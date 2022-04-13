use std::{ops::Range, time::{Duration, Instant}};
use rayon::prelude::*;
use statrs::distribution::{ContinuousCDF, Normal as NormalDistr};
use rand::prelude::*;
use rand_distr::{StandardNormal, Uniform};
use rand_xoshiro::Xoshiro256StarStar;

use crate::{timeseries::WindowedTimeseries, distance::zeucl};

// const show_a: u32 = 1172;
// const show_b: u32 = 6112;
const show_a: u32 = 7137172;
const show_b: u32 = 7414112;

struct CostEstimator {
    /// how much it costs to carry out a pair evaluation
    pair_evaluation: Duration,
    /// The time, for each prefix length, to do the preprocessing
    cumulative_preprocessing: Vec<Duration>,
    /// The number of collisions at each prefix length
    collisions: Vec<usize>,
}

impl CostEstimator {
    fn new(ts: &WindowedTimeseries) -> Self {
        Self {
            pair_evaluation: Duration::from_nanos(1500), // this number should be derived from benchmarking
            cumulative_preprocessing: vec![Duration::from_secs(0)],
            collisions: vec![(ts.num_subsequences() * (ts.num_subsequences() - 1)) / 2],
        }
    }

    fn record(&mut self, level: usize, preprocessing: Duration, collisions: usize) {
        assert_eq!(level, self.cumulative_preprocessing.len());
        self.cumulative_preprocessing.push(preprocessing + *self.cumulative_preprocessing.last().unwrap());
        self.collisions.push(collisions);
        eprintln!("at level {} preprocessing {:?} and {:?} exploring {} collisions", 
            level, 
            self.cumulative_preprocessing.last().unwrap(),
            *self.collisions.last().unwrap() as u32 * self.pair_evaluation,
            self.collisions.last().unwrap()
        );
    }

    fn estimated_cost(&self, level: usize) -> Duration {
        let preprocessing: Duration = self.cumulative_preprocessing[level];
        let collisions: Duration = self.collisions[level] as u32 * self.pair_evaluation;
        preprocessing + collisions
    }
}

pub struct AdaptiveHashCollection<'ts> {
    ts: &'ts WindowedTimeseries,
    cost_estimator: CostEstimator,
    max_memory_bytes: usize,
    /// number of concatenations to use at most, set with the cost optimizer
    pub max_k: usize,
    max_repetitions: usize,
    r: f64,
    seed: u64,
    /// the vector of the repetitions holds, for each repetition, a vector of subsequence indexes
    /// lexicographically sorted by hash code. Instead of storing the hash code, we store the prefix length
    /// at which each subsequence's hash code differs from the previous one in the order
    repetitions: Vec<Repetition>,
}

impl<'ts> AdaptiveHashCollection<'ts> {
    pub fn new(ts: &'ts WindowedTimeseries, max_memory_bytes: usize, seed: u64) -> Self {
        let mut cost_estimator = CostEstimator::new(ts);
        let mut max_k = 1usize;
        let r = 1.0;

        eprintln!("known motif distance {}", zeucl(ts, show_a as usize, show_b as usize));

        let mut v_buf = Vec::new();
        let mut dotp_buf = Vec::new();

        let max_repetitions = {
            let mut reps = 1;
            while Self::memory_usage_bytes(ts, reps) <= max_memory_bytes {
                reps += 1;
            }
            reps
        };

        eprintln!("Maximum number of repetitions {} ({} bytes per repetition)", max_repetitions, Self::memory_usage_bytes(ts, 1));


        let mut probe_repetition = Self::init_repetition(ts);
        let mut partition = vec![0..ts.num_subsequences()];
        eprintln!("Estimated cost at 0 {:?}", cost_estimator.estimated_cost(0));
        while max_k <= 32 {
            eprintln!(":::::::::::: k={max_k}");
            let start = Instant::now();
            partition_by_hash(
                ts,
                &mut probe_repetition,
                &mut partition,
                Self::derive_seed(seed, max_k, 0),
                r,
                &mut v_buf,
                &mut dotp_buf,
            );
            let elapsed = start.elapsed();
            let collisions = partition.iter().map(|r| (r.len() * (r.len()-1)) / 2).sum();
            cost_estimator.record(max_k, elapsed, collisions);
            let current = cost_estimator.estimated_cost(max_k);
            let previous = cost_estimator.estimated_cost(max_k-1);
            eprintln!("Estimated cost at {} {:?} ({})", max_k, current, current.as_secs_f64() / previous.as_secs_f64());
            if max_k > 4 && cost_estimator.estimated_cost(max_k) > cost_estimator.estimated_cost(max_k-1) {
                max_k -= 1;
                break;
            }
            max_k += 1;
        }
        eprintln!("Selected max_k = {max_k}");

        Self {
            ts,
            cost_estimator,
            max_memory_bytes,
            max_repetitions,
            r,
            max_k,
            seed,
            repetitions: vec![Repetition::from_vec(&probe_repetition)],
        }
    }

    fn memory_usage_bytes(ts: &WindowedTimeseries, n_repetitions: usize) -> usize {
        n_repetitions * ts.num_subsequences() * (
            std::mem::size_of::<u8>() +
            std::mem::size_of::<u32>()
        )
    }

    fn derive_seed(seed: u64, k: usize, repetition: usize) -> u64 {
        (repetition + 1) as u64 * seed + k as u64
    }

    fn init_repetition(ts: &WindowedTimeseries) -> Vec<(u8, u32, i32)> {
        (0..ts.num_subsequences())
            .map(|i| (0, i as u32, 0))
            .collect()
    }

    pub fn for_pairs_at(&self, repetition: usize, prefix: usize, mut action: impl FnMut(usize, usize)) {
        self.repetitions[repetition].for_pairs_at(prefix, &mut action);
    }

    pub fn add_repetitions(&mut self, nreps: usize) {
        let s = self.repetitions.len();
        let e = s + nreps;
        let ts = &self.ts;
        let max_k = self.max_k;
        let seed = self.seed;
        let r = self.r;
        let extension = (s..e).into_par_iter().map(|repetition| {
            // for debug purposes only
            let mut hash_values: Vec<Vec<i32>> = vec![Vec::new(); ts.num_subsequences()];
            let mut arr = Self::init_repetition(ts);
            let mut partition = vec![0..ts.num_subsequences()];
            let mut v_buf = Vec::new();
            let mut dotp_buf = Vec::new();
            for k in 1..=max_k {
                partition_by_hash(
                    ts,
                    &mut arr,
                    &mut partition,
                    Self::derive_seed(seed, k, repetition),
                    r,
                    &mut v_buf,
                    &mut dotp_buf,
                );
                // just for debugging
                for (_, idx, h) in arr.iter() {
                    hash_values[*idx as usize].push(*h);
                }
            }
            Repetition::from_vec(&arr)
        });
        self.repetitions.par_extend(extension);
    }

    pub fn collision_probability_at(&self, d: f64) -> f64 {
        let r = self.r;
        let normal = NormalDistr::new(0.0, 1.0).unwrap();
        1.0 - 2.0 * normal.cdf(-r / d)
            - (2.0 / ((std::f64::consts::PI * 2.0).sqrt() * (r / d)))
                * (1.0 - (-r * r / (2.0 * d * d)).exp())
    }

    pub fn best_move(&self, distance: f64, delta: f64) -> (usize, usize) {
        // If we can confirm the given distance with the current number of repetitions,
        // possibly with a shorter prefix, then we don't instantiate more repetitions
        // and we just move to the shorter prefix.
        // Otherwise we allocate a few more repetitions and see if we discover 
        // a smaller distance.
        // We do the same if the current distance can be confirmed with a slightly 
        // longer prefix and a few more repetitions (within the maximum limit) at a smaller cost
        let p = self.collision_probability_at(distance);
        let required_repetitions = |prefix: usize| {
            ((1.0/delta).log(std::f64::consts::E) / p.powi(prefix as i32)).ceil() as usize
        };
        for prefix in 1..self.max_k {
            let reps = required_repetitions(prefix);
            let cost = self.cost_estimator.estimated_cost(prefix) * reps as u32;
            eprintln!(
                "At prefix {} we should do {} repetitions, estimated cost {:?}",
                prefix, reps, cost
            );
        }
        todo!()
    }
}

fn partition_by_hash(
    ts: &WindowedTimeseries,
    indices: &mut [(u8, u32, i32)],
    partition: &mut Vec<Range<usize>>,
    seed: u64,
    r: f64,
    v_buf: &mut Vec<f64>,
    dotp_buf: &mut Vec<f64>,
) {
    // Compute hash value
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
    // Get the random vector
    v_buf.clear();
    v_buf.resize_with(ts.w, || StandardNormal.sample(&mut rng));
    let b = Uniform::new(0.0, r).sample(&mut rng);
    dotp_buf.resize(ts.num_subsequences(), 0.0);

    ts.znormalized_sliding_dot_product(&v_buf, dotp_buf);
    let mut hash_values = Vec::new();
    for dotp in dotp_buf.iter() {
        hash_values.push(((dotp + b) / r).floor() as i32);
    }
    for triplet in indices.iter_mut() {
        triplet.2 = hash_values[triplet.1 as usize];
    }
    // eprintln!(" . time hash: {:?}", t_hash.elapsed());

    // Partition by values
    let mut new_partition: Vec<Range<usize>> = Vec::new();
    for range in partition.iter() {
        let part = &mut indices[range.clone()];
        // Sort by hash value
        part.sort_by_key(|triplet| triplet.2);

        // Find the boundaries and update the differences
        let mut start = 0;
        let mut end = start + 1;
        let offset = range.start;
        loop {
            if end >= part.len() || part[end].2 != part[end - 1].2 {
                let r = (start + offset)..(end + offset);
                assert!(r.start < r.end);
                new_partition.push(r);
                start = end;
                end = start + 1;
                if end >= part.len() {
                    break;
                }
            } else {
                part[end].0 += 1;
                end += 1;
            }
        }
    }

    // Update partitions
    std::mem::swap(partition, &mut new_partition);
}

struct Repetition {
    diffs: Vec<u8>,
    indices: Vec<u32>
}

impl Repetition {
    fn from_vec(v: &Vec<(u8, u32, i32)>) -> Repetition {
        let mut diffs = Vec::with_capacity(v.len());
        let mut indices = Vec::with_capacity(v.len());
        for (diff, idx, _) in v.iter() {
            diffs.push(*diff);
            indices.push(*idx);
        }
        Repetition {diffs, indices}
    }

    // TODO: we might want to cache the boundaries?
    fn for_pairs_at(&self, prefix: usize, mut action: impl FnMut(usize, usize)) {
        // where are the two motifs?
        // for (p, (diff, idx)) in self.diffs.iter().zip(self.indices.iter()).enumerate() {
        //     if *idx == show_a || *idx == show_b {
        //         eprintln!("  {idx} has a diff of {diff} with previous and is at position {p}");
        //     }
        // }

        let mut s = 0;
        let mut e = s + 1;
        loop {
            if e == self.diffs.len() || (self.diffs[e] as usize) < prefix {
                // Invoke action on the pairs
                for i in s..e {
                    for j in (i+1)..e {
                        action(self.indices[i] as usize, self.indices[j] as usize);
                    }
                }

                if e >= self.diffs.len() {
                    return;
                }
                s = e;
                e = s + 1;
            } else {
                e += 1;
            }
        }
    }
}
