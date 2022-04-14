use rand::prelude::*;
use rand_distr::{StandardNormal, Uniform};
use rand_xoshiro::Xoshiro256StarStar;
use rayon::prelude::*;
use statrs::distribution::{ContinuousCDF, Normal as NormalDistr};
use std::{
    ops::Range,
    time::{Duration, Instant},
};

use crate::{distance::zeucl, timeseries::WindowedTimeseries};

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
        self.cumulative_preprocessing
            .push(preprocessing + *self.cumulative_preprocessing.last().unwrap());
        self.collisions.push(collisions);
        eprintln!(
            "at level {} preprocessing {:?} and {:?} exploring {} collisions",
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
    /// number of concatenations to use at most, set with the cost optimizer
    pub max_k: usize,
    pub level: usize,
    max_repetitions: usize,
    pub r: f64,
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

        let mut v_buf = Vec::new();
        let mut dotp_buf = Vec::new();

        let max_repetitions = {
            let mut reps = 1;
            while Self::memory_usage_bytes(ts, reps) <= max_memory_bytes {
                reps += 1;
            }
            reps
        };

        eprintln!(
            "Maximum number of repetitions {} ({} bytes per repetition)",
            max_repetitions,
            Self::memory_usage_bytes(ts, 1)
        );

        let mut hashes = vec![Vec::new(); ts.num_subsequences()];
        let mut probe_repetition = Self::init_repetition(ts);
        let mut partition = vec![0..ts.num_subsequences()];
        let mut old_partition = Vec::new();
        eprintln!("Estimated cost at 0 {:?}", cost_estimator.estimated_cost(0));
        while max_k <= 32 {
            eprintln!(":::::::::::: k={max_k}");
            let start = Instant::now();
            partition_by_hash(
                ts,
                &mut probe_repetition,
                &mut partition,
                &mut old_partition,
                Self::derive_seed(seed, max_k, 0),
                r,
                &mut v_buf,
                &mut dotp_buf,
            );
            for triplet in probe_repetition.iter() {
                hashes[triplet.1 as usize].push(triplet.2);
            }
            let elapsed = start.elapsed();
            let collisions = partition
                .iter()
                .map(|r| (r.len() * (r.len() - 1)) / 2)
                .sum();
            cost_estimator.record(max_k, elapsed, collisions);
            let current = cost_estimator.estimated_cost(max_k);
            let previous = cost_estimator.estimated_cost(max_k - 1);
            eprintln!(
                "Estimated cost at {} {:?} ({})",
                max_k,
                current,
                current.as_secs_f64() / previous.as_secs_f64()
            );
            if max_k > 4
                && cost_estimator.estimated_cost(max_k) > cost_estimator.estimated_cost(max_k - 1)
            {
                max_k -= 1;
                break;
            }
            max_k += 1;
        }
        eprintln!("Selected max_k = {max_k}");

        Self {
            ts,
            cost_estimator,
            max_repetitions,
            r,
            max_k,
            level: max_k,
            seed,
            // We don't use the probe repetition we built for the estimation because
            // it is partitioned according to max_k + 1, and there does not seem to be an easy way to
            // undo the sorting at the last invocation of partition_by_hash
            repetitions: vec![],
        }
    }

    pub fn current_repetitions(&self) -> usize {
        self.repetitions.len()
    }

    fn memory_usage_bytes(ts: &WindowedTimeseries, n_repetitions: usize) -> usize {
        n_repetitions
            * ts.num_subsequences()
            * (std::mem::size_of::<u8>() + std::mem::size_of::<u32>())
    }

    fn derive_seed(seed: u64, k: usize, repetition: usize) -> u64 {
        (repetition + 1) as u64 * seed + k as u64
    }

    fn init_repetition(ts: &WindowedTimeseries) -> Vec<(u8, u32, i32)> {
        (0..ts.num_subsequences())
            .map(|i| (0, i as u32, 0))
            .collect()
    }

    pub fn for_pairs(
        &self,
        repetition: usize,
        mut action: impl FnMut(usize, usize),
    ) {
        self.repetitions[repetition].for_pairs(&mut action);
    }

    pub fn for_pairs_at(
        &self,
        repetition: usize,
        prefix: usize,
        mut action: impl FnMut(usize, usize),
    ) {
        self.repetitions[repetition].for_pairs_at(prefix, &mut action);
    }

    pub fn set_repetitions(&mut self, repetitions: usize) {
        assert!(repetitions <= self.max_repetitions);
        assert!(repetitions >= self.current_repetitions());
        self.add_repetitions(repetitions - self.current_repetitions());
    }

    pub fn decrease_level(&mut self, level: usize) {
        assert!(level < self.level);
        self.repetitions
            .iter_mut()
            .for_each(|rep| rep.decrease_level(level));
        self.level = level;
    }

    fn add_repetitions(&mut self, nreps: usize) {
        let timer = Instant::now();
        let s = self.repetitions.len();
        let e = s + nreps;
        let ts = &self.ts;
        let seed = self.seed;
        let r = self.r;
        let level = self.level;
        let extension = (s..e).into_par_iter().map(|repetition| {
            let mut arr = Self::init_repetition(ts);
            let mut partition = vec![0..ts.num_subsequences()];
            let mut old_partition = Vec::new();
            let mut v_buf = Vec::new();
            let mut dotp_buf = Vec::new();
            for k in 1..=level {
                partition_by_hash(
                    ts,
                    &mut arr,
                    &mut partition,
                    &mut old_partition,
                    Self::derive_seed(seed, k, repetition),
                    r,
                    &mut v_buf,
                    &mut dotp_buf,
                );
            }
            Repetition::from_vec(level, &arr, &partition)
        });
        self.repetitions.par_extend(extension);
        eprintln!("Added {nreps} repetitions in {:?}", timer.elapsed());
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
            ((1.0 / delta).log(std::f64::consts::E) / p.powi(prefix as i32)).ceil() as usize
        };
        let (prefix, reps, _cost) = (1..=self.max_k)
            .map(|prefix| {
                let reps = required_repetitions(prefix);
                let cost = self.cost_estimator.estimated_cost(prefix) * reps as u32;
                eprintln!(
                    " . cost for {} repetitions at prefix {} would be {:?}",
                    reps, prefix, cost
                );
                (prefix, reps, cost)
            })
            .filter(|(_prefix, reps, _cost)| *reps <= self.max_repetitions)
            .min_by_key(|triplet| triplet.2)
            .unwrap();
        (prefix, reps)
    }
}

fn partition_by_hash(
    ts: &WindowedTimeseries,
    indices: &mut [(u8, u32, i32)],
    partition: &mut Vec<Range<usize>>,
    old_partition: &mut Vec<Range<usize>>,
    seed: u64,
    r: f64,
    v_buf: &mut Vec<f64>,
    dotp_buf: &mut Vec<f64>,
) {
    // TODO reduce allocations by reusing buffers in thread local storage
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
    std::mem::swap(partition, old_partition);
    partition.clear();
    for range in old_partition.iter() {
        let part = &mut indices[range.clone()];
        // Sort by hash value
        part.sort_by_key(|triplet| triplet.2);

        // Find the boundaries and update the differences
        let mut start = 0;
        let mut end = start + 1;
        let offset = range.start;
        while end <= part.len() {
            if end >= part.len() || part[end].2 != part[end - 1].2 {
            // if part[end].2 != part[end - 1].2 {
                let r = (start + offset)..(end + offset);
                assert!(r.start < r.end);
                partition.push(r);
                start = end;
                end = start + 1;
            } else {
                part[end].0 += 1;
                end += 1;
            }
        }
    }
    // Check that the partition is indeed a partition
    for i in 1..partition.len() {
        assert_eq!(
            partition[i - 1].end,
            partition[i].start,
            "partition has non contiguous intervals \nold partition{:?}\nnew partition{:?}",
            &old_partition[0..10],
            &partition[0..10]
        );
    }
}

#[derive(Debug)]
struct RepetitionSlice {
    start: usize,
    end: usize,
    children: Option<Vec<Range<usize>>>,
}

impl RepetitionSlice {
    fn new(start: usize, end: usize, children: Option<Vec<Range<usize>>>) -> Self {
        if let Some(children) = children.as_ref() {
            assert_eq!(children[0].start, start);
            assert_eq!(children[children.len() - 1].end, end);
            let len: usize = children.iter().map(|r| r.len()).sum();
            for i in 1..children.len() {
                assert!(children[i - 1].start < children[i].start);
                assert_eq!(children[i - 1].end, children[i].start);
            }
            assert_eq!(len, (end - start));
        }
        Self {
            start,
            end,
            children,
        }
    }

    fn len(&self) -> usize {
        self.end - self.start
    }

    fn from_slices(slices: &[RepetitionSlice]) -> Self {
        let children = slices.iter().map(|r| r.start..r.end).collect();
        Self::new(
            slices[0].start,
            slices[slices.len() - 1].end,
            Some(children),
        )
    }

    fn from_range(r: &Range<usize>) -> Self {
        Self::new(r.start, r.end, None)
    }

    /// Run the given action on each pair of indices represented by this slice,
    /// avoiding comparisoins that have already been done within children.
    fn for_each_pair(&self, mut action: impl FnMut(usize, usize)) {
        if let Some(children) = self.children.as_ref() {
            for child_i in 0..(children.len() - 1) {
                let range_i = children[child_i].clone();
                let range_j = children[child_i+1].start..children[children.len()-1].end;
                for i in range_i {
                    for j in range_j.clone() {
                        action(i, j);
                    }
                }
            }
        } else {
            for i in self.start..self.end {
                for j in (i + 1)..self.end {
                    action(i, j);
                }
            }
        }
    }
}

struct Repetition {
    level: usize,
    diffs: Vec<u8>,
    indices: Vec<u32>,
    slices: Vec<RepetitionSlice>,
}

impl Repetition {
    fn from_vec(level: usize, v: &Vec<(u8, u32, i32)>, slices: &[Range<usize>]) -> Repetition {
        let mut diffs = Vec::with_capacity(v.len());
        let mut indices = Vec::with_capacity(v.len());
        for (diff, idx, _) in v.iter() {
            diffs.push(*diff);
            indices.push(*idx);
        }
        // The first diff should always be zero, regardless of what we got 
        //from the input (which can be different from 0 because of the rearrangement during partitioning,
        // which might cause the first bucket to be split at a higher level than zero, of course)
        diffs[0] = 0;
        for i in 1..slices.len() {
            // check that slices are contiguous
            assert_eq!(
                slices[i - 1].end,
                slices[i].start,
                "slices are not contiguous"
            );
        }
        assert!(slices.iter().cloned().all(|s| diffs[(s.start+1..s.end)].iter().all(|d| *d as usize >= level)));
        assert!(slices.iter().cloned().all(|s| diffs[s.start] == *diffs[s.start..s.end].iter().min().unwrap()));

        let slices: Vec<RepetitionSlice> = slices.iter().map(RepetitionSlice::from_range).collect();
        for s in slices.iter() {
            assert!(
                (v[s.start].0 as usize) < level,
                "{} should be < {} (range {:?})",
                v[s.start].0,
                level,
                s.start..s.end
            );
        }
        Repetition {
            level,
            diffs,
            indices,
            slices,
        }
    }

    /// Decrease the level to the given target
    fn decrease_level(&mut self, target_level: usize) {
        assert!(target_level < self.level);
        let mut start = 0;
        // TODO: get rid of allocations
        let mut new_slices = Vec::new();
        while start < self.slices.len() {
            // We grow the slice at position i so that it includes all the next slices
            // so that the breakpoints share a common prefix of length `target_level`
            let mut end = start+1;
            while end < self.slices.len()
                && (self.diffs[self.slices[end].start] as usize) >= target_level
            {
                end += 1;
            }
            new_slices.push(RepetitionSlice::from_slices(&self.slices[start..end]));
            start = end;
        }
        self.slices = new_slices;
        self.level = target_level;

        // Check postconditions
        for s in self.slices.iter() {
            assert!(
                (self.diffs[s.start] as usize) < self.level,
                "{} should be < {} (range {:?})\n{:?}",
                self.diffs[s.start],
                self.level,
                s.start..s.end,
                self.diffs
            );
            assert!(self.diffs[(s.start+1)..s.end].iter().all(|d| *d as usize >= target_level), "common prefixes at {}..{} : {:?}", s.start, s.end, &self.diffs[s.start..s.end]);
        }
    }

    fn for_pairs(&self, mut action: impl FnMut(usize, usize)) {
        for slice in self.slices.iter() {
            slice.for_each_pair(|i, j| {
                action(self.indices[i] as usize, self.indices[j] as usize);
            });
        }
    }

    // TODO: we want to skip comparisons that have already been performed.
    fn for_pairs_at(&self, prefix: usize, mut action: impl FnMut(usize, usize)) {
        let mut s = 0;
        let mut e = s + 1;
        loop {
            if e == self.diffs.len() || (self.diffs[e] as usize) < prefix {
                // Invoke action on the pairs
                for i in s..e {
                    for j in (i + 1)..e {
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

#[test]
fn test_slice() {
    let ranges = vec![0..5, 5..8, 8..10];
    let slice = RepetitionSlice::new(0, 10, Some(ranges));
    let mut pairs = Vec::new();
    slice.for_each_pair(|i, j| {
        pairs.push((i, j));
    });
    pairs.sort();
    let mut expected = vec![
        (0, 5),
        (0, 6),
        (0, 7),
        (1, 5),
        (1, 6),
        (1, 7),
        (2, 5),
        (2, 6),
        (2, 7),
        (3, 5),
        (3, 6),
        (3, 7),
        (4, 5),
        (4, 6),
        (4, 7),
        (0, 8),
        (0, 9),
        (1, 8),
        (1, 9),
        (2, 8),
        (2, 9),
        (3, 8),
        (3, 9),
        (4, 8),
        (4, 9),
        (5, 8),
        (5, 9),
        (6, 8),
        (6, 9),
        (7, 8),
        (7, 9),
    ];
    expected.sort();
    assert_eq!(pairs, expected);
}
