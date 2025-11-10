use core::f64;
use log::debug;
use rand::prelude::*;
use rand_distr::{Normal, Uniform};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use rustfft::num_complex;
use std::{
    mem::size_of,
    ops::Range,
    time::{Duration, Instant},
};

use crate::{
    allocator::{ByteSize, Bytes},
    knn::Distance,
    observe::*,
    timeseries::{FFTData, Overlaps, WindowedTimeseries},
};

pub const K: usize = 8;
pub const MASKS: [u64; K + 1] = [
    0x0000000000000000, // unused
    0xFF00000000000000, // 1
    0xFFFF000000000000, // 2
    0xFFFFFF0000000000, // 3
    0xFFFFFFFF00000000, // 4
    0xFFFFFFFFFF000000, // 5
    0xFFFFFFFFFFFF0000, // 6
    0xFFFFFFFFFFFFFF00, // 7
    0xFFFFFFFFFFFFFFFF, // 8
];

#[derive(Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct HashValue(u64);
impl HashValue {
    fn set_byte(&mut self, k: usize, byte: u8) {
        // bytes are indexed from the left
        let byte_pos = 7 - k;
        // Clear byte
        self.0 &= !(0xFFu64 << ((size_of::<u8>() * byte_pos) * 8));
        // Set byte
        self.0 |= (byte as u64) << ((size_of::<u8>() * byte_pos) * 8);
    }

    fn prefix_eq(&self, other: &Self, prefix: usize) -> bool {
        assert!(prefix > 0);
        let mask = MASKS[prefix];
        self.0 & mask == other.0 & mask
    }
}

impl From<u64> for HashValue {
    #[inline]
    fn from(v: u64) -> Self {
        Self(v)
    }
}

impl Into<u64> for HashValue {
    #[inline]
    fn into(self) -> u64 {
        self.0
    }
}
impl Into<u64> for &HashValue {
    #[inline]
    fn into(self) -> u64 {
        self.0
    }
}

impl std::fmt::Debug for HashValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:016x}", self.0)
    }
}

#[test]
fn test_set_byte() {
    let mut h = HashValue(0);
    h.set_byte(0, 0xAA);
    assert_eq!(h, HashValue(0xaa00000000000000u64));
    h.set_byte(1, 0xBB);
    assert_eq!(h, HashValue(0xaabb000000000000u64));
}

#[test]
fn test_hash_prefix_compare() {
    let h1 = HashValue(0x1122000000000000u64);
    let h2 = HashValue(0x1133000000000000u64);
    assert!(h1.prefix_eq(&h2, 1));
    assert!(!h1.prefix_eq(&h2, 2));
    assert!(!h1.prefix_eq(&h2, 2));
}

/// Stores information to compute a single repetition of [HashValue]s
struct Hasher {
    vectors: [Vec<f64>; K],
    shifts: [f64; K],
    width: f64,
}

impl ByteSize for Hasher {
    fn byte_size(&self) -> Bytes {
        self.vectors.iter().map(|v| v.byte_size()).sum::<Bytes>()
            + self.shifts.iter().map(|v| v.byte_size()).sum()
            + self.width.byte_size()
    }

    fn mem_tree_fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.byte_size())
    }
}

impl Hasher {
    fn new<R: Rng>(dimension: usize, width: f64, rng: &mut R) -> Self {
        let uniform = Uniform::new(0.0, width);
        let shifts = [
            uniform.sample(rng),
            uniform.sample(rng),
            uniform.sample(rng),
            uniform.sample(rng),
            uniform.sample(rng),
            uniform.sample(rng),
            uniform.sample(rng),
            uniform.sample(rng),
        ];

        let vectors = [
            Self::sample_vec(dimension, rng),
            Self::sample_vec(dimension, rng),
            Self::sample_vec(dimension, rng),
            Self::sample_vec(dimension, rng),
            Self::sample_vec(dimension, rng),
            Self::sample_vec(dimension, rng),
            Self::sample_vec(dimension, rng),
            Self::sample_vec(dimension, rng),
        ];
        Self {
            vectors,
            shifts,
            width,
        }
    }

    fn sample_vec<R: Rng>(dimension: usize, rng: &mut R) -> Vec<f64> {
        let normal = Normal::new(0.0, 1.0).expect("problem instantiating normal distribution");
        normal
            .sample_iter(rng)
            .take(dimension)
            .collect::<Vec<f64>>()
    }

    pub fn collision_probability_at(&self, d: Distance) -> f64 {
        use crate::stats::Normal;
        let d = d.0;
        let r = self.width;
        let normal = Normal::default();
        1.0 - 2.0 * normal.cdf(-r / d)
            - (2.0 / ((std::f64::consts::PI * 2.0).sqrt() * (r / d)))
                * (1.0 - (-r * r / (2.0 * d * d)).exp())
    }

    /// Computes the width to be used for the random projection hashing. Setting this parameter
    /// correctly is crucial for performance.
    ///
    /// The heuristic adopted by this method is the following. First we estimate the maximum dot
    /// product and we consequently compute the width that allows to fit the discretized dot
    /// products in 8 bits.
    ///
    /// Then we want close pairs of points to collide in 8 consecutive concatenations, so that
    /// full-hashes make sense. To this end we first find a pair of close subsequences (without
    /// guarantees on the minimality of their distance: this is just a heuristic). Then we compute
    /// [[K]] dot products for the two subsequences, and record the maximum absolute difference in
    /// projections. Twice this number is our guess for the width, or the minimum width described
    /// in the previous paragraph, whichever is largest.
    pub fn compute_width<R: Rng>(
        ts: &WindowedTimeseries,
        fft_data: &FFTData,
        _exclusion_zone: usize,
        rng: &mut R,
    ) -> f64 {
        let timer = Instant::now();
        let n = ts.num_subsequences();
        let subsequence_norm = (ts.w as f64).sqrt();
        let expected_max_dotp = subsequence_norm * (2.0 * (n as f64).ln()).sqrt();
        debug!("Expected max dot product {}", expected_max_dotp);
        let mut dotps = vec![(0.0, 0); n];
        // let min_width = expected_max_dotp / 128.0;
        // Compute a few dot products
        let v = Self::sample_vec(ts.w, rng);
        ts.znormalized_sliding_dot_product_write(fft_data, &v, &mut dotps, |i, dotp, out| {
            if !dotp.is_nan() {
                *out = (dotp, i);
            } else {
                *out = (f64::INFINITY, usize::MAX);
            }
        });
        dotps.par_sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        // remove the infinity values (that are placeholders for NaN values in the dot products)
        while let Some(last) = dotps.last() {
            if last.0.is_infinite() {
                dotps.pop();
            } else {
                break;
            }
        }

        let (perc1, perc99) = (dotps[dotps.len() / 100].0, dotps[99 * dotps.len() / 100].0);
        let min_sample_width = perc1.abs().max(perc99.abs()) / 128.0;
        debug!(
            "dot product 1perc {} 99perc {} (min sample width: {}) max/min {}/{}",
            dotps[dotps.len() / 100].0,
            dotps[99 * dotps.len() / 100].0,
            min_sample_width,
            dotps.iter().min_by(|a, b| a.0.total_cmp(&b.0)).unwrap().0,
            dotps.iter().max_by(|a, b| a.0.total_cmp(&b.0)).unwrap().0,
        );
        debug!(
            "width: {} computed in {:?}",
            min_sample_width,
            timer.elapsed()
        );

        min_sample_width
    }

    /// Hash all the subsequences of the time series to 8 x 8-bit hash values each
    fn hash(
        &self,
        ts: &WindowedTimeseries,
        fft_data: &FFTData,
        output: &mut [(HashValue, u32)],
        max_k: usize,
    ) {
        assert_eq!(ts.num_subsequences(), output.len());
        output.fill((HashValue::default(), 0));
        for k in 0..max_k {
            ts.znormalized_sliding_dot_product_write(
                fft_data,
                &self.vectors[k],
                output,
                move |i, mut h, out| {
                    if out.1 == u32::MAX {
                        // the subsequence had an overflown hash value, skip it
                        return;
                    }
                    if !h.is_nan() {
                        h = (h + self.shifts[k]) / self.width;
                        if h.abs() > 127.0 {
                            out.0 = HashValue(u64::MAX);
                            out.1 = u32::MAX;
                        } else {
                            let h = ((h as i64 & 0xFFi64) as i8) as u8;
                            out.0.set_byte(k, h);
                            out.1 = i as u32;
                        }
                    } else {
                        // if the subsequence is flat don't include it
                        // in subsequent computations by giving it a hash that makes it
                        // go to the end of the sorted hash-array
                        out.0 = HashValue(u64::MAX);
                        out.1 = u32::MAX;
                    }
                },
            );
        }
    }
}

struct Repetition {
    hashes: Vec<HashValue>,
    indices: Vec<u32>,
}

impl ByteSize for Repetition {
    fn byte_size(&self) -> Bytes {
        self.hashes.iter().map(|v| v.byte_size()).sum::<Bytes>()
            + self.indices.iter().map(|v| v.byte_size()).sum()
    }

    fn mem_tree_fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.byte_size())
    }
}

impl Repetition {
    fn sample<R: Rng>(
        ts: &WindowedTimeseries,
        quantization_width: f64,
        max_k: usize,
        fft_data: &FFTData,
        rng: &mut R,
        tmp: &mut Vec<(HashValue, u32)>,
    ) -> (Hasher, Self) {
        let hasher = Hasher::new(ts.w, quantization_width, rng);
        tmp.resize(ts.num_subsequences(), (HashValue::default(), 0u32));
        hasher.hash(ts, fft_data, tmp, max_k);
        tmp.par_sort_unstable_by_key(|pair| pair.0);
        (
            hasher,
            Self::from_pairs(ts.num_subsequences(), tmp.iter().copied()),
        )
    }

    fn from_pairs<I: IntoIterator<Item = (HashValue, u32)>>(n: usize, pairs: I) -> Self {
        let mut hashes = Vec::with_capacity(n);
        let mut indices = Vec::with_capacity(n);
        for (h, idx) in pairs {
            if idx != u32::MAX {
                // remove subsequences with overflown hash
                hashes.push(h);
                indices.push(idx);
            }
        }
        hashes.shrink_to_fit();
        indices.shrink_to_fit();
        Self { hashes, indices }
    }

    fn bytes(&self) -> Bytes {
        let n = self.hashes.len();
        Bytes(n * (std::mem::size_of::<HashValue>() + std::mem::size_of::<u32>()))
    }

    fn get_hashes(&self) -> &[HashValue] {
        &self.hashes
    }

    fn get_indices(&self) -> &[u32] {
        &self.indices
    }

    fn collisions(&self, prefix: usize, prev_prefix: Option<usize>) -> CollisionEnumerator<'_> {
        CollisionEnumerator::new(&self, prefix, prev_prefix)
    }

    /// the number of collisions for each prefix, _including_ trivial collisions
    pub fn collision_profile(&self) -> Vec<f64> {
        let mut out = vec![f64::INFINITY; K + 1];
        for prefix in 1..=K {
            out[prefix] = CollisionEnumerator::new(self, prefix, None).count_collisions() as f64;
        }
        out
    }

    pub fn count_non_trivial(&self, prefix: usize, exclusion_zone: usize) -> usize {
        CollisionEnumerator::new(self, prefix, None).count_non_trivial_collisions(exclusion_zone)
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct LSHIndexStats {
    pub num_repetitions: usize,
    pub memory_usage: Bytes,
}

impl LSHIndexStats {
    #[rustfmt::skip]
    pub fn observe(&self, repetition: usize, prefix: usize) {
        observe!(repetition, prefix, "num_repetitions", self.num_repetitions);
        observe!(repetition, prefix, "main_memory_usage", self.memory_usage.0);
    }
}

#[derive(Clone, Copy, Default)]
struct Average {
    total: Duration,
    count: u32,
}

impl std::fmt::Debug for Average {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.average())
    }
}

impl Average {
    fn average(&self) -> Duration {
        self.total / self.count
    }

    fn update(&mut self, add_total: Duration, add_count: usize) {
        self.total += add_total;
        self.count += add_count as u32;
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CostEstimator {
    repetition_time: Average,
    collision_time: Average,
}

impl CostEstimator {
    pub fn repetition_time(&self) -> Duration {
        self.repetition_time.average()
    }

    pub fn collision_time(&self) -> Duration {
        self.collision_time.average()
    }

    pub fn update_repetition_time(&mut self, elapsed: Duration, num_new_repetitions: usize) {
        self.repetition_time.update(elapsed, num_new_repetitions);
    }

    pub fn update_collision_time(&mut self, elapsed: Duration, num_new_collisions: usize) {
        self.collision_time.update(elapsed, num_new_collisions);
    }
}

pub struct LSHIndex {
    rng: Xoshiro256PlusPlus,
    quantization_width: f64,
    functions: Vec<Hasher>,
    repetitions: Vec<Repetition>,
    max_repetitions: usize,
    collision_profile: Vec<f64>,
    pub cost_estimator: CostEstimator,
}

impl ByteSize for LSHIndex {
    fn byte_size(&self) -> Bytes {
        self.functions.byte_size() + self.repetitions.byte_size()
    }

    fn mem_tree_fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct(&format!("LSHIndex({})", self.byte_size()))
            .field_with("functions", |f| write!(f, "{}", self.functions.byte_size()))
            .field_with("repetitions", |f| {
                write!(f, "{}", self.repetitions.byte_size())
            })
            .finish()
    }
}

impl LSHIndex {
    /// How much memory would it be required to store information for these many repetitions?
    pub fn required_memory(ts: &WindowedTimeseries, repetitions: usize) -> Bytes {
        let hashes = repetitions * ts.num_subsequences() * size_of::<HashValue>();
        let indices = repetitions * ts.num_subsequences() * size_of::<u32>();
        Bytes(hashes + indices)
    }

    pub fn index_stats(&self) -> LSHIndexStats {
        let mut main_memory_usage = Bytes(0);

        for rep in self.repetitions.iter() {
            main_memory_usage += rep.bytes();
        }

        LSHIndexStats {
            num_repetitions: self.repetitions.len(),
            memory_usage: main_memory_usage,
        }
    }

    /// With this function we can construct a `HashCollection` from a `WindowedTimeseries`
    /// and a `Hasher`.
    pub fn from_ts(
        ts: &WindowedTimeseries,
        exclusion_zone: usize,
        fft_data: &FFTData,
        max_memory: Bytes,
        seed: u64,
    ) -> Self {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let sqrt_n = (ts.num_subsequences() as f64).sqrt().ceil() as usize;

        let t = Instant::now();
        let (qw, collision_profile) = if let Ok(qw) = std::env::var("ATTIMO_QUANTIZATION") {
            let qw = qw
                .parse::<f64>()
                .expect("Unable to parse ATTIMO_QUANTIZATION as a float");

            let mut tmp = Vec::new();
            let mut collision_profile = vec![0.0f64; K + 1];
            const SAMPLES: usize = 8;
            for _ in 0..SAMPLES {
                let cp = Repetition::sample(ts, qw, K, fft_data, &mut rng, &mut tmp)
                    .1
                    .collision_profile();
                for (acc, c) in collision_profile.iter_mut().zip(cp) {
                    *acc += c;
                }
            }
            for c in collision_profile.iter_mut() {
                *c /= SAMPLES as f64;
            }
            (qw, collision_profile)
        } else {
            let mut qw_lower: Option<f64> = None;
            let mut qw_upper: Option<f64> = None;
            let mut tmp = Vec::new();
            log::debug!("compute quantization width");
            let mut qw = Hasher::compute_width(ts, fft_data, exclusion_zone, &mut rng);
            log::debug!("initial value {}", qw);
            let mut collision_profile = vec![0.0f64; K + 1];

            loop {
                if let (Some(lower), Some(upper)) = (qw_lower, qw_upper) {
                    qw = (upper + lower) / 2.0;
                    log::debug!("new guess {}", qw);
                }

                collision_profile.fill(0.0);
                let mut longest_prefix_collisions = Vec::<usize>::new();
                const SAMPLES: usize = 8;
                for _ in 0..SAMPLES {
                    let cp = Repetition::sample(ts, qw, K, fft_data, &mut rng, &mut tmp)
                        .1
                        .collision_profile();
                    longest_prefix_collisions.push(*cp.last().unwrap() as usize);
                    for (acc, c) in collision_profile.iter_mut().zip(cp) {
                        *acc += c;
                    }
                }
                for c in collision_profile.iter_mut() {
                    *c /= SAMPLES as f64;
                }

                longest_prefix_collisions.sort();
                let median_collisions =
                    longest_prefix_collisions[longest_prefix_collisions.len() / 2];
                let avg_collisions = collision_profile.last().unwrap();

                debug!(
                "Num collisions with quantization_width={}: {} median {} average (lower {:?}, upper {:?})",
                qw, median_collisions, avg_collisions, qw_lower, qw_upper
            );
                if median_collisions < sqrt_n {
                    // the quantization width is too small
                    qw_lower.replace(qw);
                    qw *= 2.0;
                    debug!("Doubling the quantization width to {}", qw);
                } else if median_collisions > 10 * sqrt_n
                    && qw_upper.unwrap_or(f64::INFINITY) - qw_lower.unwrap_or(0.0) > 1e-7
                {
                    // the quantization width is too large: too many subsequences
                    qw_upper.replace(qw);
                    qw /= 2.0; // FIXME: this ends up repeating the qw value already inspected in the previous step
                    debug!("Halving the quantization width {}", qw);
                } else {
                    debug!("Settling on quantization width {}", qw);
                    break;
                }
            }
            (qw, collision_profile)
        };
        debug!("Collision profile: {:?}", collision_profile);
        debug!("sqrt(n) = {}", sqrt_n);

        let mut max_repetitions = 0;
        while Self::required_memory(ts, max_repetitions) < max_memory {
            max_repetitions += 1;
        }
        debug!(
            "quantization width: {}, maximum repetitions: {} ({}, {} for 8 reps)",
            qw,
            max_repetitions,
            Self::required_memory(ts, max_repetitions),
            Self::required_memory(ts, 8),
        );

        let mut slf = Self {
            rng: rng.clone(),
            quantization_width: qw,
            functions: Vec::new(),
            repetitions: Vec::new(),
            max_repetitions,
            collision_profile,
            cost_estimator: CostEstimator::default(),
        };
        slf.add_repetitions(ts, fft_data, 1, K);

        observe!(0, 0, "profile/index_setup", t.elapsed().as_secs_f64());
        return slf;
    }

    pub fn add_repetitions(
        &mut self,
        ts: &WindowedTimeseries,
        fft_data: &FFTData,
        total_repetitions: usize,
        max_k: usize,
    ) {
        assert!(
            total_repetitions > self.get_repetitions(),
            "total_repetitions {} is not > self.get_repetitions() {}",
            total_repetitions,
            self.get_repetitions()
        );
        let starting_repetitions = self.get_repetitions();
        let new_repetitions = total_repetitions - starting_repetitions;
        log::debug!("Adding {} new repetitions", new_repetitions);

        let timer = Instant::now();
        let mut tmp = Vec::new();
        for _ in 0..new_repetitions {
            let (hasher, repetition) = Repetition::sample(
                ts,
                self.quantization_width,
                max_k,
                fft_data,
                &mut self.rng,
                &mut tmp,
            );
            self.functions.push(hasher);
            self.repetitions.push(repetition);
        }

        let elapsed = timer.elapsed();
        log::debug!("Added {} new repetitions in {:?}", new_repetitions, elapsed);
        self.cost_estimator
            .update_repetition_time(elapsed, new_repetitions)
    }

    /// Get how many repetitions are available in this index
    pub fn get_repetitions(&self) -> usize {
        self.repetitions.len()
    }

    pub fn max_repetitions(&self) -> usize {
        self.max_repetitions
    }

    pub fn collision_probability_at(&self, d: Distance) -> f64 {
        self.functions[0].collision_probability_at(d)
    }

    #[cfg(test)]
    fn empirical_collision_probability(&self, i: usize, j: usize, prefix: usize) -> f64 {
        let mut cnt = 0;
        // for (hs, idxs) in self.hashes.iter().zip(&self.indices) {
        for repetition in self.repetitions.iter() {
            // let repetition = repetition.get();
            let hs = repetition.get_hashes();
            let idxs = repetition.get_indices();
            let hi = hs[*idxs.iter().find(|idx| **idx as usize == i).unwrap() as usize];
            let hj = hs[*idxs.iter().find(|idx| **idx as usize == j).unwrap() as usize];
            if hi.prefix_eq(&hj, prefix) {
                cnt += 1;
            }
        }

        cnt as f64 / self.get_repetitions() as f64
    }

    pub fn failure_probability(
        &self,
        d: Distance,
        reps: usize,
        prefix: usize,
        prev_prefix: Option<usize>,
    ) -> f64 {
        let p = self.collision_probability_at(d);
        let cur_fail = (1.0 - p.powi(prefix as i32)).powi(reps as i32);
        let prev_fail = prev_prefix
            .map(|prefix| {
                let repetitions = self.get_repetitions();
                (1.0 - p.powi(prefix as i32)).powi(0i32.max(repetitions as i32 - reps as i32))
            })
            .unwrap_or(1.0);
        cur_fail * prev_fail
    }

    /// the propobability that a pair at the given distance collides at least once
    /// in the given number of repetitions with the given prefix
    pub fn at_least_one_collision_prob(&self, d: Distance, reps: usize, prefix: usize) -> f64 {
        let p = self.collision_probability_at(d);
        let never_collide = (1.0 - p.powi(prefix as i32)).powi(reps as i32);
        let at_least_one = 1.0 - never_collide;
        at_least_one
    }

    /// get the largest distance (smaller than the given upper bound) that
    /// is confirmed using the given number of repetitions and prefix
    pub fn largest_confirmed_distance(
        &self,
        upper_bound: Distance,
        reps: usize,
        prefix: usize,
        prev_prefix: Option<usize>,
        target_failure_probability: f64,
    ) -> Distance {
        let eps = 0.01;
        let mut upper = upper_bound.0;
        let mut lower = 0.0;
        let mut d = (upper + lower) / 2.0;
        while upper - lower > eps {
            d = (upper + lower) / 2.0;
            let fp = self.failure_probability(Distance(d), reps, prefix, prev_prefix);
            if fp < target_failure_probability {
                lower = d;
            } else {
                upper = d;
            }
        }
        Distance(d)
    }

    // pub fn stats(&self, ts: &WindowedTimeseries, exclusion_zone: usize) -> IndexStats {
    //     IndexStats::new(self, ts, self.max_repetitions, exclusion_zone)
    // }

    /// estimates the average number of collisions at the given prefix,
    /// using _all_ the available repetitions
    pub fn average_collisions(&self, prefix: usize, exclusion_zone: usize) -> f64 {
        let sum: f64 = self
            .repetitions
            .par_iter()
            .map(|rep| {
                let c = CollisionEnumerator::new(rep, prefix, None)
                    .estimate_num_collisions(exclusion_zone);
                c
            })
            .sum::<usize>() as f64;
        sum / (self.repetitions.len() as f64)
    }

    /// Gets the minimum, maximum, median, and mean number of collisions at the given
    /// prefix for in all repetitions
    pub fn collisions_stats(
        &self,
        prefix: usize,
        exclusion_zone: usize,
    ) -> (usize, usize, usize, f64) {
        let mut collisions: Vec<usize> = self
            .repetitions
            .par_iter()
            .map(|rep| {
                CollisionEnumerator::new(rep, prefix, None).estimate_num_collisions(exclusion_zone)
            })
            .collect();
        collisions.sort();
        let mean = collisions.iter().sum::<usize>() as f64 / collisions.len() as f64;
        (
            collisions[0],
            collisions[collisions.len() / 2],
            collisions[collisions.len() - 1],
            mean,
        )
    }

    pub fn collisions(
        &self,
        repetition: usize,
        prefix: usize,
        prev_prefix: Option<usize>,
    ) -> CollisionEnumerator<'_> {
        assert!(prefix > 0 && prefix <= K, "illegal prefix {}", prefix);
        let rep = &self.repetitions[repetition];
        rep.collisions(prefix, prev_prefix)
    }

    pub fn collision_profile(&self) -> &[f64] {
        &self.collision_profile
    }
}

pub struct CollisionEnumerator<'index> {
    prefix: usize,
    prev_prefix: Option<usize>,
    repetition: &'index Repetition,
    current_range: Range<usize>,
    i: usize,
    j: usize,
}
impl<'index> CollisionEnumerator<'index> {
    fn new(repetition: &'index Repetition, prefix: usize, prev_prefix: Option<usize>) -> Self {
        assert!(prefix > 0 && prefix <= K);
        let mut slf = Self {
            prefix,
            prev_prefix,
            repetition,
            current_range: 0..0,
            i: 0,
            j: 1,
        };
        slf.next_range();
        slf
    }

    /// efficiently the enumerator to the next range of equal (by prefix)
    /// hash values
    fn next_range(&mut self) {
        let hashes = self.repetition.get_hashes();
        if hashes.is_empty() {
            return;
        }

        // exponential search
        let start = self.current_range.end;
        let h = hashes[start];
        let mut offset = 1;
        let mut low = start;
        if low >= hashes.len() {
            self.current_range = low..low;
            return;
        }
        while start + offset < hashes.len() && hashes[start + offset].prefix_eq(&h, self.prefix) {
            low = start + offset;
            offset *= 2;
        }
        let high = (start + offset).min(hashes.len());

        // binary search
        debug_assert!(
            hashes[low].prefix_eq(&h, self.prefix),
            "{:?} != {:?} (prefix {})",
            hashes[low],
            h,
            self.prefix
        );
        let off = hashes[low..high].partition_point(|hv| h.prefix_eq(hv, self.prefix));
        let end = low + off;
        self.current_range = start..end;
        self.i = self.current_range.start;
        self.j = self.i + 1;
    }

    /// Fills the given output buffer with colliding pairs, and returns the number
    /// of pairs that have been put into the buffer. If there were no pairs to add,
    /// return `None`.
    /// The third `f64` element is there just as a placeholder, which will be initialized
    /// as `f64::INFINITY`: actual distances must be computed somewhere else
    pub fn next(
        &mut self,
        output: &mut [(u32, u32, Distance)],
        exclusion_zone: usize,
    ) -> Option<usize> {
        let mut idx = 0;
        log::trace!(
            "Current range: {}-{} ({}% overall, i={}, range length {})",
            self.current_range.start,
            self.current_range.end,
            self.i as f64 / self.repetition.get_hashes().len() as f64 * 100.0,
            self.i,
            self.current_range.len()
        );
        while self.current_range.end < self.repetition.get_hashes().len() {
            let hashes = self.repetition.get_hashes();
            let indices = self.repetition.get_indices();
            let range = self.current_range.clone();
            while self.i < range.end {
                while self.j < range.end {
                    assert!(range.contains(&self.i));
                    assert!(range.contains(&self.j));
                    let a = indices[self.i];
                    let b = indices[self.j];
                    let ha = hashes[self.i];
                    let hb = hashes[self.j];
                    debug_assert!(ha.prefix_eq(&hb, self.prefix));
                    if
                    // did the points collide previously?
                    !(self
                        .prev_prefix
                        .map(|pp| ha.prefix_eq(&hb, pp))
                        .unwrap_or(false))
                        &&
                        // are the corresponding subsequences overlapping?
                        // this check also excludes the flat subsequences, whose ID is replaced
                        // with u32::MAX, and thus always overlaps
                        !a.overlaps(b, exclusion_zone)
                    {
                        output[idx] = (a.min(b), a.max(b), Distance(f64::INFINITY));
                        idx += 1;
                    }
                    self.j += 1;
                    if idx >= output.len() {
                        return Some(idx);
                    }
                }
                self.i += 1;
                self.j = self.i + 1;
            }

            self.next_range();
        }
        if idx == 0 {
            None
        } else {
            Some(idx)
        }
    }

    /// how many collisions, including trivial ones, occur in this enumerator
    pub fn count_collisions(mut self) -> usize {
        let mut cnt = 0;
        while self.current_range.end < self.repetition.get_hashes().len() {
            let range = self.current_range.clone();
            cnt += range.len() * (range.len() - 1) / 2;
            self.next_range();
        }
        cnt
    }

    /// how many collisions, excluding trivial ones, occur in this enumerator?
    /// **Warning**: takes a long time on short prefixes
    pub fn count_non_trivial_collisions(mut self, exclusion_zone: usize) -> usize {
        let mut cnt = 0;
        while self.current_range.end < self.repetition.get_hashes().len() {
            let range = self.current_range.clone();
            for i in range.clone() {
                for j in i..range.end {
                    if !self.repetition.indices[i]
                        .overlaps(self.repetition.indices[j], exclusion_zone)
                    {
                        cnt += 1;
                    }
                }
            }
            self.next_range();
        }
        cnt
    }

    pub fn estimate_num_collisions(mut self, exclusion_zone: usize) -> usize {
        let mut cnt = 0;
        while self.current_range.end < self.repetition.get_hashes().len() {
            let hashes = self.repetition.get_hashes();
            let indices = self.repetition.get_indices();

            let range = self.current_range.clone();
            if range.len() as f64 > (hashes.len() as f64).sqrt() {
                // the bucket is _very_ large (relative to the number of subsequences),
                // so we just pick the square of its size in order to avoid spending
                // forever in iterating over pairs checking for overlaps
                cnt += range.len() * (range.len() - 1) / 2;
            } else {
                let mut i = range.start;
                while i < range.end {
                    let mut j = i + 1;
                    while j < range.end {
                        let a = indices[i];
                        let b = indices[j];
                        if !a.overlaps(b, exclusion_zone) {
                            cnt += 1;
                        }
                        j += 1;
                    }
                    i += 1;
                }
            }

            self.next_range();
        }

        cnt
    }
}

/// A few statistics on the LSH index so to be able to estimate the cost of
/// running repetitions at a given prefix length.
// #[derive(Debug, Clone)]
// // TODO: Maybe these stats are no longer needed
// #[deprecated]
// pub struct IndexStats {
//     /// the expected number of collisions for each prefix length
//     expected_collisions: Vec<f64>,
//     /// the cost of setting up a repetition
//     repetition_setup_cost: f64,
//     /// the cost of evaluating a collision
//     collision_cost: f64,
//     /// the maximum number of repetitions, fitting in the memory limit
//     pub max_repetitions: usize,
// }
// impl IndexStats {
//     fn new(
//         index: &LSHIndex,
//         ts: &WindowedTimeseries,
//         max_repetitions: usize,
//         exclusion_zone: usize,
//     ) -> Self {
//         assert!(max_repetitions >= 4);

//         let repetition_setup_cost = index.repetitions_setup_time.as_secs_f64();
//         let t = Instant::now();
//         let expected_collisions = index.collision_profile();
//         info!(
//             "Collisions profile ({:?}): {:?}",
//             t.elapsed(),
//             expected_collisions
//         );

//         // get the prefix at which we are going to sample to estimate the
//         // cost of running a repetition
//         let (mut sampling_prefix, _collisions) = expected_collisions
//             .iter()
//             .copied()
//             .enumerate()
//             .rev()
//             .find(|(_, collisions)| *collisions > 1000.0)
//             .unwrap_or((1, f64::INFINITY));
//         sampling_prefix = sampling_prefix.max(1);
//         assert!(sampling_prefix > 0);

//         let max_samples = 10_000; // cap the maximum work we are going to spend on sampling
//         let mut buf = vec![(0, 0, Distance(0.0)); 65536];
//         let mut enumerator = index.collisions(0, sampling_prefix, None);
//         let mut samples = 0;
//         let timer = Instant::now();
//         while let Some(cnt) = enumerator.next(&mut buf, exclusion_zone) {
//             for (a, b, d) in &mut buf[..cnt] {
//                 *d =
//                     Distance(zeucl_threshold(ts, *a as usize, *b as usize, f64::INFINITY).unwrap());
//                 samples += 1;
//                 if samples > max_samples {
//                     break;
//                 }
//             }
//         }
//         let elapsed = timer.elapsed();
//         let collision_cost = elapsed.as_secs_f64() / (samples as f64);
//         assert!(!collision_cost.is_nan());
//         assert!(!repetition_setup_cost.is_nan());

//         Self {
//             expected_collisions,
//             repetition_setup_cost,
//             collision_cost,
//             max_repetitions,
//         }
//     }

//     pub fn expected_collisions(&self) -> &[f64] {
//         &self.expected_collisions
//     }

//     pub fn first_meaningful_prefix(&self) -> usize {
//         self.expected_collisions
//             .iter()
//             .enumerate()
//             .rev()
//             .find(|(_prefix, collisions)| **collisions >= 1.0)
//             .unwrap()
//             .0
//             .max(1)
//     }

//     pub fn repetition_cost_estimate(&self, prefix: usize) -> f64 {
//         self.expected_collisions[prefix] * self.collision_cost
//     }
//     pub fn repetition_setup_estimate(&self) -> f64 {
//         self.repetition_setup_cost
//     }

//     /// For each prefix length, compute the cost to confirm a pair at
//     /// the given distance.
//     #[deprecated]
//     fn costs_to_confirm(
//         &self,
//         max_prefix: usize,
//         d: Distance,
//         delta: f64,
//         index: &LSHIndex,
//     ) -> Vec<(f64, usize)> {
//         let index_repetitions = index.get_repetitions();
//         // let p = hasher.collision_probability_at(d.0);
//         self.expected_collisions[..=max_prefix]
//             .iter()
//             .enumerate()
//             .map(|(prefix, collisions)| {
//                 if prefix == 0 {
//                     return (f64::INFINITY, 0);
//                 }
//                 let maxreps = self.max_repetitions;
//                 let (nreps, fp) = {
//                     let mut nreps = 0;
//                     let mut fp = 1.0;
//                     while fp > delta && nreps < maxreps {
//                         fp = index.failure_probability(d, nreps, prefix, None);
//                         nreps += 1;
//                     }
//                     (nreps, fp)
//                 };
//                 if nreps >= maxreps {
//                     log::debug!(
//                         "distance {}, at prefix {} failure probability {} with all repetitions",
//                         d,
//                         prefix,
//                         fp
//                     );
//                     return (f64::INFINITY, maxreps);
//                 }
//                 let new_repetitions = if nreps <= index_repetitions {
//                     0
//                 } else {
//                     nreps - index_repetitions
//                 } as f64;
//                 (
//                     nreps as f64
//                         * (self.collision_cost * collisions
//                             + self.repetition_setup_cost * new_repetitions),
//                     nreps,
//                 )
//             })
//             .collect()
//     }
// }

#[test]
fn test_collision_probability() {
    let ts: Vec<f64> = crate::load::loadts("data/ecg-heartbeat-av.csv", None).unwrap();
    let w = 100;
    let ts = WindowedTimeseries::new(ts, w, false);

    let ids = vec![
        1308, 1434, 1519, 1626, 1732, 1831, 1938, 2034, 2118, 2227, 2341, 2415, 2510, 2607, 2681,
        2787,
    ];

    let fft_data = FFTData::new(&ts);
    let mut index = LSHIndex::from_ts(&ts, w / 2, &fft_data, Bytes::gbytes(2), 1234);
    index.add_repetitions(&ts, &fft_data, 4096, K);

    for &i in &ids {
        for &j in &ids {
            if i < j {
                dbg!(i, j);
                let d = Distance(crate::distance::zeucl(&ts, i, j));
                let p = index.collision_probability_at(d);

                dbg!(d);
                dbg!(p);
                dbg!(index.empirical_collision_probability(i, j, 1));
                dbg!(index.failure_probability(d, index.get_repetitions(), 1, None));

                // assert!(pool.first_collision(i, j, 1).is_some());
            }
        }
    }
}
