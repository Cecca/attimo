use log::info;
use rand::prelude::*;
use rand_distr::{Normal, Uniform};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use std::{
    mem::size_of,
    ops::Range,
    time::{Duration, Instant},
};

use crate::{
    distance::zeucl,
    knn::Distance,
    timeseries::{Bytes, FFTData, Overlaps, WindowedTimeseries},
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
        // Clear byte
        self.0 &= !(0xFFu64 << ((size_of::<u8>() * k) * 8));
        // Set byte
        self.0 |= (byte as u64) << ((size_of::<u8>() * k) * 8);
    }

    fn prefix_eq(&self, other: &Self, prefix: usize) -> bool {
        assert!(prefix > 0);
        let mask = MASKS[prefix];
        self.0 & mask == other.0 & mask
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
    assert_eq!(h, HashValue(0x0000000000000000AAu64));
    h.set_byte(1, 0xBB);
    assert_eq!(h, HashValue(0x00000000000000BBAAu64));
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

        fn sample_vec<R: Rng>(dimension: usize, rng: &mut R) -> Vec<f64> {
            let normal = Normal::new(0.0, 1.0).expect("problem instantiating normal distribution");
            normal
                .sample_iter(rng)
                .take(dimension)
                .collect::<Vec<f64>>()
        }
        let vectors = [
            sample_vec(dimension, rng),
            sample_vec(dimension, rng),
            sample_vec(dimension, rng),
            sample_vec(dimension, rng),
            sample_vec(dimension, rng),
            sample_vec(dimension, rng),
            sample_vec(dimension, rng),
            sample_vec(dimension, rng),
        ];
        Self {
            vectors,
            shifts,
            width,
        }
    }

    pub fn collision_probability_at(&self, d: Distance) -> f64 {
        use statrs::distribution::{ContinuousCDF, Normal as NormalDistr};
        let d = d.0;
        let r = self.width;
        let normal = NormalDistr::new(0.0, 1.0).unwrap();
        1.0 - 2.0 * normal.cdf(-r / d)
            - (2.0 / ((std::f64::consts::PI * 2.0).sqrt() * (r / d)))
                * (1.0 - (-r * r / (2.0 * d * d)).exp())
    }

    pub fn compute_width(ts: &WindowedTimeseries) -> f64 {
        let n = ts.num_subsequences();
        let subsequence_norm = (ts.w as f64).sqrt();
        let expected_max_dotp = subsequence_norm * (2.0 * (n as f64).ln()).sqrt();
        expected_max_dotp / 128.0
    }

    /// Hash all the subsequences of the time series to 8 x 8-bit hash values each
    fn hash(&self, ts: &WindowedTimeseries, fft_data: &FFTData, output: &mut [(HashValue, u32)]) {
        assert_eq!(ts.num_subsequences(), output.len());
        for k in 0..K {
            ts.znormalized_sliding_dot_product_for_each(fft_data, &self.vectors[k], |i, mut h| {
                h = (h + self.shifts[k]) / self.width;
                //// Count if the value is out of bounds to be repre
                if h.abs() > 128.0 {
                    h = h.signum() * 127.0;
                }
                let h = ((h as i64 & 0xFFi64) as i8) as u8;
                output[i].0.set_byte(k, h);
                output[i].1 = i as u32;
            });
        }
    }
}

pub struct LSHIndex {
    rng: Xoshiro256PlusPlus,
    functions: Vec<Hasher>,
    hashes: Vec<Vec<HashValue>>,
    indices: Vec<Vec<u32>>,
}

impl LSHIndex {
    /// How much memory would it be required to store information for these many repetitions?
    pub fn required_memory(ts: &WindowedTimeseries, repetitions: usize) -> Bytes {
        let hashes = repetitions * K * ts.num_subsequences() * size_of::<HashValue>();
        let indices = repetitions * K * ts.num_subsequences() * size_of::<u32>();
        Bytes(hashes + indices)
    }

    /// With this function we can construct a `HashCollection` from a `WindowedTimeseries`
    /// and a `Hasher`.
    pub fn from_ts(ts: &WindowedTimeseries, fft_data: &FFTData, seed: u64) -> Self {
        let rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        const INITIAL_REPETITIONS: usize = 8;

        let mut slf = Self {
            rng,
            functions: Vec::new(),
            hashes: Vec::new(),
            indices: Vec::new(),
        };

        slf.add_repetitions(ts, fft_data, INITIAL_REPETITIONS);
        slf
    }

    pub fn add_repetitions(
        &mut self,
        ts: &WindowedTimeseries,
        fft_data: &FFTData,
        total_repetitions: usize,
    ) {
        assert!(total_repetitions > self.get_repetitions());
        let dimension = ts.w;
        let n = ts.num_subsequences();
        let width = Hasher::compute_width(ts);
        let new_repetitions = total_repetitions - self.get_repetitions();

        let new_hashers: Vec<Hasher> = (0..new_repetitions)
            .map(|_| Hasher::new(dimension, width, &mut self.rng))
            .collect();

        let (new_hashes, new_indices): (Vec<Vec<HashValue>>, Vec<Vec<u32>>) = new_hashers
            .par_iter()
            .map_with(Vec::new(), |tmp, hasher| {
                tmp.resize(n, (HashValue::default(), 0u32));
                hasher.hash(ts, fft_data, tmp);
                tmp.sort();
                let res: (Vec<HashValue>, Vec<u32>) = tmp.iter().copied().unzip();
                res
            })
            .unzip();

        self.functions.extend(new_hashers);
        self.hashes.extend(new_hashes);
        self.indices.extend(new_indices);
    }

    /// Get how many repetitions are available in this index
    pub fn get_repetitions(&self) -> usize {
        self.hashes.len()
    }

    fn collision_probability_at(&self, d: Distance) -> f64 {
        self.functions[0].collision_probability_at(d)
    }

    #[cfg(test)]
    fn empirical_collision_probability(&self, i: usize, j: usize, prefix: usize) -> f64 {
        let mut cnt = 0;
        for (hs, idxs) in self.hashes.iter().zip(&self.indices) {
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
        prev_prefix_repetitions: Option<usize>,
    ) -> f64 {
        let p = self.collision_probability_at(d);
        let cur_fail = (1.0 - p.powi(prefix as i32)).powi(reps as i32);
        let prev_fail = prev_prefix
            .zip(prev_prefix_repetitions)
            .map(|(prefix, repetitions)| {
                (1.0 - p.powi(prefix as i32)).powi(0i32.max(repetitions as i32 - reps as i32))
            })
            .unwrap_or(1.0);
        cur_fail * prev_fail
    }

    pub fn stats(
        &self,
        ts: &WindowedTimeseries,
        max_memory: Bytes,
        exclusion_zone: usize,
    ) -> IndexStats {
        IndexStats::new(self, ts, max_memory, exclusion_zone)
    }

    pub fn collisions(&self, repetition: usize, prefix: usize) -> CollisionEnumerator {
        assert!(prefix > 0 && prefix <= K, "illegal prefix {}", prefix);
        CollisionEnumerator::new(&self.hashes[repetition], &self.indices[repetition], prefix)
    }
}

pub struct CollisionEnumerator<'index> {
    prefix: usize,
    hashes: &'index [HashValue],
    indices: &'index [u32],
    current_range: Range<usize>,
    i: usize,
    j: usize,
}
impl<'index> CollisionEnumerator<'index> {
    fn new(hashes: &'index [HashValue], indices: &'index [u32], prefix: usize) -> Self {
        assert!(prefix > 0 && prefix <= K);
        let mut slf = Self {
            prefix,
            hashes,
            indices,
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
        // exponential search
        let start = self.current_range.end;
        let h = self.hashes[start];
        let mut offset = 1;
        let mut low = start;
        if low >= self.hashes.len() {
            self.current_range = low..low;
            return;
        }
        while start + offset < self.hashes.len()
            && self.hashes[start + offset].prefix_eq(&h, self.prefix)
        {
            low = start + offset;
            offset *= 2;
        }
        let high = (start + offset).min(self.hashes.len());

        // binary search
        debug_assert!(
            self.hashes[low].prefix_eq(&h, self.prefix),
            "{:?} != {:?} (prefix {})",
            self.hashes[low],
            h,
            self.prefix
        );
        let off = self.hashes[low..high].partition_point(|hv| h.prefix_eq(hv, self.prefix));
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
        output.fill((0, 0, Distance(0.0)));
        while self.current_range.end < self.hashes.len() {
            let range = self.current_range.clone();
            while self.i < range.end {
                while self.j < range.end {
                    assert!(range.contains(&self.i));
                    assert!(range.contains(&self.j));
                    let a = self.indices[self.i];
                    let b = self.indices[self.j];
                    let ha = self.hashes[self.i];
                    let hb = self.hashes[self.j];
                    assert!(ha.prefix_eq(&hb, self.prefix));
                    if !a.overlaps(b, exclusion_zone) {
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

    pub fn estimate_num_collisions(mut self, exclusion_zone: usize) -> usize {
        let mut cnt = 0;
        while self.current_range.end < self.hashes.len() {
            let range = self.current_range.clone();
            if range.len() as f64 > (self.hashes.len() as f64).sqrt() {
                // the bucket if _very_ large (relative to the number of subsequences),
                // so we just pick the square of its size in order to avoid spending
                // forever in iterating over pairs checking for overlaps
                log::trace!("Large bucket detected: {}", range.len());
                cnt += range.len() * range.len();
            } else {
                let mut i = range.start;
                while i < range.end {
                    let mut j = i + 1;
                    while j < range.end {
                        let a = self.indices[i];
                        let b = self.indices[j];
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
#[derive(Debug, Clone)]
pub struct IndexStats {
    /// the expected number of collisions for each prefix length
    expected_collisions: Vec<f64>,
    /// the cost of setting up a repetition
    repetition_setup_cost: f64,
    /// the cost of evaluating a collision
    collision_cost: f64,
    /// the maximum number of repetitions, fitting in the memory limit
    max_repetitions: usize,
}
impl IndexStats {
    fn new(
        index: &LSHIndex,
        ts: &WindowedTimeseries,
        max_memory: Bytes,
        exclusion_zone: usize,
    ) -> Self {
        let mut max_repetitions = 4.min(index.get_repetitions());
        while LSHIndex::required_memory(ts, 2 * max_repetitions) <= max_memory {
            max_repetitions *= 2;
        }
        info!(
            "Maximum repetitions {} which would require {}",
            max_repetitions,
            LSHIndex::required_memory(ts, max_repetitions)
        );

        let nreps = 4;
        let mut repetition_setup_cost = 0.0;
        let repetition_setup_cost_experiments = nreps * K;
        let mut expected_collisions = vec![0.0; K + 1];
        let dat: Vec<(usize, usize, usize, Duration)> = (1..=K)
            .into_par_iter()
            .flat_map(|prefix| (0..nreps).into_par_iter().map(move |rep| (prefix, rep)))
            .map(|(prefix, rep)| {
                let start = Instant::now();
                let collisions = Self::num_collisions(index, rep, prefix, exclusion_zone);
                let elapsed = Instant::now() - start;
                (prefix, rep, collisions, elapsed)
            })
            .collect();
        for (prefix, _rep, collisions, duration) in dat {
            repetition_setup_cost += duration.as_secs_f64();
            expected_collisions[prefix] += collisions as f64;
        }
        for c in expected_collisions.iter_mut() {
            *c /= nreps as f64;
        }
        expected_collisions[0] = f64::INFINITY; // just to signal that we don't want to go to prefix 0
        repetition_setup_cost /= repetition_setup_cost_experiments as f64;

        // now we estimate the cost of running a handful of distance computations,
        // as a proxy for the cost of handling the collisions.
        // TODO: maybe update this as we collect information during the execution?
        let samples = 4096.min(ts.num_subsequences());
        let t_start = Instant::now();
        for i in 0..samples {
            std::hint::black_box(zeucl(ts, 0, i));
        }
        let elapsed = Instant::now() - t_start;
        let collision_cost = elapsed.as_secs_f64() / (samples as f64);
        assert!(!collision_cost.is_nan());
        assert!(!repetition_setup_cost.is_nan());

        Self {
            expected_collisions,
            repetition_setup_cost,
            collision_cost,
            max_repetitions,
        }
    }

    /// Counts the number of collision at a given repetition and prefix length,
    /// and the cost of setting up the repetition
    fn num_collisions(
        index: &LSHIndex,
        repetition: usize,
        prefix: usize,
        exclusion_zone: usize,
    ) -> usize {
        index
            .collisions(repetition, prefix)
            .estimate_num_collisions(exclusion_zone)
    }

    pub fn first_meaningful_prefix(&self) -> usize {
        self.expected_collisions
            .iter()
            .enumerate()
            .rev()
            .find(|(_prefix, collisions)| **collisions >= 1.0)
            .unwrap()
            .0
            .max(1)
    }

    /// For each prefix length, compute the cost to confirm a pair at
    /// the given distance.
    pub fn costs_to_confirm(
        &self,
        max_prefix: usize,
        d: Distance,
        delta: f64,
        index: &LSHIndex,
    ) -> Vec<(f64, usize)> {
        // let p = hasher.collision_probability_at(d.0);
        self.expected_collisions[..=max_prefix]
            .iter()
            .enumerate()
            .map(|(prefix, collisions)| {
                let maxreps = self.max_repetitions;
                let nreps = {
                    let mut nreps = 0;
                    let mut fp = 1.0;
                    while fp > delta && nreps < maxreps {
                        fp = index.failure_probability(d, nreps, prefix, None, None);
                        nreps += 1;
                    }
                    nreps
                };
                if nreps >= maxreps {
                    return (f64::INFINITY, maxreps);
                }
                let res = (
                    nreps as f64 * (self.collision_cost * collisions + self.repetition_setup_cost),
                    nreps,
                );
                assert!(!res.0.is_nan());
                res
            })
            .collect()
    }
}

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
    let mut index = LSHIndex::from_ts(&ts, &fft_data, 1234);
    dbg!(index.stats(&ts, Bytes::gbytes(8), w));
    index.add_repetitions(&ts, &fft_data, 4096);

    for &i in &ids {
        for &j in &ids {
            if i < j {
                dbg!(i, j);
                let d = Distance(zeucl(&ts, i, j));
                let p = index.collision_probability_at(d);

                dbg!(d);
                dbg!(p);
                dbg!(index.empirical_collision_probability(i, j, 1));
                dbg!(index.failure_probability(d, index.get_repetitions(), 1, None, None));

                // assert!(pool.first_collision(i, j, 1).is_some());
            }
        }
    }
}
