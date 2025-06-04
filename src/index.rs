use core::f64;
use log::{info, warn};
use rand::prelude::*;
use rand_distr::{Normal, Uniform};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use std::{
    fs::File,
    io::{prelude::*, BufReader, BufWriter},
    mem::size_of,
    ops::Range,
    path::PathBuf,
    time::{Duration, Instant},
};

use crate::{
    allocator::Bytes,
    distance::{zeucl, zeucl_threshold},
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
        info!("Expected max dot product {}", expected_max_dotp);
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
        info!(
            "dot product 1perc {} 99perc {} (min sample width: {}) max/min {}/{}",
            dotps[dotps.len() / 100].0,
            dotps[99 * dotps.len() / 100].0,
            min_sample_width,
            dotps.iter().min_by(|a, b| a.0.total_cmp(&b.0)).unwrap().0,
            dotps.iter().max_by(|a, b| a.0.total_cmp(&b.0)).unwrap().0,
        );
        info!(
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

impl Repetition {
    fn from_pairs<I: IntoIterator<Item = (HashValue, u32)>>(pairs: I) -> Self {
        let (hashes, indices): (Vec<HashValue>, Vec<u32>) = pairs
            .into_iter()
            // remove the overflown subsequences
            .filter(|(_, idx)| *idx != u32::MAX)
            .unzip();
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

pub struct LSHIndex {
    rng: Xoshiro256PlusPlus,
    quantization_width: f64,
    functions: Vec<Hasher>,
    repetitions: Vec<Repetition>,
    repetitions_setup_time: Duration,
    max_repetitions: usize,
}

pub const INITIAL_REPETITIONS: usize = 8;

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

        let mut qw = std::env::var("ATTIMO_QUANTIZATION")
            .map(|q| {
                q.parse::<f64>()
                    .expect("unable to parse ATTIMO_QUANTIZATION as a float")
            })
            .unwrap_or_else(|_| Hasher::compute_width(ts, fft_data, exclusion_zone, &mut rng));

        let mut max_repetitions = 0;
        while Self::required_memory(ts, max_repetitions) < max_memory {
            max_repetitions += 1;
        }
        info!(
            "initial quantization width: {}, maximum repetitions: {}",
            qw, max_repetitions
        );

        // Try to build the index until we get a version that has collisions at the deepest level
        let t = Instant::now();
        let mut qw_lower: Option<f64> = None;
        let mut qw_upper: Option<f64> = None;
        loop {
            if let (Some(lower), Some(upper)) = (qw_lower, qw_upper) {
                qw = (upper + lower) / 2.0;
            }
            let mut slf = Self {
                rng: rng.clone(),
                quantization_width: qw,
                functions: Vec::new(),
                repetitions: Vec::new(),
                repetitions_setup_time: Duration::from_secs(0),
                max_repetitions,
            };

            slf.add_repetitions(ts, fft_data, 1, K);

            let enumerator = slf.collisions(0, K, None);
            let num_collisions = enumerator.estimate_num_collisions(exclusion_zone);
            info!(
                "Num collisions with quantization_width={}: {} (lower {:?}, upper {:?})",
                qw, num_collisions, qw_lower, qw_upper
            );
            if num_collisions == 0 {
                // the quantization width is too small
                qw_lower.replace(qw);
                qw *= 2.0;
                info!("Doubling the quantization width to {}", qw);
            } else if num_collisions > sqrt_n {
                // the quantization width is too large: too many subsequences
                qw_upper.replace(qw);
                qw /= 2.0;
                info!("Halving the quantization width {}", qw);
            } else {
                info!("Settling on quantization width {}", qw);
                let avg_dur = slf.add_repetitions(ts, fft_data, INITIAL_REPETITIONS, K);
                slf.repetitions_setup_time = avg_dur;
                observe!(0, 0, "profile/index_setup", t.elapsed().as_secs_f64());
                return slf;
            }
        }
    }

    pub fn add_repetitions(
        &mut self,
        ts: &WindowedTimeseries,
        fft_data: &FFTData,
        total_repetitions: usize,
        max_k: usize,
    ) -> Duration {
        assert!(
            total_repetitions > self.get_repetitions(),
            "total_repetitions {} is not > self.get_repetitions() {}",
            total_repetitions,
            self.get_repetitions()
        );
        let dimension = ts.w;
        let n = ts.num_subsequences();
        let starting_repetitions = self.get_repetitions();
        let new_repetitions = total_repetitions - starting_repetitions;
        let max_repetitions_in_memory = self.max_repetitions;
        log::trace!("Adding {} new repetitions", new_repetitions);

        let new_hashers: Vec<Hasher> = (0..new_repetitions)
            .map(|_| Hasher::new(dimension, self.quantization_width, &mut self.rng))
            .collect();

        let timer = Instant::now();
        let mut tmp = Vec::new();
        let new_reps = new_hashers.iter().enumerate().map(|(i, hasher)| {
            let rep_idx = starting_repetitions + i;
            tmp.resize(n, (HashValue::default(), 0u32));
            hasher.hash(ts, fft_data, &mut tmp, max_k);
            tmp.par_sort_unstable_by_key(|pair| pair.0);
            if rep_idx > max_repetitions_in_memory {
                log::error!(
                    "Too many repetitions, maximum is {}",
                    max_repetitions_in_memory
                );
                panic!("Too many repetitions");
            } else {
                Repetition::from_pairs(tmp.iter().copied())
            }
        });
        self.repetitions.extend(new_reps);
        let elapsed = timer.elapsed();
        log::debug!("Added {} new repetitions in {:?}", new_repetitions, elapsed);
        let average_time = elapsed / new_repetitions as u32;

        self.functions.extend(new_hashers);

        average_time
    }

    /// Get how many repetitions are available in this index
    pub fn get_repetitions(&self) -> usize {
        self.repetitions.len()
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

    pub fn stats(&self, ts: &WindowedTimeseries, exclusion_zone: usize) -> IndexStats {
        IndexStats::new(self, ts, self.max_repetitions, exclusion_zone)
    }

    pub fn collisions(
        &self,
        repetition: usize,
        prefix: usize,
        prev_prefix: Option<usize>,
    ) -> CollisionEnumerator {
        assert!(prefix > 0 && prefix <= K, "illegal prefix {}", prefix);
        let rep = &self.repetitions[repetition];
        CollisionEnumerator::new(rep, prefix, prev_prefix)
    }

    /// Estimates, for each level in the given repetition index, the number of
    /// non-trivial collisions
    fn collision_profile_at(&self, repetition_idx: usize, exclusion_zone: usize) -> Vec<f64> {
        let repetition = &self.repetitions[repetition_idx];
        let hashes = repetition.get_hashes();
        let indices = repetition.get_indices();

        // let mut counts = vec![0.0; K + 1];
        // counts[0] = f64::INFINITY;

        // First we count, checking for overlaps, the collisions at the longest prefix, under the
        // assumption that most trivial collisions happen at the longest prefix
        let counts = (0..=K)
            .into_par_iter()
            .map(|prefix| {
                if prefix == 0 {
                    return (f64::INFINITY, 0);
                }
                let mut cnt = 0;
                let mut trivial = 0;
                let mut start = 0;
                while start < hashes.len() {
                    let end = start
                        + hashes[start..].partition_point(|h| h.prefix_eq(&hashes[start], prefix));
                    assert!(start < end);
                    if prefix == K {
                        let n = end - start;
                        let estimate_collisions = n * (n - 1) / 2;
                        let mut cnt_collisions = 0;
                        for i in start..end {
                            for j in start..i {
                                if !indices[i].overlaps(indices[j], exclusion_zone) {
                                    cnt_collisions += 1;
                                } else {
                                    trivial += 1;
                                }
                            }
                        }
                        cnt += cnt_collisions;
                    } else {
                        let n = end - start;
                        let estimate_collisions = n * (n - 1) / 2;
                        cnt += estimate_collisions;
                    }
                    start = end;
                }
                (cnt as f64, trivial)
            })
            .collect::<Vec<(f64, usize)>>();

        let trivial = counts.last().unwrap().1;
        let (mut counts, _): (Vec<f64>, Vec<usize>) = counts.into_iter().unzip();

        // Adjust the estimates by removing trivial collisions, except for the last one where
        // collisions have been counted exactly
        for c in counts[1..K].iter_mut() {
            *c -= trivial as f64;
            assert!(*c >= 0.0);
        }

        counts
    }

    fn collision_profile(&self, exclusion_zone: usize) -> Vec<f64> {
        let reps = self.get_repetitions().min(4);
        let mut counts = (0..reps)
            .into_par_iter()
            .map(|rep| {
                let mut out = vec![0.0; K + 1];
                for prefix in 1..=K {
                    let enumerator = CollisionEnumerator::new(&self.repetitions[rep], prefix, None);
                    out[prefix] = enumerator.estimate_num_collisions(exclusion_zone) as f64;
                }
                out
            })
            .reduce(
                || vec![0.0; K + 1],
                |mut a, b| {
                    for (aa, bb) in a.iter_mut().zip(b) {
                        *aa += bb;
                    }
                    a
                },
            );
        counts[0] = f64::INFINITY;

        for acc in counts.iter_mut() {
            *acc /= reps as f64;
        }

        counts
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
        check_prefix: bool,
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
                    !(check_prefix && self
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
#[derive(Debug, Clone)]
pub struct IndexStats {
    /// the expected number of collisions for each prefix length
    expected_collisions: Vec<f64>,
    /// the cost of setting up a repetition
    repetition_setup_cost: f64,
    /// the cost of evaluating a collision
    collision_cost: f64,
    /// the maximum number of repetitions, fitting in the memory limit
    pub max_repetitions: usize,
}
impl IndexStats {
    fn new(
        index: &LSHIndex,
        ts: &WindowedTimeseries,
        max_repetitions: usize,
        exclusion_zone: usize,
    ) -> Self {
        assert!(max_repetitions >= 4);

        let repetition_setup_cost = index.repetitions_setup_time.as_secs_f64();
        let t = Instant::now();
        let expected_collisions = index.collision_profile(exclusion_zone);
        info!(
            "Collisions profile ({:?}): {:?}",
            t.elapsed(),
            expected_collisions
        );

        // get the prefix at which we are going to sample to estimate the
        // cost of running a repetition
        let (sampling_prefix, _collisions) = expected_collisions
            .iter()
            .skip(1)
            .copied()
            .enumerate()
            .rev()
            .find(|(_, collisions)| *collisions > 1000.0)
            .unwrap_or((1, f64::INFINITY));

        let max_samples = 10_000; // cap the maximum work we are going to spend on sampling
        let mut buf = vec![(0, 0, Distance(0.0)); 65536];
        let mut enumerator = index.collisions(0, sampling_prefix, None);
        let mut samples = 0;
        let timer = Instant::now();
        while let Some(cnt) = enumerator.next(&mut buf, exclusion_zone, true) {
            for (a, b, d) in &mut buf[..cnt] {
                *d =
                    Distance(zeucl_threshold(ts, *a as usize, *b as usize, f64::INFINITY).unwrap());
                samples += 1;
                if samples > max_samples {
                    break;
                }
            }
        }
        let elapsed = timer.elapsed();
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

    pub fn repetition_cost_estimate(&self, prefix: usize) -> f64 {
        self.expected_collisions[prefix] * self.collision_cost
    }
    pub fn repetition_setup_estimate(&self) -> f64 {
        self.repetition_setup_cost
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
        let index_repetitions = index.get_repetitions();
        // let p = hasher.collision_probability_at(d.0);
        self.expected_collisions[..=max_prefix]
            .iter()
            .enumerate()
            .map(|(prefix, collisions)| {
                if prefix == 0 {
                    return (f64::INFINITY, 0);
                }
                let maxreps = self.max_repetitions;
                let (nreps, fp) = {
                    let mut nreps = 0;
                    let mut fp = 1.0;
                    while fp > delta && nreps < maxreps {
                        fp = index.failure_probability(d, nreps, prefix, None, None);
                        nreps += 1;
                    }
                    (nreps, fp)
                };
                if nreps >= maxreps {
                    log::debug!(
                        "distance {}, at prefix {} failure probability {} with all repetitions",
                        d,
                        prefix,
                        fp
                    );
                    return (f64::INFINITY, maxreps);
                }
                let new_repetitions = if nreps <= index_repetitions {
                    0
                } else {
                    nreps - index_repetitions
                } as f64;
                (
                    nreps as f64
                        * (self.collision_cost * collisions
                            + self.repetition_setup_cost * new_repetitions),
                    nreps,
                )
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
    let mut index = LSHIndex::from_ts(&ts, w / 2, &fft_data, Bytes::gbytes(2), 1234);
    index.add_repetitions(&ts, &fft_data, 4096, K);

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
