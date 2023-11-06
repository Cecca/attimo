//// # Locality Sensitive Hashing
////
//// The purpose of Locality Sensitive Hashing (LSH for short) is to speed up nearest neighbor
//// queries, resulting in potentially sublinear query times. The technique can also be used
//// for similarity joins, like we do in this work.
////
//// The intuition is simple: for a given distance function between points (which in our case
//// is the z-normalized Euclidean distance between subsequences), we choose a _family_ of hash
//// functions. From this family we sample an appropriate number of functions (more on this later).
//// For a given subsequence of our time series, each of the sampled functions will output a
//// _hash value_, whose domain depends on the family of hash functions.
////
//// A LSH scheme for the Euclidean distance is
//// described in [this paper](http://theory.csail.mit.edu/~mirrokni/pstable.ps).
//// The idea is rather simple: for each input vector (in our case the subsequences of the input time
//// series) we compute the inner product with a random vector, whose components are distributed
//// according to the p-stable distribution associated with the distance. For the Euclidean
//// distance such distribution is the Standard Normal. The result is then bucketed into bins whose
//// width is a parameter of the algorithm (we shall later see how to estimate this parameter automatically).
////
//// The nice property of this approach, when used with time series, is that for a given random vector
//// we can compute all the dot products with every subsequence of the time series in one go using the
//// same trick of [MASS](https://www.cs.unm.edu/~mueen/FastestSimilaitySearch.html).
//// The idea is to use the [cyclic convolution theorem](http://www.dei.unipd.it/~geppo/DA2/DOCS/FFT.pdf)
//// to perform the dot products by means of element-wise multiplication in the frequency domain.
//// As such, the dominant component of the complexity is the `O(n log n)` of the Fast Fourier Transform:
//// we save a factor `w` in the complexity, where `w` is the motif length.

use crate::knn::OrdF64;
use crate::motifs::Motif;

// TODO Remove this dependency
use crate::sort::*;
use crate::timeseries::{FFTData, Overlaps, WindowedTimeseries};
use rand::prelude::*;
use rand_distr::{Normal, Uniform};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use slog_scope::info;
use statrs::distribution::{ContinuousCDF, Normal as NormalDistr};
use std::ops::Range;
use std::time::Duration;
use std::{cell::UnsafeCell, sync::Arc, time::Instant};

pub const K: usize = 8;
pub const K_HALF: usize = K / 2;

#[derive(Debug, Ord, PartialOrd, Eq, PartialEq, Copy, Clone, Default)]
pub struct HashValue(pub u32);

impl GetByte for HashValue {
    fn num_bytes(&self) -> usize {
        4
    }
    #[inline(always)]
    fn get_byte(&self, i: usize) -> u8 {
        (self.0 >> (8 * (std::mem::size_of::<u32>() - i - 1)) & 0xFF) as u8
    }
}

#[derive(Default)]
pub struct ColumnBuffers {
    pub hashes: Vec<(HashValue, u32)>,
    pub buckets: Vec<Range<usize>>,
}
impl ColumnBuffers {
    pub fn enumerator(&self) -> Option<CollisionEnumerator<'_>> {
        let current_bucket = 0;
        if self.buckets.len() > 0 {
            let range = self.buckets[0].clone();
            let i = range.start;
            let j = i + 1;
            Some(CollisionEnumerator {
                current_bucket,
                i,
                j,
                buffers: self,
            })
        } else {
            None
        }
    }
}

pub struct CollisionEnumerator<'hashes> {
    current_bucket: usize,
    i: usize,
    j: usize,
    buffers: &'hashes ColumnBuffers,
}
impl<'hashes> CollisionEnumerator<'hashes> {
    /// Fills the given output buffer with colliding pairs, and returns the number
    /// of pairs that have been put into the buffer. If there were no pairs to add,
    /// return `None`.
    /// The third `f64` element is there just as a placeholder, which will be initialized
    /// as `f64::INFINITY`: actual distances must be computed somewhere else
    pub fn next(
        &mut self,
        output: &mut [(u32, u32, OrdF64)],
        exclusion_zone: usize,
    ) -> Option<usize> {
        let mut idx = 0;
        while self.current_bucket < self.buffers.buckets.len() {
            let range = self.buffers.buckets[self.current_bucket].clone();
            while self.i < range.end {
                while self.j < range.end {
                    assert!(range.contains(&self.i));
                    assert!(range.contains(&self.j));
                    let a = self.buffers.hashes[self.i].1;
                    let b = self.buffers.hashes[self.j].1;
                    if !a.overlaps(b, exclusion_zone) {
                        output[idx] = (a.min(b), a.max(b), OrdF64(f64::INFINITY));
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

            self.current_bucket += 1;
            if self.current_bucket < self.buffers.buckets.len() {
                let range = self.buffers.buckets[self.current_bucket].clone();
                self.i = range.start;
                self.j = range.start + 1;
            } else {
                if idx == 0 {
                    return None;
                } else {
                    return Some(idx);
                }
            }
        }
        if idx == 0 {
            return None;
        } else {
            return Some(idx);
        }
    }
}

#[test]
fn test_collision_enumerator() {
    let buf_size = 16;
    let buffers = ColumnBuffers::default();
    let enumerator = buffers.enumerator();
    assert!(enumerator.is_none());

    let mut buffers = ColumnBuffers::default();
    buffers.hashes.extend_from_slice(&[
        (HashValue(0), 0),
        (HashValue(0), 1),
        (HashValue(0), 2),
        (HashValue(0), 3),
    ]);
    let n = buffers.hashes.len();
    buffers.buckets.push(0..n);
    let mut enumerator = buffers.enumerator().unwrap();
    let mut buf = vec![(0, 0, OrdF64(f64::INFINITY)); buf_size];
    let ret = enumerator.next(&mut buf, 0);
    println!("{:?}", buf);
    assert_eq!(ret, Some(n * (n - 1) / 2));

    let mut buffers = ColumnBuffers::default();
    buffers.hashes.extend_from_slice(&[
        (HashValue(0), 0),
        (HashValue(0), 1),
        (HashValue(0), 2),
        (HashValue(0), 3),
        (HashValue(0), 4),
        (HashValue(0), 5),
        (HashValue(0), 6),
        (HashValue(0), 7),
    ]);
    let n = buffers.hashes.len();
    let tot_pairs = n * (n - 1) / 2;
    buffers.buckets.push(0..n);
    let mut enumerator = buffers.enumerator().unwrap();
    let mut buf = vec![(0, 0, OrdF64(f64::INFINITY)); buf_size];
    let mut enumerated = 0;
    while let Some(cnt) = enumerator.next(&mut buf, 0) {
        enumerated += cnt;
    }
    assert_eq!(enumerated, tot_pairs);

    let mut buffers = ColumnBuffers::default();
    buffers.hashes.extend_from_slice(&[
        (HashValue(0), 0),
        (HashValue(0), 1),
        (HashValue(0), 2),
        (HashValue(0), 3),
        (HashValue(1), 4),
        (HashValue(1), 5),
        (HashValue(1), 6),
        (HashValue(1), 7),
    ]);
    let tot_pairs = 4 * (4 - 1);
    buffers.buckets.push(0..4);
    buffers.buckets.push(4..8);
    let mut enumerator = buffers.enumerator().unwrap();
    let mut buf = vec![(0, 0, OrdF64(f64::INFINITY)); buf_size];
    let mut enumerated = 0;
    while let Some(cnt) = enumerator.next(&mut buf, 0) {
        enumerated += cnt;
    }
    assert_eq!(enumerated, tot_pairs);
}

//// This data structure is taken from [this StackOverflow answer](https://stackoverflow.com/questions/65178245/how-do-i-write-to-a-mutable-slice-from-multiple-threads-at-arbitrary-indexes-wit).
//// It is simply a wrapper around a mutable slice, providing (unsafe) concurrent mutable
//// access to its elements, without the need for synchronization primitives.
//// We use it only in one place, when building the hash pools, when accesses to the arrays
//// are by construction non-overlapping.
#[derive(Copy, Clone)]
pub struct UnsafeSlice<'a, T> {
    slice: &'a [UnsafeCell<T>],
}
unsafe impl<'a, T: Send + Sync> Send for UnsafeSlice<'a, T> {}
unsafe impl<'a, T: Send + Sync> Sync for UnsafeSlice<'a, T> {}

impl<'a, T> UnsafeSlice<'a, T> {
    pub fn new(slice: &'a mut [T]) -> Self {
        let ptr = slice as *mut [T] as *const [UnsafeCell<T>];
        Self {
            slice: unsafe { &*ptr },
        }
    }

    /// SAFETY: It is UB if two threads write to the same index without
    /// synchronization.
    pub unsafe fn write(&self, i: usize, value: T) {
        let ptr = self.slice[i].get();
        *ptr = value;
    }

    /// SAFETY: It is UB if two threads write to the same index without
    /// synchronization.
    pub unsafe fn update(&self, i: usize, mut f: impl FnMut(*mut T)) {
        let ptr = self.slice[i].get();
        f(ptr);
    }
}

fn get_minimal_index_pair(idx: usize) -> (usize, usize) {
    let sqrt = (idx as f64).sqrt() as usize;
    if idx == sqrt * sqrt + 2 * sqrt {
        (sqrt, sqrt)
    } else if idx >= sqrt * sqrt + sqrt {
        (sqrt, idx - (sqrt * sqrt + sqrt))
    } else {
        (idx - sqrt * sqrt, sqrt)
    }
}

fn get_minimal_repetition(repetitions: usize, i: usize, j: usize) -> Option<usize> {
    for rep in 0..repetitions {
        let pair = get_minimal_index_pair(rep);
        if pair == (i, j) {
            return Some(rep);
        }
    }
    None
}

#[test]
fn test_index_pair_round_trip() {
    let repetitions = 4096;
    for rep in 0..repetitions {
        let (i, j) = get_minimal_index_pair(rep);
        assert_eq!(rep, get_minimal_repetition(repetitions, i, j).unwrap());
    }
}

//// This data structure contains all the information needed to generate the hash values for all the repeititions
//// for all the subsequences.
#[derive(Clone)]
pub struct HashCollection {
    pub hasher: Arc<Hasher>,
    n_subsequences: usize,
    /// Pools of hash values, we have a vector for each tensor repetition, and each vector has an
    /// entry for the strings of K_HALF hash values, one for the left, and one for the right pool
    pools: Vec<Vec<([u8; K_HALF], [u8; K_HALF])>>,
}

impl HashCollection {
    //// LSH schemes usually have at least two parameters: the number of concatenations `K` and
    //// the number or repetitions `L`. We fix the first parameter in the constant `K` at the top of
    //// this source file. As for the second, with this function we allow the user to get an estimate of the
    //// memory that would be needed to run the given number or `repetitions`. Hence,
    //// it can be used to tune the LSH data structure to the available memory.
    pub fn required_memory(ts: &WindowedTimeseries, repetitions: usize) -> usize {
        let tensor_repetitions = (repetitions as f64).sqrt().ceil() as usize;
        //// This is the memory required by the hash pools
        let mem_pools = tensor_repetitions * K_HALF * ts.num_subsequences() * 2;
        //// And this is the memory eventually required by the hash matrix, when all the columns are materialized
        let mem_matrix = ts.num_subsequences() * (K + std::mem::size_of::<usize>()) * repetitions;

        mem_pools + mem_matrix
    }

    //// With this function we can construct a `HashCollection` from a `WindowedTimeseries`
    //// and a `Hasher`.
    pub fn from_ts(ts: &WindowedTimeseries, fft_data: &FFTData, hasher: Arc<Hasher>) -> Self {
        assert!(ts.num_subsequences() < u32::MAX as usize, "We use 32 bit integers as pointers into subsequences, this timeseries has too many subsequences.");
        let ns = ts.num_subsequences();

        let timer = Instant::now();

        let pools = (0..hasher.tensor_repetitions)
            .into_par_iter()
            .map(|trep| {
                let mut repdata = vec![([0u8; K_HALF], [0u8; K_HALF]); ns];
                let mut max_dotp = 0.0f64;
                for k in 0..K_HALF {
                    let mdp1 = hasher.hash_all(&ts, fft_data, k, trep, |i, h| {
                        let h = ((h as i64 & 0xFFi64) as i8) as u8;
                        repdata[i].0[k] = h;
                    });
                    let mdp2 = hasher.hash_all(&ts, fft_data, k + K_HALF, trep, |i, h| {
                        let h = ((h as i64 & 0xFFi64) as i8) as u8;
                        repdata[i].1[k] = h;
                    });
                    max_dotp = max_dotp.max(mdp1.max(mdp2));
                }
                repdata
            })
            .collect();

        let elapsed = timer.elapsed();
        info!("tensor pool building";
            "tag" => "profiling",
            "time_s" => elapsed.as_secs_f64()
        );

        Self {
            hasher,
            n_subsequences: ns,
            pools,
        }
    }

    #[deprecated]
    pub fn left(&self, i: usize, repetition: usize) -> &[u8] {
        let trep = repetition % self.hasher.tensor_repetitions;
        &self.pools[trep][i].0
    }

    #[deprecated]
    pub fn right(&self, i: usize, repetition: usize) -> &[u8] {
        let trep = repetition / self.hasher.tensor_repetitions;
        &self.pools[trep][i].1
    }

    pub fn half_hashes(&self, i: usize, repetition: usize) -> (&[u8], &[u8]) {
        let (l_trep, r_trep) = get_minimal_index_pair(repetition);
        let l = &self.pools[l_trep][i].0;
        let r = &self.pools[r_trep][i].1;
        (l, r)
    }

    #[cfg(test)]
    pub fn extended_hash_value(&self, i: usize, repetition: usize) -> [u8; K] {
        let mut output = [0; K];
        let (l, r) = self.half_hashes(i, repetition);
        let mut h = 0;
        while h < K_HALF {
            output[2 * h] = l[h];
            output[2 * h + 1] = r[h];
            h += 1;
        }
        output
    }

    pub fn k_pair(k: usize) -> (usize, usize) {
        assert!(k <= K);
        let k_left = (k as f64 / 2.0).ceil() as usize;
        let k_right = (k as f64 / 2.0).floor() as usize;
        (k_left, k_right)
    }

    pub fn hash_value(&self, i: usize, prefix: usize, repetition: usize) -> HashValue {
        let mut hv: [u8; 32] = [0; 32];
        let (l, r) = self.half_hashes(i, repetition);
        for h in 0..usize::div_ceil(prefix, 2) {
            hv[2 * h] = l[h];
            hv[2 * h + 1] = r[h];
        }
        HashValue(xxhash_rust::xxh32::xxh32(&hv[..prefix], 1234))
    }

    #[cfg(test)]
    pub fn first_collision_baseline(&self, i: usize, j: usize, prefix: usize) -> Option<usize> {
        (0..self.hasher.repetitions)
            .filter(|&rep| {
                let ihash = &self.extended_hash_value(i, rep)[0..prefix];
                let jhash = &self.extended_hash_value(j, rep)[0..prefix];
                ihash == jhash
            })
            .next()
    }

    pub fn first_collision(&self, i: usize, j: usize, prefix: usize) -> Option<usize> {
        let (k_left, k_right) = Self::k_pair(prefix);

        let mut lindex = None;
        let mut rindex = None;
        for trep in 0..self.hasher.tensor_repetitions {
            let (ipool_left, ipool_right) = &self.pools[trep][i];
            let (jpool_left, jpool_right) = &self.pools[trep][j];
            // check left
            if lindex.is_none() {
                let hi = &ipool_left[0..k_left];
                let hj = &jpool_left[0..k_left];
                if hi == hj {
                    lindex = Some(trep);
                }
            }
            // check right
            if rindex.is_none() {
                let hi = &ipool_right[0..k_right];
                let hj = &jpool_right[0..k_right];
                if hi == hj {
                    rindex = Some(trep);
                }
            }
            if rindex.is_some() && lindex.is_some() {
                break;
            }
        }
        let res =
            get_minimal_repetition(self.hasher.repetitions, lindex?, rindex?).and_then(|rep| {
                if rep > self.hasher.repetitions {
                    None
                } else {
                    Some(rep)
                }
            });

        #[cfg(test)]
        {
            assert_eq!(
                res,
                self.first_collision_baseline(i, j, prefix),
                "\nrepetitions: {:?}\nk_left: {:?}\nk_right: {:?}\nprefix: {}\nlindex: {:?}\nrindex: {:?}\nactual minimal index pair: {:?}",
                self.hasher.repetitions,
                k_left,
                k_right,
                prefix,
                lindex,
                rindex,
                get_minimal_index_pair(self.first_collision_baseline(i, j, prefix).unwrap())
            );
        }

        res
    }

    pub fn group_subsequences(
        &self,
        depth: usize,
        repetition: usize,
        exclusion_zone: usize,
        buffers: &mut ColumnBuffers,
        parallel: bool,
    ) -> () {
        let ns = self.n_subsequences;
        let buffer = &mut buffers.hashes;
        let output = &mut buffers.buckets;

        buffer.clear();
        output.clear();

        let start = Instant::now();
        if parallel {
            buffer.par_extend(
                (0..ns)
                    .into_par_iter()
                    .map(|i| (self.hash_value(i, depth, repetition), i as u32)),
            );
        } else {
            buffer.extend((0..ns).map(|i| (self.hash_value(i, depth, repetition), i as u32)));
        }

        let elapsed_hashes = start.elapsed();
        let start = Instant::now();
        if parallel {
            buffer.par_sort_unstable();
        } else {
            buffer.sort_unstable();
        }
        let elapsed_sort = start.elapsed();
        debug_assert!(buffer.is_sorted_by_key(|pair| pair.0.clone()));

        let mut largest_bucket = 0;
        let timer = Instant::now();
        let mut idx = 0;
        while idx < buffer.len() {
            let start = idx;
            let current: HashValue = buffer[idx].0;
            let mut min_i = buffer[idx].1 as usize;
            let mut max_i = buffer[idx].1 as usize;
            while idx < buffer.len() && buffer[idx].0 == current {
                min_i = std::cmp::min(min_i, buffer[idx].1 as usize);
                max_i = std::cmp::max(max_i, buffer[idx].1 as usize);
                idx += 1;
            }
            //// We add only if the bucket is non-trivial
            if idx - start > 1 && min_i + exclusion_zone < max_i {
                if idx - start > largest_bucket {
                    largest_bucket = idx - start;
                }
                output.push(start..idx);
            }
        }
        let elapsed_boundaries = timer.elapsed();
        info!("grouping subsequences";
            "tag" => "profiling",
            "repetition" => repetition,
            "depth" => depth,
            "largest_bucket" => largest_bucket,
            "n_buckets" => output.len(),
            "time_bounds_s" => elapsed_boundaries.as_secs_f64(),
            "time_hashes_s" => elapsed_hashes.as_secs_f64(),
            "time_sort_s" => elapsed_sort.as_secs_f64(),
            "time_s" => (elapsed_hashes + elapsed_sort + elapsed_boundaries).as_secs_f64()
        );
    }
}

/// Data structure to do LSH of subsequences.
#[derive(Clone)]
pub struct Hasher {
    pub dimension: usize,
    pub tensor_repetitions: usize,
    pub repetitions: usize,
    // this is organized like a three dimensional matrix
    vectors: Vec<f64>,
    // And this is organized as a two dimensional matrix
    shifts: Vec<f64>,
    width: f64,
}

impl Hasher {
    pub fn new(dimension: usize, repetitions: usize, width: f64, seed: u64) -> Self {
        let tensor_repetitions = (repetitions as f64).sqrt().ceil() as usize;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut vectors = Vec::with_capacity(dimension * K * tensor_repetitions);
        let mut shifts = Vec::with_capacity(K * tensor_repetitions);
        let normal = Normal::new(0.0, 1.0).expect("problem instantiating normal distribution");
        let uniform = Uniform::new(0.0, width);
        for _ in 0..(repetitions * K * dimension) {
            vectors.push(normal.sample(&mut rng));
        }
        for _ in 0..(repetitions * K) {
            shifts.push(uniform.sample(&mut rng));
        }

        Self {
            dimension,
            tensor_repetitions,
            repetitions,
            vectors,
            shifts,
            width,
        }
    }

    pub fn print_collision_probabilities(&self, max_dist: f64, concatenations: usize) {
        let step = max_dist / 10.0;
        let mut d = step;
        println!(
            "Collision probability profile with {} concatenations",
            concatenations
        );
        print!("dist: ");
        while d < max_dist {
            print!("{:>8.3} ", d);
            d += step;
        }
        print!("\nprob: ");
        let mut d = step;
        while d < max_dist {
            print!(
                "{:>8.3e} ",
                self.collision_probability_at(d).powi(concatenations as i32)
            );
            d += step;
        }
        println!();
    }

    pub fn collision_probability_at(&self, d: f64) -> f64 {
        let r = self.width;
        let normal = NormalDistr::new(0.0, 1.0).unwrap();
        1.0 - 2.0 * normal.cdf(-r / d)
            - (2.0 / ((std::f64::consts::PI * 2.0).sqrt() * (r / d)))
                * (1.0 - (-r * r / (2.0 * d * d)).exp())
    }

    pub fn failure_probability(
        &self,
        zeucl_dist: f64,
        reps: usize,
        prefix: usize,
        prev_prefix: Option<usize>,
    ) -> f64 {
        let p = self.collision_probability_at(zeucl_dist);
        let prev_prefix = prev_prefix.unwrap_or(prefix + 1);

        // TODO: precompute all these numbers
        let cur_left_bits = (prefix as f64 / 2.0).floor() as i32;
        let cur_right_bits = (prefix as f64 / 2.0).ceil() as i32;
        assert_eq!(cur_left_bits + cur_right_bits, prefix as i32);

        let prev_left_bits = ((prev_prefix) as f64 / 2.0).floor() as i32;
        let prev_right_bits = ((prev_prefix) as f64 / 2.0).ceil() as i32;
        assert_eq!(prev_left_bits + prev_right_bits, prev_prefix as i32);
        assert!(prev_left_bits >= cur_left_bits);
        assert!(prev_right_bits >= cur_right_bits);

        let up_treps = (reps as f64 + 1.0).sqrt().floor() as i32;
        let low_treps = self.tensor_repetitions as i32 - up_treps;

        // Probabilities of *not* colliding on a *single* repetition with a given number of bits
        let cur_left_fail = 1.0 - p.powi(cur_left_bits);
        let cur_right_fail = 1.0 - p.powi(cur_right_bits);

        let prev_left_fail = 1.0 - p.powi(prev_left_bits);
        let prev_right_fail = 1.0 - p.powi(prev_right_bits);

        // Probabilities of collising in *at least* on repetition
        let collide_up_left_cur = 1.0 - cur_left_fail.powi(up_treps);
        let collide_up_right_cur = 1.0 - cur_right_fail.powi(up_treps);

        let collide_up_left_prev = 1.0 - prev_left_fail.powi(up_treps);
        let collide_up_right_prev = 1.0 - prev_right_fail.powi(up_treps);

        let collide_low_left_prev = 1.0 - prev_left_fail.powi(low_treps);
        let collide_low_right_prev = 1.0 - prev_right_fail.powi(low_treps);

        (1.0 - collide_up_left_cur * collide_up_right_cur)
            * (1.0 - collide_low_left_prev * collide_low_right_prev)
            * (1.0 - collide_up_left_prev * collide_low_right_prev)
            * (1.0 - collide_low_left_prev * collide_up_right_prev)
    }

    pub fn compute_width(ts: &WindowedTimeseries) -> f64 {
        let n = ts.num_subsequences();
        let subsequence_norm = (ts.w as f64).sqrt();
        let expected_max_dotp = subsequence_norm * (2.0 * (n as f64).ln()).sqrt();
        expected_max_dotp / 128.0
    }

    /// With this function we estimate the `width` parameter for bucketing the projections in
    /// the LSH function. While the precise value of this parameter is not so important (since
    /// the effects on the collision probability of a misconfiguration can be counterbalanced by
    /// using a larger or smaller `k`), setting a sensible value can help a great deal.
    #[deprecated]
    pub fn estimate_width(
        ts: &WindowedTimeseries,
        fft_data: &FFTData,
        k: usize,
        min_dist: Option<f64>,
        seed: u64,
    ) -> f64 {
        let timer = Instant::now();

        // Determine a good first guess
        let n = ts.num_subsequences();
        let subsequence_norm = (ts.w as f64).sqrt();
        let expected_max_dotp = subsequence_norm * (2.0 * (n as f64).ln()).sqrt();
        println!("Expected max dotp: {}", expected_max_dotp);
        let mut r = expected_max_dotp / 128.0;

        let mut probe_buffers = ColumnBuffers::default();

        let mut pair_probing_time = Duration::from_secs(0);
        let mut probed_pairs = 0usize;

        loop {
            println!("Build probe buckets with r={}", r);
            let probe_hasher = Arc::new(Hasher::new(ts.w, 1, r, seed));
            let probe_collection = HashCollection::from_ts(&ts, fft_data, probe_hasher);
            let probe_collection = Arc::new(probe_collection);
            probe_collection.group_subsequences(K, 0, ts.w, &mut probe_buffers, true);
            info!("grouped subsequences");

            let mut has_collision = || {
                let mut topk = crate::motifs::TopK::new(k, ts.w);
                let timer = Instant::now();
                // let mut cnt_dists = 0;
                for bucket in probe_buffers.buckets.iter() {
                    let bucket = &probe_buffers.hashes[bucket.clone()];
                    for (_, a_idx) in bucket.iter() {
                        let a_idx = *a_idx as usize;
                        for (_, b_idx) in bucket.iter() {
                            let b_idx = *b_idx as usize;
                            if a_idx + ts.w < b_idx {
                                probed_pairs += 1;
                                if probe_collection.first_collision(a_idx, b_idx, K).is_some() {
                                    let d = crate::distance::zeucl(&ts, a_idx, b_idx);
                                    // cnt_dists += 1;
                                    if d > min_dist.unwrap_or(-1.0) {
                                        topk.insert(Motif {
                                            distance: d,
                                            elapsed: None,
                                            idx_a: a_idx,
                                            idx_b: b_idx,
                                            discovered: timer.elapsed(),
                                        });
                                    }
                                    if topk.k_th().is_some() {
                                        return Some(topk.k_th().unwrap().distance);
                                    }
                                }
                            }
                        }
                    }
                }
                pair_probing_time += timer.elapsed();
                // println!(
                //     "Found {} motifs with {} distance computations",
                //     topk.to_vec().len(),
                //     cnt_dists,
                // );
                return None;
            };

            if let Some(_kth) = has_collision() {
                break;
            } else {
                r *= 2.0;
            }
            drop(probe_collection);
            info!("width estimation"; "time_s" => timer.elapsed().as_secs_f64(), "width" => r, "tag" => "profiling");
        }

        return r;
    }

    fn get_vector(&self, repetition: usize, concat: usize) -> &'_ [f64] {
        let idx = repetition * K * self.dimension + concat * self.dimension;
        &self.vectors[idx..idx + self.dimension]
    }

    //// With this function we hash all the subsequences of the timeseries in one go.
    pub fn hash_all<F: FnMut(usize, f64)>(
        &self,
        ts: &WindowedTimeseries,
        fft_data: &FFTData,
        k: usize,
        repetition: usize,
        mut callback: F,
    ) -> f64 {
        let v = self.get_vector(repetition, k);
        let shift = self.shifts[repetition * K + k];
        let mut max_dotp = 0.0f64;
        ts.znormalized_sliding_dot_product_for_each(fft_data, v, |i, mut h| {
            max_dotp = max_dotp.max(h.abs());
            h = (h + shift) / self.width;
            //// Count if the value is out of bounds to be repre
            if h.abs() > 128.0 {
                h = h.signum() * 127.0;
            }
            callback(i, h);
        });
        max_dotp
    }
}

#[cfg(test)]
mod test {
    use crate::lsh::*;

    #[test]
    fn test_first_collision() {
        let w = 300;
        let ts = crate::load::loadts("data/ECG.csv.gz", Some(500)).expect("problem loading data");
        let ts = crate::timeseries::WindowedTimeseries::new(ts, w, true);
        let fft_data = FFTData::new(&ts);

        let repetitions = 200;

        let hasher = Arc::new(Hasher::new(w, repetitions, 5.0, 1245));
        let pools = HashCollection::from_ts(&ts, &fft_data, Arc::clone(&hasher));

        for &depth in &[K, K / 2, K / 4] {
            for i in 0..ts.num_subsequences() {
                for j in 0..ts.num_subsequences() {
                    // println!("i={} j={}", i, j);
                    assert_eq!(
                        pools.first_collision(i, j, depth),
                        pools.first_collision_baseline(i, j, depth)
                    );
                    // todo!()
                }
            }
        }
    }
}
