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

use crate::motifs::Motif;
use crate::{alloc_cnt, allocator::*};
// TODO Remove this dependency
use crate::sort::*;
use crate::timeseries::{FFTData, WindowedTimeseries};
use rand::prelude::*;
use rand_distr::{Normal, Uniform};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use slog_scope::info;
use statrs::distribution::{ContinuousCDF, Normal as NormalDistr};
use std::cell::RefCell;
use std::ops::Range;
use std::time::Duration;
use std::{cell::UnsafeCell, sync::Arc, time::Instant};
use thread_local::ThreadLocal;

//// ## Hash values
//// We consider hash values made of 8-bit words. So we have to make sure, setting the
//// `width` parameter, that the values are in the range `[-128, 127]`.
//// More on the estimation of the `width` parameter down below.

//// We only consider concatenations of hash values of a fixed length, defined in this
//// constant `K`. The reason is that this way we can inline the hash values when allocated into a
//// vector, rather than falling back to vector of vectors. Removing this dereference allows for
//// a rather large speed up.
//// Also, it is one fewer parameter for the user to set.
pub const K: usize = 32;
pub const K_HALF: usize = K / 2;

/// A string of hash values
#[derive(PartialEq, PartialOrd, Eq, Ord, Clone)]
pub struct HashString([u8; K]);
impl HashString {
    fn prefix(&self, l: usize) -> &[u8] {
        &self.0[..l]
    }
    fn new(left: &[u8], right: &[u8]) -> Self {
        assert_eq!(left.len(), K_HALF);
        assert_eq!(right.len(), K_HALF);
        let mut h = [0u8; K];
        for i in 0..K_HALF {
            h[2 * i] = left[i];
            h[2 * i + 1] = right[i];
        }
        Self(h)
    }
}

//// That said, here is the definition of a hash value, with several
//// utility implementations following.
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

//// ## Collections of hash values

//// A key part of the algorithm is the ability to process hash values related to different subsequence
//// in bulk. In particular, we want to be able to access all the hash values associated to one particular repetition,
//// so that all subsequences sharing a common prefix of a given length can be accessed together.
////
//// Furthermore, we want to reduce the number of hash function evaluations, which tend to be expensive even
//// when using the convolution trick. Implementing the LSH repetitions naively would require
//// computing `K * L` hash values for each subsequence of the time series, where `K` is the
//// number of concatenations, and `L` is the number of repetitions.
//// We can cut down these evaluations by using the [tensoring technique](https://arxiv.org/pdf/1708.07586.pdf), which
//// allows to compute just `2 * K * sqrt(L)` hash values, which are then combined to derive the
//// instances of `HashValue` we need.
////
//// **TODO**: maybe give more details on the tensoring approach.

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

//// This data structure contains all the information needed to generate the hash values for all the repeititions
//// for all the subsequences.
#[derive(Clone)]
pub struct HashCollection {
    pub hasher: Arc<Hasher>,
    n_subsequences: usize,
    oob: usize,
    // Pools are organized as three dimensional matrices, in C order.
    // The stride in the first dimension is `K * n_subsequences`, and the stride in the
    // second dimension is `K`. In the third dimension, the first `K_HALF` elements
    // are the left pool, the second are the right pool
    // TODO: More cache-friendly layout. The first index should be the repetition, the second should
    // be the index of the subsequence. We should store the left and right hashes in two separate arrays
    // to pack more data in a cache line, speeding up the
    pools: Vec<u8>,
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
        // println!(
        //     "Number of tensor repetitions: {}",
        //     hasher.tensor_repetitions
        // );
        let ns = ts.num_subsequences();

        let mut pools = alloc_cnt!("pools"; {vec![0u8; hasher.tensor_repetitions * K * ns]});
        let nhashes = pools.len();
        let uns_pools = UnsafeSlice::new(&mut pools);

        let timer = Instant::now();
        //// Instead of doing a double nested loop over the repetitions and K, we flatten all
        //// the iterations (which are independent) so to expose more parallelism
        let oob = (0..(hasher.tensor_repetitions * K))
            .into_par_iter()
            .map(|hash_idx| {
                let repetition = hash_idx / K;
                let k = hash_idx % K;

                let oob = hasher.hash_all(&ts, fft_data, k, repetition, |i, h| {
                    let idx = K * ns * repetition + i * K + k;
                    assert!(idx < nhashes);
                    let h = ((h as i64 & 0xFFi64) as i8) as u8;
                    //// This operation is `unsafe` but each index is accessed only once,
                    //// so there are no data races, despite not using synchronization.
                    unsafe { uns_pools.write(idx, h) };
                });
                oob
            })
            .sum::<usize>();

        let elapsed = timer.elapsed();
        info!("tensor pool building";
            "tag" => "profiling",
            "time_s" => elapsed.as_secs_f64(),
            "out of bounds" => oob,
            "total hashes" => nhashes
        );

        Self {
            hasher,
            oob,
            n_subsequences: ns,
            pools,
        }
    }

    pub fn left(&self, i: usize, repetition: usize) -> &[u8] {
        let trep = repetition % self.hasher.tensor_repetitions;
        let idx = K * self.n_subsequences * trep + i * K;
        &self.pools[idx..idx + K_HALF]
    }

    pub fn right(&self, i: usize, repetition: usize) -> &[u8] {
        let trep = repetition / self.hasher.tensor_repetitions;
        let idx = K * self.n_subsequences * trep + i * K + K_HALF;
        &self.pools[idx..idx + K_HALF]
    }

    #[cfg(test)]
    pub fn extended_hash_value(&self, i: usize, repetition: usize) -> [u8; 32] {
        let mut output = [0; K];
        let l = &self.left(i, repetition);
        let r = &self.right(i, repetition);
        let mut h = 0;
        while h < K_HALF {
            output[2 * h] = l[h];
            output[2 * h + 1] = r[h];
            h += 1;
        }
        output
    }

    pub fn k_pair(k: usize) -> (usize, usize) {
        let k_left = (k as f64 / 2.0).ceil() as usize;
        let k_right = (k as f64 / 2.0).floor() as usize;
        (k_left, k_right)
    }

    pub fn hash_value(&self, i: usize, prefix: usize, repetition: usize) -> HashValue {
        let mut hv: [u8; 32] = [0; 32];
        let l = &self.left(i, repetition);
        let r = &self.right(i, repetition);
        for h in 0..usize::div_ceil(prefix, 2) {
            hv[2 * h] = l[h];
            hv[2 * h + 1] = r[h];
        }
        HashValue(xxhash_rust::xxh32::xxh32(&hv[..prefix], 1234))
    }

    pub fn fraction_oob(&self) -> f64 {
        self.oob as f64 / self.pools.len() as f64
    }

    // TODO: Reimplement this test
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
        let jump = K * self.n_subsequences;
        let mut iidx = i * K;
        let mut jidx = j * K;

        let mut lindex = None;
        let mut rindex = None;
        for rep in 0..self.hasher.tensor_repetitions {
            let ipool = &self.pools[iidx..iidx + K];
            let jpool = &self.pools[jidx..jidx + K];
            // check left
            if lindex.is_none() {
                let hi = &ipool[0..k_left];
                let hj = &jpool[0..k_left];
                if hi == hj {
                    lindex = Some(rep);
                }
            }
            // check right
            if rindex.is_none() {
                let hi = &ipool[K_HALF..K_HALF + k_right];
                let hj = &jpool[K_HALF..K_HALF + k_right];
                if hi == hj {
                    rindex = Some(rep);
                }
            }
            if rindex.is_some() && lindex.is_some() {
                break;
            }
            iidx += jump;
            jidx += jump;
        }

        let idx = rindex? * self.hasher.tensor_repetitions + lindex?;
        if idx < self.hasher.repetitions {
            Some(idx)
        } else {
            None
        }
    }

    pub fn group_subsequences(
        &self,
        depth: usize,
        repetition: usize,
        exclusion_zone: usize,
        // This buffer emulates the column of the hash matrix
        buffer: &mut Vec<(HashValue, u32)>,
        output: &mut Vec<Range<usize>>,
    ) -> () {
        let ns = self.n_subsequences;

        buffer.clear();
        output.clear();

        let start = Instant::now();
        buffer.par_extend(
            (0..ns)
                .into_par_iter()
                .map(|i| (self.hash_value(i, depth, repetition), i as u32)),
        );
        let elapsed_hashes = start.elapsed();
        let start = Instant::now();
        buffer.par_sort_unstable();
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

#[derive(Clone)]
pub struct EuclHasher {
    pub dimension: usize,
    /// The quantization width
    width: f64,
    projection_direction: Vec<f64>,
    shift: f64,
}

impl EuclHasher {
    pub fn new<R: Rng>(dimension: usize, width: f64, rng: &mut R) -> Self {
        let normal = Normal::new(0.0, 1.0).expect("problem instantiating normal distribution");
        let uniform = Uniform::new(0.0, width);
        let projection_direction = rng.sample_iter(normal).take(dimension).collect();
        let shift = rng.sample(uniform);

        Self {
            dimension,
            width,
            projection_direction,
            shift,
        }
    }

    pub fn hash_all<F: FnMut(usize, f64)>(
        &self,
        ts: &WindowedTimeseries,
        fft_data: &FFTData,
        mut callback: F,
    ) -> usize {
        let shift = self.shift;
        let mut oob = 0; // count how many out of bounds we have
        ts.znormalized_sliding_dot_product_for_each(
            fft_data,
            &self.projection_direction,
            |i, mut h| {
                h = (h + shift) / self.width;
                //// Count if the value is out of bounds to be repre
                if h.abs() > 128.0 {
                    oob += 1;
                    h = h.signum() * 127.0;
                }
                callback(i, h);
            },
        );
        oob
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

    pub fn collision_probability_at(&self, d: f64) -> f64 {
        let r = self.width;
        let normal = NormalDistr::new(0.0, 1.0).unwrap();
        1.0 - 2.0 * normal.cdf(-r / d)
            - (2.0 / ((std::f64::consts::PI * 2.0).sqrt() * (r / d)))
                * (1.0 - (-r * r / (2.0 * d * d)).exp())
    }

    //// ## Estimating the width parameter

    //// With this function we estimate the `width` parameter for bucketing the projections in
    //// the LSH function. While the precise value of this parameter is not so important (since
    //// the effects on the collision probability of a misconfiguration can be counterbalanced by
    //// using a larger or smaller `k`), setting a sensible value can help a great deal.
    ////
    //// The procedure takes into account two things: that we have at least one collision at
    //// the deepest level, and that we have at most 1% of the hashes falling out of the range [-128, 128],
    //// i.e. that 99% of the hash values can be represented with 8 bits.
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
        let mut r = expected_max_dotp / 128.0;

        let mut probe_column = Vec::new();
        let mut probe_buckets = Vec::new();

        let mut pair_probing_time = Duration::from_secs(0);
        let mut probed_pairs = 0usize;

        let mut kth_upper_bound = None;
        loop {
            // println!("Build probe buckets with r={}", r);
            let probe_hasher = Arc::new(Hasher::new(ts.w, 1, r, seed));
            let probe_collection = HashCollection::from_ts(&ts, fft_data, probe_hasher);
            let fraction_oob = probe_collection.fraction_oob();
            let probe_collection = Arc::new(probe_collection);
            info!(
                "built probe collection";
                "fraction_oob" => fraction_oob
            );
            probe_collection.group_subsequences(K, 0, ts.w, &mut probe_column, &mut probe_buckets);
            info!("grouped subsequences");

            let mut has_collision = || {
                let mut topk = crate::motifs::TopK::new(k, ts.w);
                let timer = Instant::now();
                // let mut cnt_dists = 0;
                for bucket in probe_buckets.iter() {
                    let bucket = &probe_column[bucket.clone()];
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

            if let Some(kth) = has_collision() {
                if fraction_oob < 0.1 {
                    kth_upper_bound.replace(kth);
                    break;
                } else {
                    r *= 2.0;
                }
            } else {
                r *= 2.0;
            }
            drop(probe_collection);
        }
        info!("width estimation"; "time_s" => timer.elapsed().as_secs_f64(), "width" => r, "tag" => "profiling");

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
    ) -> usize {
        let v = self.get_vector(repetition, k);
        let shift = self.shifts[repetition * K + k];
        let mut oob = 0; // count how many out of bounds we have
        ts.znormalized_sliding_dot_product_for_each(fft_data, v, |i, mut h| {
            h = (h + shift) / self.width;
            //// Count if the value is out of bounds to be repre
            if h.abs() > 128.0 {
                oob += 1;
                h = h.signum() * 127.0;
            }
            callback(i, h);
        });
        oob
    }
}

struct Table {
    hashes: Vec<HashString>,
    indices: Vec<usize>,
    /// The table is navigable by considering groups all the elements with hashes that share this
    /// prefix length.
    prefix_length: usize,
    /// The breakpoints between one segment of same-prefix elements and the other
    breakpoints: Vec<usize>,
}

impl Table {
    fn new<I: IntoIterator<Item = (HashString, usize)>>(
        n: usize,
        iter_elements: I,
        scratch: &mut Vec<(HashString, usize)>,
    ) -> Self {
        scratch.clear();
        scratch.reserve_exact(n);
        for pair in iter_elements {
            scratch.push(pair);
        }
        scratch.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        let (hashes, indices): (Vec<HashString>, Vec<usize>) = scratch.drain(..).unzip();
        debug_assert!(hashes.is_sorted());

        let prefix_length = K;
        let mut breakpoints = Vec::new();

        let mut b = 0;
        breakpoints.push(b);
        while b < hashes.len() {
            let needle = &hashes[b];
            b += hashes[b..].partition_point(|h| h == needle);
            breakpoints.push(b);
        }

        Self {
            hashes,
            indices,
            prefix_length,
            breakpoints,
        }
    }

    pub fn shorten_prefix(&mut self, new_prefix: usize) {
        assert!(new_prefix < self.prefix_length);

        assert_eq!(self.breakpoints[0], 0);
        assert_eq!(*self.breakpoints.last().unwrap(), self.hashes.len());

        let prefix_length = self.prefix_length;
        let mut last_hash = &self.hashes[0];

        let hashes = &self.hashes;

        // do cleanup of breakpoints already sharing the (now old) prefix
        self.breakpoints.retain(|b| {
            if *b == 0 || *b == hashes.len() {
                // keep the first and the last
                return true;
            }
            let keep = (hashes[*b].prefix(prefix_length)) != (last_hash.prefix(prefix_length));
            last_hash = &hashes[*b];
            keep
        });

        // only *now* set the prefix length to the requested one.
        // This leaves multiple segments with the same prefix, allowing
        // to avoid visiting the same pairs over and over again.
        self.prefix_length = new_prefix;

        assert_eq!(self.breakpoints[0], 0);
        assert_eq!(*self.breakpoints.last().unwrap(), self.hashes.len());
    }

    fn for_each_segment<F: FnMut(&[usize])>(&self, mut f: F) {
        let mut b_start = 0;

        while b_start < self.breakpoints.len() - 1 {
            let mut b_end = b_start + 1;
            while b_end < self.breakpoints.len() - 1
                && self.breakpoints[b_end] < self.hashes.len()
                && self.hashes[self.breakpoints[b_end]].prefix(self.prefix_length)
                    == self.hashes[self.breakpoints[b_start]].prefix(self.prefix_length)
            {
                b_end += 1;
            }
            let start = self.breakpoints[b_start];
            let end = self.breakpoints[b_end];
            #[cfg(test)]
            {
                for a in start..end {
                    assert_eq!(
                        self.hashes[a].prefix(self.prefix_length),
                        self.hashes[start].prefix(self.prefix_length)
                    );
                }
            }

            f(&self.indices[start..end]);
            b_start = b_end;
        }
    }

    fn for_each_neighboring_segments<F: FnMut(&[usize], &[usize])>(&self, mut f: F) {
        let mut b_start = 0;

        while b_start < self.breakpoints.len() - 1 {
            let mut b_end = b_start + 1;
            while b_end < self.breakpoints.len() - 1
                && self.breakpoints[b_end] < self.hashes.len()
                && self.hashes[self.breakpoints[b_end]].prefix(self.prefix_length)
                    == self.hashes[self.breakpoints[b_start]].prefix(self.prefix_length)
            {
                b_end += 1;
            }

            let breaks = &self.breakpoints[b_start..=b_end];
            for i in 0..(breaks.len() - 1) {
                for j in (i + 1)..(breaks.len() - 1) {
                    let range_a = breaks[i]..breaks[i + 1];
                    let range_b = breaks[j]..breaks[j + 1];
                    #[cfg(test)]
                    {
                        for a in range_a.clone() {
                            for b in range_b.clone() {
                                assert_eq!(
                                    self.hashes[a].prefix(self.prefix_length),
                                    self.hashes[b].prefix(self.prefix_length)
                                );
                            }
                        }
                    }

                    f(&self.indices[range_a], &self.indices[range_b]);
                }
            }
            b_start = b_end;
        }
    }

    pub fn for_each_collision<F: FnMut(usize, usize)>(&self, exclusion_zone: usize, mut f: F) {
        if self.prefix_length == K {
            // We are at the deepest level of the tries
            self.for_each_segment(|segment| {
                for ii in 0..segment.len() {
                    let i = segment[ii];
                    for jj in (ii + 1)..segment.len() {
                        let j = segment[jj];
                        if i.max(j) - i.min(j) >= exclusion_zone {
                            f(i.min(j), i.max(j));
                        }
                    }
                }
            })
        } else {
            self.for_each_neighboring_segments(|seg1, seg2| {
                for i in seg1 {
                    for j in seg2 {
                        if i.max(j) - i.min(j) >= exclusion_zone {
                            f(*i.min(j), *i.max(j));
                        }
                    }
                }
            })
        }
    }
}

/// A collection of segments of lexicograpyically sorted elements, with the invariant
/// that all elements in the same range have the same prefix of the given number of bits.
pub struct TableSegments<'table> {
    table: &'table Table,
    prefix_length: usize,
    /// The breakpoints between one segment and the other
    breakpoints: Vec<usize>,
}

impl<'table> TableSegments<'table> {
    fn new(table: &'table Table) -> Self {
        let prefix_length = 32;
        let mut breakpoints = Vec::new();

        let mut b = 0;
        breakpoints.push(b);
        while b < table.hashes.len() {
            let needle = &table.hashes[b];
            b += table.hashes[b..].partition_point(|h| h == needle);
            breakpoints.push(b);
        }
        assert_eq!(b, table.hashes.len());

        Self {
            table,
            prefix_length,
            breakpoints,
        }
    }

    pub fn for_each_segment<F: FnMut(&[usize])>(&self, mut f: F) {
        let mut b_start = 0;

        while b_start < self.breakpoints.len() - 1 {
            let mut b_end = b_start + 1;
            while b_end < self.breakpoints.len() - 1
                && self.breakpoints[b_end] < self.table.hashes.len()
                && self.table.hashes[self.breakpoints[b_end]].prefix(self.prefix_length)
                    == self.table.hashes[self.breakpoints[b_start]].prefix(self.prefix_length)
            {
                b_end += 1;
            }
            let start = self.breakpoints[b_start];
            let end = self.breakpoints[b_end];
            #[cfg(test)]
            {
                for a in start..end {
                    assert_eq!(
                        self.table.hashes[a].prefix(self.prefix_length),
                        self.table.hashes[start].prefix(self.prefix_length)
                    );
                }
            }

            f(&self.table.indices[start..end]);
            b_start = b_end;
        }
    }

    pub fn for_each_neighboring_segments<F: FnMut(&[usize], &[usize])>(&self, mut f: F) {
        let mut b_start = 0;

        while b_start < self.breakpoints.len() - 1 {
            let mut b_end = b_start + 1;
            while b_end < self.breakpoints.len() - 1
                && self.breakpoints[b_end] < self.table.hashes.len()
                && self.table.hashes[self.breakpoints[b_end]].prefix(self.prefix_length)
                    == self.table.hashes[self.breakpoints[b_start]].prefix(self.prefix_length)
            {
                b_end += 1;
            }

            let breaks = &self.breakpoints[b_start..=b_end];
            for i in 0..(breaks.len() - 1) {
                for j in (i + 1)..(breaks.len() - 1) {
                    let range_a = breaks[i]..breaks[i + 1];
                    let range_b = breaks[j]..breaks[j + 1];
                    #[cfg(test)]
                    {
                        for a in range_a.clone() {
                            for b in range_b.clone() {
                                assert_eq!(
                                    self.table.hashes[a].prefix(self.prefix_length),
                                    self.table.hashes[b].prefix(self.prefix_length)
                                );
                            }
                        }
                    }

                    f(&self.table.indices[range_a], &self.table.indices[range_b]);
                }
            }
            b_start = b_end;
        }
    }

    pub fn shorten_prefix(&mut self, new_prefix: usize) {
        assert!(new_prefix < self.prefix_length);

        assert_eq!(self.breakpoints[0], 0);
        assert_eq!(*self.breakpoints.last().unwrap(), self.table.hashes.len());

        let prefix_length = self.prefix_length;
        let mut last_hash = &self.table.hashes[0];

        let hashes = &self.table.hashes;

        // do cleanup of breakpoints already sharing the (now old) prefix
        self.breakpoints.retain(|b| {
            if *b == 0 || *b == hashes.len() {
                // keep the first and the last
                return true;
            }
            let keep = (hashes[*b].prefix(prefix_length)) != (last_hash.prefix(prefix_length));
            last_hash = &hashes[*b];
            keep
        });

        // only *now* set the prefix length to the requested one.
        // This leaves multiple segments with the same prefix, allowing
        // to avoid visiting the same pairs over and over again.
        self.prefix_length = new_prefix;

        assert_eq!(self.breakpoints[0], 0);
        assert_eq!(*self.breakpoints.last().unwrap(), self.table.hashes.len());
    }
}

struct LSHTablesBuilder<'ts> {
    ts: &'ts WindowedTimeseries,
    width: f64,
    repetitions: usize,
    tensor_reps: usize,
    hashers: Vec<EuclHasher>,
    half_hashes: Vec<u8>,
}

impl<'ts> LSHTablesBuilder<'ts> {
    fn new(
        ts: &'ts WindowedTimeseries,
        dimension: usize,
        repetitions: usize,
        width: f64,
        seed: u64,
    ) -> Self {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let n = ts.num_subsequences();
        let tensor_reps = (repetitions as f64).sqrt().ceil() as usize;
        let hashers = (0..tensor_reps * K)
            .map(|_| EuclHasher::new(dimension, width, &mut rng))
            .collect();
        let half_hashes = vec![0u8; K * tensor_reps * n];
        eprintln!("Allocated hashers and space");
        Self {
            ts,
            width,
            repetitions,
            tensor_reps,
            hashers,
            half_hashes,
        }
    }

    fn build(mut self) -> LSHTables {
        let n = self.ts.num_subsequences();
        let ts = &self.ts;
        let fft_data = FFTData::new(ts);

        let hashers = self.hashers.chunks_exact(K_HALF);
        assert!(hashers.remainder().is_empty());
        let half_hashes = self.half_hashes.chunks_exact_mut(K_HALF * n);
        eprintln!("Compute half-width hashes");
        let start = Instant::now();
        half_hashes.zip(hashers).for_each(|(hashes, hashers)| {
            for (k, h) in hashers.iter().enumerate() {
                h.hash_all(ts, &fft_data, |i, hash| {
                    let offset = i * K_HALF + k;
                    hashes[offset] = ((hash as i64 & 0xFFi64) as i8) as u8;
                });
            }
        });
        let end = Instant::now();
        eprintln!("elapsed: {:?}", end - start);

        // let mut tables = Vec::with_capacity(self.repetitions);

        let scratch = ThreadLocal::new();
        eprintln!("setup sorted tables");

        let start = Instant::now();
        let tables = (0..self.repetitions)
            .into_par_iter()
            .map(|r| {
                let mut scratch = scratch
                    .get_or(|| RefCell::new(Vec::with_capacity(n)))
                    .borrow_mut();
                let (l_idx, r_idx) = get_minimal_index_pair(r);
                let chunk_size = n * K_HALF;
                let l_hashes = &self.half_hashes[l_idx * chunk_size..(l_idx + 1) * chunk_size];
                let r_hashes = &self.half_hashes[r_idx * chunk_size..(r_idx + 1) * chunk_size];

                let table = Table::new(
                    n,
                    (0..n).map(|i| {
                        let h = HashString::new(&l_hashes[i..i + K_HALF], &r_hashes[i..i + K_HALF]);
                        (h, i)
                    }),
                    &mut scratch,
                );

                table
            })
            .collect();
        let end = Instant::now();
        eprintln!("elapsed: {:?}", end - start);

        LSHTables {
            tensor_repetitions: self.tensor_reps,
            repetitions: self.repetitions,
            tables,
            width: self.width,
        }
    }
}

pub struct LSHTables {
    width: f64,
    repetitions: usize,
    tensor_repetitions: usize,
    tables: Vec<Table>,
}

impl LSHTables {
    pub fn from_ts<R: Rng>(
        ts: &WindowedTimeseries,
        repetitions: usize,
        width: f64,
        seed: u64,
    ) -> Self {
        let builder = LSHTablesBuilder::new(ts, ts.w, repetitions, width, seed);
        builder.build()
    }

    /// The collision probability of a single hash function at the given z-normalized Euclidean
    /// distance
    pub fn collision_probability_at(&self, d: f64) -> f64 {
        let r = self.width;
        let normal = NormalDistr::new(0.0, 1.0).unwrap();
        1.0 - 2.0 * normal.cdf(-r / d)
            - (2.0 / ((std::f64::consts::PI * 2.0).sqrt() * (r / d)))
                * (1.0 - (-r * r / (2.0 * d * d)).exp())
    }

    /// What would be the failure probability if iterations were independent
    pub fn independent_failure_probability(
        &self,
        zeucl_dist: f64,
        reps: usize,
        bits: usize,
    ) -> f64 {
        let p = self.collision_probability_at(zeucl_dist);

        let cur_failure = (1.0 - p.powi(bits as i32)).powi(reps as i32 + 1);
        let prev_failure =
            (1.0 - p.powi(bits as i32 + 1)).powi((self.repetitions - reps + 1) as i32);
        return cur_failure * prev_failure;
    }

    pub fn failure_probability(&self, dist: f64, reps: usize, bits: usize) -> f64 {
        let p = self.collision_probability_at(dist);

        // TODO: precompute all these numbers
        let cur_left_bits = (bits as f64 / 2.0).floor() as i32;
        let cur_right_bits = (bits as f64 / 2.0).ceil() as i32;
        assert_eq!(cur_left_bits + cur_right_bits, bits as i32);

        let prev_left_bits = ((bits + 1) as f64 / 2.0).floor() as i32;
        let prev_right_bits = ((bits + 1) as f64 / 2.0).ceil() as i32;
        assert_eq!(prev_left_bits + prev_right_bits, bits as i32 + 1);
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

        for &depth in &[32usize, 20, 10] {
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
