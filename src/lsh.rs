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
use std::cell::RefCell;
use std::convert::{TryFrom, TryInto};
use std::simd::{u8x16, u8x32, Simd, SimdPartialEq};
use std::sync::RwLock;
// TODO Remove this dependency
use crate::sort::*;
use crate::timeseries::{FFTData, WindowedTimeseries};
use rand::prelude::*;
use rand_distr::{Normal, Uniform};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use slog_scope::info;
use statrs::distribution::{ContinuousCDF, Normal as NormalDistr};
use std::cmp::Ordering;
use std::ops::Range;
use std::time::Duration;
use std::{cell::UnsafeCell, sync::Arc, time::Instant};

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

#[derive(Debug, Eq, PartialEq, PartialOrd, Ord, Clone, Copy)]
pub enum Breakpoint {
    /// The breakpoint refers to a previous prefix length
    Previous(usize),
    /// The breakpoint refers to the current prefix length
    Current(usize),
}

impl Into<usize> for &Breakpoint {
    fn into(self) -> usize {
        match self {
            Breakpoint::Previous(i) => *i,
            Breakpoint::Current(i) => *i,
        }
    }
}
impl Into<usize> for Breakpoint {
    fn into(self) -> usize {
        match self {
            Self::Previous(i) => i,
            Self::Current(i) => i,
        }
    }
}
impl Breakpoint {
    fn is_current(&self) -> bool {
        match self {
            Self::Current(_) => true,
            _ => false,
        }
    }
    pub fn for_each_pair<F: FnMut(usize, usize)>(
        breakpoints: &[Breakpoint],
        indices: &[usize],
        exclusion_zone: usize,
        mut fun: F,
    ) {
        #[cfg(test)]
        {
            for i in 1..breakpoints.len() {
                let bb: usize = breakpoints[i - 1].into();
                assert!(bb < breakpoints[i].into());
            }
        }
        let mut b_i = 0;
        while b_i < breakpoints.len() - 1 {
            if breakpoints[b_i].is_current() && breakpoints[b_i + 1].is_current() {
                // eprintln!("Bucket {:?} {:?}", breakpoints[b_i], breakpoints[b_i + 1]);
                let i_i: usize = breakpoints[b_i].into();
                let i_j: usize = breakpoints[b_i + 1].into();
                // do all pairwise comparisons within the chunk,
                // this chunk was not split in a previous iteration
                for (off, i) in indices[i_i..i_j].iter().enumerate() {
                    for j in &indices[i_i + off..i_j] {
                        let ii = *i.min(j);
                        let jj = *i.max(j);
                        if jj - ii >= exclusion_zone {
                            fun(ii, jj);
                        }
                    }
                }
                b_i += 1;
            } else {
                let mut b_j = b_i + 1; // the adjacent chunk
                while !breakpoints[b_j].is_current() {
                    eprintln!(
                        "Bucket [{:?} {:?}] vs [{:?} {:?}]",
                        breakpoints[b_i],
                        breakpoints[b_i + 1],
                        breakpoints[b_j],
                        breakpoints[b_j + 1],
                    );
                    // here we do comparisons between elements that were in
                    // different chunks in the previous runs
                    let i_start: usize = breakpoints[b_i].into();
                    let i_end: usize = breakpoints[b_i + 1].into();
                    let j_start: usize = breakpoints[b_j].into();
                    let j_end: usize = breakpoints[b_j + 1].into();
                    for i in &indices[i_start..i_end] {
                        for j in &indices[j_start..j_end] {
                            let ii = *i.min(j);
                            let jj = *i.max(j);
                            if jj - ii >= exclusion_zone {
                                fun(ii, jj);
                            }
                        }
                    }

                    b_j += 1;
                }
                b_i += 1;
            }
        }
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

#[inline]
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

    scratch: Arc<RwLock<Vec<(HashValue, HashValue, u32)>>>,
    ranges: Arc<RwLock<Vec<Range<usize>>>>,
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
            scratch: Default::default(),
            ranges: Default::default(),
        }
    }

    #[deprecated]
    pub fn left(&self, i: usize, repetition: usize) -> &[u8] {
        let trep = repetition % self.hasher.tensor_repetitions;
        let idx = K * self.n_subsequences * trep + i * K;
        &self.pools[idx..idx + K_HALF]
    }

    #[deprecated]
    pub fn right(&self, i: usize, repetition: usize) -> &[u8] {
        let trep = repetition / self.hasher.tensor_repetitions;
        let idx = K * self.n_subsequences * trep + i * K + K_HALF;
        &self.pools[idx..idx + K_HALF]
    }

    #[inline]
    pub fn half_hashes(
        &self,
        i: usize,
        l_trep: usize,
        r_trep: usize,
    ) -> (&[u8; K_HALF], &[u8; K_HALF]) {
        // let (l_trep, r_trep) = get_minimal_index_pair(repetition);
        let l_idx = K * self.n_subsequences * l_trep + i * K;
        let r_idx = K * self.n_subsequences * r_trep + i * K + K_HALF;
        let l: &[u8; K_HALF] = self.pools[l_idx..l_idx + K_HALF].try_into().unwrap();
        let r: &[u8; K_HALF] = self.pools[r_idx..r_idx + K_HALF].try_into().unwrap();
        (l, r)
    }

    #[inline]
    pub fn half_hashes_simd(&self, i: usize, l_trep: usize, r_trep: usize) -> (u8x16, u8x16) {
        let l_idx = K * self.n_subsequences * l_trep + i * K;
        let r_idx = K * self.n_subsequences * r_trep + i * K + K_HALF;
        let l = u8x16::try_from(&self.pools[l_idx..l_idx + K_HALF]).unwrap();
        let r = u8x16::try_from(&self.pools[r_idx..r_idx + K_HALF]).unwrap();
        (l, r)
    }

    pub fn extended_hash_value(&self, i: usize, l_trep: usize, r_trep: usize) -> [u8; 32] {
        let mut output = [0; K];
        let (l, r) = self.half_hashes(i, l_trep, r_trep);
        let mut h = 0;
        while h < K_HALF {
            output[2 * h] = l[h];
            output[2 * h + 1] = r[h];
            h += 1;
        }
        output
    }

    pub fn extended_hash_value_simd(&self, i: usize, l_trep: usize, r_trep: usize) -> u8x32 {
        let mut output = [0; K];
        let (l, r) = self.half_hashes(i, l_trep, r_trep);
        let mut h = 0;
        while h < K_HALF {
            output[2 * h] = l[h];
            output[2 * h + 1] = r[h];
            h += 1;
        }
        output.into()
    }

    pub fn k_pair(k: usize) -> (usize, usize) {
        let k_left = (k as f64 / 2.0).ceil() as usize;
        let k_right = (k as f64 / 2.0).floor() as usize;
        (k_left, k_right)
    }

    pub fn hash_value(&self, i: usize, prefix: usize, repetition: usize) -> HashValue {
        let (l_trep, r_trep) = get_minimal_index_pair(repetition);
        let mut hv: [u8; 32] = [0; 32];
        let (l, r) = self.half_hashes(i, l_trep, r_trep);
        for h in 0..usize::div_ceil(prefix, 2) {
            hv[2 * h] = l[h];
            hv[2 * h + 1] = r[h];
        }
        HashValue(xxhash_rust::xxh32::xxh32(&hv[..prefix], 1234))
    }

    fn hash_value_pair(
        &self,
        i: usize,
        prefix_l: usize,
        prefix_r: usize,
        prefix_l_prev: usize,
        prefix_r_prev: usize,
        l_trep: usize,
        r_trep: usize,
    ) -> (HashValue, HashValue) {
        const SEED: u32 = 1234u32;
        let (l, r) = self.half_hashes(i, l_trep, r_trep);

        let mut hasher = xxhash_rust::xxh32::Xxh32::new(SEED);
        hasher.update(&l[..prefix_l]);
        hasher.update(&l[..prefix_r]);
        let h_cur = hasher.digest();

        let mut hasher = xxhash_rust::xxh32::Xxh32::new(SEED);
        hasher.update(&l[..prefix_l_prev]);
        hasher.update(&l[..prefix_r_prev]);
        let h_prev = hasher.digest();

        (HashValue(h_cur), HashValue(h_prev))
    }

    pub fn fraction_oob(&self) -> f64 {
        self.oob as f64 / self.pools.len() as f64
    }

    // TODO: Reimplement this test
    #[cfg(test)]
    pub fn first_collision_baseline(&self, i: usize, j: usize, prefix: usize) -> Option<usize> {
        (0..self.hasher.repetitions)
            .filter(|&rep| {
                let (l_trep, r_trep) = get_minimal_index_pair(rep);
                let ihash = &self.extended_hash_value(i, l_trep, r_trep)[0..prefix];
                let jhash = &self.extended_hash_value(j, l_trep, r_trep)[0..prefix];
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

        get_minimal_repetition(self.hasher.repetitions, rindex?, lindex?)
    }

    #[inline]
    pub fn eq_hashes(
        &self,
        l_prefix: usize,
        r_prefix: usize,
        l_trep: usize,
        r_trep: usize,
        i: usize,
        j: usize,
    ) -> bool {
        let (l_i, r_i) = self.half_hashes(i, l_trep, r_trep);
        let (l_j, r_j) = self.half_hashes(j, l_trep, r_trep);
        l_i[..l_prefix] == l_j[..l_prefix] && r_i[..r_prefix] == r_j[..r_prefix]
    }

    #[inline]
    pub fn cmp_hashes(
        &self,
        l_trep: usize,
        r_trep: usize,
        i: usize,
        j: usize,
    ) -> std::cmp::Ordering {
        use std::simd::ToBitMask;
        let (l_i, r_i) = self.half_hashes_simd(i, l_trep, r_trep);
        let (l_j, r_j) = self.half_hashes_simd(j, l_trep, r_trep);

        let l_eq = l_i.simd_eq(l_j);
        let r_eq = r_i.simd_eq(r_j);

        let l_ne_pos = l_eq.to_bitmask().trailing_ones() as usize;
        let r_ne_pos = r_eq.to_bitmask().trailing_ones() as usize;

        if l_ne_pos == r_ne_pos && l_ne_pos == K_HALF {
            return Ordering::Equal;
        }

        let res = if l_ne_pos <= r_ne_pos {
            l_i[l_ne_pos].cmp(&l_j[l_ne_pos])
        } else {
            r_i[r_ne_pos].cmp(&r_j[r_ne_pos])
        };

        debug_assert_eq!(
            self.extended_hash_value(i, l_trep, r_trep)
                .cmp(&self.extended_hash_value(j, l_trep, r_trep)),
            res
        );

        res
    }

    /// Returns a vector of the indices of the subsequences, sorted by the
    /// lexicographic order of the hash values of the given `repetition`
    pub fn sorted_indices(
        &self,
        repetition: usize,
        scratch: &mut Vec<([u8; K], usize)>,
    ) -> Vec<usize> {
        let (l_trep, r_trep) = get_minimal_index_pair(repetition);
        scratch.clear();
        let mut indices: Vec<usize> = Vec::with_capacity(self.n_subsequences);
        scratch.extend(
            (0..self.n_subsequences)
                .into_iter()
                .map(|i| (self.extended_hash_value(i, l_trep, r_trep), i)),
        );
        scratch.radix_sort();
        indices.extend(scratch.drain(..).map(|p| p.1));
        indices
    }

    pub fn breakpoints(
        &self,
        repetition: usize,
        prefix: usize,
        previous_prefix: usize,
        indices: &[usize],
        output: &mut Vec<Breakpoint>,
    ) {
        let (l_trep, r_trep) = get_minimal_index_pair(repetition);
        let (l_prefix, r_prefix) = Self::k_pair(prefix);
        let (l_prefix_prev, r_prefix_prev) = Self::k_pair(previous_prefix);

        // TODO optimize with exponential + binary search
        output.clear();
        output.push(Breakpoint::Current(0));

        let mut last_idx = indices[0];
        for ii in 1..self.n_subsequences {
            let i = indices[ii];
            if !self.eq_hashes(l_prefix, r_prefix, l_trep, r_trep, last_idx, i) {
                output.push(Breakpoint::Current(ii));
            } else if !self.eq_hashes(l_prefix_prev, r_prefix_prev, l_trep, r_trep, last_idx, i) {
                output.push(Breakpoint::Previous(ii));
            }
            last_idx = i;
        }

        output.push(Breakpoint::Current(indices.len()));
    }

    /// Executes the given `fun`ction for each collision at the given repetition
    /// for the given `prefix`. In doing so, it avoid executing the action
    /// for pairs that already collided at the `prev_prefix`, the prefix
    /// of some earlier iteration.
    pub fn for_each_collision<F: Fn(usize, usize) + Send + Sync>(
        &self,
        prefix: usize,
        prev_prefix: usize,
        repetition: usize,
        exclusion_zone: usize,
        fun: F,
    ) {
        let (l_trep, r_trep) = get_minimal_index_pair(repetition);
        let (prefix_l, prefix_r) = Self::k_pair(prefix);
        let (prefix_l_prev, prefix_r_prev) = Self::k_pair(prev_prefix);

        // Setup the scratch space, which will group consecutive runs of hashes
        let mut scratch = self.scratch.write().unwrap();
        scratch.clear();
        scratch.par_extend((0..self.n_subsequences).into_par_iter().map(|i| {
            let (h_cur, h_prev) = self.hash_value_pair(
                i,
                prefix_l,
                prefix_r,
                prefix_l_prev,
                prefix_r_prev,
                l_trep,
                r_trep,
            );
            (h_cur, h_prev, i as u32)
        }));
        scratch.par_sort_unstable();
        drop(scratch);

        // Now find the bucket boundaries and then iterate through the buckets
        let scratch = self.scratch.read().unwrap();
        let mut ranges = self.ranges.write().unwrap();
        ranges.clear();
        let mut start = 0;
        while start < scratch.len() {
            let end = start + Self::next_breakpoint(&scratch[start..], |tup| tup.0);
            ranges.push(start..end);
            start = end;
        }
        drop(ranges);

        self.ranges
            .read()
            .unwrap()
            .par_iter()
            .for_each(move |range| {
                let scratch = self.scratch.read().unwrap();
                let start = range.start;
                let end = range.end;
                // now we need to distinguish between two cases:
                //  - either we are in a bucket at the current prefix which is
                //    not split into further buckets at the previous prefix
                //  - or there are other buckets at the previous prefix splitting
                //    the current one
                if scratch[range.start].1 == scratch[range.end - 1].1 {
                    // In this case the bucket is not further split, because all the
                    // previous-prefix (longer) hashes are also equal.
                    // in this case we simply run through the pairs of elements
                    for i in start..end {
                        let aa = scratch[i].2;
                        for j in (i + 1)..end {
                            let bb = scratch[j].2;
                            let a = aa.min(bb) as usize;
                            let b = aa.max(bb) as usize;
                            if b - a >= exclusion_zone {
                                fun(a, b);
                            }
                        }
                    }
                } else {
                    // We need to find the splits of the previous iterations,
                    // in order to avoid having duplicate invocations of `fun`
                    // on the same pair
                    let mut ranges: Vec<Range<usize>> = Vec::new();
                    let mut prev_start = start;
                    while prev_start < end {
                        let prev_end =
                            prev_start + Self::next_breakpoint(&scratch[prev_start..], |tup| tup.1);
                        ranges.push(prev_start..prev_end);
                        prev_start = prev_end;
                    }
                    // now that we identified the boundaries, we can go through all
                    // pairs of ranges, and run the function for the pairs therein
                    for i in 0..ranges.len() {
                        let ri = ranges[i].clone();
                        for j in (i + 1)..ranges.len() {
                            let rj = ranges[j].clone();
                            for (_, _, aa) in &scratch[ri.clone()] {
                                for (_, _, bb) in &scratch[rj.clone()] {
                                    let a = *aa.min(bb) as usize;
                                    let b = *aa.max(bb) as usize;
                                    if b - a >= exclusion_zone {
                                        fun(a, b);
                                    }
                                }
                            }
                        }
                    }
                }
            });

        // let mut start = 0;
        // while start < scratch.len() {
        //     let end = start + Self::next_breakpoint(&scratch[start..], |tup| tup.0);
        //
        //     // now we need to distinguish between two cases:
        //     //  - either we are in a bucket at the current prefix which is
        //     //    not split into further buckets at the previous prefix
        //     //  - or there are other buckets at the previous prefix splitting
        //     //    the current one
        //     if scratch[start].1 == scratch[end - 1].1 {
        //         // In this case the bucket is not further split, because all the
        //         // previous-prefix (longer) hashes are also equal.
        //         // in this case we simply run through the pairs of elements
        //         for i in start..end {
        //             let aa = scratch[i].2;
        //             for j in (i + 1)..end {
        //                 let bb = scratch[j].2;
        //                 let a = aa.min(bb) as usize;
        //                 let b = aa.max(bb) as usize;
        //                 if b - a >= exclusion_zone {
        //                     fun(a, b);
        //                 }
        //             }
        //         }
        //     } else {
        //         // We need to find the splits of the previous iterations,
        //         // in order to avoid having duplicate invocations of `fun`
        //         // on the same pair
        //         ranges.clear();
        //         let mut prev_start = start;
        //         while prev_start < end {
        //             let prev_end =
        //                 prev_start + Self::next_breakpoint(&scratch[prev_start..], |tup| tup.1);
        //             ranges.push(prev_start..prev_end);
        //             prev_start = prev_end;
        //         }
        //         // now that we identified the boundaries, we can go through all
        //         // pairs of ranges, and run the function for the pairs therein
        //         for i in 0..ranges.len() {
        //             let ri = ranges[i].clone();
        //             for j in (i + 1)..ranges.len() {
        //                 let rj = ranges[j].clone();
        //                 for (_, _, aa) in &scratch[ri.clone()] {
        //                     for (_, _, bb) in &scratch[rj.clone()] {
        //                         let a = *aa.min(bb) as usize;
        //                         let b = *aa.max(bb) as usize;
        //                         if b - a >= exclusion_zone {
        //                             fun(a, b);
        //                         }
        //                     }
        //                 }
        //             }
        //         }
        //     }
        //
        //     start = end;
        // }
    }

    fn next_breakpoint<T, K: Eq, F: Fn(&T) -> K>(arr: &[T], key: F) -> usize {
        let needle = key(&arr[0]);
        arr.partition_point(|elem| {
            let k = key(elem);
            k == needle
        })
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
        buffer.extend(
            (0..ns)
                .into_iter()
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

    pub fn failure_probability(&self, zeucl_dist: f64, reps: usize, prefix: usize) -> f64 {
        let p = self.collision_probability_at(zeucl_dist);

        // TODO: precompute all these numbers
        let cur_left_bits = (prefix as f64 / 2.0).floor() as i32;
        let cur_right_bits = (prefix as f64 / 2.0).ceil() as i32;
        assert_eq!(cur_left_bits + cur_right_bits, prefix as i32);

        let prev_left_bits = ((prefix + 1) as f64 / 2.0).floor() as i32;
        let prev_right_bits = ((prefix + 1) as f64 / 2.0).ceil() as i32;
        assert_eq!(prev_left_bits + prev_right_bits, prefix as i32 + 1);
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
