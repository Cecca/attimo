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

// TODO Remove this dependency
use crate::timeseries::{FFTData, WindowedTimeseries};
use crate::sort::*;
use rand::prelude::*;
use rand_distr::{Normal, Uniform};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use slog_scope::info;
use statrs::distribution::{ContinuousCDF, Normal as NormalDistr};
use std::ops::Range;
use std::{
    cell::{RefCell, UnsafeCell},
    sync::Arc,
    time::Instant,
};
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

//// That said, here is the definition of a hash value, with several
//// utility implementations following.
#[derive(Debug, Ord, PartialOrd, Eq, PartialEq, Copy, Clone)]
pub struct HashValue(u32);

impl GetByte for HashValue {
    fn num_bytes(&self) -> usize {
        4
    }
    #[inline(always)]
    fn get_byte(&self, i: usize) -> u8 {
        (self.0 >> (8 * (std::mem::size_of::<u32>() - i - 1)) & 0xFF) as u8
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
}

//// This data structure contains all the information needed to generate the hash values for all the repeititions
//// for all the subsequences.
pub struct HashCollection {
    hasher: Arc<Hasher>,
    n_subsequences: usize,
    // Both pools are organized as three dimensional matrices, in C order.
    // The stride in the first dimenson is `K_HALF*n_subsequences`, and the stride in the second
    // dimension is `K_HALF`.
    left_pools: Vec<u8>,
    right_pools: Vec<u8>,
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
    pub fn from_ts(
        ts: &WindowedTimeseries,
        hasher: Arc<Hasher>,
        fft_data: &FFTData,
    ) -> (Self, usize, usize) {
        assert!(ts.num_subsequences() < u32::MAX as usize, "We use 32 bit integers as pointers into subsequences, this timeseries has too many subsequences.");
        println!(
            "Number of tensor repetitions: {}",
            hasher.tensor_repetitions
        );
        let ns = ts.num_subsequences();

        let mut left_pools = vec![0u8; hasher.tensor_repetitions * K_HALF * ns];
        let mut right_pools = vec![0u8; hasher.tensor_repetitions * K_HALF * ns];
        let uns_left_pools = UnsafeSlice::new(&mut left_pools);
        let uns_right_pools = UnsafeSlice::new(&mut right_pools);

        let tl_buffer = ThreadLocal::new();
        // let tl_left_pools = ThreadLocal::new();
        // let tl_right_pools = ThreadLocal::new();

        let timer = Instant::now();
        //// Instead of doing a double nested loop over the repetitions and K, we flatten all
        //// the iterations (which are independent) so to expose more parallelism
        let oob = (0..(hasher.tensor_repetitions * K))
            .into_par_iter()
            .map(|hash_idx| {
                let repetition = hash_idx / K;
                let k = hash_idx % K;

                let mut buffer = tl_buffer.get_or(|| RefCell::new(vec![0; ns])).borrow_mut();
                let (pools, offset) = if k < K_HALF {
                    (&uns_left_pools, k)
                } else {
                    (&uns_right_pools, k - K_HALF)
                };

                let oob = hasher.hash_all(&ts, &fft_data, k, repetition, &mut buffer);
                for (i, h) in buffer.iter().enumerate() {
                    let idx = K_HALF * ns * repetition + i * K_HALF + offset;
                    //// This operation is `unsafe` but each index is accessed only once,
                    //// so there are no data races, despite not using synchronization.
                    unsafe { pools.write(idx, *h) };
                }
                oob
            })
            .sum::<usize>();

        let elapsed = timer.elapsed();
        let total_hashes = left_pools.len() + right_pools.len();
        info!("tensor pool building";
            "tag" => "profiling",
            "time_s" => elapsed.as_secs_f64(),
            "out of bounds" => oob,
            "total hashes" => total_hashes
        );

        (
            Self {
                hasher,
                n_subsequences: ns,
                left_pools,
                right_pools,
            },
            oob,
            total_hashes
        )
    }

    fn left(&self, i: usize, repetition: usize) -> &[u8] {
        let trep = repetition % self.hasher.tensor_repetitions;
        let idx = K_HALF * self.n_subsequences * trep + i * K_HALF;
        &self.left_pools[idx..idx + K_HALF]
    }

    fn right(&self, i: usize, repetition: usize) -> &[u8] {
        let trep = repetition / self.hasher.tensor_repetitions;
        let idx = K_HALF * self.n_subsequences * trep + i * K_HALF;
        &self.right_pools[idx..idx + K_HALF]
    }

    pub fn hash_value(&self, i: usize, depth: usize, repetition: usize) -> HashValue {
        use std::hash::Hasher;
        let mut hasher = std::collections::hash_map::DefaultHasher::default();
        hasher.write(&self.left(i, repetition)[0..std::cmp::min(K_HALF, depth)]);
        if depth > K_HALF {
            hasher.write(&self.right(i, repetition)[0..(depth - K_HALF)]);
        }
        HashValue(hasher.finish() as u32)
    }

    // TODO: Reimplement this test
    // #[cfg(test)]
    // pub fn first_collision_baseline(&self, i: usize, j: usize, depth: usize) -> Option<usize> {
    //     (0..self.hasher.repetitions)
    //         .filter(|&rep| {
    //             self.hash_value(i, rep)
    //                 .prefix_eq(&self.hash_value(j, rep), depth)
    //         })
    //         .next()
    // }

    pub fn first_collision(&self, i: usize, j: usize, depth: usize) -> Option<usize> {
        let jump = K_HALF * self.n_subsequences;

        let depth_l = std::cmp::min(depth, K_HALF);
        let mut lindex = None;
        let mut iidx = i * K_HALF;
        let mut jidx = j * K_HALF;
        for rep in 0..self.hasher.tensor_repetitions {
            if unsafe {
                let hi = &self.left_pools.get_unchecked(iidx..iidx + depth_l);
                let hj = &self.left_pools.get_unchecked(jidx..jidx + depth_l);
                hi == hj
            } {
                lindex = Some(rep);
                break;
            }
            iidx += jump;
            jidx += jump;
        }

        let lindex = lindex?;

        if depth < K_HALF {
            return Some(lindex);
        }

        let depth_r = depth - K_HALF;
        let mut rindex = None;
        let mut iidx = i * K_HALF;
        let mut jidx = j * K_HALF;
        for rep in 0..self.hasher.tensor_repetitions {
            if unsafe {
                let hi = &self.right_pools.get_unchecked(iidx..iidx + depth_r);
                let hj = &self.right_pools.get_unchecked(jidx..jidx + depth_r);
                hi == hj
            } {
                rindex = Some(rep);
                break;
            }
            iidx += jump;
            jidx += jump;
        }
        let rindex = rindex?;

        let idx = rindex * self.hasher.tensor_repetitions + lindex;
        if idx < self.hasher.repetitions {
            Some(idx)
        } else {
            None
        }
    }

    pub fn collision_probability(&self, i: usize, j: usize) -> f64 {
        let mut n_collisions = 0;
        for rep in 0..self.hasher.tensor_repetitions {
            let idx_i = rep / self.n_subsequences + i * K_HALF;
            let idx_j = rep / self.n_subsequences + j * K_HALF;
            n_collisions += self.left_pools[idx_i..idx_i + K_HALF]
                .iter()
                .zip(&self.left_pools[idx_j..idx_j + K_HALF])
                .filter(|(hi, hj)| hi == hj)
                .count();
            n_collisions += self.right_pools[idx_i..idx_i + K_HALF]
                .iter()
                .zip(&self.right_pools[idx_j..idx_j + K_HALF])
                .filter(|(hi, hj)| hi == hj)
                .count();
        }

        n_collisions as f64 / (2 * K_HALF * self.hasher.tensor_repetitions) as f64
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
        info!("column building";
            "tag" => "profiling",
            "repetition" => repetition,
            "time_hashes_s" => elapsed_hashes.as_secs_f64(),
            "time_sort_s" => elapsed_sort.as_secs_f64(),
            "time_s" => (elapsed_hashes + elapsed_sort).as_secs_f64()
        );

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
                output.push(start..idx);
            }
        }
        info!("computing bucket boundaries";
            "tag" => "profiling",
            "repetition" => repetition,
            "time_s" => timer.elapsed().as_secs_f64()
        );
    }
}

/// Data structure to do LSH of subsequences.
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
    pub fn estimate_width(ts: &WindowedTimeseries, fft_data: &FFTData, seed: u64) -> f64 {
        let timer = Instant::now();
        let mut at_least_one_collision = false;
        let mut fraction_oob = f64::INFINITY;
        let mut r = 1.0;
        while !at_least_one_collision || fraction_oob > 0.01 {
            println!("Build probe buckets with r={}", r);
            let probe_hasher = Arc::new(Hasher::new(ts.w, 1, r, seed));
            let (probe_collection, oob, total) = HashCollection::from_ts(&ts, probe_hasher, fft_data);
            fraction_oob = oob as f64 / total as f64;
            let probe_collection = Arc::new(probe_collection);
            info!(
                "built probe collection"; 
                "oob" => oob,
                "total" => total,
                "fraction_oob" => fraction_oob
            );
            let mut probe_column = Vec::new();
            let mut probe_buckets = Vec::new();
            probe_collection.group_subsequences(K, 0, ts.w, &mut probe_column, &mut probe_buckets);
            info!("grouped subsequences");
            at_least_one_collision = probe_buckets.iter().find(|b| b.len() > 1).is_some();
            if !at_least_one_collision || fraction_oob > 0.01 {
                r *= 2.0;
            }
        }
        info!("width estimation"; "time_s" => timer.elapsed().as_secs_f64(), "width" => r, "tag" => "profiling");

        return r;
    }

    fn get_vector(&self, repetition: usize, concat: usize) -> &'_ [f64] {
        let idx = repetition * K * self.dimension + concat * self.dimension;
        &self.vectors[idx..idx + self.dimension]
    }

    //// With this function we hash all the subsequences of the timeseries in one go.
    //// The hash values are placed in the output buffer, which can be reused across calls.
    pub fn hash_all(
        &self,
        ts: &WindowedTimeseries,
        fft_data: &FFTData,
        k: usize,
        repetition: usize,
        buffer: &mut [u8],
    ) -> usize {
        assert!(buffer.len() == ts.num_subsequences());
        let v = self.get_vector(repetition, k);
        let shift = self.shifts[repetition * K + k];
        let mut oob = 0; // count how many out of bounds we have
        DOTP_BUFFER.with(|dotp_buf| {
            ts.znormalized_sliding_dot_product(v, fft_data, &mut dotp_buf.borrow_mut());
            for (i, dotp) in dotp_buf.borrow().iter().enumerate() {
                let h = (dotp + shift) / self.width;
                //// Count if the value is out of bounds to be repre
                if h.abs() > 128.0 {
                    oob += 1;
                }
                //// We only use the 8 lowest-order bits of each hash value, in order to use a bit less space.
                //// In most cases we don't lose information in this way, because we are
                //// truncating zeros in all hash values. When we truncate other bit patters
                //// we are just increasing the collision probability, which harms performance
                //// but not correctness
                buffer[i] = ((h as i64 & 0xFFi64) as i8) as u8;
            }
        });
        oob
    }
}

//// And this is the buffer that is reused across sliding dot product invocations.
//// It's thread local, so it is safe to use if there are multiple concurrent threads.
thread_local! { static DOTP_BUFFER: RefCell<Vec<f64>> = RefCell::new(Vec::new()); }

#[cfg(test)]
mod test {
    use crate::lsh::*;

    #[test]
    fn test_first_collision() {
        let w = 300;
        let ts = crate::load::loadts("data/ECG.csv", Some(500)).expect("problem loading data");
        let ts = crate::timeseries::WindowedTimeseries::new(ts, w);

        let repetitions = 200;

        let hasher = Arc::new(Hasher::new(w, repetitions, 5.0, 1245));
        let fft_data = ts.fft_data();
        let pools = HashCollection::from_ts(&ts, Arc::clone(&hasher), &fft_data);

        for &depth in &[32usize, 20, 10] {
            for i in 0..ts.num_subsequences() {
                for j in 0..ts.num_subsequences() {
                    // println!("i={} j={}", i, j);
                    // assert_eq!(
                    //     pools.first_collision(i, j, depth),
                    //     pools.first_collision_baseline(i, j, depth)
                    // );
                    todo!()
                }
            }
        }
    }
}
