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
//// ## Simhash, a popular LSH family
////
//// A very popular family of hash functions is [Simhash](https://www.cs.princeton.edu/courses/archive/spr04/cos598B/bib/CharikarEstim.pdf).
//// It works for the _angular distance_ (or cosine similarity) and produces
//// binary hash values. It is very simple (just a dot product of the input vector with a random
//// vector with coordinated sampled from a standard normal distribution) but does not readily
//// support the Euclidean distance. Not all hope is lost, however, since we can _embed_ our input
//// vectors into a _kernel space_ such that the dot product in the kernel space is a function of the
//// euclidean distance of the original vectors. This is discussed in the [embedding
//// module](embedding.html).
////
//// Embedding a subsequence of length `w` into a space of dimension `w` has a cost which is
//// $O(w^2)$, since it requires a matrix-vector multiplication.
////
////
//// ## LSH for p-stable distributions
////
//// The cost of the aforementioned approach might be too high, since we have to pay w^2 time for
//// each subsequence.
//// An alternative scheme is to use the approach for distances related to p-stable distributions,
//// described in [this paper](http://theory.csail.mit.edu/~mirrokni/pstable.ps).
//// The idea is rather simple: for each input vector (in our case the subsequences of the input time
//// series) we compute the inner product with a random vector, whose components are distributed
//// according to the p-stable distribution associated with the distance. For the Euclidean
//// distance such distribution is the Standard Normal. The result is then bucketed into bins whose
//// width is a parameter of the algorithm.
//// The nice property of this approach is that we can compute the dot products in one go using the
//// same trick of [MASS](https://www.cs.unm.edu/~mueen/FastestSimilaitySearch.html), i.e. by doing
//// element-wise multiplication in the frequency domain. This way, computing a single hash
//// function for the entire time series takes time n log n. Therefore, for k concatenations
//// and L repetitions (without tensoring) we have that we require just k L n log n to compute all
//// hash values.
//// This is obviously desirable, and also leverages the fact that subsequences are overlapping, a
//// fact that makes this problem different from traditional nearest neighbor search.
////
//// ## Computing dot procuts in n log n time
////
//// This trick is at the base of the fast MASS algorithm for computing the distance profile.
//// The approach is based on the definition of convolution (see these [lecture notes](http://www.dei.unipd.it/~geppo/DA2/DOCS/FFT.pdf))

use crate::sort::*;
use crate::timeseries::WindowedTimeseries;
use deepsize::DeepSizeOf;
use rand::prelude::*;
use rand_distr::Normal;
use rand_xoshiro::Xoshiro256PlusPlus;
use slog_scope::info;
use std::{
    cell::{Cell, RefCell},
    cmp::Ordering,
    fmt::Debug,
    mem::size_of,
    ops::Range,
    rc::Rc,
};

//// ## Hash values
//// We consider hash values made of 8-bit words. So we have to make sure, setting the
//// `width` parameter, that the values are in the range `[-128, 127]`.

//// We only consider concatenations of hash values of a fixed length, defined in this
//// constant `K`. The reason is that this way we can inline the hash values when allocated into a
//// vector, rather than falling back to vector of vectors. Removing this dereference allows for
//// a rather large speed up.
pub const K: usize = 32;
pub const K_HALF: usize = K / 2;

/// Wrapper structx for vectors of 8-bits words, which sort lexicographically from the lowest significant bit.
#[derive(Clone, Eq, PartialEq)]
pub struct HashValue {
    hashes: [i8; K],
}

impl GetByte for HashValue {
    fn num_bytes(&self) -> usize {
        self.hashes.len()
    }
    fn get_byte(&self, i: usize) -> u8 {
        self.hashes[i] as u8
    }
}

impl Debug for HashValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.hashes)
    }
}

impl DeepSizeOf for HashValue {
    fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize {
        size_of::<i8>() * self.hashes.len()
    }
}

//// We also use a custom implementation of the ordering, that sorts lexicographically the
//// hash values from the first value, so that hashes _with the same prefix_ are all
//// grouped together.
impl Ord for HashValue {
    fn cmp(&self, other: &Self) -> Ordering {
        debug_assert!(self.hashes.len() == other.hashes.len());
        for (x, y) in self.hashes.iter().zip(other.hashes.iter()) {
            if x < y {
                return Ordering::Less;
            } else if x > y {
                return Ordering::Greater;
            }
        }
        return Ordering::Equal;
    }
}

impl PartialOrd for HashValue {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl HashValue {
    //// Furthermore, we need a way to compare for equality on prefixes
    //// with this method, which checks if two hash values have the same
    //// prefix of a given length.
    #[inline]
    fn prefix_eq(&self, other: &Self, l: usize) -> bool {
        debug_assert!(self.hashes.len() == other.hashes.len());
        debug_assert!(l <= self.hashes.len());
        self.hashes[..l] == other.hashes[..l]
    }
}

pub struct HashCollection<'hasher> {
    ts: Rc<WindowedTimeseries>,
    hasher: &'hasher Hasher,
    n_subsequences: usize,
    //// Both pools are organized as three dimensional matrices, in C order.
    //// The stride in the first dimenson is `K_HALF*n_subsequences`, and the stride in the second
    //// dimension is `K_HALF`.
    left_pools: Vec<i8>,
    right_pools: Vec<i8>,
    init_mark_left: Cell<usize>,
    init_mark_right: Cell<usize>,
}

impl<'hasher> HashCollection<'hasher> {
    pub fn from_ts(ts: Rc<WindowedTimeseries>, hasher: &'hasher Hasher) -> Self {
        let ns = ts.num_subsequences();
        let mut left_pools = vec![0; hasher.tensor_repetitions * K_HALF * ns];
        let mut right_pools = vec![0; hasher.tensor_repetitions * K_HALF * ns];

        #[cfg(debug)]
        let mut init_left = vec![false; hasher.tensor_repetitions * K_HALF * ns];
        #[cfg(debug)]
        let mut init_right = vec![false; hasher.tensor_repetitions * K_HALF * ns];

        let mut buffer = vec![0; ns];
        for repetition in 0..hasher.tensor_repetitions {
            for k in 0..K_HALF {
                hasher.hash_all(&ts, k, repetition, &mut buffer);
                for (i, h) in buffer.iter().enumerate() {
                    let idx = K_HALF * ns * repetition + i * K_HALF + k;
                    #[cfg(debug)]
                    assert!(!init_left[idx]);
                    left_pools[idx] = *h;
                    #[cfg(debug)]
                    {
                        init_left[idx] = true;
                    }
                }
            }
            for k in K_HALF..K {
                hasher.hash_all(&ts, k, repetition, &mut buffer);
                for (i, h) in buffer.iter().enumerate() {
                    let idx = K_HALF * ns * repetition + i * K_HALF + (k - K_HALF);
                    #[cfg(debug)]
                    assert!(!init_right[idx]);
                    right_pools[idx] = *h;
                    #[cfg(debug)]
                    {
                        init_right[idx] = true;
                    }
                }
            }
        }
        #[cfg(debug)]
        {
            assert!(init_left.iter().all());
            assert!(init_right.iter().all());
            let mut hist = [0; 256];
            for h in &left_pools {
                hist[(*h as u8) as usize] += 1;
            }
            for h in &right_pools {
                hist[(*h as u8) as usize] += 1;
            }
            dbg!(hist);
        }
        Self {
            ts,
            hasher,
            n_subsequences: ns,
            left_pools,
            right_pools,
            init_mark_left: Cell::new(0),
            init_mark_right: Cell::new(0),
        }
    }

    // fn init_repetition(&self, repetition: usize) {
    //     let rep_left = repetition % self.hasher.tensor_repetitions;
    //     let rep_right = repetition / self.hasher.tensor_repetitions;
    // }

    fn left(&self, i: usize, repetition: usize) -> &[i8] {
        // let idx = self.idx_left(i, repetition);
        let trep = repetition % self.hasher.tensor_repetitions;
        let idx = K_HALF * self.n_subsequences * trep + i * K_HALF;
        &self.left_pools[idx..idx + K_HALF]
    }

    fn right(&self, i: usize, repetition: usize) -> &[i8] {
        // let idx = self.idx_right(i, repetition);
        let trep = repetition / self.hasher.tensor_repetitions;
        let idx = K_HALF * self.n_subsequences * trep + i * K_HALF;
        &self.right_pools[idx..idx + K_HALF]
    }

    pub fn hash_value(&self, i: usize, repetition: usize) -> HashValue {
        let mut output = [0; K];
        output[0..K_HALF].copy_from_slice(self.left(i, repetition));
        output[K_HALF..K].copy_from_slice(self.right(i, repetition));
        HashValue { hashes: output }
    }

    #[cfg(test)]
    pub fn first_collision_baseline(&self, i: usize, j: usize, depth: usize) -> Option<usize> {
        (0..self.hasher.repetitions)
            .filter(|&rep| {
                self.hash_value(i, rep)
                    .prefix_eq(&self.hash_value(j, rep), depth)
            })
            .next()
    }

    pub fn first_collision(&self, i: usize, j: usize, depth: usize) -> Option<usize> {
        let depth_l = std::cmp::min(depth, K_HALF);
        let lindex = (0..self.hasher.tensor_repetitions).find(|&rep| {
            let iidx = K_HALF * self.n_subsequences * rep + i * K_HALF;
            let jidx = K_HALF * self.n_subsequences * rep + j * K_HALF;
            let hi = &self.left_pools[iidx..iidx + depth_l];
            let hj = &self.left_pools[jidx..jidx + depth_l];
            hi == hj
        })?;

        if depth < K_HALF {
            return Some(lindex);
        }

        let depth_r = depth - K_HALF;
        let rindex = (0..self.hasher.tensor_repetitions).find(|&rep| {
            let iidx = K_HALF * self.n_subsequences * rep + i * K_HALF;
            let jidx = K_HALF * self.n_subsequences * rep + j * K_HALF;
            let hi = &self.right_pools[iidx..iidx + depth_r];
            let hj = &self.right_pools[jidx..jidx + depth_r];
            hi == hj
        })?;

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

    pub fn get_hash_matrix(&self) -> HashMatrix {
        HashMatrix::new(self)
    }
}

impl<'hasher> DeepSizeOf for HashCollection<'hasher> {
    fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize {
        self.left_pools.deep_size_of() + self.right_pools.deep_size_of()
    }
}

#[derive(DeepSizeOf)]
pub struct HashMatrix<'hasher> {
    /// Outer vector has one entry per repetition, inner vector has one entry per subsequence,
    /// and items are hash values and indices into the timeseries
    hashes: Vec<Vec<(HashValue, usize)>>,
    coll: &'hasher HashCollection<'hasher>,
}

impl<'hasher> HashMatrix<'hasher> {
    fn new(coll: &'hasher HashCollection) -> Self {
        //// We just initialize the vector, which we keep empty at this point. It will
        //// be lazily populated if/when needed.
        let hashes = Vec::with_capacity(coll.hasher.repetitions);
        Self { hashes, coll }
    }

    pub fn buckets<'hashes>(
        &'hashes mut self,
        depth: usize,
        repetition: usize,
    ) -> BucketIterator<'hashes> {
        use std::time::Instant;
        //// If we didn't build the repetition yet, compute all missing repetitions
        while self.hashes.len() <= repetition {
            let rep = self.hashes.len();
            let mut rephashes = Vec::with_capacity(self.coll.n_subsequences);
            let start = Instant::now();
            for i in 0..self.coll.n_subsequences {
                rephashes.push((self.coll.hash_value(i, rep), i));
            }
            let elapsed_hashes = start.elapsed();
            let start = Instant::now();
            rephashes.sort_unstable();
            let elapsed_sort = start.elapsed();
            debug_assert!(rephashes.is_sorted_by_key(|pair| pair.0.clone()));
            info!("completed lazy hash column building"; "repetition" => rep, "time_hashes_s" => elapsed_hashes.as_secs_f64(), "time_sort_s" => elapsed_sort.as_secs_f64());
            self.hashes.push(rephashes);
        }

        BucketIterator {
            hashes: &self.hashes[repetition],
            depth,
            idx: 0,
        }
    }
}

pub struct BucketIterator<'hashes> {
    hashes: &'hashes Vec<(HashValue, usize)>,
    depth: usize,
    idx: usize,
}

impl<'hashes> Iterator for BucketIterator<'hashes> {
    type Item = (Range<usize>, &'hashes [(HashValue, usize)]);

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.hashes.len() {
            return None;
        }
        let start = self.idx;
        let current = &self.hashes[self.idx].0;
        while self.idx < self.hashes.len() && self.hashes[self.idx].0.prefix_eq(current, self.depth)
        {
            self.idx += 1;
        }
        Some((start..self.idx, &self.hashes[start..self.idx]))
    }
}

/// Data structure to do LSH of subsequences.
#[derive(DeepSizeOf)]
pub struct Hasher {
    pub dimension: usize,
    pub tensor_repetitions: usize,
    pub repetitions: usize,
    // this is organized like a three dimensional matrix
    vectors: Vec<f64>,
    width: f64,
}

impl Hasher {
    pub fn new(dimension: usize, repetitions: usize, width: f64, seed: u64) -> Self {
        let tensor_repetitions = (repetitions as f64).sqrt().ceil() as usize;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut vectors = Vec::with_capacity(dimension * K * tensor_repetitions);
        let normal = Normal::new(0.0, 1.0).expect("problem instantiating normal distribution");
        for _ in 0..(repetitions * K * dimension) {
            vectors.push(normal.sample(&mut rng));
        }

        Self {
            dimension,
            tensor_repetitions,
            repetitions,
            vectors,
            width,
        }
    }

    pub fn estimate_width(ts: &WindowedTimeseries, samples: usize, seed: u64) -> f64 {
        let mut min_dotp = f64::INFINITY;
        let mut max_dotp = f64::NEG_INFINITY;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let normal = Normal::new(0.0, 1.0).expect("problem instantiating normal distribution");
        let mut v = Vec::with_capacity(ts.w);
        let mut buf = vec![0.0; ts.num_subsequences()];
        for _ in 0..samples {
            v.clear();
            for x in normal.sample_iter(&mut rng).take(ts.w) {
                v.push(x);
            }
            ts.znormalized_sliding_dot_product(&v, &mut buf);
            let min = *buf.iter().min_by(|x, y| x.partial_cmp(y).unwrap()).unwrap();
            let max = *buf.iter().max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap();
            if min < min_dotp {
                min_dotp = min;
            }
            if max > max_dotp {
                max_dotp = max;
            }
            info!("estimating width"; "min_dotp" => min_dotp, "max_dotp" => max_dotp);
        }

        //// Pick the width so that the range is divided in 128 buckets
        (max_dotp - min_dotp) / 64.0
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
        k: usize,
        repetition: usize,
        buffer: &mut [i8],
    ) {
        assert!(buffer.len() == ts.num_subsequences());
        let v = self.get_vector(repetition, k);
        DOTP_BUFFER.with(|dotp_buf| {
            ts.znormalized_sliding_dot_product(v, &mut dotp_buf.borrow_mut());
            for (i, dotp) in dotp_buf.borrow().iter().enumerate() {
                buffer[i] = (dotp / self.width) as i8;
            }
        });
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

        let hasher = Hasher::new(w, repetitions, 5.0, 1245);
        let pools = HashCollection::from_ts(&ts, &hasher);

        for &depth in &[32usize, 20, 10] {
            println!("depth {}", depth);
            for i in 0..ts.num_subsequences() {
                for j in 0..ts.num_subsequences() {
                    // println!("i={} j={}", i, j);
                    assert_eq!(
                        pools.first_collision(i, j, depth),
                        pools.first_collision_baseline(i, j, depth)
                    );
                }
            }
        }
    }
}
