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

use crate::timeseries::WindowedTimeseries;
use crate::sort::*;
use bumpalo::Bump;
use deepsize::DeepSizeOf;
use rand::prelude::*;
use rand_distr::Normal;
use rand_xoshiro::Xoshiro256PlusPlus;
use slog_scope::info;
use std::{cell::RefCell, cmp::Ordering, fmt::Debug, mem::size_of, ops::Range};

//// ## Hash values
//// We consider hash values made of 8-bit words. So we have to make sure, setting the
//// `width` parameter, that the values are in the range `[-128, 127]`.

pub const K: usize = 32;
pub const K_HALF: usize = K / 2;

/// Wrapper structx for vectors of 8-bits words, which sort lexicographically from the lowest significant bit.
#[derive(Clone, Eq, PartialEq)]
pub struct HashValue {
    hashes: [i8; K]
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

#[derive(DeepSizeOf)]
pub struct TensorPool {
    hashes: Vec<i8>,
}

impl TensorPool {
    fn new(tensor_repetitions: usize) -> Self {
        let hashes = Vec::with_capacity(tensor_repetitions);
        Self { hashes }
    }

    fn append(&mut self, hash: i8) {
        self.hashes.push(hash);
    }
}

pub struct HashCollection<'hasher> {
    hasher: &'hasher Hasher,
    pools: Vec<TensorPool>,
}

impl<'hasher> HashCollection<'hasher> {
    pub fn from_ts(ts: &WindowedTimeseries, hasher: &'hasher Hasher) -> Self {
        let mut pools = Vec::with_capacity(ts.num_subsequences());
        for _ in 0..ts.num_subsequences() {
            pools.push(TensorPool::new(hasher.tensor_repetitions));
        }
        let mut buffer = vec![0; ts.num_subsequences()];
        for repetition in 0..hasher.tensor_repetitions {
            for k in 0..K {
                hasher.hash_all(ts, k, repetition, &mut buffer);
                for (i, h) in buffer.iter().enumerate() {
                    pools[i].append(*h);
                }
            }
        }
        Self { hasher, pools }
    }

    fn left(&self, i: usize, repetition: usize) -> &[i8] {
        let idx = repetition % self.hasher.tensor_repetitions;
        let idx = idx * K_HALF;
        &self.pools[i].hashes[idx..idx + K_HALF]
    }

    fn right(&self, i: usize, repetition: usize) -> &[i8] {
        let idx = repetition / self.hasher.tensor_repetitions;
        let idx = self.hasher.tensor_repetitions * K_HALF + idx * K_HALF;
        &self.pools[i].hashes[idx..idx + K_HALF]
    }

    pub fn hash_value(
        &self,
        i: usize,
        repetition: usize,
        arena: &Bump,
    ) -> HashValue {
        let mut output = [0; K];
        output[0..K_HALF].copy_from_slice(self.left(i, repetition));
        output[K_HALF..K].copy_from_slice(self.right(i, repetition));
        HashValue { hashes: output }
    }

    #[cfg(test)]
    pub fn first_collision_baseline(&self, i: usize, j: usize, depth: usize) -> Option<usize> {
        let arena = Bump::new();
        (0..self.hasher.repetitions)
            .filter(|&rep| {
                self.hash_value(i, rep, &arena)
                    .prefix_eq(&self.hash_value(j, rep, &arena), depth)
            })
            .next()
    }

    pub fn first_collision(&self, i: usize, j: usize, depth: usize) -> Option<usize> {
        let depth_l = std::cmp::min(depth, K_HALF);
        let lindex = (0..self.hasher.tensor_repetitions).find(|&rep| {
            let idx = rep * K_HALF;
            let hi = &self.pools[i].hashes[idx..idx + depth_l];
            let hj = &self.pools[j].hashes[idx..idx + depth_l];
            hi == hj
        })?;

        if depth < K_HALF {
            return Some(lindex);
        }

        let depth_r = depth - K_HALF;
        let rindex = (0..self.hasher.tensor_repetitions).find(|&rep| {
            let idx =
                self.hasher.tensor_repetitions * K_HALF + rep * K_HALF;
            let hi = &self.pools[i].hashes[idx..idx + depth_r];
            let hj = &self.pools[j].hashes[idx..idx + depth_r];
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
        let n_collisions = self.pools[i]
            .hashes
            .iter()
            .zip(self.pools[j].hashes.iter())
            .filter(|(hi, hj)| hi == hj)
            .count();

        n_collisions as f64 / self.pools[i].hashes.len() as f64
    }

    pub fn get_hash_matrix(&self, arena: &Bump) -> HashMatrix {
        HashMatrix::new(self, arena)
    }
}

impl<'hasher> DeepSizeOf for HashCollection<'hasher> {
    fn deep_size_of_children(&self, _context: &mut deepsize::Context) -> usize {
        self.pools.deep_size_of()
    }
}

#[derive(DeepSizeOf)]
pub struct HashMatrix {
    /// Outer vector has one entry per repetition, inner vector has one entry per subsequence,
    /// and items are hash values and indices into the timeseries
    hashes: Vec<Vec<(HashValue, usize)>>,
}

impl HashMatrix {
    fn new(coll: &HashCollection, arena: &Bump) -> Self {
        let mut hashes = Vec::with_capacity(coll.hasher.repetitions);
        for repetition in 0..coll.hasher.repetitions {
            let mut rephashes = Vec::with_capacity(coll.pools.len());
            for i in 0..coll.pools.len() {
                rephashes.push((coll.hash_value(i, repetition, arena), i));
            }
            rephashes.sort_unstable();
            debug_assert!(rephashes.is_sorted_by_key(|pair| pair.0.clone()));
            hashes.push(rephashes);
        }
        Self { hashes }
    }

    pub fn buckets<'hashes>(
        &'hashes self,
        depth: usize,
        repetition: usize,
    ) -> BucketIterator<'hashes> {
        debug_assert!(self.hashes[repetition].is_sorted_by_key(|pair| pair.0.clone()));
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

        let k = 32;
        let repetitions = 200;

        let hasher = Hasher::new(w, k, repetitions, 5.0, 1245);
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
