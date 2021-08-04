use std::{
    cell::RefCell,
    ops::{Index, IndexMut, Range},
};

use bumpalo::Bump;
use rand::prelude::*;
use rand_distr::StandardNormal;
use rand_xoshiro::{Xoshiro256Plus, Xoshiro256PlusPlus};

use crate::{distance::*, embedding::Embedder, types::WindowedTimeseries};

pub struct TensorPool<'arena> {
    left: Vec<u32, &'arena Bump>,
    right: Vec<u32, &'arena Bump>,
}

impl<'arena> TensorPool<'arena> {
    fn from_words<I: IntoIterator<Item = u64>>(
        words: I,
        arena: &'arena Bump,
        reps: usize,
        k_right: usize,
    ) -> Self {
        let mask_right = mask(k_right);
        let mut words_left = Vec::with_capacity_in(reps, arena);
        let mut words_right = Vec::with_capacity_in(reps, arena);

        for word in words.into_iter() {
            words_left.push((word >> k_right) as u32);
            words_right.push((word & mask_right) as u32);
        }
        TensorPool {
            left: words_left,
            right: words_right,
        }
    }
}

pub struct HashCollection<'hasher, 'arena> {
    hasher: &'hasher Hasher,
    pools: Vec<TensorPool<'arena>, &'arena Bump>,
}

impl<'hasher, 'arena> HashCollection<'hasher, 'arena> {
    pub fn from_ts(ts: &WindowedTimeseries, hasher: &'hasher Hasher, arena: &'arena Bump) -> Self {
        let mut pools = Vec::with_capacity_in(ts.num_subsequences(), arena);
        for i in 0..ts.num_subsequences() {
            pools.push(hasher.hash(ts, i, arena));
        }
        Self { hasher, pools }
    }

    pub fn first_collision(&self, i: usize, j: usize, depth: usize) -> Option<usize> {
        let m = mask(depth);
        (0..self.hasher.repetitions).filter(|&rep| {
            let hi = self.hash_value(i, rep) & m;
            let hj = self.hash_value(j, rep) & m;
            hi == hj
        }).next()
        // let rm = mask(depth) as u32;

        // let rindex = self.pools[i]
        //     .right
        //     .iter()
        //     .zip(self.pools[j].right.iter())
        //     .enumerate()
        //     .find(|(_i, (ibits, jbits))| (**ibits & rm) == (**jbits & rm))?
        //     .0;

        // if depth < self.hasher.k_right {
        //     return Some(rindex);
        // }
        // let lm = mask(depth - self.hasher.k_right) as u32;

        // let lindex = self.pools[i]
        //     .left
        //     .iter()
        //     .zip(self.pools[j].left.iter())
        //     .enumerate()
        //     .find(|(_i, (ibits, jbits))| (**ibits & lm) == (**jbits & lm))?
        //     .0;

        // Some(lindex * self.hasher.tensor_repetitions + rindex)
    }

    pub fn collision_probability(&self, i: usize, j: usize) -> f64 {
        let mut n_collisions = 0;
        let m_left = mask(self.hasher.k_left);
        let m_right = mask(self.hasher.k_right);

        n_collisions += count_collisions(&self.pools[i].left, &self.pools[j].left, m_left as u32);
        n_collisions +=
            count_collisions(&self.pools[i].right, &self.pools[j].right, m_right as u32);

        n_collisions as f64
            / ((self.hasher.k_left + self.hasher.k_right) * self.hasher.tensor_repetitions) as f64
    }

    pub fn get_hash_matrix(&self) -> HashMatrix {
        HashMatrix::new(self)
    }

    fn hash_value(&self, i: usize, repetition: usize) -> u64 {
        let l_idx = repetition / self.hasher.tensor_repetitions;
        let r_idx = repetition % self.hasher.tensor_repetitions;
        let pool = &self.pools[i];
        let mut w = pool.left[l_idx] as u64;
        w = w << self.hasher.k_right;
        w = w | (pool.right[r_idx] as u64);
        w
    }
}

fn count_collisions(wa: &[u32], wb: &[u32], bitmask: u32) -> usize {
    wa.iter()
        .zip(wb.iter())
        .map(|(&a, &b)| (!(a ^ b) & bitmask).count_ones() as usize)
        .sum()
}

pub struct HashMatrix {
    /// Outer vector has one entry per repetition, inner vector has one entry per subsequence,
    /// and items are hash values and indices into the timeseries
    hashes: Vec<Vec<(u64, usize)>>,
}

impl HashMatrix {
    fn new(coll: &HashCollection) -> Self {
        let mut hashes = Vec::with_capacity(coll.hasher.repetitions);
        for repetition in 0..coll.hasher.repetitions {
            let mut rephashes = Vec::with_capacity(coll.pools.len());
            for i in 0..coll.pools.len() {
                rephashes.push((coll.hash_value(i, repetition), i));
            }
            rephashes.sort_unstable();
            hashes.push(rephashes);
        }
        Self { hashes }
    }

    pub fn buckets<'hashes>(
        &'hashes self,
        depth: usize,
        repetition: usize,
    ) -> BucketIterator<'hashes> {
        BucketIterator {
            hashes: &self.hashes[repetition],
            mask: mask(depth),
            idx: 0,
        }
    }
}

pub struct BucketIterator<'hashes> {
    hashes: &'hashes Vec<(u64, usize)>,
    mask: u64,
    idx: usize,
}

impl<'hashes> Iterator for BucketIterator<'hashes> {
    type Item = (Range<usize>, &'hashes [(u64, usize)]);

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.hashes.len() {
            return None;
        }
        let start = self.idx;
        let current = self.hashes[self.idx].0 & self.mask;
        while self.idx < self.hashes.len() && (self.hashes[self.idx].0 & self.mask) == current {
            self.idx += 1;
        }
        // println!("Bucket starting with {:08b}, width {}", current, self.idx - start);
        Some((start..self.idx, &self.hashes[start..self.idx]))
    }
}

thread_local! {
    static ZNORM_BUFFER: RefCell<Vec<f64>> = RefCell::new(Vec::new());
    static EMBED_BUFFER: RefCell<Vec<f64>> = RefCell::new(Vec::new());
}

struct HyperplaneLSH {
    planes: Vec<Vec<Vec<f64>>>,
}

impl HyperplaneLSH {
    fn new(dim: usize, k: usize, repetitions: usize, seed: u64) -> Self {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut planes = Vec::with_capacity(repetitions);
        for _ in 0..repetitions {
            let mut ks = Vec::with_capacity(k);
            for _ in 0..k {
                let v: Vec<f64> = StandardNormal.sample_iter(&mut rng).take(dim).collect();
                ks.push(v);
            }
            planes.push(ks);
        }

        Self { planes }
    }

    fn hash<'hasher>(&'hasher self, v: &'hasher [f64]) -> impl Iterator<Item = u64> + 'hasher {
        self.planes.iter().map(move |rep_planes| {
            let mut word = 0u64;
            for plane in rep_planes.iter() {
                if dot(v, plane) >= 0.0 {
                    word = (word << 1) | 1;
                } else {
                    word = word << 1;
                }
            }
            word
        })
    }
}

/// Data structure to do LSH of subsequences.
pub struct Hasher {
    pub k: usize,
    pub k_left: usize,
    pub k_right: usize,
    pub tensor_repetitions: usize,
    pub repetitions: usize,
    hyperplanes: HyperplaneLSH,
    embedder: Embedder,
}

pub fn mask(bits: usize) -> u64 {
    let mut m = 0;
    for _ in 0..bits {
        m = (m << 1) | 1;
    }
    m
}

impl Hasher {
    pub fn new(k: usize, repetitions: usize, embedder: Embedder, seed: u64) -> Self {
        let k_left = k / 2;
        let k_right = (k as f64 / 2.0).ceil() as usize;
        let tensor_repetitions = (repetitions as f64).sqrt().ceil() as usize;
        let hyperplanes = HyperplaneLSH::new(embedder.dim_in, k, tensor_repetitions, seed);

        Self {
            k,
            k_left,
            k_right,
            tensor_repetitions,
            repetitions,
            hyperplanes,
            embedder,
        }
    }

    pub fn hash<'arena>(
        &self,
        ts: &WindowedTimeseries,
        i: usize,
        arena: &'arena Bump,
    ) -> TensorPool<'arena> {
        assert!(ts.w == self.embedder.dim_in);

        ZNORM_BUFFER.with(|zbuf| {
            // do z normalization
            ts.znormalized(i, &mut zbuf.borrow_mut());

            // do the embedding
            EMBED_BUFFER.with(|ebuf| {
                let mut ebuf = ebuf.borrow_mut();
                ebuf.resize(self.embedder.dim_out, 0.0);
                self.embedder.embed(&zbuf.borrow(), &mut ebuf);

                // and finally do the hashing itself and construct the tensor pool
                TensorPool::from_words(
                    self.hyperplanes.hash(&ebuf),
                    arena,
                    self.tensor_repetitions,
                    self.k_right,
                )
            })
        })
    }
}

#[cfg(test)]
mod test {
    use crate::lsh::{HyperplaneLSH, TensorPool, mask};
    use crate::distance::normalize;

    #[test]
    fn test_probability() {
        use rand::prelude::*;
        use rand_distr::StandardNormal;
        use rand_xoshiro::Xoroshiro128PlusPlus;
        let mut rng = Xoroshiro128PlusPlus::seed_from_u64(3462);
        let dim = 2;
        let repetitions = 10000;
        let n = 1000;

        let hasher = HyperplaneLSH::new(dim, 1, repetitions, 13462);

        let a: Vec<f64> = StandardNormal.sample_iter(&mut rng).take(dim).collect();
        let a = normalize(&a);

        assert_eq!(
            1.0,
            hasher
                .hash(&a)
                .zip(hasher.hash(&a))
                .filter(|(ha, hb)| ha == hb)
                .count() as f64
                / repetitions as f64
        );

        let mut angles = Vec::new();
        let mut probs = Vec::new();
        let mut expected = Vec::new();
        for _ in 0..n {
            let b: Vec<f64> = StandardNormal.sample_iter(&mut rng).take(dim).collect();
            let b = normalize(&b);

            let angle = dbg!(crate::distance::dot(&a, &b).acos());

            let prob = hasher
                .hash(&a)
                .zip(hasher.hash(&b))
                .filter(|(ha, hb)| ha == hb)
                .count() as f64
                / repetitions as f64;
            let expected_prob = 1.0 - angle / (std::f64::consts::PI);

            angles.push(angle);
            probs.push(prob);
            expected.push(expected_prob);
            assert!(
                (expected_prob - prob).abs() <= 0.01,
                "discrepancy in probabilities:\n\texpected: {}\n\tactual:   {}",
                expected_prob,
                prob
            );
        }

        // let exp_trace = Scatter::new(angles.clone(), expected)
        //     .name("expected")
        //     .mode(Mode::Markers);
        // let actual_trace = Scatter::new(angles.clone(), probs)
        //     .name("actual")
        //     .mode(Mode::Markers);
        // let mut p = Plot::new();
        // p.add_trace(exp_trace);
        // p.add_trace(actual_trace);
        // p.show();
    }

    #[test]
    fn test_mask() {
        for i in 0..64 {
            let m = mask(i);
            assert_eq!(m.count_ones(), i as u32);
        }
    }
}
