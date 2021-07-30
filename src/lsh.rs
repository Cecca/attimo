use std::{
    cell::RefCell,
    ops::{Index, IndexMut, Range},
};

use bumpalo::Bump;
use rand::prelude::*;
use rand_distr::StandardNormal;
use rand_xoshiro::Xoshiro256Plus;

use crate::{distance::*, embedding::Embedder, types::WindowedTimeseries};

pub struct TensorPool<'arena> {
    left: Vec<u32, &'arena Bump>,
    right: Vec<u32, &'arena Bump>,
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
        // dbg!(depth);
        let rm = mask(depth) as u32;

        let rindex = 
            self.pools[i]
                .right
                .iter()
                .zip(self.pools[j].right.iter())
                .enumerate()
                .find(|(_i, (ibits, jbits))| (**ibits & rm) == (**jbits & rm))?.0;

        if depth < self.hasher.k_right {
            return Some(rindex);
        }
        let lm = mask(depth - self.hasher.k_right) as u32;

        let lindex = 
            self.pools[i]
                .left
                .iter()
                .zip(self.pools[j].left.iter())
                .enumerate()
                .find(|(_i, (ibits, jbits))| (**ibits & lm) == (**jbits & lm))?.0;
            
        Some(lindex * self.hasher.tensor_repetitions + rindex)
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
        Some((start..self.idx, &self.hashes[start..self.idx]))
    }
}

thread_local! {
    static ZNORM_BUFFER: RefCell<Vec<f64>> = RefCell::new(Vec::new());
    static EMBED_BUFFER: RefCell<Vec<f64>> = RefCell::new(Vec::new());
}

/// Data structure to do LSH of subsequences.
pub struct Hasher {
    pub k: usize,
    pub k_left: usize,
    pub k_right: usize,
    pub tensor_repetitions: usize,
    pub repetitions: usize,
    pub planes: Vec<Vec<Vec<f64>>>,
    pub embedder: Embedder,
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

        let mut rng = Xoshiro256Plus::seed_from_u64(seed);
        let mut planes = Vec::with_capacity(tensor_repetitions);
        for _ in 0..tensor_repetitions {
            let mut ks = Vec::with_capacity(k);
            for _ in 0..k {
                let mut v = Vec::with_capacity(embedder.dim_in);
                for _ in 0..embedder.dim_in {
                    v.push(rng.sample(StandardNormal));
                }
                ks.push(v);
            }
            planes.push(ks);
        }

        Self {
            k,
            k_left,
            k_right,
            tensor_repetitions,
            repetitions,
            planes,
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
            {
                let mut zbuf = zbuf.borrow_mut();
                zbuf.resize(ts.w, 0.0);
                let m = ts.mean(i);
                let sd = ts.sd(i);
                for (j, &x) in ts.subsequence(i).iter().enumerate() {
                    zbuf[j] = (x - m) / sd;
                }
            }

            // do the embedding
            EMBED_BUFFER.with(|ebuf| {
                let mut ebuf = ebuf.borrow_mut();
                ebuf.resize(self.embedder.dim_out, 0.0);
                self.embedder.embed(&zbuf.borrow(), &mut ebuf);

                // and finally do the hashing itself
                let mask_right = mask(self.k_right);
                let mut words_left = Vec::with_capacity_in(self.tensor_repetitions, arena);
                let mut words_right = Vec::with_capacity_in(self.tensor_repetitions, arena);

                for rep in 0..self.tensor_repetitions {
                    let mut word = 0u64;
                    for bit in 0..self.k {
                        if dot(&ebuf, &self.planes[rep][bit]) >= 0.0 {
                            word = (word << 1) | 1;
                        } else {
                            word = word << 1;
                        }
                    }
                    words_left.push((word >> self.k_right) as u32);
                    words_right.push((word & mask_right) as u32);
                }
                TensorPool {
                    left: words_left,
                    right: words_right,
                }
            })
        })
    }
}
