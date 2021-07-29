use std::cell::RefCell;

use bumpalo::Bump;
use rand::prelude::*;
use rand_distr::StandardNormal;
use rand_xoshiro::Xoshiro256Plus;

use crate::{distance::*, embedding::Embedder, types::WindowedTimeseries};

#[derive(Debug)]
pub struct TensorPool<'arena> {
    left: Vec<u32, &'arena Bump>,
    right: Vec<u32, &'arena Bump>,
}

#[derive(Debug)]
pub struct HashCollection<'arena> {
    pools: Vec<TensorPool<'arena>, &'arena Bump>,
}

impl<'arena> HashCollection<'arena> {
    pub fn from_ts(ts: &WindowedTimeseries, hasher: &Hasher, arena: &'arena Bump) -> Self {
        let mut pools = Vec::with_capacity_in(ts.num_subsequences(), arena);
        for i in 0..ts.num_subsequences() {
            pools.push(hasher.hash(ts, i, arena));
        }
        Self { pools }
    }
}

thread_local! {
    static ZNORM_BUFFER: RefCell<Vec<f64>> = RefCell::new(Vec::new());
    static EMBED_BUFFER: RefCell<Vec<f64>> = RefCell::new(Vec::new());
}

/// Data structure to do LSH of subsequences.
pub struct Hasher {
    k: usize,
    k_left: usize,
    k_right: usize,
    tensor_repetitions: usize,
    repetitions: usize,
    planes: Vec<Vec<Vec<f64>>>,
    embedder: Embedder,
}

fn mask(bits: usize) -> u64 {
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
