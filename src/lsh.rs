use std::{
    cell::RefCell,
    cmp::Ordering,
    fmt::Debug,
    ops::{BitAnd, BitOr, BitXor, Not, Range, Shl, Shr},
};

use bumpalo::Bump;
use rand::prelude::*;
use rand_distr::Normal;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::{distance::*, embedding::Embedder, types::WindowedTimeseries};

/// Wrapper structx for 64-bits words, which sort lexicographically from the lowest significant bit
#[derive(Clone, Copy, Eq, PartialEq)]
pub struct HashValue(u64);

impl Debug for HashValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:064b}", self.0)
    }
}

impl PartialOrd for HashValue {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HashValue {
    fn cmp(&self, other: &Self) -> Ordering {
        // This comparator is equivalent to the following implementation
        // let selfstr = format!("{:064b}", self.0).chars().rev().collect::<String>();
        // let otherstr = format!("{:064b}", other.0).chars().rev().collect::<String>();
        // selfstr.cmp(&otherstr)
        if self.0 == other.0 {
            return Ordering::Equal;
        }
        let mut selfword = self.0;
        let mut otherword = other.0;
        for _ in 0..64 {
            let selfbit = selfword & 1;
            let otherbit = otherword & 1;
            if selfbit < otherbit {
                return Ordering::Less;
            }
            if selfbit > otherbit {
                return Ordering::Greater;
            }
            selfword = selfword >> 1;
            otherword = otherword >> 1;
        }
        panic!("equality is handled with the special case above");
    }
}

impl BitAnd for HashValue {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output {
        Self(self.0 & rhs.0)
    }
}

impl BitOr for HashValue {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitXor for HashValue {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self::Output {
        Self(self.0 ^ rhs.0)
    }
}

impl Not for HashValue {
    type Output = Self;

    fn not(self) -> Self::Output {
        Self(!self.0)
    }
}

impl Shr<usize> for HashValue {
    type Output = Self;

    fn shr(self, shift: usize) -> Self::Output {
        Self(self.0 >> shift)
    }
}

impl Shl<usize> for HashValue {
    type Output = Self;

    fn shl(self, shift: usize) -> Self::Output {
        Self(self.0 << shift)
    }
}

pub fn mask(bits: usize) -> HashValue {
    let mut m = 0;
    for _ in 0..bits {
        m = (m << 1) | 1;
    }
    HashValue(m)
}

pub struct TensorPool<'arena> {
    words: Vec<HashValue, &'arena Bump>,
}

impl<'arena> TensorPool<'arena> {
    fn from_words<I: IntoIterator<Item = HashValue>>(
        input_words: I,
        arena: &'arena Bump,
        reps: usize,
    ) -> Self {
        let mut words = Vec::with_capacity_in(reps, arena);
        for word in input_words.into_iter() {
            words.push(word);
        }
        TensorPool { words }
    }
}

pub struct HashCollection<'hasher, 'arena> {
    hasher: &'hasher Hasher,
    pools: Vec<TensorPool<'arena>, &'arena Bump>,
    select_left_mask: HashValue,
    select_right_mask: HashValue,
}

impl<'hasher, 'arena> HashCollection<'hasher, 'arena> {
    pub fn from_ts(ts: &WindowedTimeseries, hasher: &'hasher Hasher, arena: &'arena Bump) -> Self {
        let mut pools = Vec::with_capacity_in(ts.num_subsequences(), arena);
        for i in 0..ts.num_subsequences() {
            pools.push(hasher.hash(ts, i, arena));
        }
        let select_left_mask = !mask(hasher.k_right);
        let select_right_mask = mask(hasher.k_right);
        Self {
            hasher,
            pools,
            select_left_mask,
            select_right_mask,
        }
    }

    #[cfg(test)]
    pub fn first_collision_baseline(&self, i: usize, j: usize, depth: usize) -> Option<usize> {
        let m = mask(depth);
        (0..self.hasher.repetitions)
            .filter(|&rep| {
                let hi = self.hash_value(i, rep) & m;
                let hj = self.hash_value(j, rep) & m;
                hi == hj
            })
            .next()
    }

    pub fn first_collision(&self, i: usize, j: usize, depth: usize) -> Option<usize> {
        let rm = mask(depth) & self.select_right_mask;

        let rindex = self.pools[i]
            .words
            .iter()
            .zip(self.pools[j].words.iter())
            .enumerate()
            .find(|(_i, (ibits, jbits))| (**ibits & rm) == (**jbits & rm))?
            .0;

        if depth < self.hasher.k_right {
            return Some(rindex);
        }
        let lm = (mask(depth - self.hasher.k_right) << self.hasher.k_right) & self.select_left_mask;

        let lindex = self.pools[i]
            .words
            .iter()
            .zip(self.pools[j].words.iter())
            .enumerate()
            .find(|(_i, (ibits, jbits))| (**ibits & lm) == (**jbits & lm))?
            .0;

        let idx = lindex * self.hasher.tensor_repetitions + rindex;
        if idx < self.hasher.repetitions {
            Some(idx)
        } else {
            None
        }
    }

    pub fn collision_probability(&self, i: usize, j: usize) -> f64 {
        let m = mask(self.hasher.k);
        let n_collisions = count_collisions(&self.pools[i].words, &self.pools[j].words, m);

        n_collisions as f64
            / ((self.hasher.k_left + self.hasher.k_right) * self.hasher.tensor_repetitions) as f64
    }

    pub fn get_hash_matrix(&self) -> HashMatrix {
        HashMatrix::new(self)
    }

    fn hash_value(&self, i: usize, repetition: usize) -> HashValue {
        let l_idx = repetition / self.hasher.tensor_repetitions;
        let r_idx = repetition % self.hasher.tensor_repetitions;
        let pool = &self.pools[i];
        let l = pool.words[l_idx] & self.select_left_mask;
        let r = pool.words[r_idx] & self.select_right_mask;
        l | r
    }
}

fn count_collisions(wa: &[HashValue], wb: &[HashValue], bitmask: HashValue) -> usize {
    wa.iter()
        .zip(wb.iter())
        .map(|(&a, &b)| (!(a ^ b) & bitmask).0.count_ones() as usize)
        .sum()
}

pub struct HashMatrix {
    /// Outer vector has one entry per repetition, inner vector has one entry per subsequence,
    /// and items are hash values and indices into the timeseries
    hashes: Vec<Vec<(HashValue, usize)>>,
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
            debug_assert!(rephashes.is_sorted_by_key(|pair| pair.0));
            hashes.push(rephashes);
        }
        Self { hashes }
    }

    pub fn buckets<'hashes>(
        &'hashes self,
        depth: usize,
        repetition: usize,
    ) -> BucketIterator<'hashes> {
        debug_assert!(self.hashes[repetition].is_sorted_by_key(|pair| pair.0));
        BucketIterator {
            hashes: &self.hashes[repetition],
            mask: mask(depth),
            idx: 0,
        }
    }
}

pub struct BucketIterator<'hashes> {
    hashes: &'hashes Vec<(HashValue, usize)>,
    mask: HashValue,
    idx: usize,
}

impl<'hashes> Iterator for BucketIterator<'hashes> {
    type Item = (Range<usize>, &'hashes [(HashValue, usize)]);

    fn next(&mut self) -> Option<Self::Item> {
        if self.idx >= self.hashes.len() {
            return None;
        }
        let start = self.idx;
        let current = self.hashes[self.idx].0 & self.mask;
        while self.idx < self.hashes.len() && (self.hashes[self.idx].0 & self.mask) == current {
            self.idx += 1;
        }
        // println!("boundary {:?} (mask {:?}) size {}", current, self.mask, self.idx - start);
        Some((start..self.idx, &self.hashes[start..self.idx]))
    }
}

thread_local! {
    static ZNORM_BUFFER: RefCell<Vec<f64>> = RefCell::new(Vec::new());
    static EMBED_BUFFER: RefCell<Vec<f64>> = RefCell::new(Vec::new());
}

struct HyperplaneLSH {
    /// Organized like a three dimensional matrix
    rands: Vec<f64>,
    repetitions: usize,
    k: usize,
    dim: usize,
}

impl HyperplaneLSH {
    fn new(dim: usize, k: usize, repetitions: usize, seed: u64) -> Self {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut rands = Vec::with_capacity(dim * k * repetitions);
        let normal = Normal::new(0.0, 1.0).expect("problem instantiating normal distribution");
        for _ in 0..(repetitions * k * dim) {
            rands.push(normal.sample(&mut rng));
        }

        Self {
            rands,
            repetitions,
            k,
            dim,
        }
    }

    fn plane(&self, repetition: usize, bit: usize) -> &'_ [f64] {
        let idx = repetition * self.k * self.dim + bit * self.dim;
        &self.rands[idx..idx + self.dim]
    }

    fn hash<'hasher>(
        &'hasher self,
        v: &'hasher [f64],
    ) -> impl Iterator<Item = HashValue> + 'hasher {
        let mut repetition = 0;
        std::iter::from_fn(move || {
            if repetition >= self.repetitions {
                return None;
            }
            let mut word = 0u64;
            for bit in 0..self.k {
                if dot(v, &self.plane(repetition, bit)) > 0.0 {
                    word = (word << 1) | 1;
                } else {
                    word = word << 1;
                }
            }
            repetition += 1;
            Some(HashValue(word))
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
            zbuf.borrow_mut().clear();
            ts.znormalized(i, &mut zbuf.borrow_mut());

            // do the embedding
            EMBED_BUFFER.with(|ebuf| {
                let mut ebuf = ebuf.borrow_mut();
                ebuf.clear();
                ebuf.resize(self.embedder.dim_out, 0.0);
                self.embedder.embed(&zbuf.borrow(), &mut ebuf);

                // and finally do the hashing itself and construct the tensor pool
                TensorPool::from_words(self.hyperplanes.hash(&ebuf), arena, self.tensor_repetitions)
            })
        })
    }
}

#[cfg(test)]
mod test {
    use crate::distance::*;
    use crate::embedding::*;
    use crate::lsh::*;

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

            let angle = crate::distance::dot(&a, &b).acos();

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
    fn test_probability_k() {
        use rand::prelude::*;
        use rand_distr::StandardNormal;
        use rand_xoshiro::Xoroshiro128PlusPlus;
        let mut rng = Xoroshiro128PlusPlus::seed_from_u64(3462);
        let dim = 10;
        let repetitions = 10000;
        let n = 10;

        for k in 1..=8 {
            let hasher = HyperplaneLSH::new(dim, k, repetitions, 13462);

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

            let mut bs: Vec<(f64, Vec<f64>)> = (0..n)
                .map(|_| {
                    let b: Vec<f64> = StandardNormal.sample_iter(&mut rng).take(dim).collect();
                    let b = normalize(&b);
                    let angle = crate::distance::dot(&a, &b).acos();
                    let expected_prob = (1.0 - angle / (std::f64::consts::PI)).powi(k as i32);
                    (expected_prob, b)
                })
                .collect();
            bs.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap().reverse());

            for (expected_prob, b) in bs {
                if expected_prob >= 0.8 {
                    continue;
                }
                let prob = hasher
                    .hash(&a)
                    .zip(hasher.hash(&b))
                    .filter(|(ha, hb)| ha == hb)
                    .count() as f64
                    / repetitions as f64;
                println!("expected: {} actual: {}", expected_prob, prob);
                println!("dot product {}", dot(&a, &b));

                assert!(
                    (expected_prob - prob).abs() <= 0.05,
                    "discrepancy in probabilities for k={}:\n\texpected: {}\n\tactual:   {}",
                    k,
                    expected_prob,
                    prob
                );
            }
        }
    }

    #[test]
    fn test_first_collision() {
        let w = 300;
        let ts = crate::load::loadts("data/ECG.csv", Some(500)).expect("problem loading data");
        let ts = crate::WindowedTimeseries::new(ts, w);
        let sf = scaling_factor(&ts, zeucl, 0.01);

        let k = 32;
        let repetitions = 200;

        let hasher = Hasher::new(k, repetitions, Embedder::new(ts.w, ts.w, sf, 1245), 1245);
        let arena = bumpalo::Bump::new();
        let pools = HashCollection::from_ts(&ts, &hasher, &arena);

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
