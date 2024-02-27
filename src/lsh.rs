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

use crate::allocator::*;
use crate::distance::zeucl;
use crate::knn::Distance;
use crate::motifs::Motif;
use crate::timeseries::{FFTData, Overlaps, WindowedTimeseries};
use log::info;
use rand::prelude::*;
use rand_distr::{Normal, Uniform};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
// use statrs::distribution::{ContinuousCDF, Normal as NormalDistr};
use std::ops::Range;
use std::time::Duration;
use std::{sync::Arc, time::Instant};

pub const K: usize = 8;
pub const K_HALF: usize = K / 2;

#[derive(Ord, PartialOrd, Eq, PartialEq, Copy, Clone, Default)]
pub struct HashValue(pub u64);

impl From<[u8; K]> for HashValue {
    fn from(value: [u8; K]) -> Self {
        let hash = u64::from_le_bytes(value);
        Self(hash)
    }
}
impl std::fmt::Debug for HashValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let bytes = self.0.to_le_bytes();
        f.debug_list().entries(bytes.iter()).finish()
    }
}

#[derive(Default, Clone)]
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
        output: &mut [(u32, u32, Distance)],
        exclusion_zone: usize,
    ) -> Option<usize> {
        let mut idx = 0;
        output.fill((0, 0, Distance(0.0)));
        while self.current_bucket < self.buffers.buckets.len() {
            let range = self.buffers.buckets[self.current_bucket].clone();
            while self.i < range.end {
                while self.j < range.end {
                    debug_assert!(range.contains(&self.i));
                    debug_assert!(range.contains(&self.j));
                    let (ha, a) = self.buffers.hashes[self.i];
                    let (hb, b) = self.buffers.hashes[self.j];
                    debug_assert_eq!(ha, hb);
                    if !a.overlaps(b, exclusion_zone) {
                        output[idx] = (a.min(b), a.max(b), Distance(f64::INFINITY));
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
            } else if idx == 0 {
                return None;
            } else {
                return Some(idx);
            }
        }
        if idx == 0 {
            None
        } else {
            Some(idx)
        }
    }

    pub fn estimate_num_collisions(mut self, exclusion_zone: usize) -> usize {
        let mut cnt = 0;
        while self.current_bucket < self.buffers.buckets.len() {
            let range = self.buffers.buckets[self.current_bucket].clone();
            if range.len() as f64 > (self.buffers.hashes.len() as f64).sqrt() {
                // the bucket if _very_ large (relative to the number of subsequences),
                // so we just pick the square of its size in order to avoid spending
                // forever in iterating over pairs checking for overlaps
                log::trace!("Large bucket detected: {}", range.len());
                cnt += range.len() * range.len();
            } else {
                let mut i = range.start;
                while i < range.end {
                    let mut j = i + 1;
                    while j < range.end {
                        assert!(range.contains(&i));
                        assert!(range.contains(&j));
                        let (ha, a) = self.buffers.hashes[i];
                        let (hb, b) = self.buffers.hashes[j];
                        assert_eq!(ha, hb);
                        if !a.overlaps(b, exclusion_zone) {
                            cnt += 1;
                        }
                        j += 1;
                    }
                    i += 1;
                }
            }

            self.current_bucket += 1;
        }

        cnt
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
    let mut buf = vec![(0, 0, Distance(f64::INFINITY)); buf_size];
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
    let mut buf = vec![(0, 0, Distance(f64::INFINITY)); buf_size];
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
    let mut buf = vec![(0, 0, Distance(f64::INFINITY)); buf_size];
    let mut enumerated = 0;
    while let Some(cnt) = enumerator.next(&mut buf, 0) {
        enumerated += cnt;
    }
    assert_eq!(enumerated, tot_pairs);
}

/// Encapsulate a repetition index along with its left and
/// right tensor decomposition so that we don't have to recompute them all the time.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd, Ord, Eq)]
pub struct RepetitionIndex {
    pub repetition: usize,
    pub left_tensor_repetition: usize,
    pub right_tensor_repetition: usize,
}
impl From<usize> for RepetitionIndex {
    fn from(repetition: usize) -> Self {
        let (l, r) = get_minimal_index_pair(repetition);
        Self {
            repetition,
            left_tensor_repetition: l,
            right_tensor_repetition: r,
        }
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

//// This data structure contains all the information needed to generate the hash values for all the repeititions
//// for all the subsequences.
#[derive(Clone)]
pub struct HashCollection {
    pub hasher: Hasher,
    n_subsequences: usize,
    /// Pools of hash values, we have a vector for each tensor repetition, and each vector has an
    /// entry for the strings of K_HALF hash values, one for the left, and one for the right pool
    pools: Vec<Vec<([u8; K_HALF], [u8; K_HALF])>>,
    /// This table caches, in row-major order, the results of calls to
    /// [get_minimal_repetition] for different pairs i and j
    minimal_repetition_table: Vec<Option<usize>>,
}

impl HashCollection {
    /// How much memory would it be required to store information for these many repetitions?
    pub fn required_memory(ts: &WindowedTimeseries, repetitions: usize) -> Bytes {
        let tensor_repetitions = (repetitions as f64).sqrt().ceil() as usize;
        let bytes = tensor_repetitions * K * ts.num_subsequences() * 8;
        Bytes(bytes)
    }

    /// Get how many repetitions are being run
    pub fn get_repetitions(&self) -> usize {
        self.hasher.repetitions
    }

    pub fn collision_probability_at(&self, d: Distance) -> f64 {
        self.hasher.collision_probability_at(d.0)
    }

    pub fn failure_probability(
        &self,
        d: Distance,
        reps: usize,
        prefix: usize,
        prev_prefix: Option<usize>,
        prev_prefix_repetitions: Option<usize>,
    ) -> f64 {
        self.hasher
            .failure_probability(d.0, reps, prefix, prev_prefix, prev_prefix_repetitions)
    }

    pub fn failure_probability_independent(
        &self,
        d: Distance,
        reps: usize,
        prefix: usize,
        prev_prefix: Option<usize>,
        prev_prefix_repetitions: Option<usize>,
    ) -> f64 {
        self.hasher.failure_probability_independent(
            d.0,
            reps,
            prefix,
            prev_prefix,
            prev_prefix_repetitions,
        )
    }

    pub fn get_hasher(&self) -> &Hasher {
        &self.hasher
    }

    /// With this function we can construct a `HashCollection` from a `WindowedTimeseries`
    /// and a `Hasher`.

    pub fn from_ts(ts: &WindowedTimeseries, fft_data: &FFTData, hasher: Hasher) -> Self {
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
        info!("tensor pool building in {:?}", elapsed.as_secs_f64());

        let reps = hasher.repetitions;
        let minimal_repetition_table =
            Self::build_minimal_repetition_table(reps, hasher.tensor_repetitions);

        Self {
            hasher,
            n_subsequences: ns,
            pools,
            minimal_repetition_table,
        }
    }

    fn build_minimal_repetition_table(reps: usize, trep: usize) -> Vec<Option<usize>> {
        use std::collections::HashMap;
        let mut htable = HashMap::<(usize, usize), Option<usize>>::new();
        for i in 0..trep {
            for j in 0..trep {
                htable.insert((i, j), None);
            }
        }
        for rep in 0..reps {
            htable
                .entry(get_minimal_index_pair(rep))
                .and_modify(|entry| *entry = Some(rep));
        }
        let mut minimal_repetition_table: Vec<((usize, usize), Option<usize>)> =
            htable.into_iter().collect();

        minimal_repetition_table.sort();
        let table: Vec<Option<usize>> = minimal_repetition_table
            .into_iter()
            .map(|tup| tup.1)
            .collect();

        table
    }

    pub fn add_repetitions(
        &mut self,
        ts: &WindowedTimeseries,
        fft_data: &FFTData,
        total_repetitions: usize,
    ) {
        info!(
            "Adding new repetitions, up to a total of {}",
            total_repetitions
        );
        assert!(total_repetitions.is_power_of_two());
        assert!(total_repetitions > self.hasher.repetitions);
        self.hasher.add_repetitions(total_repetitions);

        let hasher = &self.hasher;

        let ns = ts.num_subsequences();

        let old_tensor_repetitions = self.pools.len();

        let it = (old_tensor_repetitions..hasher.tensor_repetitions)
            .into_par_iter()
            .map(|trep| {
                let mut repdata = vec![([0u8; K_HALF], [0u8; K_HALF]); ns];
                let mut max_dotp = 0.0f64;
                for k in 0..K_HALF {
                    let mdp1 = hasher.hash_all(ts, fft_data, k, trep, |i, h| {
                        let h = ((h as i64 & 0xFFi64) as i8) as u8;
                        repdata[i].0[k] = h;
                    });
                    let mdp2 = hasher.hash_all(ts, fft_data, k + K_HALF, trep, |i, h| {
                        let h = ((h as i64 & 0xFFi64) as i8) as u8;
                        repdata[i].1[k] = h;
                    });
                    max_dotp = max_dotp.max(mdp1.max(mdp2));
                }
                repdata
            });
        self.pools.par_extend(it);

        let reps = hasher.repetitions;
        self.minimal_repetition_table =
            Self::build_minimal_repetition_table(reps, hasher.tensor_repetitions);
    }

    pub fn half_hashes(&self, i: usize, repetition: RepetitionIndex) -> (&[u8], &[u8]) {
        let l = &self.pools[repetition.left_tensor_repetition][i].0;
        let r = &self.pools[repetition.right_tensor_repetition][i].1;
        (l, r)
    }

    #[cfg(test)]
    pub fn extended_hash_value(&self, i: usize, repetition: RepetitionIndex) -> [u8; K] {
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

    pub fn hash_value(&self, i: usize, prefix: usize, repetition: RepetitionIndex) -> HashValue {
        let mut hv: [u8; K] = [0; K];
        let (l, r) = self.half_hashes(i, repetition);
        for h in 0..(prefix / 2) {
            hv[2 * h] = l[h];
            hv[2 * h + 1] = r[h];
        }
        if prefix % 2 != 0 {
            let h = prefix / 2;
            hv[2 * h] = l[h];
        }
        let hash = u64::from_le_bytes(hv);
        HashValue(hash)
    }

    pub fn empirical_collision_probability(&self, i: usize, j: usize) -> f64 {
        fn count_equal(x: [u8; 4], y: [u8; 4]) -> usize {
            let mut cnt = 0;
            for (a, b) in x.iter().zip(&y) {
                if a == b {
                    cnt += 1;
                }
            }
            cnt
        }

        let collisions = self
            .pools
            .iter()
            .map(|rep| {
                let (hl_i, hr_i) = rep[i];
                let (hl_j, hr_j) = rep[j];
                count_equal(hl_i, hl_j) + count_equal(hr_i, hr_j)
            })
            .sum::<usize>();
        collisions as f64 / (self.pools.len() * 8) as f64
    }

    #[cfg(test)]
    pub fn first_collision_baseline(&self, i: usize, j: usize, prefix: usize) -> Option<usize> {
        (0..self.hasher.repetitions)
            .filter(|&rep| {
                let rep = rep.into();
                let ihash = &self.extended_hash_value(i, rep)[0..prefix];
                let jhash = &self.extended_hash_value(j, rep)[0..prefix];
                ihash == jhash
            })
            .next()
    }

    pub fn get_minimal_repetition(&self, lindex: usize, rindex: usize) -> Option<usize> {
        let idx = lindex * self.hasher.tensor_repetitions + rindex;
        self.minimal_repetition_table[idx]
    }

    /// Counts the number of collision at a given repetition and prefix length
    fn num_collisions(
        &self,
        repetition: RepetitionIndex,
        prefix: usize,
        exclusion_zone: usize,
        buffers: &mut ColumnBuffers,
    ) -> usize {
        self.group_subsequences(prefix, repetition, exclusion_zone, buffers, false);
        if let Some(mut enumerator) = buffers.enumerator() {
            let mut dummy_buffer: [(u32, u32, Distance); 1024] =
                [(0, 0, Distance::infinity()); 1024];
            let mut count = 0;
            while let Some(cnt) = enumerator.next(&mut dummy_buffer, exclusion_zone) {
                count += cnt;
            }
            count
        } else {
            0
        }
    }

    /// Estimates the number of collisions per prefix length and puts
    /// them in a vector. Uses only the first few repetitions to do the
    /// estimate
    pub fn average_num_collisions(
        &self,
        exclusion_zone: usize,
        buffers: &mut ColumnBuffers,
    ) -> Vec<f64> {
        let nreps = 8;
        let mut output = vec![0.0; K + 1];
        for prefix in (1..=K).rev() {
            for rep in 0..nreps {
                let rep = rep.into();
                output[prefix] += self.num_collisions(rep, prefix, exclusion_zone, buffers) as f64;
            }
        }
        for c in output.iter_mut() {
            *c /= nreps as f64;
        }
        output[0] = f64::INFINITY; // just to signal that we don't want to go to prefix 0
        output
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
        let res = self
            .get_minimal_repetition(lindex?, rindex?)
            .and_then(|rep| {
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
        repetition: RepetitionIndex,
        exclusion_zone: usize,
        buffers: &mut ColumnBuffers,
        parallel: bool,
    ) {
        assert!(repetition.left_tensor_repetition < self.hasher.tensor_repetitions);
        assert!(repetition.right_tensor_repetition < self.hasher.tensor_repetitions);
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
            // TODO: re-enable bucket sort here
            buffer.sort_unstable();
            // let mut scratch = vec![Default::default(); buffer.len()];
            // sort_hash_pairs(buffer, &mut scratch);
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
        log::trace!(
            "tag" = "profiling",
            "repetition" = repetition.repetition,
            "depth" = depth,
            "largest_bucket" = largest_bucket,
            "n_buckets" = output.len(),
            "time_bounds_s" = elapsed_boundaries.as_secs_f64(),
            "time_hashes_s" = elapsed_hashes.as_secs_f64(),
            "time_sort_s" = elapsed_sort.as_secs_f64(),
            "time_s" = (elapsed_hashes + elapsed_sort + elapsed_boundaries).as_secs_f64();
            "grouping subsequences"
        );
    }

    pub fn stats(
        &self,
        ts: &WindowedTimeseries,
        max_memory: Bytes,
        exclusion_zone: usize,
    ) -> HashCollectionStats {
        HashCollectionStats::new(self, ts, max_memory, exclusion_zone)
    }
}

/// A few statistics on a hash pool so to be able to estimate the cost of
/// running repetitions at a given prefix length.
#[derive(Debug, Clone)]
pub struct HashCollectionStats {
    /// the expected number of collisions for each prefix length
    expected_collisions: Vec<f64>,
    /// the cost of setting up a repetition
    repetition_setup_cost: f64,
    /// the cost of evaluating a collision
    collision_cost: f64,
    /// the maximum number of repetitions, fitting in the memory limit
    max_repetitions: usize,
}
impl HashCollectionStats {
    fn new(
        pool: &HashCollection,
        ts: &WindowedTimeseries,
        max_memory: Bytes,
        exclusion_zone: usize,
    ) -> Self {
        let mut max_repetitions = 16;
        while HashCollection::required_memory(ts, 2 * max_repetitions) <= max_memory {
            max_repetitions *= 2;
        }
        info!(
            "Maximum repetitions {} which would require {}",
            max_repetitions,
            HashCollection::required_memory(ts, max_repetitions)
        );

        let mut repetition_setup_cost = 0.0;
        let mut repetition_setup_cost_experiments = 0;
        let mut buffers = ColumnBuffers::default();
        let nreps = 4;
        let mut expected_collisions = vec![0.0; K + 1];
        let dat: Vec<(usize, usize, usize, f64)> = (1..=K)
            .into_par_iter()
            .flat_map(|prefix| (0..nreps).into_par_iter().map(move |rep| (prefix, rep)))
            .map_with(ColumnBuffers::default(), |mut buffers, (prefix, rep)| {
                let (collisions, setup_cost) =
                    Self::num_collisions(pool, rep.into(), prefix, exclusion_zone, &mut buffers);
                (prefix, rep, collisions, setup_cost)
            })
            .collect();
        for (prefix, _rep, collisions, setup_cost) in dat {
            expected_collisions[prefix] += collisions as f64;
            repetition_setup_cost += setup_cost;
            repetition_setup_cost_experiments += 1;
        }
        for c in expected_collisions.iter_mut() {
            *c /= nreps as f64;
        }
        expected_collisions[0] = f64::INFINITY; // just to signal that we don't want to go to prefix 0
        repetition_setup_cost /= repetition_setup_cost_experiments as f64;

        // now we estimate the cost of running a handful of distance computations,
        // as a proxy for the cost of handling the collisions.
        // TODO: maybe update this as we collect information during the execution?
        let mut collision_cost = 0.0;
        for (prefix, estimated_collisions) in expected_collisions.iter().enumerate().rev() {
            if *estimated_collisions > 100.0 {
                let mut cnt_dists = 0;
                pool.group_subsequences(prefix, 0.into(), exclusion_zone, &mut buffers, false);
                let t_start = Instant::now();
                let mut dists_buf = [(0, 0, Distance::infinity()); 1024];
                let mut enumerator = buffers.enumerator().unwrap();
                while let Some(cnt) = enumerator.next(&mut dists_buf, exclusion_zone) {
                    for (i, j, d) in &mut dists_buf[..cnt] {
                        *d = Distance(zeucl(ts, *i as usize, *j as usize));
                    }
                    cnt_dists += cnt;
                }
                let elapsed = Instant::now() - t_start;
                collision_cost = elapsed.as_secs_f64() / cnt_dists as f64;
                break;
            }
        }

        Self {
            expected_collisions,
            repetition_setup_cost,
            collision_cost,
            max_repetitions,
        }
    }

    /// Counts the number of collision at a given repetition and prefix length,
    /// and the cost of setting up the repetition
    fn num_collisions(
        pool: &HashCollection,
        repetition: RepetitionIndex,
        prefix: usize,
        exclusion_zone: usize,
        buffers: &mut ColumnBuffers,
    ) -> (usize, f64) {
        let t_start = Instant::now();
        pool.group_subsequences(prefix, repetition, exclusion_zone, buffers, false);
        let t_elapsed = Instant::now() - t_start;
        let count = buffers.enumerator().map_or(0, |enumerator| {
            enumerator.estimate_num_collisions(exclusion_zone)
        });
        (count, t_elapsed.as_secs_f64())
    }

    pub fn first_meaningful_prefix(&self) -> usize {
        self.expected_collisions
            .iter()
            .enumerate()
            .rev()
            .find(|(_prefix, collisions)| **collisions >= 1.0)
            .unwrap()
            .0
            .max(1)
    }

    /// For each prefix length, compute the cost to confirm a pair at
    /// the given distance.
    pub fn costs_to_confirm(
        &self,
        max_prefix: usize,
        d: Distance,
        delta: f64,
        hasher: &Hasher,
    ) -> Vec<(f64, usize)> {
        // let p = hasher.collision_probability_at(d.0);
        self.expected_collisions[..=max_prefix]
            .iter()
            .enumerate()
            .map(|(prefix, collisions)| {
                let maxreps = self.max_repetitions;
                let nreps = {
                    let mut nreps = 0;
                    let mut fp = 1.0;
                    while fp > delta && nreps < maxreps {
                        // FIXME: use the actual failure probability, which
                        // however depends on the number of repetitions which
                        // cannot be exceeded.
                        // fp = (1.0 - p.powi(prefix as i32)).powi(nreps as i32);
                        fp = hasher.failure_probability(d.0, nreps, prefix, None, None);
                        nreps += 1;
                    }
                    nreps
                };
                if nreps >= maxreps {
                    return (f64::INFINITY, maxreps);
                }
                (
                    nreps as f64 * (self.collision_cost * collisions + self.repetition_setup_cost),
                    nreps,
                )
            })
            .collect()
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
    rng: Xoshiro256PlusPlus,
}

impl Hasher {
    pub fn new(dimension: usize, repetitions: usize, width: f64, seed: u64) -> Self {
        let tensor_repetitions = (repetitions as f64).sqrt().ceil() as usize;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut vectors = Vec::with_capacity(dimension * K * tensor_repetitions);
        let mut shifts = Vec::with_capacity(K * tensor_repetitions);
        let normal = Normal::new(0.0, 1.0).expect("problem instantiating normal distribution");
        let uniform = Uniform::new(0.0, width);
        for _ in 0..(tensor_repetitions * K * dimension) {
            vectors.push(normal.sample(&mut rng));
        }
        for _ in 0..(tensor_repetitions * K) {
            shifts.push(uniform.sample(&mut rng));
        }

        Self {
            dimension,
            tensor_repetitions,
            repetitions,
            vectors,
            shifts,
            width,
            rng,
        }
    }

    /// add repetitions so that overall we have the required
    /// `total_repetitions`.
    pub fn add_repetitions(&mut self, total_repetitions: usize) {
        assert!(total_repetitions.is_power_of_two());
        assert!(total_repetitions > self.repetitions);
        let total_tensor_repetitions = (total_repetitions as f64).sqrt().ceil() as usize;
        let new_tensor_repetitions = total_tensor_repetitions - self.tensor_repetitions;
        let normal = Normal::new(0.0, 1.0).expect("problem instantiating normal distribution");
        let uniform = Uniform::new(0.0, self.width);
        for _ in 0..(new_tensor_repetitions * K * self.dimension) {
            self.vectors.push(normal.sample(&mut self.rng));
        }
        for _ in 0..(new_tensor_repetitions * K) {
            self.shifts.push(uniform.sample(&mut self.rng));
        }
        self.tensor_repetitions = total_tensor_repetitions;
        self.repetitions = total_repetitions;
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
        // let normal = NormalDistr::new(0.0, 1.0).unwrap();
        let normal = crate::stats::Normal::default();
        1.0 - 2.0 * normal.cdf(-r / d)
            - (2.0 / ((std::f64::consts::PI * 2.0).sqrt() * (r / d)))
                * (1.0 - (-r * r / (2.0 * d * d)).exp())
    }

    pub fn failure_probability_independent(
        &self,
        zeucl_dist: f64,
        reps: usize,
        prefix: usize,
        prev_prefix: Option<usize>,
        prev_prefix_repetitions: Option<usize>,
    ) -> f64 {
        let p = self.collision_probability_at(zeucl_dist);
        let cur_fail = (1.0 - p.powi(prefix as i32)).powi(reps as i32);
        let prev_fail = prev_prefix
            .zip(prev_prefix_repetitions)
            .map(|(prefix, repetitions)| {
                (1.0 - p.powi(prefix as i32)).powi(0i32.max(repetitions as i32 - reps as i32))
            })
            .unwrap_or(1.0);
        cur_fail * prev_fail
    }

    pub fn failure_probability(
        &self,
        zeucl_dist: f64,
        reps: usize,
        prefix: usize,
        prev_prefix: Option<usize>,
        prev_prefix_repetitions: Option<usize>,
    ) -> f64 {
        let p = self.collision_probability_at(zeucl_dist);

        // TODO: precompute all these numbers
        let cur_left_bits = (prefix as f64 / 2.0).floor() as i32;
        let cur_right_bits = (prefix as f64 / 2.0).ceil() as i32;
        assert_eq!(cur_left_bits + cur_right_bits, prefix as i32);

        let prev_prefixes = prev_prefix.map(|prefix| {
            (
                (prefix as f64 / 2.0).ceil() as i32,
                (prefix as f64 / 2.0).floor() as i32,
            )
        });

        let up_treps = (reps as f64 + 1.0).sqrt().floor() as i32;
        let low_treps = prev_prefix_repetitions
            .map(|repetitions| {
                let tensor_repetitions = (repetitions as f64).sqrt().ceil() as i32;
                if tensor_repetitions > up_treps {
                    tensor_repetitions - up_treps
                } else {
                    0
                }
            })
            .unwrap_or(0);

        // Probabilities of *not* colliding on a *single* repetition with a given number of bits
        let cur_left_fail = 1.0 - p.powi(cur_left_bits);
        let cur_right_fail = 1.0 - p.powi(cur_right_bits);

        let (prev_left_fail, prev_right_fail) = prev_prefixes
            .map(|(prev_left_bits, prev_right_bits)| {
                (1.0 - p.powi(prev_left_bits), 1.0 - p.powi(prev_right_bits))
            })
            // default to failure probability 1 on the previous
            // level if there was no previous level
            .unwrap_or((1.0, 1.0));

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

    /// Given a failure probability, returns a vectors of the distances
    /// achieving that failure probability for each prefix length and
    /// repetition.
    pub fn inverse_failure_probability(
        &self,
        ts: &WindowedTimeseries,
        delta: f64,
    ) -> Vec<(usize, usize, f64)> {
        let max_dist = ts.maximum_distance();
        let step = max_dist / 1000000.0;
        eprintln!("Step is {}", step);

        let mut res = Vec::new();
        let mut dist = 0.0;
        let mut previous_prefix = None;
        let previous_prefix_repetitions = Some(self.repetitions);
        for prefix in (1..K).rev() {
            eprintln!("prefix {prefix}");
            for rep in 0..self.repetitions {
                while self.failure_probability(
                    dist,
                    rep,
                    prefix,
                    previous_prefix,
                    previous_prefix_repetitions,
                ) < delta
                    && dist < max_dist
                {
                    dist += step;
                }
                res.push((prefix, rep, dist));
            }
            previous_prefix.replace(prefix);
        }

        res
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
            let probe_hasher = Hasher::new(ts.w, 1, r, seed);
            let probe_collection = HashCollection::from_ts(&ts, fft_data, probe_hasher);
            let probe_collection = Arc::new(probe_collection);
            probe_collection.group_subsequences(K, 0.into(), ts.w, &mut probe_buffers, true);
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

        let hasher = Hasher::new(w, repetitions, 5.0, 1245);
        let pools = HashCollection::from_ts(&ts, &fft_data, hasher);

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

    #[test]
    fn distance_progression_gap() {
        use std::io::prelude::*;
        let delta = 0.01;
        let w = 600;
        let ts = crate::load::loadts("data/GAP.csv.gz", Some(10000)).expect("problem loading data");
        let ts = crate::timeseries::WindowedTimeseries::new(ts, w, true);

        let repetitions = 1024;

        let width = Hasher::compute_width(&ts);
        let hasher = Arc::new(Hasher::new(w, repetitions, width, 1245));

        let dists = hasher.inverse_failure_probability(&ts, delta);
        let mut f = std::fs::File::create("dists-gap.csv").unwrap();
        writeln!(f, "prefix, rep, dist").unwrap();
        for (prefix, rep, dist) in dists {
            writeln!(f, "{prefix}, {rep}, {dist}").unwrap();
        }
    }

    #[test]
    fn distance_progression_ecg() {
        use std::io::prelude::*;
        let delta = 0.01;
        let w = 1000;
        let ts = crate::load::loadts("data/ECG.csv.gz", Some(10000)).expect("problem loading data");
        let ts = crate::timeseries::WindowedTimeseries::new(ts, w, true);

        let width = Hasher::compute_width(&ts);

        for repetitions in [1024, 4096] {
            let hasher = Arc::new(Hasher::new(w, repetitions, width, 1245));
            let dists = hasher.inverse_failure_probability(&ts, delta);
            let mut f = std::fs::File::create(format!("dists-ecg-{}.csv", repetitions)).unwrap();
            writeln!(f, "prefix, rep, dist").unwrap();
            for (prefix, rep, dist) in dists {
                writeln!(f, "{prefix}, {rep}, {dist}").unwrap();
            }
        }
    }
}
