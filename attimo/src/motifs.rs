//// # Motifs

//// Finding motifs in time series. Instead of computing the full matrix profile,
//// leverage [LSH](src/lsh.html) to check only pairs that are probably near.
//// The data structure used for the task is adaptive to the data, and is configured
//// to respect the limits of the system in terms of memory.

use crate::alloc_cnt;
use crate::allocator::allocated;
use crate::distance::*;
use crate::lsh::*;
use crate::timeseries::*;
use indicatif::ProgressBar;
use indicatif::ProgressStyle;
use rayon::prelude::*;
use slog_scope::info;
use std::cell::RefCell;
use std::ops::Range;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;
use thread_local::ThreadLocal;
#[derive(Debug, PartialEq, PartialOrd)]
struct OrderedF64(f64);

impl Eq for OrderedF64 {}

impl Ord for OrderedF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

//// ## Support data structures
//// ### Motifs

//// This data structure stores information about a motif:
////
////  - The index of the two subsequences defining the motif
////  - The distance between the two subsequences
////  - The LSH collision probability two subsequences
////  - The elapsed time since the start of the algorithm until
////    when the motif was found
////
//// Some utility functions follow.
#[derive(Clone, Copy, Debug)]
pub struct Motif {
    pub idx_a: usize,
    pub idx_b: usize,
    pub distance: f64,
    /// When the motif was confirmed
    pub elapsed: Option<Duration>,
}

impl Eq for Motif {}
impl PartialEq for Motif {
    fn eq(&self, other: &Self) -> bool {
        self.idx_a == other.idx_a && self.idx_b == other.idx_b && self.distance == other.distance
    }
}

impl Ord for Motif {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl PartialOrd for Motif {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.distance
            .partial_cmp(&other.distance)
            .map(|ord| ord.then_with(|| self.idx_a.cmp(&other.idx_a)))
    }
}

//// An important part of working with motifs is defining and removing
//// _trivial matches_. With the function `Motif::overlaps` we can detect
//// whether two motifs overlap according to the given `exclusion_zone`:
//// if any two indices in the two motifs are at distance less than
//// `exclusion_zone` from each other, then the motifs overlap and one of them
//// shall be discarded.
impl Motif {
    /// Tells whether the two motifs overlap, in order to avoid storing trivial matches
    fn overlaps(&self, other: &Self, exclusion_zone: usize) -> bool {
        let mut idxs = [self.idx_a, self.idx_b, other.idx_a, other.idx_b];
        idxs.sort_unstable();

        idxs[0] + exclusion_zone > idxs[1]
            || idxs[1] + exclusion_zone > idxs[2]
            || idxs[2] + exclusion_zone > idxs[3]
    }
}

//// ### Top-k data structure

//// With our algorithm we look for the top motifs, that is a configurable
//// number of non-overlapping motifs in increasing order of distance.
//// This data structure implements a buffer, holding up to `k` sorted motifs,
//// such that no two motifs in the data structure are overlapping,
//// according to the parameter `exclusion_zone`.
#[derive(Clone)]
pub struct TopK {
    k: usize,
    exclusion_zone: usize,
    top: Vec<Motif>,
}

impl std::fmt::Debug for TopK {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, m) in self.top.iter().enumerate() {
            writeln!(
                f,
                "  {} ::: {} -- {}  ({:.4}) ({:?})",
                i, m.idx_a, m.idx_b, m.distance, m.elapsed
            )?;
        }

        Ok(())
    }
}

impl TopK {
    pub fn new(k: usize, exclusion_zone: usize) -> Self {
        Self {
            k,
            exclusion_zone,
            top: Vec::new(),
        }
    }

    //// When inserting into the data structure, we first check, in order of distance,
    //// if there is a pair whose defining motif is closer than the one being inserted,
    //// and which is also overlapping.
    pub fn insert(&mut self, motif: Motif) {
        let mut i = 0;
        while i < self.top.len() && self.top[i].distance <= motif.distance {
            if motif.overlaps(&self.top[i], self.exclusion_zone) {
                //// If this is the case, we don't insert the motif, and return.
                return;
            }
            i += 1;
        }

        //// Otherwise, we insert the motif in the correct position.
        //// Because of this the `top` array is always in sorted
        //// order of increasing distance
        self.top.insert(i, motif);

        //// After the insertion we make sure that there are no other
        //// motifs overlapping with the one just inserted.
        //// To this end we remove from the tail of the vector all motifs
        //// that overlap with the one just inserted.
        ////
        //// One consequence of this is that among several trivial matches of
        //// the same motif, the one with the smallest distance is selected.
        //// In fact, this should be equivalent to just sorting all pairs of subsequences
        //// based on their distance, and then proceed from the one with smallest distance
        //// removing trivial matches along the way.
        i += 1;
        while i < self.top.len() {
            if self.top[i].overlaps(&motif, self.exclusion_zone) {
                self.top.remove(i);
            } else {
                i += 1;
            }
        }

        debug_assert!(self.top.is_sorted());

        //// Finally, we retain only `k` elements
        if self.top.len() > self.k {
            for m in &self.top[self.k..] {
                assert!(m.elapsed.is_none());
            }
            self.top.truncate(self.k);
        }
    }

    //// This function is used to access the k-th motif, if
    //// we already found it, even if not confirmed yet
    pub fn k_th(&self) -> Option<Motif> {
        if self.top.len() == self.k {
            self.top.last().map(|mot| *mot)
        } else {
            None
        }
    }

    pub fn first_not_confirmed(&self) -> Option<Motif> {
        self.top
            .iter()
            .filter(|m| m.elapsed.is_none())
            .next()
            .map(|m| *m)
    }

    pub fn first_outstanding_mut(&mut self) -> Option<&mut Motif> {
        self.top
            .iter_mut()
            .filter(|m| m.elapsed.is_none())
            .next()
    }

    pub fn first_outstanding_dist(&self) -> Option<f64> {
        self.top
            .iter()
            .filter(|m| m.elapsed.is_none())
            .next()
            .map(|m| m.distance)
    }

    pub fn last_confirmed(&self) -> Option<Motif> {
        self.top
            .iter()
            .filter(|m| m.elapsed.is_some())
            .last()
            .map(|m| *m)
    }

    pub fn num_confirmed(&self) -> usize {
        self.confirmed().count()
    }

    pub fn confirmed(&self) -> impl Iterator<Item = Motif> + '_ {
        self.top.iter().filter(|m| m.elapsed.is_some()).map(|m| *m)
    }

    pub fn for_each(&mut self, f: impl FnMut(&mut Motif)) {
        self.top.iter_mut().for_each(f)
    }

    pub fn len(&self) -> usize {
        self.top.len()
    }

    pub fn to_vec(&self) -> Vec<Motif> {
        self.top.clone().into_iter().collect()
    }

    pub fn add_all(&mut self, other: &mut TopK) {
        for m in other.top.drain(..) {
            self.insert(m);
        }
    }
}

/// Iterator over the motifs of a time series, which are returned in increasing order of distance
/// 
/// # Examples
/// 
/// ```
/// use attimo::timeseries::WindowedTimeseries;
/// use attimo::motifs::*;
/// 
/// let w = 100;
/// // Load some data, the first 10000 data points of an ECG trace
/// let ts = WindowedTimeseries::new(attimo::load::loadts("data/ECG.csv.gz", Some(10000)).unwrap(), w, false);
/// let mut motifs = MotifIterator::new(&ts, 10, 100, 0.01, 1234);
/// let m = motifs.next().unwrap();
/// assert_eq!(m.idx_a, 616);
/// assert_eq!(m.idx_b, 2780);
/// assert_eq!(m.distance, 0.17614364917722336);
/// ```
pub struct MotifIterator<'a> {
    ts: &'a WindowedTimeseries,
    max_topk: usize,
    repetitions: usize,
    delta: f64,
    start: Instant,
    exclusion_zone: usize,
    hasher: Arc<Hasher>,
    pools: Arc<HashCollection>,
    prev_prefix: Option<usize>,
    cur_prefix: usize,
    cur_repetition: usize,
    //// This vector holds the (sorted) hashed subsequences, and their index
    column_buffer: Vec<(HashValue, u32)>,
    //// This vector holds the boundaries between buckets. We reuse the allocations
    buckets: Vec<Range<usize>>,
    tl_top: ThreadLocal<RefCell<TopK>>,
    top: TopK,
}

impl<'a> MotifIterator<'a> {
    pub fn new(ts: &'a WindowedTimeseries, max_topk: usize, repetitions: usize, delta: f64, seed: u64) -> Self {
        let start = Instant::now();
        let exclusion_zone = ts.w;

        let fft_data = FFTData::new(&ts);

        let hasher_width = Hasher::estimate_width(
            &ts,
            &fft_data,
            1,
            None,
            seed,
        );
        let hasher = Arc::new(Hasher::new(ts.w, repetitions, hasher_width, seed));
        let pools = HashCollection::from_ts(ts, &fft_data, Arc::clone(&hasher));
        let pools = Arc::new(pools);
        drop(fft_data);


        Self {
            ts,
            max_topk,
            repetitions,
            delta,
            start,
            exclusion_zone,
            hasher,
            pools,
            prev_prefix: None,
            cur_prefix: K,
            cur_repetition: 0,
            column_buffer: Vec::new(),
            buckets: Vec::new(),
            tl_top: ThreadLocal::new(),
            top: TopK::new(max_topk, exclusion_zone)
        }
    }

    fn stopping_condition(&self, d: f64, prefix: usize, prev_prefix: Option<usize>, repetition: usize) -> bool {
        let p = self.hasher.collision_probability_at(d);
        let i_half = prefix as f64 / 2.0;
        let sqrt = (self.repetitions as f64).sqrt().ceil() as i32;
        let j_left = repetition as i32 / sqrt;
        let j_right = repetition as i32 % sqrt;
        let failure_p = if let Some(previous) = prev_prefix {
            let prev_half = previous as f64 / 2.0;
            let lu_i = 1.0 - (1.0 - p.powf(i_half)).powi(j_left);
            let ru_i = 1.0 - (1.0 - p.powf(i_half)).powi(j_right);
            let lu_ip = 1.0 - (1.0 - p.powf(prev_half)).powi(j_left);
            let ru_ip = 1.0 - (1.0 - p.powf(prev_half)).powi(j_right);
            let ll_ip = 1.0 - (1.0 - p.powf(prev_half)).powi(sqrt - j_left);
            let rl_ip = 1.0 - (1.0 - p.powf(prev_half)).powi(sqrt - j_right);
            (1.0 - lu_i * ru_i)
                * (1.0 - lu_ip * rl_ip)
                * (1.0 - ll_ip * ru_ip)
                * (1.0 - ll_ip * rl_ip)
        } else {
            let lu_i = 1.0 - (1.0 - p.powf(i_half)).powi(j_left);
            let ru_i = 1.0 - (1.0 - p.powf(i_half)).powi(j_right);
            1.0 - lu_i * ru_i
        };
        failure_p <= self.delta
    }

    fn prefix_for_distance(&self, d: f64, mut prefix: usize) -> usize {
        let initial = prefix;
        while prefix > 0 {
            for rep in 0..self.repetitions {
                if self.stopping_condition(d, prefix, Some(initial), rep) {
                    return prefix;
                }
            }
            prefix -= 1;
        }
        panic!()
    }

}

impl<'a> Iterator for MotifIterator<'a> {
    type Item = Motif;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            assert!(self.cur_prefix > 0, "Empty prefixes are not supported!");

            // Emit the first outstanding motif, if any
            if let Some(d) = self.top.first_outstanding_dist() {
                if self.stopping_condition(d, self.cur_prefix, self.prev_prefix, self.cur_repetition) {
                    let outstanding = self.top.first_outstanding_mut().unwrap();
                    assert!(outstanding.distance == d);
                    outstanding.elapsed.replace(self.start.elapsed());
                    return Some(outstanding.clone())
                }
            }
            if self.top.num_confirmed() == self.max_topk {
                // We are done with this iterator
                return None;
            }

            // advance the iterator by one repetition, or decrease the prefix and reset repetition
            if self.cur_repetition + 1 == self.repetitions {
                self.cur_repetition = 0;
                self.prev_prefix.replace(self.cur_prefix);
                if let Some(d) = self.top.first_outstanding_dist() {
                    self.cur_prefix = self.prefix_for_distance(d, self.cur_prefix);
                } else {
                    self.cur_prefix -= 1;
                }
                assert!(self.cur_prefix > 0, "Empty prefixes are not supported!");
            } else {
                self.cur_repetition += 1;
            }

            // Do the repetition
            self.pools.group_subsequences(self.cur_prefix, self.cur_repetition, self.exclusion_zone, &mut self.column_buffer, &mut self.buckets);
            let n_buckets = self.buckets.len();
            // Each thread works on these many buckets at one time, to reduce the
            // overhead of scheduling.
            let chunk_size = std::cmp::max(1, n_buckets / (4 * rayon::current_num_threads()));

            (0..n_buckets / chunk_size)
                .into_par_iter()
                .for_each(|chunk_i| {
                    // let tl_top = tl_top.get_or(|| RefCell::new(TopK::new(topk, exclusion_zone)));
                    let tl_top = self.tl_top.get_or(|| RefCell::new(self.top.clone()));

                    for i in (chunk_i * chunk_size)..((chunk_i + 1) * chunk_size) {
                        let bucket = &self.column_buffer[self.buckets[i].clone()];

                        for (_, a_idx) in bucket.iter() {
                            let a_idx = *a_idx as usize;
                            // let a_already_checked = rep_bounds[a_idx].clone();
                            // let a_hash_idx = hash_range.start + a_offset;
                            for (_, b_idx) in bucket.iter() {
                                let b_idx = *b_idx as usize;
                                //// Here we handle trivial matches: we don't consider a pair if the difference between
                                //// the subsequence indexes is smaller than the exclusion zone, which is set to `w/4`.
                                if a_idx + self.exclusion_zone < b_idx {

                                    //// We only process the pair if this is the first repetition in which
                                    //// they collide. We get this information from the pool of bits
                                    //// from which hash values for all repetitions are extracted.
                                    if let Some(first_colliding_repetition) =
                                        self.pools.first_collision(a_idx, b_idx, self.cur_prefix)
                                    {
                                        //// This is the first collision in this iteration, _and_ the pair didn't collide
                                        //// at a deeper level.
                                        if first_colliding_repetition == self.cur_repetition
                                            && self.prev_prefix
                                                .map(|d| {
                                                    self.pools.first_collision(a_idx, b_idx, d).is_none()
                                                })
                                                .unwrap_or(true)
                                        {
                                            //// After computing the distance between the two subsequences,
                                            //// we try to insert the pair in the top data structure
                                            let d = zeucl(self.ts, a_idx, b_idx);
                                            if d.is_finite() {
                                                let m = Motif {
                                                    idx_a: a_idx as usize,
                                                    idx_b: b_idx as usize,
                                                    distance: d,
                                                    elapsed: None,
                                                };
                                                tl_top.borrow_mut().insert(m);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                });

            // Add to the top-k all the new top pairs that have been found
            let found_motifs = self.tl_top
                .iter_mut()
                .flat_map(|top| top.borrow().to_vec());
            for m in found_motifs {
                self.top.insert(m);
            }
        }
    }
}

pub fn motifs(
    ts: &WindowedTimeseries,
    topk: usize,
    repetitions: usize,
    delta: f64,
    seed: u64,
) -> Vec<Motif> {
    MotifIterator::new(ts, topk, repetitions, delta, seed).collect()
}

#[cfg(test)]
mod test {
    use crate::{load::loadts, timeseries::WindowedTimeseries};

    use super::*;

    #[test]
    #[ignore]
    fn test_motif_ecg_10000() {
        // The indices and distances in this test have been computed
        // using SCAMP: https://github.com/zpzim/SCAMP
        // The distances are slightly different, due to numerical approximation
        // and a different normalization in their computation of the standard deviation
        for (w, a, b, d) in [
            (100, 616, 2780, 0.1761538477),
            (200, 416, 2580, 0.3602377446),
        ] {
            let ts: Vec<f64> = loadts("../data/ECG.csv.gz", Some(10000)).unwrap();
            let ts = WindowedTimeseries::new(ts, w, true);

            let motif = *motifs(&ts, 1,20, 0.001, 12435)
                .first()
                .unwrap();
            println!(
                "{} -- {} actual {} expected {}",
                motif.idx_a, motif.idx_b, motif.distance, d
            );
            assert!((motif.idx_a as isize - a as isize).abs() < w as isize);
            assert!((motif.idx_b as isize - b as isize).abs() < w as isize);
            assert!(motif.distance <= d + 0.00001);
        }
    }

    #[test]
    #[ignore]
    fn test_motif_ecg_full() {
        // The indices and distances in this test have been computed
        // using SCAMP: https://github.com/zpzim/SCAMP
        // The distances are slightly different, due to numerical approximation
        // and a different normalization in their computation of the standard deviation
        for (w, a, b, d) in [(1000, 7137168, 7414108, 0.3013925657)] {
            let ts: Vec<f64> = loadts("../data/ECG.csv.gz", None).unwrap();
            let ts = WindowedTimeseries::new(ts, w, true);
            // assert!((crate::distance::zeucl(&ts, a, b) - d) < 0.00000001);

            let motif = *motifs(&ts, 1, 200, 0.001, 12435)
                .first()
                .unwrap();
            println!("Motif distance {}", motif.distance);
            // We consider the test passed if we find a distance smaller than the one found by SCAMP,
            // and the motif instances are located within w steps from the ones found by SCAMP.
            // These differences are due to differences in floating point computations
            assert!(motif.distance <= d);
            assert!((motif.idx_a as isize - a as isize).abs() < w as isize);
            assert!((motif.idx_b as isize - b as isize).abs() < w as isize);
        }
    }

    #[test]
    #[ignore]
    fn test_motif_ecg_top10() {
        // as in the other examples, the ground truth is obtained using SCAMP run on the GPU
        let top10 = [
            (7137166, 7414106, 0.3013925657),
            (7377870, 7383302, 0.343015406),
            (7553828, 7587436, 0.3612951315),
            (6779076, 7379224, 0.3880223353),
            (7238944, 7264944, 0.3938163096),
            (7574696, 7611520, 0.3942701023),
            (7094136, 7220980, 0.3981813093),
            (6275400, 6298896, 0.3989290683),
            (6625400, 7479248, 0.4026470338),
            (6961239, 7385163, 0.4042721064),
        ];

        let w = 1000;
        let ts: Vec<f64> = loadts("../data/ECG.csv.gz", None).unwrap();
        let ts = WindowedTimeseries::new(ts, w, false);

        let motifs = motifs(&ts, 10, 200, 0.01, 12435);
        for (a, b, dist) in top10 {
            // look for this in the motifs, allowing up to w displacement
            println!("looking for ({a} {b} {dist})");
            let mut found = false;
            for motif in &motifs {
                found |= (motif.idx_a as isize - a as isize).abs() <= w as isize;
                found |= (motif.idx_b as isize - b as isize).abs() <= w as isize;
                if found {
                    println!(
                        "   found at ({} {} {})",
                        motif.idx_a, motif.idx_b, motif.distance
                    );
                    break;
                }
            }
            assert!(
                found,
                "Could not find ({}, {}, {}) in {:?}",
                a, b, dist, motifs
            );
        }
    }

    #[test]
    #[ignore]
    fn test_motif_astro_top10() {
        // as in the other examples, the ground truth is obtained using SCAMP run on the GPU
        let top10 = [
            (609810, 888455, 1.264327903),
            (502518, 656063, 1.312459673),
            (321598, 423427, 1.368041725),
            (342595, 625081, 1.403194924),
            (218448, 1006871, 1.442935122),
            (192254, 466432, 1.523167513),
            (527024, 533903, 1.526611152),
            (520191, 743708, 1.558780057),
            (192097, 193569, 1.583277835),
            (267982, 512333, 1.617081054),
        ];

        let w = 100;
        let ts: Vec<f64> = loadts("../data/ASTRO.csv.gz", None).unwrap();
        let ts = WindowedTimeseries::new(ts, w, false);

        let motifs = motifs(&ts, 10, 800, 0.01, 12435);
        for (a, b, dist) in top10 {
            // look for this in the motifs, allowing up to w displacement
            println!("looking for ({a} {b} {dist})");
            let mut found = false;
            for motif in &motifs {
                found |= (motif.idx_a as isize - a as isize).abs() <= w as isize;
                found |= (motif.idx_b as isize - b as isize).abs() <= w as isize;
                if found {
                    println!(
                        "   found at ({} {} {})",
                        motif.idx_a, motif.idx_b, motif.distance
                    );
                    break;
                }
            }
            assert!(
                found,
                "Could not find ({}, {}, {}) in {:?}",
                a, b, dist, motifs
            );
        }
    }

}
