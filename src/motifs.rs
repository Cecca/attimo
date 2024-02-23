//// # Motifs

//// Finding motifs in time series. Instead of computing the full matrix profile,
//// leverage [LSH](src/lsh.html) to check only pairs that are probably near.
//// The data structure used for the task is adaptive to the data, and is configured
//// to respect the limits of the system in terms of memory.

use crate::distance::*;
use crate::knn::*;
use crate::lsh::*;
use crate::timeseries::*;
use indicatif::ProgressBar;
use indicatif::ProgressStyle;
use log::*;
use rayon::prelude::*;
use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
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
    /// when the motif was first found
    pub discovered: Duration,
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
impl Overlaps<Motif> for Motif {
    /// Tells whether the two motifs overlap, in order to avoid storing trivial matches
    fn overlaps(&self, other: Self, exclusion_zone: usize) -> bool {
        let mut idxs = [self.idx_a, self.idx_b, other.idx_a, other.idx_b];
        idxs.sort_unstable();

        idxs[0] + exclusion_zone > idxs[1]
            || idxs[1] + exclusion_zone > idxs[2]
            || idxs[2] + exclusion_zone > idxs[3]
    }
}

fn nearest_neighbor_bf(
    ts: &WindowedTimeseries,
    from: usize,
    fft_data: &FFTData,
    exclusion_zone: usize,
    distances: &mut [f64],
    buf: &mut [f64],
) -> (Distance, usize) {
    // Check that the auxiliary memory buffers are correctly sized
    assert_eq!(distances.len(), ts.num_subsequences());
    assert_eq!(buf.len(), ts.w);

    // Compute the distance profile using the MASS algorithm
    ts.distance_profile(&fft_data, from, distances, buf);

    // Pick the nearest neighbor skipping overlapping subsequences
    let mut nearest = f64::INFINITY;
    let mut nearest_idx = 0;
    for (j, &d) in distances.iter().enumerate() {
        if !j.overlaps(from, exclusion_zone) {
            if d < nearest {
                nearest = d;
                nearest_idx = j;
            }
        }
    }
    (Distance(nearest), nearest_idx)
}

pub fn brute_force_motifs(ts: &WindowedTimeseries, k: usize, exclusion_zone: usize) -> Vec<Motif> {
    let start = Instant::now();
    // pre-compute the FFT for the time series
    let fft_data = FFTData::new(&ts);
    let n = ts.num_subsequences();

    // initialize some auxiliary buffers, which will be cloned on a
    // per-thread basis.
    let mut distances = Vec::new();
    distances.resize(n, 0.0f64);
    let mut buf = Vec::new();
    buf.resize(ts.w, 0.0f64);

    // compute all k-nearest neighborhoods
    let mut nns: Vec<Motif> = (0..n)
        .into_par_iter()
        .map_with((distances, buf), |(distances, buf), i| {
            let (d, j) = nearest_neighbor_bf(ts, i, &fft_data, exclusion_zone, distances, buf);
            Motif {
                idx_a: i.min(j),
                idx_b: i.max(j),
                distance: d.0,
                elapsed: Some(start.elapsed()),
                discovered: start.elapsed(),
            }
        })
        .collect();

    nns.sort_unstable();

    let mut res = Vec::new();
    let mut i = 0;
    while res.len() < k && i < nns.len() {
        if !nns[i].overlaps(res.as_slice(), exclusion_zone) {
            res.push(nns[i]);
        }
        i += 1;
    }

    res
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
    current_non_overlapping: Vec<Motif>,
    should_update: bool,
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
            current_non_overlapping: Vec::new(),
            should_update: false,
        }
    }

    //// When inserting into the data structure, we first check, in order of distance,
    //// if there is a pair whose defining motif is closer than the one being inserted,
    //// and which is also overlapping.
    pub fn insert(&mut self, motif: Motif) {
        let mut i = 0;
        while i < self.top.len() && self.top[i].distance <= motif.distance {
            i += 1;
        }
        self.top.insert(i, motif);

        debug_assert!(self.top.is_sorted());

        self.cleanup();
        assert!(self.top.len() <= (self.k + 1) * (self.k + 1));
        self.should_update = true;
    }

    fn cleanup(&mut self) {
        let k = self.k;
        let mut i = 0;
        while i < self.top.len() {
            if overlap_count(&self.top[i], &self.top[..i], self.exclusion_zone) >= k {
                self.top.remove(i);
            } else {
                i += 1;
            }
        }
    }

    fn update_non_overlapping(&mut self) {
        if !self.should_update {
            return;
        }
        self.current_non_overlapping.clear();
        for i in 0..self.top.len() {
            if !self.top[i].overlaps(self.current_non_overlapping.as_slice(), self.exclusion_zone) {
                self.current_non_overlapping.push(self.top[i]);
            }
        }
        self.should_update = false;
    }

    //// This function is used to access the k-th motif, if
    //// we already found it, even if not confirmed yet
    pub fn k_th(&mut self) -> Option<Motif> {
        self.update_non_overlapping();
        let current = &self.current_non_overlapping;
        if current.len() == self.k {
            current.last().map(|mot| *mot)
        } else {
            None
        }
    }

    pub fn first_not_confirmed(&mut self) -> Option<Motif> {
        self.update_non_overlapping();
        self.current_non_overlapping
            .iter()
            .filter(|m| m.elapsed.is_none())
            .next()
            .map(|m| *m)
    }

    pub fn last_confirmed(&mut self) -> Option<Motif> {
        self.update_non_overlapping();
        self.current_non_overlapping
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

    pub fn to_vec(&mut self) -> Vec<Motif> {
        self.update_non_overlapping();
        self.current_non_overlapping.clone()
    }

    pub fn add_all(&mut self, other: &mut TopK) {
        for m in other.top.drain(..) {
            self.insert(m);
        }
    }
}

//// ## Motif finding algorithm

//// At last, this is the algorithm to find the motifs.
//// It takes, as parameters:
////
////  - the time series windowed with windows of length `w`
////  - the number of top motifs we want to find
////  - the memory we allow the algorithm to use
////  - the probability `delta`  of making an error
////  - the seed of the random number generator
////
//// These are the general steps of the algorithm, which are then detailed below:
////
////  1. Compute the tensored hash pools for all the subsequences, taking into account
////     the allowed memory when setting the number of repetitions
////  2. Initialize the hash matrix, with one column per repetition
////  3. Then, starting from depth `K`, iterate through all repetitions, updating the
////     estimated nearest neighbor of each subsequence
////     - Whenever a nearest neighbor is updated, we try to insert it in the top-k set
////     - If the k-th motif in the top-k data structure is at distance such that the
////       probability of not having seen a closer pair is small enough, we stop.
////     - Otherwise, if we exhaust all available repetitions, it means that hash values
////       of the current length are too selective. Therefore, we relax them by considering
////       hash values one value shorter, repeating from the beginning of point 3.
pub fn motifs(
    ts: Arc<WindowedTimeseries>,
    topk: usize,
    repetitions: usize,
    delta: f64,
    seed: u64,
) -> Vec<Motif> {
    let exclusion_zone = ts.w;
    let mut enumerator = MotifsEnumerator::<PairMotifState>::new(
        ts,
        topk,
        repetitions,
        delta,
        || PairMotifState::new(topk, exclusion_zone),
        seed,
        true,
    );
    let mut res = Vec::new();
    while let Some(m) = enumerator.next() {
        res.push(m);
    }
    res.sort_unstable();
    res.truncate(topk);
    assert_eq!(res.len(), topk);
    eprintln!("{:?}", enumerator.exec_stats);
    res
}

#[derive(Debug, Default, Copy, Clone)]
pub struct Stats {
    pub distances: u64,
    pub candidates: u64,
}
impl Stats {
    #[inline]
    pub fn inc_dists(&mut self) {
        self.distances += 1;
    }
    #[inline]
    pub fn inc_cands(&mut self) {
        self.candidates += 1;
    }
    pub fn merge(&self, other: &Self) -> Self {
        Self {
            distances: self.distances + other.distances,
            candidates: self.candidates + other.candidates,
        }
    }
}

pub trait State: std::fmt::Debug {
    type Output: Sync + Send + std::fmt::Debug + Clone + Ord;
    /// update the state with the given pair of subsequences
    fn update(&self, ts: &WindowedTimeseries, a: usize, b: usize);
    /// are we done with the computation?
    fn is_done(&mut self) -> bool;
    /// emit a bunch of output, if any
    fn emit<F: Fn(f64) -> bool>(
        &mut self,
        ts: &WindowedTimeseries,
        predicate: F,
    ) -> Vec<Self::Output>;
    /// the next distance we are interested in looking at
    fn next_distance(&mut self) -> Option<f64>;
}

#[derive(Debug)]
pub struct PairMotifState {
    k: usize,
    exclusion_zone: usize,
    start: Instant,
    tl_topk: ThreadLocal<RefCell<TopK>>,
    topk: TopK,
}

impl PairMotifState {
    pub fn new(k: usize, exclusion_zone: usize) -> Self {
        Self {
            k,
            exclusion_zone,
            start: Instant::now(),
            tl_topk: Default::default(),
            topk: TopK::new(k, exclusion_zone),
        }
    }

    /// merge the thread-local topk queues
    fn merge_threads(&mut self) {
        let tmp = self.tl_topk.iter_mut().reduce(|a, b| {
            a.borrow_mut().add_all(&mut b.borrow_mut());
            a
        });
        if let Some(tmp) = tmp {
            self.topk.add_all(&mut tmp.borrow_mut());
        }
    }
}

impl State for PairMotifState {
    type Output = Motif;

    fn update(&self, ts: &WindowedTimeseries, a_idx: usize, b_idx: usize) {
        let d = zeucl(ts, a_idx, b_idx);
        if d.is_finite() {
            let m = Motif {
                idx_a: a_idx as usize,
                idx_b: b_idx as usize,
                distance: d,
                elapsed: None,
                discovered: self.start.elapsed(),
            };
            self.tl_topk
                .get_or(|| RefCell::new(TopK::new(self.k, self.exclusion_zone)))
                .borrow_mut()
                .insert(m);
        }
    }

    fn is_done(&mut self) -> bool {
        self.merge_threads();
        self.topk.num_confirmed() == self.k
    }

    fn emit<F: Fn(f64) -> bool>(&mut self, _ts: &WindowedTimeseries, predicate: F) -> Vec<Motif> {
        self.merge_threads();
        let mut ret: Vec<Motif> = Vec::new();
        let elapsed = self.start.elapsed();
        self.topk.for_each(|m| {
            if m.elapsed.is_none() {
                if predicate(m.distance) {
                    m.elapsed.replace(elapsed);
                    ret.push(*m);
                }
            }
        });
        ret
    }

    fn next_distance(&mut self) -> Option<f64> {
        self.topk.first_not_confirmed().map(|m| m.distance)
    }
}

pub struct MotifsEnumerator<S: State> {
    ts: Arc<WindowedTimeseries>,
    pub max_k: usize,
    state: S,
    to_return: BinaryHeap<Reverse<S::Output>>,
    /// used to cache the motifs already discovered so that we can use the enumerator as a collection
    returned: Vec<S::Output>,
    repetitions: usize,
    delta: f64,
    exclusion_zone: usize,
    pools: Arc<HashCollection>,
    buffers: ColumnBuffers,
    /// the current repetition
    rep: usize,
    /// the current depth
    depth: usize,
    /// the previous depth
    previous_depth: Option<usize>,
    /// the progress bar
    pbar: Option<ProgressBar>,
    /// the execution statistics
    exec_stats: Stats,
}

impl<S: State + Send + Sync> MotifsEnumerator<S> {
    pub fn new<F: FnOnce() -> S>(
        ts: Arc<WindowedTimeseries>,
        max_k: usize,
        repetitions: usize,
        delta: f64,
        init: F,
        seed: u64,
        show_progress: bool,
    ) -> Self {
        let start = Instant::now();
        let exclusion_zone = ts.w;
        let fft_data = FFTData::new(&ts);

        let hasher_width = Hasher::compute_width(&ts);
        debug!("hasher_width" = hasher_width; "computed hasher width");

        let hasher = Hasher::new(ts.w, repetitions, hasher_width, seed);
        let pools = HashCollection::from_ts(&ts, &fft_data, hasher);
        let pools = Arc::new(pools);
        info!("Computed hash values in {:?}", start.elapsed());
        drop(fft_data);

        let pbar = if show_progress {
            Some(Self::build_progress_bar(K, repetitions))
        } else {
            None
        };

        Self {
            ts,
            max_k,
            state: init(),
            to_return: BinaryHeap::new(),
            returned: Vec::new(),
            repetitions,
            delta,
            exclusion_zone,
            pools,
            buffers: ColumnBuffers::default(),
            rep: 0,
            depth: K,
            previous_depth: None,
            pbar,
            exec_stats: Stats::default(),
        }
    }

    fn build_progress_bar(depth: usize, repetitions: usize) -> ProgressBar {
        let pbar = ProgressBar::new(repetitions as u64);
        pbar.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {msg} {bar:40.cyan/blue} {pos:>7}/{len:7}")
                .unwrap(),
        );
        pbar.set_message(format!("depth {}", depth));
        pbar
    }

    pub fn get_ts(&self) -> Arc<WindowedTimeseries> {
        Arc::clone(&self.ts)
    }

    //// Find the level for which the given distance has a good probability of being
    //// found withing the allowed number of repetitions
    fn level_for_distance(&self, d: f64, mut prefix: usize) -> usize {
        let orig_prefix = prefix;
        while prefix > 0 {
            for rep in 0..self.repetitions {
                if self
                    .pools
                    .failure_probability_independent(d.into(), rep, prefix, None, None)
                    < self.delta
                {
                    return prefix;
                }
            }
            prefix -= 1;
        }
        let ret = orig_prefix - 1;
        assert!(ret > 0);
        ret
    }

    /// Return the next motif, or `None` if we already returned `max_k` motifs
    pub fn next(&mut self) -> Option<S::Output> {
        // First, try to empty the buffer of motifs to return, if any
        if let Some(motif) = self.to_return.pop() {
            self.returned.push(motif.0.clone());
            return Some(motif.0);
        }

        // check we already returned all we could
        if self.state.is_done() {
            self.pbar.as_ref().map(|pbar| pbar.finish_and_clear());
            return None;
        }

        // repeat until we are able to buffer some motifs
        while self.to_return.is_empty() {
            assert!(self.depth > 0);
            assert!(self.rep < self.repetitions);

            // Set up buckets for the current repetition
            self.pools.group_subsequences(
                self.depth,
                self.rep.into(),
                self.exclusion_zone,
                &mut self.buffers,
                true,
            );
            let n_buckets = self.buffers.buckets.len();
            let chunk_size = std::cmp::max(1, n_buckets / (4 * rayon::current_num_threads()));

            // counters for profiling
            let mut stats = ThreadLocal::new();

            (0..n_buckets / chunk_size)
                .into_par_iter()
                .for_each(|chunk_i| {
                    let mut tl_stats = stats.get_or(|| RefCell::new(Stats::default())).borrow_mut();

                    for i in (chunk_i * chunk_size)..((chunk_i + 1) * chunk_size) {
                        let bucket = &self.buffers.hashes[self.buffers.buckets[i].clone()];

                        for (_, a_idx) in bucket.iter() {
                            let a_idx = *a_idx as usize;
                            // let a_already_checked = rep_bounds[a_idx].clone();
                            // let a_hash_idx = hash_range.start + a_offset;
                            for (_, b_idx) in bucket.iter() {
                                let b_idx = *b_idx as usize;
                                //// Here we handle trivial matches: we don't consider a pair if the difference between
                                //// the subsequence indexes is smaller than the exclusion zone, which is set to `w/4`.
                                if a_idx + self.exclusion_zone < b_idx {
                                    tl_stats.inc_cands();

                                    //// We only process the pair if this is the first repetition in which
                                    //// they collide. We get this information from the pool of bits
                                    //// from which hash values for all repetitions are extracted.
                                    if let Some(first_colliding_repetition) =
                                        self.pools.first_collision(a_idx, b_idx, self.depth)
                                    {
                                        //// This is the first collision in this iteration, _and_ the pair didn't collide
                                        //// at a deeper level.
                                        if first_colliding_repetition == self.rep
                                            && self
                                                .previous_depth
                                                .map(|d| {
                                                    self.pools
                                                        .first_collision(a_idx, b_idx, d)
                                                        .is_none()
                                                })
                                                .unwrap_or(true)
                                        {
                                            tl_stats.inc_dists();
                                            self.state.update(
                                                &self.ts,
                                                a_idx as usize,
                                                b_idx as usize,
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                });

            // Add to the output all the new top pairs that have been found
            self.exec_stats = stats
                .iter_mut()
                .map(|s| s.take())
                .reduce(|a, b| a.merge(&b))
                .unwrap_or_default()
                .merge(&self.exec_stats);

            // Confirm the pairs that can be confirmed in this iteration
            let depth = self.depth;
            let rep = self.rep;
            let delta = self.delta;
            let pools = &self.pools;
            let mut buf = self.state.emit(&self.ts, |d| {
                pools.failure_probability_independent(d.into(), rep, depth, None, None) <= delta
            });
            self.to_return.extend(buf.drain(..).map(|m| Reverse(m)));

            self.pbar.as_ref().map(|pbar| pbar.inc(1));

            // set up next repetition
            self.rep += 1;
            if self.rep >= self.repetitions {
                self.rep = 0;
                self.previous_depth.replace(self.depth);
                if let Some(distance) = self.state.next_distance() {
                    if let Some(pbar) = &self.pbar {
                        pbar.println(format!("next distance {:?}", distance));
                        pbar.println(format!("{:?}", self.state));
                    }
                    let new_depth = self.level_for_distance(distance, self.depth);
                    if new_depth == depth {
                        self.depth -= 1;
                    } else {
                        self.depth = new_depth;
                    }
                } else {
                    self.depth -= 1;
                }
                if self.pbar.is_some() {
                    self.pbar = Some(Self::build_progress_bar(self.depth, self.repetitions));
                }
                assert!(self
                    .previous_depth
                    .map(|prev| self.depth < prev)
                    .unwrap_or(true));
            }
        }

        // return the found motif with the smallest distance
        let m = self.to_return.pop().map(|m| m.0);
        if let Some(m) = m.clone() {
            self.returned.push(m);
        }
        m
    }

    /// Gets the motif at the given rank, possibly computing it if not already discovered.
    pub fn get_ranked(&mut self, rank: usize) -> &S::Output {
        if rank >= self.max_k {
            panic!("Index out of bounds");
        }
        while self.returned.len() <= rank {
            self.next();
        }
        &self.returned[rank]
    }
}

impl Iterator for MotifsEnumerator<PairMotifState> {
    type Item = Motif;

    fn next(&mut self) -> Option<Self::Item> {
        self.next()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{load::loadts, timeseries::WindowedTimeseries};

    fn run_motif_test(
        ts: Arc<WindowedTimeseries>,
        k: usize,
        repetitions: usize,
        seed: u64,
        ground_truth: Option<Vec<(usize, usize, f64)>>,
    ) {
        let failure_probability = 0.01;
        let exclusion_zone = ts.w;
        let ground_truth: Vec<(usize, usize, f64)> = ground_truth.unwrap_or_else(|| {
            eprintln!(
                "Running brute force algorithm on {} subsequences",
                ts.num_subsequences()
            );
            brute_force_motifs(&ts, k, exclusion_zone)
                .into_iter()
                .map(|m| (m.idx_a, m.idx_b, m.distance))
                .collect()
        });
        dbg!(&ground_truth);
        let motifs = motifs(ts, k, repetitions, failure_probability, seed);
        assert_eq!(motifs.len(), k);
        let mut cnt = 0;
        for m in &motifs {
            println!("{:?}", m);
            if m.distance <= ground_truth.last().unwrap().2 + 0.0000001 {
                cnt += 1;
            }
        }
        assert_eq!(cnt, k);
    }

    #[test]
    #[ignore]
    fn test_ecg() {
        let ts: Vec<f64> = loadts("data/ECG.csv.gz", Some(10000)).unwrap();
        let ts = Arc::new(WindowedTimeseries::new(ts, 100, false));
        run_motif_test(ts, 10, 512, 12345, None);

        let ts: Vec<f64> = loadts("data/ECG.csv.gz", Some(10000)).unwrap();
        let ts = Arc::new(WindowedTimeseries::new(ts, 200, false));
        run_motif_test(ts, 10, 512, 12345, None);
    }

    #[test]
    #[ignore]
    fn test_astro() {
        let ts: Vec<f64> = loadts("data/ASTRO.csv.gz", Some(10000)).unwrap();
        let ts = Arc::new(WindowedTimeseries::new(ts, 20, false));
        run_motif_test(ts, 1, 512, 12345, None);
    }
}
