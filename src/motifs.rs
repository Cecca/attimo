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
use rayon::prelude::*;
use slog_scope::info;
use std::cell::RefCell;
use std::cmp::Reverse;
use std::collections::BTreeMap;
use std::collections::BinaryHeap;
use std::ops::Range;
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

#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Motiflet {
    indices: Vec<usize>,
    extent: f64,
}
impl Eq for Motiflet {}
impl Ord for Motiflet {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.extent.partial_cmp(&other.extent).unwrap()
    }
}
impl Motiflet {
    pub fn support(&self) -> usize {
        self.indices.len()
    }
    pub fn extent(&self) -> f64 {
        self.extent
    }
    pub fn indices(&self) -> Vec<usize> {
        self.indices.clone()
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
    eprintln!("{:?}", enumerator.exec_stats);
    res
}

pub fn motiflets(
    ts: Arc<WindowedTimeseries>,
    support: usize,
    repetitions: usize,
    delta: f64,
    seed: u64,
) -> Vec<Motiflet> {
    let exclusion_zone = ts.w;
    let mut enumerator = MotifsEnumerator::new(
        ts,
        1,
        repetitions,
        delta,
        || KMotifletState::new(support, exclusion_zone),
        seed,
        true,
    );
    let mut res = Vec::new();
    while let Some(m) = enumerator.next() {
        eprintln!("Confirm {:?}", m);
        res.push(m);
    }
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
    fn emit<F: Fn(f64) -> bool>(&mut self, predicate: F) -> Vec<Self::Output>;
    /// the next distance we are interested in looking at
    fn next_distance(&self) -> Option<f64>;
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

    fn emit<F: Fn(f64) -> bool>(&mut self, predicate: F) -> Vec<Motif> {
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

    fn next_distance(&self) -> Option<f64> {
        self.topk.first_not_confirmed().map(|m| m.distance)
    }
}

pub struct KMotifletState {
    support: usize,
    exclusion_zone: usize,
    current_best: Option<f64>,
    done: bool,
    tl_neighborhoods: ThreadLocal<RefCell<BTreeMap<usize, SubsequenceNeighborhood>>>,
    neighborhoods: BTreeMap<usize, SubsequenceNeighborhood>,
}
impl KMotifletState {
    pub fn new(support: usize, exclusion_zone: usize) -> Self {
        assert!(support >= 2);
        Self {
            support,
            exclusion_zone,
            current_best: None,
            done: false,
            tl_neighborhoods: Default::default(),
            neighborhoods: Default::default(),
        }
    }

    fn merge_threads(&mut self) {
        for tl_neighs in self.tl_neighborhoods.iter_mut() {
            for (id, neighs) in tl_neighs.borrow_mut().iter() {
                self.neighborhoods
                    .entry(*id)
                    .or_insert_with(|| SubsequenceNeighborhood::new(*id))
                    .merge(neighs);
            }
            tl_neighs.borrow_mut().clear();
        }
    }
}
impl std::fmt::Debug for KMotifletState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let nn_entries = self
            .neighborhoods
            .iter()
            .map(|(_, v)| v.len())
            .sum::<usize>();
        let nn_k = self
            .neighborhoods
            .iter()
            .map(|(_, v)| v.len())
            .max()
            .unwrap_or(0);
        writeln!(
            f,
            "motiflets state: nearest neighbor entries: {} max neighbors: {}",
            nn_entries, nn_k
        )
    }
}
impl State for KMotifletState {
    type Output = Motiflet;
    fn update(&self, ts: &WindowedTimeseries, a: usize, b: usize) {
        let d = zeucl(ts, a, b);
        if let Some(best) = self.current_best {
            if d > best {
                // there's no point in keeping track of pairs that cannot
                // beat the current candidate
                return;
            }
        }
        self.tl_neighborhoods
            .get_or_default()
            .borrow_mut()
            .entry(a)
            .or_insert_with(|| SubsequenceNeighborhood::new(a))
            .update(d, b);
        self.tl_neighborhoods
            .get_or_default()
            .borrow_mut()
            .entry(b)
            .or_insert_with(|| SubsequenceNeighborhood::new(b))
            .update(d, a);
    }
    fn is_done(&mut self) -> bool {
        self.merge_threads();
        self.done
    }
    fn emit<F: Fn(f64) -> bool>(&mut self, predicate: F) -> Vec<Self::Output> {
        if self.done {
            return Vec::new();
        }
        self.merge_threads();

        // cleanup
        if let Some(d) = self.next_distance() {
            for neighs in self.neighborhoods.values_mut() {
                neighs.neighbors.retain(|(k, _)| k.0 <= d);
            }
        }
        self.current_best = self.next_distance();

        let res: Vec<Self::Output> = self
            .neighborhoods
            .iter()
            .filter(|(_, neighborhood)| {
                neighborhood
                    .distance_at(self.support - 1, self.exclusion_zone)
                    .is_some()
            })
            .min_by_key(|(_, neighborhood)| {
                OrdF64(
                    neighborhood
                        .distance_at(self.support - 1, self.exclusion_zone)
                        .unwrap(),
                )
            })
            .into_iter()
            .filter_map(|(k, neighborhood)| {
                // let neighborhood = entry.value();
                if let Some(d) = neighborhood.distance_at(self.support - 1, self.exclusion_zone) {
                    if predicate(d) {
                        let mut knn = neighborhood.knn(self.support - 1, self.exclusion_zone);
                        knn.push(*k);
                        Some(Motiflet {
                            indices: knn,
                            extent: d,
                        })
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .into_iter()
            .collect();
        self.done = !res.is_empty();
        res
    }
    fn next_distance(&self) -> Option<f64> {
        self.neighborhoods
            .iter()
            .filter_map(|(_, neighborhood)| {
                neighborhood
                    .distance_at(self.support - 1, self.exclusion_zone)
                    .map(|d| OrdF64(d))
            })
            .min()
            .map(|d| d.0)
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
    hasher: Arc<Hasher>,
    pools: Arc<HashCollection>,
    column_buffer: Vec<(HashValue, u32)>,
    buckets: Vec<Range<usize>>,
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

        let hasher_width = Hasher::estimate_width(&ts, &fft_data, max_k, None, seed);
        info!("Computed hasher width"; "hasher_width" => hasher_width);

        info!("hash computation"; "tag" => "phase");
        let hasher = Arc::new(Hasher::new(ts.w, repetitions, hasher_width, seed));
        let pools = HashCollection::from_ts(&ts, &fft_data, Arc::clone(&hasher));
        let pools = Arc::new(pools);
        eprintln!("Computed hash values in {:?}", start.elapsed());
        drop(fft_data);

        info!("tries exploration"; "tag" => "phase");
        // This vector holds the (sorted) hashed subsequences, and their index
        let column_buffer = Vec::new();
        // This vector holds the boundaries between buckets. We reuse the allocations
        let buckets = Vec::new();

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
            hasher,
            pools,
            column_buffer,
            buckets,
            rep: 0,
            depth: K,
            previous_depth: None,
            pbar,
            exec_stats: Stats::default(),
        }
    }

    fn build_progress_bar(depth: usize, repetitions: usize) -> ProgressBar {
        let pbar = ProgressBar::new(repetitions as u64);
        pbar.set_draw_rate(1);
        pbar.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {msg} {bar:40.cyan/blue} {pos:>7}/{len:7}"),
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
        while prefix > 0 {
            for rep in 0..self.repetitions {
                if self.hasher.failure_probability(d, rep, prefix) < self.delta {
                    return prefix;
                }
            }
            prefix -= 1;
        }
        panic!("Got to prefix of length 0!");
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
                self.rep,
                self.exclusion_zone,
                &mut self.column_buffer,
                &mut self.buckets,
            );
            let n_buckets = self.buckets.len();
            let chunk_size = std::cmp::max(1, n_buckets / (4 * rayon::current_num_threads()));

            // counters for profiling
            let mut stats = ThreadLocal::new();

            (0..n_buckets / chunk_size)
                .into_par_iter()
                .for_each(|chunk_i| {
                    let mut tl_stats = stats.get_or(|| RefCell::new(Stats::default())).borrow_mut();

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
            let hasher = Arc::clone(&self.hasher);
            let mut buf = self
                .state
                .emit(|d| hasher.failure_probability(d, rep, depth) <= delta);
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
            (1000, 1172, 6112, 2.133571168),
        ] {
            let ts: Vec<f64> = loadts("data/ECG.csv.gz", Some(10000)).unwrap();
            let ts = Arc::new(WindowedTimeseries::new(ts, w, true));

            let motif = *motifs(ts, 1, 256, 0.001, 12435).first().unwrap();
            dbg!(motif);
            assert!((motif.idx_a as isize - a as isize).abs() < w as isize);
            assert!((motif.idx_b as isize - b as isize).abs() < w as isize);
            assert!(motif.distance <= d + 0.01);
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
            let ts: Vec<f64> = loadts("data/ECG.csv.gz", None).unwrap();
            let ts = Arc::new(WindowedTimeseries::new(ts, w, true));
            // assert!((crate::distance::zeucl(&ts, a, b) - d) < 0.00000001);

            let motif = *motifs(ts, 1, 200, 0.001, 12435).first().unwrap();
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
        let ts: Vec<f64> = loadts("data/ECG.csv.gz", None).unwrap();
        let ts = Arc::new(WindowedTimeseries::new(ts, w, false));

        let motifs = motifs(ts, 10, 200, 0.01, 12435);
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
    fn test_motif_astro_top10_enumerate() {
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
        let ts: Vec<f64> = loadts("data/ASTRO.csv.gz", None).unwrap();
        let ts = Arc::new(WindowedTimeseries::new(ts, w, false));

        let motifs: Vec<Motif> = MotifsEnumerator::new(
            ts,
            10,
            800,
            0.01,
            || PairMotifState::new(10, w),
            12435,
            false,
        )
        .collect();

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
        let ts: Vec<f64> = loadts("data/ASTRO.csv.gz", None).unwrap();
        let ts = Arc::new(WindowedTimeseries::new(ts, w, false));

        let motifs = motifs(ts, 10, 800, 0.01, 12435);
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
