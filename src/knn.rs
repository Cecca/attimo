use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::{cell::RefCell, collections::BTreeMap, ops::Range, sync::Arc, time::Instant};
use thread_local::ThreadLocal;

use crate::{
    distance::zeucl,
    lsh::{ColumnBuffers, HashCollection, HashValue, Hasher},
    timeseries::{overlap_count, FFTData, Overlaps, WindowedTimeseries},
};

#[derive(PartialEq, PartialOrd, Clone, Copy, Debug)]
// TODO: turn this struct into a `Distance` struct with [From] and [Into] impls for `f64`
pub struct OrdF64(pub f64);
impl Eq for OrdF64 {}
impl Ord for OrdF64 {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap()
    }
}

impl Overlaps<(OrdF64, usize)> for (OrdF64, usize) {
    fn overlaps(&self, other: (OrdF64, usize), exclusion_zone: usize) -> bool {
        self.1.overlaps(other.1, exclusion_zone)
    }
}

/// A triple where the third element denotes whether the corresponding tuple is "active"
/// overlaps with another one only if the other one is active and their indices overlap
impl Overlaps<(OrdF64, usize, bool)> for (OrdF64, usize, bool) {
    fn overlaps(&self, other: (OrdF64, usize, bool), exclusion_zone: usize) -> bool {
        other.2 && self.1.overlaps(other.1, exclusion_zone)
    }
}

pub struct SupportBuffers {
    indices: Vec<usize>,
    distances: Vec<f64>,
    buf: Vec<f64>,
}
impl SupportBuffers {
    pub fn new(ts: &WindowedTimeseries) -> Self {
        let n = ts.num_subsequences();
        let mut indices = Vec::new();
        indices.resize(n, 0usize);
        let mut distances = Vec::new();
        distances.resize(n, 0.0f64);
        let mut buf = Vec::new();
        buf.resize(ts.w, 0.0f64);
        Self {
            indices,
            distances,
            buf,
        }
    }
}

struct EvolvingNeighborhood {
    subsequence: usize,
    max_k: usize,
    /// When true, requires to update the flags in the [neighbors] vector.
    dirty: bool,
    /// For each neighbor we consider its distance, its index and whether it is currently
    /// part of the k-nearest neighbors, i.e. if it overlaps with any othe preceding
    /// selected neighbors
    neighbors: Vec<(OrdF64, usize, bool)>,
}
impl EvolvingNeighborhood {
    fn new(subsequence: usize, max_k: usize) -> Self {
        Self {
            subsequence,
            max_k,
            dirty: false,
            neighbors: Vec::with_capacity(max_k),
        }
    }
    fn update(&mut self, dist: OrdF64, neigh: usize) {
        self.dirty = true;
        let tuple = (dist, neigh, false);
        let mut i = 0;
        while i < self.neighbors.len() && tuple.0 > self.neighbors[i].0 {
            i += 1;
        }
        self.neighbors.insert(i, tuple);
    }
    fn clean(&mut self, exclusion_zone: usize) {
        if !self.dirty {
            return;
        }
        for tup in self.neighbors.iter_mut() {
            tup.2 = false;
        }
        let mut i = 0;
        while i < self.neighbors.len() {
            if !self.neighbors[i].overlaps(&self.neighbors[..i], exclusion_zone) {
                self.neighbors[i].2 = true;
            }
            i += 1;
        }
        self.dirty = false;
    }
    /// Returns an upper bound to the extent of this neighborhood
    fn extent(&mut self, k: usize, exclusion_zone: usize) -> OrdF64 {
        self.clean(exclusion_zone);
        self.neighbors
            .iter()
            .filter_map(|(d, _, is_neighbor)| {
                if *is_neighbor {
                    Some(OrdF64(2.0 * d.0))
                } else {
                    None
                }
            })
            .nth(k)
            .unwrap_or(OrdF64(f64::INFINITY))
    }
    /// Counts how many neighbors have a distance _strictly_ larger than the given distance
    fn count_larger_than(&mut self, k: usize, exclusion_zone: usize, d: OrdF64) -> usize {
        self.clean(exclusion_zone);
        self.neighbors
            .iter()
            .filter_map(
                |(d, _, is_neighbor)| {
                    if *is_neighbor {
                        Some(2.0 * d.0)
                    } else {
                        None
                    }
                },
            )
            .take(k)
            .filter(|nn_dist| *nn_dist > d.0)
            .count()
    }
}

pub enum SubsequenceNeighborhood {
    /// This subsequence neighborhood has been "brute forced", i.e.
    /// it has been computed exactly. It ignores any proposed update.
    Exact {
        subsequence: usize,
        extents: Vec<OrdF64>,
        ids: Vec<usize>,
    },
    /// This neighborhood can be updated by adding new points that might possibly
    /// replace the current nearest neighbors
    Evolving { neighborhood: EvolvingNeighborhood },
    /// This neighborhood has been discarded as it has been deemed not useful for computing the
    /// final result. It ignores any proposed update, and does not consume memory for the
    /// neighbors.
    Discarded { subsequence: usize },
}
impl SubsequenceNeighborhood {
    pub fn evolving(max_k: usize, subsequence: usize) -> Self {
        Self::Evolving {
            neighborhood: EvolvingNeighborhood::new(max_k, subsequence),
        }
    }
    pub fn exact(
        k: usize,
        subsequence: usize,
        ts: &WindowedTimeseries,
        fft_data: &FFTData,
        exclusion_zone: usize,
        buffers: &mut SupportBuffers,
    ) -> Self {
        let indices = &mut buffers.indices;
        let distances = &mut buffers.distances;
        let buf = &mut buffers.buf;
        assert_eq!(indices.len(), ts.num_subsequences());
        assert_eq!(distances.len(), ts.num_subsequences());
        assert_eq!(buf.len(), ts.w);

        // Compute the distance profile using the MASS algorithm
        ts.distance_profile(&fft_data, subsequence, distances, buf);

        // Reset the indices of the subsequences
        for i in 0..ts.num_subsequences() {
            indices[i] = i;
        }
        // Find the likely candidates by a (partial) indirect sort of
        // the indices by increasing distance.
        let n_candidates = (k * exclusion_zone).min(ts.num_subsequences());
        indices.select_nth_unstable_by_key(n_candidates, |j| OrdF64(distances[*j]));

        // Sort the candidate indices by increasing distance (the previous step)
        // only partitioned the indices in two groups with the guarantee that the first
        // `n_candidates` indices are the ones at shortest distance from the `from` point,
        // but they are not guaranteed to be sorted
        let indices = &mut indices[..n_candidates];
        indices.sort_unstable_by_key(|j| OrdF64(distances[*j]));

        // Pick the k-neighborhood skipping overlapping subsequences
        let mut ids = Vec::new();
        ids.push(subsequence);
        let mut j = 1;
        while ids.len() < k && j < indices.len() {
            // find the non-overlapping subsequences
            let jj = indices[j];
            let mut overlaps = false;
            for h in 0..ids.len() {
                let hh = ids[h];
                if jj.max(hh) - jj.min(hh) < exclusion_zone {
                    overlaps = true;
                    break;
                }
            }
            if !overlaps {
                ids.push(jj);
            }
            j += 1;
        }
        assert_eq!(ids.len(), k);

        let mut extents = Vec::new();
        extents.push(OrdF64(0.0));
        for i in 0..k {
            let mut extent = 0.0f64;
            for j in (i + 1)..k {
                let d = zeucl(ts, ids[i], ids[j]);
                extent = extent.max(d);
            }
            extents.push(OrdF64(extent));
        }

        Self::Exact {
            subsequence,
            ids,
            extents,
        }
    }

    pub fn discard(&mut self) {
        // Only discard the evolving one.
        match self {
            Self::Evolving { neighborhood } => {
                *self = Self::Discarded {
                    subsequence: neighborhood.subsequence,
                }
            }

            Self::Exact {
                subsequence: _,
                extents: _,
                ids: _,
            } => panic!("should not try to discard an exact neighborhood"),
            Self::Discarded { subsequence: _ } => (), // do nothing
        }
    }
    pub fn brute_force(
        &mut self,
        ts: &WindowedTimeseries,
        fft_data: &FFTData,
        exclusion_zone: usize,
        buffers: &mut SupportBuffers,
    ) {
        match self {
            Self::Evolving { neighborhood } => {
                *self = Self::exact(
                    neighborhood.max_k,
                    neighborhood.subsequence,
                    ts,
                    fft_data,
                    exclusion_zone,
                    buffers,
                )
            }
            Self::Exact {
                subsequence: _,
                extents: _,
                ids: _,
            } => (), // do nothing
            Self::Discarded { subsequence: _ } => {
                panic!("should not try to brute force a discarded subsequence")
            }
        }
    }
    pub fn extent(&mut self, k: usize, exclusion_zone: usize) -> OrdF64 {
        match self {
            Self::Evolving { neighborhood } => neighborhood.extent(k, exclusion_zone),
            Self::Exact {
                subsequence: _,
                extents,
                ids: _,
            } => extents[k],
            Self::Discarded { subsequence: _ } => OrdF64(std::f64::INFINITY),
        }
    }
    pub fn update(&mut self, dist: OrdF64, neigh: usize) {
        match self {
            Self::Evolving { neighborhood } => {
                neighborhood.update(dist, neigh);
            }
            _ => (),
        }
    }
    /// Compute the probability that this subsequence neighborhood
    /// fails based on how many neighbors would need to be changed in order to
    /// possibly bring its extent above the given lower bound
    pub fn failure_probability(
        &mut self,
        k: usize,
        extent_lower_bound: OrdF64,
        lower_bound_fp: f64,
        exclusion_zone: usize,
    ) -> f64 {
        match self {
            Self::Evolving { neighborhood } => {
                let h = neighborhood.count_larger_than(k, exclusion_zone, extent_lower_bound);
                assert!(h <= k);
                lower_bound_fp.powi((k - h) as i32)
            }
            Self::Exact {
                subsequence: _,
                extents: _,
                ids: _,
            } => 0.0,
            Self::Discarded { subsequence: _ } => 1.0,
        }
    }
}

#[derive(Clone, Ord, Eq, PartialEq, PartialOrd)]
pub struct SubsequenceNeighborhoodOld {
    pub id: usize,
    k: usize,
    exclusion_zone: usize,
    neighbors: Vec<(OrdF64, usize)>,
    current_non_overlapping: Vec<(OrdF64, usize)>,
    should_update: bool,
}
impl std::fmt::Debug for SubsequenceNeighborhoodOld {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SubsequenceNeighborhood {{ id: {}, k: {}, n_neighbors: {} n_non_overlapping: {}",
            self.id,
            self.k,
            self.neighbors.len(),
            self.current_non_overlapping.len()
        )?;
        write!(f, " neighbors ")?;
        f.debug_list().entries(self.neighbors.iter()).finish()?;
        write!(f, " non_overlapping ")?;
        f.debug_list()
            .entries(self.current_non_overlapping.iter())
            .finish()?;
        write!(f, "}}")
    }
}
impl SubsequenceNeighborhoodOld {
    pub fn new(id: usize, k: usize, exclusion_zone: usize) -> Self {
        Self {
            id,
            k,
            exclusion_zone,
            neighbors: Vec::with_capacity(k + 1),
            current_non_overlapping: Vec::new(),
            should_update: false,
        }
    }
    pub fn len(&self) -> usize {
        self.neighbors.len()
    }
    fn cleanup(&mut self) {
        let k = self.k;
        let mut i = 0;
        while i < self.neighbors.len() {
            if overlap_count(
                &self.neighbors[i],
                &self.neighbors[..i],
                self.exclusion_zone,
            ) >= k
            {
                self.neighbors.remove(i);
            } else {
                i += 1;
            }
        }
        assert!(self.neighbors.is_sorted());
    }
    fn update_non_overlapping(&mut self) {
        if !self.should_update {
            return;
        }
        self.current_non_overlapping.clear();
        for i in 0..self.neighbors.len() {
            if !self.neighbors[i]
                .overlaps(self.current_non_overlapping.as_slice(), self.exclusion_zone)
            {
                self.current_non_overlapping.push(self.neighbors[i]);
            }
        }
        assert!(self.current_non_overlapping.is_sorted());
        self.should_update = false;
    }
    pub fn update(&mut self, dist: f64, neighbor: usize) {
        let dist = OrdF64(dist);
        let pair = (dist, neighbor);

        let mut i = 0;
        while i < self.neighbors.len() && pair > self.neighbors[i] {
            i += 1;
        }

        // we insert the neighbor in the correct position
        self.neighbors.insert(i, (dist, neighbor));
        assert!(
            self.neighbors.is_sorted(),
            "neighbors are not sorted: {:?}",
            self.neighbors
        );

        // remove neighbors that are overlapped by more
        // than k higher-ranked subsequences
        self.cleanup();
        assert!(self.neighbors.len() <= (self.k + 1) * (self.k + 1));
        self.should_update = true;
    }
    pub fn merge(&mut self, other: &Self) {
        assert_eq!(self.id, other.id);
        for (d, neigh) in &other.neighbors {
            self.update(d.0, *neigh);
        }
    }
    pub fn farthest_distance(&mut self) -> Option<f64> {
        self.update_non_overlapping();
        self.current_non_overlapping
            .iter()
            .nth(self.k)
            .map(|pair| (pair.0).0)
    }
    pub fn extent(&mut self, ts: &WindowedTimeseries) -> Option<f64> {
        let ids = self.knn();
        if ids.len() < self.k {
            return None;
        }
        let mut extent = (self.current_non_overlapping[self.k - 1].0).0;
        for i in 0..ids.len() {
            let ii = ids[i];
            for j in (i + 1)..ids.len() {
                let jj = ids[j];
                let d = zeucl(ts, ii, jj);
                extent = extent.max(d);
            }
        }
        // We verify that the triangle inequality holds.
        assert!(extent <= 2.0 * (self.current_non_overlapping[self.k - 1].0).0);
        Some(extent)
    }
    pub fn knn(&mut self) -> Vec<usize> {
        self.update_non_overlapping();
        self.current_non_overlapping
            .iter()
            .take(self.k)
            .map(|pair| pair.1)
            .collect()
    }
    pub fn to_knn(&mut self) -> Knn {
        self.update_non_overlapping();
        let (neighbors, distances): (Vec<usize>, Vec<f64>) = self
            .current_non_overlapping
            .iter()
            .take(self.k)
            .map(|pair| (pair.1, (pair.0).0))
            .unzip();
        Knn {
            id: self.id,
            neighbors,
            distances,
        }
    }
}

struct KnnState {
    k: usize,
    exclusion_zone: usize,
    done: Vec<bool>,
    tl_neighborhoods: ThreadLocal<RefCell<BTreeMap<usize, SubsequenceNeighborhoodOld>>>,
    neighborhoods: BTreeMap<usize, SubsequenceNeighborhoodOld>,
}

impl KnnState {
    pub fn new(k: usize, exclusion_zone: usize, n: usize) -> Self {
        assert!(k >= 2);
        Self {
            k,
            exclusion_zone,
            done: vec![false; n],
            tl_neighborhoods: Default::default(),
            neighborhoods: Default::default(),
        }
    }

    fn merge_threads(&mut self) {
        let k = self.k;
        let exclusion_zone = self.exclusion_zone;
        for tl_neighs in self.tl_neighborhoods.iter_mut() {
            for (id, neighs) in tl_neighs.borrow_mut().iter() {
                if !self.done[*id] {
                    self.neighborhoods
                        .entry(*id)
                        .or_insert_with(|| SubsequenceNeighborhoodOld::new(*id, k, exclusion_zone))
                        .merge(neighs);
                }
            }
            tl_neighs.borrow_mut().clear();
        }
    }

    fn next_distance(&mut self) -> Option<f64> {
        let k = self.k;
        self.neighborhoods
            .iter_mut()
            .filter_map(|(_, neighborhood)| neighborhood.farthest_distance().map(|d| OrdF64(d)))
            .min()
            .map(|d| d.0)
    }
}
impl std::fmt::Debug for KnnState {
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
            "k-nn iterator: nearest neighbor entries: {} max neighbors: {}",
            nn_entries, nn_k
        )
    }
}
impl KnnState {
    fn update(&self, ts: &WindowedTimeseries, a: usize, b: usize) {
        if self.done[a] && self.done[b] {
            return;
        }
        let d = zeucl(ts, a, b);
        if !self.done[a] {
            self.tl_neighborhoods
                .get_or_default()
                .borrow_mut()
                .entry(a)
                .or_insert_with(|| SubsequenceNeighborhoodOld::new(a, self.k, self.exclusion_zone))
                .update(d, b);
        }
        if !self.done[b] {
            self.tl_neighborhoods
                .get_or_default()
                .borrow_mut()
                .entry(b)
                .or_insert_with(|| SubsequenceNeighborhoodOld::new(b, self.k, self.exclusion_zone))
                .update(d, a);
        }
    }
    fn is_done(&mut self) -> bool {
        self.merge_threads();
        self.done.iter().all(|b| *b)
    }
    fn emit<F: Fn(f64) -> bool>(&mut self, predicate: F) -> Vec<Knn> {
        self.merge_threads();

        let k = self.k;
        let mut res: Vec<Knn> = Vec::new();
        for (id, neighborhood) in self.neighborhoods.iter_mut() {
            if let Some(d) = neighborhood.farthest_distance() {
                if !self.done[*id] && predicate(d) {
                    res.push(neighborhood.to_knn());
                }
            }
        }

        for knn in res.iter() {
            self.done[knn.id] = true;
            self.neighborhoods.remove(&knn.id).unwrap();
        }

        res
    }
}

#[derive(PartialEq, PartialOrd)]
pub struct Knn {
    pub id: usize,
    pub neighbors: Vec<usize>,
    pub distances: Vec<f64>,
}
impl Eq for Knn {}
impl Ord for Knn {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distances
            .last()
            .unwrap()
            .partial_cmp(&other.distances.last().unwrap())
            .unwrap()
    }
}

pub struct KnnIter {
    ts: Arc<WindowedTimeseries>,
    state: KnnState,
    buffer: Vec<Knn>,
    repetitions: usize,
    delta: f64,
    exclusion_zone: usize,
    hasher: Arc<Hasher>,
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
}
impl KnnIter {
    pub fn new(
        ts: Arc<WindowedTimeseries>,
        k: usize,
        w: usize,
        repetitions: usize,
        delta: f64,
        seed: u64,
        show_progress: bool,
    ) -> Self {
        let start = Instant::now();
        let exclusion_zone = ts.w;
        let fft_data = FFTData::new(&ts);

        let hasher_width = Hasher::estimate_width(&ts, &fft_data, 10, None, seed);

        let hasher = Arc::new(Hasher::new(ts.w, repetitions, hasher_width, seed));
        let pools = HashCollection::from_ts(&ts, &fft_data, Arc::clone(&hasher));
        let pools = Arc::new(pools);
        eprintln!("Computed hash values in {:?}", start.elapsed());
        drop(fft_data);

        let pbar = if show_progress {
            Some(Self::build_progress_bar(crate::lsh::K, repetitions))
        } else {
            None
        };

        let state = KnnState::new(k, w, ts.num_subsequences());

        Self {
            ts,
            state,
            buffer: Vec::new(),
            repetitions,
            delta,
            exclusion_zone,
            hasher,
            pools,
            buffers: ColumnBuffers::default(),
            rep: 0,
            depth: crate::lsh::K,
            previous_depth: None,
            pbar,
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
}
impl Iterator for KnnIter {
    type Item = Knn;

    fn next(&mut self) -> Option<Self::Item> {
        // check we already returned all we could
        if self.state.is_done() {
            self.pbar.as_ref().map(|pbar| pbar.finish_and_clear());
            return None;
        }

        // repeat until we are able to buffer some motifs
        while self.buffer.is_empty() {
            assert!(self.depth > 0);
            assert!(self.rep < self.repetitions);

            // Set up buckets for the current repetition
            self.pools.group_subsequences(
                self.depth,
                self.rep,
                self.exclusion_zone,
                &mut self.buffers,
            );
            let n_buckets = self.buffers.buckets.len();
            let chunk_size = std::cmp::max(1, n_buckets / (4 * rayon::current_num_threads()));

            (0..n_buckets / chunk_size)
                .into_par_iter()
                .for_each(|chunk_i| {
                    for i in (chunk_i * chunk_size)..((chunk_i + 1) * chunk_size) {
                        let bucket = &self.buffers.buffer[self.buffers.buckets[i].clone()];

                        for (_, a_idx) in bucket.iter() {
                            let a_idx = *a_idx as usize;
                            for (_, b_idx) in bucket.iter() {
                                let b_idx = *b_idx as usize;
                                if a_idx + self.exclusion_zone < b_idx {
                                    if let Some(first_colliding_repetition) =
                                        self.pools.first_collision(a_idx, b_idx, self.depth)
                                    {
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

            // Confirm the knn that can be confirmed in this iteration
            let depth = self.depth;
            let rep = self.rep;
            let delta = self.delta;
            let hasher = Arc::clone(&self.hasher);
            let mut buf: Vec<Knn> = self
                .state
                .emit(|d| hasher.failure_probability(d, rep, depth) <= delta);
            self.buffer.extend(buf.drain(..));
            self.buffer.sort_by(|knn1, knn2| knn1.cmp(knn2).reverse());

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

        self.buffer.pop()
    }
}
