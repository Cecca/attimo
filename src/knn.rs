use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::{cell::RefCell, collections::BTreeMap, ops::Range, sync::Arc, time::Instant};
use thread_local::ThreadLocal;

use crate::{
    distance::zeucl,
    lsh::{HashCollection, HashValue, Hasher},
    timeseries::{FFTData, Overlaps, WindowedTimeseries},
};

#[derive(PartialEq, PartialOrd, Clone, Copy, Debug)]
pub struct OrdF64(pub f64);
impl Eq for OrdF64 {}
impl Ord for OrdF64 {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap()
    }
}

#[derive(Debug, Clone, Ord, Eq, PartialEq, PartialOrd)]
pub struct SubsequenceNeighborhood {
    pub id: usize,
    k: usize,
    exclusion_zone: usize,
    pub neighbors: Vec<(OrdF64, usize)>,
}
impl SubsequenceNeighborhood {
    pub fn new(id: usize, k: usize, exclusion_zone: usize) -> Self {
        Self {
            id,
            k,
            exclusion_zone,
            neighbors: Vec::with_capacity(k + 1),
        }
    }
    pub fn len(&self) -> usize {
        self.neighbors.len()
    }
    pub fn update(&mut self, dist: f64, neighbor: usize) {
        let dist = OrdF64(dist);

        let mut i = 0;
        while i < self.neighbors.len() && dist > self.neighbors[i].0 {
            if neighbor.overlaps(self.neighbors[i].1, self.exclusion_zone) {
                // The candidate neighbor overlaps with one that is closer to
                // the root point, hence we discard it.
                return;
            }
            i += 1;
        }

        // we insert the neighbor in the correct position
        self.neighbors.insert(i, (dist, neighbor));

        // now we remove other neighbors that might be
        // overlapping with the one just inserted
        i += 1;
        while i < self.neighbors.len() {
            if neighbor.overlaps(self.neighbors[i].1, self.exclusion_zone) {
                self.neighbors.remove(i);
            } else {
                i += 1;
            }
        }

        // and now we remove any excess elements
        while self.neighbors.len() > self.k {
            self.neighbors.pop();
        }
    }
    pub fn merge(&mut self, other: &Self) {
        assert_eq!(self.id, other.id);
        for (d, neigh) in &other.neighbors {
            self.update(d.0, *neigh);
        }
    }
    pub fn farthest_up_to(&self, k: usize) -> Option<f64> {
        self.neighbors.iter().take(k).map(|pair| (pair.0).0).last()
    }
    pub fn distance_at(&self, k: usize) -> Option<f64> {
        self.neighbors.iter().nth(k).map(|pair| (pair.0).0)
    }
    pub fn extent(&self, k: usize, ts: &WindowedTimeseries) -> Option<f64> {
        let ids = self.knn(k);
        if ids.len() < k {
            return None;
        }
        let mut extent = zeucl(ts, self.id, *ids.last().unwrap());
        for i in 0..ids.len() {
            let ii = ids[i];
            for j in (i + 1)..ids.len() {
                let jj = ids[j];
                let d = zeucl(ts, ii, jj);
                extent = extent.max(d);
            }
        }
        Some(extent)
    }
    pub fn knn(&self, k: usize) -> Vec<usize> {
        self.neighbors.iter().take(k).map(|pair| pair.1).collect()
    }
    pub fn to_knn(&self, k: usize) -> Knn {
        let (neighbors, distances): (Vec<usize>, Vec<f64>) = self
            .neighbors
            .iter()
            .take(k)
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
    tl_neighborhoods: ThreadLocal<RefCell<BTreeMap<usize, SubsequenceNeighborhood>>>,
    neighborhoods: BTreeMap<usize, SubsequenceNeighborhood>,
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
                        .or_insert_with(|| SubsequenceNeighborhood::new(*id, k, exclusion_zone))
                        .merge(neighs);
                }
            }
            tl_neighs.borrow_mut().clear();
        }
    }

    fn next_distance(&self) -> Option<f64> {
        self.neighborhoods
            .iter()
            .filter_map(|(_, neighborhood)| neighborhood.distance_at(self.k).map(|d| OrdF64(d)))
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
                .or_insert_with(|| SubsequenceNeighborhood::new(a, self.k, self.exclusion_zone))
                .update(d, b);
        }
        if !self.done[b] {
            self.tl_neighborhoods
                .get_or_default()
                .borrow_mut()
                .entry(b)
                .or_insert_with(|| SubsequenceNeighborhood::new(b, self.k, self.exclusion_zone))
                .update(d, a);
        }
    }
    fn is_done(&mut self) -> bool {
        self.merge_threads();
        self.done.iter().all(|b| *b)
    }
    fn emit<F: Fn(f64) -> bool>(&mut self, predicate: F) -> Vec<Knn> {
        self.merge_threads();

        let res: Vec<Knn> = self
            .neighborhoods
            .iter()
            .filter(|(_, neighborhood)| neighborhood.distance_at(self.k).is_some())
            .into_iter()
            .filter_map(|(id, neighborhood)| {
                if let Some(d) = neighborhood.distance_at(self.k) {
                    if !self.done[*id] && predicate(d) {
                        Some(neighborhood.to_knn(self.k))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .into_iter()
            .collect();

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

        // This vector holds the (sorted) hashed subsequences, and their index
        let column_buffer = Vec::new();
        // This vector holds the boundaries between buckets. We reuse the allocations
        let buckets = Vec::new();

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
            column_buffer,
            buckets,
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
                &mut self.column_buffer,
                &mut self.buckets,
            );
            let n_buckets = self.buckets.len();
            let chunk_size = std::cmp::max(1, n_buckets / (4 * rayon::current_num_threads()));

            (0..n_buckets / chunk_size)
                .into_par_iter()
                .for_each(|chunk_i| {
                    for i in (chunk_i * chunk_size)..((chunk_i + 1) * chunk_size) {
                        let bucket = &self.column_buffer[self.buckets[i].clone()];

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
