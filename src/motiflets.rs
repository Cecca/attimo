use crate::allocator::*;
use crate::distance::zeucl_threshold;
use crate::graph::{AdjacencyGraph, GraphStats};
use crate::index::{CostEstimator, LSHIndexStats};
use crate::observe::*;
use crate::timeseries::{overlap_count_iter, TimeseriesStats};
use crate::{
    index::LSHIndex,
    knn::*,
    timeseries::{FFTData, Overlaps, WindowedTimeseries},
};
use log::*;
use rand::prelude::*;
use rand::SeedableRng;
use rand_distr::Uniform;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use std::collections::BTreeSet;
use std::time::Duration;
use std::{sync::Arc, time::Instant};

#[allow(clippy::too_many_arguments)]
fn k_extents_bf(
    ts: &WindowedTimeseries,
    from: usize,
    fft_data: &FFTData,
    k: usize,
    exclusion_zone: usize,
    indices: &mut [usize],
    distances: &mut [f64],
    buf: &mut [f64],
) -> (Vec<Distance>, Vec<usize>) {
    // Check that the auxiliary memory buffers are correctly sized
    assert_eq!(indices.len(), ts.num_subsequences());
    assert_eq!(distances.len(), ts.num_subsequences());
    assert_eq!(buf.len(), ts.w);
    assert!(!ts.is_flat(from));

    // Compute the distance profile using the MASS algorithm
    ts.distance_profile(fft_data, from, distances, buf);

    // Reset the indices of the subsequences
    (0..ts.num_subsequences()).for_each(|i| {
        indices[i] = i;
    });
    indices.sort_unstable_by_key(|j| Distance(distances[*j]));

    // Pick the k-neighborhood skipping overlapping subsequences
    let mut ret = Vec::new();
    ret.push(from);
    let mut j = 1;
    while ret.len() < k && j < indices.len() {
        // find the non-overlapping subsequences
        let jj = indices[j];
        let mut overlaps = false;
        for h in 0..ret.len() {
            let hh = ret[h];
            if jj.overlaps(hh, exclusion_zone) {
                overlaps = true;
                break;
            }
        }
        if !overlaps {
            ret.push(jj);
        }
        j += 1;
    }
    // assert_eq!(ret.len(), k);

    (compute_extents(ts, &ret), ret)
}

/// Find the (approximate) motiflets by brute force: for each subsequence find its
/// k-nearest neighbors, compute their extents, and pick the neighborhood with the
/// smallest extent.
pub fn brute_force_motiflets(
    ts: &WindowedTimeseries,
    k: usize,
    exclusion_zone: usize,
) -> Vec<(Distance, Vec<usize>)> {
    #[cfg(not(test))]
    debug_assert!(false, "Should run only in `release mode`");
    // pre-compute the FFT for the time series
    let fft_data = FFTData::new(ts);
    let n = ts.num_subsequences();

    let average_pairwise_distance = ts.average_pairwise_distance(1234, exclusion_zone);
    info!(
        "Average pairwise distance: {}, maximum pairwise distance: {}",
        average_pairwise_distance,
        ts.maximum_distance()
    );

    let pl = indicatif::ProgressBar::new(n as u64);

    // initialize some auxiliary buffers, which will be cloned on a
    // per-thread basis.
    let indices = vec![0usize; n];
    let distances = vec![0.0f64; n];
    let buf = vec![0.0f64; ts.w];

    // compute all k-nearest neighborhoods
    let motiflets = (0..n)
        .into_par_iter()
        .map_with((indices, distances, buf), |(indices, distances, buf), i| {
            pl.inc(1);
            let (extents, indices) =
                k_extents_bf(ts, i, &fft_data, k, exclusion_zone, indices, distances, buf);
            (extents, indices, i)
        })
        .fold(
            || vec![(Distance::infinity(), Vec::new()); k],
            |mut minima, (extents, indices, _root)| {
                for i in 1..extents.len() {
                    if extents[i] < minima[i].0 {
                        minima[i] = (extents[i], indices[..=i].to_owned());
                    }
                }
                minima
            },
        )
        .reduce(
            || vec![(Distance::infinity(), Vec::new()); k],
            |mut m1, m2| {
                for i in 1..k {
                    if m2[i].0 < m1[i].0 {
                        m1[i] = m2[i].clone();
                    }
                }
                m1
            },
        )
        .into_iter()
        .filter(|pair| pair.0.is_finite())
        .collect();
    pl.finish_and_clear();
    motiflets
}

#[derive(Clone, Debug, PartialEq)]
pub struct Motiflet {
    indices: Vec<usize>,
    extent: f64, // FIXME: make this a `Distance`
    /// the relative contrast of this motiflet
    relative_contrast: f64,
    lower_bound: f64,
}
impl Eq for Motiflet {}
impl Ord for Motiflet {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}
#[allow(clippy::non_canonical_partial_ord_impl)]
impl PartialOrd for Motiflet {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.extent.partial_cmp(&other.extent)
    }
}
impl ByteSize for Motiflet {
    fn byte_size(&self) -> Bytes {
        self.indices.byte_size() + self.extent.byte_size() + self.relative_contrast.byte_size()
    }

    fn mem_tree_fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.byte_size())
    }
}

impl Overlaps<Self> for Motiflet {
    fn overlaps(&self, other: Self, exclusion_zone: usize) -> bool {
        let other_indices = other.indices.as_slice();
        for i in &self.indices {
            if i.overlaps(other_indices, exclusion_zone) {
                return true;
            }
        }
        false
    }
}

impl Overlaps<&Self> for Motiflet {
    fn overlaps(&self, other: &Self, exclusion_zone: usize) -> bool {
        let other_indices = other.indices.as_slice();
        for i in &self.indices {
            if i.overlaps(other_indices, exclusion_zone) {
                return true;
            }
        }
        false
    }
}

impl Motiflet {
    pub fn new(indices: Vec<usize>, extent: f64, avg_dist: Distance, lower_bound: f64) -> Self {
        Self {
            indices,
            extent,
            relative_contrast: avg_dist.0 / extent,
            lower_bound,
        }
    }
    pub fn support(&self) -> usize {
        self.indices.len()
    }
    pub fn extent(&self) -> f64 {
        self.extent
    }
    pub fn lower_bound(&self) -> f64 {
        self.lower_bound
    }
    pub fn relative_contrast(&self) -> f64 {
        self.relative_contrast
    }
    pub fn indices(&self) -> Vec<usize> {
        self.indices.clone()
    }
}

#[derive(Debug)]
struct TopK {
    k: usize,
    exclusion_zone: usize,
    threshold: usize,
    top: BTreeSet<Motiflet>,
    emitted: BTreeSet<Motiflet>,
    /// Smallest Not Emitted Distance
    sned: Option<Distance>,
    disabled: bool,
}
impl ByteSize for TopK {
    fn byte_size(&self) -> Bytes {
        self.k.byte_size()
            + self.exclusion_zone.byte_size()
            + self.threshold.byte_size()
            + self.top.byte_size()
            + self.emitted.byte_size()
            + self.sned.byte_size()
    }

    fn mem_tree_fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct(&format!("TopK({})", self.byte_size()))
            .field_with("k", |f| write!(f, "{}", self.k.byte_size()))
            .field_with("exclusion_zone", |f| {
                write!(f, "{}", self.exclusion_zone.byte_size())
            })
            .field_with("threshold", |f| write!(f, "{}", self.threshold.byte_size()))
            .field_with("top", |f| write!(f, "{}", self.top.byte_size()))
            .field_with("emitted", |f| write!(f, "{}", self.emitted.byte_size()))
            .field_with("sned", |f| write!(f, "{}", self.sned.byte_size()))
            .finish()
    }
}

impl TopK {
    fn disable(&mut self) {
        self.disabled = true;
    }
    fn new(k: usize, exclusion_zone: usize) -> Self {
        let threshold = (k + 1).pow(2);
        Self {
            k,
            exclusion_zone,
            threshold,
            top: Default::default(),
            emitted: Default::default(),
            sned: None,
            disabled: false,
        }
    }

    fn is_complete(&self) -> bool {
        self.disabled || self.emitted.len() == self.k
    }

    fn insert(&mut self, motiflet: Motiflet) {
        if self.is_complete() {
            return;
        }
        if self.top.len() == self.threshold {
            if let Some(last) = self.top.last() {
                if &motiflet > last {
                    return;
                }
            }
        }
        if let Some(sned) = self.sned {
            if Distance(motiflet.extent) < sned {
                self.sned.replace(Distance(motiflet.extent));
            }
        }
        self.top.insert(motiflet);
        if self.top.len() > self.threshold {
            self.cleanup();
        }
        assert!(self.top.len() <= self.threshold);
    }

    fn cleanup(&mut self) {
        let mut clean: BTreeSet<Motiflet> = BTreeSet::new();
        for motiflet in self.top.iter() {
            if overlap_count_iter(motiflet, &clean, self.exclusion_zone) < self.k {
                clean.insert(motiflet.clone());
            }
            if clean.len() >= self.threshold {
                break;
            }
        }
        self.top = clean;
    }

    fn smallest_non_emitted_distance(&self) -> Option<Distance> {
        self.sned
    }

    fn kth_distance(&self) -> Option<Distance> {
        let mut non_overlapping: Vec<Motiflet> = Vec::new();
        for motiflet in self.top.iter() {
            if !motiflet.overlaps_iter(&non_overlapping, self.exclusion_zone) {
                non_overlapping.push(motiflet.clone());
            }
            if non_overlapping.len() == self.k {
                return non_overlapping.last().map(|m| Distance(m.extent));
            }
        }
        None
    }

    fn emit<P: Fn(Distance) -> bool>(&mut self, predicate: P) -> Vec<Motiflet> {
        let mut res: Vec<Motiflet> = Vec::new();
        for motiflet in self.top.iter() {
            if self.emitted.len() >= self.k {
                break;
            }
            if !motiflet.overlaps_iter(&self.emitted, self.exclusion_zone) {
                if predicate(Distance(motiflet.extent)) {
                    self.sned.take();
                    res.push(motiflet.clone());
                    self.emitted.insert(motiflet.clone());
                } else {
                    self.sned.replace(Distance(motiflet.extent));
                    break;
                }
            }
        }
        res
    }
}

#[derive(Debug, Clone, Default)]
pub struct MotifletsIteratorStats {
    average_distance: Distance,
    cnt_confirmed: usize,
    next_distance: Distance,
    cnt_candidates: usize,
    cnt_skipped: usize,
    cnt_truncated: usize,
    timeseries_stats: TimeseriesStats,
    graph_stats: GraphStats,
    index_stats: LSHIndexStats,
}

impl MotifletsIteratorStats {
    /// dump observations about the statistics collected
    #[rustfmt::skip]
    fn observe(&self, repetition: usize, prefix: usize) {
        observe!(repetition, prefix, "average_distance", self.average_distance);
        observe!(repetition, prefix, "cnt_confirmed", self.cnt_confirmed);
        observe!(repetition, prefix, "next_distance", self.next_distance);
        observe!(repetition, prefix, "cnt_candidates", self.cnt_candidates);
        observe!(repetition, prefix, "cnt_skipped", self.cnt_skipped);
        observe!(repetition, prefix, "cnt_truncated", self.cnt_truncated);
        if repetition == 0 && prefix == 0 {
            self.timeseries_stats.observe(repetition, prefix);
        }
        self.graph_stats.observe(repetition, prefix);
        self.index_stats.observe(repetition, prefix);
    }

    /// how much effort has been invested in the computation so far?
    fn effort_so_far(&self) -> usize {
        let hashing_cost = crate::index::K
            * self.index_stats.num_repetitions
            * self.timeseries_stats.num_subsequences
            / self.timeseries_stats.window;
        self.cnt_candidates //+ hashing_cost
    }
}

fn build_rooted_motiflets(
    ts: &WindowedTimeseries,
    from: usize,
    fft_data: &FFTData,
    k: usize,
    exclusion_zone: usize,
    avg_distance: f64,
) -> Vec<Motiflet> {
    let mut indices = vec![0usize; ts.num_subsequences()];
    let mut distances = vec![0.0; ts.num_subsequences()];
    let mut buf = vec![0.0; ts.w];
    let (distances, selected_indices) = k_extents_bf(
        ts,
        from,
        fft_data,
        k,
        exclusion_zone,
        &mut indices,
        &mut distances,
        &mut buf,
    );

    let mut res = Vec::new();
    for i in 1..distances.len() {
        let mut motiflet_indices = vec![];
        motiflet_indices.extend_from_slice(&selected_indices[..=i]);
        let extent = distances[i].0;
        res.push(Motiflet {
            indices: motiflet_indices,
            extent,
            relative_contrast: avg_distance / extent,
            lower_bound: extent,
        });
    }

    res
}

pub struct MotifletsIterator {
    pub max_k: usize,
    ts: Arc<WindowedTimeseries>,
    fft_data: FFTData,
    top: Vec<TopK>,
    next_to_confirm: Option<Distance>,
    graph: AdjacencyGraph,
    to_return: Vec<Motiflet>,
    delta: f64,
    exclusion_zone: usize,
    index: LSHIndex,
    // index_stats: IndexStats,
    pairs_buffer: Vec<(u32, u32, Distance)>,
    /// the current repetition
    rep: usize,
    /// the current hash prefix
    prefix: usize,
    /// the previous prefix
    previous_prefix: Option<usize>,
    /// some statistics on the execution
    stats: MotifletsIteratorStats,
    collisions_threshold: usize,
    stop_on_collisions_threshold: bool,
    rng: Xoshiro256PlusPlus,
}

impl ByteSize for MotifletsIterator {
    fn byte_size(&self) -> Bytes {
        self.ts.byte_size()
            + self.fft_data.byte_size()
            + self.top.byte_size()
            + self.to_return.byte_size()
            + self.index.byte_size()
            + self.graph.byte_size()
            + self.pairs_buffer.byte_size()
    }

    fn mem_tree_fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct(&format!("MotifletsIterator({})", self.byte_size()))
            .field_with("ts", |f| write!(f, "{}", self.ts.byte_size()))
            .field_with("fft_data", |f| write!(f, "{}", self.fft_data.byte_size()))
            .field_with("top", |f| write!(f, "{}", self.top.byte_size()))
            .field_with("to_return", |f| write!(f, "{}", self.to_return.byte_size()))
            .field_with("index", |f| self.index.mem_tree_fmt(f))
            .field_with("graph", |f| self.graph.mem_tree_fmt(f))
            .field_with("pairs_buffer", |f| {
                write!(f, "{}", self.pairs_buffer.byte_size())
            })
            .finish()
    }
}

impl MotifletsIterator {
    pub fn new(
        ts: Arc<WindowedTimeseries>,
        max_k: usize, // TODO: rename support
        top_k: usize,
        max_memory: Bytes,
        delta: f64,
        exclusion_zone: usize,
        seed: u64,
        _show_progress: bool,
    ) -> Self {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let start = Instant::now();
        let n = ts.num_subsequences();
        debug!("computing FFT data");
        let fft_data = FFTData::new(&ts);

        let mem_gauge = MemoryGauge::allocated();
        let index = LSHIndex::from_ts(&ts, exclusion_zone, &fft_data, max_memory, seed);
        debug!(
            "Computed initial hash values in {:?}, {}",
            start.elapsed(),
            mem_gauge.measure()
        );

        // get some stats about distances
        let average_pairwise_distance = ts.average_pairwise_distance(1234, exclusion_zone);

        let pairs_buffer = vec![(0, 0, Distance(0.0)); 1 << 20];

        let mut stats = MotifletsIteratorStats::default();
        stats.timeseries_stats = ts.stats();
        stats.average_distance = average_pairwise_distance.into();
        stats.next_distance = Distance::infinity();

        let mut top = Vec::with_capacity(max_k + 1);
        top.resize_with(max_k + 1, || TopK::new(top_k, exclusion_zone));
        top[0].disable();
        top[1].disable();
        // we initialize the top queue with motiflets rooted at
        // a random index, provided the corresponding subsequence is not flat
        let random_root = Uniform::new(0, ts.num_subsequences())
            .sample_iter(&mut rng)
            .filter(|t| !ts.is_flat(*t))
            .next()
            .unwrap();
        for motiflet in build_rooted_motiflets(
            &ts,
            random_root,
            &fft_data,
            max_k,
            exclusion_zone,
            average_pairwise_distance,
        ) {
            top[motiflet.support()].insert(motiflet);
        }

        stats.observe(0, 0);

        // by default, if we do more than 10% of the possible collision comparisons we signal that
        // we have a problem
        let collisions_threshold = stats.timeseries_stats.num_subsequence_pairs / 10;

        let next_to_confirm = top[2].smallest_non_emitted_distance();
        let slf = Self {
            ts,
            fft_data,
            top,
            next_to_confirm,
            graph: AdjacencyGraph::new(n, exclusion_zone),
            max_k,
            to_return: Vec::new(),
            delta,
            exclusion_zone,
            index,
            pairs_buffer,
            rep: 0,
            prefix: crate::index::K,
            previous_prefix: None,
            stats,
            collisions_threshold,
            stop_on_collisions_threshold: false,
            rng,
        };

        if slf.byte_size() > get_maximum_allocation_limit().divide(5) {
            let bytes_per_subsequence =
                get_maximum_allocation_limit().divide(slf.ts.num_subsequences());
            let recommended = Bytes(512);
            log::warn!(
                "The maximum memory setting might be too low (only {} per subsequence).
                The code might crash. It is recommended to provide at least {} per subsequence ({} for this dataset)",
                bytes_per_subsequence,
                recommended,
                recommended * slf.ts.num_subsequences() as f64
            );
        }

        slf
    }

    pub fn set_collision_threshold(&mut self, threshold: usize) {
        self.collisions_threshold = threshold;
    }

    pub fn set_stop_on_collisions_threshold(&mut self, should_stop: bool) {
        self.stop_on_collisions_threshold = should_stop;
    }

    pub fn get_ts(&self) -> Arc<WindowedTimeseries> {
        Arc::clone(&self.ts)
    }

    /// Update the graph
    fn update_graph(&mut self) {
        let timer = Instant::now();
        let prefix = self.prefix;
        let rep = self.rep;
        assert!(rep < self.index.get_repetitions());
        let exclusion_zone = self.exclusion_zone;
        let ts = &self.ts;
        let graph = &mut self.graph;
        let t = Instant::now();
        graph.reset_updated();
        #[rustfmt::skip]
        observe!(rep, prefix, "profile/repetition/update_graph/graph_reset_flags", t.elapsed().as_secs_f64());

        let threshold = self.top[self.max_k]
            .kth_distance()
            .unwrap_or(Distance::infinity());

        let index = &mut self.index;

        let mut time_distance_computation = Duration::default();
        let mut count_collisions = 0;
        let mut time_update_graph = Duration::default();
        let mut enumerator = index.collisions(rep, prefix, self.previous_prefix);
        loop {
            let rng = &mut self.rng;
            let t = Instant::now();
            let maybe_cnt = enumerator.next(self.pairs_buffer.as_mut_slice(), exclusion_zone);
            #[rustfmt::skip]
            observe!(rep, prefix, "profile/repetition/update_graph/enumerator_next", t.elapsed().as_secs_f64());
            if maybe_cnt.is_none() {
                break;
            }
            let cnt = maybe_cnt.unwrap();
            count_collisions += cnt;
            let t = Instant::now();
            // Fixup the distances
            let pairs_buffer = &mut self.pairs_buffer[0..cnt];
            let num_threads = rayon::current_num_threads();
            let cnt_dist_computed = rayon::scope(move |scope| {
                let (cnt_dist_send, cnt_dist_recv) = std::sync::mpsc::channel();
                let cnt_dist_send = Arc::new(cnt_dist_send);
                let chunk_size = cnt.div_ceil(num_threads);
                for chunk in pairs_buffer.chunks_mut(chunk_size) {
                    let mut trng = rng.clone();
                    let cnt_dist_send = Arc::clone(&cnt_dist_send);
                    trng.jump();
                    scope.spawn(move |_| {
                        let mut cnt_dist = 0;
                        let mut cnt_dist_below_threshold = 0;
                        for (a, b, dist) in chunk.iter_mut() {
                            let a = *a as usize;
                            let b = *b as usize;
                            assert!(a < b);
                            cnt_dist += 1;
                            if let Some(d) = zeucl_threshold(ts, a, b, threshold.0) {
                                // we only schedule the pair to update the respective
                                // neighborhoods if it can result in a better motiflet.
                                let d = Distance(d);
                                cnt_dist_below_threshold += 1;
                                *dist = d;
                            } else {
                                *dist = Distance::infinity();
                            }
                        }

                        #[rustfmt::skip]
                        observe!(rep, prefix, "cnt/distcomp", cnt_dist);
                        #[rustfmt::skip]
                        observe!(rep, prefix, "cnt/dist_below_threshold", cnt_dist_below_threshold);
                        cnt_dist_send.send(cnt_dist).unwrap();
                    });
                }
                drop(cnt_dist_send);
                cnt_dist_recv.into_iter().sum::<usize>()
            });
            self.stats.cnt_candidates += cnt_dist_computed;

            time_distance_computation += t.elapsed();

            // Update the neighborhoods
            let t_graph = Instant::now();
            let mut cnt_edges = 0;
            let mem_gauge = MemoryGauge::allocated();
            for (a, b, d) in self.pairs_buffer.iter() {
                if d.is_finite() && !a.overlaps(b, exclusion_zone) {
                    graph.insert(*d, *a as usize, *b as usize);
                    cnt_edges += 1;
                }
            }
            trace!(
                "(rep: {} prefix: {}) inserted {} edges (added {}, stats {:?})",
                rep,
                prefix,
                cnt_edges,
                mem_gauge.measure(),
                graph.stats(),
            );
            time_update_graph += t_graph.elapsed();

            if self.stats.cnt_candidates >= self.collisions_threshold {
                log::info!("Early return from update_graph");
                return;
            }
        } // while there are collisions
        self.index
            .cost_estimator
            .update_collision_time(time_distance_computation, count_collisions);

        log::debug!("{}", self.mem_tree());
        #[rustfmt::skip]
        observe!(rep, prefix, "profile/repetition/update_graph/distance_computation", time_distance_computation.as_secs_f64());
        #[rustfmt::skip]
        observe!(self.rep, self.prefix, "profile/repetition/update_graph/update_graph", time_update_graph.as_secs_f64());
        #[rustfmt::skip]
        observe!(self.rep, self.prefix, "profile/repetition/update_graph", timer.elapsed().as_secs_f64());
    }

    /// adds to `self.to_return` the motiflets that can
    /// be confirmed in this iteration
    fn emit_confirmed(&mut self) {
        let timer = Instant::now();
        let prefix = self.prefix;
        let previous_prefix = self.previous_prefix;
        let rep = self.rep;
        let largest_confirmed = self.index.largest_confirmed_distance(
            self.stats.average_distance,
            self.rep,
            self.prefix,
            self.previous_prefix,
            self.delta,
        );

        let mut time_extents = Duration::from_secs(0);
        let mut cnt_extents = 0;
        let mut cnt_skipped = 0;
        // TODO: check here the neighborhoods that are returned
        for (neighborhoods_ids, dists) in self.graph.neighborhoods(self.max_k + 1) {
            // compute all the extents in one go if one of the
            // distances is smaller than the correponding extent.
            if true
                || dists
                    .iter()
                    .skip(1)
                    .zip(self.top.iter().skip(2).map(|pair| pair.kth_distance()))
                    .any(|(d, ld)| *d < ld.unwrap_or(Distance::infinity()))
            {
                cnt_extents += 1;
                let t = Instant::now();
                let extents = compute_extents(&self.ts, &neighborhoods_ids);
                time_extents += t.elapsed();
                for i in 1..neighborhoods_ids.len() {
                    let ids = &neighborhoods_ids[..=i];
                    let extent = extents[i];
                    let k = ids.len();
                    if k <= self.max_k {
                        let top = &mut self.top[k];
                        top.insert(Motiflet::new(
                            ids.to_owned(),
                            extent.0,
                            self.stats.average_distance,
                            if largest_confirmed < extent {
                                largest_confirmed.0
                            } else {
                                extent.0
                            },
                        ));
                    }
                }
            } else {
                cnt_skipped += 1;
            }
        }
        log::debug!(
            "Time spent computing {} extents: {:?} ({} skipped)",
            cnt_extents,
            time_extents,
            cnt_skipped
        );
        observe!(
            self.rep,
            self.prefix,
            "time_extents_s",
            time_extents.as_secs_f64()
        );

        // finally, we possibly output the points
        for k in 0..=self.max_k {
            let top = &mut self.top[k];
            if !top.is_complete() {
                let new_motiflets = top.emit(|extent| {
                    let fp = self.index.failure_probability(
                        extent,
                        rep + 1, // the number of repetitions we did is the repetition index + 1
                        prefix,
                        previous_prefix,
                    );
                    fp < self.delta
                });
                self.to_return.extend(new_motiflets);
            }
        }
        self.next_to_confirm = self
            .top
            .iter()
            .filter(|top| !top.is_complete())
            .filter_map(|top| top.smallest_non_emitted_distance())
            .min();

        // FIXME: adjust these
        self.stats.cnt_confirmed = self.top[2..]
            .iter()
            .map(|top| top.emitted.len())
            .sum::<usize>();
        self.stats.next_distance = self.next_to_confirm.unwrap_or(Distance::infinity());

        #[rustfmt::skip]
        observe!(self.rep, self.prefix, "profile/repetition/emit_confirmed", timer.elapsed().as_secs_f64());
    }

    fn repetitions_to_confirm(
        &self,
        d: Distance,
        prefix: usize,
        prev_prefix: Option<usize>,
    ) -> Option<usize> {
        let mut nreps = 0;
        let mut fp = 1.0;
        while fp > self.delta && nreps < self.index.max_repetitions() {
            fp = self
                .index
                .failure_probability(d, nreps, prefix, prev_prefix);
            nreps += 1;
        }
        if fp <= self.delta {
            Some(nreps)
        } else {
            None
        }
    }

    pub fn pick_next_repetition(&mut self) {
        // Picking the right prefix and repetition is a crucial part of the practical efficiency
        // of the algorithm. We continue on the current prefix in two cases:
        //  - if the probability of finding anything good is high enough.
        //    In this case we are making a bet: considering
        //    the largest distance confirmed so far, how likely it is to finding something at the
        //    same distance at least once in the next repetitions? Anything farther away is going
        //    to be less likely. If the probability of finding such good candidate is larger than
        //    0.5 then we continue on the current prefix, otherwise we move to the shorter prefix
        //  - if the next distance to confirm is going to be confirmed at the current prefix
        //    within the maximum number of repetitions, and the estimated cost of doing so is smaller
        //    than on shorter prefixes
        //
        // If neither of the two cases above holds, then we move to a shorter prefix (possibly the
        // one minimizing the number of collisions) and restart from the first repetition

        if self.rep + 1 < self.index.max_repetitions() {
            // in this case we can potentially add an additional repetition

            let collisions_so_far = self.stats.cnt_candidates;
            let expected_collisions = self.index.collision_profile();
            self.rep += 1;

            let largest_confirmed = self.index.largest_confirmed_distance(
                self.stats.average_distance,
                self.rep,
                self.prefix,
                self.previous_prefix,
                self.delta,
            );
            // when the `hope` of finding a very good next candidate drops
            // below half, we switch to a shorter prefix.
            // Maybe if this is too extreme we might consider
            // to take the distance halfway from the lartest confirmed and the
            // next to confirm
            let hope = self.index.at_least_one_collision_prob(
                largest_confirmed,
                self.index.max_repetitions() - self.rep,
                self.prefix,
            );

            // will be None if the next distance to confirm is either None or cannot
            // be confirmed at the current prefix
            let additional_collisions_and_repetitions_at_prefix =
                self.next_to_confirm.and_then(|d| {
                    self.repetitions_to_confirm(d, self.prefix, self.previous_prefix)
                        .map(|expected_reps_at_level| {
                            // compute the additional collisions at the current prefix
                            let additional_collisions = (expected_reps_at_level - self.rep) as f64
                                * expected_collisions[self.prefix];
                            let new_repetitions =
                                expected_reps_at_level - self.index.get_repetitions();
                            (additional_collisions, new_repetitions)
                        })
                });

            // will be None if either there is no next distance to confirm or if
            // the current next distance to confirm cannot be confirmed at any
            // shorter prefix
            let collisions_at_shorter = self.next_to_confirm.and_then(|d| {
                (1..self.prefix)
                    .filter_map(|prefix| {
                        // Here we consider the repetitions that are needed to confirm
                        // `d` at a shorter prefix `prefix`. If this number of repetitions
                        // is smaller than the current one and the number of collisions
                        // is also smaller than the expected one we would see if we
                        // stayed at the current level, then we propose this prefix
                        // as a candidate
                        self.repetitions_to_confirm(d, prefix, Some(self.prefix))
                            .map(|est_reps| {
                                // the overall number of expected collisions.
                                // We remove the number of collisions seen so far, since those are skipped.
                                let collisions_at_prefix = est_reps as f64
                                    * expected_collisions[prefix]
                                    - collisions_so_far as f64;
                                (collisions_at_prefix, prefix)
                            })
                    })
                    // Then we pick the minimum, by number of collisions
                    .min_by(|a, b| a.0.total_cmp(&b.0))
            });

            let next_prefix = if hope >= 0.5 {
                log::debug!(
                    "continuing on prefix {}, hope for something at distance {} is {}",
                    self.prefix,
                    largest_confirmed,
                    hope
                );
                self.prefix
            } else {
                match (
                    additional_collisions_and_repetitions_at_prefix,
                    collisions_at_shorter,
                ) {
                    (
                        Some((current_collisions, repetitions_to_add)),
                        Some((shorter_collisions, prefix)),
                    ) => {
                        let cost_estimator = self.index.cost_estimator;
                        let current_estimated_cost = cost_estimator.collision_time().as_secs_f64()
                            * current_collisions
                            + cost_estimator.repetition_time().as_secs_f64()
                                * repetitions_to_add as f64;
                        let shorter_estimated_cost =
                            cost_estimator.collision_time().as_secs_f64() * shorter_collisions;
                        dbg!(current_estimated_cost);
                        dbg!(shorter_estimated_cost);
                        if current_estimated_cost < shorter_estimated_cost {
                            self.prefix
                        } else {
                            prefix
                        }
                    }
                    // (None, Some((_, prefix))) => prefix,
                    (None, None) | (None, Some(_)) => self.prefix - 1,
                    (Some(_), None) => unreachable!(),
                }
            };

            // Adjust things if we shorten the prefix
            if next_prefix < self.prefix {
                self.previous_prefix.replace(self.prefix);
                self.rep = 0;
                self.prefix = next_prefix;
            }

            // possibly add the repetition, if we didn't materialize it previously
            if self.rep + 1 > self.index.get_repetitions() {
                self.index
                    .add_repetitions(&self.ts, &self.fft_data, self.rep + 1, self.prefix);
            }
        } else {
            // we take this branch if we exhausted all possible repetitions
            self.rep = 0;
            self.previous_prefix.replace(self.prefix);
            self.prefix -= 1;
            assert!(self.prefix > 0);
        }
        debug!("Going to prefix {} repetition {}", self.prefix, self.rep);
        self.stats.index_stats = self.index.index_stats();
    }

    pub fn next_interruptible<E, F: FnMut() -> Result<(), E>>(
        &mut self,
        mut f: F,
    ) -> Result<Option<Motiflet>, E> {
        let mut repetition_timer = Instant::now();
        while self.to_return.is_empty() {
            // Give the chance to the caller to interrupt the computation
            // by returning an `Err` in this call
            f()?;
            // check the stopping condition: everything is confirmed
            if self.top.iter().all(|top| top.is_complete()) {
                info!("Execution stats: {:#?}", self.stats);
                info!("Maximum allocated memory: {:?}", Bytes::max_allocated());
                return Ok(None);
            }

            let timer = Instant::now();
            observe_iter(self.rep, self.prefix);
            self.update_graph();
            self.emit_confirmed();
            let repetition_elapsed = timer.elapsed();
            #[rustfmt::skip]
            observe!("repetition_elapsed_s", repetition_elapsed.as_secs_f64());
            #[rustfmt::skip]
            observe!("allocated_bytes", Bytes::allocated().0);

            if self.stats.effort_so_far() > self.collisions_threshold || self.prefix == 1 {
                warn!(
                    "Too much effort! {} > {} (max support smallest non-emitted distance {:?})",
                    self.stats.effort_so_far(),
                    self.collisions_threshold,
                    self.top.last().unwrap().smallest_non_emitted_distance()
                );
                if self.stop_on_collisions_threshold {
                    let largest_confirmed = self.index.largest_confirmed_distance(
                        self.stats.average_distance,
                        self.rep,
                        self.prefix,
                        self.previous_prefix,
                        self.delta,
                    );

                    // emit all the partial candidates
                    for (_, queue) in self.top.iter_mut().enumerate() {
                        self.to_return
                            .extend(queue.emit(|_| true).into_iter().map(|mut m| {
                                m.lower_bound = largest_confirmed.0;
                                m
                            }));
                        queue.disable();
                    }
                    // break from the loop, so that
                    // we can emit the results
                    break;
                }
            }

            self.stats.graph_stats = self.graph.stats();
            debug!("[{}@{}] {:#?}", self.rep, self.prefix, self.stats);
            debug!(
                "[{}@{}] First non confirmed distance {:?} (fp={:?})",
                self.rep,
                self.prefix,
                self.next_to_confirm,
                self.next_to_confirm.map(|d| self.index.failure_probability(
                    d,
                    self.rep,
                    self.prefix,
                    self.previous_prefix
                ))
            );
            self.stats.observe(self.rep, self.prefix);

            #[rustfmt::skip]
            observe!(self.rep, self.prefix, "profile/repetition", repetition_timer.elapsed().as_secs_f64());

            repetition_timer = Instant::now();
            self.pick_next_repetition();
            #[rustfmt::skip]
            observe!(self.rep, self.prefix, "profile/repetition/setup", repetition_timer.elapsed().as_secs_f64());
        }

        Ok(self.to_return.pop())
    }
}

impl Iterator for MotifletsIterator {
    type Item = Motiflet;

    fn next(&mut self) -> Option<Self::Item> {
        // call the interruptive iterator, without interrupting
        self.next_interruptible(|| Ok::<(), ()>(())).unwrap()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use crate::load::loadts;
    use std::collections::HashMap;
    use std::sync::Arc;

    fn run_motiflet_test(ts: Arc<WindowedTimeseries>, k: usize, seed: u64) {
        let failure_probability = 0.01;
        let exclusion_zone = ts.w;

        let iter = MotifletsIterator::new(
            Arc::clone(&ts),
            k,
            1,
            Bytes::gbytes(1),
            failure_probability,
            exclusion_zone,
            seed,
            false,
        );
        let motiflets: HashMap<usize, Motiflet> = iter.map(|m| (m.support(), m)).collect();
        dbg!(&motiflets);

        let brute_force: HashMap<usize, (Distance, Vec<usize>)> =
            brute_force_motiflets(&ts, k, exclusion_zone)
                .into_iter()
                .map(|pair| (pair.1.len(), pair))
                .collect();

        for k in 2..=k {
            dbg!(k);
            let motiflet = &motiflets[&k];
            let mut motiflet_indices = motiflet.indices();
            eprintln!("Extent of discovered motiflet {}", motiflet.extent());
            motiflet_indices.sort();

            let (ground_extent, mut ground_indices) = brute_force[&k].clone();
            eprintln!("Ground distance of {} motiflet: {}", k, ground_extent);
            eprintln!("Ground motiflet is {:?}", ground_indices);
            ground_indices.sort();
            // check that the indices of the motiflet found are not too far away from
            // the true ones.
            for (actual_i, ground_i) in motiflet_indices.iter().zip(&ground_indices) {
                assert!((*actual_i as isize - *ground_i as isize).abs() <= (ts.w / 2) as isize);
            }
        }
    }

    #[test]
    fn test_ecg_motiflet_k2() {
        let ts: Vec<f64> = loadts("data/ECG.csv.gz", Some(10000)).unwrap();
        let ts = Arc::new(WindowedTimeseries::new(ts, 100, false));
        run_motiflet_test(ts, 2, 123456);
    }

    #[test]
    fn test_ecg_motiflet_k5() {
        let ts: Vec<f64> = loadts("data/ECG.csv.gz", Some(10000)).unwrap();
        let ts = Arc::new(WindowedTimeseries::new(ts, 100, false));
        run_motiflet_test(ts, 5, 123456);
    }

    #[test]
    fn test_ecg_motiflet_k8() {
        env_logger::init();
        let ts: Vec<f64> = loadts("data/ECG.csv.gz", Some(10000)).unwrap();
        let ts = Arc::new(WindowedTimeseries::new(ts, 100, false));
        run_motiflet_test(ts, 8, 123456);
    }

    #[test]
    #[ignore]
    fn test_ecg_motiflet_k10() {
        let ts: Vec<f64> = loadts("data/ECG.csv.gz", Some(20000)).unwrap();
        let ts = Arc::new(WindowedTimeseries::new(ts, 50, false));
        run_motiflet_test(ts, 10, 1234567);
    }

    #[test]
    fn test_ecg_motiflet_ground_truth() {
        let ts: Vec<f64> = loadts("data/ecg-heartbeat-av.csv", None).unwrap();
        let w = 100;
        let exclusion_zone = w / 2;
        let ts = Arc::new(WindowedTimeseries::new(ts, w, false));

        let mut ground = HashMap::new();
        ground.insert(6, &[113, 241, 369, 497]);

        let grounds = &[
            vec![113, 241, 369, 497],
            vec![31, 159, 287, 415, 543, 671],
            vec![
                1308, 1434, 1519, 1626, 1732, 1831, 1938, 2034, 2118, 2227, 2341, 2415, 2510, 2607,
                2681, 2787,
            ],
        ];

        let mut iter = MotifletsIterator::new(
            Arc::clone(&ts),
            grounds.last().unwrap().len(),
            1,
            Bytes::gbytes(1),
            0.01,
            exclusion_zone,
            1234,
            false,
        );
        let prob_motiflets: HashMap<usize, Motiflet> =
            (&mut iter).map(|m| (m.support(), m)).collect();

        for ground in grounds {
            let k = ground.len();
            let brute_force = brute_force_motiflets(&ts, k, exclusion_zone);
            let (_bf_extent, mut bf_idxs): (Distance, Vec<usize>) = brute_force
                .iter()
                .find(|pair| pair.1.len() == k)
                .unwrap()
                .clone();
            dbg!(_bf_extent);
            bf_idxs.sort();
            for (i_expected, i_actual) in ground.iter().zip(&bf_idxs) {
                assert!((*i_expected as isize - *i_actual as isize).abs() <= 1);
            }

            let m = &prob_motiflets[&k];
            dbg!(m.extent());
            let mut p_idxs = m.indices();
            p_idxs.sort();
            dbg!(&p_idxs);
            dbg!(&ground);
            for (i_expected, i_actual) in ground.iter().zip(&p_idxs) {
                assert!(
                    (*i_expected as isize - *i_actual as isize).abs() <= 2 // <= exclusion_zone as isize / 2
                );
            }
        }
    }
}
