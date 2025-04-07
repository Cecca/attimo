use crate::allocator::*;
use crate::distance::zeucl_threshold;
use crate::graph::{AdjacencyGraph, GraphStats};
use crate::index::{LSHIndexStats, INITIAL_REPETITIONS};
use crate::observe::*;
use crate::timeseries::{overlap_count_iter, TimeseriesStats};
use crate::{
    index::{IndexStats, LSHIndex},
    knn::*,
    timeseries::{FFTData, Overlaps, WindowedTimeseries},
};
use log::*;
use rand::prelude::*;
use rand::SeedableRng;
use rand_distr::{Bernoulli, Uniform};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use rustfft::num_complex::ComplexFloat;
use std::collections::BTreeSet;
use std::sync::atomic::AtomicUsize;
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

    // Compute the distance profile using the MASS algorithm
    ts.distance_profile(fft_data, from, distances, buf);

    // Reset the indices of the subsequences
    (0..ts.num_subsequences()).for_each(|i| {
        indices[i] = i;
    });
    // // Find the likely candidates by a (partial) indirect sort of
    // // the indices by increasing distance.
    // let n_candidates = (k * exclusion_zone).min(indices.len() - 1);
    // assert!(n_candidates <= indices.len());
    // indices.select_nth_unstable_by_key(n_candidates, |j| Distance(distances[*j]));
    //
    // // Sort the candidate indices by increasing distance (the previous step)
    // // only partitioned the indices in two groups with the guarantee that the first
    // // `n_candidates` indices are the ones at shortest distance from the `from` point,
    // // but they are not guaranteed to be sorted
    // let indices = &mut indices[..n_candidates];
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

// Find the (approximate) motiflets by brute force: for each subsequence find its
// k-nearest neighbors, compute their extents, and pick the neighborhood with the
// smallest extent.
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
    /// the relative contrast of this motifle
    relative_contrast: f64,
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
    pub fn new(indices: Vec<usize>, extent: f64, avg_dist: Distance) -> Self {
        Self {
            indices,
            extent,
            relative_contrast: avg_dist.0 / extent,
        }
    }
    pub fn support(&self) -> usize {
        self.indices.len()
    }
    pub fn extent(&self) -> f64 {
        self.extent
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
        self.sned.take();
        self.top.insert(motiflet);
        if self.top.len() > self.threshold {
            self.cleanup();
        }
        assert!(self.top.len() <= self.threshold);
    }

    fn len(&self) -> usize {
        self.top.len()
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
    index_stats: IndexStats,
    pairs_buffer: Vec<(u32, u32, Distance)>,
    /// the current repetition
    rep: usize,
    /// the current hash prefix
    prefix: usize,
    /// the previous prefix
    previous_prefix: Option<usize>,
    /// the repetitions done at the previous prefix
    previous_prefix_repetitions: Option<usize>,
    /// some statistics on the execution
    stats: MotifletsIteratorStats,
    collisions_threshold: usize,
    stop_on_collisions_threshold: bool,
    rng: Xoshiro256PlusPlus,
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
        let start = Instant::now();
        let n = ts.num_subsequences();
        let mem_gauge = MemoryGauge::allocated();
        let fft_data = FFTData::new(&ts);
        debug!("Computed fft_data: {}", mem_gauge.measure());

        let mem_gauge = MemoryGauge::allocated();
        let index = LSHIndex::from_ts(&ts, exclusion_zone, &fft_data, seed);
        debug!(
            "Computed initial hash values in {:?}, {}",
            start.elapsed(),
            mem_gauge.measure()
        );

        // get some stats about distances
        let average_pairwise_distance = ts.average_pairwise_distance(1234, exclusion_zone);
        let (min_nn_estimate, avg_nn_estimate, sampled_min_index_pair) =
            ts.nearest_neighbor_distance_stats(&fft_data, 12435, exclusion_zone);
        info!(
            "Average pairwise distance: {}, maximum pairwise distance: {}, min/average nearest neighbor dist: {:?}",
            average_pairwise_distance,
            ts.maximum_distance(),
            (min_nn_estimate, avg_nn_estimate)
        );

        let pairs_buffer = vec![(0, 0, Distance(0.0)); 1 << 20];

        let index_stats = index.stats(&ts, max_memory, exclusion_zone);
        info!("Pools stats: {:?}", index_stats);
        let first_meaningful_prefix = index_stats.first_meaningful_prefix();
        assert!(first_meaningful_prefix > 0);

        let mut stats = MotifletsIteratorStats::default();
        stats.timeseries_stats = ts.stats();
        stats.average_distance = average_pairwise_distance.into();
        stats.next_distance = Distance::infinity();

        let mut top = Vec::with_capacity(max_k + 1);
        top.resize_with(max_k + 1, || TopK::new(top_k, exclusion_zone));
        top[0].disable();
        top[1].disable();
        // we initialize the top queue for the second motiflet
        // with the sampled closest distance pair
        top[2].insert(Motiflet::new(
            vec![sampled_min_index_pair.0, sampled_min_index_pair.1],
            min_nn_estimate,
            stats.average_distance,
        ));

        stats.observe(0, 0);

        // by default, if we do more than 10% of the possible collision comparisons we signal that
        // we have a problem
        let collisions_threshold = stats.timeseries_stats.num_subsequence_pairs / 10;

        Self {
            ts,
            fft_data,
            top,
            // the minimum sampled nearest neighbor is of course a
            // natural candidate for the 2-motiflet
            next_to_confirm: Some(min_nn_estimate.into()),
            graph: AdjacencyGraph::new(n, exclusion_zone),
            max_k,
            to_return: Vec::new(),
            delta,
            exclusion_zone,
            index,
            index_stats,
            pairs_buffer,
            rep: 0,
            // this way we avoid meaningless repetitions that have no collisions
            prefix: first_meaningful_prefix,
            previous_prefix: None,
            previous_prefix_repetitions: None,
            stats,
            collisions_threshold,
            stop_on_collisions_threshold: false,
            rng: Xoshiro256PlusPlus::seed_from_u64(seed),
        }
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
        let p_threshold = self.index.collision_probability_at(threshold);
        let p_sample = if p_threshold.is_nan() {
            // this happens on the first iteration, when the threshold is infinity
            1.0
        } else {
            let p_sample = 8.0 / (p_threshold * self.index.get_repetitions() as f64);
            p_sample.min(1.0)
        };
        let coin = Bernoulli::new(p_sample).unwrap();

        let index = &mut self.index;

        let mut time_distance_computation = Duration::default();
        let mut time_update_graph = Duration::default();
        let mut enumerator = index.collisions(rep, prefix, self.previous_prefix);
        loop {
            let rng = &mut self.rng;
            let t = Instant::now();
            let check_prefix = rep < self.previous_prefix_repetitions.unwrap_or(0);
            let maybe_cnt = enumerator.next(
                self.pairs_buffer.as_mut_slice(),
                exclusion_zone,
                check_prefix,
            );
            #[rustfmt::skip]
            observe!(rep, prefix, "profile/repetition/update_graph/enumerator_next", t.elapsed().as_secs_f64());
            if maybe_cnt.is_none() {
                break;
            }
            let cnt = maybe_cnt.unwrap();
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
                        let mut cnt_skipped = 0;
                        let mut cnt_dist = 0;
                        let mut cnt_dist_below_threshold = 0;
                        for (a, b, dist) in chunk.iter_mut() {
                            if !coin.sample(&mut trng) {
                                *dist = Distance::infinity();
                                cnt_skipped += 1;
                                continue;
                            }
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
                        observe!(rep, prefix, "cnt/sampling_skipped", cnt_skipped);
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
            for (a, b, d) in self.pairs_buffer.iter() {
                if d.is_finite() && !a.overlaps(b, exclusion_zone) {
                    graph.insert(*d, *a as usize, *b as usize);
                }
            }
            time_update_graph += t_graph.elapsed();
        } // while there are collisions
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
        let previous_prefix_repetitions = self.previous_prefix_repetitions;
        let rep = self.rep;

        let mut time_extents = Duration::from_secs(0);
        let mut cnt_extents = 0;
        let mut cnt_skipped = 0;
        for (neighborhoods_ids, dists) in self.graph.neighborhoods(self.max_k + 1) {
            // compute all the extents in one go if one of the
            // distances is smaller than the correponding extent.
            if dists
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
                        previous_prefix_repetitions,
                    );
                    fp < self.delta
                });
                self.to_return.extend(new_motiflets);
            }
        }
        self.next_to_confirm = self
            .top
            .iter()
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
        // self.graph
        //     .remove_larger_than(self.best_motiflet.last().unwrap().0);
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
                return Ok(None);
            }

            if self.stats.cnt_candidates > self.collisions_threshold {
                warn!(
                    "Too many collisions! {} > {} (max support smallest non-emitted distance {:?})",
                    self.stats.cnt_candidates,
                    self.collisions_threshold,
                    self.top.last().unwrap().smallest_non_emitted_distance()
                );
                if self.stop_on_collisions_threshold {
                    // emit all the partial candidates
                    for (supp, queue) in self.top.iter_mut().enumerate() {
                        self.to_return.extend(queue.emit(|_| true));
                        queue.disable();
                    }
                    // break from the loop, so that
                    // we can emit the results
                    break;
                }
            }

            let timer = Instant::now();
            self.update_graph();
            self.emit_confirmed();
            let repetition_elapsed = timer.elapsed();
            #[rustfmt::skip]
            observe!(self.rep, self.prefix, "repetition_elapsed_s", repetition_elapsed.as_secs_f64());
            #[rustfmt::skip]
            observe!(self.rep, self.prefix, "repetition_estimate_s", self.index_stats.repetition_cost_estimate(self.prefix));

            self.stats.graph_stats = self.graph.stats();
            debug!("[{}@{}] {:#?}", self.rep, self.prefix, self.stats);
            // debug!("[{}@{}] {:?}", self.rep, self.prefix, self.best_motiflet);
            debug!(
                "[{}@{}] First non confirmed distance {:?}",
                self.rep, self.prefix, self.next_to_confirm
            );
            self.stats.observe(self.rep, self.prefix);

            #[rustfmt::skip]
            observe!(self.rep, self.prefix, "profile/repetition", repetition_timer.elapsed().as_secs_f64());
            repetition_timer = Instant::now();

            // We consider two cases:
            //  - The current best candidate can be confirmed with more repetitions on
            //    this level: instantiate the repetitions and keep the prefix length as it is
            //  - We need a shorter prefix. In this case rather than jumping to the
            //    shorter prefix (which at times is just 1) we shorten the prefix
            //    just by one, in the hope of discovering better candidates
            let next_prefix = if (self.rep + 1 < INITIAL_REPETITIONS
                && self.previous_prefix.is_none())
                || self.rep + 1
                    < self
                        .previous_prefix_repetitions
                        .unwrap_or(INITIAL_REPETITIONS)
            {
                // stay at the current prefix for the first few repetitions, to allow discovery
                // of at least a few pairs
                debug!("Still in the initial repetitions, continuing with the current prefix");
                self.prefix
            } else if let Some(first_unconfirmed) = self.next_to_confirm {
                let (best_prefix, _required_repetitions) = if first_unconfirmed.is_finite() {
                    let costs = self.index_stats.costs_to_confirm(
                        self.prefix,
                        first_unconfirmed,
                        self.delta,
                        &self.index,
                    );
                    debug!("Costs: {:?}", costs);
                    let (best_prefix, (best_cost, required_repetitions)) = costs
                        .iter()
                        .enumerate()
                        .min_by(|(_, tup1), (_, tup2)| tup1.0.total_cmp(&tup2.0))
                        .unwrap();
                    if best_cost.is_infinite() {
                        warn!("Best prefix would be 0, continuing on this level");
                        (self.prefix, self.index_stats.max_repetitions)
                    } else {
                        debug!(
                            "Best prefix to confirm {} is {} with {} repetitions with cost {}",
                            first_unconfirmed, best_prefix, required_repetitions, best_cost
                        );
                        (best_prefix, *required_repetitions)
                    }
                } else if self.rep < self.index_stats.max_repetitions {
                    let required_repetitions =
                        (self.index.get_repetitions() + 32).min(self.index_stats.max_repetitions);
                    warn!(
                        "No motiflet to confirm, continuing at level {} with new repetitions",
                        self.prefix
                    );
                    (self.prefix, required_repetitions)
                } else {
                    warn!(
                        "No motiflet to confirm and built all repetitions, going to level {}",
                        self.prefix - 1
                    );
                    (self.prefix - 1, 1)
                };
                best_prefix
            } else {
                self.prefix
            };
            let index_stats = self.index.index_stats();
            self.stats.index_stats = index_stats;

            if next_prefix >= self.prefix {
                // Advance on the current prefix
                self.rep += 1;
                debug!("Advancing to repetition {}", self.rep);
                if self.rep >= self.index.get_repetitions() {
                    let elapsed = self.index.add_repetitions(
                        &self.ts,
                        &self.fft_data,
                        self.rep + 1,
                        next_prefix + 1,
                    );
                    #[rustfmt::skip]
                    observe!(self.rep, self.prefix, "repetition_setup_s", elapsed.as_secs_f64());
                    #[rustfmt::skip]
                    observe!(self.rep, self.prefix, "repetition_setup_estimate_s", self.index_stats.repetition_setup_estimate());
                }
            } else {
                // Go to the suggested prefix, and start from the first repetition there
                self.previous_prefix_repetitions.replace(self.rep + 1);
                self.rep = 0;
                self.previous_prefix.replace(self.prefix);
                self.prefix -= 1;
                debug!("Going to prefix {}", self.prefix);
            }
            assert!(self.prefix > 0);

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
            Bytes::gbytes(4),
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
        env_logger::init();
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
            Bytes::gbytes(8),
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
