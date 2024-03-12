use crate::allocator::*;
use crate::distance::zeucl;
use crate::graph::Graph;
use crate::index::INITIAL_REPETITIONS;
use crate::{
    index::{IndexStats, LSHIndex},
    knn::*,
    // lsh::{ColumnBuffers, HashCollection, HashCollectionStats, Hasher, RepetitionIndex},
    timeseries::{FFTData, Overlaps, WindowedTimeseries},
};
use log::*;
use rayon::prelude::*;
use std::{sync::Arc, time::Instant};

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
    ts.distance_profile(&fft_data, from, distances, buf);

    // Reset the indices of the subsequences
    for i in 0..ts.num_subsequences() {
        indices[i] = i;
    }
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
                // if jj.max(hh) - jj.min(hh) < exclusion_zone {
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
    pub fn new(indices: Vec<usize>, extent: f64) -> Self {
        Self { indices, extent }
    }
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

#[derive(Debug, Clone, Default)]
pub struct MotifletsIteratorStats {
    cnt_candidates: usize,
}

pub struct MotifletsIterator {
    pub max_k: usize,
    ts: Arc<WindowedTimeseries>,
    fft_data: FFTData,
    best_motiflet: Vec<(Distance, Vec<usize>, bool)>,
    next_to_confirm: Option<Distance>,
    graph: Graph,
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
}

impl MotifletsIterator {
    pub fn new(
        ts: Arc<WindowedTimeseries>,
        max_k: usize,
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
        info!("Computed fft_data: {}", mem_gauge.measure());

        let mem_gauge = MemoryGauge::allocated();
        let index = LSHIndex::from_ts(&ts, &fft_data, seed);
        info!(
            "Computed initial hash values in {:?}, {}",
            start.elapsed(),
            mem_gauge.measure()
        );

        let average_pairwise_distance = ts.average_pairwise_distance(1234, exclusion_zone);
        debug!(
            "Average pairwise distance: {}, maximum pairwise distance: {}",
            average_pairwise_distance,
            ts.maximum_distance(),
        );

        let mut best_motiflet = vec![(Distance(std::f64::INFINITY), Vec::new(), false); max_k + 1];
        best_motiflet[0].2 = true;
        best_motiflet[1].2 = true;
        let pairs_buffer = vec![(0, 0, Distance(0.0)); 65536];

        let index_stats = index.stats(&ts, max_memory, exclusion_zone);
        info!("Pools stats: {:?}", index_stats);
        let first_meaningful_prefix = index_stats.first_meaningful_prefix();
        assert!(first_meaningful_prefix > 0);

        Self {
            ts,
            fft_data,
            best_motiflet,
            next_to_confirm: None,
            graph: Graph::new(exclusion_zone),
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
            stats: MotifletsIteratorStats::default(),
        }
    }

    pub fn get_ts(&self) -> Arc<WindowedTimeseries> {
        Arc::clone(&self.ts)
    }

    /// Update the graph
    fn update_graph(&mut self) {
        let prefix = self.prefix;
        let rep = self.rep;
        assert!(rep < self.index.get_repetitions());
        let exclusion_zone = self.exclusion_zone;
        let index = &mut self.index;
        let ts = &self.ts;
        let graph = &mut self.graph;

        let num_collisions_threshold = ts.num_subsequence_pairs() / 2;

        let threshold = self.best_motiflet[self.max_k - 1].0;

        let mut cnt_candidates = 0;
        let mut enumerator = index.collisions(rep, prefix, self.previous_prefix);
        while let Some(cnt) = enumerator.next(self.pairs_buffer.as_mut_slice(), exclusion_zone) {
            cnt_candidates += cnt;
            if cnt_candidates > num_collisions_threshold {
                panic!(
                    "Too many collisions: {} out of {} possible pairs",
                    cnt_candidates,
                    ts.num_subsequence_pairs()
                );
            }
            self.stats.cnt_candidates += cnt;
            // Fixup the distances
            let (d, c): (f64, usize) = self.pairs_buffer[0..cnt]
                .par_iter_mut()
                .with_min_len(1024)
                .map(|(a, b, dist)| {
                    let a = *a as usize;
                    let b = *b as usize;
                    assert!(a < b);
                    let d = Distance(zeucl(ts, a, b));
                    if d <= threshold {
                        // we only schedule the pair to update the respective
                        // neighborhoods if it can result in a better motiflet.
                        *dist = d;
                        (d.0, 1)
                    } else {
                        *dist = Distance(std::f64::INFINITY);
                        (0.0, 0)
                    }
                })
                .reduce(
                    || (0.0f64, 0usize),
                    |accum, pair| (accum.0 + pair.0, accum.1 + pair.1),
                );

            // Update the neighborhoods
            for (a, b, d) in self.pairs_buffer.iter() {
                if d.is_finite() && !a.overlaps(b, exclusion_zone) {
                    graph.insert(*d, *a as usize, *b as usize);
                }
            }
        } // while there are collisions
        debug!("collisions at prefix {}: {}", prefix, cnt_candidates);
    }

    /// adds to `self.to_return` the motiflets that can
    /// be confirmed in this iteration
    fn emit_confirmed(&mut self) {
        let prefix = self.prefix;
        let previous_prefix = self.previous_prefix;
        let previous_prefix_repetitions = self.previous_prefix_repetitions;
        let rep = self.rep;
        let mut failure_probabilities = vec![1.0; self.max_k + 1];

        for (dist, ids) in self.graph.neighborhoods() {
            let k = ids.len();
            if k <= self.max_k {
                // `dist` is a lower bound to the extent. If it is larger than
                // the current extent at k we can simply skip running this set of points
                if dist < self.best_motiflet[k].0 {
                    let extent = compute_extent(&self.ts, &ids);
                    let (best_extent, best_indices, emitted) = &mut self.best_motiflet[k];
                    if !*emitted && extent < *best_extent {
                        *best_extent = extent;
                        *best_indices = ids;
                    }
                }
            }
            if let Some(max_extent) = self.best_motiflet.iter().map(|tup| tup.0).max() {
                if max_extent < dist {
                    break;
                }
            }
        }

        // finally, we possibly output the points
        for k in 0..=self.max_k {
            let (extent, indices, emitted) = &mut self.best_motiflet[k];
            if !*emitted && extent.0.is_finite() {
                self.next_to_confirm.replace(*extent);
                let fp = self.index.failure_probability(
                    *extent,
                    rep,
                    prefix,
                    previous_prefix,
                    previous_prefix_repetitions,
                );
                // NOTE: here we used to raise the failure probability by the number of
                // neighbors that need to be replaced in order for this candidate to be overcome.
                // It turns out that this is too aggressive (i.e. it makes the algorithm fail
                // more often than we'd like), therefore we keep as a failure probability
                // the probability of having missed a _single_ neighbor.
                assert!(fp <= 1.0);
                failure_probabilities[k] = fp;

                if fp < self.delta {
                    *emitted = true;
                    let m = Motiflet::new(indices.clone(), extent.0);
                    self.to_return.push(m);
                }
            }
        }

        self.graph
            .remove_larger_than(self.best_motiflet.last().unwrap().0);
    }

    pub fn next_interruptible<E, F: FnMut() -> Result<(), E>>(
        &mut self,
        mut f: F,
    ) -> Result<Option<Motiflet>, E> {
        while self.to_return.is_empty() {
            // Give the chance to the caller to interrupt the computation
            // by returning an `Err` in this call
            f()?;
            // check the stopping condition: everything is confirmed
            if self.best_motiflet.iter().all(|tup| tup.2) {
                info!("Execution stats: {:#?}", self.stats);
                return Ok(None);
            }

            self.update_graph();
            self.emit_confirmed();

            debug!("[{}@{}] {:?}", self.rep, self.prefix, self.graph.stats());
            debug!("[{}@{}] {:?}", self.rep, self.prefix, self.best_motiflet);
            debug!(
                "[{}@{}] First non confirmed distance {:?}",
                self.rep, self.prefix, self.next_to_confirm
            );

            let next_prefix =
                if self.rep + 1 < INITIAL_REPETITIONS && self.previous_prefix.is_none() {
                    // stay at the current prefix for the first few repetitions, to allow discovery
                    // of at least a few pairs
                    debug!("Still in the initial repetitions, continuing with the current prefix");
                    self.prefix
                } else if let Some(first_unconfirmed) = self.next_to_confirm {
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
                    if best_prefix <= self.prefix {
                        debug!(
                            "Best prefix to confirm {} is {} with {} repetitions with cost {}",
                            first_unconfirmed, best_prefix, required_repetitions, best_cost
                        );
                    }
                    if *required_repetitions > self.index.get_repetitions() {
                        let new_total_repetitions: usize = (*required_repetitions)
                            .min(self.index.get_repetitions() + rayon::current_num_threads());
                        self.index
                            .add_repetitions(&self.ts, &self.fft_data, new_total_repetitions);
                    }
                    best_prefix
                } else {
                    self.prefix
                };

            if next_prefix >= self.prefix {
                // Advance on the current prefix
                self.rep += 1;
                debug!("Advancing to repetition {}", self.rep);
                if self.rep >= self.index.get_repetitions() {
                    self.previous_prefix_repetitions.replace(self.rep);
                    self.rep = 0;
                    self.previous_prefix.replace(self.prefix);
                    self.prefix -= 1;
                    debug!("Not enough repetitions, going to prefix {}", self.prefix);
                }
            } else {
                // Go to the suggested prefix, and start from the first repetition there
                self.previous_prefix_repetitions.replace(self.rep);
                self.rep = 0;
                self.previous_prefix.replace(self.prefix);
                self.prefix = next_prefix;
            }
            assert!(
                self.prefix > 0,
                "prefix got to zero, motiflets situation:\n{:?}",
                self.best_motiflet
            );
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
