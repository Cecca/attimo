use crate::{
    distance::zeucl,
    knn::*,
    lsh::{ColumnBuffers, HashCollection, HashCollectionStats, Hasher, RepetitionIndex},
    timeseries::{Bytes, FFTData, Overlaps, WindowedTimeseries},
};
use log::*;
use rayon::prelude::*;
use std::{sync::Arc, time::Instant};

/// Find the `k` nearest neighbors of the given `from` subsequence, respecting the
/// given `exclusion_zone`. Returns a pair where the first element is the extent
/// of the neighborhood thus found (i.e. the maximum z-normalized Eucliedean distance
/// between any two points in the neighborhood, non squared), and the second element
/// is a vector of the actual neighbors found.
fn k_nearest_neighbors_bf(
    ts: &WindowedTimeseries,
    from: usize,
    fft_data: &FFTData,
    k: usize,
    exclusion_zone: usize,
    indices: &mut [usize],
    distances: &mut [f64],
    buf: &mut [f64],
) -> (Distance, Vec<usize>) {
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
    // Find the likely candidates by a (partial) indirect sort of
    // the indices by increasing distance.
    let n_candidates = (k * exclusion_zone).min(indices.len() - 1);
    assert!(n_candidates <= indices.len());
    indices.select_nth_unstable_by_key(n_candidates, |j| Distance(distances[*j]));

    // Sort the candidate indices by increasing distance (the previous step)
    // only partitioned the indices in two groups with the guarantee that the first
    // `n_candidates` indices are the ones at shortest distance from the `from` point,
    // but they are not guaranteed to be sorted
    let indices = &mut indices[..n_candidates];
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
    assert_eq!(ret.len(), k);

    (compute_extent(ts, &ret), ret)
}

// Find the (approximate) motiflets by brute force: for each subsequence find its
// k-nearest neighbors, compute their extents, and pick the neighborhood with the
// smallest extent.
pub fn brute_force_motiflets(
    ts: &WindowedTimeseries,
    k: usize,
    exclusion_zone: usize,
) -> (f64, Vec<usize>) {
    debug_assert!(false, "Should run only in `release mode`");
    // pre-compute the FFT for the time series
    let fft_data = FFTData::new(&ts);
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
    let mut indices = Vec::new();
    indices.resize(n, 0usize);
    let mut distances = Vec::new();
    distances.resize(n, 0.0f64);
    let mut buf = Vec::new();
    buf.resize(ts.w, 0.0f64);

    // compute all k-nearest neighborhoods
    let (extent, indices, root) = (0..n)
        .into_par_iter()
        .map_with((indices, distances, buf), |(indices, distances, buf), i| {
            pl.inc(1);
            let (extent, indices) = k_nearest_neighbors_bf(
                ts,
                i,
                &fft_data,
                k,
                exclusion_zone,
                indices,
                distances,
                buf,
            );
            (extent, indices, i)
        })
        .min()
        .unwrap();
    pl.finish_and_clear();
    info!("Root of motiflet is {root}");
    (extent.0, indices)
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
    best_motiflet: Vec<(Distance, usize, bool)>,
    graph: KnnGraph,
    to_return: Vec<Motiflet>,
    repetitions: usize,
    delta: f64,
    exclusion_zone: usize,
    // hasher: Arc<Hasher>,
    pools: HashCollection,
    pools_stats: HashCollectionStats,
    buffers: ColumnBuffers,
    pairs_buffer: Vec<(u32, u32, Distance)>,
    /// the current repetition
    rep: usize,
    /// the current hash prefix
    prefix: usize,
    /// the previous prefix
    previous_prefix: Option<usize>,
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
        let fft_data = FFTData::new(&ts);

        let repetitions = 16;
        let hasher_width = Hasher::compute_width(&ts);

        let hasher = Hasher::new(ts.w, repetitions, hasher_width, seed);
        let pools = HashCollection::from_ts(&ts, &fft_data, hasher);
        info!("Computed hash values in {:?}", start.elapsed());

        let average_pairwise_distance = ts.average_pairwise_distance(1234, exclusion_zone);
        info!(
            "Average pairwise distance: {}, maximum pairwise distance: {}",
            average_pairwise_distance,
            ts.maximum_distance(),
        );

        let best_motiflet = vec![(Distance(std::f64::INFINITY), 0, false); max_k];
        let pairs_buffer = vec![(0, 0, Distance(0.0)); 65536];
        let buffers = ColumnBuffers::default();

        let pools_stats = pools.stats(&ts, max_memory, exclusion_zone);
        info!("Pools stats: {:?}", pools_stats);
        let first_meaningful_prefix = pools_stats.first_meaningful_prefix();

        Self {
            ts,
            fft_data,
            best_motiflet,
            graph: KnnGraph::new(max_k, n, exclusion_zone),
            max_k,
            to_return: Vec::new(),
            repetitions,
            delta,
            exclusion_zone,
            pools,
            pools_stats,
            buffers,
            pairs_buffer,
            rep: 0,
            // this way we avoid meaningless repetitions that have no collisions
            prefix: first_meaningful_prefix,
            previous_prefix: None,
            stats: MotifletsIteratorStats::default(),
        }
    }

    pub fn get_ts(&self) -> Arc<WindowedTimeseries> {
        Arc::clone(&self.ts)
    }

    /// Update the neighborhoods with collisions
    fn update_neighborhoods(&mut self) {
        let prefix = self.prefix;
        let previous_prefix = self.previous_prefix;
        let rep: RepetitionIndex = self.rep.into();
        let exclusion_zone = self.exclusion_zone;
        let pools = &mut self.pools;
        let ts = &self.ts;
        let graph = &mut self.graph;

        let threshold = self.best_motiflet[self.max_k - 1].0;

        pools.group_subsequences(prefix, rep, exclusion_zone, &mut self.buffers, true);

        let mut cnt_candidates = 0;
        let mut sum_dist = 0.0;
        let mut cnt_distances = 0;
        if let Some(mut enumerator) = self.buffers.enumerator() {
            while let Some(cnt) = enumerator.next(self.pairs_buffer.as_mut_slice(), exclusion_zone)
            {
                cnt_candidates += cnt;
                self.stats.cnt_candidates += cnt;
                // Fixup the distances
                let (d, c): (f64, usize) = self.pairs_buffer[0..cnt]
                    .par_iter_mut()
                    .with_min_len(1024)
                    .map(|(a, b, dist)| {
                        let a = *a as usize;
                        let b = *b as usize;
                        // TODO: Re-introduce the check to verify if a pair has been verified iwth a
                        // longer prefix. The caveat now is that we might do different number of
                        // repetitions at different prefixes
                        // TODO: maybe skip pairs with only one collision
                        if true {
                            let d = Distance(zeucl(ts, a, b));
                            if d <= threshold {
                                // we only schedule the pair to update the respective
                                // neighborhoods if it can result in a better motiflet.
                                *dist = d;
                            } else {
                                *dist = Distance(std::f64::INFINITY);
                            }
                            (d.0, 1)
                        } else {
                            (0.0, 0)
                        }
                    })
                    .reduce(
                        || (0.0f64, 0usize),
                        |accum, pair| (accum.0 + pair.0, accum.1 + pair.1),
                    );

                sum_dist += d;
                cnt_distances += c;

                // Update the neighborhoods
                graph.update_batch(self.pairs_buffer.as_mut_slice());
            }
            // while there are collisions
        }
        let average_distance = sum_dist / cnt_distances as f64;
        let average_distance_probability = self
            .pools
            .collision_probability_at(Distance(average_distance))
            .powi(prefix as i32);
        debug!(
            "Candidate pairs {}, distances computed {}, average distance {} (p={})",
            cnt_candidates, cnt_distances, average_distance, average_distance_probability
        );
        // assert!(average_distance_probability.is_nan() || average_distance_probability >= 0.000001);
    }

    /// adds to `self.to_return` the motiflets that can
    /// be confirmed in this iteration
    fn emit_confirmed(&mut self) {
        let prefix = self.prefix;
        let previous_prefix = self.previous_prefix;
        let rep = self.rep;
        let mut failure_probabilities = vec![1.0; self.max_k];

        // first we update the best motiflets
        self.graph.update_extents(&self.ts);
        let min_extents = self.graph.min_extents();
        let mut thresholds = Vec::with_capacity(self.max_k);
        for k in 0..self.max_k {
            let (extent, root_idx, emitted) = &mut self.best_motiflet[k];
            if !*emitted {
                if min_extents[k].0 < *extent {
                    *extent = min_extents[k].0;
                    *root_idx = min_extents[k].1;
                }
                thresholds.push(*extent);
            } else {
                thresholds.push(Distance::infinity());
            }
        }

        // then, we populate the min_to replace vector
        let min_to_replace = self.graph.min_count_above_many(&thresholds);

        // finally, we possibly output the points
        for k in 0..self.max_k {
            let (extent, root_idx, emitted) = &mut self.best_motiflet[k];
            if !*emitted && extent.0.is_finite() {
                assert!(min_to_replace[k] > 0);
                assert!(min_to_replace[k] <= self.max_k);
                // FIXME: fix the computation of failure probability with the
                // caveat that different prefixes might use a different number
                // of repetitions
                let fp = self.pools.failure_probability_independent(
                    *extent,
                    rep,
                    prefix,
                    previous_prefix,
                );
                let fp = fp.powi(min_to_replace[k] as i32);
                assert!(fp <= 1.0);
                failure_probabilities[k] = fp;

                if fp < self.delta {
                    let indices = self.graph.get(*root_idx, k);
                    *emitted = true;
                    let m = Motiflet::new(indices, extent.0);
                    self.to_return.push(m);
                } else if rep % 512 == 0 {
                    info!(
                        "[{}@{}] failure probability for k={}: {}",
                        rep, prefix, k, fp
                    );
                }
            }
        }
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

            self.update_neighborhoods();
            self.emit_confirmed();

            if self.rep % 512 == 0 {
                info!("[{}@{}] {:?}", self.rep, self.prefix, self.graph.stats());
                info!("[{}@{}] {:?}", self.rep, self.prefix, self.best_motiflet);
            }

            let first_unconfirmed = self
                .best_motiflet
                .iter()
                .filter(|tup| tup.0.is_finite())
                .find_map(|tup| if !tup.2 { Some(tup.0) } else { None });
            info!("First non confirmed distance {:?}", first_unconfirmed);

            let best_prefix = if let Some(first_unconfirmed) = first_unconfirmed {
                let costs = self.pools_stats.costs_to_confirm(
                    self.prefix,
                    first_unconfirmed,
                    self.delta,
                    &self.pools.get_hasher(),
                );
                let (best_prefix, (best_cost, required_repetitions)) = costs
                    .iter()
                    .enumerate()
                    .min_by(|(_, tup1), (_, tup2)| tup1.0.total_cmp(&tup2.0))
                    .unwrap();
                info!(
                    "Best prefix to confirm {} is {} with {} repetitions with cost {}",
                    first_unconfirmed, best_prefix, required_repetitions, best_cost
                );
                if *required_repetitions > self.pools.get_repetitions() {
                    self.pools.add_repetitions(
                        &self.ts,
                        &self.fft_data,
                        required_repetitions.next_power_of_two(),
                    );
                }
                best_prefix
            } else {
                self.prefix
            };

            if best_prefix >= self.prefix {
                // Advance on the current prefix
                self.rep += 1;
                if self.rep >= self.repetitions {
                    self.rep = 0;
                    self.previous_prefix.replace(self.prefix);
                    self.prefix -= 1;
                }
            } else {
                // Go to the suggested prefix, and start from the first repetition there
                self.rep = 0;
                self.previous_prefix.replace(self.prefix);
                self.prefix = best_prefix;
            }
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

    fn run_motiflet_test(ts: Arc<WindowedTimeseries>, k: usize, repetitions: usize, seed: u64) {
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

        for k in 2..=k {
            dbg!(k);
            let motiflet = &motiflets[&k];
            let mut motiflet_indices = motiflet.indices();
            eprintln!("Extent of discovered motiflet {}", motiflet.extent());
            motiflet_indices.sort();

            let (ground_extent, mut ground_indices): (f64, Vec<usize>) =
                brute_force_motiflets(&ts, k, exclusion_zone);
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
        run_motiflet_test(ts, 2, 8192, 123456);
    }

    #[test]
    fn test_ecg_motiflet_k5() {
        let ts: Vec<f64> = loadts("data/ECG.csv.gz", Some(10000)).unwrap();
        let ts = Arc::new(WindowedTimeseries::new(ts, 100, false));
        run_motiflet_test(ts, 5, 8192, 123456);
    }

    #[test]
    fn test_ecg_motiflet_k8() {
        let ts: Vec<f64> = loadts("data/ECG.csv.gz", Some(10000)).unwrap();
        let ts = Arc::new(WindowedTimeseries::new(ts, 100, false));
        run_motiflet_test(ts, 8, 8192, 123456);
    }

    #[test]
    #[ignore]
    fn test_ecg_motiflet_k10() {
        let ts: Vec<f64> = loadts("data/ECG.csv.gz", Some(20000)).unwrap();
        let ts = Arc::new(WindowedTimeseries::new(ts, 50, false));
        run_motiflet_test(ts, 10, 8192, 1234567);
    }
}
