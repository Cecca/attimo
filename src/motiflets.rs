use crate::{
    distance::zeucl,
    knn::*,
    lsh::{self, ColumnBuffers, HashCollection, Hasher},
    timeseries::{FFTData, Overlaps, WindowedTimeseries},
};
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
    let n_candidates = (k * exclusion_zone).min(ts.num_subsequences());
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

    eprintln!(
        "Average pairwise distance: {}, maximum pairwise distance: {}",
        ts.average_pairwise_distance(1234, exclusion_zone),
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
    eprintln!("Root of motiflet is {root}");
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

pub struct MotifletsIterator {
    pub max_k: usize,
    ts: Arc<WindowedTimeseries>,
    best_motiflet: Vec<(Distance, usize, bool)>,
    graph: KnnGraph,
    to_return: Vec<Motiflet>,
    repetitions: usize,
    delta: f64,
    exclusion_zone: usize,
    hasher: Arc<Hasher>,
    pools: Arc<HashCollection>,
    buffers: Vec<ColumnBuffers>,
    pairs_buffer: Vec<(u32, u32, Distance)>,
    /// the current repetition
    rep: usize,
    /// the current hash prefix
    prefix: usize,
    /// the previous prefix
    previous_prefix: Option<usize>,
    /// the number of threads used
    threads: usize,
}

impl MotifletsIterator {
    pub fn new(
        ts: Arc<WindowedTimeseries>,
        max_k: usize,
        repetitions: usize,
        delta: f64,
        exclusion_zone: usize,
        seed: u64,
        _show_progress: bool,
    ) -> Self {
        let start = Instant::now();
        let n = ts.num_subsequences();
        let fft_data = FFTData::new(&ts);

        let hasher_width = Hasher::compute_width(&ts);
        eprintln!(
            "Average pairwise distance: {}",
            ts.average_pairwise_distance(seed, exclusion_zone)
        );

        let threads = rayon::current_num_threads();
        assert!(
            repetitions % threads == 0,
            "The number of repetitions ({}) should be a multiple of the number of available threads ({})", 
            repetitions, threads
        );

        let hasher = Arc::new(Hasher::new(ts.w, repetitions, hasher_width, seed));
        let pools = HashCollection::from_ts(&ts, &fft_data, Arc::clone(&hasher));
        let pools = Arc::new(pools);
        eprintln!("Computed hash values in {:?}", start.elapsed());

        let best_motiflet = vec![(Distance(std::f64::INFINITY), 0, false); max_k];
        let pairs_buffer = vec![(0, 0, Distance(0.0)); 65536];
        let mut buffers = Vec::with_capacity(threads);
        for _ in 0..threads {
            buffers.push(ColumnBuffers::default());
        }

        Self {
            ts,
            best_motiflet,
            graph: KnnGraph::new(max_k, n, exclusion_zone),
            max_k,
            to_return: Vec::new(),
            repetitions,
            delta,
            exclusion_zone,
            hasher,
            pools,
            buffers,
            pairs_buffer,
            rep: 0,
            prefix: lsh::K,
            previous_prefix: None,
            threads,
        }
    }

    pub fn get_ts(&self) -> Arc<WindowedTimeseries> {
        Arc::clone(&self.ts)
    }

    /// Update the neighborhoods with collisions
    fn update_neighborhoods(&mut self) {
        let prefix = self.prefix;
        let previous_prefix = self.previous_prefix;
        let base_rep = self.rep;
        let exclusion_zone = self.exclusion_zone;
        let pools = &self.pools;
        let ts = &self.ts;
        let graph = &mut self.graph;

        let threshold = self.best_motiflet[self.max_k - 1].0;

        (base_rep..base_rep + self.threads)
            .into_par_iter()
            .zip(&mut self.buffers)
            .for_each(|(rep, buffer)| {
                pools.group_subsequences(prefix, rep, exclusion_zone, buffer, false);
            });

        let mut cnt_candidates = 0;
        for buffer in self.buffers.iter() {
            if let Some(mut enumerator) = buffer.enumerator() {
                while let Some(cnt) =
                    enumerator.next(self.pairs_buffer.as_mut_slice(), exclusion_zone)
                {
                    cnt_candidates += cnt;
                    // Fixup the distances
                    self.pairs_buffer[0..cnt]
                        .par_iter_mut()
                        .for_each(|(a, b, dist)| {
                            let a = *a as usize;
                            let b = *b as usize;
                            if previous_prefix
                                .map(|prefix| pools.first_collision(a, b, prefix).is_none())
                                .unwrap_or(true)
                            {
                                let d = Distance(zeucl(ts, a, b));
                                if d <= threshold {
                                    // we only schedule the pair to update the respective
                                    // neighborhoods if it can result in a better motiflet.
                                    *dist = d;
                                } else {
                                    *dist = Distance(std::f64::INFINITY);
                                }
                            }
                            // }
                        });

                    // Update the neighborhoods
                    graph.update_batch(self.pairs_buffer.as_mut_slice());
                }
            }
            // while there are collisions
        }
        eprintln!("Candidate pairs {}", cnt_candidates);
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
                let fp = self.hasher.failure_probability_independent(
                    extent.0,
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
                }
            }
        }
    }
}

impl Iterator for MotifletsIterator {
    type Item = Motiflet;

    fn next(&mut self) -> Option<Self::Item> {
        while self.to_return.is_empty() {
            // check the stopping condition: everything is confirmed
            if self.best_motiflet[1..].iter().all(|tup| tup.2) {
                return None;
            }

            self.update_neighborhoods();
            self.emit_confirmed();

            if self.rep % 512 == 0 {
                eprintln!("[{}@{}] {:?}", self.rep, self.prefix, self.graph.stats());
                eprintln!("[{}@{}] {:?}", self.rep, self.prefix, self.best_motiflet);
            }

            // Advance
            self.rep += self.threads;
            if self.rep >= self.repetitions {
                self.rep = 0;
                self.previous_prefix.replace(self.prefix);
                self.prefix -= 1;
            }
        }

        self.to_return.pop()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::load::loadts;
    use std::collections::HashMap;
    use std::sync::Arc;

    fn run_motiflet_test(ts: Arc<WindowedTimeseries>, k: usize, repetitions: usize, seed: u64) {
        let failure_probability = 0.1;
        let exclusion_zone = ts.w;

        let iter = MotifletsIterator::new(
            Arc::clone(&ts),
            k,
            repetitions,
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
            eprintln!("Motiflet is {:?}", ground_indices);
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
