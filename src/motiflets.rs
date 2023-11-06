use crate::{
    distance::zeucl,
    knn::*,
    lsh::{self, ColumnBuffers, HashCollection, Hasher},
    motifs::Stats,
    timeseries::{FFTData, WindowedTimeseries},
};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::{collections::HashMap, time::Instant};
use std::{
    collections::{BinaryHeap, HashSet},
    sync::Arc,
};

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
) -> (OrdF64, Vec<usize>) {
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
    indices.select_nth_unstable_by_key(n_candidates, |j| OrdF64(distances[*j]));

    // Sort the candidate indices by increasing distance (the previous step)
    // only partitioned the indices in two groups with the guarantee that the first
    // `n_candidates` indices are the ones at shortest distance from the `from` point,
    // but they are not guaranteed to be sorted
    let indices = &mut indices[..n_candidates];
    indices.sort_unstable_by_key(|j| OrdF64(distances[*j]));

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
            if jj.max(hh) - jj.min(hh) < exclusion_zone {
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

    let mut extent = 0.0f64;
    for i in 0..k {
        for j in (i + 1)..k {
            let d = zeucl(ts, ret[i], ret[j]);
            extent = extent.max(d);
        }
    }

    (OrdF64(extent), ret)
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
    eprintln!("Root of motiflet is {root}");
    (extent.0, indices)
}

pub fn probabilistic_motiflets(
    ts: &WindowedTimeseries,
    k: usize,
    exclusion_zone: usize,
    repetitions: usize,
    target_failure_probability: f64,
    seed: u64,
) -> (f64, Vec<usize>) {
    const BUFFER_SIZE: usize = 1 << 16;
    let fft_data = FFTData::new(&ts);
    let average_distance = ts.average_pairwise_distance(seed, exclusion_zone);
    println!(
        "Average subsequence distance is {} (maximum {})",
        average_distance,
        ts.maximum_distance()
    );

    let hasher_width = Hasher::estimate_width(&ts, &fft_data, 1, None, seed);

    let hasher = Arc::new(Hasher::new(ts.w, repetitions, hasher_width, seed));
    let pools = HashCollection::from_ts(&ts, &fft_data, Arc::clone(&hasher));
    let mut hash_buffers = ColumnBuffers::default();
    let mut support_buffers = SupportBuffers::new(ts);
    let mut pairs_buffer = vec![(0u32, 0u32, OrdF64(f64::INFINITY)); BUFFER_SIZE];

    let mut neighborhoods: HashMap<usize, SubsequenceNeighborhood> = HashMap::new();

    let mut brute_forced = 0;

    let mut last_extent = f64::INFINITY;
    let mut cnt_distances = 0;

    let mut previous_prefix = None;
    let mut prefix = lsh::K;
    while prefix > 0 {
        eprintln!("=== {prefix}");
        eprintln!(
            "Computed {} distances so far, {}% of all possible",
            cnt_distances,
            100.0 * cnt_distances as f64 / ts.num_subsequence_pairs() as f64
        );
        hasher.print_collision_probabilities(average_distance, prefix);
        for rep in 0..repetitions {
            pools.group_subsequences(prefix, rep, exclusion_zone, &mut hash_buffers, false);
            let mut rep_collisions = 0;
            if let Some(mut enumerator) = hash_buffers.enumerator() {
                while let Some(cnt) = enumerator.next(&mut pairs_buffer, exclusion_zone) {
                    // Fixup the distances
                    pairs_buffer[0..cnt]
                        // .par_iter_mut()
                        // .with_min_len(1024)
                        .iter_mut()
                        .for_each(|(a, b, dist)| {
                            let a = *a as usize;
                            let b = *b as usize;
                            rep_collisions += 1;
                            if let Some(first_colliding_repetition) =
                                pools.first_collision(a, b, prefix)
                            {
                                if first_colliding_repetition == rep
                                    && previous_prefix
                                        .map(|prefix| pools.first_collision(a, b, prefix).is_none())
                                        .unwrap_or(true)
                                {
                                    cnt_distances += 1;
                                    *dist = OrdF64(zeucl(ts, a, b));
                                }
                            }
                        });

                    // Update the neighborhoods
                    for (a, b, dist) in &pairs_buffer[0..cnt] {
                        assert!(dist.0 > 0.0);
                        if dist.0.is_finite() {
                            let a = *a as usize;
                            let b = *b as usize;
                            neighborhoods
                                .entry(a)
                                .or_insert_with(|| SubsequenceNeighborhood::evolving(k, a))
                                .update(*dist, b);
                            neighborhoods
                                .entry(b)
                                .or_insert_with(|| SubsequenceNeighborhood::evolving(k, b))
                                .update(*dist, a);
                        }
                    }
                }
            } // while there are collisions
            if rep_collisions == 0 {
                eprintln!("WARNING: No collisions in repetition {}@{}", rep, prefix);
            }

            let num_top_extents = 10;
            let mut top_extents = BinaryHeap::new();
            for (idx, neighborhood) in neighborhoods.iter_mut() {
                if neighborhood.is_evolving() {
                    let ext = neighborhood.extent(k - 1, exclusion_zone);
                    // we brute force only the extents that can
                    // overtake the current best motiflet
                    if ext.0 <= 2.0 * last_extent {
                        top_extents.push((ext, *idx));
                    }
                    if top_extents.len() > num_top_extents {
                        top_extents.pop();
                    }
                }
            }
            for (_, idx) in top_extents {
                brute_forced += 1;
                neighborhoods.get_mut(&idx).unwrap().brute_force(
                    ts,
                    &fft_data,
                    k,
                    &mut support_buffers,
                );
            }

            // Find the smallest extent so far
            if let Some((smallest_extent, smallest_extent_idx)) = neighborhoods
                .iter_mut()
                .map(|(idx, neighborhood)| (neighborhood.extent(k - 1, exclusion_zone), idx))
                .min()
            {
                if smallest_extent.0.is_finite() {
                    let fp_smallest_extent =
                        hasher.failure_probability(smallest_extent.0, rep, prefix, previous_prefix);

                    if smallest_extent.0 < last_extent {
                        println!(
                            "[{}@{}] Updated best extent: {} rooted at {}",
                            rep, prefix, smallest_extent.0, smallest_extent_idx
                        );
                        last_extent = smallest_extent.0;
                    }

                    // Find the failure probabilities
                    let mut max_fp = fp_smallest_extent.powi((k - 1) as i32);
                    for neighborhood in neighborhoods.values_mut() {
                        let fp = neighborhood.failure_probability(
                            k - 1,
                            smallest_extent,
                            fp_smallest_extent,
                            exclusion_zone,
                        );
                        max_fp = max_fp.max(fp);
                    }
                    if rep == 0 {
                        eprintln!(
                            "smallest_extent {} failure probability {}",
                            smallest_extent.0, max_fp
                        );
                    }
                    if max_fp < target_failure_probability {
                        let n_neighborhoods = neighborhoods.len();
                        // pick the neighborhood with the best extent
                        let (extent, best_idx, best_neighborhood) = neighborhoods
                            .iter_mut()
                            .map(|(idx, ns)| (ns.extent(k - 1, exclusion_zone), idx, ns))
                            .min_by_key(|tup| tup.0)
                            .unwrap();
                        let ids = best_neighborhood.neighbors(k);
                        eprintln!(
                            "Returning motiflet {:?} with extent {} rooted at {} with failure probability {} (brute forced {} over {}, computed {} distances)",
                            ids, extent.0, best_idx, max_fp, brute_forced, n_neighborhoods, cnt_distances
                        );
                        return (extent.0, ids);
                    }
                }
            }
        }
        previous_prefix.replace(prefix);

        prefix -= 1;
        eprintln!(
            "Next prefix is {}, where the failure probability will be {} in the first iteration",
            prefix,
            hasher.failure_probability(last_extent, 0, prefix, previous_prefix)
        );
    }

    unreachable!()
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
    max_k: usize,
    ts: Arc<WindowedTimeseries>,
    fft_data: FFTData,
    best_motiflet: Vec<(OrdF64, usize, bool)>,
    neighborhoods: HashMap<usize, SubsequenceNeighborhood>,
    to_return: Vec<Motiflet>,
    repetitions: usize,
    delta: f64,
    exclusion_zone: usize,
    hasher: Arc<Hasher>,
    pools: Arc<HashCollection>,
    buffers: ColumnBuffers,
    pairs_buffer: Vec<(u32, u32, OrdF64)>,
    support_buffers: SupportBuffers,
    /// the current repetition
    rep: usize,
    /// the current hash prefix
    prefix: usize,
    /// the previous prefix
    previous_prefix: Option<usize>,
    /// the progress bar
    pbar: Option<ProgressBar>,
    /// the execution statistics
    exec_stats: Stats,
}

impl MotifletsIterator {
    pub fn new(
        ts: Arc<WindowedTimeseries>,
        max_k: usize,
        repetitions: usize,
        delta: f64,
        exclusion_zone: usize,
        seed: u64,
        show_progress: bool,
    ) -> Self {
        let start = Instant::now();
        let fft_data = FFTData::new(&ts);

        let hasher_width = Hasher::estimate_width(&ts, &fft_data, 1, None, seed);

        let hasher = Arc::new(Hasher::new(ts.w, repetitions, hasher_width, seed));
        let pools = HashCollection::from_ts(&ts, &fft_data, Arc::clone(&hasher));
        let pools = Arc::new(pools);
        eprintln!("Computed hash values in {:?}", start.elapsed());

        let pbar = if show_progress {
            Some(Self::build_progress_bar(lsh::K, repetitions))
        } else {
            None
        };

        let best_motiflet = vec![(OrdF64(std::f64::INFINITY), 0, false); max_k + 1];
        let pairs_buffer = vec![(0, 0, OrdF64(0.0)); 65536];
        let support_buffers = SupportBuffers::new(&ts);

        Self {
            ts,
            fft_data,
            best_motiflet,
            neighborhoods: HashMap::new(),
            max_k,
            to_return: Vec::new(),
            repetitions,
            delta,
            exclusion_zone,
            hasher,
            pools,
            buffers: ColumnBuffers::default(),
            pairs_buffer,
            support_buffers,
            rep: 0,
            prefix: lsh::K,
            previous_prefix: None,
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
}

impl Iterator for MotifletsIterator {
    type Item = Motiflet;

    fn next(&mut self) -> Option<Self::Item> {
        while self.to_return.is_empty() {
            // check the stopping condition: everything is confirmed
            if self.best_motiflet[2..].iter().all(|tup| tup.2) {
                return None;
            }

            let prefix = self.prefix;
            let previous_prefix = self.previous_prefix;
            let rep = self.rep;
            let exclusion_zone = self.exclusion_zone;
            let max_k = self.max_k;
            let pools = &self.pools;
            let ts = &self.ts;

            self.pools
                .group_subsequences(prefix, rep, exclusion_zone, &mut self.buffers, false);
            let mut rep_collisions = 0;
            if let Some(mut enumerator) = self.buffers.enumerator() {
                while let Some(cnt) =
                    enumerator.next(self.pairs_buffer.as_mut_slice(), exclusion_zone)
                {
                    // Fixup the distances
                    self.pairs_buffer[0..cnt]
                        // .par_iter_mut()
                        // .with_min_len(1024)
                        .iter_mut()
                        .for_each(|(a, b, dist)| {
                            let a = *a as usize;
                            let b = *b as usize;
                            rep_collisions += 1;
                            if let Some(first_colliding_repetition) =
                                pools.first_collision(a, b, prefix)
                            {
                                if first_colliding_repetition == rep
                                    && previous_prefix
                                        .map(|prefix| pools.first_collision(a, b, prefix).is_none())
                                        .unwrap_or(true)
                                {
                                    *dist = OrdF64(zeucl(ts, a, b));
                                }
                            }
                        });

                    // Update the neighborhoods
                    for (a, b, dist) in &self.pairs_buffer[0..cnt] {
                        assert!(dist.0 > 0.0);
                        if dist.0.is_finite() {
                            let a = *a as usize;
                            let b = *b as usize;
                            self.neighborhoods
                                .entry(a)
                                .or_insert_with(|| SubsequenceNeighborhood::evolving(max_k, a))
                                .update(*dist, b);
                            self.neighborhoods
                                .entry(b)
                                .or_insert_with(|| SubsequenceNeighborhood::evolving(max_k, b))
                                .update(*dist, a);
                        }
                    }
                }
            } // while there are collisions

            // Resolve the most promising candidates
            let mut exts = vec![OrdF64(0.0); self.max_k + 1];
            let mut to_brute_force = HashSet::new();
            for (idx, neighborhood) in self.neighborhoods.iter_mut() {
                if neighborhood.is_evolving() {
                    neighborhood.extents(exclusion_zone, &mut exts);
                    // we brute force only the extents that can
                    // overtake the current best motiflet for a given k
                    for k in 2..=self.max_k {
                        if exts[k].0 <= 2.0 * (self.best_motiflet[k].0).0 {
                            to_brute_force.insert(*idx);
                        }
                    }
                }
            }
            for idx in to_brute_force {
                let neighborhood = self.neighborhoods.get_mut(&idx).unwrap();
                neighborhood.brute_force(ts, &self.fft_data, self.max_k, &mut self.support_buffers);
                neighborhood.extents(exclusion_zone, &mut exts);
                for k in 2..=self.max_k {
                    if !self.best_motiflet[k].2 && exts[k] <= self.best_motiflet[k].0 {
                        self.best_motiflet[k] = (exts[k], idx, false);
                    }
                }
            }

            // And now try to emit the motiflets confirmed in this iteration, if any
            for k in 2..=self.max_k {
                let (extent, root_idx, emitted) = self.best_motiflet[k];
                if !emitted {
                    let fp = self
                        .hasher
                        .failure_probability(extent.0, rep, prefix, previous_prefix)
                        .powi(k as i32);
                    if fp < self.delta {
                        let neighborhood = self.neighborhoods.get_mut(&root_idx).unwrap();
                        self.best_motiflet[k].2 = true;
                        let indices = neighborhood.neighbors(k);
                        let m = Motiflet::new(indices, extent.0);
                        self.to_return.push(m);
                    }
                }
            }

            // Advance
            self.rep += 1;
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
    use std::sync::Arc;

    fn run_motiflet_test(
        ts: Arc<WindowedTimeseries>,
        k: usize,
        repetitions: usize,
        seed: u64,
        ground_truth: Option<(f64, Vec<usize>)>,
    ) {
        let failure_probability = 0.01;
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
        let result: Vec<Motiflet> = iter.collect();
        dbg!(result);

        let (ground_extent, mut ground_indices): (f64, Vec<usize>) =
            ground_truth.unwrap_or_else(|| {
                eprintln!(
                    "Running brute force algorithm on {} subsequences",
                    ts.num_subsequences()
                );
                brute_force_motiflets(&ts, k, exclusion_zone)
            });
        eprintln!("Ground distance of {} motiflet: {}", k, ground_extent);
        eprintln!("Motiflet is {:?}", ground_indices);
        ground_indices.sort();
        let (_motiflet_extent, mut motiflet_indices) = probabilistic_motiflets(
            &ts,
            k,
            exclusion_zone,
            repetitions,
            failure_probability,
            seed,
        );
        motiflet_indices.sort();
        assert_eq!(motiflet_indices, ground_indices);
    }

    #[test]
    fn test_ecg_motiflet_k5() {
        let ts: Vec<f64> = loadts("data/ECG.csv.gz", Some(10000)).unwrap();
        let ts = Arc::new(WindowedTimeseries::new(ts, 100, false));
        run_motiflet_test(ts, 5, 8192, 123456, None);
    }

    #[test]
    fn test_ecg_motiflet_k8() {
        let ts: Vec<f64> = loadts("data/ECG.csv.gz", Some(10000)).unwrap();
        let ts = Arc::new(WindowedTimeseries::new(ts, 100, false));
        run_motiflet_test(ts, 8, 8192, 123456, None);
    }

    #[test]
    fn test_ecg_motiflet_k10() {
        let ts: Vec<f64> = loadts("data/ECG.csv.gz", Some(20000)).unwrap();
        let ts = Arc::new(WindowedTimeseries::new(ts, 50, false));
        run_motiflet_test(ts, 10, 8192, 123456, None);
    }
}
