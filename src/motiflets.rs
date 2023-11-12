use crate::{
    distance::zeucl,
    knn::*,
    lsh::{self, ColumnBuffers, HashCollection, Hasher},
    timeseries::{FFTData, Overlaps, WindowedTimeseries},
};
use rayon::prelude::*;
use std::{cmp::Reverse, collections::BinaryHeap, collections::HashMap, sync::Arc, time::Instant};

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

    let mut extent = 0.0f64;
    for i in 0..k {
        for j in (i + 1)..k {
            let d = zeucl(ts, ret[i], ret[j]);
            extent = extent.max(d);
        }
    }

    (Distance(extent), ret)
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
    fft_data: FFTData,
    threads: usize,
    best_motiflet: Vec<(Distance, usize, bool)>,
    neighborhoods: HashMap<usize, SubsequenceNeighborhood>,
    to_return: Vec<Motiflet>,
    repetitions: usize,
    delta: f64,
    exclusion_zone: usize,
    hasher: Arc<Hasher>,
    pools: Arc<HashCollection>,
    buffers: ColumnBuffers,
    pairs_buffer: Vec<(u32, u32, Distance)>,
    support_buffers: SupportBuffers,
    /// the current repetition
    rep: usize,
    /// the current hash prefix
    prefix: usize,
    /// the previous prefix
    previous_prefix: Option<usize>,
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
        let support_buffers = SupportBuffers::new(&ts);

        Self {
            ts,
            fft_data,
            threads,
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
        }
    }

    pub fn get_ts(&self) -> Arc<WindowedTimeseries> {
        Arc::clone(&self.ts)
    }

    /// Update the neighborhoods with collisions
    fn update_neighborhoods(&mut self) {
        let n = self.ts.num_subsequences();
        let prefix = self.prefix;
        let previous_prefix = self.previous_prefix;
        let rep = self.rep;
        let exclusion_zone = self.exclusion_zone;
        let max_k = self.max_k;
        let pools = &self.pools;
        let ts = &self.ts;
        let neighborhoods = &mut self.neighborhoods;
        eprintln!(
            "[{}@{}] number of neighborhoods: {}/{}, of which exact {}",
            rep,
            prefix,
            neighborhoods.len(),
            n,
            neighborhoods.values().filter(|n| !n.is_evolving()).count()
        );
        eprintln!("[{}@{}] {:?}", rep, prefix, self.best_motiflet);

        let threshold = self.best_motiflet[self.max_k - 1].0;

        let mut cnt_distances = 0;
        self.pools
            .group_subsequences(prefix, rep, exclusion_zone, &mut self.buffers, false);
        if let Some(mut enumerator) = self.buffers.enumerator() {
            while let Some(cnt) = enumerator.next(self.pairs_buffer.as_mut_slice(), exclusion_zone)
            {
                // Fixup the distances
                self.pairs_buffer[0..cnt]
                    .iter_mut()
                    .for_each(|(a, b, dist)| {
                        let a = *a as usize;
                        let b = *b as usize;
                        // TODO: keep track of the evolving and exact neighborhoods separately
                        if neighborhoods.contains_key(&a)
                            && neighborhoods.contains_key(&b)
                            && !(neighborhoods[&a].is_evolving() && neighborhoods[&b].is_evolving())
                        {
                            // just skip the computation if both neighborhoods were already
                            // resolved
                            *dist = Distance(std::f64::INFINITY);
                        } else if let Some(first_colliding_repetition) =
                            pools.first_collision(a, b, prefix)
                        {
                            if first_colliding_repetition == rep
                                && previous_prefix
                                    .map(|prefix| pools.first_collision(a, b, prefix).is_none())
                                    .unwrap_or(true)
                            {
                                cnt_distances += 1;
                                let d = Distance(zeucl(ts, a, b));
                                if d <= threshold {
                                    // we only schedule the pair to update the respective
                                    // neighborhoods if it can result in a better motiflet.
                                    *dist = d;
                                } else {
                                    *dist = Distance(std::f64::INFINITY);
                                }
                            }
                        }
                    });

                // Update the neighborhoods
                for (a, b, dist) in &self.pairs_buffer[0..cnt] {
                    assert!(dist.0 > 0.0);
                    if dist.0.is_finite() {
                        let a = *a as usize;
                        let b = *b as usize;
                        neighborhoods
                            .entry(a)
                            .or_insert_with(|| SubsequenceNeighborhood::evolving(max_k, a))
                            .update(*dist, b);
                        neighborhoods
                            .entry(b)
                            .or_insert_with(|| SubsequenceNeighborhood::evolving(max_k, b))
                            .update(*dist, a);
                    }
                }
            }
        } // while there are collisions
        eprintln!("[{}@{}] computed {} distances", rep, prefix, cnt_distances);
    }

    /// Brute forces the most promising neighborhoods
    fn resolve_most_promising(&mut self) {
        let exclusion_zone = self.exclusion_zone;
        let ts = &self.ts;

        // Resolve the most promising candidates
        let mut exts = vec![Distance(0.0); self.max_k];
        let mut to_brute_force = vec![BinaryHeap::new(); self.max_k];
        for (idx, neighborhood) in self.neighborhoods.iter_mut() {
            if neighborhood.is_evolving() {
                neighborhood.extents(exclusion_zone, &mut exts);
                // we brute force only the extents that can
                // overtake the current best motiflet for a given k
                for k in 1..self.max_k {
                    if exts[k] <= self.best_motiflet[k].0 {
                        to_brute_force[k].push((Reverse(exts[k]), *idx));
                    }
                }
            }
        }
        let num_to_brute_force = to_brute_force
            .iter()
            .map(|queue| queue.len())
            .sum::<usize>();
        if num_to_brute_force > 0 {
            eprintln!(
                "[{}@{}] Potential neighborhoods to resolve {}",
                self.rep, self.prefix, num_to_brute_force
            );
        }
        let minimum_brute_force = self.threads;
        let mut cnt_brute_forced = 0;
        let mut some_updated = false;
        for k in 1..self.max_k {
            if !self.best_motiflet[k].2 {
                while let Some((Reverse(extent_lower_bound), idx)) = to_brute_force[k].pop() {
                    if extent_lower_bound.0 > (self.best_motiflet[k].0).0 {
                        break;
                    }
                    let neighborhood = self.neighborhoods.get_mut(&idx).unwrap();
                    if neighborhood.is_evolving() {
                        cnt_brute_forced += 1;
                        eprintln!(
                            "[{}@{}] Brute forcing {} (lower bound@{} = {} <= {})",
                            self.rep,
                            self.prefix,
                            idx,
                            k,
                            extent_lower_bound.0,
                            (self.best_motiflet[k].0).0
                        );
                        neighborhood.brute_force(
                            ts,
                            &self.fft_data,
                            self.max_k,
                            &mut self.support_buffers,
                        );
                    }
                    neighborhood.extents(exclusion_zone, &mut exts);
                    // tentatively update all extents
                    for k in 1..self.max_k {
                        if !self.best_motiflet[k].2 && exts[k] <= self.best_motiflet[k].0 {
                            eprintln!("{} is updating at {} with {}", idx, k, exts[k]);
                            self.best_motiflet[k] = (exts[k], idx, false);
                            some_updated = true;
                        }
                    }
                }
            }
        }
        if cnt_brute_forced < minimum_brute_force {
            let target_brute_force = minimum_brute_force - cnt_brute_forced;
            let mut to_brute_force = BinaryHeap::new();

            for (idx, neighborhood) in self.neighborhoods.iter_mut() {
                if neighborhood.is_evolving() {
                    neighborhood.extents(exclusion_zone, &mut exts);
                    let k = 1;
                    to_brute_force.push((exts[k], *idx));
                }
                while to_brute_force.len() > target_brute_force {
                    to_brute_force.pop();
                }
            }

            while let Some((_, idx)) = to_brute_force.pop() {
                let neighborhood = self.neighborhoods.get_mut(&idx).unwrap();
                if neighborhood.is_evolving() {
                    cnt_brute_forced += 1;
                    // eprintln!("[{}@{}] Brute forcing {}", self.rep, self.prefix, idx,);
                    neighborhood.brute_force(
                        ts,
                        &self.fft_data,
                        self.max_k,
                        &mut self.support_buffers,
                    );
                }
                neighborhood.extents(exclusion_zone, &mut exts);
                // tentatively update all extents
                for k in 1..self.max_k {
                    if !self.best_motiflet[k].2 && exts[k] <= self.best_motiflet[k].0 {
                        // eprintln!("{} is updating at {} with {}", idx, k, exts[k]);
                        self.best_motiflet[k] = (exts[k], idx, false);
                        some_updated = true;
                    }
                }
            }
        }
        if cnt_brute_forced > 0 {
            eprintln!(
                "[{}@{}] Brute forced: {}",
                self.rep, self.prefix, cnt_brute_forced
            );
        }
        if some_updated {
            eprintln!(
                "[{}@{}] Updated motiflets: {:?}",
                self.rep, self.prefix, self.best_motiflet
            );
        }
    }

    /// adds to `self.to_return` the motiflets that can
    /// be confirmed in this iteration
    fn emit_confirmed(&mut self) {
        let exclusion_zone = self.exclusion_zone;
        let prefix = self.prefix;
        let previous_prefix = self.previous_prefix;
        let rep = self.rep;
        let mut failure_probabilities = vec![0.0; self.max_k];
        let mut min_to_replace = vec![0; self.max_k];
        for k in 1..self.max_k {
            let (extent, root_idx, emitted) = self.best_motiflet[k];
            if !emitted {
                min_to_replace[k] = self
                    .neighborhoods
                    .values_mut()
                    .filter_map(|neighborhood| {
                        if neighborhood.is_evolving() {
                            Some(neighborhood.count_larger_than(k, exclusion_zone, extent))
                        } else {
                            None
                        }
                    })
                    .min()
                    .unwrap_or(k);
                assert!(min_to_replace[k] > 0);
                let fp = self
                    .hasher
                    .failure_probability_independent(extent.0, rep, prefix, previous_prefix)
                    .powi(min_to_replace[k] as i32);
                failure_probabilities[k] = fp;
                if fp < self.delta {
                    eprintln!(
                        "[{}@{}] Failure probability for k={} is {} (extent {}, threshold {})",
                        rep, prefix, k, fp, extent.0, self.delta
                    );
                    let neighborhood = self.neighborhoods.get_mut(&root_idx).unwrap();
                    self.best_motiflet[k].2 = true;
                    let indices = neighborhood.neighbors(k);
                    let m = Motiflet::new(indices, extent.0);
                    self.to_return.push(m);
                }
            }
        }
        eprintln!("[{}@{}] min_to_replace {:?}", rep, prefix, min_to_replace);
        eprintln!(
            "[{}@{}] Failure probabilities {:?}",
            rep, prefix, failure_probabilities
        );
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
            self.resolve_most_promising();
            self.emit_confirmed();

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

    fn run_motiflet_test(ts: Arc<WindowedTimeseries>, k: usize, repetitions: usize, seed: u64) {
        let failure_probability = 0.01; // / (k as f64);
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
        // let result: Vec<Motiflet> = iter.collect();
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
            assert_eq!(motiflet_indices, ground_indices);
        }
    }

    #[test]
    fn test_ecg_motiflet_k2() {
        let ts: Vec<f64> = loadts("data/ECG.csv.gz", Some(20000)).unwrap();
        let ts = Arc::new(WindowedTimeseries::new(ts, 50, false));
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
    fn test_ecg_motiflet_k10() {
        let ts: Vec<f64> = loadts("data/ECG.csv.gz", Some(20000)).unwrap();
        let ts = Arc::new(WindowedTimeseries::new(ts, 50, false));
        run_motiflet_test(ts, 10, 8192, 123456);
    }
}
