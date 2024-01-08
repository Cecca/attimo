use crate::{
    distance::zeucl,
    timeseries::{FFTData, Overlaps, WindowedTimeseries},
};

#[derive(PartialEq, PartialOrd, Clone, Copy)]
pub struct Distance(pub f64);
impl Eq for Distance {}
impl Ord for Distance {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap()
    }
}

impl Distance {
    pub fn infinity() -> Self {
        Self(f64::INFINITY)
    }
    pub fn is_finite(&self) -> bool {
        self.0.is_finite()
    }
    pub fn is_infinite(&self) -> bool {
        self.0.is_infinite()
    }
}

impl From<f64> for Distance {
    fn from(value: f64) -> Self {
        Self(value)
    }
}
impl Into<f64> for Distance {
    fn into(self) -> f64 {
        self.0
    }
}

impl std::fmt::Debug for Distance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::fmt::Display for Distance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Overlaps<(Distance, usize)> for (Distance, usize) {
    fn overlaps(&self, other: (Distance, usize), exclusion_zone: usize) -> bool {
        self.1.overlaps(other.1, exclusion_zone)
    }
}

/// A triple where the third element denotes whether the corresponding tuple is "active"
/// overlaps with another one only if the other one is active and their indices overlap
impl Overlaps<(Distance, usize, bool)> for (Distance, usize, bool) {
    fn overlaps(&self, other: (Distance, usize, bool), exclusion_zone: usize) -> bool {
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

/// An iterator over the elements of a neighborhood that are not shadowed by others
struct ActiveNeighborhood<'neighs> {
    i: usize,
    neighborhood: &'neighs [(Distance, usize, bool)],
}
impl<'neighs> ActiveNeighborhood<'neighs> {
    fn new(neighborhood: &'neighs [(Distance, usize, bool)]) -> Self {
        Self { i: 0, neighborhood }
    }
}
impl<'neighs> Iterator for ActiveNeighborhood<'neighs> {
    type Item = (Distance, usize);
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        while self.i < self.neighborhood.len() && !self.neighborhood[self.i].2 {
            self.i += 1;
        }
        if self.i < self.neighborhood.len() {
            let (dist, idx, _) = self.neighborhood[self.i];
            self.i += 1;
            Some((dist, idx))
        } else {
            None
        }
    }
}

/// Compute the extent of the given set of indices of the time series.
pub fn compute_extent(ts: &WindowedTimeseries, indices: &[usize]) -> Distance {
    let k = indices.len();
    let mut extent = 0.0f64;
    for i in 0..k {
        for j in (i + 1)..k {
            let d = zeucl(ts, indices[i], indices[j]);
            extent = extent.max(d);
        }
    }

    Distance(extent)
}

/// this structure reports some statistics, mainly for debugging purposes
#[derive(Debug, Clone, Copy, Default)]
pub struct KnnGraphStats {
    total_neighbors: usize,
    max_neighbors: usize,
}

pub struct KnnGraph {
    max_k: usize,
    exclusion_zone: usize,
    neighborhoods: Vec<Vec<(Distance, usize, bool)>>,
    extents: Vec<Vec<Distance>>,
    dirty: Vec<bool>,
    changed: Vec<bool>,
}

impl KnnGraph {
    pub fn new(max_k: usize, n: usize, exclusion_zone: usize) -> Self {
        Self {
            max_k,
            exclusion_zone,
            neighborhoods: vec![Vec::new(); n],
            dirty: vec![false; n],
            extents: vec![Vec::new(); n],
            changed: vec![false; n],
        }
    }

    pub fn stats(&self) -> KnnGraphStats {
        let mut stats = KnnGraphStats::default();

        for neighborhood in &self.neighborhoods {
            let n_neighs = neighborhood.len();
            stats.total_neighbors += n_neighs;
            stats.max_neighbors = stats.max_neighbors.max(n_neighs);
        }

        stats
    }

    fn get_distance(
        neighborhoods: &[Vec<(Distance, usize, bool)>],
        from: usize,
        to: usize,
    ) -> Option<Distance> {
        neighborhoods[from]
            .iter()
            .find(|tup| tup.1 == to)
            .map(|tup| tup.0)
    }

    pub fn extent(&self, idx: usize, k: usize) -> Distance {
        assert!(!self.dirty[idx], "you should first call `update_extents`");
        self.extents[idx][k]
    }

    /// Mark the neighbors that are not overlapped by others
    fn fix_flags(&mut self) {
        for idx in 0..self.dirty.len() {
            if !self.dirty[idx] {
                self.changed[idx] = false;
                continue;
            }
            let neighborhood = &mut self.neighborhoods[idx];
            // for tup in neighborhood.iter_mut() {
            //     tup.2 = false;
            // }
            let mut i = 0;
            let mut changed = false;
            while i < neighborhood.len() {
                // is the subsequence shadowed because it overlaps with an earlier one?
                let shadowed = neighborhood[i].overlaps(&neighborhood[..i], self.exclusion_zone);
                // store if we are flipping the flag
                changed |= neighborhood[i].2 != !shadowed;
                // possibly flip the flag
                neighborhood[i].2 = !shadowed;
                i += 1;
            }
            self.changed[idx] = changed;
            self.dirty[idx] = false;
        }
    }

    pub fn num_non_empty(&self) -> usize {
        self.neighborhoods.len()
    }

    /// Return the maximum distance of the k-th neighbor in
    /// this graph.
    pub fn farthest_kth(&self) -> Option<Distance> {
        self.neighborhoods
            .iter()
            .filter_map(|nn| {
                let mut active = ActiveNeighborhood::new(&nn);
                active.nth(self.max_k).map(|tup| tup.0)
            })
            .max()
    }

    pub fn extents(&self, idx: usize) -> &[Distance] {
        &self.extents[idx]
    }

    pub fn min_count_above_many(&self, thresholds: &[Distance]) -> Vec<usize> {
        assert!(!self.neighborhoods.is_empty());
        let mut output = vec![0; thresholds.len()];
        let mut local_counts = vec![0; thresholds.len()];

        self.neighborhoods.iter().for_each(|neighborhood| {
            local_counts.fill(0);
            for (i, tup) in ActiveNeighborhood::new(neighborhood)
                .take(self.max_k - 1)
                .enumerate()
            {
                if tup.0 <= thresholds[i] {
                    local_counts[i] += 1;
                }
            }

            for (below, cnt) in output.iter_mut().zip(local_counts.iter()) {
                if cnt > below {
                    *below = *cnt;
                }
            }
        });

        for below in output.iter_mut() {
            assert!(*below <= self.max_k);
            *below = self.max_k - *below;
        }
        output
    }

    pub fn min_count_above(&self, threshold: Distance) -> usize {
        assert!(!self.neighborhoods.is_empty());
        let below = self
            .neighborhoods
            .iter()
            .map(|neighborhood| {
                let active = ActiveNeighborhood::new(neighborhood);
                let cnt = active
                    .filter(|tup| tup.0 <= threshold)
                    .take(self.max_k - 1)
                    .count();
                cnt
            })
            .max()
            .unwrap();
        assert!(below <= self.max_k);
        self.max_k - below
    }

    pub fn update_extents(&mut self, ts: &WindowedTimeseries) {
        use rayon::prelude::*;
        self.fix_flags();

        let max_k = self.max_k;
        let neighborhoods = &self.neighborhoods;

        // now we can compute the extents
        self.extents
            .par_iter_mut()
            .zip(neighborhoods)
            .zip(&self.changed)
            .enumerate()
            .for_each(|(_subsequence_idx, ((extents, neighborhood), changed))| {
                if !neighborhood.is_empty() && *changed {
                    extents.resize(max_k, Distance::infinity());
                    extents.fill(Distance::infinity());
                    assert_eq!(extents.len(), max_k);
                    for (k, (dist, i)) in ActiveNeighborhood::new(neighborhood)
                        .enumerate()
                        .take(max_k)
                    {
                        if k == 0 {
                            extents[k] = dist;
                        } else {
                            extents[k] = dist.max(extents[k - 1]);
                        }
                        for (_, j) in ActiveNeighborhood::new(neighborhood).take(k) {
                            let d = Self::get_distance(&neighborhoods, i, j)
                                .unwrap_or_else(|| Distance(zeucl(ts, i, j)));
                            extents[k] = extents[k].max(d);
                        }
                    }
                    assert!(extents.is_sorted());
                }
            });
    }

    pub fn min_extents(&self) -> Vec<(Distance, usize)> {
        let mut minima = vec![(Distance::infinity(), 0); self.max_k];

        for (idx, extents) in self.extents.iter().enumerate() {
            for k in 0..extents.len() {
                if extents[k] < minima[k].0 {
                    minima[k] = (extents[k], idx);
                }
            }
        }
        for ext in &minima {
            assert!(
                ext.0 > Distance(0.0),
                "Got a 0 extent for {}, whose neighbors are {:?}",
                ext.1,
                self.neighborhoods[ext.1]
            );
        }
        minima
    }

    pub fn get(&self, idx: usize, k: usize) -> Vec<usize> {
        let mut indices: Vec<usize> = ActiveNeighborhood::new(&self.neighborhoods[idx])
            .map(|tup| tup.1)
            .take(k + 1)
            .collect();
        indices.push(idx);
        // check that the indices are all distinct
        indices.sort();
        for i in 1..indices.len() {
            assert_ne!(indices[i - 1], indices[i]);
        }
        indices
    }

    pub fn update_batch(&mut self, batch: &mut [(u32, u32, Distance)]) {
        fn do_insert<K: Fn(&(u32, u32, Distance)) -> (usize, usize)>(
            exclusion_zone: usize,
            neighborhoods: &mut [Vec<(Distance, usize, bool)>],
            dirty: &mut [bool],
            batch: &[(u32, u32, Distance)],
            extract: K,
        ) {
            // TODO: batch accesses to the same neighborhood
            for tup in batch {
                let (src, dst) = extract(tup);
                let d = tup.2;
                if d.0.is_infinite() || src.overlaps(dst, exclusion_zone) {
                    continue;
                }
                // find the place in the neighborhood of `src` to insert `dst`, by distance
                let neighborhood = &mut neighborhoods[src];
                let mut i = 0;
                while i < neighborhood.len() && neighborhood[i].0 < d {
                    i += 1;
                }
                // TODO: also skip inserting if the distance is too large
                if i < neighborhood.len() && neighborhood[i].1 == dst {
                    // continue with the next tuple, the element is already inserted
                    continue;
                }
                neighborhood.insert(i, (d, dst, false));
                assert!(neighborhood.is_sorted_by_key(|tup| tup.0));

                dirty[src] = true;
            }
        }

        do_insert(
            self.exclusion_zone,
            &mut self.neighborhoods,
            &mut self.dirty,
            &batch,
            |tup| (tup.0 as usize, tup.1 as usize),
        );

        do_insert(
            self.exclusion_zone,
            &mut self.neighborhoods,
            &mut self.dirty,
            &batch,
            |tup| (tup.1 as usize, tup.0 as usize),
        );
    }
}

#[derive(Debug)]
pub struct EvolvingNeighborhood {
    subsequence: usize,
    max_k: usize,
    /// When true, requires to update the flags in the [neighbors] vector.
    dirty: bool,
    /// For each neighbor we consider its distance, its index and whether it is currently
    /// part of the k-nearest neighbors, i.e. if it overlaps with any othe preceding
    /// selected neighbors
    neighbors: Vec<(Distance, usize, bool)>,
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
    fn update(&mut self, dist: Distance, neigh: usize) {
        if let Some(kth_dist) = self
            .neighbors
            .iter()
            .filter_map(|tup| if tup.2 { Some(tup.0) } else { None })
            .nth(self.max_k)
        {
            if dist > kth_dist {
                // no point in adding this distance, it would
                // be removed on the next call of `clean`
                return;
            }
        }
        self.dirty = true;
        let tuple = (dist, neigh, false);
        let mut i = 0;
        while i < self.neighbors.len() && tuple.0 > self.neighbors[i].0 {
            i += 1;
        }
        if i >= self.neighbors.len() || self.neighbors[i].1 != neigh {
            self.neighbors.insert(i, tuple);
        }
    }
    fn clean(&mut self, exclusion_zone: usize) {
        if !self.dirty {
            return;
        }
        for tup in self.neighbors.iter_mut() {
            tup.2 = false;
        }
        let mut i = 0;
        let mut cnt_neighbors = 0;
        let mut farthest_idx = 0;
        while i < self.neighbors.len() {
            if !self.neighbors[i].overlaps(&self.neighbors[..i], exclusion_zone) {
                self.neighbors[i].2 = true;
                cnt_neighbors += 1;
                farthest_idx = i;
            }
            i += 1;
        }
        if cnt_neighbors == self.max_k {
            // Remove all the subsequences after the k-th nearest neighbor
            self.neighbors.truncate(farthest_idx + 1);
        }
        self.dirty = false;
    }
    /// Returns an upper bound to the extent of this neighborhood
    fn extent(&mut self, k: usize, exclusion_zone: usize) -> Distance {
        self.clean(exclusion_zone);
        let ext = self
            .neighbors
            .iter()
            .filter_map(|(d, _, is_neighbor)| {
                if *is_neighbor {
                    Some(Distance(2.0 * d.0))
                } else {
                    None
                }
            })
            // offset by one because we are not
            // storing the subsequence itself in the
            // neighbors array
            .nth(k - 1)
            .unwrap_or(Distance(f64::INFINITY));
        ext
    }
    fn extents(&mut self, exclusion_zone: usize, out: &mut [Distance]) {
        assert_eq!(out.len(), self.max_k);
        self.clean(exclusion_zone);

        let n = self.neighbors.len();
        out.fill(Distance(std::f64::INFINITY));
        out[0] = Distance(0.0);
        let mut i = 1;
        let mut nn = 0;
        while i < self.max_k && nn < n {
            let (d, _, included) = self.neighbors[nn];
            if included {
                out[i] = d;
                i += 1;
            }
            nn += 1;
        }
    }
    /// Counts how many neighbors have a distance _strictly_ larger than the given distance
    fn count_larger_than(&mut self, k: usize, exclusion_zone: usize, d: Distance) -> usize {
        self.clean(exclusion_zone);
        let cnt_smaller = self
            .neighbors
            .iter()
            .filter_map(|(d, _, is_neighbor)| if *is_neighbor { Some(*d) } else { None })
            .take(k)
            .filter(|nn_dist| *nn_dist <= d)
            .count();
        k - cnt_smaller
    }
}

#[derive(Debug)]
pub enum SubsequenceNeighborhood {
    /// This subsequence neighborhood has been "brute forced", i.e.
    /// it has been computed exactly. It ignores any proposed update.
    Exact {
        subsequence: usize,
        extents: Vec<Distance>,
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
            neighborhood: EvolvingNeighborhood::new(subsequence, max_k),
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
        let n_candidates = (2 * k * exclusion_zone).min(ts.num_subsequences());
        indices.select_nth_unstable_by_key(n_candidates, |j| Distance(distances[*j]));

        // Sort the candidate indices by increasing distance (the previous step)
        // only partitioned the indices in two groups with the guarantee that the first
        // `n_candidates` indices are the ones at shortest distance from the `from` point,
        // but they are not guaranteed to be sorted
        let indices = &mut indices[..n_candidates];
        indices.sort_unstable_by_key(|j| Distance(distances[*j]));

        // Pick the k-neighborhood skipping overlapping subsequences
        let mut ids = Vec::new();
        ids.push(subsequence);
        let mut j = 1;
        while ids.len() <= k && j < indices.len() {
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
        assert_eq!(ids.len(), k + 1);

        let mut extents = vec![Distance(0.0); k];
        for i in 0..k {
            let mut extent = 0.0f64;
            for j in 0..i {
                let d = zeucl(ts, ids[i], ids[j]);
                extent = extent.max(d);
            }
            extents[i] =
                Distance(extent).max(*extents[0..i].iter().max().unwrap_or(&Distance(0.0)));
        }
        assert_eq!(extents.len(), k);

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
    pub fn is_evolving(&self) -> bool {
        match self {
            Self::Evolving { neighborhood: _ } => true,
            _ => false,
        }
    }
    pub fn count_larger_than(&mut self, k: usize, exclusion_zone: usize, d: Distance) -> usize {
        match self {
            Self::Evolving { neighborhood } => neighborhood.count_larger_than(k, exclusion_zone, d),
            _ => unreachable!(),
        }
    }
    pub fn extents(&mut self, exclusion_zone: usize, out: &mut [Distance]) {
        match self {
            Self::Evolving { neighborhood } => {
                neighborhood.extents(exclusion_zone, out);
            }
            Self::Exact {
                subsequence: _,
                extents,
                ids: _,
            } => out.copy_from_slice(extents),
            Self::Discarded { subsequence: _ } => unreachable!(),
        }
    }
    pub fn extent(&mut self, k: usize, exclusion_zone: usize) -> Distance {
        match self {
            Self::Evolving { neighborhood } => neighborhood.extent(k, exclusion_zone),
            Self::Exact {
                subsequence: _,
                extents,
                ids: _,
            } => extents[k],
            Self::Discarded { subsequence: _ } => Distance(std::f64::INFINITY),
        }
    }
    pub fn neighbors(&mut self, k: usize) -> Vec<usize> {
        match self {
            Self::Exact {
                subsequence: _,
                extents: _,
                ids,
            } => ids[..=k].to_vec(),
            _ => panic!(),
        }
    }
    pub fn update(&mut self, dist: Distance, neigh: usize) {
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
        extent_lower_bound: Distance,
        lower_bound_fp: f64,
        exclusion_zone: usize,
    ) -> f64 {
        match self {
            Self::Evolving { neighborhood } => {
                let h = neighborhood.count_larger_than(k, exclusion_zone, extent_lower_bound);
                if h == 0 {
                    return 0.0;
                }
                assert!(h <= k);
                let ret = lower_bound_fp.powi(h as i32);
                ret
            }
            Self::Exact {
                subsequence: _,
                extents: _,
                ids: _,
            } => 0.0,
            Self::Discarded { subsequence: _ } => 0.0,
        }
    }
}
