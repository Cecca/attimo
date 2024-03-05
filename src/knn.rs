use std::collections::BTreeSet;

use thread_local::ThreadLocal;

use crate::{
    distance::zeucl,
    timeseries::{Overlaps, WindowedTimeseries},
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

pub fn compute_extents(ts: &WindowedTimeseries, indices: &[usize]) -> Vec<Distance> {
    let k = indices.len();
    let mut extents = vec![Distance(0.0f64); indices.len()];
    for i in 1..k {
        extents[i] = extents[i - 1];
        for j in 0..i {
            let d = Distance(zeucl(ts, indices[i], indices[j]));
            extents[i] = extents[i].max(d);
        }
    }

    extents
}

/// this structure reports some statistics, mainly for debugging purposes
#[derive(Debug, Clone, Copy, Default)]
pub struct KnnGraphStats {
    pub total_neighbors: usize,
    pub max_neighbors: usize,
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
        use rayon::prelude::*;
        self.dirty
            .par_iter_mut()
            .zip(self.changed.par_iter_mut())
            .zip(self.neighborhoods.par_iter_mut())
            .for_each_with(
                ExclusionVec::new(self.exclusion_zone),
                |exclusion_vec, ((dirty, changed), neighborhood)| {
                    if !*dirty {
                        *changed = false;
                        return;
                    }
                    exclusion_vec.clear();
                    let mut i = 0;
                    let mut change = false;
                    while i < neighborhood.len() {
                        // is the subsequence shadowed because it overlaps with an earlier one?
                        let shadowed = exclusion_vec.overlaps(neighborhood[i].1);
                        if !shadowed {
                            exclusion_vec.insert(neighborhood[i].1);
                        }
                        // let shadowed =
                        //     neighborhood[i].overlaps(&neighborhood[..i], self.exclusion_zone);
                        // store if we are flipping the flag
                        change |= neighborhood[i].2 != !shadowed;
                        // possibly flip the flag
                        neighborhood[i].2 = !shadowed;
                        i += 1;
                    }
                    *changed = change;
                    *dirty = false;
                },
            );
    }

    pub fn extents(&self, idx: usize) -> &[Distance] {
        &self.extents[idx]
    }

    pub fn min_count_above_many(&self, thresholds: &[Distance]) -> Vec<usize> {
        use rayon::prelude::*;
        assert!(!self.neighborhoods.is_empty());
        let output = ThreadLocal::new(); // vec![0; thresholds.len()];
        let local_counts = vec![0; thresholds.len()];

        self.neighborhoods
            .par_iter()
            .for_each_with(local_counts, |local_counts, neighborhood| {
                let mut output = output
                    .get_or(|| std::cell::RefCell::new(vec![0usize; thresholds.len()]))
                    .borrow_mut();
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

        let mut output = output
            .into_iter()
            .map(|cell| cell.take())
            .reduce(|a, b| a.into_iter().zip(b).map(|(a, b)| a.max(b)).collect())
            .unwrap();

        for below in output.iter_mut() {
            assert!(*below <= self.max_k);
            *below = self.max_k - *below;
        }
        output
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
                            let d = Self::get_distance(neighborhoods, i, j)
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
            for (k, &ext) in extents.iter().enumerate() {
                if ext < minima[k].0 {
                    minima[k] = (ext, idx);
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
                debug_assert!(neighborhood.is_sorted_by_key(|tup| tup.0));

                dirty[src] = true;
            }
        }

        do_insert(
            self.exclusion_zone,
            &mut self.neighborhoods,
            &mut self.dirty,
            batch,
            |tup| (tup.0 as usize, tup.1 as usize),
        );

        do_insert(
            self.exclusion_zone,
            &mut self.neighborhoods,
            &mut self.dirty,
            batch,
            |tup| (tup.1 as usize, tup.0 as usize),
        );
    }
}

#[derive(Debug, Clone)]
struct ExclusionVec {
    exclusion_zone: usize,
    intervals: BTreeSet<usize>,
}

impl ExclusionVec {
    fn new(exclusion_zone: usize) -> Self {
        Self {
            exclusion_zone,
            intervals: Default::default(),
        }
    }

    fn insert(&mut self, x: usize) {
        self.intervals.insert(x);
    }

    /// returns true if the given x is withing `exclusion_zone` from any
    /// of the indices in this exclusion_vec.
    fn overlaps(&self, x: usize) -> bool {
        let start = if x < self.exclusion_zone {
            0
        } else {
            x - self.exclusion_zone
        };
        let end = x + self.exclusion_zone;
        self.intervals.range(start..=end).next().is_some()
    }

    fn clear(&mut self) {
        self.intervals.clear()
    }
}
