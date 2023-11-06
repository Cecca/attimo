use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::{cell::RefCell, collections::BTreeMap, ops::Range, sync::Arc, time::Instant};
use thread_local::ThreadLocal;

use crate::{
    distance::zeucl,
    lsh::{ColumnBuffers, HashCollection, HashValue, Hasher},
    timeseries::{overlap_count, FFTData, Overlaps, WindowedTimeseries},
};

#[derive(PartialEq, PartialOrd, Clone, Copy, Debug)]
// TODO: turn this struct into a `Distance` struct with [From] and [Into] impls for `f64`
pub struct OrdF64(pub f64);
impl Eq for OrdF64 {}
impl Ord for OrdF64 {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap()
    }
}

impl Overlaps<(OrdF64, usize)> for (OrdF64, usize) {
    fn overlaps(&self, other: (OrdF64, usize), exclusion_zone: usize) -> bool {
        self.1.overlaps(other.1, exclusion_zone)
    }
}

/// A triple where the third element denotes whether the corresponding tuple is "active"
/// overlaps with another one only if the other one is active and their indices overlap
impl Overlaps<(OrdF64, usize, bool)> for (OrdF64, usize, bool) {
    fn overlaps(&self, other: (OrdF64, usize, bool), exclusion_zone: usize) -> bool {
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

#[derive(Debug)]
pub struct EvolvingNeighborhood {
    subsequence: usize,
    max_k: usize,
    /// When true, requires to update the flags in the [neighbors] vector.
    dirty: bool,
    /// For each neighbor we consider its distance, its index and whether it is currently
    /// part of the k-nearest neighbors, i.e. if it overlaps with any othe preceding
    /// selected neighbors
    neighbors: Vec<(OrdF64, usize, bool)>,
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
    fn update(&mut self, dist: OrdF64, neigh: usize) {
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
        while i < self.neighbors.len() {
            if !self.neighbors[i].overlaps(&self.neighbors[..i], exclusion_zone) {
                self.neighbors[i].2 = true;
            }
            i += 1;
        }
        self.dirty = false;
    }
    /// Returns an upper bound to the extent of this neighborhood
    fn extent(&mut self, k: usize, exclusion_zone: usize) -> OrdF64 {
        self.clean(exclusion_zone);
        let ext = self
            .neighbors
            .iter()
            .filter_map(|(d, _, is_neighbor)| {
                if *is_neighbor {
                    Some(OrdF64(2.0 * d.0))
                } else {
                    None
                }
            })
            .nth(k - 1)
            .unwrap_or(OrdF64(f64::INFINITY));
        ext
    }
    fn extents(&mut self, exclusion_zone: usize, out: &mut [OrdF64]) {
        assert_eq!(out.len(), self.max_k + 1);
        self.clean(exclusion_zone);

        out[0] = OrdF64(0.0);
        for (i, (d, _, included)) in self.neighbors.iter().filter(|tup| tup.2).enumerate() {
            if *included {
                out[i + 1] = OrdF64(d.0 * 2.0);
            }
        }
    }
    /// Counts how many neighbors have a distance _strictly_ larger than the given distance
    fn count_larger_than(&mut self, k: usize, exclusion_zone: usize, d: OrdF64) -> usize {
        self.clean(exclusion_zone);
        let cnt_smaller = self
            .neighbors
            .iter()
            .filter_map(
                |(d, _, is_neighbor)| {
                    if *is_neighbor {
                        Some(2.0 * d.0)
                    } else {
                        None
                    }
                },
            )
            .take(k)
            .filter(|nn_dist| *nn_dist <= d.0)
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
        extents: Vec<OrdF64>,
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
        // let n_candidates = (k * exclusion_zone).min(ts.num_subsequences());
        // indices.select_nth_unstable_by_key(n_candidates, |j| OrdF64(distances[*j]));

        // Sort the candidate indices by increasing distance (the previous step)
        // only partitioned the indices in two groups with the guarantee that the first
        // `n_candidates` indices are the ones at shortest distance from the `from` point,
        // but they are not guaranteed to be sorted
        // let indices = &mut indices[..n_candidates];
        indices.sort_unstable_by_key(|j| OrdF64(distances[*j]));

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

        let mut extents = vec![OrdF64(0.0); k + 1];
        for i in 0..=k {
            let mut extent = 0.0f64;
            for j in 0..i {
                let d = zeucl(ts, ids[i], ids[j]);
                extent = extent.max(d);
            }
            extents[i] = OrdF64(extent).max(*extents[0..i].iter().max().unwrap_or(&OrdF64(0.0)));
        }
        assert_eq!(extents.len(), k + 1);

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
    pub fn extents(&mut self, exclusion_zone: usize, out: &mut [OrdF64]) {
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
    pub fn extent(&mut self, k: usize, exclusion_zone: usize) -> OrdF64 {
        match self {
            Self::Evolving { neighborhood } => neighborhood.extent(k, exclusion_zone),
            Self::Exact {
                subsequence: _,
                extents,
                ids: _,
            } => extents[k],
            Self::Discarded { subsequence: _ } => OrdF64(std::f64::INFINITY),
        }
    }
    pub fn neighbors(&mut self, k: usize) -> Vec<usize> {
        match self {
            Self::Exact {
                subsequence: _,
                extents: _,
                ids,
            } => ids[..k].to_vec(),
            _ => panic!(),
        }
    }
    pub fn update(&mut self, dist: OrdF64, neigh: usize) {
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
        extent_lower_bound: OrdF64,
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
