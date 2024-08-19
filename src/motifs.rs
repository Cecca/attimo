//// # Motifs

//// Finding motifs in time series. Instead of computing the full matrix profile,
//// leverage [LSH](src/lsh.html) to check only pairs that are probably near.
//// The data structure used for the task is adaptive to the data, and is configured
//// to respect the limits of the system in terms of memory.

use crate::knn::Distance;
use crate::motiflets::Motiflet;
use crate::timeseries::*;
use rayon::prelude::*;
use std::time::Instant;

/// This data structure stores information about a motif:
///
///  - The index of the two subsequences defining the motif
///  - The distance between the two subsequences
///  - The LSH collision probability two subsequences
///  - The elapsed time since the start of the algorithm until
///    when the motif was found
///
/// Some utility functions follow.
#[derive(Clone, Copy, Debug)]
pub struct Motif {
    pub idx_a: usize,
    pub idx_b: usize,
    pub distance: f64,
    pub confirmed: bool,
}

impl TryFrom<Motiflet> for Motif {
    type Error = &'static str;
    fn try_from(motiflet: Motiflet) -> Result<Self, Self::Error> {
        let indices = motiflet.indices();
        if indices.len() != 2 {
            Err("only motiflets of support 2 can be converted to motifs")
        } else {
            let idx_a = indices[0].min(indices[1]);
            let idx_b = indices[0].max(indices[1]);
            Ok(Motif {
                idx_a,
                idx_b,
                distance: motiflet.extent(),
                confirmed: true,
            })
        }
    }
}

impl Eq for Motif {}
impl PartialEq for Motif {
    fn eq(&self, other: &Self) -> bool {
        self.idx_a == other.idx_a && self.idx_b == other.idx_b && self.distance == other.distance
    }
}

impl Ord for Motif {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

#[allow(clippy::non_canonical_partial_ord_impl)]
impl PartialOrd for Motif {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.distance
            .partial_cmp(&other.distance)
            .map(|ord| ord.then_with(|| self.idx_a.cmp(&other.idx_a)))
    }
}

/// An important part of working with motifs is defining and removing
/// _trivial matches_. With the function `Motif::overlaps` we can detect
/// whether two motifs overlap according to the given `exclusion_zone`:
/// if any two indices in the two motifs are at distance less than
/// `exclusion_zone` from each other, then the motifs overlap and one of them
/// shall be discarded.
impl Overlaps<Motif> for Motif {
    /// Tells whether the two motifs overlap, in order to avoid storing trivial matches
    fn overlaps(&self, other: Self, exclusion_zone: usize) -> bool {
        let mut idxs = [self.idx_a, self.idx_b, other.idx_a, other.idx_b];
        idxs.sort_unstable();

        idxs[0] + exclusion_zone > idxs[1]
            || idxs[1] + exclusion_zone > idxs[2]
            || idxs[2] + exclusion_zone > idxs[3]
    }
}

fn nearest_neighbor_bf(
    ts: &WindowedTimeseries,
    from: usize,
    fft_data: &FFTData,
    exclusion_zone: usize,
    distances: &mut [f64],
    buf: &mut [f64],
) -> (Distance, usize) {
    // Check that the auxiliary memory buffers are correctly sized
    assert_eq!(distances.len(), ts.num_subsequences());
    assert_eq!(buf.len(), ts.w);

    // Compute the distance profile using the MASS algorithm
    ts.distance_profile(&fft_data, from, distances, buf);

    // Pick the nearest neighbor skipping overlapping subsequences
    let mut nearest = f64::INFINITY;
    let mut nearest_idx = 0;
    for (j, &d) in distances.iter().enumerate() {
        if !j.overlaps(from, exclusion_zone) && d < nearest {
            nearest = d;
            nearest_idx = j;
        }
    }
    (Distance(nearest), nearest_idx)
}

pub fn brute_force_motifs(ts: &WindowedTimeseries, k: usize, exclusion_zone: usize) -> Vec<Motif> {
    // pre-compute the FFT for the time series
    let fft_data = FFTData::new(&ts);
    let n = ts.num_subsequences();

    // initialize some auxiliary buffers, which will be cloned on a
    // per-thread basis.
    let mut distances = Vec::new();
    distances.resize(n, 0.0f64);
    let mut buf = Vec::new();
    buf.resize(ts.w, 0.0f64);

    // compute all k-nearest neighborhoods
    let mut nns: Vec<Motif> = (0..n)
        .into_par_iter()
        .map_with((distances, buf), |(distances, buf), i| {
            let (d, j) = nearest_neighbor_bf(ts, i, &fft_data, exclusion_zone, distances, buf);
            Motif {
                idx_a: i.min(j),
                idx_b: i.max(j),
                distance: d.0,
                confirmed: true,
            }
        })
        .collect();

    nns.sort_unstable();

    let mut res = Vec::new();
    let mut i = 0;
    while res.len() < k && i < nns.len() {
        if !nns[i].overlaps(res.as_slice(), exclusion_zone) {
            res.push(nns[i]);
        }
        i += 1;
    }

    res
}

/// This data structure implements a buffer, holding up to `k` sorted motifs,
/// such that no two motifs in the data structure are overlapping,
/// according to the parameter `exclusion_zone`.
#[derive(Clone)]
pub struct TopK {
    k: usize,
    exclusion_zone: usize,
    top: Vec<Motif>,
    current_non_overlapping: Vec<Motif>,
    should_update: bool,
}

impl std::fmt::Debug for TopK {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, m) in self.top.iter().enumerate() {
            writeln!(
                f,
                "  {} ::: {} -- {}  ({:.4})",
                i, m.idx_a, m.idx_b, m.distance
            )?;
        }

        Ok(())
    }
}

impl TopK {
    pub fn new(k: usize, exclusion_zone: usize) -> Self {
        Self {
            k,
            exclusion_zone,
            top: Vec::new(),
            current_non_overlapping: Vec::new(),
            should_update: false,
        }
    }

    /// When inserting into the data structure, we first check, in order of distance,
    /// if there is a pair whose defining motif is closer than the one being inserted,
    /// and which is also overlapping.
    pub fn insert(&mut self, motif: Motif) {
        let mut i = 0;
        while i < self.top.len() && self.top[i].distance <= motif.distance {
            i += 1;
        }
        self.top.insert(i, motif);

        debug_assert!(self.top.is_sorted());

        self.cleanup();
        assert!(self.top.len() <= (self.k + 1) * (self.k + 1));
        self.should_update = true;
    }

    fn cleanup(&mut self) {
        let k = self.k;
        let mut i = 0;
        while i < self.top.len() {
            if overlap_count(&self.top[i], &self.top[..i], self.exclusion_zone) >= k {
                self.top.remove(i);
            } else {
                i += 1;
            }
        }
    }

    fn update_non_overlapping(&mut self) {
        if !self.should_update {
            return;
        }
        self.current_non_overlapping.clear();
        for i in 0..self.top.len() {
            if !self.top[i].overlaps(self.current_non_overlapping.as_slice(), self.exclusion_zone) {
                self.current_non_overlapping.push(self.top[i]);
            }
        }
        self.should_update = false;
    }

    /// This function is used to access the k-th motif, if
    /// we already found it, even if not confirmed yet
    pub fn k_th(&mut self) -> Option<Motif> {
        self.update_non_overlapping();
        let current = &self.current_non_overlapping;
        if current.len() == self.k {
            current.last().map(|mot| *mot)
        } else {
            None
        }
    }

    pub fn first_not_confirmed(&mut self) -> Option<Motif> {
        self.update_non_overlapping();
        self.current_non_overlapping
            .iter()
            .find(|m| m.confirmed)
            .copied()
    }

    pub fn last_confirmed(&mut self) -> Option<Motif> {
        self.update_non_overlapping();
        self.current_non_overlapping
            .iter()
            .filter(|m| m.confirmed)
            .last()
            .copied()
    }

    pub fn num_confirmed(&self) -> usize {
        self.confirmed().count()
    }

    pub fn confirmed(&self) -> impl Iterator<Item = Motif> + '_ {
        self.top.iter().filter(|m| m.confirmed).copied()
    }

    pub fn for_each(&mut self, f: impl FnMut(&mut Motif)) {
        self.top.iter_mut().for_each(f)
    }

    pub fn len(&self) -> usize {
        self.top.len()
    }

    pub fn to_vec(&mut self) -> Vec<Motif> {
        self.update_non_overlapping();
        self.current_non_overlapping.clone()
    }

    pub fn add_all(&mut self, other: &mut TopK) {
        for m in other.top.drain(..) {
            self.insert(m);
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::motiflets::MotifletsIterator;
    use crate::{load::loadts, timeseries::WindowedTimeseries};
    use std::sync::Arc;

    fn run_motif_test(
        ts: Arc<WindowedTimeseries>,
        k: usize,
        repetitions: usize,
        seed: u64,
        ground_truth: Option<Vec<(usize, usize, f64)>>,
    ) {
        let failure_probability = 0.01;
        let exclusion_zone = ts.w;
        let ground_truth: Vec<(usize, usize, f64)> = ground_truth.unwrap_or_else(|| {
            eprintln!(
                "Running brute force algorithm on {} subsequences",
                ts.num_subsequences()
            );
            brute_force_motifs(&ts, k, exclusion_zone)
                .into_iter()
                .map(|m| (m.idx_a, m.idx_b, m.distance))
                .collect()
        });
        dbg!(&ground_truth);
        let iter = MotifletsIterator::new(
            ts,
            2,
            k,
            crate::allocator::Bytes::gbytes(2),
            failure_probability,
            exclusion_zone,
            seed,
            false,
        )
        .map(|motiflet| Motif::try_from(motiflet).unwrap());
        let motifs: Vec<Motif> = iter.collect();
        assert_eq!(motifs.len(), k);
        let mut cnt = 0;
        for m in &motifs {
            println!("{:?}", m);
            if m.distance <= ground_truth.last().unwrap().2 + 0.0000001 {
                cnt += 1;
            }
        }
        assert_eq!(cnt, k);
    }

    #[test]
    fn test_ecg() {
        let ts: Vec<f64> = loadts("data/ECG.csv.gz", Some(10000)).unwrap();
        let ts = Arc::new(WindowedTimeseries::new(ts, 100, false));
        run_motif_test(ts, 10, 512, 12345, None);

        let ts: Vec<f64> = loadts("data/ECG.csv.gz", Some(10000)).unwrap();
        let ts = Arc::new(WindowedTimeseries::new(ts, 200, false));
        run_motif_test(ts, 10, 512, 12345, None);
    }
}
