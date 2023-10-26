use crate::{
    distance::zeucl,
    knn::OrdF64,
    timeseries::{FFTData, WindowedTimeseries},
};
use rayon::prelude::*;

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
    let (extent, indices) = (0..n)
        .into_par_iter()
        .map_with((indices, distances, buf), |(indices, distances, buf), i| {
            k_nearest_neighbors_bf(ts, i, &fft_data, k, exclusion_zone, indices, distances, buf)
        })
        .min()
        .unwrap();
    (extent.0, indices)
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{load::loadts, motifs::motiflets};
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
        let (_ground_dist, mut ground_indices): (f64, Vec<usize>) =
            ground_truth.unwrap_or_else(|| {
                eprintln!(
                    "Running brute force algorithm on {} subsequences",
                    ts.num_subsequences()
                );
                brute_force_motiflets(&ts, k, exclusion_zone)
            });
        ground_indices.sort();
        let motiflet = motiflets(ts, k, repetitions, failure_probability, seed)
            .first()
            .unwrap()
            .clone();
        let mut motiflet_indices = motiflet.indices();
        motiflet_indices.sort();
        assert_eq!(motiflet_indices, ground_indices);
    }

    #[test]
    #[ignore]
    fn test_ecg_motiflet() {
        let ts: Vec<f64> = loadts("data/ECG.csv.gz", Some(10000)).unwrap();
        let ts = Arc::new(WindowedTimeseries::new(ts, 100, false));
        run_motiflet_test(ts, 10, 1024, 12345, None);
    }
}
