use crate::{
    distance::zeucl,
    knn::OrdF64,
    timeseries::{FFTData, WindowedTimeseries},
};
use rayon::prelude::*;

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
    assert_eq!(indices.len(), ts.num_subsequences());
    assert_eq!(distances.len(), ts.num_subsequences());
    assert_eq!(buf.len(), ts.w);
    ts.distance_profile(&fft_data, from, distances, buf);
    for i in 0..ts.num_subsequences() {
        indices[i] = i;
    }
    indices.sort_unstable_by_key(|j| OrdF64(distances[*j]));

    let mut ret = Vec::new();
    assert!(indices[0] == from);
    ret.push(indices[0]);
    let mut j = 1;
    while ret.len() < k {
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

    let mut extent = 0.0f64;
    for i in 0..k {
        for j in (i + 1)..k {
            let d = zeucl(ts, ret[i], ret[j]);
            extent = extent.max(d);
        }
    }

    (OrdF64(extent), ret)
}

pub fn brute_force_motiflets(
    ts: &WindowedTimeseries,
    k: usize,
    exclusion_zone: usize,
) -> (f64, Vec<usize>) {
    debug_assert!(false, "Should run only in `release mode`");
    let fft_data = FFTData::new(&ts);
    let n = ts.num_subsequences();

    let mut indices = Vec::new();
    indices.resize(n, 0usize);
    let mut distances = Vec::new();
    distances.resize(n, 0.0f64);
    let mut buf = Vec::new();
    buf.resize(ts.w, 0.0f64);

    let (extent, indices) = (0..n)
        .into_par_iter()
        .map_with((indices, distances, buf), |(indices, distances, buf), i| {
            k_nearest_neighbors_bf(ts, i, &fft_data, k, exclusion_zone, indices, distances, buf)
        })
        .min()
        .unwrap();
    (extent.0, indices)
}
