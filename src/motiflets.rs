use crate::{
    distance::zeucl,
    knn::*,
    lsh::{self, ColumnBuffers, HashCollection, Hasher},
    timeseries::{FFTData, WindowedTimeseries},
};
use rayon::prelude::*;
use std::{collections::BinaryHeap, sync::Arc};
use std::{collections::HashMap, time::Instant};

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
    let n = ts.num_subsequences();

    let hasher_width = Hasher::estimate_width(&ts, &fft_data, 1, None, seed);

    let hasher = Arc::new(Hasher::new(ts.w, repetitions, hasher_width, seed));
    let pools = HashCollection::from_ts(&ts, &fft_data, Arc::clone(&hasher));
    let mut hash_buffers = ColumnBuffers::default();
    let mut support_buffers = SupportBuffers::new(ts);
    let mut pairs_buffer = vec![(0u32, 0u32, OrdF64(f64::INFINITY)); BUFFER_SIZE];

    let mut neighborhoods: HashMap<usize, SubsequenceNeighborhood> = HashMap::new();

    let mut brute_forced = 0;

    let mut last_extent = f64::INFINITY;

    let mut previous_prefix = None;
    let mut prefix = lsh::K;
    while prefix > 0 {
        eprintln!("=== {prefix}");
        for rep in 0..repetitions {
            pools.group_subsequences(prefix, rep, exclusion_zone, &mut hash_buffers, true);
            if let Some(mut enumerator) = hash_buffers.enumerator() {
                while let Some(cnt) = enumerator.next(&mut pairs_buffer, exclusion_zone) {
                    // Fixup the distances
                    pairs_buffer[0..cnt]
                        .par_iter_mut()
                        .with_min_len(1024)
                        .for_each(|(a, b, dist)| {
                            let a = *a as usize;
                            let b = *b as usize;
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
                    for (a, b, dist) in &pairs_buffer[0..cnt] {
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
                            "Returning motiflet {:?} with extent {} rooted at {} (brute forced {} over {})",
                            ids, extent.0, best_idx, brute_forced, n_neighborhoods
                        );
                        return (extent.0, ids);
                    }
                }
            }
        }
        previous_prefix.replace(prefix);

        if last_extent.is_finite() && prefix > 1 {
            while prefix > 0 {
                if (0..repetitions).any(|rep| {
                    hasher.failure_probability(last_extent, rep, prefix, previous_prefix)
                        <= target_failure_probability
                }) {
                    break;
                }
                prefix -= 1;
            }
            if prefix == 0 {
                prefix = 1;
            }
        } else {
            prefix -= 1;
        }
        eprintln!(
            "Next prefix is {}, where the failure probability will be {} in the first iteration",
            prefix,
            hasher.failure_probability(last_extent, 0, prefix, previous_prefix)
        );
    }

    unreachable!()
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
        let (_motiflet_extent, mut motiflet_indices) = if false {
            let motiflet = motiflets(ts, k, repetitions, failure_probability, seed)
                .first()
                .unwrap()
                .clone();
            (motiflet.extent(), motiflet.indices())
        } else {
            probabilistic_motiflets(&ts, k, exclusion_zone, repetitions, 0.01, 1234)
        };
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
