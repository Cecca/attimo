// Computation of the approximate matrix profile, as a 1-NN join based on LSH.
// The algorithm is inspired by [PUFFINN [Aumüller et al. 2019]](https://arxiv.org/pdf/1906.12211.pdf),
// which is a LSH adaptive nearest neighbor algorithm.

use crate::distance::*;
use crate::lsh::*;
use crate::types::*;
use indicatif::ProgressBar;
use indicatif::ProgressStyle;
use slog_scope::info;
use std::ops::Range;
use std::time::Instant;
use bumpalo::Bump;

pub fn approx_mp(
    ts: &WindowedTimeseries,
    k: usize,
    repetitions: usize,
    delta: f64,
    seed: u64,
) -> Vec<(f64, usize)> {
    let start = Instant::now();

    let exclusion_zone = ts.w / 4;
    info!("Approximate Matrix Profile setup"; "k" => k, "repetitions" => repetitions, "delta" => delta, "seed" => seed, "exclusion_zone" => exclusion_zone);

    // FIXME find a good width parameter
    let hasher_width = 10.0;
    let hasher = Hasher::new(ts.w, k, repetitions, hasher_width, seed);
    let pools = HashCollection::from_ts(&ts, &hasher);
    println!(
        "[{:?}] Computed hash pools, taking {}",
        start.elapsed(),
        pools.bytes_size()
    );
    let hashes_arena = Bump::new();
    let hashes = pools.get_hash_matrix(&hashes_arena);
    println!(
        "[{:?}] Computed hash matrix, taking {}",
        start.elapsed(),
        hashes.bytes_size()
    );

    // Define upper and lower bounds, to avoid repeating already-done comparisons
    // We have a range of already examined hash indices for each element and repetition
    let mut bounds: Vec<Vec<Range<usize>>> = vec![vec![0..0; ts.num_subsequences()]; repetitions];

    // keep track of active subsequences
    let mut active: Vec<bool> = vec![true; ts.num_subsequences()];
    // These two array of counters is just for inspection
    let mut count_evaluations = vec![0usize; ts.num_subsequences()];
    let mut thresholds = vec![usize::MAX; ts.num_subsequences()];

    // In this vector we will hold the solution: we have an entry for each subsequence, initialized
    // to the empty value, and eventually replaced with a pair whose first element is the
    // distance to the (estimated) nearest neighbor, and the second element is the index
    // of the nearest neigbor itself.
    let mut nearest_neighbor: Vec<Option<(f64, usize)>> = vec![None; ts.num_subsequences()];

    let mut cnt_dist = 0;

    // for decreasing depths
    for depth in (0..=k).rev() {
        let n_active = active.iter().filter(|a| **a).count();
        info!(""; "depth" => depth, "active" => n_active);
        if n_active == 0 {
            break;
        }
        let pbar = ProgressBar::new(repetitions as u64);
        pbar.set_draw_rate(4);
        pbar.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {msg} {bar:40.cyan/blue} {pos:>7}/{len:7}"),
        );

        for rep in 0..repetitions {
            let mut rep_cnt_dists = 0;
            pbar.set_message(format!(
                "depth {}, active {}",
                depth,
                active.iter().filter(|a| **a).count()
            ));
            for (hash_range, bucket) in hashes.buckets(depth, rep) {
                for (a_offset, &(_, a_idx)) in bucket.iter().enumerate() {
                    if active[a_idx] {
                        let a_already_checked = &bounds[rep][a_idx];
                        let a_hash_idx = hash_range.start + a_offset;
                        for (b_offset, &(_, b_idx)) in bucket.iter().enumerate() {
                            // Here we handle trivial matches: we don't consider a pair if the difference between
                            // the subsequence indexes is smaller than the exclusion zone, which is set to `w/4`.
                            if a_idx + exclusion_zone < b_idx {
                                let b_hash_idx = hash_range.start + b_offset;
                                let b_already_checked = &bounds[rep][b_idx];
                                let check_a = !a_already_checked.contains(&b_hash_idx);
                                let check_b = !b_already_checked.contains(&a_hash_idx);
                                if check_a || check_b {
                                    // We only process the pair if this is the first repetition in which
                                    // they collide. We get this information from the pool of bits
                                    // from which hash values for all repetitions are extracted.
                                    let first_colliding_repetition: usize = pools
                                        .first_collision(a_idx, b_idx, depth)
                                        .expect("hashes must collide in buckets");
                                    if first_colliding_repetition == rep {
                                        // After computing the distance between the two subsequences,
                                        // we set `b` as the nearest neigbor of `a`, if it is closer
                                        // than the previous candidate.
                                        let d = zeucl(ts, a_idx, b_idx);
                                        cnt_dist += 1;
                                        rep_cnt_dists += 1;
                                        count_evaluations[a_idx] += 1;
                                        count_evaluations[b_idx] += 1;
                                        if nearest_neighbor[a_idx].is_none()
                                            || d < nearest_neighbor[a_idx].unwrap().0
                                        {
                                            nearest_neighbor[a_idx] = Some((d, b_idx));
                                        }
                                        // Similarly, set `a` as the nearest neighbor of `b`.
                                        if nearest_neighbor[b_idx].is_none()
                                            || d < nearest_neighbor[b_idx].unwrap().0
                                        {
                                            nearest_neighbor[b_idx] = Some((d, a_idx));
                                        }
                                    }
                                }
                            }
                        }

                        // Mark the bucket as seen for the ref_idx subsequence. All the points in the
                        // bucket go through here, irrespective of how they were processed in
                        // the loop above.
                        bounds[rep][a_idx] = hash_range.clone();
                        // Check the stopping condition, and if possible deactivate the subsequence
                        if let Some((d, nn_idx)) = nearest_neighbor[a_idx] {
                            let p = pools.collision_probability(a_idx, nn_idx);
                            let threshold =
                                ((1.0 / delta).ln() / p.powi(depth as i32)).ceil() as usize;
                            thresholds[a_idx] = threshold;
                            active[a_idx] = rep < threshold;
                            if !active[a_idx] {
                                info!("report nearest neighbor"; 
                                    "elapsed" => start.elapsed().as_secs_f64(),
                                    "depth" => depth,
                                    "repetition" => rep,
                                    "idx" => a_idx,
                                    "nn_idx" => nn_idx,
                                    "distance" => d,
                                    "threshold" => threshold,
                                    "single_collision_probability" => p,
                                    "collision_probability_at_depth" => p.powi(depth as i32),
                                );
                            }
                        }
                    }
                }
            }
            info!("completed repetition"; "computed_distances" => rep_cnt_dists, "depth" => depth, "repetition" => rep);
            pbar.inc(1);
            if depth == 0 {
                break;
            }
        }
        pbar.finish();
        info!("completed depth"; 
            "min_threshold" => thresholds.iter().min().unwrap(), 
            "max_threshold" => thresholds.iter().max().unwrap(), 
            "avg_threshold" => thresholds.iter().sum::<usize>() as f64 / thresholds.len() as f64
        );
    }
    let total_distances = ts.num_subsequences() * (ts.num_subsequences() - 1) / 2;
    println!(
        "[{:?}] done! Computed {}/{} distances ({:.2}%)",
        start.elapsed(),
        cnt_dist,
        total_distances,
        (cnt_dist as f64 / total_distances as f64) * 100.0
    );
    println!(
        "distance evaluations min {} mean {} max {}",
        count_evaluations.iter().min().unwrap(),
        count_evaluations.iter().sum::<usize>() as f64 / count_evaluations.len() as f64,
        count_evaluations.iter().max().unwrap()
    );
    println!(
        "thresholds min {} mean {} max {}",
        thresholds.iter().min().unwrap(),
        thresholds.iter().sum::<usize>() as f64 / count_evaluations.len() as f64,
        thresholds.iter().max().unwrap()
    );
    nearest_neighbor
        .iter()
        .map(|opt| opt.expect("missing nearest neighbor"))
        .collect()
}
