use crate::distance::*;
use crate::embedding::*;
use crate::lsh::*;
use crate::types::*;
use bumpalo::Bump;
use indicatif::ProgressBar;
use indicatif::ProgressStyle;
use std::collections::HashMap;
use std::fmt::Debug;
use std::ops::Range;
use std::time::Instant;

pub fn approx_mp(
    ts: &WindowedTimeseries,
    k: usize,
    repetitions: usize,
    delta: f64,
    seed: u64,
) -> Vec<(f64, usize)> {
    let start = Instant::now();
    let sf = scaling_factor(ts, zeucl, 0.01);
    assert!(!sf.is_nan());
    println!("[{:?}] Scaling factor: {}", start.elapsed(), sf);

    let hasher = Hasher::new(k, repetitions, Embedder::new(ts.w, ts.w, 1.0, seed), seed);
    let arena = Bump::new();
    let pools = HashCollection::from_ts(&ts, &hasher, &arena);
    println!(
        "[{:?}] Computed hash pools, taking {}",
        start.elapsed(),
        PrettyBytes(arena.allocated_bytes())
    );
    let hashes = pools.get_hash_matrix();
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

    // In this vector we will hold the solution: we have an entry for each subsequence, initialized
    // to the empty value, and eventually replaced with a pair whose first element is the
    // distance to the (estimated) nearest neighbor, and the second element is the index
    // of the nearest neigbor itself.
    let mut nearest_neighbor: Vec<Option<(f64, usize)>> = vec![None; ts.num_subsequences()];

    let mut cnt_dist = 0;

    // for decreasing depths
    for depth in (0..=k).rev() {
        if active.iter().filter(|a| **a).count() == 0 {
            break;
        }
        let pbar = ProgressBar::new(repetitions as u64);
        pbar.set_draw_rate(4);
        pbar.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {msg} {bar:40.cyan/blue} {pos:>7}/{len:7}"),
        );

        for rep in 0..repetitions {
            pbar.set_message(format!(
                "depth {}, active {}",
                depth,
                active.iter().filter(|a| **a).count()
            ));
            for (hash_range, bucket) in hashes.buckets(depth, rep) {
                // stats.push_bucket_size(depth, bucket.len());
                for (a_offset, &(_, a_idx)) in bucket.iter().enumerate() {
                    if active[a_idx] {
                        let a_already_checked = &bounds[rep][a_idx];
                        let a_hash_idx = hash_range.start + a_offset;
                        for (b_offset, &(_, b_idx)) in bucket.iter().enumerate() {
                            if a_idx < b_idx {
                                // FIXME: Maybe we can exclude trivial matches already here?
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
                        if let Some((_, nn_idx)) = nearest_neighbor[a_idx] {
                            let p = pools.collision_probability(a_idx, nn_idx);
                            let threshold =
                                ((2.0 / delta).ln() / p.powi(depth as i32)).ceil() as usize;
                            active[a_idx] = rep < threshold;
                        }
                    }
                }
            }
            pbar.inc(1);
            if depth == 0 {
                break;
            }
        }
        pbar.finish();
    }
    let total_distances = ts.num_subsequences() * (ts.num_subsequences() - 1) / 2;
    println!(
        "[{:?}] done! Computed {}/{} distances ({:.2}%)",
        start.elapsed(),
        cnt_dist,
        total_distances,
        (cnt_dist as f64 / total_distances as f64) * 100.0
    );
    nearest_neighbor
        .iter()
        .map(|opt| opt.expect("missing nearest neighbor"))
        .collect()
}

struct Stats {
    bucket_size: HashMap<usize, Vec<usize>>,
}

impl Stats {
    fn new() -> Self {
        Self {
            bucket_size: HashMap::new(),
        }
    }

    fn push_bucket_size(&mut self, depth: usize, size: usize) {
        self.bucket_size
            .entry(depth)
            .or_insert_with(Vec::new)
            .push(size);
    }
}

impl Debug for Stats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut formatted: Vec<(usize, String)> = self
            .bucket_size
            .iter()
            .map(|(depth, buckets)| {
                if buckets.is_empty() {
                    return (*depth, format!("no buckets at depth {}", depth));
                }
                let mut buckets = buckets.clone();
                buckets.sort();
                let median = buckets[buckets.len() / 2];
                let min = buckets[0];
                let max = buckets[buckets.len() - 1];
                (
                    *depth,
                    format!(
                        "bucket size at depth {}: min={} median={} max={}",
                        depth, min, median, max
                    ),
                )
            })
            .collect();
        formatted.sort();
        formatted.reverse();
        for (_, s) in formatted {
            writeln!(f, "{}", s)?
        }
        Ok(())
    }
}
