use bumpalo::Bump;
use indicatif::ProgressStyle;

use crate::distance::*;
use crate::embedding::*;
use crate::lsh::*;
use crate::types::*;
use indicatif::ProgressBar;
use std::cmp::Ordering;
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
    println!("[{:?}] Computed hash pools", start.elapsed());
    let hashes = pools.get_hash_matrix();
    println!("[{:?}] Computed hash matrix", start.elapsed());

    // Define upper and lower bounds, to avoid repeating already-done comparisons
    // We have a range of already examined hash indices for each element and repetition
    let mut bounds: Vec<Vec<Range<usize>>> = vec![vec![0..0; ts.num_subsequences()]; repetitions];

    // keep track of active subsequences
    let mut active: Vec<bool> = vec![true; ts.num_subsequences()];

    let mut nearest_neighbor: Vec<Option<(f64, usize)>> = vec![None; ts.num_subsequences()];

    let mut stats = Stats::new();

    // for decreasing depths
    for depth in (0..=k).rev() {
        let pbar = ProgressBar::new(repetitions as u64);
        pbar.set_draw_rate(4);
        pbar.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {msg} {bar:40.cyan/blue} {pos:>7}/{len:7}"),
        );
        // println!("==== depth {}", depth);
        for rep in 0..repetitions {
            // println!("----");
            pbar.set_message(format!(
                "depth {}, active {}",
                depth,
                active.iter().filter(|a| **a).count()
            ));
            for (hash_range, bucket) in hashes.buckets(depth, rep) {
                stats.push_bucket_size(depth, bucket.len());
                for &(_, ref_idx) in bucket {
                    if active[ref_idx] {
                        let already_checked = &bounds[rep][ref_idx];
                        // FIXME: we can leverage the fact that the indexes are sorted to halve the number of computations
                        for (offset, &(_, cand_idx)) in bucket.iter().enumerate() {
                            // FIXME: Maybe we can exclude trivial matches already here?
                            let hash_idx = hash_range.start + offset;
                            if cand_idx != ref_idx && !already_checked.contains(&hash_idx)
                            {
                                // FIXME: Fix issues with first colliding repetition
                                let first_colliding_repetition: usize = pools
                                    .first_collision(ref_idx, cand_idx, depth)
                                    .expect("hashes must collide in buckets");
                                // if first_colliding_repetition == rep {
                                    let d = zeucl(ts, ref_idx, cand_idx);
                                    if nearest_neighbor[ref_idx].is_none()
                                        || d < nearest_neighbor[ref_idx].unwrap().0
                                    {
                                        nearest_neighbor[ref_idx] = Some((d, cand_idx));
                                    }
                                // }
                            }
                        }

                        // mark the bucket as seen for the ref_idx subsequence
                        bounds[rep][ref_idx] = hash_range.clone();
                        // Check the stopping condition, and if possible deactivate the subsequence
                        if let Some((_, nn_idx)) = nearest_neighbor[ref_idx] {
                            let p = pools.collision_probability(ref_idx, nn_idx);
                            let threshold =
                                ((2.0 / delta).ln() / p.powi(depth as i32)).ceil() as usize;
                            active[ref_idx] = rep < threshold;
                        }
                    }
                }
            }
            pbar.inc(1);
            if depth==0 {
                break;
            }
        }
        pbar.finish();
    }
    println!("[{:?}] done!", start.elapsed());
    println!("Stats: \n{:?}", stats);
    nearest_neighbor
        .iter()
        .map(|opt| opt.expect("missing nearest neighbor"))
        .collect()
}

struct Stats {
    bucket_size: HashMap<usize, Vec<usize>>
}

impl Stats {
    fn new() -> Self {
        Self {
            bucket_size: HashMap::new()
        }
    }

    fn push_bucket_size(&mut self, depth: usize, size: usize) {
        self.bucket_size.entry(depth).or_insert_with(Vec::new).push(size);
    }
}

impl Debug for Stats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut formatted: Vec<(usize, String)> = self.bucket_size.iter().map(|(depth, buckets)| {
            if buckets.is_empty() {
                return (*depth, format!("no buckets at depth {}", depth));
            }
            let mut buckets = buckets.clone();
            buckets.sort();
            let median = buckets[buckets.len()/2];
            let min = buckets[0];
            let max = buckets[buckets.len()-1];
            (*depth, format!("bucket size at depth {}: min={} median={} max={}", depth, min, median, max))
        }).collect();
        formatted.sort();
        formatted.reverse();
        for (_, s) in formatted {
            writeln!(f, "{}", s)?
        }
        Ok(())
    }
}

