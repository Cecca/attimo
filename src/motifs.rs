//// # Motifs
////
//// Finding motifs in time series. Instead of computing the full matrix profile,
//// leverage [LSH](src/lsh.html) to check only pairs that are probably near.

use crate::distance::*;
use crate::lsh::*;
use crate::timeseries::*;
use indicatif::ProgressBar;
use indicatif::ProgressStyle;
use slog_scope::info;
use std::collections::BTreeSet;
use std::ops::Range;
use std::rc::Rc;
use std::time::Duration;
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
pub struct Motif {
    pub idx_a: usize,
    pub idx_b: usize,
    pub distance: f64,
    pub collision_probability: f64,
    pub elapsed: Duration,
}

impl Eq for Motif {}
impl PartialEq for Motif {
    fn eq(&self, other: &Self) -> bool {
        self.idx_a == other.idx_a
            && self.idx_b == other.idx_b
            && self.distance == other.distance
            && self.collision_probability == other.collision_probability
            && self.elapsed == other.elapsed
    }
}

impl Ord for Motif {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl PartialOrd for Motif {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

struct TopK {
    k: usize,
    top: BTreeSet<Motif>,
}

impl TopK {
    fn new(k: usize) -> Self {
        Self {
            k,
            top: BTreeSet::new(),
        }
    }

    fn insert(&mut self, motif: Motif) {
        self.top.insert(motif);
        if self.top.len() > self.k {
            self.top.pop_last();
        }
    }

    // FIXME: We should set the probabilistic stopping condition here
    fn is_complete(&self) -> bool {
        self.top.len() == self.k
    }

    fn to_vec(self) -> Vec<Motif> {
        self.top.into_iter().collect()
    }
}

pub fn motifs(
    ts: Rc<WindowedTimeseries>,
    topk: usize,
    repetitions: usize,
    delta: f64,
    seed: u64,
) -> Vec<Motif> {
    let start = Instant::now();

    let exclusion_zone = ts.w / 4;
    info!("Motifs setup"; "topk" => topk, "repetitions" => repetitions, "delta" => delta, "seed" => seed, "exclusion_zone" => exclusion_zone);

    let hasher_width = Hasher::estimate_width(&ts, 20, seed);
    info!("Computed hasher width"; "hasher_width" => hasher_width);
    let hasher = Hasher::new(ts.w, repetitions, hasher_width, seed);
    let pools = HashCollection::from_ts(Rc::clone(&ts), &hasher);
    println!(
        "[{:?}] Computed hash pools, taking {}",
        start.elapsed(),
        pools.bytes_size()
    );
    let mut hashes = pools.get_hash_matrix();

    //// Define upper and lower bounds, to avoid repeating already-done comparisons
    //// We have a range of already examined hash indices for each element and repetition
    let mut bounds: Vec<Vec<Range<usize>>> = vec![vec![0..0; ts.num_subsequences()]; repetitions];

    //// keep track of active subsequences
    let mut active: Vec<bool> = vec![true; ts.num_subsequences()];
    //// These two array of counters is just for inspection
    let mut count_evaluations = vec![0usize; ts.num_subsequences()];
    let mut thresholds = vec![usize::MAX; ts.num_subsequences()];

    //// In this vector we will hold the solution: we have an entry for each subsequence, initialized
    //// to the empty value, and eventually replaced with a pair whose first element is the
    //// distance to the (estimated) nearest neighbor, and the second element is the index
    //// of the nearest neigbor itself.
    let mut nearest_neighbor: Vec<Option<(f64, usize)>> = vec![None; ts.num_subsequences()];

    let mut cnt_dist = 0;

    let mut top = TopK::new(topk);

    //// Keep track of the evolution of the minimum required number of repetitions
    let mut min_threshold = std::usize::MAX;

    //// for decreasing depths
    for depth in (0..=crate::lsh::K).rev() {
        if top.is_complete() {
            break;
        }
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
            if top.is_complete() {
                break;
            }
            let mut rep_cnt_dists = 0;
            pbar.set_message(format!(
                "depth {}, active {}",
                depth,
                active.iter().filter(|a| **a).count()
            ));
            for (hash_range, bucket) in hashes.buckets(depth, rep) {
                if top.is_complete() {
                    break;
                }
                for (a_offset, &(_, a_idx)) in bucket.iter().enumerate() {
                    if top.is_complete() {
                        break;
                    }
                    if active[a_idx] {
                        let a_already_checked = &bounds[rep][a_idx];
                        let a_hash_idx = hash_range.start + a_offset;
                        for (b_offset, &(_, b_idx)) in bucket.iter().enumerate() {
                            //// Here we handle trivial matches: we don't consider a pair if the difference between
                            //// the subsequence indexes is smaller than the exclusion zone, which is set to `w/4`.
                            if a_idx + exclusion_zone < b_idx {
                                let b_hash_idx = hash_range.start + b_offset;
                                let b_already_checked = &bounds[rep][b_idx];
                                let check_a = !a_already_checked.contains(&b_hash_idx);
                                let check_b = !b_already_checked.contains(&a_hash_idx);
                                if check_a || check_b {
                                    //// We only process the pair if this is the first repetition in which
                                    //// they collide. We get this information from the pool of bits
                                    //// from which hash values for all repetitions are extracted.
                                    let first_colliding_repetition: usize = pools
                                        .first_collision(a_idx, b_idx, depth)
                                        .expect("hashes must collide in buckets");
                                    if first_colliding_repetition == rep {
                                        //// After computing the distance between the two subsequences,
                                        //// we set `b` as the nearest neigbor of `a`, if it is closer
                                        //// than the previous candidate.
                                        let d = zeucl(&ts, a_idx, b_idx);
                                        cnt_dist += 1;
                                        rep_cnt_dists += 1;
                                        count_evaluations[a_idx] += 1;
                                        count_evaluations[b_idx] += 1;
                                        if nearest_neighbor[a_idx].is_none()
                                            || d < nearest_neighbor[a_idx].unwrap().0
                                        {
                                            nearest_neighbor[a_idx] = Some((d, b_idx));
                                        }
                                        //// Similarly, set `a` as the nearest neighbor of `b`.
                                        if nearest_neighbor[b_idx].is_none()
                                            || d < nearest_neighbor[b_idx].unwrap().0
                                        {
                                            nearest_neighbor[b_idx] = Some((d, a_idx));
                                        }
                                    }
                                }
                            }
                        }

                        //// Mark the bucket as seen for the ref_idx subsequence. All the points in the
                        //// bucket go through here, irrespective of how they were processed in
                        //// the loop above.
                        bounds[rep][a_idx] = hash_range.clone();
                        //// Check the stopping condition, and if possible deactivate the subsequence
                        if let Some((d, nn_idx)) = nearest_neighbor[a_idx] {
                            let p = hasher.collision_probability_at(d);
                            assert!(p <= 1.0);
                            let threshold =
                                ((1.0 / delta).ln() / p.powi(depth as i32)).ceil() as usize;
                            thresholds[a_idx] = threshold;
                            min_threshold = std::cmp::min(threshold, min_threshold);
                            active[a_idx] = rep < threshold;
                            if !active[a_idx] {
                                let motif = Motif {
                                    idx_a: a_idx,
                                    idx_b: nn_idx,
                                    distance: d,
                                    elapsed: start.elapsed(),
                                    collision_probability: p,
                                };
                                top.insert(motif);
                            }
                        }
                    }
                }
            }
            info!("completed repetition"; "computed_distances" => rep_cnt_dists, "depth" => depth, "repetition" => rep, "min_threshold" => min_threshold);
            pbar.inc(1);
            if depth == 0 {
                break;
            }
        }
        pbar.finish();
    }
    println!(
        "[{:?}] hash matrix matrix used {}",
        start.elapsed(),
        hashes.bytes_size()
    );
    let total_distances = ts.num_subsequences() * (ts.num_subsequences() - 1) / 2;
    println!(
        "[{:?}] done! Computed {}/{} distances ({}%)",
        start.elapsed(),
        cnt_dist,
        total_distances,
        (cnt_dist as f64 / total_distances as f64) * 100.0
    );
    top.to_vec()
}
