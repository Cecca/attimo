//// # Motifs

//// Finding motifs in time series. Instead of computing the full matrix profile,
//// leverage [LSH](src/lsh.html) to check only pairs that are probably near.
//// The data structure used for the task is adaptive to the data, and is configured
//// to respect the limits of the system in terms of memory.

use crate::distance::*;
use crate::lsh::*;
use crate::timeseries::*;
use indicatif::ProgressBar;
use indicatif::ProgressStyle;
use rayon::prelude::*;
use slog_scope::info;
use std::ops::Range;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Duration;
use std::time::Instant;

//// ## Support data structures
//// ### Motifs

//// This data structure stores information about a motif:
////
////  - The index of the two subsequences defining the motif
////  - The distance between the two subsequences
////  - The LSH collision probability two subsequences
////  - The elapsed time since the start of the algorithm until
////    when the motif was found
////
//// Some utility functions follow.
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

//// An important part of working with motifs is defining and removing
//// _trivial matches_. With the function `Motif::overlaps` we can detect
//// whether two motifs overlap according to the given `exclusion_zone`:
//// if any two indices in the two motifs are at distance less than
//// `exclusion_zone` from each other, then the motifs overlap and one of them
//// shall be discarded.
impl Motif {
    /// Tells whether the two motifs overlap, in order to avoid storing trivial matches
    fn overlaps(&self, other: &Self, exclusion_zone: usize) -> bool {
        let mut idxs = [self.idx_a, self.idx_b, other.idx_a, other.idx_b];
        idxs.sort_unstable();

        idxs[0] + exclusion_zone > idxs[1]
            || idxs[1] + exclusion_zone > idxs[2]
            || idxs[2] + exclusion_zone > idxs[3]
    }
}

//// ### Top-k data structure

//// With our algorithm we look for the top motifs, that is a configurable
//// number of non-overlapping motifs in increasing order of distance.
//// This data structure implements a buffer, holding up to `k` sorted motifs,
//// such that no two motifs in the data structure are overlapping,
//// according to the parameter `exclusion_zone`.
#[derive(Clone)]
struct TopK {
    k: usize,
    exclusion_zone: usize,
    top: Vec<Motif>,
}

impl TopK {
    fn new(k: usize, exclusion_zone: usize) -> Self {
        Self {
            k,
            exclusion_zone,
            top: Vec::new(),
        }
    }

    //// When inserting into the data structure, we first check, in order of distance,
    //// if there is a pair whose defining motif is closer than the one being inserted,
    //// and which is also overlapping.
    fn insert(&mut self, motif: Motif) {
        let mut i = 0;
        while i < self.top.len() && self.top[i].distance < motif.distance {
            //// If this is the case, we don't insert the motif, and return.
            if motif.overlaps(&self.top[i], self.exclusion_zone) {
                return;
            }
            i += 1;
        }

        //// Otherwise, we insert the motif in the correct position.
        //// Because of this the `top` array is always in sorted
        //// order of increasing distance
        self.top.insert(i, motif);

        //// After the insertion we make sure that there are no other
        //// motifs overlapping with the one just inserted.
        //// To this end we remove from the tail of the vector all motifs
        //// that overlap with the one just inserted.
        ////
        //// One consequence of this is that among several trivial matches of
        //// the same motif, the one with the smallest distance is selected.
        //// In fact, this should be equivalent to just sorting all pairs of subsequences
        //// based on their distance, and then proceed from the one with smallest distance
        //// removing trivial matches along the way.
        i += 1;
        while i < self.top.len() {
            if self.top[i].overlaps(&motif, self.exclusion_zone) {
                self.top.remove(i);
            } else {
                i += 1;
            }
        }

        debug_assert!(self.top.is_sorted());

        //// Finally, we ratain only `k` elements
        if self.top.len() > self.k {
            self.top.truncate(self.k);
        }
    }

    fn merge(&mut self, other: &Self) {
        let kth_d = self
            .k_th()
            .map(|m| m.distance)
            .unwrap_or(std::f64::INFINITY);
        for m in other.top.iter() {
            if m.distance < kth_d {
                self.insert(*m);
            }
        }
    }

    //// This function is used to access the k-th motif, if we already found it.
    fn k_th(&self) -> Option<Motif> {
        if self.top.len() == self.k {
            self.top.last().map(|mot| *mot)
        } else {
            None
        }
    }

    fn to_vec(&self) -> Vec<Motif> {
        self.top.clone().into_iter().collect()
    }
}

//// ## Motif finding algorithm

//// At last, this is the algorithm to find the motifs.
//// It takes, as parameters:
////
////  - the time series windowed with windows of length `w`
////  - the number of top motifs we want to find
////  - the memory we allow the algorithm to use
////  - the probability `delta`  of making an error
////  - the seed of the random number generator
////
//// These are the general steps of the algorithm, which are then detailed below:
////
////  1. Compute the tensored hash pools for all the subsequences, taking into account
////     the allowed memory when setting the number of repetitions
////  2. Initialize the hash matrix, with one column per repetition
////  3. Then, starting from depth `K`, iterate through all repetitions, updating the
////     estimated nearest neighbor of each subsequence
////     - Whenever a nearest neighbor is updated, we try to insert it in the top-k set
////     - If the k-th motif in the top-k data structure is at distance such that the
////       probability of not having seen a closer pair is small enough, we stop.
////     - Otherwise, if we exhaust all available repetitions, it means that hash values
////       of the current length are too selective. Therefore, we relax them by considering
////       hash values one value shorter, repeating from the beginning of point 3.
pub fn motifs(
    ts: &WindowedTimeseries,
    topk: usize,
    repetitions: usize,
    delta: f64,
    seed: u64,
) -> Vec<Motif> {
    let start = Instant::now();

    //// We set the exclusion zone to the motif length, so that motifs cannot overlap at all.
    let exclusion_zone = ts.w;
    info!("Motifs setup";
        "topk" => topk,
        "repetitions" => repetitions,
        "delta" => delta,
        "seed" => seed,
        "exclusion_zone" => exclusion_zone
    );

    let hasher_width = Hasher::estimate_width(&ts, 20, seed);
    info!("Computed hasher width"; "hasher_width" => hasher_width);
    let hasher = Hasher::new(ts.w, repetitions, hasher_width, seed);
    let pools = HashCollection::from_ts(ts, &hasher);
    println!(
        "[{:?}] Computed hash pools, taking {}",
        start.elapsed(),
        pools.bytes_size()
    );
    let mut hashes = pools.get_hash_matrix();
    hashes.setup_hashes();
    println!("[{:?}] Computed hash matrix columns", start.elapsed());

    //// Define upper and lower bounds, to avoid repeating already-done comparisons
    //// We have a range of already examined hash indices for each element and repetition
    let mut bounds: Vec<Vec<Range<usize>>> = vec![vec![0..0; ts.num_subsequences()]; repetitions];

    let cnt_dist = AtomicUsize::new(0);

    let mut top = TopK::new(topk, exclusion_zone);

    let mut stop = false;

    let num_threads = rayon::current_num_threads();

    //// We proceed for decreasing depths in the tries, starting from the full hash values.
    let mut depth = crate::lsh::K as isize;
    while depth >= 0 && !stop {
        let depth_timer = Instant::now();
        let pbar = ProgressBar::new(repetitions as u64);
        pbar.set_draw_rate(4);
        pbar.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {msg} {bar:40.cyan/blue} {pos:>7}/{len:7}"),
        );
        pbar.set_message(format!("depth {}", depth));

        let mut start_rep = 0;
        while start_rep < repetitions {
            let end_rep = std::cmp::min(start_rep + num_threads, repetitions);
            let repetition_range = start_rep..end_rep;
            let local_tops: Vec<TopK> = repetition_range
                .clone()
                .into_par_iter()
                .zip(bounds[repetition_range].par_iter_mut())
                .map(|(rep, rep_bounds)| {
                    let mut local_top = top.clone();
                    let mut rep_cnt_dists = 0;
                    let mut rep_candidate_pairs = 0;
                    let rep_timer = Instant::now();
                    let buckets = hashes.buckets_vec(depth as usize, rep);

                    for (hash_range, bucket) in buckets.iter() {
                        //// We first sort by index, which improves locality in accessing the
                        //// subsequences of the time series
                        let mut bucket: Vec<(usize, usize)> = bucket
                            .iter()
                            .enumerate()
                            .map(|(offset, (_, idx))| (*idx, offset))
                            .collect();
                        bucket.sort();

                        for &(a_idx, a_offset) in bucket.iter() {
                            let a_already_checked = &rep_bounds[a_idx];
                            let a_hash_idx = hash_range.start + a_offset;
                            for &(b_idx, b_offset) in bucket.iter() {
                                //// Here we handle trivial matches: we don't consider a pair if the difference between
                                //// the subsequence indexes is smaller than the exclusion zone, which is set to `w/4`.
                                if a_idx + exclusion_zone < b_idx {
                                    let b_hash_idx = hash_range.start + b_offset;
                                    let b_already_checked = &rep_bounds[b_idx];
                                    let check_a = !a_already_checked.contains(&b_hash_idx);
                                    let check_b = !b_already_checked.contains(&a_hash_idx);
                                    if check_a || check_b {
                                        rep_candidate_pairs += 1;
                                        //// We only process the pair if this is the first repetition in which
                                        //// they collide. We get this information from the pool of bits
                                        //// from which hash values for all repetitions are extracted.
                                        let first_colliding_repetition: usize = pools
                                            .first_collision(a_idx, b_idx, depth as usize)
                                            .expect("hashes must collide in buckets");
                                        if first_colliding_repetition == rep {
                                            //// After computing the distance between the two subsequences,
                                            //// we try to insert the pair in the top data structure
                                            let d = zeucl(&ts, a_idx, b_idx);
                                            rep_cnt_dists += 1;

                                            //// We insert the motif into the `top` data structure only if
                                            //// its distance is smaller than the k-th in in `top`.
                                            let should_insert = local_top
                                                .k_th()
                                                .map(|kth| d < kth.distance)
                                                .unwrap_or(true);
                                            if should_insert {
                                                //// This is the collision probability for this distance
                                                let p = hasher.collision_probability_at(d);

                                                let motif = Motif {
                                                    idx_a: a_idx,
                                                    idx_b: b_idx,
                                                    distance: d,
                                                    elapsed: start.elapsed(),
                                                    collision_probability: p,
                                                };
                                                local_top.insert(motif);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    //// Now we update the bounds that have already been explored in this repetition
                    //// for each node. This works and can be done here instead of the loop above
                    //// because each subsequence falls into a single bucket in any given
                    //// repetition.
                    for (hash_range, bucket) in buckets.iter() {
                        for &(_, idx) in bucket.iter() {
                            //// Mark the bucket as seen for the ref_idx subsequence. All the points in the
                            //// bucket go through here, irrespective of how they were processed in
                            //// the loop above.
                            rep_bounds[idx] = hash_range.clone();
                        }
                    }

                    let rep_elapsed = rep_timer.elapsed();
                    info!("completed repetition";
                        "tag" => "profiling",
                        "computed_distances" => rep_cnt_dists,
                        "candidate_pairs" => rep_candidate_pairs,
                        "depth" => depth,
                        "repetition" => rep,
                        "time_s" => rep_elapsed.as_secs_f64()
                    );
                    cnt_dist.fetch_add(rep_cnt_dists, Ordering::SeqCst);
                    pbar.inc(1);
                    local_top
                })
                .collect();
            for t in local_tops {
                top.merge(&t);
            }

            //// Now we check the stopping condition. If we have seen enough
            //// repetitions to make it unlikely (where _unlikely_ is quantified
            //// by the parameter `delta`) that we have missed a pair
            //// closer than the k-th in the `top` data structure, then
            //// we stop the computation.
            //// We check the condition even if no update happened, because there is
            //// one more repetition that could have made us pass the threshold.
            if let Some(kth) = top.k_th() {
                let threshold = ((1.0 / delta).ln() / kth.collision_probability.powi(depth as i32))
                    .ceil() as usize;
                info!("check stopping condition"; "threshold" => threshold);
                if end_rep >= threshold {
                    stop = true;
                    break;
                }
            }

            start_rep += num_threads;
        }

        pbar.finish();
        info!("level completed";
            "tag" => "profiling",
            "depth" => depth,
            "time_s" => depth_timer.elapsed().as_secs_f64()
        );

        //// If we are not done, we decide to which level of the trie to jump, based on the distance of the k-th motif.
        //// This heuristic does not hurt correctness. Even if the current k-th motif is not the correct one, then
        //// it means that we are jumping on a level too low. But this only hurts performance, since it means we
        //// are going to evaluate more pairs, but we will not miss any pair that would have been evaluated at
        //// deeper levels.
        if !stop {
            if let Some(kth) = top.k_th() {
                let orig_depth = depth;
                while depth >= 0 {
                    let threshold = ((1.0 / delta).ln()
                        / kth.collision_probability.powi(depth as i32))
                    .ceil() as usize;
                    if threshold <= repetitions {
                        break;
                    }
                    depth -= 1;
                }
                assert!(depth < orig_depth, "we are not making progress in depth");
            } else {
                depth -= 1;
            }
        }
    }
    println!(
        "[{:?}] hash matrix matrix used {}",
        start.elapsed(),
        hashes.bytes_size()
    );
    let total_distances = ts.num_subsequences() * (ts.num_subsequences() - 1) / 2;
    let cnt_dist = cnt_dist.load(Ordering::SeqCst);
    info!("motifs completed";
        "tag" => "profiling",
        "time_s" => start.elapsed().as_secs_f64(),
        "cnt_dist" => cnt_dist,
        "total_distances" => total_distances,
        "distances_fraction" => (cnt_dist as f64 / total_distances as f64)
    );
    println!(
        "[{:?}] done! Computed {}/{} distances ({}%)",
        start.elapsed(),
        cnt_dist,
        total_distances,
        (cnt_dist as f64 / total_distances as f64) * 100.0,
    );
    top.to_vec()
}

#[cfg(test)]
mod test {
    use crate::{load::loadts, timeseries::WindowedTimeseries};

    use super::motifs;

    #[test]
    fn test_ecg_10000() {
        // The indices and distances in this test have been computed 
        // using STUMPY: https://github.com/TDAmeritrade/stumpy
        // The distances are slightly different, due to numerical approximation
        // and a different normalization in their computation of the standard deviation
        for (w, a, b, d) in [
            (100, 616, 2780, 0.17526071805739987),
            (200, 416, 2580, 0.35932460689995877),
            (1000, 1172, 6112, 2.1325079069545545)
        ] {
            let ts: Vec<f64> = loadts("data/ECG-10000.csv", None).unwrap();
            let ts = WindowedTimeseries::new(ts, w);

            let motif = *motifs(&ts, 1, 100, 0.01, 12435).first().unwrap();
            assert_eq!(motif.idx_a, a);
            assert_eq!(motif.idx_b, b);
            println!("{}", motif.distance);
            assert!((motif.distance - d).abs() < 0.0000001);
        }
    }
}
