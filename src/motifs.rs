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
use slog_scope::info;
use std::ops::Range;
use std::rc::Rc;
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

    fn len(&self) -> usize {
        self.top.len()
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

    //// This function is used to access the k-th motif, if we already found it.
    fn k_th(&self) -> Option<Motif> {
        if self.top.len() == self.k {
            self.top.last().map(|mot| *mot)
        } else {
            None
        }
    }

    fn to_vec(self) -> Vec<Motif> {
        self.top.into_iter().collect()
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
    ts: Rc<WindowedTimeseries>,
    topk: usize,
    memory: PrettyBytes,
    delta: f64,
    seed: u64,
) -> Vec<Motif> {
    let start = Instant::now();

    //// We set the exclusion zone to the motif length, so that motifs cannot overlap at all.
    let exclusion_zone = ts.w;
    info!("Motifs setup"; "topk" => topk, "memory" => format!("{}", memory), "delta" => delta, "seed" => seed, "exclusion_zone" => exclusion_zone);

    //// First, we have to determine how many repetitions we can afford with the allowed memory
    let repetitions = {
        let mut r = 1;
        let target = memory.0;
        loop {
            if HashCollection::required_memory(&ts, r + 1) > target {
                break;
            }
            r += 1;
        }
        r
    };
    println!("Performing {} repetitions", repetitions);

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

    let mut cnt_dist = 0;

    let mut top = TopK::new(topk, exclusion_zone);

    //// Keep track of the evolution of the minimum required number of repetitions
    let mut min_threshold = std::usize::MAX;

    let mut insertions_cnt = 0;

    //// Flag to signal if we have to continue the computation
    let mut stop = false;

    //// for decreasing depths
    for depth in (0..=crate::lsh::K).rev() {
        if stop {
            break;
        }
        let pbar = ProgressBar::new(repetitions as u64);
        pbar.set_draw_rate(4);
        pbar.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {msg} {bar:40.cyan/blue} {pos:>7}/{len:7}"),
        );

        for rep in 0..repetitions {
            if stop {
                break;
            }
            let mut rep_cnt_dists = 0;
            pbar.set_message(format!(
                "depth {} pq={} thresh={}",
                depth,
                top.len(),
                min_threshold
            ));
            for (hash_range, bucket) in hashes.buckets(depth, rep) {
                if stop {
                    break;
                }
                for (a_offset, &(_, a_idx)) in bucket.iter().enumerate() {
                    if stop {
                        break;
                    }
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
                                    //// we try to insert the pair in the top data structure
                                    let d = zeucl(&ts, a_idx, b_idx);
                                    cnt_dist += 1;
                                    rep_cnt_dists += 1;

                                    //// We insert the motif into the `top` data structure only if
                                    //// its distance is smaller than the k-th in in `top`.
                                    if top.k_th().map(|kth| d < kth.distance).unwrap_or(true) {
                                        //// This is the collision probability for this distance
                                        let p = hasher.collision_probability_at(d);

                                        let motif = Motif {
                                            idx_a: a_idx,
                                            idx_b: b_idx,
                                            distance: d,
                                            elapsed: start.elapsed(),
                                            collision_probability: p,
                                        };
                                        top.insert(motif);
                                        insertions_cnt += 1;
                                    }
                                }
                            }
                        }
                    }

                    //// Mark the bucket as seen for the ref_idx subsequence. All the points in the
                    //// bucket go through here, irrespective of how they were processed in
                    //// the loop above.
                    bounds[rep][a_idx] = hash_range.clone();

                    //// Now we check the stopping condition. If we have seen enough
                    //// repetitions to make it unlikely (where _unlikely_ is quantified
                    //// by the parameter `delta`) that we have missed a pair
                    //// closer than the k-th in the `top` data structure, then
                    //// we stop the computation.
                    //// We check the condition even if no update happened, because there is
                    //// one more repetition that could have made us pass the threshold.
                    if let Some(kth) = top.k_th() {
                        let threshold = ((1.0 / delta).ln()
                            / kth.collision_probability.powi(depth as i32))
                        .ceil() as usize;
                        min_threshold = threshold;
                        stop = rep >= threshold;
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
        "[{:?}] done! Computed {}/{} distances ({}%), {} insertions in queue",
        start.elapsed(),
        cnt_dist,
        total_distances,
        (cnt_dist as f64 / total_distances as f64) * 100.0,
        insertions_cnt
    );
    top.to_vec()
}
