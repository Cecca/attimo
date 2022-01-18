//// # Motifs

//// Finding motifs in time series. Instead of computing the full matrix profile,
//// leverage [LSH](src/lsh.html) to check only pairs that are probably near.
//// The data structure used for the task is adaptive to the data, and is configured
//// to respect the limits of the system in terms of memory.

use crate::alloc_cnt;
use crate::allocator::allocated;
use crate::distance::*;
use crate::lsh::*;
use crate::timeseries::*;
use indicatif::ProgressBar;
use indicatif::ProgressStyle;
use rayon::prelude::*;
use slog_scope::info;
use std::cell::RefCell;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;
use thread_local::ThreadLocal;

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
    /// When the motif was confirmed
    pub elapsed: Option<Duration>,
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
pub struct TopK {
    k: usize,
    exclusion_zone: usize,
    top: Vec<Motif>,
}

impl TopK {
    pub fn new(k: usize, exclusion_zone: usize) -> Self {
        Self {
            k,
            exclusion_zone,
            top: Vec::new(),
        }
    }

    //// When inserting into the data structure, we first check, in order of distance,
    //// if there is a pair whose defining motif is closer than the one being inserted,
    //// and which is also overlapping.
    pub fn insert(&mut self, motif: Motif) {
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

    fn for_each<F: FnMut(&mut Motif)>(&mut self, f: F) {
        self.top.iter_mut().for_each(f);
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
    pub fn k_th(&self) -> Option<Motif> {
        if self.top.len() == self.k {
            self.top.last().map(|mot| *mot)
        } else {
            None
        }
    }

    pub fn first_not_confirmed(&self) -> Option<Motif> {
        self.top
            .iter()
            .filter(|m| m.elapsed.is_none())
            .next()
            .map(|m| *m)
    }

    pub fn to_vec(&self) -> Vec<Motif> {
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
    max_correlation: Option<f64>,
    min_correlation: Option<f64>,
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
        "max_correlation" => max_correlation,
        "min_correlation" => min_correlation,
        "exclusion_zone" => exclusion_zone
    );

    let max_dist = min_correlation.map(|c| ((1.0 - c) * (2.0 * ts.w as f64)).sqrt());
    let min_dist = max_correlation.map(|c| ((1.0 - c) * (2.0 * ts.w as f64)).sqrt());
    println!(
        "Distance constrained between {:?} and {:?}",
        min_dist, max_dist
    );

    info!("fft computation"; "tag" => "phase");
    println!("Computing FFT data");
    let timer = Instant::now();
    let fft_data = ts.fft_data();
    println!("Computed FFT data in {:?}", timer.elapsed());

    info!("quantization width estimation"; "tag" => "phase");
    let hasher_width = Hasher::estimate_width(&ts, topk, &fft_data, min_dist, seed);
    info!("Computed hasher width"; "hasher_width" => hasher_width);

    info!("hash computation"; "tag" => "phase");
    let hasher = Arc::new(Hasher::new(ts.w, repetitions, hasher_width, seed));
    let mem_before = allocated();
    let pools = HashCollection::from_ts(ts, Arc::clone(&hasher), &fft_data);
    let pools = Arc::new(pools);
    let pools_size = allocated() - mem_before;
    println!(
        "[{:?}] Computed hash pools, taking {}",
        start.elapsed(),
        PrettyBytes(pools_size)
    );
    //// Drop the fft, which we don't need from now on.
    drop(fft_data);

    //// This function is used in the stopping condition
    let threshold_fn = |d: f64, depth: isize| {
        let p = hasher.collision_probability_at(d);
        ((1.0 / delta).ln() / p.powi(depth as i32)).ceil() as usize
    };

    //// Find the level for which the given distance has a good probability of being
    //// found withing the allowed number of repetitions
    let level_for_distance = |d: f64, mut depth: isize| {
        while depth >= 0 {
            let threshold = threshold_fn(d, depth);
            if threshold < repetitions {
                break;
            }
            depth -= 1;
        }
        depth
    };

    let cnt_dist = AtomicUsize::new(0);

    info!("tries exploration"; "tag" => "phase");
    let mut top = TopK::new(topk, exclusion_zone);

    let mut stop = false;

    //// This vector holds the (sorted) hashed subsequences, and their index
    let mut column_buffer = Vec::new();
    //// This vector holds the boundaries between buckets. We reuse the allocations
    let mut buckets = Vec::new();

    //// We proceed for decreasing depths in the tries, starting from the full hash values.
    let mut depth = K as isize;
    // let mut depth = if let Some(min_dist) = min_dist {
    //     level_for_distance(min_dist, K as isize)
    // } else {
    //     K as isize
    // };
    let mut previous_depth = None;
    while depth >= 0 && !stop {
        let depth_timer = Instant::now();
        let pbar = ProgressBar::new(repetitions as u64);
        pbar.set_draw_rate(1);
        pbar.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {msg} {bar:40.cyan/blue} {pos:>7}/{len:7}"),
        );
        pbar.set_message(format!("depth {}", depth));

        for rep in 0..repetitions {
            let rep_cnt_dists = AtomicUsize::new(0);
            let spurious_collisions_cnt = AtomicUsize::new(0);
            let rep_candidate_pairs = AtomicUsize::new(0);
            let rep_timer = Instant::now();
            alloc_cnt!("column_buffer"; {
                pools.group_subsequences(depth as usize, rep, exclusion_zone, &mut column_buffer, &mut buckets);
            });
            let snap_subsequences = rep_timer.elapsed();
            let n_buckets = buckets.len();

            let tl_top = ThreadLocal::new();

            //// Each thread works on these many buckets at one time, to reduce the
            //// overhead of scheduling.
            let chunk_size = std::cmp::max(1, n_buckets / (4 * rayon::current_num_threads()));

            (0..n_buckets / chunk_size)
                .into_par_iter()
                .for_each(|chunk_i| {
                    let tl_top = tl_top.get_or(|| RefCell::new(top.clone()));
                    for i in (chunk_i * chunk_size)..((chunk_i + 1) * chunk_size) {
                        let bucket = &column_buffer[buckets[i].clone()];
                        let bpbar = if bucket.len() > 1000000 {
                            Some(ProgressBar::new(bucket.len() as u64).with_style(
                                ProgressStyle::default_bar()
                                    .template("  [{elapsed_precise}] {msg} {bar:40.white/red} {pos:>7}/{len:7}")
                            ))
                        } else {
                            None
                        };

                        for (_, a_idx) in bucket.iter() {
                            let a_idx = *a_idx as usize;
                            // let a_already_checked = rep_bounds[a_idx].clone();
                            // let a_hash_idx = hash_range.start + a_offset;
                            for (_, b_idx) in bucket.iter() {
                                let b_idx = *b_idx as usize;
                                //// Here we handle trivial matches: we don't consider a pair if the difference between
                                //// the subsequence indexes is smaller than the exclusion zone, which is set to `w/4`.
                                if a_idx + exclusion_zone < b_idx {
                                    rep_candidate_pairs.fetch_add(1, Ordering::SeqCst);
                                    //// We only process the pair if this is the first repetition in which
                                    //// they collide. We get this information from the pool of bits
                                    //// from which hash values for all repetitions are extracted.
                                    if let Some(first_colliding_repetition) =
                                        pools.first_collision(a_idx, b_idx, depth as usize)
                                    {
                                        //// This is the first collision in this iteration, _and_ the pair didn't collide
                                        //// at a deeper level.
                                        if first_colliding_repetition == rep
                                            && previous_depth
                                                .map(|d| pools.first_collision(a_idx, b_idx, d).is_none())
                                                .unwrap_or(true)
                                        {
                                            //// After computing the distance between the two subsequences,
                                            //// we try to insert the pair in the top data structure
                                            let d = zeucl(&ts, a_idx, b_idx);
                                            if d.is_finite() && d > min_dist.unwrap_or(-1.0) {
                                                rep_cnt_dists.fetch_add(1, Ordering::SeqCst);

                                                //// This is the collision probability for this distance
                                                let p = hasher.collision_probability_at(d);

                                                let m = Motif {
                                                    idx_a: a_idx,
                                                    idx_b: b_idx,
                                                    distance: d,
                                                    elapsed: None,
                                                    collision_probability: p,
                                                };
                                                tl_top.borrow_mut().insert(m);
                                            }
                                        }
                                    } else {
                                        spurious_collisions_cnt.fetch_add(1, Ordering::SeqCst);
                                    }
                                }
                            }
                            bpbar.iter().for_each(|b| {
                                let top_d = tl_top.borrow().k_th().map(|m| m.distance);
                                b.inc(1);
                                b.set_message(format!("spurious {}/{} d={:?}", spurious_collisions_cnt.load(Ordering::SeqCst), rep_candidate_pairs.load(Ordering::SeqCst), top_d));
                            });
                        }
                        bpbar.iter().for_each(|b| b.finish_and_clear());
                    }
                });

            let snap_bucket_solve = rep_timer.elapsed();

            //// Now merge the information from the thread local tops
            for tl_top in tl_top.into_iter() {
                top.merge(&tl_top.borrow());
            }

            let rep_elapsed = rep_timer.elapsed();
            info!("completed repetition";
                "tag" => "profiling",
                "computed_distances" => rep_cnt_dists.load(Ordering::SeqCst),
                "candidate_pairs" => rep_candidate_pairs.load(Ordering::SeqCst),
                "spurious_collisions" => spurious_collisions_cnt.load(Ordering::SeqCst),
                "depth" => depth,
                "repetition" => rep,
                "time_s" => rep_elapsed.as_secs_f64(),
                "time_hash_grouping_s" => snap_subsequences.as_secs_f64(),
                "time_solve_buckets_s" => (snap_bucket_solve - snap_subsequences).as_secs_f64()
            );
            cnt_dist.fetch_add(rep_cnt_dists.load(Ordering::SeqCst), Ordering::SeqCst);
            pbar.inc(1);

            //// Report on the console the motifs new motifs that have been confirmed, if any,
            //// and update their confirmation time on the go.
            top.for_each(|motif| {
                let t = threshold_fn(motif.distance, depth);
                if rep >= t && motif.elapsed.is_none() {
                    motif.elapsed.replace(start.elapsed());
                    let correlation = 1.0 - motif.distance.powi(2) / (2.0 * ts.w as f64);
                    info!("output reporting"; "tag" => "output", "distance" => motif.distance, "correlation" => correlation);
                    pbar.println(format!(
                        "Found motif at distance {:.4} ({} -- {}, corr {:.4}) after {:?} (depth {} repetition {})",
                        motif.distance,
                        motif.idx_a,
                        motif.idx_b,
                        correlation,
                        motif.elapsed.unwrap(),
                        depth,
                        rep
                    ));
                }
            });

            //// Now we check the stopping condition. If we have seen enough
            //// repetitions to make it unlikely (where _unlikely_ is quantified
            //// by the parameter `delta`) that we have missed a pair
            //// closer than the k-th in the `top` data structure, then
            //// we stop the computation.
            //// We check the condition even if no update happened, because there is
            //// one more repetition that could have made us pass the threshold.
            if let Some(m) = top.k_th() {
                let threshold = threshold_fn(m.distance, depth);
                if rep >= threshold {
                    stop = true;
                    break;
                }
            }
            if let Some(max_dist) = max_dist {
                let threshold = threshold_fn(max_dist, depth);
                if rep >= threshold {
                    stop = true;
                    break;
                }
            }
        }

        pbar.finish_and_clear();
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
            previous_depth.replace(depth as usize);
            if let Some(m) = top.first_not_confirmed() {
                let orig_depth = depth;
                let d = if let Some(max_dist) = max_dist {
                    std::cmp::min_by(max_dist, m.distance, |a, b| a.partial_cmp(b).unwrap())
                } else {
                    m.distance
                };
                depth = level_for_distance(d, depth);
                assert!(depth < orig_depth, "we are not making progress in depth");
            } else {
                depth -= 1;
            }
        }
    }
    let total_distances = ts.num_subsequences() * (ts.num_subsequences() - 1) / 2;
    let cnt_dist = cnt_dist.load(Ordering::SeqCst);
    info!("end"; "tag" => "phase");
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
        // using SCAMP: https://github.com/zpzim/SCAMP
        // The distances are slightly different, due to numerical approximation
        // and a different normalization in their computation of the standard deviation
        for (w, a, b, d) in [
            (100, 616, 2780, 0.1761538477),
            (200, 416, 2580, 0.3602377446),
            (1000, 1172, 6112, 2.133571168),
        ] {
            let ts: Vec<f64> = loadts("data/ECG-10000.csv", None).unwrap();
            let ts = WindowedTimeseries::new(ts, w, true);

            let motif = *motifs(&ts, 1, 20, 0.001, None, None, 12435)
                .first()
                .unwrap();
            println!("{}", motif.distance);
            assert!((motif.idx_a as isize - a as isize).abs() < w as isize);
            assert!((motif.idx_b as isize - b as isize).abs() < w as isize);
            assert!(motif.distance <= d + 0.00001);
        }
    }

    #[test]
    fn test_ecg_full() {
        // The indices and distances in this test have been computed
        // using SCAMP: https://github.com/zpzim/SCAMP
        // The distances are slightly different, due to numerical approximation
        // and a different normalization in their computation of the standard deviation
        for (w, a, b, d) in [(1000, 7137168, 7414108, 0.3013925657)] {
            let ts: Vec<f64> = loadts("data/ECG.csv", None).unwrap();
            let ts = WindowedTimeseries::new(ts, w, true);
            assert!((crate::distance::zeucl(&ts, a, b) - d) < 0.00000001);

            let motif = *motifs(&ts, 1, 20, 0.001, None, None, 12435)
                .first()
                .unwrap();
            println!("Motif distance {}", motif.distance);
            // We consider the test passed if we find a distance smaller than the one found by SCAMP,
            // and the motif instances are located within w steps from the ones found by SCAMP.
            // These differences are due to differences in floating point computations
            assert!(motif.distance <= d);
            assert!((motif.idx_a as isize - a as isize).abs() < w as isize);
            assert!((motif.idx_b as isize - b as isize).abs() < w as isize);
        }
    }
}
