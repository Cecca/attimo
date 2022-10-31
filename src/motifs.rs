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
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::ops::Range;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;
use thread_local::ThreadLocal;
#[derive(Debug, PartialEq, PartialOrd)]
struct OrderedF64(f64);

impl Eq for OrderedF64 {}

impl Ord for OrderedF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

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
    /// When the motif was confirmed
    pub elapsed: Option<Duration>,
}

impl Eq for Motif {}
impl PartialEq for Motif {
    fn eq(&self, other: &Self) -> bool {
        self.idx_a == other.idx_a && self.idx_b == other.idx_b && self.distance == other.distance
    }
}

impl Ord for Motif {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl PartialOrd for Motif {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.distance
            .partial_cmp(&other.distance)
            .map(|ord| ord.then_with(|| self.idx_a.cmp(&other.idx_a)))
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

impl std::fmt::Debug for TopK {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, m) in self.top.iter().enumerate() {
            writeln!(
                f,
                "  {} ::: {} -- {}  ({:.4}) ({:?})",
                i, m.idx_a, m.idx_b, m.distance, m.elapsed
            )?;
        }

        Ok(())
    }
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
        while i < self.top.len() && self.top[i].distance <= motif.distance {
            if motif.overlaps(&self.top[i], self.exclusion_zone) {
                //// If this is the case, we don't insert the motif, and return.
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

        //// Finally, we retain only `k` elements
        if self.top.len() > self.k {
            for m in &self.top[self.k..] {
                assert!(m.elapsed.is_none());
            }
            self.top.truncate(self.k);
        }
    }

    //// This function is used to access the k-th motif, if
    //// we already found it, even if not confirmed yet
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

    pub fn last_confirmed(&self) -> Option<Motif> {
        self.top
            .iter()
            .filter(|m| m.elapsed.is_some())
            .last()
            .map(|m| *m)
    }

    pub fn num_confirmed(&self) -> usize {
        self.confirmed().count()
    }

    pub fn confirmed(&self) -> impl Iterator<Item = Motif> + '_ {
        self.top.iter().filter(|m| m.elapsed.is_some()).map(|m| *m)
    }

    pub fn for_each(&mut self, f: impl FnMut(&mut Motif)) {
        self.top.iter_mut().for_each(f)
    }

    pub fn len(&self) -> usize {
        self.top.len()
    }

    pub fn to_vec(&self) -> Vec<Motif> {
        self.top.clone().into_iter().collect()
    }

    pub fn add_all(&mut self, other: &mut TopK) {
        for m in other.top.drain(..) {
            self.insert(m);
        }
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
    start: Instant, // when the program started, to compute elapsed confirmation times
) -> Vec<Motif> {
    //// We set the exclusion zone to the motif length, so that motifs cannot overlap at all.
    let exclusion_zone = ts.w;
    let fft_data = FFTData::new(&ts);

    let max_dist = min_correlation.map(|c| ((1.0 - c) * (2.0 * ts.w as f64)).sqrt());
    let min_dist = max_correlation.map(|c| ((1.0 - c) * (2.0 * ts.w as f64)).sqrt());
    println!(
        "Distance constrained between {:?} and {:?}",
        min_dist, max_dist
    );

    let hasher_width = Hasher::estimate_width(&ts, &fft_data, topk, min_dist, seed);
    info!("Computed hasher width"; "hasher_width" => hasher_width);

    info!("hash computation"; "tag" => "phase");
    let hasher = Arc::new(Hasher::new(ts.w, repetitions, hasher_width, seed));
    let mem_before = allocated();
    let pools = HashCollection::from_ts(ts, &fft_data, Arc::clone(&hasher));
    let pools = Arc::new(pools);
    let pools_size = allocated() - mem_before;
    println!(
        "[{:?}] Computed hash pools, taking {}",
        start.elapsed(),
        PrettyBytes(pools_size)
    );
    eprintln!("Dropping fft data");
    drop(fft_data);

    let cnt_dist = AtomicUsize::new(0);

    let mut output = TopK::new(topk, exclusion_zone);

    info!("tries exploration"; "tag" => "phase");
    //// This vector holds the (sorted) hashed subsequences, and their index
    let mut column_buffer = Vec::new();
    //// This vector holds the boundaries between buckets. We reuse the allocations
    let mut buckets = Vec::new();

    explore_tries(
        ts,
        Arc::clone(&pools),
        &cnt_dist,
        topk,
        delta,
        exclusion_zone,
        min_dist,
        max_dist,
        start,
        &mut output,
        &mut column_buffer,
        &mut buckets,
    );

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
    output.to_vec()
}

fn explore_tries(
    ts: &WindowedTimeseries,
    pools: Arc<HashCollection>,
    cnt_dist: &AtomicUsize,
    topk: usize,
    delta: f64,
    exclusion_zone: usize,
    min_dist: Option<f64>,
    max_dist: Option<f64>,
    start: Instant,
    output: &mut TopK,
    column_buffer: &mut Vec<(HashValue, u32)>,
    buckets: &mut Vec<Range<usize>>,
) {
    let mut tl_top = ThreadLocal::new();

    let repetitions = pools.hasher.repetitions;
    let hasher = Arc::clone(&pools.hasher);

    let stopping_condition = |d: f64, prefix: isize, previous: Option<usize>, repetition: usize| {
        let p = hasher.collision_probability_at(d);
        let i_half = prefix as f64 / 2.0;
        let sqrt = (repetitions as f64).sqrt().ceil() as i32;
        let j_left = repetition as i32 / sqrt;
        let j_right = repetition as i32 % sqrt;
        let failure_p = if let Some(previous) = previous {
            let prev_half = previous as f64 / 2.0;
            let lu_i = 1.0 - (1.0 - p.powf(i_half)).powi(j_left);
            let ru_i = 1.0 - (1.0 - p.powf(i_half)).powi(j_right);
            let lu_ip = 1.0 - (1.0 - p.powf(prev_half)).powi(j_left);
            let ru_ip = 1.0 - (1.0 - p.powf(prev_half)).powi(j_right);
            let ll_ip = 1.0 - (1.0 - p.powf(prev_half)).powi(sqrt - j_left);
            let rl_ip = 1.0 - (1.0 - p.powf(prev_half)).powi(sqrt - j_right);
            (1.0 - lu_i * ru_i)
                * (1.0 - lu_ip * rl_ip)
                * (1.0 - ll_ip * ru_ip)
                * (1.0 - ll_ip * rl_ip)
        } else {
            let lu_i = 1.0 - (1.0 - p.powf(i_half)).powi(j_left);
            let ru_i = 1.0 - (1.0 - p.powf(i_half)).powi(j_right);
            1.0 - lu_i * ru_i
        };
        failure_p <= delta / (topk as f64)
    };

    //// Find the level for which the given distance has a good probability of being
    //// found withing the allowed number of repetitions
    let level_for_distance = |d: f64, mut prefix: isize| {
        let initial = prefix as usize;
        while prefix >= 0 {
            for rep in 0..repetitions {
                if stopping_condition(d, prefix, Some(initial), rep) {
                    return prefix;
                }
            }
            prefix -= 1;
        }
        panic!()
    };

    //// We proceed for decreasing depths in the tries, starting from the full hash values.
    let mut depth = if let Some(min_dist) = min_dist {
        level_for_distance(min_dist, K as isize)
    } else {
        K as isize
    };
    println!(
        "Exploring tries starting from minimum distance {:?} at depth {}",
        min_dist, depth
    );
    let mut previous_depth = None;
    while depth >= 0 {
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
                pools.group_subsequences(depth as usize, rep, exclusion_zone, column_buffer, buckets);
            });
            let snap_subsequences = rep_timer.elapsed();
            let n_buckets = buckets.len();

            //// Each thread works on these many buckets at one time, to reduce the
            //// overhead of scheduling.
            let chunk_size = std::cmp::max(1, n_buckets / (4 * rayon::current_num_threads()));

            (0..n_buckets / chunk_size)
                .into_par_iter()
                .for_each(|chunk_i| {
                    // let tl_top = tl_top.get_or(|| RefCell::new(TopK::new(topk, exclusion_zone)));
                    let tl_top = tl_top.get_or(|| RefCell::new(output.clone()));

                    // counters
                    let mut cands = 0;
                    let mut dists = 0;
                    let mut spurious = 0;

                    for i in (chunk_i * chunk_size)..((chunk_i + 1) * chunk_size) {
                        let bucket = &column_buffer[buckets[i].clone()];

                        for (_, a_idx) in bucket.iter() {
                            let a_idx = *a_idx as usize;
                            // let a_already_checked = rep_bounds[a_idx].clone();
                            // let a_hash_idx = hash_range.start + a_offset;
                            for (_, b_idx) in bucket.iter() {
                                let b_idx = *b_idx as usize;
                                //// Here we handle trivial matches: we don't consider a pair if the difference between
                                //// the subsequence indexes is smaller than the exclusion zone, which is set to `w/4`.
                                if a_idx + exclusion_zone < b_idx {
                                    cands += 1;

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
                                                .map(|d| {
                                                    pools.first_collision(a_idx, b_idx, d).is_none()
                                                })
                                                .unwrap_or(true)
                                        {
                                            //// After computing the distance between the two subsequences,
                                            //// we try to insert the pair in the top data structure
                                            let d = zeucl(&ts, a_idx, b_idx);
                                            if d.is_finite() && d > min_dist.unwrap_or(-1.0) {
                                                dists += 1;

                                                let m = Motif {
                                                    idx_a: a_idx as usize,
                                                    idx_b: b_idx as usize,
                                                    distance: d,
                                                    elapsed: None,
                                                };
                                                tl_top.borrow_mut().insert(m);
                                            }
                                        }
                                    } else {
                                        spurious += 1;
                                    }
                                }
                            }
                        }
                    }
                    rep_candidate_pairs.fetch_add(cands, Ordering::SeqCst);
                    rep_cnt_dists.fetch_add(dists, Ordering::SeqCst);
                    spurious_collisions_cnt.fetch_add(spurious, Ordering::SeqCst);
                });

            let snap_bucket_solve = rep_timer.elapsed();

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

            // Add to the output all the new top pairs that have been found
            tl_top
                .iter_mut()
                .for_each(|top| output.add_all(&mut top.borrow_mut()));
            tl_top.iter_mut().for_each(|top| {
                top.replace(output.clone());
            });

            // Confirm the pairs that can be confirmed in this iteration
            output.for_each(|m| {
                if m.elapsed.is_none() {
                    if stopping_condition(m.distance, depth, previous_depth, rep) {
                        m.elapsed.replace(start.elapsed());
                        pbar.println(format!(
                            "Confirm {} -- {} @ {:.4} ({:?})",
                            m.idx_a,
                            m.idx_b,
                            m.distance,
                            m.elapsed.unwrap()
                        ));
                    }
                }
            });

            //// Now we check the stopping condition. If we have seen enough
            //// repetitions to make it unlikely (where _unlikely_ is quantified
            //// by the parameter `delta`) that we have missed a pair
            //// closer than the k-th in the `top` data structure, then
            //// we stop the computation.
            //// We check the condition even if no update happened, because there is
            //// one more repetition that could have made us pass the threshold.
            if output.num_confirmed() == topk {
                return;
            }
            if let Some(max_dist) = max_dist {
                if stopping_condition(max_dist, depth, previous_depth, rep) {
                    return;
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
        previous_depth.replace(depth as usize);
        if let Some(m) = output.first_not_confirmed() {
            let orig_depth = depth;
            let d = if let Some(max_dist) = max_dist {
                std::cmp::min_by(max_dist, m.distance, |a, b| a.partial_cmp(b).unwrap())
            } else {
                m.distance
            };
            depth = level_for_distance(d, depth);
            if depth == orig_depth {
                depth -= 1;
            }
            pbar.println(format!(
                "Next candidate at distance {:.4}, going at depth {}",
                d, depth
            ));
            // assert!(depth < orig_depth, "we are not making any progress");
        } else {
            depth -= 1;
        }
    }
}

pub struct MotifsEnumerator {
    start: Instant,
    ts: Arc<WindowedTimeseries>,
    pub max_k: usize,
    topk: TopK,
    to_return: BinaryHeap<Reverse<Motif>>,
    /// used to cache the motifs already discovered so that we can use the enumerator as a collection
    returned: Vec<Motif>,
    repetitions: usize,
    delta: f64,
    exclusion_zone: usize,
    hasher: Arc<Hasher>,
    pools: Arc<HashCollection>,
    column_buffer: Vec<(HashValue, u32)>,
    buckets: Vec<Range<usize>>,
    tl_top: ThreadLocal<RefCell<TopK>>,
    /// the current repetition
    rep: usize,
    /// the current depth
    depth: usize,
    /// the previous depth
    previous_depth: Option<usize>,
}

impl MotifsEnumerator {
    pub fn new(
        ts: Arc<WindowedTimeseries>,
        max_k: usize,
        repetitions: usize,
        delta: f64,
        seed: u64,
    ) -> Self {
        let start = Instant::now();
        let exclusion_zone = ts.w;
        let fft_data = FFTData::new(&ts);

        let hasher_width = Hasher::estimate_width(&ts, &fft_data, max_k, None, seed);
        info!("Computed hasher width"; "hasher_width" => hasher_width);

        info!("hash computation"; "tag" => "phase");
        let hasher = Arc::new(Hasher::new(ts.w, repetitions, hasher_width, seed));
        let mem_before = allocated();
        let pools = HashCollection::from_ts(&ts, &fft_data, Arc::clone(&hasher));
        let pools = Arc::new(pools);
        let pools_size = allocated() - mem_before;
        drop(fft_data);

        let cnt_dist = AtomicUsize::new(0);

        let mut topk = TopK::new(max_k, exclusion_zone);

        info!("tries exploration"; "tag" => "phase");
        // This vector holds the (sorted) hashed subsequences, and their index
        let column_buffer = Vec::new();
        // This vector holds the boundaries between buckets. We reuse the allocations
        let buckets = Vec::new();
        let tl_top = ThreadLocal::new();

        Self {
            start,
            ts,
            max_k,
            topk,
            to_return: BinaryHeap::new(),
            returned: Vec::new(),
            repetitions,
            delta,
            exclusion_zone,
            hasher,
            pools,
            column_buffer,
            buckets,
            tl_top,
            rep: 0,
            depth: K,
            previous_depth: None,
        }
    }

    pub fn get_ts(&self) -> Arc<WindowedTimeseries> {
        Arc::clone(&self.ts)
    }

    pub fn stopping_condition(
        p: f64,
        prefix: usize,
        previous: Option<usize>,
        repetition: usize,
        repetitions: usize,
        delta: f64,
    ) -> bool {
        let i_half = prefix as f64 / 2.0;
        let sqrt = (repetitions as f64).sqrt().ceil() as i32;
        let j_left = repetition as i32 / sqrt;
        let j_right = repetition as i32 % sqrt;
        let failure_p = if let Some(previous) = previous {
            let prev_half = previous as f64 / 2.0;
            let lu_i = 1.0 - (1.0 - p.powf(i_half)).powi(j_left);
            let ru_i = 1.0 - (1.0 - p.powf(i_half)).powi(j_right);
            let lu_ip = 1.0 - (1.0 - p.powf(prev_half)).powi(j_left);
            let ru_ip = 1.0 - (1.0 - p.powf(prev_half)).powi(j_right);
            let ll_ip = 1.0 - (1.0 - p.powf(prev_half)).powi(sqrt - j_left);
            let rl_ip = 1.0 - (1.0 - p.powf(prev_half)).powi(sqrt - j_right);
            (1.0 - lu_i * ru_i)
                * (1.0 - lu_ip * rl_ip)
                * (1.0 - ll_ip * ru_ip)
                * (1.0 - ll_ip * rl_ip)
        } else {
            let lu_i = 1.0 - (1.0 - p.powf(i_half)).powi(j_left);
            let ru_i = 1.0 - (1.0 - p.powf(i_half)).powi(j_right);
            1.0 - lu_i * ru_i
        };
        failure_p <= delta
    }

    //// Find the level for which the given distance has a good probability of being
    //// found withing the allowed number of repetitions
    fn level_for_distance(&self, d: f64, mut prefix: usize, delta: f64) -> usize {
        let p = self.hasher.collision_probability_at(d);
        let initial = prefix as usize;
        while prefix > 0 {
            for rep in 0..self.repetitions {
                if Self::stopping_condition(p, prefix, Some(initial), rep, self.repetitions, delta)
                {
                    return prefix;
                }
            }
            prefix -= 1;
        }
        panic!("Got to prefix of length 0!");
    }

    /// Return the next motif, or `None` if we already returned `max_k` motifs
    pub fn next_motif(&mut self) -> Option<Motif> {
        // First, try to empty the buffer of motifs to return, if any
        if let Some(motif) = self.to_return.pop() {
            self.returned.push(motif.0.clone());
            return Some(motif.0);
        }

        // check we already returned all we could
        if self.topk.num_confirmed() == self.max_k {
            return None;
        }

        // repeat until we are able to buffer some motifs
        while self.to_return.is_empty() {
            assert!(self.depth > 0);
            assert!(self.rep < self.repetitions);

            // Set up buckets for the current repetition
            self.pools.group_subsequences(
                self.depth,
                self.rep,
                self.exclusion_zone,
                &mut self.column_buffer,
                &mut self.buckets,
            );
            let n_buckets = self.buckets.len();
            let chunk_size = std::cmp::max(1, n_buckets / (4 * rayon::current_num_threads()));

            // counters for profiling
            let rep_cnt_dists = AtomicUsize::new(0);
            let spurious_collisions_cnt = AtomicUsize::new(0);
            let rep_candidate_pairs = AtomicUsize::new(0);

            (0..n_buckets / chunk_size)
                .into_par_iter()
                .for_each(|chunk_i| {
                    // let tl_top = tl_top.get_or(|| RefCell::new(TopK::new(topk, exclusion_zone)));
                    let tl_top = self.tl_top.get_or(|| RefCell::new(self.topk.clone()));

                    // counters
                    let mut cands = 0;
                    let mut dists = 0;
                    let mut spurious = 0;

                    for i in (chunk_i * chunk_size)..((chunk_i + 1) * chunk_size) {
                        let bucket = &self.column_buffer[self.buckets[i].clone()];

                        for (_, a_idx) in bucket.iter() {
                            let a_idx = *a_idx as usize;
                            // let a_already_checked = rep_bounds[a_idx].clone();
                            // let a_hash_idx = hash_range.start + a_offset;
                            for (_, b_idx) in bucket.iter() {
                                let b_idx = *b_idx as usize;
                                //// Here we handle trivial matches: we don't consider a pair if the difference between
                                //// the subsequence indexes is smaller than the exclusion zone, which is set to `w/4`.
                                if a_idx + self.exclusion_zone < b_idx {
                                    cands += 1;

                                    //// We only process the pair if this is the first repetition in which
                                    //// they collide. We get this information from the pool of bits
                                    //// from which hash values for all repetitions are extracted.
                                    if let Some(first_colliding_repetition) =
                                        self.pools.first_collision(a_idx, b_idx, self.depth)
                                    {
                                        //// This is the first collision in this iteration, _and_ the pair didn't collide
                                        //// at a deeper level.
                                        if first_colliding_repetition == self.rep
                                            && self
                                                .previous_depth
                                                .map(|d| {
                                                    self.pools
                                                        .first_collision(a_idx, b_idx, d)
                                                        .is_none()
                                                })
                                                .unwrap_or(true)
                                        {
                                            //// After computing the distance between the two subsequences,
                                            //// we try to insert the pair in the top data structure
                                            let d = zeucl(&self.ts, a_idx, b_idx);
                                            if d.is_finite() {
                                                dists += 1;

                                                let m = Motif {
                                                    idx_a: a_idx as usize,
                                                    idx_b: b_idx as usize,
                                                    distance: d,
                                                    elapsed: None,
                                                };
                                                tl_top.borrow_mut().insert(m);
                                            }
                                        }
                                    } else {
                                        spurious += 1;
                                    }
                                }
                            }
                        }
                    }
                    rep_candidate_pairs.fetch_add(cands, Ordering::SeqCst);
                    rep_cnt_dists.fetch_add(dists, Ordering::SeqCst);
                    spurious_collisions_cnt.fetch_add(spurious, Ordering::SeqCst);
                });

            // Add to the output all the new top pairs that have been found
            let mut tmp_top = TopK::new(self.max_k, self.exclusion_zone);
            self.tl_top
                .iter_mut()
                .for_each(|top| tmp_top.add_all(&mut top.borrow_mut()));
            self.topk.add_all(&mut tmp_top);

            // Confirm the pairs that can be confirmed in this iteration
            let elapsed = self.start.elapsed();
            let depth = self.depth;
            let previous_depth = self.previous_depth;
            let rep = self.rep;
            let repetitions = self.repetitions;
            let delta = self.delta;
            let hasher = Arc::clone(&self.hasher);
            let mut buf = Vec::new();
            self.topk.for_each(|m| {
                if m.elapsed.is_none() {
                    let p = hasher.collision_probability_at(m.distance);
                    if Self::stopping_condition(p, depth, previous_depth, rep, repetitions, delta) {
                        m.elapsed.replace(elapsed);
                        buf.push(m.clone());
                    }
                }
            });
            self.to_return.extend(buf.drain(..).map(|m| Reverse(m)));

            // set up next repetition
            self.rep += 1;
            if self.rep >= self.repetitions {
                self.rep = 0;
                self.previous_depth.replace(self.depth);
                if let Some(first_not_confirmed) = self.topk.first_not_confirmed() {
                    let new_depth =
                        self.level_for_distance(first_not_confirmed.distance, self.depth, delta);
                    if new_depth == depth {
                        self.depth -= 1;
                    } else {
                        self.depth = new_depth;
                    }
                } else {
                    self.depth -= 1;
                }
                assert!(self
                    .previous_depth
                    .map(|prev| self.depth < prev)
                    .unwrap_or(true));
            }
        }

        // return the found motif with the smallest distance
        let m = self.to_return.pop().map(|m| m.0);
        if let Some(m) = m.clone() {
            self.returned.push(m);
        }
        m
    }

    /// Gets the motif at the given rank, possibly computing it if not already discovered.
    pub fn get_ranked(&mut self, rank: usize) -> &Motif {
        if rank >= self.max_k {
            panic!("Index out of bounds");
        }
        while self.returned.len() <= rank {
            self.next();
        }
        &self.returned[rank]
    }
}

impl Iterator for MotifsEnumerator {
    type Item = Motif;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_motif()
    }
}

#[cfg(test)]
mod test {
    use crate::{load::loadts, timeseries::WindowedTimeseries};

    use super::*;

    #[test]
    #[ignore]
    fn test_motif_ecg_10000() {
        // The indices and distances in this test have been computed
        // using SCAMP: https://github.com/zpzim/SCAMP
        // The distances are slightly different, due to numerical approximation
        // and a different normalization in their computation of the standard deviation
        for (w, a, b, d) in [
            (100, 616, 2780, 0.1761538477),
            (200, 416, 2580, 0.3602377446),
            (1000, 1172, 6112, 2.133571168),
        ] {
            let ts: Vec<f64> = loadts("data/ECG.csv.gz", Some(10000)).unwrap();
            let ts = WindowedTimeseries::new(ts, w, true);

            let motif = *motifs(&ts, 1, 20, 0.001, None, None, 12435, Instant::now())
                .first()
                .unwrap();
            println!(
                "{} -- {} actual {} expected {}",
                motif.idx_a, motif.idx_b, motif.distance, d
            );
            assert!((motif.idx_a as isize - a as isize).abs() < w as isize);
            assert!((motif.idx_b as isize - b as isize).abs() < w as isize);
            assert!(motif.distance <= d + 0.00001);
        }
    }

    #[test]
    #[ignore]
    fn test_motif_ecg_full() {
        // The indices and distances in this test have been computed
        // using SCAMP: https://github.com/zpzim/SCAMP
        // The distances are slightly different, due to numerical approximation
        // and a different normalization in their computation of the standard deviation
        for (w, a, b, d) in [(1000, 7137168, 7414108, 0.3013925657)] {
            let ts: Vec<f64> = loadts("data/ECG.csv.gz", None).unwrap();
            let ts = WindowedTimeseries::new(ts, w, true);
            // assert!((crate::distance::zeucl(&ts, a, b) - d) < 0.00000001);

            let motif = *motifs(&ts, 1, 200, 0.001, None, None, 12435, Instant::now())
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

    #[test]
    #[ignore]
    fn test_motif_ecg_top10() {
        // as in the other examples, the ground truth is obtained using SCAMP run on the GPU
        let top10 = [
            (7137166, 7414106, 0.3013925657),
            (7377870, 7383302, 0.343015406),
            (7553828, 7587436, 0.3612951315),
            (6779076, 7379224, 0.3880223353),
            (7238944, 7264944, 0.3938163096),
            (7574696, 7611520, 0.3942701023),
            (7094136, 7220980, 0.3981813093),
            (6275400, 6298896, 0.3989290683),
            (6625400, 7479248, 0.4026470338),
            (6961239, 7385163, 0.4042721064),
        ];

        let w = 1000;
        let ts: Vec<f64> = loadts("data/ECG.csv.gz", None).unwrap();
        let ts = WindowedTimeseries::new(ts, w, false);

        let motifs = motifs(&ts, 10, 200, 0.01, None, None, 12435, Instant::now());
        for (a, b, dist) in top10 {
            // look for this in the motifs, allowing up to w displacement
            println!("looking for ({a} {b} {dist})");
            let mut found = false;
            for motif in &motifs {
                found |= (motif.idx_a as isize - a as isize).abs() <= w as isize;
                found |= (motif.idx_b as isize - b as isize).abs() <= w as isize;
                if found {
                    println!(
                        "   found at ({} {} {})",
                        motif.idx_a, motif.idx_b, motif.distance
                    );
                    break;
                }
            }
            assert!(
                found,
                "Could not find ({}, {}, {}) in {:?}",
                a, b, dist, motifs
            );
        }
    }

    #[test]
    #[ignore]
    fn test_motif_astro_top10_enumerate() {
        // as in the other examples, the ground truth is obtained using SCAMP run on the GPU
        let top10 = [
            (609810, 888455, 1.264327903),
            (502518, 656063, 1.312459673),
            (321598, 423427, 1.368041725),
            (342595, 625081, 1.403194924),
            (218448, 1006871, 1.442935122),
            (192254, 466432, 1.523167513),
            (527024, 533903, 1.526611152),
            (520191, 743708, 1.558780057),
            (192097, 193569, 1.583277835),
            (267982, 512333, 1.617081054),
        ];

        let w = 100;
        let ts: Vec<f64> = loadts("data/ASTRO.csv.gz", None).unwrap();
        let ts = Arc::new(WindowedTimeseries::new(ts, w, false));

        let motifs: Vec<Motif> = MotifsEnumerator::new(ts, 10, 800, 0.01, 12435).collect();

        for (a, b, dist) in top10 {
            // look for this in the motifs, allowing up to w displacement
            println!("looking for ({a} {b} {dist})");
            let mut found = false;
            for motif in &motifs {
                found |= (motif.idx_a as isize - a as isize).abs() <= w as isize;
                found |= (motif.idx_b as isize - b as isize).abs() <= w as isize;
                if found {
                    println!(
                        "   found at ({} {} {})",
                        motif.idx_a, motif.idx_b, motif.distance
                    );
                    break;
                }
            }
            assert!(
                found,
                "Could not find ({}, {}, {}) in {:?}",
                a, b, dist, motifs
            );
        }
    }

    #[test]
    #[ignore]
    fn test_motif_astro_top10() {
        // as in the other examples, the ground truth is obtained using SCAMP run on the GPU
        let top10 = [
            (609810, 888455, 1.264327903),
            (502518, 656063, 1.312459673),
            (321598, 423427, 1.368041725),
            (342595, 625081, 1.403194924),
            (218448, 1006871, 1.442935122),
            (192254, 466432, 1.523167513),
            (527024, 533903, 1.526611152),
            (520191, 743708, 1.558780057),
            (192097, 193569, 1.583277835),
            (267982, 512333, 1.617081054),
        ];

        let w = 100;
        let ts: Vec<f64> = loadts("data/ASTRO.csv.gz", None).unwrap();
        let ts = WindowedTimeseries::new(ts, w, false);

        let motifs = motifs(&ts, 10, 800, 0.01, None, None, 12435, Instant::now());
        for (a, b, dist) in top10 {
            // look for this in the motifs, allowing up to w displacement
            println!("looking for ({a} {b} {dist})");
            let mut found = false;
            for motif in &motifs {
                found |= (motif.idx_a as isize - a as isize).abs() <= w as isize;
                found |= (motif.idx_b as isize - b as isize).abs() <= w as isize;
                if found {
                    println!(
                        "   found at ({} {} {})",
                        motif.idx_a, motif.idx_b, motif.distance
                    );
                    break;
                }
            }
            assert!(
                found,
                "Could not find ({}, {}, {}) in {:?}",
                a, b, dist, motifs
            );
        }
    }
}
