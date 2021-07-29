use bumpalo::Bump;

use crate::distance::*;
use crate::embedding::*;
use crate::lsh::*;
use crate::types::*;
use std::ops::Range;
use std::time::Instant;

pub fn approx_mp(
    ts: &WindowedTimeseries,
    knn: usize,
    k: usize,
    repetitions: usize,
    delta: f64,
    seed: u64,
) {
    let start = Instant::now();
    let sf = scaling_factor(ts, zeucl, 0.01);
    println!("[{:?}] Scaling factor: {}", start.elapsed(), sf);

    let hasher = Hasher::new(32, 200, Embedder::new(ts.w, ts.w, 1.0, 1234), 49875);
    let arena = Bump::new();
    let pools = HashCollection::from_ts(&ts, &hasher, &arena);
    println!("[{:?}] Computed hash pools", start.elapsed());
    let hashes = pools.get_hash_matrix();
    println!("[{:?}] Computed hash matrix", start.elapsed());

    // Define upper and lower bounds, to avoid repeating already-done comparisons
    // We have a range of already examined hash indices for each element and repetition
    let bounds: Vec<Vec<Range<usize>>> = vec![vec![0..0; ts.num_subsequences()]; repetitions];

    // keep track of active subsequences
    let active: Vec<bool> = vec![true; ts.num_subsequences()];

    // for decreasing depths
    for depth in (0..k).rev() {
        for rep in 0..repetitions {
            for (hash_range, bucket) in hashes.buckets(depth, rep) {
                for &(_, ref_idx) in bucket {
                    if active[ref_idx] {
                        let already_checked = &bounds[rep][ref_idx];
                        // FIXME: here you can leverage that the indexes are sorted to halve the number of computations
                        for &(_, cand_idx) in bucket {
                            if !already_checked.contains(&cand_idx) {
                                let first_colliding_repetition: usize = pools
                                    .first_collision(ref_idx, cand_idx, depth)
                                    .expect("hashes must collide in buckets");
                                if first_colliding_repetition == rep {
                                    // push into the buffer
                                    if cand_idx != ref_idx {
                                        //
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
