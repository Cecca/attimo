use anyhow::Result;
use attimo::distance::zeucl;
use attimo::load::loadts;
use attimo::lsh::*;
use attimo::motifs::{Motif, TopK};
use attimo::sort::*;
use attimo::timeseries::WindowedTimeseries;
use std::convert::TryInto;
use std::time::{Duration, Instant};

fn main() -> Result<()> {
    let ts = loadts("data/ECG.csv.gz", None)?;
    let ts = WindowedTimeseries::new(ts, 1000, false);

    let n = ts.num_subsequences();

    let delta = 0.01;
    let top_dist = 0.403;

    let mut accum = 0.0;
    println!("r,k,collisions,preparation,exploration,total,repetitions");
    for r in &[1.0, 2.0, 4.0, 8.0] {
        let mut topk = TopK::new(10, ts.w);
        let mut cost = CostEstimator::new(n);
        let hasher = Hasher::new(ts.w, 1, *r, 1234);
        let mut ids: Vec<(u8, usize)> = (0..ts.num_subsequences()).map(|i| (0, i)).collect();
        let mut boundaries = vec![(0..ts.num_subsequences())];
        let mut buf = vec![0; ts.num_subsequences()];

        let top_p = hasher.collision_probability_at(top_dist);

        for k in 0..32 {
            let required_repetitions = ((1.0f64/delta).log(std::f64::consts::E) / top_p.powi(k as i32)).ceil() as usize;
            let start_hash = Instant::now();
            hasher.hash_all(&ts, k, 0, &mut buf);
            for (h, i) in ids.iter_mut() {
                *h = buf[*i];
            }
            let elapsed_hash = Instant::now() - start_hash;
            eprintln!("Hash in {:?}", elapsed_hash);

            let start_sort = Instant::now();
            let mut new_boundaries = Vec::new();
            for r in &boundaries {
                ids[r.clone()].sort_unstable_by_key(|p| p.0);
                // ids[r.clone()].radix_sort();
            }
            let elapsed_sort = Instant::now() - start_sort;
            eprintln!("Sort in {:?}", elapsed_sort);

            let start_boundaries = Instant::now();
            for r in &boundaries {
                let mut start = r.start;
                let mut end = start + 1;
                while end < r.end {
                    if ids[end].0 != ids[start].0 {
                        new_boundaries.push(start..end);
                        start = end;
                        end = start + 1;
                    } else {
                        end += 1;
                    }
                }
                new_boundaries.push(start..end);
            }
            let elapsed_boundaries = Instant::now() - start_boundaries;
            let actual_elapsed = Instant::now() - start_hash;

            eprintln!("Boundaries in {:?}", elapsed_boundaries);
            std::mem::swap(&mut boundaries, &mut new_boundaries);
            let tot_len: usize = boundaries.iter().map(|r| r.len()).sum();
            assert!(tot_len == n);

            let num_collisions: usize =
                boundaries.iter().map(|r| r.len() * (r.len() - 1) / 2).sum();
            cost.append(LevelMeasurements {
                num_collisions,
                required_repetitions,
                hash: elapsed_hash,
                sort: elapsed_sort,
                boundaries: elapsed_boundaries,
            });

            let estimated = cost.precomputation_at(k);
            eprintln!(
                "Actual elapsed {:?} estimated {:?}",
                actual_elapsed, estimated
            );

            if false { // num_collisions < 10_000_000 && num_collisions > 0 {
                let bs = boundaries.clone();
                let start_distances = Instant::now();
                let mut cnt = 0;
                for r in bs {
                    let buck = &ids[r];
                    for (_, i) in buck.iter() {
                        for (_, j) in buck.iter() {
                            if i < j {
                                cnt += 1;
                                let d = zeucl(&ts, *i, *j);
                                topk.insert(Motif {
                                    idx_a: *i,
                                    idx_b: *j,
                                    distance: d,
                                    elapsed: None,
                                });
                            }
                        }
                    }
                }
                assert!(cnt == num_collisions);
                let elapsed = Instant::now() - start_distances;
                cost.add_distances(cnt, elapsed);
                eprintln!(
                    "Computed {} distances in {:?} ({:?}/pair) (estimated {:?})",
                    cnt,
                    elapsed,
                    elapsed / cnt as u32,
                    cost.exploration_cost_at(k)
                );
            }
        }
        eprintln!("top-10 dist {:?}", topk.k_th());

        for level in 0..32 {
            println!(
                "{}, {}, {}, {}, {}, {}, {}",
                r,
                level,
                cost.collisions_at(level),
                cost.preparation_cost_at(level).as_secs_f64(),
                cost.exploration_cost_at(level).as_secs_f64(),
                cost.total_cost_at(level).as_secs_f64(),
                cost.required_repetitions_at(level)
            );
        }
    }

    Ok(())
}

struct LevelMeasurements {
    num_collisions: usize,
    required_repetitions: usize,
    hash: Duration,
    sort: Duration,
    boundaries: Duration,
}

struct CostEstimator {
    n: usize,
    levels: Vec<LevelMeasurements>,
    distance: Duration,
    n_distances: usize,
}

impl CostEstimator {
    fn new(n: usize) -> Self {
        Self {
            n,
            levels: Vec::new(),
            n_distances: 0,
            distance: Duration::from_secs(0),
        }
    }

    fn append(&mut self, measurements: LevelMeasurements) {
        self.levels.push(measurements);
    }

    fn add_distances(&mut self, ndist: usize, duration: Duration) {
        self.distance += duration;
        self.n_distances += ndist;
    }

    fn required_repetitions_at(&self, level: usize) -> usize {
        self.levels[level].required_repetitions
    }

    fn collisions_at(&self, level: usize) -> usize {
        self.levels[level].num_collisions
    }

    fn total_cost_at(&self, level: usize) -> Duration {
        self.preparation_cost_at(level) + self.exploration_cost_at(level)
    }

    fn precomputation_at(&self, level: usize) -> Duration {
        let level = &self.levels[level];
        (level.hash + level.sort + level.boundaries)
    }

    fn preparation_cost_at(&self, level: usize) -> Duration {
        (0..=level).map(|l| self.precomputation_at(l)).sum()
    }

    fn exploration_cost_at(&self, level: usize) -> Duration {
        let level = &self.levels[level];
        let s_per_distance = self.distance.as_secs_f64() / self.n_distances as f64;
        let tot = s_per_distance * level.num_collisions as f64;
        // Duration::from_secs_f64(tot)
        Duration::from_nanos(560 * level.num_collisions as u64)
    }
}
