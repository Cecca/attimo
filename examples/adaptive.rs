use anyhow::Result;
use attimo::adaptive::AdaptiveHashCollection;
use attimo::distance::zeucl;
use attimo::load::loadts;
use attimo::motifs::{Motif, TopK};
use attimo::timeseries::WindowedTimeseries;
use rayon::prelude::*;
use std::time::Instant;

const MB: usize = 1024 * 1024;

pub fn collision_probability_at(r: f64, d: f64) -> f64 {
    use statrs::distribution::ContinuousCDF;
    use statrs::distribution::Normal;
    let normal = Normal::new(0.0, 1.0).unwrap();
    1.0 - 2.0 * normal.cdf(-r / d)
        - (2.0 / ((std::f64::consts::PI * 2.0).sqrt() * (r / d)))
            * (1.0 - (-r * r / (2.0 * d * d)).exp())
}

fn main() -> Result<()> {
    let ts = loadts("data/ECG.csv.gz", None)?;
    let ts = WindowedTimeseries::new(ts, 1000, false);
    let start = Instant::now();

    let exclusion_zone = ts.w;
    let delta = 0.01;
    let num_motifs = 1;
    let timer = Instant::now();
    let mut topk = TopK::new(num_motifs, exclusion_zone);
    let mut coll = AdaptiveHashCollection::new(&ts, 4096 * MB, 12345);
    eprintln!("Estimation and construction {:?}", timer.elapsed());
    let r = coll.r;

    let stopping_condition = |d: f64, prefix: usize, repetition: usize| {
        let p = collision_probability_at(r, d);
        repetition as f64
            >= (num_motifs as f64 / delta).log(std::f64::consts::E) / p.powi(prefix as i32)
    };

    let repetitions = 10;
    let t_add_reps = Instant::now();
    coll.set_repetitions(repetitions);
    eprintln!("Adding repetitions {:?}", t_add_reps.elapsed());

    let mut prefix = coll.max_k;
    let mut rep = 0;
    loop {
        let t_rep = Instant::now();
        eprintln!("Doing repetitions at level {}", coll.level);
        let mut new_topk = (rep..coll.current_repetitions())
            .into_par_iter()
            .map(|rep| {
                let mut topk = TopK::new(num_motifs, exclusion_zone);
                coll.for_pairs(rep, |pi, pj| {
                    let i = std::cmp::min(pi, pj);
                    let j = std::cmp::max(pi, pj);
                    let d = zeucl(&ts, i, j);
                    let m = Motif {
                        idx_a: i,
                        idx_b: j,
                        distance: d,
                        elapsed: None,
                    };
                    topk.insert(m);
                });
                // eprintln!("repetition {} at prefix {} took {:?}", rep, prefix, t_rep.elapsed());
                topk
            })
            .reduce(
                || topk.clone(),
                |mut a, mut b| {
                    a.add_all(&mut b);
                    a
                },
            );
        topk.add_all(&mut new_topk);
        eprintln!("Time to run repetitions {:?}", t_rep.elapsed());
        rep = coll.current_repetitions();

        // Confirm the pairs that can be confirmed in this iteration
        topk.for_each(|m| {
            if m.elapsed.is_none() {
                if stopping_condition(m.distance, prefix, coll.current_repetitions()) {
                    m.elapsed.replace(start.elapsed());
                    eprintln!(
                        "Confirm {} -- {} @ {:.4} ({:?})",
                        m.idx_a,
                        m.idx_b,
                        m.distance,
                        m.elapsed.unwrap()
                    );
                }
            }
        });

        if topk.num_confirmed() == num_motifs {
            break;
        }

        let (candidate_prefix, candidate_reps) = coll.best_move(
            topk.first_not_confirmed().unwrap().distance,
            delta / num_motifs as f64,
        );
        eprintln!("Candidate prefix is {candidate_prefix}, candidate repetitions {candidate_reps} (current prefix {prefix} , current repetitions {})", coll.current_repetitions());
        assert!(candidate_prefix < prefix || candidate_reps > coll.current_repetitions());
        if candidate_reps <= coll.current_repetitions() {
            // Reset the repetitions and do them from scratch
            rep = 0;
            prefix = candidate_prefix;
            coll.decrease_level(candidate_prefix);
        } else {
            coll.set_repetitions(candidate_reps);
        }
    }
    Ok(())
}
