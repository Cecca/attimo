use anyhow::Result;
use attimo::adaptive::AdaptiveHashCollection;
use attimo::distance::zeucl;
use attimo::load::loadts;
use attimo::lsh::*;
use attimo::motifs::{Motif, TopK};
use attimo::sort::*;
use attimo::timeseries::WindowedTimeseries;
use std::convert::TryInto;
use std::time::{Duration, Instant};

const MB: usize = 1024*1024;

fn main() -> Result<()> {
    let ts = loadts("data/ECG.csv.gz", None)?;
    let ts = WindowedTimeseries::new(ts, 1000, false);
    // let ts = WindowedTimeseries::gen_randomwalk(30, 3, 234);

    let timer = Instant::now();
    let mut topk = TopK::new(1, ts.w);
    let mut coll = AdaptiveHashCollection::new(&ts, 1024 * MB, 12345);
    eprintln!("Estimation and construction {:?}", timer.elapsed());

    let repetitions = 10;
    let t_add_reps = Instant::now();
    coll.add_repetitions(repetitions);
    eprintln!("Adding repetitions {:?}", t_add_reps.elapsed());

    for prefix in (1..=coll.max_k).rev() {
        for rep in 0..repetitions {
            let t_explore = Instant::now();
            coll.for_pairs_at(rep, prefix, |pi, pj| {
                let i = std::cmp::min(pi, pj);
                let j = std::cmp::max(pi, pj);
                let d = zeucl(&ts, i, j);
                let m = Motif {
                    idx_a: i,
                    idx_b: j,
                    distance: d,
                    elapsed: None
                };
                topk.insert(m);
            });
            // eprintln!("Explore in {:?}", t_explore.elapsed());
        }
        let d = topk.first_not_confirmed().unwrap().distance;
        eprintln!("First non confirmed distance: {d}");
        if d <= 0.3014 {
            eprintln!("Found!");
            return Ok(())
        }
        // let best = coll.best_move(d, 0.01);
    }
    eprintln!("Not found!");
    Ok(())
}
