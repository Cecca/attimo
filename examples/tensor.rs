use attimo::distance::zeucl;
use attimo::{load::loadts, lsh::spherical::*, timeseries::WindowedTimeseries};
use std::io::Write;

const K: usize = 32;

fn main() {
    let w = 1000;
    let ts = loadts("data/ECG.csv.gz", Some(3000)).unwrap();
    let ts = WindowedTimeseries::new(ts, w, false);
    let mut rng = rand::thread_rng();
    let repetitions = 128;
    let tables = LSHTables::from_ts(&ts, repetitions, &mut rng);
    let zdist = zeucl(&ts, 0, 10);
    dbg!(zdist);
    let dotp = (2.0 * (w as f64) - zdist.powi(2)) / 2.0;
    let p = tables.collision_probability_at(dotp);
    dbg!(dotp);
    dbg!(p);
    println!("bits,rep,fp,independent");
    let mut printed = false;
    for bits in (1..=K).rev() {
        for rep in 0..repetitions {
            let fp = tables.failure_probability(dotp, rep, bits);
            let fp_independent = tables.independent_failure_probability(dotp, rep, bits);
            dbg!(fp);
            assert!(fp >= 0.0);
            // assert!(fp >= fp_independent);
            if fp < 0.01 && !printed {
                eprintln!("bits {} rep {}", bits, rep);
                printed = true;
            }
            println!("{},{},{},{}", bits, rep, fp, false);
            println!("{},{},{},{}", bits, rep, fp_independent, true);
        }
    }
}
