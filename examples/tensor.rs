use attimo::distance::zeucl;
use attimo::{load::loadts, lsh::spherical::*, timeseries::WindowedTimeseries};
use std::time::Instant;

const K: usize = 32;

fn main() {
    let w = 1000;
    let ts = loadts("data/ECG.csv.gz", None).unwrap();
    let ts = WindowedTimeseries::new(ts, w, false);
    eprintln!("Loaded time series");
    let mut rng = rand::thread_rng();
    let repetitions = 256;
    let start = Instant::now();
    let tables = LSHTables::from_ts(&ts, repetitions, &mut rng);
    let end = Instant::now();
    eprintln!("Building {} hash tables: {:?}", repetitions, end - start);
    let zdist = zeucl(&ts, 0, 20);
    dbg!(zdist);
    println!("bits,rep,fp,independent");
    let mut printed = false;
    for bits in (1..=K).rev() {
        for rep in 0..repetitions {
            let fp = tables.failure_probability(zdist, rep, bits);
            let fp_independent = tables.independent_failure_probability(zdist, rep, bits);
            if fp < 0.01 && !printed {
                eprintln!("bits {} rep {}", bits, rep);
                printed = true;
            }
            println!("{},{},{},{}", bits, rep, fp, false);
            println!("{},{},{},{}", bits, rep, fp_independent, true);
        }
    }
}
