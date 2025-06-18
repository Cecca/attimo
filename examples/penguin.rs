use std::{
    fs::File,
    io::{BufRead, BufReader},
    sync::Arc,
    time::Instant,
};

use attimo::{
    allocator::Bytes,
    motiflets::{brute_force_motiflets, MotifletsIterator},
    timeseries::WindowedTimeseries,
};

fn main() {
    env_logger::init();
    let mode = std::env::args()
        .nth(1)
        .unwrap_or("probabilistic".to_owned());

    #[cfg(feature = "pprof")]
    let _guard = attimo::profiler::Profiler::start();

    let w = 125;
    let max_k = 20;

    // for n in [10_000, 20_000, 30_000, 40_000, 60_000] {
    // for n in [1000, 2000, 3000, 10000, 20000] {
    for n in [200_000] {
        attimo::observe::reset_observer(format!("/tmp/attimo-n{}.csv", n));
        let ts = load_penguin_ts(w, n);
        let timer = Instant::now();
        match mode.as_str() {
            "exact" => {
                brute_force_motiflets(&ts, max_k, w / 2);
            }
            "probabilistic" => {
                let mut iter = MotifletsIterator::new(
                    Arc::new(ts),
                    max_k,
                    1,
                    Bytes::system_memory().divide(2),
                    0.05,
                    w / 2,
                    21234,
                    false,
                );
                iter.set_stop_on_collisions_threshold(false);
                for motiflet in iter {
                    eprintln!(
                        "support: {} extent: {} RC: {}",
                        motiflet.support(),
                        motiflet.extent(),
                        motiflet.relative_contrast()
                    );
                }
            }
            _ => {
                panic!("provide either 'exact' or 'probabilistic'")
            }
        }
        let elapsed = timer.elapsed();
        println!("{}, {}", n, elapsed.as_secs_f64());
        attimo::observe::flush_observer();
    }
}

fn load_penguin_ts(w: usize, n: usize) -> WindowedTimeseries {
    let center = 497699;
    let v = load_penguin_raw();
    dbg!(v.len());
    let ts = v[center - n..center + n].to_owned();
    WindowedTimeseries::new(ts, w, false)
}

fn load_penguin_raw() -> Vec<f64> {
    let path = "data/penguin.txt";
    let file = BufReader::new(File::open(path).unwrap());
    let mut res = Vec::new();
    for line in file.lines() {
        let line = line.unwrap();
        let mut tokens = line.split("\t");
        let x = tokens.next().unwrap().parse::<f64>().unwrap();
        res.push(x);
    }
    res
}
