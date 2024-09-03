use std::{
    fs::File,
    io::{prelude::*, BufRead, BufReader},
    sync::Arc,
    time::Instant,
};

use attimo::{allocator::Bytes, motiflets::MotifletsIterator, timeseries::WindowedTimeseries};

fn main() {
    env_logger::init();

    let w = 125;
    let max_k = 20;

    for n in [10_000] {
        //[10_000, 50_000, 100_000, 150_000] {
        attimo::observe::reset_observer(format!("/tmp/attimo-n{}.csv", n));
        let ts = load_penguin_ts(w, n);
        let timer = Instant::now();
        let mut iter = MotifletsIterator::new(
            Arc::new(ts),
            max_k,
            1,
            // Bytes::system_memory().divide(2),
            Bytes::gbytes(1),
            0.05,
            w / 2,
            21234,
            false,
        );
        let motiflets: Vec<_> = iter.collect();
        let elapsed = timer.elapsed();
        println!("{}, {}", n, elapsed.as_secs_f64());
        attimo::observe::flush_observer();
    }
}

fn load_penguin_ts(w: usize, n: usize) -> WindowedTimeseries {
    let center = 497699;
    let v = load_penguin_raw();
    let ts = v[center - n..center + n].to_owned();
    WindowedTimeseries::new(ts, w, false)
}

fn load_penguin_raw() -> Vec<f64> {
    let path = "data/penguin.txt";
    let mut file = BufReader::new(File::open(path).unwrap());
    let mut res = Vec::new();
    for line in file.lines() {
        let line = line.unwrap();
        let mut tokens = line.split("\t");
        let x = tokens.next().unwrap().parse::<f64>().unwrap();
        res.push(x);
    }
    res
}
