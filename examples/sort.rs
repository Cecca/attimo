use attimo::{
    lsh::{HashCollection, Hasher},
    timeseries::WindowedTimeseries,
};

use std::sync::Arc;
use std::time::Instant;

fn main() {
    let n = std::env::args().nth(1).unwrap().parse::<usize>().unwrap();

    let w = 1000;
    println!("Reading time series");
    let start = Instant::now();
    let ts: Vec<f64> = attimo::load::loadts("data/ECG.csv.gz", Some(n)).unwrap();
    let ts = WindowedTimeseries::new(ts, w, false);
    println!("...{:?}", start.elapsed());
    println!("Computing hashes");
    let start = Instant::now();
    let hasher = Arc::new(Hasher::new(ts.w, 4, 2.0, 123));
    let hc = HashCollection::from_ts(&ts, hasher);
    println!("...{:?}", start.elapsed());

    println!("Radix sort");
    let mut v: Vec<u32> = (0u32..ts.num_subsequences() as u32).collect();
    let start = Instant::now();
    hc.lexi_sort(0, &mut v);
    let elapsed = start.elapsed();
    println!(
        "Sorted {} values in {:?} ({} elems/s)",
        v.len(),
        elapsed,
        v.len() as f64 / elapsed.as_secs_f64()
    );

}
