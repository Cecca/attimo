use attimo::{
    lsh::{HashCollection, HashValue, Hasher},
    sort::RadixSort,
    timeseries::{FFTData, WindowedTimeseries},
};

use std::sync::Arc;
use std::time::Instant;

fn main() {
    let n = std::env::args().nth(1).unwrap().parse::<usize>().unwrap();

    // let mut rng = Xoroshiro128PlusPlus::seed_from_u64(1234);
    // let uniform = Uniform::new(i8::MIN, i8::MAX);
    // let geom = Geometric::new(0.25).unwrap();
    // let v: Vec<(HashValue, usize)> = (0..n).map(|idx| {
    //     let mut hashes: [i8; 32] = [0; 32];
    //     for (i, x) in uniform.sample_iter(&mut rng).take(32).enumerate() {
    //         hashes[i] = x as i8;
    //     }
    //     (HashValue{hashes}, idx)
    // }).collect();

    println!("Generating time series");
    let start = Instant::now();
    let ts: Vec<f64> = attimo::load::loadts("data/ECG.csv", Some(n)).unwrap();
    let ts = WindowedTimeseries::new(ts, 300, true);
    let fft_data = FFTData::new(&ts);
    println!("...{:?}", start.elapsed());
    // let ts = Rc::new(WindowedTimeseries::gen_randomwalk(n, 300, 1243));
    println!("Computing hashes");
    let start = Instant::now();
    let hasher = Arc::new(Hasher::new(300, 200, 2.0, 123));
    let hc = HashCollection::from_ts(&ts, &fft_data, hasher);
    let v: Vec<(HashValue, usize)> = (0..n)
        .map(|i| (hc.hash_value(i, attimo::lsh::K, 0), i))
        .collect();
    println!("...{:?}", start.elapsed());

    println!("Radix sort");
    let mut v1 = v.clone();
    let start = Instant::now();
    v1.radix_sort();
    let elapsed = start.elapsed();
    println!(
        "Sorted {} values in {:?} ({} elems/s)",
        v.len(),
        elapsed,
        v.len() as f64 / elapsed.as_secs_f64()
    );

    println!("Rust unstable sort");
    let mut v2 = v.clone();
    let start = Instant::now();
    v2.sort_unstable();
    let elapsed = start.elapsed();
    println!(
        "Sorted {} values in {:?} ({} elems/s)",
        v.len(),
        elapsed,
        v.len() as f64 / elapsed.as_secs_f64()
    );
}
