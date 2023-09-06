#![feature(portable_simd)]
use attimo::{
    lsh::{HashCollection, HashValue, Hasher},
    sort::{GetByte, RadixSort},
    timeseries::{FFTData, WindowedTimeseries},
};
use pprof::protos::Message;
use rayon::prelude::*;
use std::sync::Arc;
use std::time::Instant;
use std::{io::Write, time::Duration};

fn bench<T: Ord + GetByte + Clone, F: Fn(Vec<T>)>(name: &str, v: &[T], f: F) {
    println!("{}", name);
    let runs = 10;
    let mut times = Vec::new();
    for _ in 0..runs {
        let v = v.to_vec();
        let start = Instant::now();
        f(v);
        let elapsed = start.elapsed();
        times.push(elapsed);
    }
    let elapsed = times.into_iter().sum::<Duration>() / runs;
    println!(
        "   Sorted {} values in {:?} ({} elems/s)",
        v.len(),
        elapsed,
        v.len() as f64 / elapsed.as_secs_f64()
    );
}

fn main() {
    let n = std::env::args().nth(1).unwrap().parse::<usize>().unwrap();

    let w = 300;
    println!("Reading time series");
    let start = Instant::now();
    let ts: Vec<f64> = attimo::load::loadts("data/ECG.csv", Some(n)).unwrap();
    let ts = WindowedTimeseries::new(ts, w, true);
    let fft_data = FFTData::new(&ts);
    println!("...{:?}", start.elapsed());
    // let ts = Rc::new(WindowedTimeseries::gen_randomwalk(n, w, 1243));
    println!("Computing hashes");
    let start = Instant::now();
    let hasher = Arc::new(Hasher::new(w, 4, 2.0, 123));
    let hc = HashCollection::from_ts(&ts, &fft_data, hasher);
    let n = ts.num_subsequences();
    let v_u32: Vec<(HashValue, usize)> = (0..n).map(|i| (hc.hash_value(i, 32, 0), i)).collect();
    let v_arr: Vec<([u8; 32], usize)> = (0..n)
        .map(|i| (hc.extended_hash_value(i, 0, 0), i))
        .collect();
    println!("...{:?}", start.elapsed());

    bench("Radix sort u32", &v_u32, |mut v| v.radix_sort());
    bench("Rust unstable sort u32", &v_u32, |mut v| v.sort_unstable());
    bench("Rust stable sort u32", &v_u32, |mut v| v.sort());
    bench("Rayon unstable sort u32", &v_u32, |mut v| {
        v.par_sort_unstable()
    });

    bench("Radix sort arr", &v_arr, |mut v| v.radix_sort());
    bench("Rust unstable sort arr", &v_arr, |mut v| v.sort_unstable());
    bench("Rust stable sort arr", &v_arr, |mut v| v.sort());
    bench("Rayon unstable sort arr", &v_arr, |mut v| {
        v.par_sort_unstable()
    });
}
