use std::{rc::Rc, time::Instant};

use attimo::{
    lsh::{HashCollection, HashValue, Hasher},
    sort::RadixSort,
    timeseries::WindowedTimeseries,
};

fn main() {
    let w = 500;
    let ts = Rc::new(WindowedTimeseries::gen_randomwalk(1000000, w, 12345));
    let h = Hasher::new(w, 200, 10.0, 12345);
    let hasher = HashCollection::from_ts(Rc::clone(&ts), &h);
    let mut hashes: Vec<HashValue> = (0..ts.num_subsequences())
        .map(|i| hasher.hash_value(i, 0))
        .collect();

    let start = Instant::now();
    hashes.radix_sort();
    let elapsed = start.elapsed();
    println!(
        "Sorted {} hashes in {:?} ({} elems/s)",
        hashes.len(),
        elapsed,
        hashes.len() as f64 / elapsed.as_secs_f64()
    );
}
