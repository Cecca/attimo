use std::{sync::Arc, collections::HashMap};
use std::time::Instant;
use anyhow::Result;
use attimo::{timeseries::{WindowedTimeseries, FFTData}, load::loadts, lsh::{HashCollection, Hasher}};
use xxhash_rust::xxh32::xxh32;

fn hash_xxh32(pool: &HashCollection, i: usize, repetition: usize, prefix: usize) -> u32 {
    let mut hv: [u8; 32] = [0; 32];
    let l = &pool.left(i, repetition);
    let r = &pool.right(i, repetition);
    for h in 0..(prefix/2) {
        hv[2*h] = l[h];
        hv[2*h+1] = r[h];
    }
    xxh32(&hv[..prefix], 1234)
}

fn hash_default2(pool: &HashCollection, i: usize, repetition: usize, prefix: usize) -> u32 {
    use std::hash::Hasher;
    let mut hv: [u8; 32] = [0; 32];
    let mut hasher = std::collections::hash_map::DefaultHasher::default();
    let l = &pool.left(i, repetition);
    let r = &pool.right(i, repetition);
    for h in 0..(prefix/2) {
        hv[2*h] = l[h];
        hv[2*h+1] = r[h];
    }
    hasher.write(&hv[..prefix]);
    hasher.finish() as u32
}

fn hash_default(pool: &HashCollection, i: usize, repetition: usize, prefix: usize) -> u32 {
    use std::hash::Hasher;
    let mut hasher = std::collections::hash_map::DefaultHasher::default();
    let (k_left, k_right) = HashCollection::k_pair(prefix);
    let l = &pool.left(i, repetition);
    let r = &pool.right(i, repetition);
    let mut h = 0;
    while h < k_left || h < k_right {
        if h < k_left {
            hasher.write_u8(l[h]);
        }
        if h < k_right {
            hasher.write_u8(r[h]);
        }
        h += 1;
    }
    hasher.finish() as u32
}

fn bench<F: Fn(&HashCollection, usize, usize, usize) -> u32>(name: &str, ns: usize, pools: &HashCollection, repetition: usize, prefix: usize, func: F) {
    let t = Instant::now();
    let hash_values: Vec<u32> = (0..ns).map(|i| func(pools, i, repetition, prefix)).collect();
    let d = t.elapsed();
    let mut collisions = HashMap::new();
    for h in hash_values {
        collisions.entry(h).and_modify(|c| *c += 1usize).or_insert(1);
    }
    let largest = collisions.values().max().unwrap();
    println!("{},{},{},{},{}", name, prefix, d.as_secs_f64(), ns as f64 / d.as_secs_f64(), largest);
}

fn main() -> Result<()> {
    let ts = WindowedTimeseries::new(loadts("data/HumanY.txt.gz", None)?, 18000, false);
    // let ts = WindowedTimeseries::new(loadts("data/ECG.csv.gz", None)?, 1000, false);
    let fft_data = FFTData::new(&ts);
    let repetitions = 2;
    let hasher_width = 2.0;
    let seed = 1234;
    let hasher = Arc::new(Hasher::new(ts.w, repetitions, hasher_width, seed));
    let pools = HashCollection::from_ts(&ts, &fft_data, hasher);

    let ns = ts.num_subsequences();

    println!("type,prefix,time_s,throughput_s,largest");
    for prefix in 4..32 {
        bench("default", ns, &pools, 0, prefix, hash_default);
        bench("default2", ns, &pools, 0, prefix, hash_default2);
        bench("xxh32", ns, &pools, 0, prefix, hash_xxh32);
    }

    Ok(())
}
