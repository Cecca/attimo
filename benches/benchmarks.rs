
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use attimo::{lsh::*, types::WindowedTimeseries};

pub fn bench_construct_ts(c: &mut Criterion) {
    use rand::prelude::*;
    use rand_distr::StandardNormal;
    use rand_xoshiro::Xoroshiro128Plus;

    let n = 10000;

    let rng = Xoroshiro128Plus::seed_from_u64(12344);
    let mut ts: Vec<f64> = rng.sample_iter(StandardNormal).take(n).collect();
    for i in 1..n {
        ts[i] = ts[i-1] + ts[i];
    }

    let w = 200;

    c.bench_function("construct windowed time series", |b| b.iter(|| WindowedTimeseries::new(ts.clone(), black_box(w))));
}

pub fn bench_hash_ts(c: &mut Criterion) {
    let w = 200;
    let ts = WindowedTimeseries::gen_randomwalk(10000, w, 12345);
    let hasher = Hasher::new(w, 32, 200, 10.0, 12345);
    c.bench_function("hash time series", |b| b.iter(|| HashCollection::from_ts(&ts, &hasher)));
}

criterion_group!(benches, bench_construct_ts, bench_hash_ts);
criterion_main!(benches);