
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use attimo::{lsh::*, types::WindowedTimeseries};

pub fn bench_hash_ts(c: &mut Criterion) {
    let w = 200;
    let ts = WindowedTimeseries::gen_randomwalk(1000, w, 12345);
    let hasher = Hasher::new(w, 32, 100, 10.0, 12345);
    c.bench_function("hash time series", |b| b.iter(|| HashCollection::from_ts(&ts, &hasher)));
}

criterion_group!(benches, bench_hash_ts);
criterion_main!(benches);