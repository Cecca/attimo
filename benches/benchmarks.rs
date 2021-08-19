use attimo::{lsh::*, types::WindowedTimeseries};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

pub fn bench_construct_ts(c: &mut Criterion) {
    use rand::prelude::*;
    use rand_distr::StandardNormal;
    use rand_xoshiro::Xoroshiro128Plus;

    let n = 10000;

    let rng = Xoroshiro128Plus::seed_from_u64(12344);
    let mut ts: Vec<f64> = rng.sample_iter(StandardNormal).take(n).collect();
    for i in 1..n {
        ts[i] = ts[i - 1] + ts[i];
    }

    let w = 200;

    c.bench_function("construct windowed time series", |b| {
        b.iter(|| WindowedTimeseries::new(ts.clone(), black_box(w)))
    });
}

pub fn bench_sliding_dot_product(c: &mut Criterion) {
    use rand::prelude::*;
    use rand_distr::StandardNormal;
    use rand_xoshiro::Xoroshiro128Plus;

    let mut group = c.benchmark_group("Sliding dot product");

    let w = 400;

    for n in [1, 2, 3, 4, 5, 6] {
        let n = n*10000;
        group.bench_with_input(
            BenchmarkId::new("sliding dot product slow", n),
            &n,
            |b, n| {
                let ts = WindowedTimeseries::gen_randomwalk(*n, w, 12345);

                let rng = Xoroshiro128Plus::seed_from_u64(12344);
                let v: Vec<f64> = rng.sample_iter(StandardNormal).take(w).collect();
                let mut output = vec![0.0; ts.num_subsequences()];
                b.iter(|| ts.sliding_dot_product_slow(&v, &mut output))
            },
        );
        group.bench_with_input(
            BenchmarkId::new("sliding dot product fast", n),
            &n,
            |b, n| {
                let ts = WindowedTimeseries::gen_randomwalk(*n, w, 12345);

                let rng = Xoroshiro128Plus::seed_from_u64(12344);
                let v: Vec<f64> = rng.sample_iter(StandardNormal).take(w).collect();
                let mut output = vec![0.0; ts.num_subsequences()];
                b.iter(|| ts.sliding_dot_product(&v, &mut output))
            },
        );
    }
    group.finish();
}

pub fn bench_hash_ts(c: &mut Criterion) {
    let mut group = c.benchmark_group("hash-time-series");
    // Configure Criterion.rs to detect smaller differences and increase sample size to improve
    // precision and counteract the resulting noise.
    group.sample_size(10);
    let w = 200;
    let ts = WindowedTimeseries::gen_randomwalk(10000, w, 12345);
    let hasher = Hasher::new(w, 32, 200, 10.0, 12345);
    group.bench_function("hash time series", |b| {
        b.iter(|| HashCollection::from_ts(&ts, &hasher))
    });
    group.finish()
}

criterion_group!(
    benches,
    bench_sliding_dot_product,
    bench_construct_ts,
    bench_hash_ts
);
criterion_main!(benches);
