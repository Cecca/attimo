use std::rc::Rc;

use attimo::sort::*;
use attimo::{lsh::*, timeseries::WindowedTimeseries};
use criterion::{BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main};

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

    for n in [1, 10, 100] {
        let n = n * 10000;
        group.throughput(Throughput::Elements((n - w) as u64));
        // group.bench_with_input(
        //     BenchmarkId::new("sliding dot product slow", n),
        //     &n,
        //     |b, n| {
        //         let ts = WindowedTimeseries::gen_randomwalk(*n, w, 12345);

        //         let rng = Xoroshiro128Plus::seed_from_u64(12344);
        //         let v: Vec<f64> = rng.sample_iter(StandardNormal).take(w).collect();
        //         let mut output = vec![0.0; ts.num_subsequences()];
        //         b.iter(|| ts.sliding_dot_product_slow(&v, &mut output))
        //     },
        // );
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
    group.sampling_mode(criterion::SamplingMode::Flat);
    group.sample_size(10);
    let w = 500;
    let ts = Rc::new(WindowedTimeseries::gen_randomwalk(1000000, w, 12345));
    let hasher = Hasher::new(w, 200, 10.0, 12345);
    group.bench_function("hash time series", |b| {
        b.iter(|| HashCollection::from_ts(Rc::clone(&ts), &hasher))
    });
    group.finish()
}

pub fn bench_sort_usize(c: &mut Criterion) {
    use rand::prelude::*;
    use rand_xoshiro::Xoshiro256PlusPlus;
    use rand_distr::Uniform;

    let mut group = c.benchmark_group("sorting usize");
    let rng = Xoshiro256PlusPlus::seed_from_u64(1234);
    let vals: Vec<usize> = Uniform::new(0,usize::MAX).sample_iter(rng).take(10000000).collect();

    group.bench_function("rust unstable sort", |b| {
        b.iter_batched(
            || vals.clone(),
            |mut vals| vals.sort_unstable(),
            criterion::BatchSize::LargeInput,
        )
    });

    group.bench_function("radix sort", |b| {
        b.iter_batched(
            || vals.clone(),
            |mut vals| vals.radix_sort(),
            criterion::BatchSize::LargeInput,
        )
    });

    group.finish()
}

pub fn bench_sort_hashes(c: &mut Criterion) {
    let mut group = c.benchmark_group("sorting hashes");
    let w = 500;
    let ts = Rc::new(WindowedTimeseries::gen_randomwalk(1000000, w, 12345));
    let h = Hasher::new(w, 200, 10.0, 12345);
    let hasher = HashCollection::from_ts(Rc::clone(&ts), &h);
    let hashes: Vec<HashValue> = (0..ts.num_subsequences())
        .map(|i| hasher.hash_value(i, 0))
        .collect();

    group.bench_function("rust unstable sort", |b| {
        b.iter_batched(
            || hashes.clone(),
            |mut hashes| hashes.sort_unstable(),
            criterion::BatchSize::LargeInput,
        )
    });

    group.bench_function("radix sort", |b| {
        b.iter_batched(
            || hashes.clone(),
            |mut hashes| hashes.radix_sort(),
            criterion::BatchSize::LargeInput,
        )
    });

    group.finish()
}

criterion_group!(
    benches,
    bench_sliding_dot_product,
    bench_construct_ts,
    bench_hash_ts,
    bench_sort_hashes,
    bench_sort_usize
);
criterion_main!(benches);
