use std::cmp::Reverse;
use std::rc::Rc;

use attimo::distance::zdot;
use attimo::sort::*;
use attimo::{lsh::*, timeseries::WindowedTimeseries};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Uniform};
use rand_xoshiro::{Xoroshiro128Plus, Xoroshiro128PlusPlus};

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
                let fft_data = ts.fft_data();

                let rng = Xoroshiro128Plus::seed_from_u64(12344);
                let v: Vec<f64> = rng.sample_iter(StandardNormal).take(w).collect();
                let mut output = vec![0.0; ts.num_subsequences()];
                b.iter(|| ts.sliding_dot_product(&v, &fft_data, &mut output))
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
    let ts = WindowedTimeseries::gen_randomwalk(1000000, w, 12345);
    let hasher = Hasher::new(w, 200, 10.0, 12345);
    group.bench_function("hash time series", |b| {
        b.iter(|| HashCollection::from_ts(&ts, &hasher))
    });
    group.finish()
}

pub fn bench_sort_u8(c: &mut Criterion) {
    use rand::prelude::*;
    use rand_distr::Uniform;
    use rand_xoshiro::Xoshiro256PlusPlus;

    let mut group = c.benchmark_group("sorting u8");
    let rng = Xoshiro256PlusPlus::seed_from_u64(1234);
    let vals: Vec<u8> = Uniform::new(0, u8::MAX)
        .sample_iter(rng)
        .take(10000000)
        .collect();

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

pub fn bench_sort_usize(c: &mut Criterion) {
    use rand::prelude::*;
    use rand_distr::Uniform;
    use rand_xoshiro::Xoshiro256PlusPlus;

    let mut group = c.benchmark_group("sorting usize");
    let rng = Xoshiro256PlusPlus::seed_from_u64(1234);
    let vals: Vec<usize> = Uniform::new(0, usize::MAX)
        .sample_iter(rng)
        .take(10000000)
        .collect();

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
    let ts = WindowedTimeseries::gen_randomwalk(1000000, w, 12345);
    let h = Hasher::new(w, 200, 10.0, 12345);
    let hasher = HashCollection::from_ts(&ts, &h);
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

pub fn bench_sort_uniform_hashes(c: &mut Criterion) {
    let mut group = c.benchmark_group("sorting hashes uniform");
    let n = 1000000;
    let mut rng = Xoroshiro128PlusPlus::seed_from_u64(1234);
    let uniform = Uniform::new(i8::MIN, i8::MAX);
    let hashes: Vec<HashValue> = (0..n)
        .map(|_| {
            let mut hashes: [i8; 32] = [0; 32];
            for (i, x) in uniform.sample_iter(&mut rng).take(32).enumerate() {
                hashes[i] = x;
            }
            HashValue { hashes }
        })
        .collect(); // uniform.sample_iter(&mut rng).take(n).collect();

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

pub fn bench_zdot(c: &mut Criterion) {
    let n = 3000;
    let mut rng = Xoroshiro128Plus::seed_from_u64(342);
    let a: Vec<f64> = (&mut rng)
        .sample_iter(Uniform::new(0.0, 1.0))
        .take(n)
        .collect();
    let b: Vec<f64> = (&mut rng)
        .sample_iter(Uniform::new(0.0, 1.0))
        .take(n)
        .collect();
    let ma = a.iter().sum::<f64>() / a.len() as f64;
    let mb = b.iter().sum::<f64>() / a.len() as f64;
    let sa = ((a.iter().map(|x| (x - ma).powi(2)).sum::<f64>()) / (a.len() - 1) as f64).sqrt();
    let sb = ((b.iter().map(|x| (x - mb).powi(2)).sum::<f64>()) / (b.len() - 1) as f64).sqrt();

    c.bench_function("zdot bench", move |bencher| {
        bencher.iter(|| zdot(&a, ma, sa, &b, mb, sb))
    });
}

pub fn bench_first_collision(c: &mut Criterion) {
    let repetitions = 200;
    let depth = 16;

    let w = 300;
    let ts = WindowedTimeseries::gen_randomwalk(1000000, w, 12345);

    let h = Hasher::new(w, repetitions, 10.0, 12345);
    let pools = HashCollection::from_ts(&ts, &h);
    let mut hash_matrix = pools.get_hash_matrix();
    hash_matrix.setup_hashes();

    let mut buckets = hash_matrix.buckets_vec(depth, 0);
    buckets.sort_by_key(|bucket| Reverse(bucket.0.len()));
    // Take the largest bucket
    let bucket = &buckets[0];
    let bucket: Vec<usize> = bucket.1.iter().map(|(_, i)| *i + bucket.0.start).collect();
    println!("Bucket with {} elements", bucket.len());

    c.bench_function("first collision", move |bencher| {
        bencher.iter(|| {
            for i in 0..bucket.len() {
                for j in i..bucket.len() {
                    pools.first_collision(i, j, depth);
                }
            }
        });
    });
}

criterion_group!(
    benches,
    bench_sliding_dot_product,
    bench_construct_ts,
    bench_hash_ts,
    bench_sort_hashes,
    bench_sort_uniform_hashes,
    bench_sort_usize,
    bench_sort_u8,
    bench_zdot,
    bench_first_collision
);
criterion_main!(benches);
