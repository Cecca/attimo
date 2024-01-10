use attimo::distance::{zdot, zeucl};
use attimo::load::loadts;
use attimo::sort::*;
use attimo::timeseries::FFTData;
use attimo::{lsh::*, timeseries::WindowedTimeseries};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::{Rng, SeedableRng};
use rand_distr::Uniform;
use rand_xoshiro::Xoroshiro128Plus;

pub fn bench_construct_ts(c: &mut Criterion) {
    use rand::prelude::*;
    use rand_distr::StandardNormal;

    let n = 10000;

    let rng = Xoroshiro128Plus::seed_from_u64(12344);
    let mut ts: Vec<f64> = rng.sample_iter(StandardNormal).take(n).collect();
    for i in 1..n {
        ts[i] = ts[i - 1] + ts[i];
    }

    let w = 200;

    c.bench_function("construct windowed time series", |b| {
        b.iter(|| WindowedTimeseries::new(ts.clone(), black_box(w), false))
    });
}

pub fn bench_sliding_dot_product(c: &mut Criterion) {
    use rand::prelude::*;
    use rand_distr::StandardNormal;

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
                let fft_data = FFTData::new(&ts);

                let rng = Xoroshiro128Plus::seed_from_u64(12344);
                let v: Vec<f64> = rng.sample_iter(StandardNormal).take(w).collect();
                let mut output = vec![0.0; ts.num_subsequences()];
                b.iter(|| ts.sliding_dot_product(&fft_data, &v, &mut output))
            },
        );
    }
    group.finish();
}

pub fn bench_sort_u8(c: &mut Criterion) {
    use rand::prelude::*;
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

    group.finish()
}

pub fn bench_sort_usize(c: &mut Criterion) {
    use rand::prelude::*;
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

    group.finish()
}

pub fn bench_sort_hashes(c: &mut Criterion) {
    use rayon::prelude::*;
    let mut group = c.benchmark_group("sorting hashes");
    let w = 1000;
    let repetitions = 1;

    let ts = loadts("data/ECG.csv", None).unwrap();
    let ts = WindowedTimeseries::new(ts, w, false);
    let fft_data = FFTData::new(&ts);

    let h = Hasher::new(w, repetitions, 16.0, 12345);
    let pools = HashCollection::from_ts(&ts, &fft_data, h);
    let vals: Vec<(HashValue, u32)> = (0..ts.num_subsequences())
        .map(|i| (pools.hash_value(i, K, 0.into()), i as u32))
        .collect();
    let mut scratch: Vec<(HashValue, u32)> = Vec::new();
    scratch.resize(vals.len(), Default::default());

    group.bench_function("rust unstable sort", |b| {
        b.iter_batched(
            || vals.clone(),
            |mut vals| vals.sort_unstable(),
            criterion::BatchSize::LargeInput,
        )
    });

    // group.bench_function("radix sort", |b| {
    //     b.iter_batched(
    //         || vals.clone(),
    //         |mut vals| vals.radix_sort(),
    //         criterion::BatchSize::LargeInput,
    //     )
    // });

    group.bench_function("radix sort 9pass", |b| {
        b.iter_batched(
            || vals.clone(),
            |mut vals| sort_hash_pairs(&mut vals, &mut scratch),
            criterion::BatchSize::LargeInput,
        )
    });

    group.bench_function("parallel radix sort 9pass", |b| {
        b.iter_batched(
            || vals.clone(),
            |mut vals| par_sort_hash_pairs(&mut vals, &mut scratch),
            criterion::BatchSize::LargeInput,
        )
    });

    group.bench_function("rayon sort", |b| {
        b.iter_batched(
            || vals.clone(),
            |mut vals| vals.par_sort_unstable(),
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

pub fn bench_zeucl(c: &mut Criterion) {
    let ts = loadts("data/ECG.csv", Some(10000)).unwrap();
    let ts = WindowedTimeseries::new(ts, 1000, false);

    c.bench_function("ops/zeucl/ECG", move |bencher| {
        bencher.iter(|| zeucl(&ts, 0, 1340))
    });

    let ts = loadts("data/HumanY.txt", Some(1000000)).unwrap();
    let ts = WindowedTimeseries::new(ts, 18000, false);

    c.bench_function("ops/zeucl/HumanY", move |bencher| {
        bencher.iter(|| zeucl(&ts, 0, 130040))
    });

    let ts = loadts("data/ASTRO.csv", None).unwrap();
    let ts = WindowedTimeseries::new(ts, 100, false);

    c.bench_function("ops/zeucl/ASTRO", move |bencher| {
        bencher.iter(|| zeucl(&ts, 0, 50000))
    });
}

pub fn bench_first_collision(c: &mut Criterion) {
    let repetitions = 200;

    for depth in [32, 16] {
        let w = 1000;
        let ts = loadts("data/ECG.csv", Some(10000)).unwrap();
        let ts = WindowedTimeseries::new(ts, w, false);
        let fft_data = FFTData::new(&ts);

        let h = Hasher::new(w, repetitions, 16.0, 12345);
        let pools = HashCollection::from_ts(&ts, &fft_data, h);

        c.bench_function(
            &format!("ops/first_collision/{}/ECG/far", depth),
            |bencher| {
                bencher.iter(|| {
                    pools.first_collision(0, 1340, depth);
                });
            },
        );

        c.bench_function(
            &format!("ops/first_collision/{}/ECG/close", depth),
            |bencher| {
                bencher.iter(|| {
                    pools.first_collision(1172, 6112, depth);
                });
            },
        );

        let w = 18000;
        let ts = loadts("data/HumanY.txt", Some(1000000)).unwrap();
        let ts = WindowedTimeseries::new(ts, w, false);
        let fft_data = FFTData::new(&ts);

        let h = Hasher::new(w, repetitions, 16.0, 12345);
        let pools = HashCollection::from_ts(&ts, &fft_data, h);

        c.bench_function(
            &format!("ops/first_collision/{}/HumanY/far", depth),
            |bencher| {
                bencher.iter(|| {
                    pools.first_collision(0, 130040, depth);
                });
            },
        );

        c.bench_function(
            &format!("ops/first_collision/{}/HumanY/close", depth),
            |bencher| {
                bencher.iter(|| {
                    pools.first_collision(35154, 56012, depth);
                });
            },
        );

        let w = 100;
        let ts = loadts("data/ASTRO.csv", None).unwrap();
        let ts = WindowedTimeseries::new(ts, w, false);
        let fft_data = FFTData::new(&ts);

        let h = Hasher::new(w, repetitions, 8.0, 12345);
        let pools = HashCollection::from_ts(&ts, &fft_data, h);

        c.bench_function(
            &format!("ops/first_collision/{}/ASTRO/far", depth),
            |bencher| {
                bencher.iter(|| {
                    pools.first_collision(0, 50000, depth);
                });
            },
        );

        c.bench_function(
            &format!("ops/first_collision/{}/ASTRO/close", depth),
            |bencher| {
                bencher.iter(|| {
                    pools.first_collision(609810, 888455, depth);
                });
            },
        );
    }
}

criterion_group!(
    benches,
    // bench_sliding_dot_product,
    // bench_construct_ts,
    // bench_hash_ts,
    // bench_sort_usize,
    // bench_sort_u8,
    bench_sort_hashes // bench_zdot,
                      // bench_first_collision,
                      // bench_zeucl
);
criterion_main!(benches);
