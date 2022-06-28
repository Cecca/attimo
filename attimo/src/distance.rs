use std::cell::RefCell;

use crate::timeseries::*;

#[cfg(test)]
pub fn euclidean(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = x - y;
        s += d * d;
    }
    s.sqrt()
}

#[allow(dead_code)]
pub fn eucl(ts: &WindowedTimeseries, i: usize, j: usize) -> f64 {
    let mut s = 0.0;
    for (&x, &y) in ts.subsequence(i).iter().zip(ts.subsequence(j).iter()) {
        let d = x - y;
        s += d * d;
    }
    s.sqrt()
}

thread_local! {
    static BUFFER_I: RefCell<Vec<f64>> = RefCell::new(Vec::new());
    static BUFFER_J: RefCell<Vec<f64>> = RefCell::new(Vec::new());
}

pub fn zeucl(ts: &WindowedTimeseries, i: usize, j: usize) -> f64 {
    if i == j {
        return 0.0;
    }
    let dotp = zdot(
        ts.subsequence(i),
        ts.mean(i),
        ts.sd(i),
        ts.subsequence(j),
        ts.mean(j),
        ts.sd(j),
    );
    // The norm of z-normalized vectors of length w is w
    (2.0 * ts.w as f64 - 2.0 * dotp).sqrt()
}

pub fn zeucl_slow(ts: &WindowedTimeseries, i: usize, j: usize) -> f64 {
    let mut s = 0.0;
    let mi = ts.mean(i);
    let mj = ts.mean(j);
    let si = ts.sd(i);
    let sj = ts.sd(j);
    for (&x, &y) in ts.subsequence(i).iter().zip(ts.subsequence(j).iter()) {
        let d = ((x - mi) / si) - ((y - mj) / sj);
        s += d * d;
    }
    s.sqrt()
}

#[cfg(test)]
pub fn dot_slow(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut s = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        s += x * y;
    }
    s
}

#[inline]
pub fn zdot(a: &[f64], ma: f64, sda: f64, b: &[f64], mb: f64, sdb: f64) -> f64 {
    if sda == 0.0 || sdb == 0.0 {
        return f64::NAN;
    }
    use packed_simd::f64x8;
    let ac = a.chunks_exact(8);
    let bc = b.chunks_exact(8);
    let rem = ac
        .remainder()
        .iter()
        .zip(bc.remainder().iter())
        .map(|(a, b)| (a - ma) * (b - mb))
        .sum::<f64>() as f64;
    let ma = f64x8::splat(ma);
    let mb = f64x8::splat(mb);
    let part = ac
        .map(f64x8::from_slice_unaligned)
        .zip(bc.map(f64x8::from_slice_unaligned))
        .map(|(a, b)| (a - ma) * (b - mb))
        .sum::<f64x8>()
        .sum() as f64;
    (part + rem) / (sda * sdb)
}

#[inline]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    use packed_simd::f64x8;
    let ac = a.chunks_exact(8);
    let bc = b.chunks_exact(8);
    let rem = ac
        .remainder()
        .iter()
        .zip(bc.remainder().iter())
        .map(|(a, b)| a * b)
        .sum::<f64>() as f64;
    let part = ac
        .map(f64x8::from_slice_unaligned)
        .zip(bc.map(f64x8::from_slice_unaligned))
        .map(|(a, b)| a * b)
        .sum::<f64x8>()
        .sum() as f64;
    part + rem
}

pub fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

#[test]
fn test_zeucl() {
    use rand::prelude::*;
    use rand_distr::Uniform;
    use rand_xoshiro::Xoroshiro128Plus;
    let rng = Xoroshiro128Plus::seed_from_u64(3462);
    let ts: Vec<f64> = rng.sample_iter(Uniform::new(0.0, 1.0)).take(1000).collect();
    let w = 100;
    let ts = WindowedTimeseries::new(ts, w, true);

    let euclidean = |a: &[f64], b: &[f64]| {
        a.iter()
            .zip(b.iter())
            .map(|(ai, bi)| {
                let d = ai - bi;
                d * d
            })
            .sum::<f64>()
            .sqrt()
    };

    for i in 0..ts.num_subsequences() {
        for j in i..ts.num_subsequences() {
            dbg!((i, j));
            let mut za = vec![0.0; w];
            let mut zb = vec![0.0; w];
            ts.znormalized(i, &mut za);
            ts.znormalized(j, &mut zb);
            let expected = euclidean(&za, &zb);
            let slow = zeucl_slow(&ts, i, j);
            let actual = zeucl(&ts, i, j);
            assert_eq!(expected, slow);
            assert!(
                (expected - actual).abs() < 0.0001,
                "distances are too different: \n\texpected={} \n\tactual={}",
                expected,
                actual
            );
        }
    }
}

#[cfg(test)]
pub fn normalize(x: &[f64]) -> Vec<f64> {
    let norm = crate::distance::norm(x);
    let mut y = vec![0.0; x.len()];
    for (i, xi) in x.iter().enumerate() {
        y[i] = xi / norm;
    }
    y
}
