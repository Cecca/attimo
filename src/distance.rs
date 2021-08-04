use crate::types::*;

pub fn euclidean(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = x - y;
        s += d * d;
    }
    s.sqrt()
}

pub fn eucl(ts: &WindowedTimeseries, i: usize, j: usize) -> f64 {
    let mut s = 0.0;
    for (&x, &y) in ts.subsequence(i).iter().zip(ts.subsequence(j).iter()) {
        let d = x - y;
        s += d * d;
    }
    s.sqrt()
}

pub fn zeucl(ts: &WindowedTimeseries, i: usize, j: usize) -> f64 {
    // let dotp = dot(ts.subsequence(i), ts.subsequence(j));
    // let meanp = ts.w as f64 * ts.mean(i) * ts.mean(j);
    // let sdp = (ts.w as f64 * ts.sd(i) * ts.sd(j));
    // (2.0 * ts.w as f64 * ((dotp - meanp) / sdp)).sqrt()
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

pub fn dot_slow(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        s += x * y;
    }
    s
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
    let ts = WindowedTimeseries::new(ts, w);

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
        for j in 0..ts.num_subsequences() {
            let mut za = Vec::new();
            let mut zb = Vec::new();
            ts.znormalized(i, &mut za);
            ts.znormalized(j, &mut zb);
            let expected = euclidean(&za, &zb);
            let actual = zeucl(&ts, i, j);
            assert_eq!(expected, actual);
        }
    }
}

pub fn normalize(x: &[f64]) -> Vec<f64> {
    let norm = crate::distance::norm(x);
    let mut y = vec![0.0; x.len()];
    for (i, xi) in x.iter().enumerate() {
        y[i] = xi / norm;
    }
    y
}

