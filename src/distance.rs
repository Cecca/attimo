use crate::types::*;

pub fn eucl(ts: &WindowedTimeseries, i: usize, j: usize) -> f64 {
    let mut s = 0.0;
    for (&x, &y) in ts.subsequence(i).iter().zip(ts.subsequence(j).iter()) {
        let d = x - y;
        s += d * d;
    }
    s.sqrt()
}

pub fn zeucl(ts: &WindowedTimeseries, i: usize, j: usize) -> f64 {
    let dotp = dot(ts.subsequence(i), ts.subsequence(j));
    let meanp = ts.w as f64 * ts.mean(i) * ts.mean(j);
    let sdp = (ts.w as f64 * ts.sd(i) * ts.sd(j));
    (2.0 * ts.w as f64 * ((dotp - meanp) / sdp)).sqrt()
    // let mut s = 0.0;
    // let mi = ts.mean(i);
    // let mj = ts.mean(j);
    // let si = ts.sd(i);
    // let sj = ts.sd(j);
    // for (&x, &y) in ts.subsequence(i).iter().zip(ts.subsequence(j).iter()) {
    //     let d = ((x - mi) / si) - ((y - mj) / sj);
    //     s += d * d;
    // }
    // s.sqrt()
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
    dot(a, a)
}
