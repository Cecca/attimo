use crate::types::*;

pub fn eucl(ts: &WindowedTimeseries, i: usize, j: usize) -> f64 {
    let mut s = 0.0;
    for (&x, &y) in ts.subsequence(i).iter().zip(ts.subsequence(j).iter()) {
        let d = x - y;
        s += d*d;
    }
    s.sqrt()
}


pub fn zeucl(ts: &WindowedTimeseries, i: usize, j: usize) -> f64 {
    let mut s = 0.0;
    let mi = ts.mean(i);
    let mj = ts.mean(j);
    let si = ts.sd(i);
    let sj = ts.sd(j);
    for (&x, &y) in ts.subsequence(i).iter().zip(ts.subsequence(j).iter()) {
        let d = ((x - mi) / si) - ((y - mj) / sj);
        s += d*d;
    }
    s.sqrt()
}

pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        s += x * y;
    }
    s
}

pub fn norm(a: &[f64]) -> f64 {
    dot(a, a)
}

