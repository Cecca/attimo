use std::{fmt::Display};

use crate::distance::{dot};

pub struct WindowedTimeseries {
    pub data: Vec<f64>,
    pub w: usize,
    rolling_avg: Vec<f64>,
    rolling_sd: Vec<f64>,
    /// The squared norm of the z-normalized vector, to speed up the computation of the euclidean distance
    squared_norms: Vec<f64>,
}

impl WindowedTimeseries {
    pub fn new(ts: Vec<f64>, w: usize) -> Self {
        let mut rolling_avg = Vec::with_capacity(ts.len() - w);
        let mut rolling_sd = Vec::with_capacity(ts.len() - w);
        let mut squared_norms = Vec::with_capacity(ts.len() - w);

        // FIXME: compute it using sliding window
        let mut buffer = vec![0.0; w];
        for i in 0..ts.len() - w {
            let mean = ts[i..i + w].iter().sum::<f64>() / w as f64;
            let sd =
                ((ts[i..i + w].iter().map(|x| x * x).sum::<f64>() - mean * mean) / w as f64).sqrt();
            buffer.fill(0.0);
            for (i, x) in ts[i..i + w].iter().enumerate() {
                buffer[i] = (x - mean) / sd;
            }
            rolling_avg.push(mean);
            rolling_sd.push(sd);
            squared_norms.push(dot(&buffer, &buffer));
        }

        WindowedTimeseries {
            data: ts,
            w,
            rolling_avg,
            rolling_sd,
            squared_norms,
        }
    }

    //// We have the possiblity of generating a random walk windowed 
    //// time series for testing purposes
    pub fn gen_randomwalk(n: usize, w: usize, seed: u64) -> Self {
        use rand::prelude::*;
        use rand_distr::StandardNormal;
        use rand_xoshiro::Xoroshiro128Plus;
        let rng = Xoroshiro128Plus::seed_from_u64(seed);
        let mut ts: Vec<f64> = rng.sample_iter(StandardNormal).take(n).collect();
        for i in 1..n {
            ts[i] = ts[i-1] + ts[i];
        }
        Self::new(ts, w)
    }

    pub fn subsequence<'a>(&'a self, i: usize) -> &'a [f64] {
        &self.data[i..i + self.w]
    }

    pub fn znormalized(&self, i: usize, output: &mut Vec<f64>) {
        output.resize(self.w, 0.0);
        let m = self.mean(i);
        let s = self.sd(i);
        for (i, &x) in self.subsequence(i).iter().enumerate() {
            output[i] = (x - m) / s;
        }
    }

    pub fn mean(&self, i: usize) -> f64 {
        self.rolling_avg[i]
    }

    pub fn sd(&self, i: usize) -> f64 {
        self.rolling_sd[i]
    }

    pub fn squared_norm(&self, i: usize) -> f64 {
        self.squared_norms[i]
    }

    pub fn num_subsequences(&self) -> usize {
        self.data.len() - self.w
    }

    pub fn sliding_dot_product(&self, v: &[f64], output: &mut Vec<f64>) {
        assert!(v.len() == self.w);
        //// Pre-allocate the output
        output.clear();
        output.resize(self.num_subsequences(), 0.0);

        for i in 0..self.num_subsequences() {
            output[i] = dot(v, self.subsequence(i));
        }
    }

    pub fn znormalized_sliding_dot_product(&self, v: &[f64], output: &mut Vec<f64>) {
        assert!(v.len() == self.w);
        //// Pre-allocate the output
        output.clear();
        output.resize(self.num_subsequences(), 0.0);

        let mut buffer = vec![0.0; self.w];
        let (mv, sv) = meansd(v);
        let mut vnorm = vec![0.0; self.w];
        for i in 0..v.len() {
            vnorm[i] = (v[i] - mv) / sv;
        }

        for i in 0..self.num_subsequences() {
            self.znormalized(i, &mut buffer);
            output[i] = dot(&vnorm, &buffer);
        }
    }

    pub fn distance_profile<D: Fn(&WindowedTimeseries, usize, usize) -> f64>(
        &self,
        from: usize,
        d: D,
    ) -> Vec<f64> {
        let mut dp = vec![0.0; self.num_subsequences()];

        for i in 0..self.num_subsequences() {
            dp[i] = d(self, from, i);
            assert!(!dp[i].is_nan());
        }
        dp
    }
}

fn meansd(v: &[f64]) -> (f64, f64) {
    let mean: f64 = v.iter().sum::<f64>() / v.len() as f64;
    let sd = ((v.iter().map(|x| x * x).sum::<f64>() - mean * mean) / v.len() as f64).sqrt();
    (mean, sd)
}

#[test]
fn test_meanstd() {
    use rand::prelude::*;
    use rand_distr::Uniform;
    use rand_xoshiro::Xoroshiro128Plus;
    let rng = Xoroshiro128Plus::seed_from_u64(3462);
    let ts: Vec<f64> = rng.sample_iter(Uniform::new(0.0, 1.0)).take(1000).collect();
    let w = 100;
    let ts = WindowedTimeseries::new(ts, w);

    for i in 0..ts.num_subsequences() {
        let a = ts.subsequence(i);
        let mean: f64 = a.iter().sum::<f64>() / a.len() as f64;
        let actual_mean = ts.mean(i);
        let sd = ((a.iter().map(|x| x * x).sum::<f64>() - mean * mean) / a.len() as f64).sqrt();
        let actual_sd = ts.sd(i);
        assert_eq!(mean, actual_mean);
        assert_eq!(sd, actual_sd);
    }
}

pub struct PrettyBytes(pub usize);

impl Display for PrettyBytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0 >= 1024 * 1024 * 1024 {
            write!(f, "{} Gbytes", self.0 / (1024 * 1024 * 1024))
        } else if self.0 >= 1024 * 1024 {
            write!(f, "{} Mbytes", self.0 / (1024 * 1024))
        } else if self.0 >= 1024 {
            write!(f, "{} Kbytes", self.0 / 1024)
        } else {
            write!(f, "{} bytes", self.0)
        }
    }
}

pub trait BytesSize {
    fn bytes_size(&self) -> PrettyBytes;
}

impl BytesSize for WindowedTimeseries {
    fn bytes_size(&self) -> PrettyBytes {
        PrettyBytes(8 * (self.data.len() + (self.num_subsequences() * 3)))
    }
}

impl<T> BytesSize for Vec<T> {
    fn bytes_size(&self) -> PrettyBytes {
        if self.is_empty() {
            PrettyBytes(0)
        } else {
            PrettyBytes(self.len() * std::mem::size_of::<T>())
        }
    }
}
