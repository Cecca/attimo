use crate::distance::{dot, norm};

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

        // let mut avg: f64 = ts[0..w].iter().sum::<f64>() / w as f64;
        // let mut var: f64 = (ts[0..w].iter().map(|x| x*x).sum::<f64>() - avg*avg) / w as f64;

        // rolling_avg.push(avg);
        // assert!(!var.is_nan() && var.is_finite());
        // rolling_sd.push((var.sqrt()) / w as f64);

        // for i in 1..ts.len() - w {
        //     let new = ts[i + w];
        //     let old = ts[i - 1];
        //     let oldavg = avg;
        //     avg = oldavg + (new - old) / w as f64;
        //     var = (new - old) * (new - avg + old - oldavg) / w as f64;
        //     rolling_avg.push(avg);
        //     assert!(!var.is_nan() && var.is_finite());
        //     rolling_sd.push(var.sqrt());
        // }
        let mut buffer = vec![0.0; w];
        for i in 0..ts.len() - w {
            let mean = ts[i..i+w].iter().sum::<f64>() / w as f64;
            let sd = ((ts[i..i+w].iter().map(|x| x*x).sum::<f64>() - mean*mean) / w as f64).sqrt();
            buffer.fill(0.0);
            for (i, x) in ts[i..i+w].iter().enumerate() {
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

#[test]
fn test_meanstd() {
    use rand::prelude::*;
    use rand_xoshiro::Xoroshiro128Plus;
    use rand_distr::Uniform;
    let rng = Xoroshiro128Plus::seed_from_u64(3462);
    let ts: Vec<f64> = rng.sample_iter(Uniform::new(0.0, 1.0)).take(1000).collect();
    let w = 100;
    let ts = WindowedTimeseries::new(ts, w);

    for i in 0..ts.num_subsequences() {
        let a = ts.subsequence(i);
        let mean: f64 = a.iter().sum::<f64>() / a.len() as f64;
        let actual_mean = ts.mean(i);
        let sd = ((a.iter().map(|x| x*x).sum::<f64>() - mean*mean) / a.len() as f64).sqrt();
        let actual_sd = ts.sd(i);
        assert_eq!(mean, actual_mean);
        assert_eq!(sd, actual_sd);
    }
}
