
pub struct WindowedTimeseries {
    data: Vec<f64>,
    pub w: usize,
    rolling_avg: Vec<f64>,
    rolling_sd: Vec<f64>,
}

impl WindowedTimeseries {
    pub fn new(ts: Vec<f64>, w: usize) -> Self {
        let mut rolling_avg = Vec::with_capacity(ts.len() - w);
        let mut rolling_sd = Vec::with_capacity(ts.len() - w);

        let mut s: f64 = ts[0..w].iter().sum();
        let mut s2: f64 = ts[0..w].iter().map(|x| x * x).sum();
        rolling_avg.push(s / w as f64);
        rolling_sd.push((s2 - s * s) / w as f64);

        for i in 1..ts.len() - w {
            s = s - ts[i - 1] + ts[i + w];
            s2 = s2 - ts[i - 1].powi(2) + ts[i + w].powi(2);
            rolling_avg.push(s / w as f64);
            rolling_sd.push((s2 - s * s) / w as f64);
        }

        WindowedTimeseries {
            data: ts,
            w,
            rolling_avg,
            rolling_sd,
        }
    }

    pub fn subsequence<'a>(&'a self, i: usize) -> &'a [f64] {
        &self.data[i..i+self.w]
    }

    pub fn mean(&self, i: usize) -> f64 {
        self.rolling_avg[i]
    }

    pub fn sd(&self, i: usize) -> f64 {
        self.rolling_sd[i]
    }

    pub fn num_subsequences(&self) -> usize {
        self.data.len() - self.w
    }

    pub fn distance_profile<D: Fn(&WindowedTimeseries, usize, usize) -> f64>(&self, from: usize, d: D) -> Vec<f64> {
        let mut dp = vec![0.0; self.num_subsequences()];

        for i in 0..self.num_subsequences() {
            dp[i] = d(self, from, i);
        }

        dp
    }
}

