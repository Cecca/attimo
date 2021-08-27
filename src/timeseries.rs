use crate::distance::dot;
use deepsize::DeepSizeOf;
use rand_distr::num_traits::Zero;
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::{cell::RefCell, fmt::Display, mem::size_of, sync::Arc};

pub struct WindowedTimeseries {
    pub data: Vec<f64>,
    pub w: usize,
    rolling_avg: Vec<f64>,
    rolling_sd: Vec<f64>,
    /// The squared norm of the z-normalized vector, to speed up the computation of the euclidean distance
    squared_norms: Vec<f64>,
    fft_length: usize,
    /// We maintain the fft transform for computing dot products in chunks
    /// of size `fft_length` (or the smallest power of 2 after w, whichever largest), in order to
    /// be able to tile the computation of the dot products.
    fft_chunks: Vec<Vec<Complex<f64>>>,
    fftfun: Arc<dyn Fft<f64>>,
    ifftfun: Arc<dyn Fft<f64>>,
    buf_vfft: RefCell<Vec<Complex<f64>>>,
    buf_ivfft: RefCell<Vec<Complex<f64>>>,
}

impl WindowedTimeseries {
    pub fn new(ts: Vec<f64>, w: usize) -> Self {
        assert!(w <= ts.len());
        let n_subs = ts.len() - w;
        let mut rolling_avg = Vec::with_capacity(n_subs);
        let mut rolling_sd = Vec::with_capacity(n_subs);
        let mut squared_norms = Vec::with_capacity(n_subs);

        // FIXME: compute it using sliding window
        let mut buffer = vec![0.0; w];
        for i in 0..n_subs {
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

        //// Now we pre-compute the FFT of the time series, which will be needed to compute
        //// dot products for the hash values faster.
        let fft_length = std::cmp::min(
            std::cmp::max(1 << 14, w.next_power_of_two()),
            ts.len().next_power_of_two(),
        );
        let mut fft_chunks = Vec::new();

        let mut planner = FftPlanner::new();
        let fftfun = planner.plan_fft_forward(fft_length);
        let ifftfun = planner.plan_fft_inverse(fft_length);

        let mut begin = 0;
        while begin < ts.len() {
            let end = std::cmp::min(begin + fft_length, ts.len());
            let mut chunk: Vec<Complex<f64>> = ts[begin..end]
                .iter()
                .map(|x| Complex { re: *x, im: 0.0 })
                .collect();
            chunk.resize(fft_length, Complex::zero());
            fftfun.process(&mut chunk);
            fft_chunks.push(chunk);

            //// We shift the window taking into account the width of the window `w`,
            //// since we need to compute all dot products
            begin += fft_length - w;
        }
        println!(
            "FFT chunk length {}, for a total of {} chunks",
            fft_length,
            fft_chunks.len()
        );

        WindowedTimeseries {
            data: ts,
            w,
            rolling_avg,
            rolling_sd,
            squared_norms,
            fft_length,
            fft_chunks,
            fftfun,
            ifftfun,
            buf_vfft: RefCell::new(vec![Complex::zero(); fft_length]),
            buf_ivfft: RefCell::new(vec![Complex::zero(); fft_length]),
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
            ts[i] = ts[i - 1] + ts[i];
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
        // let n = self.data_fft.len();
        assert!(v.len() == self.w);

        let stride = self.fft_length - self.w;

        //// Pre-allocate the output
        output.clear();
        output.resize(self.num_subsequences(), 0.0);

        //// Get local scratch vectors, to avoid allocations
        let mut vfft = self.buf_vfft.borrow_mut();
        let mut ivfft = self.buf_ivfft.borrow_mut();

        //// Then compute the FFT of the reversed input vector, padded with zeros
        vfft.fill(Complex::zero());
        for (i, &x) in v.iter().enumerate() {
            vfft[self.w - i - 1] = Complex { re: x, im: 0.0 };
        }
        self.fftfun.process(&mut vfft);

        //// Iterate over the chunks
        for (chunk_idx, chunk) in self.fft_chunks.iter().enumerate() {
            ivfft.fill(Complex::zero());
            //// Then compute the element-wise multiplication between the dot products, inplace
            for i in 0..self.fft_length {
                ivfft[i] = vfft[i] * chunk[i];
            }

            //// And go back to the time domain
            self.ifftfun.process(&mut ivfft);

            //// Copy the values to the output buffer, rescaling on the go (`rustfft`
            //// does not perform normalization automatically)
            let offset = chunk_idx * stride;
            for i in 0..(self.fft_length - self.w) {
                if i + offset < self.num_subsequences() {
                    output[i + offset] =
                        ivfft[(i + v.len() - 1) % self.fft_length].re / self.fft_length as f64
                }
            }
        }
    }

    pub fn sliding_dot_product_slow(&self, v: &[f64], output: &mut Vec<f64>) {
        assert!(v.len() == self.w);
        //// Pre-allocate the output
        output.clear();
        output.resize(self.num_subsequences(), 0.0);

        for i in 0..self.num_subsequences() {
            output[i] = dot(v, self.subsequence(i));
        }
    }

    //// This function allows to compute the sliding dot product between the input vector
    //// and the z-normalized subsequences of the time series. Note that the input
    //// is not z-normalized by this function.
    pub fn znormalized_sliding_dot_product(&self, v: &[f64], output: &mut Vec<f64>) {
        self.sliding_dot_product(v, output);
        let sumv: f64 = v.iter().sum();
        for i in 0..self.num_subsequences() {
            let m = self.mean(i);
            let sd = self.sd(i);
            output[i] = output[i] / sd - sumv * m / sd;
        }
    }

    pub fn znormalized_sliding_dot_product_slow(&self, v: &[f64], output: &mut Vec<f64>) {
        assert!(v.len() == self.w);
        //// Pre-allocate the output
        output.clear();
        output.resize(self.num_subsequences(), 0.0);

        let mut buffer = vec![0.0; self.w];

        for i in 0..self.num_subsequences() {
            self.znormalized(i, &mut buffer);
            output[i] = dot(&v, &buffer);
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

impl<T: DeepSizeOf> BytesSize for T {
    fn bytes_size(&self) -> PrettyBytes {
        PrettyBytes(self.deep_size_of())
    }
}

impl DeepSizeOf for WindowedTimeseries {
    fn deep_size_of_children(&self, context: &mut deepsize::Context) -> usize {
        self.w.deep_size_of()
            + self.data.deep_size_of_children(context)
            + self.rolling_avg.deep_size_of_children(context)
            + self.rolling_sd.deep_size_of_children(context)
            + self.squared_norms.deep_size_of_children(context)
            + size_of::<Complex<f64>>() * self.fft_length * self.fft_chunks.len()
    }
}

#[cfg(test)]
mod test {
    use crate::timeseries::WindowedTimeseries;

    #[test]
    fn test_sliding_dot_product() {
        use rand::prelude::*;
        use rand_distr::StandardNormal;
        use rand_xoshiro::Xoroshiro128Plus;

        for n in [10, 100, 1234, 4000, 50000] {
            for w in [3, 100, 200, 500] {
                if w < n {
                    println!("n={}, w={}", n, w);
                    let ts = WindowedTimeseries::gen_randomwalk(n, w, 12345);

                    let rng = Xoroshiro128Plus::seed_from_u64(12344);
                    let v: Vec<f64> = rng.sample_iter(StandardNormal).take(w).collect();
                    let mut expected = vec![0.0; ts.num_subsequences()];
                    let mut actual = vec![0.0; ts.num_subsequences()];

                    ts.sliding_dot_product_slow(&v, &mut expected);
                    ts.sliding_dot_product(&v, &mut actual);
                    // dump_vec(format!("/tmp/sliding-expected-{}-{}.txt", n, w), &expected);
                    // dump_vec(format!("/tmp/sliding-actual-{}-{}.txt", n, w), &actual);

                    assert!(expected
                        .into_iter()
                        .zip(actual.into_iter())
                        .all(|(e, a)| (e - a).abs() <= 0.0000001));
                }
            }
        }
    }

    #[allow(dead_code)]
    fn dump_vec<D: std::fmt::Display>(fname: String, v: &[D]) {
        use std::io::prelude::*;
        let mut f = std::fs::File::create(fname).unwrap();
        for x in v.iter() {
            writeln!(f, "{}", x).unwrap();
        }
    }

    #[test]
    fn test_znormalized_sliding_dot_product() {
        use rand::prelude::*;
        use rand_distr::StandardNormal;
        use rand_xoshiro::Xoroshiro128Plus;

        for n in [10, 100, 1234, 4000, 50000] {
            for w in [3, 100, 200, 500] {
                if w < n {
                    println!("n={}, w={}", n, w);
                    let ts = WindowedTimeseries::gen_randomwalk(n, w, 12345);

                    let rng = Xoroshiro128Plus::seed_from_u64(12344);
                    let v: Vec<f64> = rng.sample_iter(StandardNormal).take(w).collect();
                    let mut expected = vec![0.0; ts.num_subsequences()];
                    let mut actual = vec![0.0; ts.num_subsequences()];

                    ts.znormalized_sliding_dot_product_slow(&v, &mut expected);
                    ts.znormalized_sliding_dot_product(&v, &mut actual);
                    // dump_vec(format!("/tmp/expected-{}-{}.txt", n, w), &expected);
                    // dump_vec(format!("/tmp/actual-{}-{}.txt", n, w), &actual);

                    assert!(expected
                        .into_iter()
                        .zip(actual.into_iter())
                        .all(|(e, a)| (e - a).abs() <= 0.00001));
                }
            }
        }
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
}
