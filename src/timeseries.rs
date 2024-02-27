use crate::allocator::Bytes;
use crate::distance::{zdot, zeucl};
use rand_distr::num_traits::Zero;
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::cell::RefCell;
use std::sync::Arc;
use thread_local::ThreadLocal;

pub trait Overlaps<T> {
    fn overlaps(&self, other: T, exclusion_zone: usize) -> bool;
}

impl Overlaps<usize> for usize {
    #[inline]
    fn overlaps(&self, other: usize, exclusion_zone: usize) -> bool {
        ((*self as isize - other as isize).abs() - exclusion_zone as isize) < 0isize
    }
}

impl Overlaps<u32> for u32 {
    #[inline]
    fn overlaps(&self, other: u32, exclusion_zone: usize) -> bool {
        ((*self as isize - other as isize).abs() - exclusion_zone as isize) < 0isize
    }
}

impl<T> Overlaps<&T> for T
where
    T: Overlaps<T> + Copy,
{
    #[inline]
    fn overlaps(&self, other: &T, exclusion_zone: usize) -> bool {
        self.overlaps(*other, exclusion_zone)
    }
}

impl<T> Overlaps<&[T]> for T
where
    T: for<'a> Overlaps<&'a T>,
{
    fn overlaps(&self, other: &[T], exclusion_zone: usize) -> bool {
        other.iter().any(|x| self.overlaps(x, exclusion_zone))
    }
}

/// Given the slice `other`, counts how many items are overlapping
/// with `x`, as per the type's implementation of [Overlaps].
pub fn overlap_count<T: for<'a> Overlaps<&'a T>>(
    x: &T,
    others: &[T],
    exclusion_zone: usize,
) -> usize {
    others
        .iter()
        .filter(|o| x.overlaps(o, exclusion_zone))
        .count()
}

pub struct WindowedTimeseries {
    pub data: Vec<f64>,
    pub w: usize,
    rolling_avg: Vec<f64>,
    rolling_sd: Vec<f64>,
}

impl WindowedTimeseries {
    pub fn new(ts: Vec<f64>, w: usize, precise: bool) -> Self {
        assert!(w <= ts.len());

        //// First we compute rolling statistics
        let (rolling_avg, rolling_sd) = if precise {
            // println!("Compute slow rolling statistics");
            rolling_stat_slow(&ts, w)
        } else {
            // println!("Computing fast rolling statistics");
            rolling_stat(&ts, w)
        };

        WindowedTimeseries {
            data: ts,
            w,
            rolling_avg,
            rolling_sd,
        }
    }

    pub fn memory(&self) -> Bytes {
        let m = std::mem::size_of::<f64>()
            * (self.data.len() + self.rolling_sd.len() + self.rolling_avg.len());
        Bytes(m)
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
        Self::new(ts, w, true)
    }

    pub fn subsequence<'a>(&'a self, i: usize) -> &'a [f64] {
        &self.data[i..i + self.w]
    }

    pub fn znormalized(&self, i: usize, output: &mut [f64]) {
        assert!(output.len() == self.w);
        // output.resize(self.w, 0.0);
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

    pub fn num_subsequences(&self) -> usize {
        self.data.len() - self.w
    }

    pub fn num_subsequence_pairs(&self) -> usize {
        let n = self.num_subsequences();
        n * (n - 1) / 2
    }

    pub fn maximum_distance(&self) -> f64 {
        2.0 * (self.w as f64).sqrt()
    }

    pub fn average_pairwise_distance(&self, seed: u64, exclusion_zone: usize) -> f64 {
        use rand::prelude::*;
        use rand_distr::Uniform;
        use rand_xoshiro::Xoshiro256PlusPlus;

        const SAMPLES: usize = 100000;
        let uniform = Uniform::new(0, self.num_subsequences());
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut sum = 0.0;
        let mut sampled = 0;
        while sampled < SAMPLES {
            loop {
                let i = uniform.sample(&mut rng);
                let j = uniform.sample(&mut rng);
                if !i.overlaps(j, exclusion_zone) {
                    sum += zeucl(self, i, j);
                    sampled += 1;
                    break;
                }
            }
        }

        sum / SAMPLES as f64
    }

    pub fn sliding_dot_product_for_each<F: FnMut(usize, f64)>(
        &self,
        fft_data: &FFTData,
        v: &[f64],
        mut action: F,
    ) {
        // let n = self.data_fft.len();
        assert!(v.len() == self.w);

        let stride = fft_data.fft_length - self.w;

        //// Get local scratch vectors, to avoid allocations
        let fft_length = fft_data.fft_length;
        let mut vfft = fft_data
            .buf_vfft
            .get_or(|| RefCell::new(vec![Complex::zero(); fft_length]))
            .borrow_mut();
        let mut ivfft = fft_data
            .buf_ivfft
            .get_or(|| RefCell::new(vec![Complex::zero(); fft_length]))
            .borrow_mut();
        let mut scratch = fft_data
            .scratch
            .get_or(|| {
                RefCell::new(vec![
                    Complex::zero();
                    fft_data.ifftfun.get_inplace_scratch_len()
                ])
            })
            .borrow_mut();

        //// Then compute the FFT of the reversed input vector, padded with zeros
        vfft.fill(Complex::zero());
        for (i, &x) in v.iter().enumerate() {
            vfft[self.w - i - 1] = Complex {
                re: x as f64,
                im: 0.0,
            };
        }
        fft_data
            .fftfun
            .process_with_scratch(&mut vfft, &mut scratch);

        //// Iterate over the chunks
        for (chunk_idx, chunk) in fft_data.fft_chunks.iter().enumerate() {
            ivfft.fill(Complex::zero());
            //// Then compute the element-wise multiplication between the dot products, inplace
            for i in 0..fft_data.fft_length {
                ivfft[i] = vfft[i] * chunk[i];
            }

            //// And go back to the time domain
            fft_data
                .ifftfun
                .process_with_scratch(&mut ivfft, &mut scratch);

            //// Callback with the output value, rescaling on the go (`rustfft`
            //// does not perform normalization automatically)
            let offset = chunk_idx * stride;
            for i in 0..(fft_data.fft_length - self.w) {
                if i + offset < self.num_subsequences() {
                    let val = (ivfft[(i + v.len() - 1) % fft_data.fft_length].re
                        / fft_data.fft_length as f64) as f64;
                    action(i + offset, val);
                }
            }
        }
    }

    pub fn sliding_dot_product(&self, fft_data: &FFTData, v: &[f64], output: &mut [f64]) {
        self.sliding_dot_product_for_each(fft_data, v, |i, v| output[i] = v);
    }

    #[cfg(test)]
    pub fn sliding_dot_product_slow(&self, v: &[f64], output: &mut Vec<f64>) {
        use crate::distance::dot;
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
    pub fn znormalized_sliding_dot_product(
        &self,
        fft_data: &FFTData,
        v: &[f64],
        output: &mut [f64],
    ) {
        self.znormalized_sliding_dot_product_for_each(fft_data, v, |i, val| output[i] = val);
    }

    pub fn znormalized_sliding_dot_product_for_each<F: FnMut(usize, f64)>(
        &self,
        fft_data: &FFTData,
        v: &[f64],
        mut action: F,
    ) {
        let sumv: f64 = v.iter().sum();
        self.sliding_dot_product_for_each(fft_data, v, |i, val| {
            let m = self.mean(i);
            let sd = self.sd(i);
            assert!(sd > 0.0);
            action(i, val / sd - sumv * m / sd);
        });
    }

    #[cfg(test)]
    pub fn znormalized_sliding_dot_product_slow(&self, v: &[f64], output: &mut Vec<f64>) {
        use crate::distance::dot;
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

    pub fn distance_profile(
        &self,
        fft_data: &FFTData,
        from: usize,
        out: &mut [f64],
        buf: &mut [f64],
    ) {
        assert!(out.len() == self.num_subsequences());
        assert!(buf.len() == self.w);
        // let mut dp = vec![0.0; self.num_subsequences()];
        // let mut buf = vec![0.0; self.w];

        self.znormalized(from, buf);
        self.znormalized_sliding_dot_product(&fft_data, &buf, out);

        for i in 0..self.num_subsequences() {
            if i == from {
                out[i] = 0.0;
            } else {
                debug_assert!(
                    self.w as f64 > out[i],
                    "i = {} w = {} dp[i] = {} zdot = {} zeucl = {}",
                    i,
                    self.w,
                    out[i],
                    zdot(
                        self.subsequence(from),
                        self.mean(from),
                        self.sd(from),
                        self.subsequence(i),
                        self.mean(i),
                        self.sd(i)
                    ),
                    zeucl(self, i, from)
                );
                debug_assert!(
                    (zdot(
                        self.subsequence(from),
                        self.mean(from),
                        self.sd(from),
                        self.subsequence(i),
                        self.mean(i),
                        self.sd(i)
                    ) - out[i])
                        .abs()
                        <= 0.0000000001,
                    "i = {} w = {} dp[i] = {} zdot = {} zeucl = {}",
                    i,
                    self.w,
                    out[i],
                    zdot(
                        self.subsequence(from),
                        self.mean(from),
                        self.sd(from),
                        self.subsequence(i),
                        self.mean(i),
                        self.sd(i)
                    ),
                    zeucl(self, i, from)
                );
                out[i] = (2.0 * self.w as f64 - 2.0 * out[i]).sqrt();
                assert!(!out[i].is_nan());
                debug_assert!(
                    (out[i] - zeucl(self, from, i)).abs() < 0.0001,
                    "dp[i]={} zeucl={} diff={}",
                    out[i],
                    zeucl(self, from, i),
                    (out[i] - zeucl(self, from, i))
                );
            }
        }
    }

    // #[cfg(test)]
    pub fn distance_profile_slow<D: Fn(&WindowedTimeseries, usize, usize) -> f64>(
        &self,
        from: usize,
        d: D,
    ) -> Vec<f64> {
        let mut dp = vec![0.0; self.num_subsequences()];

        for i in 0..self.num_subsequences() {
            dp[i] = d(self, from, i);
            // assert!(!dp[i].is_nan());
        }
        dp
    }
}

fn rolling_stat(ts: &[f64], w: usize) -> (Vec<f64>, Vec<f64>) {
    let n_subs = ts.len() - w;

    let mut rolling_avg = vec![0.0; n_subs];
    let mut rolling_sd = vec![0.0; n_subs];

    let chunk_size = 1_000_000;
    let chunks = n_subs / chunk_size;
    for i in 0..chunks {
        let start = i * chunk_size;
        let end = (i + 1) * chunk_size;
        _rolling_stat(
            &ts[start..(end + w)],
            w,
            &mut rolling_avg[start..end],
            &mut rolling_sd[start..end],
        );
    }
    if n_subs % chunk_size > 0 {
        let start = chunks * chunk_size;
        let end = n_subs;
        _rolling_stat(
            &ts[start..(end + w)],
            w,
            &mut rolling_avg[start..end],
            &mut rolling_sd[start..end],
        );
    }

    (rolling_avg, rolling_sd)
}

fn relative_error(a: f64, b: f64) -> f64 {
    (a - b).abs() / std::cmp::max_by(a, b, |a, b| a.partial_cmp(b).unwrap())
}

fn _rolling_stat(ts: &[f64], w: usize, rolling_avg: &mut [f64], rolling_sd: &mut [f64]) {
    let n_subs = ts.len() - w;

    let comp_d_squared = |i: usize| {
        let mean = ts[i..i + w].iter().sum::<f64>() / w as f64;
        ts[i..i + w].iter().map(|x| (x - mean).powi(2)).sum::<f64>()
    };

    let mut mean = ts[0..w].iter().sum::<f64>() / w as f64;
    let mut d_squared = ts[0..w].iter().map(|x| (x - mean).powi(2)).sum::<f64>();

    rolling_avg[0] = mean;
    rolling_sd[0] = (d_squared / w as f64).sqrt();

    for i in 1..n_subs {
        let old_mean = mean;
        let new = ts[i + w - 1];
        let old = ts[i - 1];
        mean += (new - old) / w as f64;
        let new_d_squared = d_squared + (new - old) * (new - mean + old - old_mean);
        d_squared = if new_d_squared > 0.0 {
            new_d_squared
        } else {
            comp_d_squared(i)
        };
        debug_assert!(
            comp_d_squared(i) == 0.0 || relative_error(d_squared, comp_d_squared(i)) < 0.000001,
            "({}) d_squared: rolling {} scratch {} relative error {}",
            i,
            d_squared,
            comp_d_squared(i),
            relative_error(d_squared, comp_d_squared(i))
        );

        assert!(mean.is_finite());
        debug_assert!((mean - average(&ts[i..(i + w)]).abs()) < 0.00000001);
        rolling_avg[i] = mean;

        let sd = (d_squared / w as f64).sqrt();
        debug_assert!(
            (sd - variance(&ts[i..(i + w)], mean).sqrt()).abs() < 0.0000001,
            "({}) computed sd is {}, actual is {}",
            i,
            sd,
            variance(&ts[i..i + w], mean).sqrt()
        );
        assert!(
            sd.is_finite(),
            "standard deviation is {}, d_squared {}\n{:?}",
            sd,
            d_squared,
            &ts[i..(i + w)]
        );
        rolling_sd[i] = sd;
    }
}

fn average(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn variance(v: &[f64], mean: f64) -> f64 {
    v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / v.len() as f64
}

// #[cfg(test)]
fn rolling_stat_slow(ts: &[f64], w: usize) -> (Vec<f64>, Vec<f64>) {
    let n_subs = ts.len() - w;
    let mut rolling_avg = Vec::with_capacity(n_subs);
    let mut rolling_sd = Vec::with_capacity(n_subs);

    let mut buffer = vec![0.0; w];
    for i in 0..n_subs {
        let mean = ts[i..i + w].iter().sum::<f64>() / w as f64;
        let sd = (ts[i..i + w].iter().map(|x| (x - mean).powi(2)).sum::<f64>() / w as f64).sqrt();
        buffer.fill(0.0);
        for (i, x) in ts[i..i + w].iter().enumerate() {
            buffer[i] = (x - mean) / sd;
        }
        assert!(mean.is_finite());
        assert!(sd.is_finite());
        rolling_avg.push(mean);
        rolling_sd.push(sd);
    }

    (rolling_avg, rolling_sd)
}

pub struct FFTData {
    fft_length: usize,
    /// We maintain the fft transform for computing dot products in chunks
    /// of size `fft_length` (or the smallest power of 2 after w, whichever largest), in order to
    /// be able to tile the computation of the dot products.
    fft_chunks: Vec<Vec<Complex<f64>>>,
    fftfun: Arc<dyn Fft<f64>>,
    ifftfun: Arc<dyn Fft<f64>>,
    buf_vfft: ThreadLocal<RefCell<Vec<Complex<f64>>>>,
    buf_ivfft: ThreadLocal<RefCell<Vec<Complex<f64>>>>,
    scratch: ThreadLocal<RefCell<Vec<Complex<f64>>>>,
}

impl FFTData {
    pub fn new(ts: &WindowedTimeseries) -> Self {
        let fft_length = std::cmp::min(
            std::cmp::max(1 << 14, ts.w.next_power_of_two()),
            ts.data.len().next_power_of_two(),
        );
        let mut fft_chunks = Vec::new();

        let mut planner = FftPlanner::new();
        let fftfun = planner.plan_fft_forward(fft_length);
        let ifftfun = planner.plan_fft_inverse(fft_length);

        let mut begin = 0;
        while begin < ts.data.len() {
            let end = std::cmp::min(begin + fft_length, ts.data.len());
            let mut chunk: Vec<Complex<f64>> = ts.data[begin..end]
                .iter()
                .map(|x| Complex {
                    re: *x as f64,
                    im: 0.0f64,
                })
                .collect();
            chunk.resize(fft_length, Complex::zero());
            fftfun.process(&mut chunk);
            fft_chunks.push(chunk);

            //// We shift the window taking into account the width of the window `w`,
            //// since we need to compute all dot products
            begin += fft_length - ts.w;
        }
        Self {
            fft_length,
            fft_chunks,
            fftfun,
            ifftfun,
            buf_vfft: ThreadLocal::new(),
            buf_ivfft: ThreadLocal::new(),
            scratch: ThreadLocal::new(),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{distance::zeucl, timeseries::*};

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
                    let fft_data = FFTData::new(&ts);

                    let rng = Xoroshiro128Plus::seed_from_u64(12344);
                    let v: Vec<f64> = rng.sample_iter(StandardNormal).take(w).collect();
                    let mut expected = vec![0.0; ts.num_subsequences()];
                    let mut actual = vec![0.0; ts.num_subsequences()];

                    ts.sliding_dot_product_slow(&v, &mut expected);
                    ts.sliding_dot_product(&fft_data, &v, &mut actual);
                    // dump_vec(format!("/tmp/sliding-expected-{}-{}.txt", n, w), &expected);
                    // dump_vec(format!("/tmp/sliding-actual-{}-{}.txt", n, w), &actual);

                    for (e, a) in expected.into_iter().zip(actual) {
                        assert!(
                            (e - a).abs() <= 0.000000001,
                            "{} != {} (expected != actual)",
                            e,
                            a
                        );
                    }
                    // assert!(expected
                    //     .into_iter()
                    //     .zip(actual.into_iter())
                    //     .all(|(e, a)| (e - a).abs() <= 0.0000001));
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
                    let fft_data = FFTData::new(&ts);

                    let rng = Xoroshiro128Plus::seed_from_u64(12344);
                    let v: Vec<f64> = rng.sample_iter(StandardNormal).take(w).collect();
                    let mut expected = vec![0.0; ts.num_subsequences()];
                    let mut actual = vec![0.0; ts.num_subsequences()];

                    ts.znormalized_sliding_dot_product_slow(&v, &mut expected);
                    ts.znormalized_sliding_dot_product(&fft_data, &v, &mut actual);

                    assert!(expected
                        .into_iter()
                        .zip(actual.into_iter())
                        .all(|(e, a)| (e - a).abs() <= 0.000000001));
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
        let ts = WindowedTimeseries::new(ts, w, true);

        for i in 0..ts.num_subsequences() {
            let a = ts.subsequence(i);
            let mean: f64 = a.iter().sum::<f64>() / a.len() as f64;
            let actual_mean = ts.mean(i);
            let sd = ((a.iter().map(|x| (x - mean).powi(2)).sum::<f64>()) / a.len() as f64).sqrt();
            let actual_sd = ts.sd(i);
            assert!((mean - actual_mean).abs() < 0.0000000000001);
            assert!((sd - actual_sd).abs() < 0.0000000000001);
        }
    }

    #[test]
    #[ignore]
    fn test_rolling_stats() {
        let w = 1000;
        let ts =
            crate::load::loadts("data/ECG.csv.gz", Some(1_000_000)).expect("problem loading data");

        let (a_mean, a_std) = rolling_stat(&ts, w);
        let (e_mean, e_std) = rolling_stat_slow(&ts, w);

        assert_eq!(a_mean.len(), e_mean.len());
        for (i, (a, e)) in a_mean.iter().zip(e_mean.iter()).enumerate() {
            assert!(
                (a - e).abs() < 0.000000000001,
                "[{}] a = {} e = {}",
                i,
                a,
                e
            );
        }

        assert_eq!(a_std.len(), e_std.len());
        for (i, (a, e)) in a_std.iter().zip(e_std.iter()).enumerate() {
            assert!(
                (a - e).abs() < 0.000000000001,
                "[{}] a = {} e = {}",
                i,
                a,
                e
            );
        }
    }

    #[test]
    fn test_distance_profile() {
        let w = 1000;
        let ts =
            crate::load::loadts("data/ECG.csv.gz", Some(100000)).expect("problem loading data");
        let ts = WindowedTimeseries::new(ts, w, true);
        let fft_data = FFTData::new(&ts);

        let mut actual = vec![0.0; ts.num_subsequences()];
        let mut buf = vec![0.0; w];

        ts.distance_profile(&fft_data, 0, &mut actual, &mut buf);
        let expected = ts.distance_profile_slow(0, zeucl);

        assert_eq!(actual.len(), expected.len());
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!((a - e).abs() < 0.00000000001, "[{}] a = {} e = {}", i, a, e);
        }
    }
}
