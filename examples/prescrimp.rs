use anyhow::Result;
use argh::FromArgs;
use attimo::distance::*;
use attimo::load::loadts;
use attimo::motifs::{Motif, TopK};
use attimo::timeseries::*;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::cell::RefCell;
use std::time::Instant;
use thread_local::ThreadLocal;

fn seq_by(min: usize, max: usize, by: usize) -> Vec<usize> {
    let mut i = min;
    let res: Vec<usize> = if by == 1 {
        (min..max).collect()
    } else {
        std::iter::from_fn(|| {
            if i >= max {
                return None;
            }
            let res = Some(i);
            i += by;
            res
        })
        .collect()
    };
    res
}

fn zeucl_to_dotp(w: usize, d: f64, ma: f64, mb: f64, sa: f64, sb: f64) -> f64 {
    let w = w as f64;
    (1.0 - d * d / (2.0 * w)) * w * sa * sb + w * ma * mb
}

fn dotp_to_zeucl(w: usize, q: f64, ma: f64, mb: f64, sa: f64, sb: f64) -> f64 {
    let w = w as f64;
    (2.0 * w * (1.0 - (q - w * ma * mb) / (w * sa * sb))).sqrt()
}

struct MatrixProfile {
    dists: Vec<f64>,
    indices: Vec<usize>,
}

impl MatrixProfile {
    fn new(n: usize) -> Self {
        Self {
            dists: vec![f64::INFINITY; n],
            indices: vec![0; n],
        }
    }

    fn iter_mut(&mut self) -> impl Iterator<Item = (&mut f64, &mut usize)> {
        self.dists.iter_mut().zip(self.indices.iter_mut())
    }
    fn iter(&self) -> impl Iterator<Item = (&f64, &usize)> {
        self.dists.iter().zip(self.indices.iter())
    }
}

fn relative_error(actual: f64, expected: f64) -> f64 {
    (actual - expected).abs() / expected
}

fn pre_scrimp(
    ts: &WindowedTimeseries,
    s: usize,
    exclusion_zone: usize,
    mp: &mut [f64],
    indices: &mut [usize],
    fft_data: &FFTData,
) {
    // note that here we are considering sequences which are s > ts.w apart, so
    // we have no trivial matches
    let ns = ts.num_subsequences();
    let w = ts.w;

    let tl_mp = ThreadLocal::new();
    let tl_dists = ThreadLocal::new();
    let tl_buf = ThreadLocal::new();

    let pbar = ProgressBar::new((ns / s) as u64);
    pbar.set_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {bar:40.cyan/blue} ETA: {eta} {pos:>7}/{len:7} {msg}")
            .unwrap(),
    );

    // We don't shuffle the indices, since we are not going to stop pre_scrimp
    // arbitrarily. This saves a a little time.
    seq_by(0, ns, s).into_par_iter().for_each(|i| {
        let mut mp = tl_mp
            .get_or(|| RefCell::new(MatrixProfile::new(ns)))
            .borrow_mut();
        let mut dists = tl_dists
            .get_or(|| RefCell::new(vec![0.0; ts.num_subsequences()]))
            .borrow_mut();
        let mut buf = tl_buf.get_or(|| RefCell::new(vec![0.0; ts.w])).borrow_mut();
        ts.distance_profile(fft_data, i, &mut dists, &mut buf);

        // update the running matrix profile
        // for (j, ((mp_val, index_val), dp_val)) in mp.dists.iter_mut().zip(mp.indices.iter_mut()).zip(dists.iter()).enumerate() {
        for (j, ((mp_val, index_val), dp_val)) in mp.iter_mut().zip(dists.iter()).enumerate() {
            if (i as isize - j as isize).abs() >= exclusion_zone as isize && *dp_val < *mp_val {
                *mp_val = *dp_val;
                *index_val = i;
            }
        }

        // nearest neighbor index, excluding trivial matches
        let (j, d) = dists
            .iter()
            .enumerate()
            .filter(|(j, _)| (*j as isize - i as isize).abs() >= exclusion_zone as isize)
            .min_by(|p1, p2| p1.1.partial_cmp(p2.1).unwrap())
            .unwrap();

        let mut q = zeucl_to_dotp(w, *d, ts.mean(i), ts.mean(j), ts.sd(i), ts.sd(j));
        debug_assert!(
            relative_error(q, dot(ts.subsequence(i), ts.subsequence(j))) <= 0.00001,
            "zeucl_to_dotp: actual {} expectd {}",
            q,
            dot(ts.subsequence(i), ts.subsequence(j))
        );
        let q_prime = q;

        for k in 1..(s - 1) {
            if i + k >= ns || j + k >= ns {
                break;
            }
            q = q - ts.data[i + k - 1] * ts.data[j + k - 1]
                + ts.data[i + k + w - 1] * ts.data[j + k + w - 1];
            debug_assert!(
                relative_error(q, dot(ts.subsequence(i + k), ts.subsequence(j + k))) <= 0.00001,
                "sliding dot product: actual {} expectd {}",
                q,
                dot(ts.subsequence(i + k), ts.subsequence(j + k))
            );
            let d = dotp_to_zeucl(
                w,
                q,
                ts.mean(i + k),
                ts.mean(j + k),
                ts.sd(i + k),
                ts.sd(j + k),
            );
            debug_assert!(
                relative_error(d, zeucl(ts, i + k, j + k)) <= 0.00001,
                "dotp_to_zeucl: actual {} expected {}",
                d,
                zeucl(ts, i + k, j + k)
            );
            if d < mp.dists[i + k] {
                mp.dists[i + k] = d;
                mp.indices[i + k] = j + k;
            }
            if d < mp.dists[j + k] {
                mp.dists[j + k] = d;
                mp.indices[j + k] = i + k;
            }
        }

        q = q_prime;
        for k in 1..(s - 1) {
            if k > i || k > j {
                break;
            }
            q = q - ts.data[i - k + w] * ts.data[j - k + w] + ts.data[j - k] * ts.data[i - k];
            debug_assert!(
                relative_error(q, dot(ts.subsequence(i - k), ts.subsequence(j - k))) <= 0.00001,
                "sliding dot product: actual {} expectd {}",
                q,
                dot(ts.subsequence(i - k), ts.subsequence(j - k))
            );
            let d = dotp_to_zeucl(
                w,
                q,
                ts.mean(i - k),
                ts.mean(j - k),
                ts.sd(i - k),
                ts.sd(j - k),
            );
            debug_assert!(
                relative_error(d, zeucl(ts, i - k, j - k)) <= 0.00001,
                "dotp_to_zeucl: actual {} expected {}",
                d,
                zeucl(ts, i - k, j - k)
            );
            if d < mp.dists[i - k] {
                mp.dists[i - k] = d;
                mp.indices[i - k] = j - k;
            }
            if d < mp.dists[j - k] {
                mp.dists[j - k] = d;
                mp.indices[j - k] = i - k;
            }
        }
        pbar.inc(1);
    });
    pbar.finish_and_clear();

    tl_mp.into_iter().for_each(|tmp| {
        for ((out_d, out_i), (d, i)) in mp
            .iter_mut()
            .zip(indices.iter_mut())
            .zip(tmp.borrow().iter())
        {
            if *d < *out_d {
                *out_d = *d;
                *out_i = *i;
            }
        }
    });
}

/// Compute the approximate matrix profile using SCRIMP++
#[derive(FromArgs)]
struct Args {
    #[argh(option, short = 'w')]
    /// subsequcence length
    pub window: usize,

    #[argh(option, default = "1")]
    /// the number of motifs to look for
    pub motifs: usize,

    #[argh(option)]
    /// consider only the given number of points from the input
    pub prefix: Option<usize>,

    #[argh(option, default = "0.25")]
    /// the skip for pre-scrimp, as a fraction of the window size
    pub skip: f64,

    #[argh(switch)]
    /// wether meand and std computations should be at the best precision, at the expense of running time
    pub precise: bool,

    #[argh(option, default = "default_output()")]
    /// path to the output file
    pub output: String,

    #[argh(positional)]
    /// path to the data file
    pub path: String,
}

fn default_output() -> String {
    "scrimp.csv".to_owned()
}

fn output_csv(path: &str, motifs: &[Motif]) -> Result<()> {
    use std::io::prelude::*;
    let mut f = std::fs::File::create(path)?;
    for m in motifs {
        if let Some(confirmation_time) = m.elapsed {
            writeln!(
                f,
                "{}, {}, {}, {}",
                m.idx_a,
                m.idx_b,
                m.distance,
                confirmation_time.as_secs_f64()
            )?;
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    debug_assert!(
        false,
        "This software should run only in Release mode, times are important"
    );
    let args: Args = argh::from_env();

    eprintln!("Reading input");
    let w = args.window;
    let ts = WindowedTimeseries::new(loadts(args.path, args.prefix)?, w, args.precise);
    eprintln!("Computing FFT");
    let fft_data = FFTData::new(&ts);
    let s = (args.skip * (w as f64)) as usize;
    let exclusion_zone = w;

    let timer = Instant::now();
    let mut mp = vec![f64::INFINITY; ts.num_subsequences()];
    let mut indices = vec![0; ts.num_subsequences()];

    eprintln!("Running PreSCRIMP");
    pre_scrimp(&ts, s, exclusion_zone, &mut mp, &mut indices, &fft_data);

    eprintln!("Getting motifs");
    let mut topk = TopK::new(args.motifs, exclusion_zone);
    for (i, (j, d)) in indices.iter().zip(mp.iter()).enumerate() {
        let m = Motif {
            idx_a: std::cmp::min(i, *j),
            idx_b: std::cmp::max(i, *j),
            distance: *d,
            elapsed: None,
            discovered: timer.elapsed(),
        };
        topk.insert(m);
    }
    eprintln!("Writing output");
    let mut motifs = topk.to_vec();
    for m in motifs.iter_mut() {
        debug_assert!(
            m.distance >= zeucl(&ts, m.idx_a, m.idx_b) - 0.000001, // allow for some tolerance for numerical instabilities
            "Computed distance is smaller than true distance: {} < {}",
            m.distance,
            zeucl(&ts, m.idx_a, m.idx_b)
        );
        m.elapsed.replace(timer.elapsed());
    }
    output_csv(&args.output, &motifs)?;
    eprintln!("Done");

    Ok(())
}
