use anyhow::{Context, Result};
use argh::*;
use attimo::load::loadts;
use attimo::motifs::*;
use attimo::timeseries::*;
use indicatif::{ProgressBar, ProgressStyle};
use rand::prelude::*;
use rand_distr::Uniform;
use rand_xoshiro::Xoshiro256StarStar;
use rayon::prelude::*;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::{prelude::*, BufWriter};
use std::path::PathBuf;
use std::str::FromStr;

#[derive(FromArgs)]
/// Compute the Local Intrinsic Dimensionality of datasets
struct Args {
    #[argh(option, short = 'w')]
    /// the window size
    window: usize,
    #[argh(option, short = 's')]
    /// number of samples to use
    samples: usize,
    #[argh(option, short = 'r', default = "50")]
    /// the number of repetitions to find motifs
    repetitions: usize,
    #[argh(positional)]
    /// path to file
    path: PathBuf,
}

struct Measures {
    lid: f64,
    rc1: f64,
    rc10: f64,
}

fn main() -> Result<()> {
    let args: Args = argh::from_env();
    let ts = WindowedTimeseries::new(loadts(&args.path, None)?, args.window);
    let fft_data = ts.fft_data();
    let seed = 1234;
    let rng = Xoshiro256StarStar::seed_from_u64(seed);

    // Find the top motif, and estimate its dimensionality measures along with the samples
    let motif = *motifs(&ts, 1, args.repetitions, 0.01, None, seed)
        .first()
        .unwrap();

    let mut idxs = vec![motif.idx_a, motif.idx_b];
    idxs.extend(
        Uniform::new(0, ts.num_subsequences())
            .sample_iter(rng)
            .take(args.samples),
    );

    let pbar = ProgressBar::new(idxs.len() as u64).with_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {msg} {bar:40.cyan/blue} {pos:>7}/{len:7}"),
    );

    let lids: Vec<(usize, Measures)> = idxs
        .into_par_iter()
        .map(|i| {
            let mut dp = ts.distance_profile(i, &fft_data);
            let mut dp: Vec<f64> = dp
                .drain(..)
                .enumerate()
                .filter_map(|(j, d)| {
                    if (i as isize - j as isize).abs() > ts.w as isize {
                        Some(d)
                    } else {
                        None
                    }
                })
                .collect();
            let valid = dp.len();
            dp.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            let w = *dp.last().unwrap();
            let mut sum_distances = 0.0;
            let mut sum_lid = 0.0;
            for &d in &dp {
                sum_distances += d;
                if d < w / 2.0 {
                    sum_lid += (d / w).ln();
                } else {
                    sum_lid += ((d - w) / w).ln_1p();
                }
            }
            let dist_mean = sum_distances / valid as f64;
            pbar.inc(1);
            (
                i,
                Measures {
                    lid: -(valid as f64) / sum_lid,
                    // There is no distance to self since we removed trivial matches, hence the
                    // closest neighbor distance is in position 0.
                    rc1: dist_mean / dp[0],
                    rc10: dist_mean / dp[9],
                },
            )
        })
        .collect();

    pbar.finish_and_clear();

    let output_path = PathBuf::from_str(&format!("{}.measures", args.path.to_str().unwrap()))?;
    let write_header = !output_path.is_file();
    let mut output_file = BufWriter::new(
        OpenOptions::new()
            .append(true)
            .create(true)
            .open(output_path)?,
    );
    if write_header {
        writeln!(output_file, "id,w,lid,rc1,rc10")?;
    }
    for (i, m) in lids {
        writeln!(output_file, "{},{},{},{},{}", i, ts.w, m.lid, m.rc1, m.rc10)?;
    }

    Ok(())
}
