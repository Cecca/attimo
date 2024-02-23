use anyhow::Result;
use argh::*;
use attimo::load::loadts;
use attimo::timeseries::*;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::fs::OpenOptions;
use std::io::{prelude::*, BufWriter};
use std::path::PathBuf;

#[derive(FromArgs)]
/// Compute the Local Intrinsic Dimensionality of datasets
struct Args {
    #[argh(option, short = 'w')]
    /// the window size
    window: usize,
    #[argh(option)]
    /// path to file
    path: PathBuf,
    #[argh(option)]
    /// path to output
    output: PathBuf,
    #[argh(positional)]
    /// number of samples to use
    idxs: Vec<usize>,
}

struct Measures {
    nn: f64,
    lid: f64,
    rc1: f64,
    rc10: f64,
    rc100: f64,
}

fn main() -> Result<()> {
    let args: Args = argh::from_env();
    let ts = WindowedTimeseries::new(loadts(&args.path, None)?, args.window, false);
    let fft_data = FFTData::new(&ts);

    let idxs = args.idxs;

    let pbar = ProgressBar::new(idxs.len() as u64).with_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {msg} {bar:40.cyan/blue} {pos:>7}/{len:7}")
            .unwrap(),
    );

    let lids: Vec<(usize, Measures)> = idxs
        .into_par_iter()
        .map(|i| {
            let mut dp = vec![0.0; ts.num_subsequences()];
            let mut buf = vec![0.0; ts.w];
            ts.distance_profile(&fft_data, i, &mut dp, &mut buf);
            let mut dp: Vec<f64> = dp
                .drain(..)
                .enumerate()
                .filter_map(|(j, d)| {
                    if d.is_finite() && (i as isize - j as isize).abs() > ts.w as isize {
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
                    nn: dp[0],
                    lid: -(valid as f64) / sum_lid,
                    // There is no distance to self since we removed trivial matches, hence the
                    // closest neighbor distance is in position 0.
                    rc1: dist_mean / dp[0],
                    rc10: dist_mean / dp[9],
                    rc100: dist_mean / dp[99],
                },
            )
        })
        .collect();

    pbar.finish_and_clear();

    let output_path = args.output;
    let write_header = !output_path.is_file();
    let mut output_file = BufWriter::new(
        OpenOptions::new()
            .append(true)
            .create(true)
            .open(output_path)?,
    );
    if write_header {
        writeln!(output_file, "id,w,nn,lid,rc1,rc10,rc100")?;
    }
    for (i, m) in lids {
        writeln!(
            output_file,
            "{},{},{},{},{},{},{}",
            i, ts.w, m.nn, m.lid, m.rc1, m.rc10, m.rc100
        )?;
    }

    Ok(())
}
