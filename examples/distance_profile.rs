use anyhow::Result;
use argh::*;
use attimo::distance::zeucl;
use attimo::load::loadts;
use attimo::timeseries::*;
use indicatif::{ProgressBar, ProgressStyle};
use rand::prelude::*;
use rand_distr::Uniform;
use rayon::prelude::*;
use std::io::{prelude::*, BufWriter};
use std::path::PathBuf;
use std::time::Instant;

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
    #[argh(option)]
    /// query subsequence
    from: usize,
    /// optional threshold to upper bound the reported distances
    #[argh(option)]
    threshold: Option<f64>,
}

fn main() -> Result<()> {
    let timer = Instant::now();
    let args: Args = argh::from_env();
    println!("[{:?}] Reading input", timer.elapsed());
    let ts = WindowedTimeseries::new(loadts(&args.path, None)?, args.window, false);

    println!("[{:?}] Computing distance profile", timer.elapsed());
    let dp = ts.distance_profile(args.from, &ts.fft_data());

    let thresh = args.threshold.unwrap_or(f64::INFINITY);

    println!("[{:?}] Writing output", timer.elapsed());
    let output_path = args.output;
    let mut output_file = BufWriter::new(std::fs::File::create(output_path)?);
    writeln!(output_file, "i, distance")?;
    for (i, d) in dp.into_iter().enumerate() {
        if d <= thresh {
            let check = zeucl(&ts, i, args.from);
            assert!((check - d).abs() < 0.0000001, "d={} check={}", d, check);
            writeln!(output_file, "{},{}", i, d)?;
        }
    }

    println!("[{:?}] Done", timer.elapsed());
    Ok(())
}
