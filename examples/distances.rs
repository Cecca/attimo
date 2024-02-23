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
    output: Option<PathBuf>,
    #[argh(option)]
    /// number of pairs to sample
    samples: usize,
    #[argh(option)]
    /// prefix of the time series to load
    prefix: Option<usize>,
}

fn main() -> Result<()> {
    let args: Args = argh::from_env();
    let ts = WindowedTimeseries::new(loadts(&args.path, args.prefix)?, args.window, false);

    let mut rng = rand_xoshiro::Xoshiro256StarStar::seed_from_u64(1234);
    let uniform = Uniform::new(0, ts.num_subsequences());
    let pairs: Vec<(usize, usize)> = (0..(10 * args.samples))
        .filter_map(|_| {
            let i = uniform.sample(&mut rng);
            let j = uniform.sample(&mut rng);
            if (i as isize - j as isize).abs() < ts.w as isize {
                None
            } else {
                Some((i, j))
            }
        })
        .take(args.samples)
        .collect();

    let pbar = ProgressBar::new(args.samples as u64).with_style(
        ProgressStyle::default_bar()
            .template("[{elapsed_precise}] {msg} {bar:40.cyan/blue} {pos:>7}/{len:7}")
            .unwrap(),
    );

    let distances: Vec<f64> = pairs
        .into_par_iter()
        .map(|(i, j)| zeucl(&ts, i, j))
        .collect();

    pbar.finish_and_clear();

    if let Some(output_path) = args.output {
        let mut output_file = BufWriter::new(std::fs::File::create(output_path)?);
        writeln!(output_file, "w, distance")?;
        for d in distances {
            writeln!(output_file, "{},{}", ts.w, d)?;
        }
    } else {
        let mean = distances.iter().sum::<f64>() / (distances.len() as f64);
        println!("{mean}");
    }

    Ok(())
}
