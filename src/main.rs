#![feature(allocator_api)]
#![feature(is_sorted)]

mod approx_mp;
mod distance;
mod embedding;
mod load;
mod lsh;
mod types;

use std::fs::OpenOptions;
use slog::*;
use anyhow::Result;
use approx_mp::*;
use argh::FromArgs;
use load::*;
use plotly::common::Mode;
use plotly::layout::LayoutGrid;
use plotly::{Layout, Plot, Scatter};
use slog_scope::GlobalLoggerGuard;
use types::*;

#[derive(FromArgs)]
/// ATTIMO computes ApproximaTe TImeseries MOtifs.
struct Config {
    #[argh(option, short = 'w')]
    /// subsequcence length
    pub window: usize,

    #[argh(option, default = "default_k()")]
    /// number of hash functions
    pub k: usize,

    #[argh(option, default = "default_delta()")]
    /// failure probability of the LSH scheme
    pub delta: f64,

    #[argh(option, short = 'r')]
    /// the number of LSH repetitions
    pub repetitions: usize,

    #[argh(switch)]
    /// open a browser window with a plot of the approximate matrix profile
    pub plot: bool,

    #[argh(option)]
    /// consider only the given number of points from the input
    pub prefix: Option<usize>,

    #[argh(option, default = "default_seed()")]
    /// seed for the psudorandom number generator
    pub seed: u64,

    #[argh(positional)]
    /// path to the data file
    pub path: String,
}

fn default_seed() -> u64 {
    12453
}

fn default_k() -> usize {
    64
}

fn default_delta() -> f64 {
    0.001
}

fn main() -> Result<()> {
    let _guard = setup_logger()?;

    // read configuration
    let config: Config = argh::from_env();
    let path = config.path;
    let w = config.window;
    let ts: Vec<f64> = loadts(path, config.prefix)?;
    let ts = WindowedTimeseries::new(ts, w);
    println!("Loaded time series, taking {}", ts.bytes_size());
    let amp = approx_mp(&ts, config.k, config.repetitions, config.delta, config.seed);
    let amp: Vec<f64> = amp.into_iter().map(|pair| pair.0).collect();

    if config.plot {
        let ts_lines = Scatter::new(0..ts.data.len(), ts.data.clone())
            .name("time series")
            .mode(Mode::Lines);
        let amp_lines = Scatter::new(0..ts.num_subsequences(), amp)
            .name("approximate matrix profile")
            .mode(Mode::Lines);
        let mut plot = Plot::new();
        plot.set_layout(Layout::new().grid(LayoutGrid::new().rows(2)));
        plot.add_trace(ts_lines);
        plot.add_trace(amp_lines);
        plot.show();
    }
    Ok(())
}

fn setup_logger() -> Result<GlobalLoggerGuard> {
    let log_path = ".trace.log";
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(log_path)
        .unwrap();

    let drain = slog_json::Json::new(file)
        .set_pretty(false)
        .add_default_keys()
        .build()
        .fuse();
    let drain = slog_async::Async::new(drain).build().fuse();
    let logger = slog::Logger::root(drain, o!());

    let guard = slog_scope::set_global_logger(logger);

    Ok(guard)
}
