#![feature(allocator_api)]

mod distance;
mod embedding;
mod load;
mod lsh;
mod types;
mod approx_mp;

use anyhow::{Context, Result};
use bumpalo::Bump;
use distance::*;
use embedding::*;
use load::*;
use lsh::*;
use plotly::common::Mode;
use plotly::{Plot, Scatter};
use types::*;
use approx_mp::*;
use std::time::Instant;

fn main() -> Result<()> {
    let path = std::env::args().nth(1).context("missing path to dataset")?;
    let w = 300;
    let ts: Vec<f64> = loadts(path)?.into_iter().take(100000).collect();
    let ts = WindowedTimeseries::new(ts, w);
    let amp = approx_mp(&ts, 64, 500, 0.01, 1234);
    let amp: Vec<f64> = amp.into_iter().map(|pair| pair.0).collect();

    // let dp = ts.distance_profile(0, zeucl);

    let lines = Scatter::new(0..amp.len(), amp)
        .name("distance profile")
        .mode(Mode::Lines);
    let mut plot = Plot::new();
    plot.add_trace(lines);
    plot.show();

    Ok(())
}
