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
    let ts: Vec<f64> = loadts(path)?.into_iter().take(10000).collect();
    let ts = WindowedTimeseries::new(ts, w);
    approx_mp(&ts, 1, 64, 200, 0.01, 1234);

    // let dp = ts.distance_profile(0, eucl);

    // let lines = Scatter::new(0..probs.len(), probs)
    //     .name("collision probabilities")
    //     .mode(Mode::Lines);
    // let mut plot = Plot::new();
    // plot.add_trace(lines);
    // plot.show();

    Ok(())
}
