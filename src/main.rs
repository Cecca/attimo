#![feature(allocator_api)]

mod distance;
mod embedding;
mod load;
mod lsh;
mod types;

use anyhow::{Context, Result};
use bumpalo::Bump;
use distance::*;
use embedding::*;
use load::*;
use lsh::*;
use plotly::common::Mode;
use plotly::{Plot, Scatter};
use types::*;
use std::time::Instant;

fn main() -> Result<()> {
    let start = Instant::now();
    let path = std::env::args().nth(1).context("missing path to dataset")?;
    let w = 300;

    let hasher = Hasher::new(32, 200, Embedder::new(w, w, 1.0, 1234), 49875);
    let arena = Bump::new();

    println!("{:?} loading time series", start.elapsed());
    let ts: Vec<f64> = loadts(path)?.into_iter().take(100000).collect();
    println!("{:?} computing windowed stats", start.elapsed());
    let ts = WindowedTimeseries::new(ts, w);
    println!("{:?} computing hash pools", start.elapsed());
    let pools = HashCollection::from_ts(&ts, &hasher, &arena);
    println!("{:?} done", start.elapsed());
    // let dp = ts.distance_profile(560, eucl);

    // let lines = Scatter::new(0..dp.len(), dp)
    //     .name("input")
    //     .mode(Mode::Lines);
    // let mut plot = Plot::new();
    // plot.add_trace(lines);
    // plot.show();

    Ok(())
}
