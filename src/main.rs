#![feature(allocator_api)]

mod distance;
mod embedding;
mod load;
mod lsh;
mod types;
mod approx_mp;

use anyhow::{Context, Result};
use load::*;
use plotly::common::Mode;
use plotly::{Plot, Scatter};
use types::*;
use approx_mp::*;
use distance::*;
use lsh::*;
use embedding::*;
use bumpalo::Bump;

fn main() -> Result<()> {
    let path = std::env::args().nth(1).context("missing path to dataset")?;
    let w = 300;
    let ts: Vec<f64> = loadts(path)?.into_iter().take(10000).collect();
    let ts = WindowedTimeseries::new(ts, w);
    // let amp = approx_mp(&ts, 32, 200, 0.0001, 1234);
    // let amp: Vec<f64> = amp.into_iter().map(|pair| pair.0).collect();

    let dp = ts.distance_profile(0, zeucl);
    let hasher = Hasher::new(32, 2000, Embedder::new(ts.w, ts.w, 1.0, 123), 123);
    let arena = Bump::new();
    let pools = HashCollection::from_ts(&ts, &hasher, &arena);
    let ps: Vec<f64> = (0..ts.num_subsequences()).map(|i| {
        pools.collision_probability(0, i)
    }).collect();

    let mut data: Vec<(f64, f64)> = dp.into_iter().zip(ps.into_iter()).collect();
    data.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let dists: Vec<f64> = data.iter().map(|p| p.0).collect();
    let probs: Vec<f64> = data.iter().map(|p| p.1).collect();

    let lines = Scatter::new(dists, probs)
        .name("distance profile")
        .mode(Mode::LinesMarkers);
    let mut plot = Plot::new();
    plot.add_trace(lines);
    plot.show();

    Ok(())
}
