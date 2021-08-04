#![feature(allocator_api)]
#![feature(is_sorted)]

mod approx_mp;
mod distance;
mod embedding;
mod load;
mod lsh;
mod types;

use anyhow::{Context, Result};
use approx_mp::*;
use bumpalo::Bump;
use distance::*;
use embedding::*;
use load::*;
use lsh::*;
use plotly::common::Mode;
use plotly::layout::LayoutGrid;
use plotly::{Layout, Plot, Scatter};
use types::*;

fn main() -> Result<()> {
    let path = std::env::args().nth(1).context("missing path to dataset")?;
    let k = std::env::args().nth(2).context("missing k")?.parse::<usize>().context("parsing k")?;
    let w = 300;
    let ts: Vec<f64> = loadts(path)?.into_iter().take(10000).collect();
    let ts = WindowedTimeseries::new(ts, w);
    let amp = approx_mp(&ts, k, 200, 0.001, 1234);
    let amp: Vec<f64> = amp.into_iter().map(|pair| pair.0).collect();

    // let dp = ts.distance_profile(0, zeucl);
    // let hasher = Hasher::new(32, 2000, Embedder::new(ts.w, ts.w, 1.0, 123), 123);
    // let arena = Bump::new();
    // let pools = HashCollection::from_ts(&ts, &hasher, &arena);
    // let ps: Vec<f64> = (0..ts.num_subsequences()).map(|i| {
    //     pools.collision_probability(0, i)
    // }).collect();

    // let mut data: Vec<(f64, f64)> = dp.into_iter().zip(ps.into_iter()).collect();
    // data.sort_by(|x, y| x.partial_cmp(y).unwrap());
    // let dists: Vec<f64> = data.iter().map(|p| p.0).collect();
    // let probs: Vec<f64> = data.iter().map(|p| p.1).collect();

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

    Ok(())
}
