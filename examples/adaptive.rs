use anyhow::Result;
use attimo::adaptive::AdaptiveHashCollection;
use attimo::distance::zeucl;
use attimo::load::loadts;
use attimo::lsh::*;
use attimo::motifs::{Motif, TopK};
use attimo::sort::*;
use attimo::timeseries::WindowedTimeseries;
use std::convert::TryInto;
use std::time::{Duration, Instant};

const MB: usize = 1024*1024;

fn main() -> Result<()> {
    let ts = loadts("data/ECG.csv.gz", None)?;
    let ts = WindowedTimeseries::new(ts, 1000, false);
    // let ts = WindowedTimeseries::gen_randomwalk(30, 3, 234);

    let timer = Instant::now();
    let coll = AdaptiveHashCollection::new(&ts, 1000 * MB, 1234);
    eprintln!("Estimation and construction {:?}", timer.elapsed());

    Ok(())
}
