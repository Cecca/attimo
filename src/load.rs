use anyhow::Context;
use anyhow::Result;
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::path::Path;
use std::time::Instant;

pub fn loadts<P: AsRef<Path>>(path: P, prefix: Option<usize>) -> Result<Vec<f64>> {
    let start = Instant::now();
    let to_take = prefix.unwrap_or(usize::MAX);
    let f = File::open(path.as_ref()).with_context(|| format!("reading {:?}", path.as_ref()))?;
    let f = BufReader::new(f);
    let res = Ok(f.lines()
        .take(to_take)
        .map(|l| l.unwrap())
        .filter(|l| !l.is_empty())
        .map(|l| l.parse::<f64>().unwrap())
        .collect());
    slog_scope::info!("input reading"; "tag" => "profiling", "time_s" => start.elapsed().as_secs_f64());
    res
}
