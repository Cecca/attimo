use anyhow::Context;
use anyhow::{Result};
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::path::Path;

pub fn loadts<P: AsRef<Path>>(path: P) -> Result<Vec<f64>> {
    let f = File::open(path.as_ref()).with_context(|| format!("reading {:?}", path.as_ref()))?;
    let f = BufReader::new(f);
    Ok(f.lines()
        .map(|l| l.unwrap())
        .filter(|l| !l.is_empty())
        .map(|l| l.parse::<f64>().unwrap())
        .collect())
}
