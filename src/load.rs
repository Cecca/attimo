use anyhow::{bail, Context, Result};
use std::fs::File;
use std::io::prelude::*;
use std::io::BufReader;
use std::path::Path;

pub fn loadts<P: AsRef<Path>>(path: P, prefix: Option<usize>) -> Result<Vec<f64>> {
    if path
        .as_ref()
        .extension()
        .map(|ext| ext.to_str().unwrap().ends_with("gz"))
        .unwrap_or(false)
    {
        let f =
            File::open(path.as_ref()).with_context(|| format!("reading {:?}", path.as_ref()))?;
        let f = BufReader::new(f);
        let f = flate2::read::GzDecoder::new(f);
        let f = BufReader::new(f);
        load_from(f, prefix)
    } else {
        let f =
            File::open(path.as_ref()).with_context(|| format!("reading {:?}", path.as_ref()))?;
        let f = BufReader::new(f);
        load_from(f, prefix)
    }
}

fn load_from<R: BufRead>(mut reader: R, prefix: Option<usize>) -> Result<Vec<f64>> {
    let to_take = prefix.unwrap_or(usize::MAX);
    let mut res: Vec<f64> = Vec::with_capacity(10000000);
    let mut buf = String::new();
    let mut cnt = 0;
    while cnt < to_take {
        buf.clear();
        match reader.read_line(&mut buf) {
            Ok(0) => break, // EOF reached
            Ok(_) => {
                if !buf.trim_end().is_empty() {
                    let (x, rest) = fast_float::parse_partial(&buf)
                        .with_context(|| format!("parsing `{}`", buf))?;
                    if !buf[rest..].trim_end().is_empty() {
                        bail!("Cannot parse `{}` into a number", buf);
                    }
                    res.push(x);
                }
            }
            Err(e) => anyhow::bail!(e),
        }
        cnt += 1;
    }
    Ok(res)
}
