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
    let mut f = BufReader::new(f);
    let mut res: Vec<f64> = Vec::with_capacity(10000000);
    let mut buf = String::new();
    let mut cnt = 0;
    while cnt < to_take {
        buf.clear();
        match f.read_line(&mut buf) {
            Ok(0) => break, // EOF reached
            Ok(_) => {
                if !buf.trim_end().is_empty(){
                    res.push(fast_float::parse_partial(&buf).with_context(|| format!("parsing `{}`", buf))?.0);
                }
            }
            Err(e) => anyhow::bail!(e),
        }
        cnt += 1;
    }
    // for l in f.lines().take(to_take) {
    //     let l = l.unwrap();
    //     if !l.is_empty() {
    //         res.push(fast_float::parse(l).unwrap());
    //     }
    // }
    slog_scope::info!("input reading"; "tag" => "profiling", "time_s" => start.elapsed().as_secs_f64());
    Ok(res)
}
