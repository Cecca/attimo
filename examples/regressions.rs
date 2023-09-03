use anyhow::{Context, Result};
use attimo::{load::loadts, motifs::motifs, timeseries::WindowedTimeseries};
use std::io::prelude::*;
use std::sync::Arc;
use std::{path::PathBuf, str::FromStr, time::Instant};

fn get_git_sha() -> Result<String> {
    std::process::Command::new("git")
        .arg("rev-parse")
        .arg("--short")
        .arg("HEAD")
        .output()
        .context("getting git sha")
        .and_then(|output| {
            let bytes = output.stdout;
            String::from_utf8(bytes)
                .context("decoding output")
                .map(|s| s.trim().to_owned())
        })
}

fn run(path: &str, w: usize, topk: usize, reps: usize, runs: usize, csv: &str) -> Result<()> {
    let date = chrono::offset::Utc::now();
    let sha = get_git_sha()?;
    let mut f = std::fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open(csv)?;
    let threads = rayon::current_num_threads();
    let ts = Arc::new(WindowedTimeseries::new(
        loadts("data/ECG.csv.gz", None)?,
        1000,
        false,
    ));
    for _ in 0..runs {
        let timer = Instant::now();
        motifs(Arc::clone(&ts), 10, 200, 0.01, 1234);
        writeln!(
            f,
            "{},{},{},{},{},{},{},{}",
            date.to_rfc3339(),
            sha,
            threads,
            path,
            w,
            topk,
            reps,
            timer.elapsed().as_secs_f64()
        )?;
    }
    Ok(())
}

fn main() -> Result<()> {
    debug_assert!(
        false,
        "This program should be executed only in release mode"
    );
    let output = "regressions.csv";
    if !PathBuf::from_str(&output)?.is_file() {
        let mut f = std::fs::File::create(&output)?;
        writeln!(f, "date, sha, threads, path, w, topk, repetitions, time_s")?;
    }
    let runs = 5;
    run("data/ECG.csv.gz", 1000, 10, 200, runs, output)?;

    Ok(())
}
