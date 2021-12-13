use anyhow::Result;
use argh::FromArgs;
use attimo::allocator::{self, CountingAllocator};
use attimo::distance::zeucl;
use attimo::load::*;
use attimo::motifs::{motifs, Motif};
use attimo::timeseries::*;
use plotly::common::{Line, Mode};
use plotly::{Layout, Plot, Scatter};
use slog::*;
use slog_scope::GlobalLoggerGuard;
use std::fs::OpenOptions;
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::{Duration, Instant};

const VERSION: u32 = 2;

#[global_allocator]
static A: CountingAllocator = CountingAllocator;

#[derive(FromArgs)]
/// ATTIMO computes ApproximaTe TImeseries MOtifs.
struct Config {
    #[argh(option, short = 'w')]
    /// subsequcence length
    pub window: usize,

    #[argh(option, default = "default_motifs()")]
    /// the number of motifs to look for
    pub motifs: usize,

    #[argh(option, default = "default_delta()")]
    /// failure probability of the LSH scheme
    pub delta: f64,

    #[argh(option)]
    /// the number of repetitions to perform
    pub repetitions: usize,

    #[argh(switch)]
    /// open a browser window with a plot of the approximate matrix profile
    pub plot: bool,

    #[argh(option)]
    /// consider only the given number of points from the input
    pub prefix: Option<usize>,

    #[argh(option, default = "default_seed()")]
    /// seed for the psudorandom number generator
    pub seed: u64,

    #[argh(option, default = "default_log_path()")]
    /// the file in which to store the detailed execution log
    pub log_path: String,

    #[argh(option, default = "default_output()")]
    /// path to the output file
    pub output: String,

    #[argh(positional)]
    /// path to the data file
    pub path: String,
}

fn default_seed() -> u64 {
    12453
}

fn default_delta() -> f64 {
    0.001
}

fn default_motifs() -> usize {
    1
}

fn default_output() -> String {
    "motifs.csv".to_owned()
}

fn default_log_path() -> String {
    ".trace.json".to_owned()
}

fn main() -> Result<()> {
    if std::env::args().filter(|arg| arg == "--version").count() == 1 {
        println!("{}", VERSION);
        return Ok(());
    }

    // read configuration
    let config: Config = argh::from_env();

    let monitor_flag = Arc::new(AtomicBool::new(true));
    let monitor = allocator::monitor(Duration::from_secs(1), Arc::clone(&monitor_flag));

    let _guard = setup_logger(&config.log_path)?;
    let path = config.path;
    let w = config.window;
    let timer = Instant::now();
    let ts: Vec<f64> = loadts(path, config.prefix)?;
    let ts = WindowedTimeseries::new(ts, w);
    let input_elapsed = timer.elapsed();
    println!(
        "Loaded time series in {:?}, taking {}",
        input_elapsed,
        ts.bytes_size()
    );
    slog_scope::info!("input reading";
        "tag" => "profiling",
        "time_s" => input_elapsed.as_secs_f64()
    );

    let motifs = motifs(
        &ts,
        config.motifs,
        config.repetitions,
        config.delta,
        config.seed,
    );

    monitor_flag.store(false, std::sync::atomic::Ordering::SeqCst);
    monitor.join().unwrap();

    output_csv(&config.output, &motifs)?;

    if config.plot {
        let mut plot = Plot::new();
        let layout = Layout::default().height(500);
        plot.set_layout(layout);

        let motif_range = 0..ts.w;
        // Provide some context time series
        let n_context = 50;
        let stride = ts.num_subsequences() / n_context;
        let mut idx = 0;
        while idx < ts.num_subsequences() {
            let mut vals = vec![0.0; ts.w];
            ts.znormalized(idx, &mut vals);
            let l = Scatter::new(motif_range.clone(), vals)
                .line(Line::new().color("#a0a0a0").width(0.5))
                .show_legend(false)
                .hover_info(plotly::common::HoverInfo::Skip)
                .mode(Mode::Lines);
            plot.add_trace(l);
            idx += stride;
        }

        let occs = find_occurences(&ts, &motifs[0]);
        for i in occs {
            let mut vals = vec![0.0; ts.w];
            ts.znormalized(i, &mut vals);
            let l = Scatter::new(motif_range.clone(), vals)
                .line(Line::new().color("#f7a61b").width(1.5))
                .show_legend(false)
                .mode(Mode::Lines);
            plot.add_trace(l);
        }
        for i in [motifs[0].idx_a, motifs[0].idx_b] {
            let mut vals = vec![0.0; ts.w];
            ts.znormalized(i, &mut vals);
            let l = Scatter::new(motif_range.clone(), vals)
                .line(Line::new().color("#db4620").width(3.0))
                .show_legend(false)
                .mode(Mode::Lines);
            plot.add_trace(l);
        }
        plot.use_local_plotly();
        plot.show();
    }
    Ok(())
}

fn output_csv<P: AsRef<Path>>(path: P, motifs: &[Motif]) -> Result<()> {
    use std::io::prelude::*;
    let mut f = std::fs::File::create(path.as_ref())?;
    for m in motifs {
        writeln!(f, "{}, {}, {}", m.idx_a, m.idx_b, m.distance)?;
    }
    Ok(())
}

fn find_occurences(ts: &WindowedTimeseries, motif: &Motif) -> Vec<usize> {
    let mdist = motif.distance;
    let w = ts.w;
    let mut idxs: Vec<usize> = ts
        .distance_profile(motif.idx_a, zeucl)
        .iter()
        .enumerate()
        .filter(|&(i, _d)| i != motif.idx_a && i != motif.idx_b)
        .filter_map(|(i, d)| if *d <= 2.0 * mdist { Some(i) } else { None })
        .collect();

    idxs.sort();
    let mut output_idxs = Vec::new();
    if !idxs.is_empty() {
        output_idxs.push(idxs[0]);
        for &i in &idxs[1..] {
            if output_idxs.last().unwrap() + w < i {
                output_idxs.push(i);
            }
        }
    }

    output_idxs
}

fn setup_logger(log_path: &str) -> Result<GlobalLoggerGuard> {
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(log_path)
        .unwrap();

    let drain = slog_json::Json::new(file)
        .set_pretty(false)
        .add_default_keys()
        .build()
        .fuse();
    let drain = slog_async::Async::new(drain).build().fuse();
    let logger = slog::Logger::root(drain, o!());

    let guard = slog_scope::set_global_logger(logger);

    Ok(guard)
}
