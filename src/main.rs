use anyhow::Result;
use argh::FromArgs;
use attimo::distance::zeucl;
use attimo::load::*;
use attimo::motifs::{motifs, Motif};
use attimo::timeseries::*;
use plotly::common::{Line, Mode};
use plotly::{Plot, Scatter};
use slog::*;
use slog_scope::GlobalLoggerGuard;
use std::fs::OpenOptions;
use std::ops::Range;

#[derive(FromArgs)]
/// ATTIMO computes ApproximaTe TImeseries MOtifs.
struct Config {
    #[argh(option, short = 'w')]
    /// subsequcence length
    pub window: usize,

    #[argh(option)]
    /// the number of motifs to look for
    pub motifs: usize,

    #[argh(option, default = "default_delta()")]
    /// failure probability of the LSH scheme
    pub delta: f64,

    #[argh(option, short = 'r')]
    /// the number of LSH repetitions
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

fn main() -> Result<()> {
    let _guard = setup_logger()?;

    // read configuration
    let config: Config = argh::from_env();
    let path = config.path;
    let w = config.window;
    let ts: Vec<f64> = loadts(path, config.prefix)?;
    let ts = WindowedTimeseries::new(ts, w);
    println!("Loaded time series, taking {}", ts.bytes_size());
    let motifs = motifs(
        &ts,
        config.motifs,
        config.repetitions,
        config.delta,
        config.seed,
    );

    // TODO: output all motifs to CSV
    println!("Top motif: {:?}", motifs[0]);

    if config.plot {
        let mut plot = Plot::new();
        let ts_lines = Scatter::new(0..ts.data.len(), ts.data.clone())
            .name("time series")
            .mode(Mode::Lines);
        // plot.set_layout(Layout::new().grid(LayoutGrid::new().rows(2)));
        plot.add_trace(ts_lines);
        let occs = find_occurences(&ts, &motifs[0]);
        for (idxs, vals) in occs {
            let l = Scatter::new(idxs, vals)
                .line(Line::new().color("#FF0000").width(5.0))
                .mode(Mode::Lines);
            plot.add_trace(l);
        }
        plot.show();
    }
    Ok(())
}

fn find_occurences(ts: &WindowedTimeseries, motif: &Motif) -> Vec<(Range<usize>, Vec<f64>)> {
    let mdist = motif.distance;
    let w = ts.w;
    let thresh = (w as f64 / 4.0) as isize;
    let idxs: Vec<usize> = ts
        .distance_profile(motif.idx_a, zeucl)
        .iter()
        .enumerate()
        .filter_map(|(i, d)| if *d <= 2.0 * mdist { Some(i) } else { None })
        .collect();

    let mut idxs: Vec<usize> = idxs
        .into_iter()
        .filter(|&i| {
            (i as isize - motif.idx_a as isize).abs() >= thresh
                || (i as isize - motif.idx_b as isize).abs() >= thresh
        })
        .collect();
    idxs.sort();
    let mut output_idxs = Vec::new();
    output_idxs.push(idxs[0]);
    for i in idxs.into_iter() {
        if (i as isize - *output_idxs.last().unwrap() as isize).abs() >= thresh {
            output_idxs.push(i);
        }
    }

    output_idxs
        .into_iter()
        .map(|i| {
            let r = i..i + w;
            (r.clone(), ts.data[r].to_vec())
        })
        .collect()
}

fn setup_logger() -> Result<GlobalLoggerGuard> {
    let log_path = ".trace.log";
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
