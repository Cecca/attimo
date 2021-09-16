use anyhow::Result;
use argh::FromArgs;
use attimo::distance::zeucl;
use attimo::load::*;
use attimo::motifs::{motifs, Motif};
use attimo::timeseries::*;
use plotly::common::{Line, Mode};
use plotly::{Layout, Plot, Scatter};
use slog::*;
use slog_scope::GlobalLoggerGuard;
use std::fs::OpenOptions;
use std::rc::Rc;

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

fn main() -> Result<()> {
    let _guard = setup_logger()?;

    // read configuration
    let config: Config = argh::from_env();
    let path = config.path;
    let w = config.window;
    let ts: Vec<f64> = loadts(path, config.prefix)?;
    let ts = Rc::new(WindowedTimeseries::new(ts, w));
    println!("Loaded time series, taking {}", ts.bytes_size());
    let motifs = motifs(
        Rc::clone(&ts),
        config.motifs,
        config.repetitions,
        config.delta,
        config.seed,
    );
    println!("{:#?}", motifs);

    // TODO: output all motifs to CSV
    println!("Top motif: {:?}", motifs[0]);

    if config.plot {
        let mut plot = Plot::new();
        let layout = Layout::default().height(500);
        plot.set_layout(layout);

        // let ts_lines = Scatter::new(0..ts.data.len(), ts.data.clone())
        //     .name("time series")
        //     .line(Line::new().color("#667393").width(0.5))
        //     .mode(Mode::Lines);
        // plot.add_trace(ts_lines);

        let motif_range = 0..ts.w;
        // Provide some context time series
        let n_context = 50;
        let stride = ts.num_subsequences() / n_context;
        let mut idx = 0;
        while idx < ts.num_subsequences() {
            let mut vals = vec![0.0;ts.w];
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
            let mut vals = vec![0.0;ts.w];
            ts.znormalized(i, &mut vals);
            let l = Scatter::new(motif_range.clone(), vals)
                .line(Line::new().color("#f7a61b").width(1.5))
                .show_legend(false)
                .mode(Mode::Lines);
            plot.add_trace(l);
        }
        for i in [motifs[0].idx_a, motifs[0].idx_b] {
            let mut vals = vec![0.0;ts.w];
            ts.znormalized(i, &mut vals);
            let l = Scatter::new(motif_range.clone(), vals)
                .line(Line::new().color("#db4620").width(3.0))
                .show_legend(false)
                .mode(Mode::Lines);
            plot.add_trace(l);
        }
        plot.show();
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
