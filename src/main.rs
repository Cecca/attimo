use anyhow::Result;
use argh::FromArgs;
use attimo::allocator::{self, allocated, CountingAllocator};
use attimo::load::*;
use attimo::motifs::{motiflets, motifs, Motif, Motiflet};
use attimo::timeseries::*;
use slog::*;
use slog_scope::GlobalLoggerGuard;
use std::fs::OpenOptions;
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::{Duration, Instant};

const VERSION: u32 = 21;

#[global_allocator]
static A: CountingAllocator = CountingAllocator;

#[derive(FromArgs)]
/// ATTIMO computes AdapTive TImeseries MOtifs.
struct Config {
    #[argh(option, short = 'w')]
    /// subsequcence length
    pub window: usize,

    #[argh(option, default = "1")]
    /// the number of motifs to look for
    pub motifs: usize,

    #[argh(option, default = "0.001")]
    /// failure probability of the LSH scheme
    pub failure_probability: f64,

    #[argh(option)]
    /// the number of repetitions to perform
    pub repetitions: usize,

    #[argh(option)]
    /// find motiflets, with the specified support
    pub motiflets: Option<usize>,

    #[argh(option)]
    /// consider only the given number of points from the input
    pub prefix: Option<usize>,

    #[argh(option, default = "12453")]
    /// seed for the psudorandom number generator
    pub seed: u64,

    #[argh(switch)]
    /// profile the code while running, and save a `profile.pb` file to open with pprof
    pub profile: bool,

    #[argh(switch)]
    /// wether meand and std computations should be at the best precision, at the expense of running time
    pub precise: bool,

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

fn default_output() -> String {
    "motifs.csv".to_owned()
}

fn default_log_path() -> String {
    ".trace.json".to_owned()
}

fn main() -> Result<()> {
    debug_assert!(false, "This executable should be run in release mode");
    let total_timer = Instant::now();
    if std::env::args().filter(|arg| arg == "--version").count() == 1 {
        println!("{}", VERSION);
        return Ok(());
    }

    // read configuration
    let config: Config = argh::from_env();

    let monitor_flag = Arc::new(AtomicBool::new(true));
    let monitor = allocator::monitor(Duration::from_millis(200), Arc::clone(&monitor_flag));

    let _guard = setup_logger(&config.log_path)?;
    slog_scope::info!("input reading"; "tag" => "phase");
    let path = config.path.clone();
    let w = config.window;
    let timer = Instant::now();
    let ts: Vec<f64> = loadts(path, config.prefix)?;
    println!("Loaded raw data in {:?}", timer.elapsed());
    let timer = Instant::now();
    let mem_before = allocated();
    let ts = WindowedTimeseries::new(ts, w, config.precise);
    let ts_bytes = allocated() - mem_before;
    let input_elapsed = timer.elapsed();
    println!(
        "Create windowed time series in {:?}, taking {}",
        input_elapsed,
        PrettyBytes(ts_bytes)
    );
    slog_scope::info!("input reading";
        "tag" => "profiling",
        "time_s" => input_elapsed.as_secs_f64()
    );

    let profiler = if config.profile {
        Some(
            pprof::ProfilerGuardBuilder::default()
                .frequency(1000)
                .blocklist(&["libc", "libgcc", "pthread", "vdso"])
                .build()
                .unwrap(),
        )
    } else {
        None
    };

    if let Some(support) = config.motiflets {
        let motiflets: Vec<Motiflet> = motiflets(
            Arc::new(ts),
            support,
            config.repetitions,
            config.failure_probability,
            config.seed,
        );
        eprintln!("Result: {:?}", motiflets);
    } else {
        let motifs: Vec<Motif> = motifs(
            Arc::new(ts),
            config.motifs,
            config.repetitions,
            config.failure_probability,
            config.seed,
        );
        output_csv(&config.output, &motifs)?;
    }

    if let Some(profiler) = profiler {
        use pprof::protos::Message;
        use std::io::Write;
        match profiler.report().build() {
            Ok(report) => {
                let mut file = std::fs::File::create("profile.pb").unwrap();
                let profile = report.pprof().unwrap();

                let mut content = Vec::new();
                profile.encode(&mut content).unwrap();
                file.write_all(&content).unwrap();
            }
            Err(_) => {}
        };
    }

    monitor_flag.store(false, std::sync::atomic::Ordering::SeqCst);
    monitor.join().unwrap();

    println!("Total time {:?}", total_timer.elapsed());

    Ok(())
}

fn output_csv<P: AsRef<Path>>(path: P, motifs: &[Motif]) -> Result<()> {
    use std::io::prelude::*;
    let mut f = std::fs::File::create(path.as_ref())?;
    for m in motifs {
        if let Some(confirmation_time) = m.elapsed {
            writeln!(
                f,
                "{}, {}, {}, {}",
                m.idx_a,
                m.idx_b,
                m.distance,
                confirmation_time.as_secs_f64()
            )?;
        }
    }
    Ok(())
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
