use anyhow::Result;
use argh::FromArgs;
use attimo::allocator::{self, Bytes, CountingAllocator, MemoryGauge};
use attimo::load::*;
use attimo::motiflets::{brute_force_motiflets, Motiflet, MotifletsIterator};
use attimo::timeseries::*;
use std::path::Path;
use std::str::FromStr;
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

    #[argh(option, default = "0.01")]
    /// failure probability of the LSH scheme
    pub failure_probability: f64,

    #[argh(option)]
    /// the number of repetitions to perform
    pub max_memory: Option<String>,

    #[argh(switch)]
    /// use the exact algorithm
    pub exact: bool,

    #[argh(option, default = "2")]
    /// the maximum support of the motiflets to find
    pub support: usize,

    #[argh(option, default = "1")]
    /// number of motiflets to find for each support value
    pub topk: usize,

    #[argh(option)]
    /// consider only the given number of points from the input
    pub prefix: Option<usize>,

    #[argh(option, default = "12453")]
    /// seed for the psudorandom number generator
    pub seed: u64,

    #[cfg(pprof)]
    #[argh(switch)]
    /// profile the code while running, and save a `profile.pb` file to open with pprof
    pub profile: bool,

    #[argh(switch)]
    /// wether meand and std computations should be at the best precision, at the expense of running time
    pub precise: bool,

    #[argh(option, default = "default_output()")]
    /// path to the output file
    pub output: String,

    #[argh(positional)]
    /// path to the data file
    pub path: String,
}

fn default_output() -> String {
    "motiflets.csv".to_owned()
}

fn main() -> Result<()> {
    debug_assert!(false, "This executable should be run in release mode");
    env_logger::init();
    let total_timer = Instant::now();
    if std::env::args().filter(|arg| arg == "--version").count() == 1 {
        println!("{}", VERSION);
        return Ok(());
    }

    // read configuration
    let config: Config = argh::from_env();

    let monitor_flag = Arc::new(AtomicBool::new(true));
    let monitor = allocator::monitor(Duration::from_millis(200), Arc::clone(&monitor_flag));

    let path = config.path.clone();
    let w = config.window;
    let mem = MemoryGauge::allocated();
    let timer = Instant::now();
    let ts: Vec<f64> = loadts(path, config.prefix)?;
    log::info!(
        "Loaded raw data in {:?}: {}",
        timer.elapsed(),
        mem.measure()
    );
    let timer = Instant::now();
    let ts = WindowedTimeseries::new(ts, w, config.precise);
    let ts_bytes = mem.measure();
    let input_elapsed = timer.elapsed();
    log::info!(
        "Create windowed time series with {} subsequences in {:?}, taking {} ({}), {} flat subsequences",
        ts.num_subsequences(),
        input_elapsed,
        ts_bytes,
        ts.memory(),
        ts.count_flat(),
    );

    #[cfg(pprof)]
    let _profiler = if config.profile {
        use pprof::ProfilerGuard;
        // The profile will be saved on drop
        Some(Profiler::start())
    } else {
        None
    };

    let support = config.support;
    let motiflets: Vec<Motiflet> = if config.exact {
        let motiflets = brute_force_motiflets(&ts, support, ts.w / 2);
        motiflets
            .into_iter()
            .map(|(extent, indices)| Motiflet::new(indices, extent.into()))
            .collect()
    } else {
        let max_memory = if let Some(max_mem_str) = config.max_memory {
            Bytes::from_str(&max_mem_str)?
        } else {
            let sysmem = Bytes::system_memory();
            let mem = sysmem.divide(2);
            log::info!("System has {} memory, using {} at most", sysmem, mem);
            mem
        };
        let start = Instant::now();
        let exclusion_zone = ts.w / 2;
        MotifletsIterator::new(
            Arc::new(ts),
            support,
            config.topk,
            max_memory,
            config.failure_probability,
            exclusion_zone,
            config.seed,
            false,
        )
        .map(|m| {
            eprintln!(
                "[{:?}] discovered motiflet with support {} and extent {}",
                start.elapsed(),
                m.support(),
                m.extent()
            );
            m
        })
        .collect()
    };
    eprintln!("Result: {:?}", motiflets);
    output_csv(config.output, &motiflets)?;

    monitor_flag.store(false, std::sync::atomic::Ordering::SeqCst);
    monitor.join().unwrap();

    eprintln!("Total time {:?}", total_timer.elapsed());

    #[cfg(feature = "observe")]
    attimo::observe::OBSERVER.lock().unwrap().flush();

    Ok(())
}

#[cfg(pprof)]
struct Profiler<'a> {
    profiler: ProfilerGuard<'a>,
}

#[cfg(pprof)]
impl<'a> Profiler<'a> {
    fn start() -> Self {
        log::info!("Start profiler");
        let profiler = pprof::ProfilerGuardBuilder::default()
            .frequency(999)
            .blocklist(&["libc", "libgcc", "pthread", "vdso", "rayon_core"])
            .build()
            .unwrap();
        Self { profiler }
    }
}

#[cfg(pprof)]
impl<'a> Drop for Profiler<'a> {
    fn drop(&mut self) {
        use pprof::protos::Message;
        use std::io::Write;

        log::info!("Saving profile");
        match self.profiler.report().build() {
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
}

fn output_csv<P: AsRef<Path>>(path: P, motifs: &[Motiflet]) -> Result<()> {
    use std::io::prelude::*;
    let mut f = std::fs::File::create(path.as_ref())?;
    for m in motifs {
        let istrings: Vec<String> = m.indices().into_iter().map(|i| format!("{}", i)).collect();
        writeln!(f, "{}, {}", m.extent(), istrings.join(", "))?;
    }
    Ok(())
}
