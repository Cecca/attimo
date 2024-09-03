use std::io::prelude::*;
use std::ops::DerefMut;
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;
use std::{fmt::Display, fs::File, io::BufWriter};

pub struct Observer {
    start: Instant,
    output: BufWriter<File>,
}

impl Observer {
    fn new<P: AsRef<Path>>(path: P) -> Self {
        let mut output = BufWriter::new(File::create(path).unwrap());
        writeln!(output, "elapsed_s,repetition,prefix,name,value").unwrap();
        let start = Instant::now();
        Self { start, output }
    }

    pub fn append<V: Display>(&mut self, repetition: usize, prefix: usize, name: &str, value: V) {
        writeln!(
            self.output,
            "{},{},{},{},{}",
            self.start.elapsed().as_secs_f64(),
            repetition,
            prefix,
            name,
            value
        )
        .unwrap()
    }

    pub fn flush(&mut self) {
        self.output.flush().unwrap()
    }
}

impl Drop for Observer {
    fn drop(&mut self) {
        self.output.flush().unwrap();
    }
}

pub static OBSERVER: Lazy<Mutex<Observer>> = Lazy::new(|| Mutex::new(Observer::new("observe.csv")));

pub fn reset_observer<P: AsRef<Path>>(path: P) {
    let mut obs = OBSERVER.lock().unwrap();
    let obs = obs.deref_mut();
    *obs = Observer::new(path.as_ref());
}

pub fn flush_observer() {
    #[cfg(feature = "observe")]
    OBSERVER.lock().unwrap().flush()
}

#[cfg(feature = "observe")]
macro_rules! observe {
    ($rep: expr, $prefix: expr, $name: literal, $value: expr) => {
        crate::observe::OBSERVER
            .lock()
            .unwrap()
            .append($rep, $prefix, $name, $value);
    };
}

#[cfg(not(feature = "observe"))]
macro_rules! observe {
    ($rep: expr, $prefix: expr, $name: literal, $value: expr) => {};
}

pub(crate) use observe;
use once_cell::sync::Lazy;
