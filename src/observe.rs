use std::io::prelude::*;
use std::sync::Mutex;
use std::time::Instant;
use std::{fmt::Display, fs::File, io::BufWriter};

pub struct Observer {
    start: Instant,
    output: BufWriter<File>,
}

impl Observer {
    fn new(path: &str) -> Self {
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

#[cfg(feature = "observe")]
macro_rules! observe {
    ($rep: expr, $prefix: expr, $name: literal, $value: expr) => {
        OBSERVER
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
