/// This module implements a thin wrapper around the system allocator
/// that allows to count how many bytes are allocated.
///
/// Taken from the [rust documentation](https://doc.rust-lang.org/std/alloc/struct.System.html)
use std::alloc::{GlobalAlloc, Layout, System};
use std::fmt::Display;
use std::ops::{Add, Sub};
use std::str::FromStr;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct ParseBytesError;
impl std::error::Error for ParseBytesError {}
impl std::fmt::Display for ParseBytesError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "error in parsing memory specification")
    }
}

#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Clone, Copy)]
pub struct Bytes(pub usize);
impl Bytes {
    pub fn kbytes(kb: usize) -> Self {
        Self(kb * 1024)
    }
    pub fn mbytes(mb: usize) -> Self {
        Self(mb * 1024 * 1024)
    }
    pub fn gbytes(gb: usize) -> Self {
        Self(gb * 1024 * 1024 * 1024)
    }
    pub fn system_memory() -> Self {
        let mut system = sysinfo::System::new_all();
        system.refresh_memory();
        let mem = system.total_memory();
        Self(mem.try_into().expect("Cannot convert u64 to usize"))
    }
    pub fn allocated() -> Self {
        Self(allocated())
    }
    pub fn divide(&self, divisor: usize) -> Self {
        Self(self.0 / divisor)
    }
}

impl Add<Bytes> for Bytes {
    type Output = Bytes;
    fn add(self, rhs: Bytes) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl Sub<Bytes> for Bytes {
    type Output = Bytes;
    fn sub(self, rhs: Bytes) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl FromStr for Bytes {
    type Err = ParseBytesError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.is_empty() {
            return Err(ParseBytesError);
        }
        let s = s.to_lowercase();
        let s = s.trim_end_matches("bytes").trim_end_matches('b');
        let suffix = s.chars().last().unwrap();
        let suffix = if suffix.is_alphabetic() {
            Some(suffix)
        } else {
            None
        };
        let num = if suffix.is_some() {
            &s[..s.len() - 1]
        } else {
            s
        };
        let num = num.trim().parse::<usize>().map_err(|_| ParseBytesError)?;
        let mult = match suffix {
            None => 1,
            Some('k') => 1024,
            Some('m') => 1024 * 1024,
            Some('g') => 1024 * 1024 * 1024,
            _ => unreachable!(),
        };
        Ok(Self(num * mult))
    }
}

#[test]
fn test_parse_bytes_string() {
    assert_eq!(Bytes::from_str("10"), Ok(Bytes(10)));
    assert_eq!(Bytes::from_str("10Kb"), Ok(Bytes(10 * 1024)));
    assert_eq!(Bytes::from_str("10Mbytes"), Ok(Bytes(10 * 1024 * 1024)));
    assert_eq!(Bytes::from_str("10 GB"), Ok(Bytes(10 * 1024 * 1024 * 1024)));
}

impl Display for Bytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0 >= 1024 * 1024 * 1024 {
            write!(f, "{:.2} Gbytes", self.0 as f64 / (1024. * 1024. * 1024.0))
        } else if self.0 >= 1024 * 1024 {
            write!(f, "{:.2} Mbytes", self.0 as f64 / (1024. * 1024.))
        } else if self.0 >= 1024 {
            write!(f, "{:.2} Kbytes", self.0 as f64 / 1024.)
        } else {
            write!(f, "{} bytes", self.0)
        }
    }
}

impl TryFrom<String> for Bytes {
    type Error = anyhow::Error;

    fn try_from(s: String) -> Result<Self, Self::Error> {
        let mut s = s.clone();
        if s.ends_with("Gb") {
            s.remove_matches("Gb");
            let num = s.parse::<f64>().map_err(|e| anyhow::anyhow!(e))?;
            let bytes = num * 1024.0 * 1024.0 * 1024.0;
            Ok(Bytes(bytes as usize))
        } else if s.ends_with("Mb") {
            s.remove_matches("Mb");
            let num = s.parse::<f64>().map_err(|e| anyhow::anyhow!(e))?;
            let bytes = num * 1024.0 * 1024.0;
            Ok(Bytes(bytes as usize))
        } else if s.ends_with("Kb") {
            s.remove_matches("Kb");
            let num = s.parse::<f64>().map_err(|e| anyhow::anyhow!(e))?;
            let bytes = num * 1024.0;
            Ok(Bytes(bytes as usize))
        } else {
            let num = s.parse::<f64>().map_err(|e| anyhow::anyhow!(e))?;
            let bytes = num;
            Ok(Bytes(bytes as usize))
        }
    }
}

pub struct CountingAllocator;

static ALLOCATED: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ret = System.alloc(layout);
        if !ret.is_null() {
            ALLOCATED.fetch_add(layout.size(), SeqCst);
        }
        ret
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        ALLOCATED.fetch_sub(layout.size(), SeqCst);
    }
}

pub fn allocated() -> usize {
    ALLOCATED.load(SeqCst)
}

pub struct MemoryGauge {
    start: Bytes,
}
impl MemoryGauge {
    pub fn allocated() -> Self {
        Self {
            start: Bytes(allocated()),
        }
    }

    /// Measure the variation in bytes since the call to [allocated] that created this instance.
    pub fn measure(&self) -> Bytes {
        let curr = Bytes(allocated());
        curr - self.start
    }
}

pub fn monitor(period: Duration, flag: Arc<AtomicBool>) -> JoinHandle<()> {
    std::thread::spawn(move || {
        let mut last = 0;
        while flag.load(SeqCst) {
            let mem = allocated();
            if mem != last {
                last = mem;
            }
            std::thread::sleep(period);
        }
    })
}
