/// This module implements a thin wrapper around the system allocator
/// that allows to count how many bytes are allocated.
///
/// Taken from the [rust documentation](https://doc.rust-lang.org/std/alloc/struct.System.html)
use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::BTreeSet;
use std::fmt::{Display, FormattingOptions};
use std::iter::Sum;
use std::ops::{Add, AddAssign, Mul, Sub};
use std::str::FromStr;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use crate::index::HashValue;
use crate::knn::Distance;

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct ParseBytesError;
impl std::error::Error for ParseBytesError {}
impl std::fmt::Display for ParseBytesError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "error in parsing memory specification")
    }
}

#[derive(Eq, PartialEq, Ord, PartialOrd, Clone, Copy, Default)]
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
    pub fn max_allocated() -> Self {
        Self(max_allocated())
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

impl Sum<Bytes> for Bytes {
    fn sum<I: Iterator<Item = Bytes>>(iter: I) -> Self {
        let mut bb = Bytes(0);
        for b in iter {
            bb += b;
        }
        bb
    }
}

impl Mul<f64> for Bytes {
    type Output = Bytes;
    fn mul(self, rhs: f64) -> Self::Output {
        Self((self.0 as f64 * rhs) as usize)
    }
}

impl AddAssign<Bytes> for Bytes {
    fn add_assign(&mut self, rhs: Bytes) {
        self.0 += rhs.0;
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
impl std::fmt::Debug for Bytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self, f)
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

static HARD_ALLOCATION_LIMIT: AtomicUsize = AtomicUsize::new(usize::MAX);
static ALLOCATED: AtomicUsize = AtomicUsize::new(0);
static MAX_ALLOCATED: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ret = System.alloc(layout);
        if !ret.is_null() {
            let currently_allocated = ALLOCATED.fetch_add(layout.size(), SeqCst);
            MAX_ALLOCATED.fetch_max(currently_allocated, SeqCst);
            if currently_allocated > HARD_ALLOCATION_LIMIT.load(SeqCst) {
                eprintln!(
                    "maximum memory allocation limit exceeded! {}",
                    Bytes(currently_allocated)
                );
                panic!(
                    "maximum memory allocation limit exceeded! {}",
                    Bytes(currently_allocated)
                );
            }
        }
        ret
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        ALLOCATED.fetch_sub(layout.size(), SeqCst);
    }
}

pub fn set_maximum_allocation_limit(maximum: Bytes) {
    log::debug!("setting maximum allocation limit to {}", maximum);
    HARD_ALLOCATION_LIMIT.store(maximum.0, SeqCst);
}
pub fn get_maximum_allocation_limit() -> Bytes {
    Bytes(HARD_ALLOCATION_LIMIT.load(SeqCst))
}

pub fn allocated() -> usize {
    ALLOCATED.load(SeqCst)
}

pub fn max_allocated() -> usize {
    MAX_ALLOCATED.load(SeqCst)
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

pub trait ByteSize {
    /// get the size, in bytes, of `Self`
    fn byte_size(&self) -> Bytes;
    /// allows to format, recursively, the size of each element
    fn mem_tree_fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result;

    fn mem_tree(&self) -> String {
        let mut s = String::new();
        let mut fmt = FormattingOptions::new()
            .alternate(true)
            .create_formatter(&mut s);
        self.mem_tree_fmt(&mut fmt).unwrap();
        s
    }
}

impl ByteSize for usize {
    fn byte_size(&self) -> Bytes {
        Bytes(std::mem::size_of_val(self))
    }
    fn mem_tree_fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.byte_size())
    }
}
impl ByteSize for u32 {
    fn byte_size(&self) -> Bytes {
        Bytes(std::mem::size_of_val(self))
    }
    fn mem_tree_fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.byte_size())
    }
}
impl ByteSize for f64 {
    fn byte_size(&self) -> Bytes {
        Bytes(std::mem::size_of_val(self))
    }
    fn mem_tree_fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.byte_size())
    }
}
impl ByteSize for HashValue {
    fn byte_size(&self) -> Bytes {
        Bytes(std::mem::size_of_val(self))
    }
    fn mem_tree_fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.byte_size())
    }
}
impl ByteSize for Distance {
    fn byte_size(&self) -> Bytes {
        Bytes(std::mem::size_of_val(self))
    }
    fn mem_tree_fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.byte_size())
    }
}
impl<A, B> ByteSize for (A, B)
where
    A: ByteSize,
    B: ByteSize,
{
    fn byte_size(&self) -> Bytes {
        self.0.byte_size() + self.1.byte_size()
    }
    fn mem_tree_fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let a = self.0.mem_tree_fmt(f);
        let b = self.0.mem_tree_fmt(f);
        f.debug_tuple("").field(&a).field(&b).finish()
    }
}
impl<A, B, C> ByteSize for (A, B, C)
where
    A: ByteSize,
    B: ByteSize,
    C: ByteSize,
{
    fn byte_size(&self) -> Bytes {
        self.0.byte_size() + self.1.byte_size() + self.2.byte_size()
    }
    fn mem_tree_fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let a = self.0.mem_tree_fmt(f);
        let b = self.1.mem_tree_fmt(f);
        let c = self.2.mem_tree_fmt(f);
        f.debug_tuple("").field(&a).field(&b).field(&c).finish()
    }
}
impl<A> ByteSize for Option<A>
where
    A: ByteSize,
{
    fn byte_size(&self) -> Bytes {
        if let Some(s) = self {
            s.byte_size()
        } else {
            Bytes(std::mem::size_of_val::<Option<A>>(&None))
        }
    }

    fn mem_tree_fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.byte_size())
    }
}

impl<A> ByteSize for Vec<A>
where
    A: ByteSize,
{
    fn byte_size(&self) -> Bytes {
        let slf = Bytes(std::mem::size_of_val(self));
        let data = self.iter().map(|elem| elem.byte_size()).sum::<Bytes>();
        slf + data
    }

    fn mem_tree_fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut d = f.debug_list();
        for elem in self.iter() {
            d.entry_with(|f| elem.mem_tree_fmt(f));
        }
        d.finish()
    }
}
impl<A> ByteSize for BTreeSet<A>
where
    A: ByteSize,
{
    fn byte_size(&self) -> Bytes {
        // this is a bit of an approximation, but we don't have access to the internal structure
        // of the implementation of the b-tree
        let data = self.iter().map(|elem| elem.byte_size()).sum::<Bytes>();
        data
    }

    fn mem_tree_fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut d = f.debug_set();
        for elem in self.iter() {
            d.entry_with(|f| elem.mem_tree_fmt(f));
        }
        d.finish()
    }
}
