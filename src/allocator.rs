/// This module implements a thin wrapper around the system allocator
/// that allows to count how many bytes are allocated.
///
/// Taken from the [rust documentation](https://doc.rust-lang.org/std/alloc/struct.System.html)
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
use std::thread::JoinHandle;
use std::time::Duration;
use slog_scope::info;

pub struct CountingAllocator;

static ALLOCATED: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ret = System.alloc(layout);
        if !ret.is_null() {
            ALLOCATED.fetch_add(layout.size(), SeqCst);
        }
        return ret;
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        ALLOCATED.fetch_sub(layout.size(), SeqCst);
    }
}

pub fn allocated() -> usize {
    ALLOCATED.load(SeqCst)
}

pub fn monitor(period: Duration, flag: Arc<AtomicBool>) -> JoinHandle<()> {
    std::thread::spawn(move || {
        while flag.load(SeqCst) {
            info!("memory"; "tag" => "memory", "mem_bytes" => allocated());
            std::thread::sleep(period);
        }
        ()
    })
}
