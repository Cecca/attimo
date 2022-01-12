use slog_scope::info;
/// This module implements a thin wrapper around the system allocator
/// that allows to count how many bytes are allocated.
///
/// Taken from the [rust documentation](https://doc.rust-lang.org/std/alloc/struct.System.html)
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

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
        let mut last = 0;
        let start = Instant::now();
        while flag.load(SeqCst) {
            let mem = allocated();
            if mem != last {
                let elapsed = start.elapsed().as_millis();
                info!("memory"; "tag" => "memory", "elapsed_ms" => elapsed, "mem_bytes" => allocated());
                last = mem;
            }
            std::thread::sleep(period);
        }
        ()
    })
}

#[macro_export]
macro_rules! alloc_cnt {
    ($what:literal; $body:block) => {
        {
            let __mem_before = allocated() as isize;
            let r = $body;
            let __mem_after = allocated() as isize;
            info!(
                "allocated";
                "what" => $what,
                "memory_bytes" => __mem_after - __mem_before,
                "memory_gb" => (__mem_after - __mem_before) as f64 / (1024.0 * 1024.0 * 1024.0),
            );
            r
        }
    };
}
