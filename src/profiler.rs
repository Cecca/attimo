use pprof::ProfilerGuard;

pub struct Profiler<'a> {
    profiler: ProfilerGuard<'a>,
}

impl<'a> Profiler<'a> {
    pub fn start() -> Self {
        log::info!("Start profiler");
        let profiler = pprof::ProfilerGuardBuilder::default()
            .frequency(999)
            .blocklist(&["libc", "libgcc", "pthread", "vdso", "rayon_core"])
            .build()
            .unwrap();
        Self { profiler }
    }
}

impl<'a> Drop for Profiler<'a> {
    fn drop(&mut self) {
        use pprof::protos::Message;
        use std::io::Write;

        log::info!("Saving profile");
        if let Ok(report) = self.profiler.report().build() {
            let mut file = std::fs::File::create("profile.pb").unwrap();
            let profile = report.pprof().unwrap();

            let mut content = Vec::new();
            profile.encode(&mut content).unwrap();
            file.write_all(&content).unwrap();
        };
    }
}
