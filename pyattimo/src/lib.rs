use anyhow::Context;
use attimo::motifs::MotifsEnumerator;
use attimo::timeseries::WindowedTimeseries;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

#[pyclass]
struct Motif {
    #[pyo3(get)]
    a: usize,
    #[pyo3(get)]
    b: usize,
    #[pyo3(get)]
    distance: f64,
}

impl From<attimo::motifs::Motif> for Motif {
    fn from(m: attimo::motifs::Motif) -> Motif {
        Self {
            a: m.idx_a,
            b: m.idx_b,
            distance: m.distance,
        }
    }
}

#[pyclass]
struct MotifsIterator {
    inner: MotifsEnumerator,
}

#[pymethods]
impl MotifsIterator {
    #[new]
    #[args(seed = "1234", delta = "0.05", repetitions = "200")]
    fn new(
        ts: Vec<f64>,
        w: usize,
        max_k: usize,
        repetitions: usize,
        delta: f64,
        seed: u64,
    ) -> Self {
        let ts = Arc::new(WindowedTimeseries::new(ts, w, false));
        let inner = MotifsEnumerator::new(ts, max_k, repetitions, delta, seed);
        Self { inner }
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<Motif> {
        slf.inner.next_motif().map(Motif::from)
    }
}

#[pyfunction(prefix = "None")]
fn load_dataset(dataset: &str, prefix: Option<usize>) -> anyhow::Result<Vec<f64>> {
    use std::path::PathBuf;

    let mapping: HashMap<String, String> = [
        ("astro", "https://figshare.com/ndownloader/files/36982360"),
        ("ecg", "https://figshare.com/ndownloader/files/36982384"),
        ("freezer", "https://figshare.com/ndownloader/files/36982390"),
        ("gap", "https://figshare.com/ndownloader/files/36982396"),
    ]
    .into_iter()
    .map(|pair| (pair.0.to_owned(), pair.1.to_owned()))
    .collect();
    let mut outfname = PathBuf::new();
    outfname.push(dataset);
    outfname.set_extension("csv.gz");
    if !outfname.is_file() {
        let url = mapping.get(dataset).with_context(|| {
            let available = mapping.keys().cloned().collect::<Vec<String>>().join(", ");
            format!("Dataset {} not available [try {}]", dataset, available)
        })?; // FIXME better error handling
        eprintln!("Downloading file from {} into {}", url, outfname.display());
        let mut out = std::fs::File::create(&outfname)?;
        let resp = ureq::get(url).call().unwrap(); // FIXME handle errors in a better way
        let mut reader = resp.into_reader();
        std::io::copy(&mut reader, &mut out)?;
    }

    attimo::load::loadts(&outfname, prefix)
}

#[pymodule]
fn pyattimo(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load_dataset, m)?)?;
    m.add_class::<MotifsIterator>()?;
    Ok(())
}
