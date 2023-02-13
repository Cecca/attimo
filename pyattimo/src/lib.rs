use anyhow::Context;
use attimo::motifs::MotifsEnumerator;
use attimo::timeseries::WindowedTimeseries;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::HashMap;
use std::sync::Arc;

#[pyclass]
#[derive(Clone)]
struct Motif {
    #[pyo3(get)]
    a: usize,
    #[pyo3(get)]
    b: usize,
    #[pyo3(get)]
    distance: f64,
    ts: Arc<WindowedTimeseries>,
}

impl Motif {
    fn with_context(m: attimo::motifs::Motif, ts: Arc<WindowedTimeseries>) -> Self {
        Self {
            a: m.idx_a,
            b: m.idx_b,
            distance: m.distance,
            ts,
        }
    }
}

const PLOT_SCRIPT: &str = r#"
import matplotlib.pyplot as plt
fig, axs = plt.subplots(3, gridspec_kw={'height_ratios': [0.5, 1, 0.5]})
axs[0].plot(timeseries, color = "gray")
axs[0].axvline(a, color="red")
axs[0].axvline(b, color="red")
axs[0].set_title("motif in context")

axs[1].plot(motif.values_a())
axs[1].plot(motif.values_b())
axs[1].set_title("original motif subsequences")

axs[2].plot(motif.zvalues_a())
axs[2].plot(motif.zvalues_b())
axs[2].set_title("z-normalized subsequences")
fig.suptitle("z-normalized distance {}".format(distance))

plt.tight_layout()

if show:
    plt.show()
"#;

#[pymethods]
impl Motif {
    fn values_a(&self) -> Vec<f64> {
        self.ts.subsequence(self.a).to_vec()
    }
    fn values_b(&self) -> Vec<f64> {
        self.ts.subsequence(self.b).to_vec()
    }

    fn zvalues_a(&self) -> Vec<f64> {
        let mut z = vec![0.0; self.ts.w];
        self.ts.znormalized(self.a, &mut z);
        z
    }

    fn zvalues_b(&self) -> Vec<f64> {
        let mut z = vec![0.0; self.ts.w];
        self.ts.znormalized(self.b, &mut z);
        z
    }


    #[args(show = "false")]
    fn plot(&self, show: bool) -> Result<(), PyErr> {
        // Downsample the original data, if needed
        let downsampled_len = 100000;
        let (timeseries, a, b) = if self.ts.data.len() > downsampled_len {
            let keep_every = self.ts.data.len() / downsampled_len;
            let timeseries: Vec<f64> = self.ts.data.iter().step_by(keep_every).cloned().collect();
            let a = self.a / keep_every;
            let b = self.b / keep_every;
            (timeseries, a, b)
        } else {
            (self.ts.data.clone(), self.a, self.b)
        };
        Python::with_gil(|py| {
            let locals = PyDict::new(py);
            locals.set_item("motif", PyCell::new(py, self.clone()).unwrap());
            locals.set_item("timeseries", timeseries);
            locals.set_item("a", a);
            locals.set_item("b", b);
            locals.set_item("show", show);
            locals.set_item("distance", self.distance);
            py.run(PLOT_SCRIPT, None, Some(locals))
        })
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
        slf.inner
            .next_motif()
            .map(|m| Motif::with_context(m, slf.inner.get_ts()))
    }

    fn __len__(&self) -> usize {
        self.inner.max_k
    }

    fn __getitem__(&mut self, idx: isize) -> Motif {
        assert!(idx >= 0);
        Motif::with_context(self.inner.get_ranked(idx as usize).clone(), self.inner.get_ts())
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
