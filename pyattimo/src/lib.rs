use attimo::allocator::Bytes;
use attimo::motiflets::brute_force_motiflets;
use attimo::motifs::{MotifsEnumerator, PairMotifState};
use attimo::timeseries::WindowedTimeseries;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::str::FromStr;
use std::sync::Arc;

#[pyclass]
#[derive(Clone)]
pub struct Motif {
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

#[pyclass]
#[derive(Clone)]
pub struct KMotiflet {
    #[pyo3(get)]
    support: usize,
    #[pyo3(get)]
    indices: Vec<usize>,
    #[pyo3(get)]
    extent: f64,
    ts: Arc<WindowedTimeseries>,
}

impl KMotiflet {
    fn new(extent: f64, indices: Vec<usize>, support: usize, ts: Arc<WindowedTimeseries>) -> Self {
        Self {
            support,
            indices,
            extent,
            ts,
        }
    }
}

#[pymethods]
impl KMotiflet {
    fn values(&self, i: usize) -> Vec<f64> {
        self.ts.subsequence(self.indices[i]).to_vec()
    }

    fn zvalues(&self, i: usize) -> Vec<f64> {
        let mut z = vec![0.0; self.ts.w];
        self.ts.znormalized(self.indices[i], &mut z);
        z
    }

    #[pyo3(signature = (show = false))]
    fn plot(&self, show: bool) -> Result<(), PyErr> {
        // Downsample the original data, if needed
        let downsampled_len = 100000;
        let timeseries = if self.ts.data.len() > downsampled_len {
            let keep_every = self.ts.data.len() / downsampled_len;
            let timeseries: Vec<f64> = self.ts.data.iter().step_by(keep_every).cloned().collect();
            timeseries
        } else {
            self.ts.data.clone()
        };
        Python::with_gil(|py| {
            let locals = PyDict::new(py);
            locals.set_item("motif", PyCell::new(py, self.clone()).unwrap())?;
            locals.set_item("timeseries", timeseries)?;
            locals.set_item("show", show)?;
            locals.set_item("indices", &self.indices)?;
            py.run(PLOT_SCRIPT_MULTI, None, Some(locals))
        })
    }

    fn __str__(&self) -> String {
        format!("motiflet: {:?} extent={}", self.indices, self.extent)
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

const PLOT_SCRIPT_MULTI: &str = r#"
import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, gridspec_kw={'height_ratios': [0.5, 1]})
axs[0].plot(timeseries, color = "gray")
axs[0].set_title("motiflet in context")

for i in range(len(indices)):
    axs[0].axvline(indices[i], color="red")
    axs[1].plot(motif.zvalues(i))

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

    #[pyo3(signature = (show = false))]
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
            locals.set_item("motif", PyCell::new(py, self.clone()).unwrap())?;
            locals.set_item("timeseries", timeseries)?;
            locals.set_item("a", a)?;
            locals.set_item("b", b)?;
            locals.set_item("show", show)?;
            locals.set_item("distance", self.distance)?;
            py.run(PLOT_SCRIPT, None, Some(locals))
        })
    }

    fn __str__(&self) -> String {
        format!("motif: ({}, {}) d={}", self.a, self.b, self.distance)
    }
}

#[pyclass]
struct MotifsIterator {
    inner: MotifsEnumerator<PairMotifState>,
}

#[pymethods]
impl MotifsIterator {
    #[new]
    #[pyo3(signature=(ts, w, max_k = 100, repetitions=1000, delta = 0.05, seed = 1234))]
    fn new(
        ts: Vec<f64>,
        w: usize,
        max_k: usize,
        repetitions: usize,
        delta: f64,
        seed: u64,
    ) -> Self {
        let ts = Arc::new(WindowedTimeseries::new(ts, w, false));
        let inner = MotifsEnumerator::new(
            ts,
            max_k,
            repetitions,
            delta,
            || PairMotifState::new(max_k, w),
            seed,
            false,
        );
        Self { inner }
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<Motif> {
        slf.inner
            .next()
            .map(|m| Motif::with_context(m, slf.inner.get_ts()))
    }

    fn __len__(&self) -> usize {
        self.inner.max_k
    }

    fn __getitem__(&mut self, idx: isize) -> Motif {
        assert!(idx >= 0);
        Motif::with_context(
            self.inner.get_ranked(idx as usize).clone(),
            self.inner.get_ts(),
        )
    }
}

/// Inner implementation of the motiflets iterator, which precomputes
/// the motiflets if the time series is small enough, otherwise defers
/// the computation to the enumerator proper.
enum MotifletsIteratorImpl {
    Enumerator(attimo::motiflets::MotifletsIterator),
    BruteForce(usize, Vec<KMotiflet>),
}

#[pyclass]
struct MotifletsIterator {
    // inner: attimo::motiflets::MotifletsIterator,
    inner: MotifletsIteratorImpl,
}

#[pymethods]
impl MotifletsIterator {
    #[new]
    #[pyo3(signature=(ts, w, max_k = 10, max_memory=None, exclusion_zone=None, delta = 0.05, seed = 1234, brute_force_threshold=1000))]
    fn new(
        ts: Vec<f64>,
        w: usize,
        max_k: usize,
        max_memory: Option<String>,
        exclusion_zone: Option<usize>,
        delta: f64,
        seed: u64,
        brute_force_threshold: usize,
    ) -> Self {
        let ts = Arc::new(WindowedTimeseries::new(ts, w, false));
        let exclusion_zone = exclusion_zone.unwrap_or(w);
        assert!(
            max_k * exclusion_zone <= ts.num_subsequences(),
            "max_k * exclusion_zone should be less than the number of subsequences. We have instead {} * {} > {}",
            max_k, exclusion_zone, ts.num_subsequences()
        );
        if ts.num_subsequences() > brute_force_threshold {
            let max_memory = if let Some(max_mem_str) = max_memory {
                Bytes::from_str(&max_mem_str).expect("cannot parse memory string")
            } else {
                let sysmem = Bytes::system_memory();
                sysmem.divide(2)
            };
            let inner =
                MotifletsIteratorImpl::Enumerator(attimo::motiflets::MotifletsIterator::new(
                    ts,
                    max_k,
                    max_memory,
                    delta,
                    exclusion_zone,
                    seed,
                    false,
                ));
            Self { inner }
        } else {
            println!(
                "Brute forcing the solution, as the instance is smaller than {} subsequences",
                brute_force_threshold
            );
            let motiflets = brute_force_motiflets(&ts, max_k, exclusion_zone)
                .into_iter()
                .map(|(extent, indices)| KMotiflet {
                    support: indices.len(),
                    indices,
                    extent: extent.into(),
                    ts: Arc::clone(&ts),
                })
                .collect();
            Self {
                inner: MotifletsIteratorImpl::BruteForce(0, motiflets),
            }
        }
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<KMotiflet>> {
        let py = slf.py();
        match &mut slf.inner {
            MotifletsIteratorImpl::Enumerator(inner) => {
                let res = inner
                    .next_interruptible(|| Python::check_signals(py))?
                    .map(|m| KMotiflet::new(m.extent(), m.indices(), m.support(), inner.get_ts()));
                Ok(res)
            }
            MotifletsIteratorImpl::BruteForce(pos, motiflets) => {
                if *pos >= motiflets.len() {
                    Ok(None)
                } else {
                    let m = motiflets[*pos].clone();
                    *pos += 1;
                    Ok(Some(m))
                }
            }
        }
    }
}

#[pyfunction]
#[pyo3(signature=(ts, w, support=3, exclusion_zone=None))]
pub fn motiflet_brute_force(
    ts: Vec<f64>,
    w: usize,
    support: usize,
    exclusion_zone: Option<usize>,
) -> Vec<KMotiflet> {
    use attimo::motiflets::*;
    let ts = Arc::new(WindowedTimeseries::new(ts, w, false));
    let exclusion_zone = exclusion_zone.unwrap_or(w / 2);
    assert!(
        support * exclusion_zone <= ts.num_subsequences(),
        "support * exclusion_zone should be less than the number of subsequences. We have instead {} * {} > {}",
        support, exclusion_zone, ts.num_subsequences()
    );
    let motiflets = brute_force_motiflets(&ts, support, exclusion_zone);
    motiflets
        .into_iter()
        .map(|(extent, indices)| KMotiflet {
            support: indices.len(),
            indices,
            extent: extent.into(),
            ts: Arc::clone(&ts),
        })
        .collect()
}

#[pymodule]
fn pyattimo(_py: Python, m: &PyModule) -> PyResult<()> {
    pyo3_log::init();
    m.add_function(wrap_pyfunction!(motiflet_brute_force, m)?)?;
    m.add_class::<MotifsIterator>()?;
    m.add_class::<MotifletsIterator>()?;
    Ok(())
}
