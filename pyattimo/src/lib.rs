use attimo::allocator::*;
use attimo::motiflets::brute_force_motiflets;
use attimo::timeseries::WindowedTimeseries;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;

#[global_allocator]
static A: CountingAllocator = CountingAllocator;

#[pyclass]
#[derive(Clone)]
pub struct Motif {
    #[pyo3(get)]
    a: usize,
    #[pyo3(get)]
    b: usize,
    #[pyo3(get)]
    distance: f64,
    #[pyo3(get)]
    relative_contrast: f64,
    ts: Arc<WindowedTimeseries>,
}

impl TryFrom<KMotiflet> for Motif {
    type Error = &'static str;
    fn try_from(motiflet: KMotiflet) -> Result<Self, Self::Error> {
        if motiflet.indices.len() != 2 {
            Err("only motiflets of support 2 can be converted to motifs")
        } else {
            let a = motiflet.indices[0].min(motiflet.indices[1]);
            let b = motiflet.indices[0].max(motiflet.indices[1]);
            Ok(Motif {
                a,
                b,
                distance: motiflet.extent,
                ts: Arc::clone(&motiflet.ts),
                relative_contrast: motiflet.relative_contrast,
            })
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
    #[pyo3(get)]
    relative_contrast: f64,
    #[pyo3(get)]
    lower_bound: f64,
    ts: Arc<WindowedTimeseries>,
}

impl KMotiflet {
    // fn new(extent: f64, indices: Vec<usize>, support: usize, ts: Arc<WindowedTimeseries>) -> Self {
    //     todo!()
    //     // Self {
    //     //     support,
    //     //     indices,
    //     //     extent,
    //     //     ts,
    //     // }
    // }
    fn with_context(motiflet: attimo::motiflets::Motiflet, ts: Arc<WindowedTimeseries>) -> Self {
        Self {
            support: motiflet.support(),
            indices: motiflet.indices(),
            extent: motiflet.extent(),
            relative_contrast: motiflet.relative_contrast(),
            lower_bound: motiflet.lower_bound(),
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
        let (timeseries, indices) = if self.ts.data.len() > downsampled_len {
            let keep_every = self.ts.data.len() / downsampled_len;
            let timeseries: Vec<f64> = self.ts.data.iter().step_by(keep_every).cloned().collect();
            let indices: Vec<usize> = self.indices.iter().map(|i| *i / keep_every).collect();
            (timeseries, indices)
        } else {
            (self.ts.data.clone(), self.indices.clone())
        };
        Python::with_gil(|py| {
            let locals = PyDict::new_bound(py);
            locals.set_item("motif", Bound::new(py, self.clone()).unwrap())?;
            locals.set_item("timeseries", timeseries)?;
            locals.set_item("show", show)?;
            locals.set_item("indices", &indices)?;
            py.run_bound(PLOT_SCRIPT_MULTI, None, Some(&locals))
        })
    }

    fn __str__(&self) -> String {
        format!("motiflet: {:?} extent={}", self.indices, self.extent)
    }

    fn __repr__(&self) -> String {
        format!("KMotiflet{{ {:?}, extent={} }}", self.indices, self.extent)
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
            let locals = PyDict::new_bound(py);
            locals.set_item("motif", Bound::new(py, self.clone()).unwrap())?;
            locals.set_item("timeseries", timeseries)?;
            locals.set_item("a", a)?;
            locals.set_item("b", b)?;
            locals.set_item("show", show)?;
            locals.set_item("distance", self.distance)?;
            py.run_bound(PLOT_SCRIPT, None, Some(&locals))
        })
    }

    fn __str__(&self) -> String {
        format!("motif: ({}, {}) d={}", self.a, self.b, self.distance)
    }
}

#[pyclass]
struct MotifsIterator {
    inner: MotifletsIterator,
}

#[pymethods]
impl MotifsIterator {
    #[new]
    #[pyo3(signature=(ts, w, top_k=1, max_memory=None, exclusion_zone=None, delta = 0.05, seed = 1234, brute_force_threshold=1000, observability_file= None, fraction_threshold=0.1, stop_on_threshold=false))]
    fn new(
        ts: Vec<f64>,
        w: usize,
        top_k: usize,
        max_memory: Option<String>,
        exclusion_zone: Option<usize>,
        delta: f64,
        seed: u64,
        brute_force_threshold: usize,
        observability_file: Option<PathBuf>,
        fraction_threshold: f64,
        stop_on_threshold: bool,
    ) -> Self {
        let inner = MotifletsIterator::new(
            ts,
            w,
            2,
            top_k,
            max_memory,
            exclusion_zone,
            delta,
            seed,
            brute_force_threshold,
            observability_file,
            fraction_threshold,
            stop_on_threshold,
        );

        Self { inner }
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<Motif>> {
        let py = slf.py();
        slf.inner.next(py).map(|m| m.map(|m| m.try_into().unwrap()))
    }
}

/// Inner implementation of the motiflets iterator, which precomputes
/// the motiflets if the time series is small enough, otherwise defers
/// the computation to the enumerator proper.
#[allow(clippy::large_enum_variant)]
enum MotifletsIteratorImpl {
    Enumerator(attimo::motiflets::MotifletsIterator),
    BruteForce(usize, Vec<KMotiflet>),
}

#[pyclass]
struct MotifletsIterator {
    // inner: attimo::motiflets::MotifletsIterator,
    inner: MotifletsIteratorImpl,
    #[pyo3(get)]
    observability_file: PathBuf,
}

#[pymethods]
impl MotifletsIterator {
    #[new]
    #[pyo3(signature=(ts, w, support=2, top_k=1, max_memory=None, exclusion_zone=None, delta = 0.05, seed = 1234, brute_force_threshold=1000, observability_file=None, fraction_threshold=0.1, stop_on_threshold=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        ts: Vec<f64>,
        w: usize,
        support: usize,
        top_k: usize,
        max_memory: Option<String>,
        exclusion_zone: Option<usize>,
        delta: f64,
        seed: u64,
        brute_force_threshold: usize,
        observability_file: Option<PathBuf>,
        fraction_threshold: f64,
        stop_on_threshold: bool,
    ) -> Self {
        let ts = Arc::new(WindowedTimeseries::new(ts, w, false));
        let exclusion_zone = exclusion_zone.unwrap_or(w);
        assert!(
            support * exclusion_zone <= ts.num_subsequences(),
            "max_k * exclusion_zone should be less than the number of subsequences. We have instead {} * {} > {}",
            support, exclusion_zone, ts.num_subsequences()
        );
        let observability_file = observability_file.unwrap_or_else(|| {
            std::env::temp_dir().join(
                format!(
                    "attimo-{}.csv",
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_millis()
                )
                .as_str(),
            )
        });
        attimo::observe::reset_observer(&observability_file);
        let num_pairs = ts.num_subsequence_pairs();
        if ts.num_subsequences() > brute_force_threshold {
            let max_memory = if let Some(max_mem_str) = max_memory {
                Bytes::from_str(&max_mem_str).expect("cannot parse memory string")
            } else {
                let sysmem = Bytes::system_memory();
                sysmem.divide(2)
            };
            attimo::allocator::set_maximum_allocation_limit(max_memory * 1.2);
            let mut iter = attimo::motiflets::MotifletsIterator::new(
                ts,
                support,
                top_k,
                max_memory,
                delta,
                exclusion_zone,
                seed,
                false,
            );
            iter.set_collision_threshold((num_pairs as f64 * fraction_threshold) as usize);
            iter.set_stop_on_collisions_threshold(stop_on_threshold);
            let inner = MotifletsIteratorImpl::Enumerator(iter);
            Self {
                inner,
                observability_file,
            }
        } else {
            println!(
                "Brute forcing the solution, as the instance is smaller than {} subsequences",
                brute_force_threshold
            );
            let average_distance = ts.average_pairwise_distance(1234, exclusion_zone);
            let motiflets = brute_force_motiflets(&ts, support, exclusion_zone)
                .into_iter()
                .map(|(extent, indices)| KMotiflet {
                    support: indices.len(),
                    indices,
                    extent: extent.into(),
                    ts: Arc::clone(&ts),
                    lower_bound: extent.into(),
                    relative_contrast: average_distance / extent.0,
                })
                .collect();
            Self {
                inner: MotifletsIteratorImpl::BruteForce(0, motiflets),
                observability_file,
            }
        }
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<KMotiflet>> {
        let py = slf.py();
        let res = slf.next(py);
        if let Ok(None) = res.as_ref() {
            // flush the observer if we are done with it
            attimo::observe::flush_observer();
        }

        res
    }
}

impl MotifletsIterator {
    fn next(&mut self, py: Python) -> PyResult<Option<KMotiflet>> {
        match &mut self.inner {
            MotifletsIteratorImpl::Enumerator(inner) => {
                let res = inner
                    .next_interruptible(|| Python::check_signals(py))?
                    .map(|m| KMotiflet::with_context(m, inner.get_ts()));
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
    let average_distance = ts.average_pairwise_distance(1234, exclusion_zone);
    let motiflets = brute_force_motiflets(&ts, support, exclusion_zone);
    motiflets
        .into_iter()
        .map(|(extent, indices)| KMotiflet {
            support: indices.len(),
            indices,
            extent: extent.into(),
            ts: Arc::clone(&ts),
            lower_bound: extent.into(),
            relative_contrast: average_distance / extent.0,
        })
        .collect()
}

#[pyfunction]
#[pyo3(signature=(path, prefix=None))]
pub fn loadts(path: &str, prefix: Option<usize>) -> Vec<f64> {
    attimo::load::loadts(path, prefix).expect("error loading time series")
}

#[pymodule]
fn pyattimo(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();
    m.add_function(wrap_pyfunction!(motiflet_brute_force, m)?)?;
    m.add_function(wrap_pyfunction!(loadts, m)?)?;
    m.add_class::<MotifsIterator>()?;
    m.add_class::<MotifletsIterator>()?;
    Ok(())
}
