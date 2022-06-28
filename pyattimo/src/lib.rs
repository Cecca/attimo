use std::sync::Arc;

use attimo::{timeseries::WindowedTimeseries};
use pyo3::prelude::*;

#[pyclass]
#[derive(Debug)]
pub struct Motif {
    #[pyo3(get)]
    a: usize,
    #[pyo3(get)]
    b: usize,
    #[pyo3(get)]
    distance: f64
}

#[pymethods]
impl Motif {
    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self))
    }
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self))
    }
}

impl From<attimo::motifs::Motif> for Motif {
    fn from(m: attimo::motifs::Motif) -> Self {
        Self { a: m.idx_a, b: m.idx_b, distance: m.distance }
    }
}

#[pyclass]
pub struct MotifIterator {
    inner: attimo::motifs::MotifIterator
}

#[pymethods]
impl MotifIterator {
    #[new]
    #[args(
        max_topk = "1000",
        repetitions = "200",
        delta = "0.01",
        seed = "1234"
    )]
    fn __new__(ts: Vec<f64>, w: usize, max_topk: usize, repetitions: usize, delta: f64, seed: u64) -> Self {
        let ts = Arc::new(WindowedTimeseries::new(ts, w, false));
        Self {
            inner: attimo::motifs::MotifIterator::new(ts, max_topk, repetitions, delta, seed)
        }
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<Motif> {
        slf.inner.next().map(|m| m.into())
    }
}

#[pymodule]
fn pyattimo(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<Motif>()?;
    m.add_class::<MotifIterator>()?;
    Ok(())
}
