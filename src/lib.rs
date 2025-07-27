#![feature(allocator_api)]
#![feature(debug_closure_helpers)]
#![feature(formatting_options)]
#![feature(string_remove_matches)]
#![feature(portable_simd)]
#![feature(int_roundings)]

pub mod allocator;
pub mod distance;
pub mod graph;
pub mod index;
pub mod knn;
pub mod load;
pub mod motiflets;
pub mod motifs;
#[macro_use]
pub mod observe;
#[cfg(feature = "pprof")]
pub mod profiler;
pub mod sort;
mod stats;
/// Implements an LSH-based index for a time series.
pub mod timeseries;
