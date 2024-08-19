#![feature(allocator_api)]
#![feature(is_sorted)]
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
pub mod observe;
pub mod sort;
mod stats;
/// Implements an LSH-based index for a time series.
pub mod timeseries;
