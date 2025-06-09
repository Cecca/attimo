use crate::{
    distance::zeucl,
    timeseries::{Overlaps, WindowedTimeseries},
};

#[derive(PartialEq, PartialOrd, Clone, Copy)]
pub struct Distance(pub f64);
impl Eq for Distance {}
impl Ord for Distance {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap()
    }
}

impl Default for Distance {
    fn default() -> Self {
        Self(0.0)
    }
}

impl Distance {
    pub fn infinity() -> Self {
        Self(f64::INFINITY)
    }
    pub fn is_finite(&self) -> bool {
        self.0.is_finite()
    }
    pub fn is_infinite(&self) -> bool {
        self.0.is_infinite()
    }
}

impl From<f64> for Distance {
    fn from(value: f64) -> Self {
        Self(value)
    }
}
impl Into<f64> for Distance {
    fn into(self) -> f64 {
        self.0
    }
}

impl std::fmt::Debug for Distance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::fmt::Display for Distance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Overlaps<(Distance, usize)> for (Distance, usize) {
    fn overlaps(&self, other: (Distance, usize), exclusion_zone: usize) -> bool {
        self.1.overlaps(other.1, exclusion_zone)
    }
}

/// A triple where the third element denotes whether the corresponding tuple is "active"
/// overlaps with another one only if the other one is active and their indices overlap
impl Overlaps<(Distance, usize, bool)> for (Distance, usize, bool) {
    fn overlaps(&self, other: (Distance, usize, bool), exclusion_zone: usize) -> bool {
        other.2 && self.1.overlaps(other.1, exclusion_zone)
    }
}

pub fn compute_extents(ts: &WindowedTimeseries, indices: &[usize]) -> Vec<Distance> {
    let k = indices.len();
    let mut extents = vec![Distance(0.0f64); indices.len()];
    for i in 1..k {
        extents[i] = extents[i - 1];
        for j in 0..i {
            let ii = indices[i];
            let jj = indices[j];
            assert!(!ts.is_flat(ii));
            assert!(!ts.is_flat(jj));
            let d = Distance(zeucl(ts, ii, jj));
            assert!(
                !d.0.is_nan(),
                "distance between {} and {} is NaN (stds: {} and {}, means {} and {})",
                ii,
                jj,
                ts.sd(ii),
                ts.sd(jj),
                ts.mean(ii),
                ts.mean(jj)
            );
            extents[i] = extents[i].max(d);
        }
    }

    extents
}
