use crate::{allocator::Bytes, knn::Distance, timeseries::Overlaps};
use bitvec::prelude::*;
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, Default)]
pub struct GraphStats {
    pub num_edges: usize,
    pub num_nodes: usize,
    pub max_neighborhood_size: usize,
    pub used_memory: Bytes,
}

#[derive(Default, Clone, Copy)]
struct DistanceWithFlag(f64);
impl DistanceWithFlag {
    fn distance(&self) -> Distance {
        Distance(self.0.abs())
    }
    fn flag(&self) -> bool {
        self.0.is_sign_positive()
    }
    fn set_flag(&mut self, flag: bool) {
        let flip_sign = flag != self.flag();
        if flip_sign {
            self.0 = -self.0;
        }
    }
}
impl From<Distance> for DistanceWithFlag {
    fn from(value: Distance) -> Self {
        Self(value.0)
    }
}
impl Eq for DistanceWithFlag {}
impl PartialEq for DistanceWithFlag {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Ord for DistanceWithFlag {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance()
            .cmp(&other.distance())
            .then(self.flag().cmp(&other.flag()))
    }
}
impl PartialOrd for DistanceWithFlag {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

pub struct AdjacencyGraph {
    exclusion_zone: usize,
    neighborhoods: Vec<Vec<(DistanceWithFlag, usize)>>,
    updated: BitVec,
}

impl AdjacencyGraph {
    pub fn new(n: usize, exclusion_zone: usize) -> Self {
        let mut updated = BitVec::new();
        updated.resize(n, false);
        Self {
            exclusion_zone,
            neighborhoods: vec![Default::default(); n],
            updated,
        }
    }

    pub fn stats(&self) -> GraphStats {
        let num_nodes = self
            .neighborhoods
            .iter()
            .filter(|nn| !nn.is_empty())
            .count();
        let num_edges = self.neighborhoods.iter().map(|nn| nn.len()).sum::<usize>();
        let max_neighborhood_size = self.neighborhoods.iter().map(|nn| nn.len()).max().unwrap();
        let entrysize = std::mem::size_of::<Distance>() + std::mem::size_of::<usize>();
        let used_memory = self
            .neighborhoods
            .iter()
            .map(|nn| nn.len() * entrysize)
            .sum::<usize>();
        let used_memory = Bytes(used_memory);

        GraphStats {
            num_nodes,
            num_edges,
            max_neighborhood_size,
            used_memory,
        }
    }

    pub fn insert(&mut self, d: Distance, a: usize, b: usize) {
        // duplicates will be handled later
        self.neighborhoods[a].push((d.into(), b));
        self.neighborhoods[b].push((d.into(), a));
        self.updated.set(a, true);
        self.updated.set(b, true);
    }

    pub fn has_edge(&self, a: usize, b: usize) -> bool {
        self.neighborhoods[a].iter().any(|(_, x)| *x == b)
    }

    pub fn reset_flags(&mut self) {
        self.updated.fill(false);
        self.neighborhoods.par_iter_mut().for_each(|nn| {
            for x in nn.iter_mut() {
                x.0.set_flag(false);
            }
        });
    }

    fn remove_duplicates(&mut self) {
        let updated = &self.updated;
        self.neighborhoods
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, nn)| {
                if updated[i] {
                    nn.sort();
                    nn.dedup_by_key(|pair| pair.1);
                }
            });
    }

    pub fn neighborhoods(
        &mut self,
        k: usize,
    ) -> impl Iterator<Item = (Vec<usize>, Vec<Distance>)> + '_ {
        self.remove_duplicates();
        let exclusion_zone = self.exclusion_zone;
        let updated = &self.updated;
        self.neighborhoods
            .iter()
            .enumerate()
            .filter_map(move |(i, nn)| {
                if !updated[i] {
                    return None;
                }
                let mut indices = Vec::new();
                let mut distances = Vec::new();
                indices.push(i);
                distances.push(Distance(0.0));
                let mut emit = false;
                let mut j = 0;
                while indices.len() < k && j < nn.len() {
                    // find the non-overlapping subsequences
                    let (jd, jj) = nn[j];
                    if !jj.overlaps(indices.as_slice(), exclusion_zone) {
                        indices.push(jj);
                        distances.push(jd.distance());
                        emit |= jd.flag(); // collect if there is at least one updated edge
                    }
                    j += 1;
                }
                if emit {
                    Some((indices, distances))
                } else {
                    None
                }
            })
    }
}

/// check that the deduplication by partial keys keeps the first entry
#[test]
fn dedup_with_flags() {
    assert!(false < true);
    let mut edges = vec![
        (Distance(0.3), 0, 2, true),
        (Distance(0.3), 0, 2, false),
        (Distance(0.5), 1, 2, true),
    ];

    let expected = vec![(Distance(0.3), 0, 2, false), (Distance(0.5), 1, 2, true)];

    edges.sort();
    edges.dedup_by_key(|tup| (tup.0, tup.1, tup.2));
    assert_eq!(edges, expected);
}
