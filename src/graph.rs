use crate::{
    allocator::{ByteSize, Bytes},
    knn::Distance,
    observe::observe,
    timeseries::Overlaps,
};
use bitvec::prelude::*;
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, Default)]
pub struct GraphStats {
    pub num_edges: usize,
    pub num_nodes: usize,
    pub max_neighborhood_size: usize,
    pub used_memory: Bytes,
}

impl GraphStats {
    #[rustfmt::skip]
    pub fn observe(&self, repetition: usize, prefix: usize) {
        observe!(repetition, prefix, "num_edges", self.num_edges);
        observe!(repetition, prefix, "num_nodes", self.num_nodes);
        observe!(repetition, prefix, "max_neighborhood_size", self.max_neighborhood_size);
        observe!(repetition, prefix, "used_memory", self.used_memory.0);
    }
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

impl ByteSize for DistanceWithFlag {
    fn byte_size(&self) -> Bytes {
        Bytes(std::mem::size_of_val(self))
    }

    fn mem_tree_fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.byte_size())
    }
}

impl ByteSize for BitVec {
    fn byte_size(&self) -> Bytes {
        Bytes(self.len() / 8)
    }

    fn mem_tree_fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.byte_size())
    }
}

impl ByteSize for AdjacencyGraph {
    fn byte_size(&self) -> Bytes {
        self.neighborhoods.byte_size() + self.updated.byte_size()
    }

    fn mem_tree_fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct(&format!("AdjacencyGraph({})", self.byte_size()))
            .field_with("neighborhoods", |f| {
                write!(f, "{}", self.neighborhoods.byte_size())
            })
            .field_with("updated", |f| write!(f, "{}", self.updated.byte_size()))
            .finish()
    }
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

    pub fn reset_updated(&mut self) {
        self.updated.fill(false);
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
            .iter_mut()
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
                    let (jd, jj) = &mut nn[j];
                    if !jj.overlaps(indices.as_slice(), exclusion_zone) {
                        indices.push(*jj);
                        distances.push(jd.distance());
                        emit |= jd.flag(); // collect if there is at least one updated edge
                        jd.set_flag(false);
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
