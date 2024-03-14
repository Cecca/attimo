use crate::{knn::Distance, timeseries::Overlaps};
use bitvec::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug)]
pub struct GraphStats {
    num_edges: usize,
    num_nodes: usize,
    max_neighborhood_size: usize,
}

/// This graph data structure maintains the edges in increasing order
pub struct Graph {
    n: usize,
    exclusion_zone: usize,
    edges: Vec<(Distance, usize, usize, bool)>,
    adjacencies: HashMap<usize, Vec<(Distance, usize)>>,
}

impl Graph {
    pub fn new(n: usize, exclusion_zone: usize) -> Self {
        Self {
            n,
            exclusion_zone,
            edges: Default::default(),
            adjacencies: Default::default(),
        }
    }

    // pub fn stats(&self) -> GraphStats {
    //     GraphStats {
    //         num_edges: self.num_edges(),
    //     }
    // }

    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// inserts an edge in the graph, if it does not exist already
    pub fn insert(&mut self, d: Distance, a: usize, b: usize) {
        assert!(a < b, "{} >= {}", a, b);
        assert!(d.is_finite());
        assert!(!a.overlaps(b, self.exclusion_zone));
        self.edges.push((d, a, b, true));
    }

    pub fn remove_larger_than(&mut self, dist: Distance) {
        if dist.is_finite() {
            let pos = match self.edges.binary_search(&(dist, 0, 0, false)) {
                Ok(p) | Err(p) => p,
            };
            let removed = self.edges.len() - pos;
            self.edges.truncate(pos);
            log::debug!("Removed {} edges larger than {}", removed, dist);
        }
    }

    pub fn reset_flags(&mut self) {
        for e in self.edges.iter_mut() {
            e.3 = false;
        }
    }

    pub fn neighborhoods(&mut self) -> NeighborhoodsIter {
        use rayon::prelude::*;
        self.edges.par_sort();
        // remove duplicates ignoring the "new" flag: since false < true the effect is that
        // edges that are duplicates because they were already inserted are not considered new
        self.edges.dedup_by_key(|tup| (tup.0, tup.1, tup.2));
        NeighborhoodsIter::from_graph(self)
    }
}

pub struct NeighborhoodsIter<'graph> {
    exclusion_zone: usize,
    edges: std::slice::Iter<'graph, (Distance, usize, usize, bool)>,
    neighborhoods: &'graph mut HashMap<usize, Vec<(Distance, usize)>>,
    updated: BitVec,
    parking: Option<(Distance, Vec<usize>)>,
    cnt_emitted: usize,
}
impl<'graph> NeighborhoodsIter<'graph> {
    fn from_graph(graph: &'graph mut Graph) -> Self {
        let neighborhoods = &mut graph.adjacencies;
        let mut updated = BitVec::new();
        updated.resize(graph.n, false);

        Self {
            exclusion_zone: graph.exclusion_zone,
            edges: graph.edges.iter(),
            neighborhoods,
            updated,
            parking: None,
            cnt_emitted: 0,
        }
    }

    /// updates the neighborhood of `src` and returns it if it was updated, otherwise
    /// it returns None.
    fn update_and_get(
        &mut self,
        new_edge: bool,
        src: usize,
        dst: usize,
        dist: Distance,
        exclusion_zone: usize,
    ) -> Option<Vec<usize>> {
        if new_edge || self.updated[src] {
            let neighborhood = self.neighborhoods.entry(src).or_insert_with(|| {
                self.updated.set(src, true);
                vec![(Distance(0.0), src)]
            });

            let cutoff = neighborhood.partition_point(|(d, _)| d <= &dist);
            neighborhood.truncate(cutoff);
            if !neighborhood
                .iter()
                .any(|(_, x)| x.overlaps(dst, exclusion_zone))
            {
                // no overlap
                neighborhood.push((dist, dst));
                self.updated.set(src, true);
                debug_assert!(neighborhood.is_sorted_by_key(|pair| pair.0));
                Some(neighborhood.iter().map(|pair| pair.1).collect())
            } else {
                None
            }
        } else {
            None
        }
    }
}

impl<'graph> Iterator for NeighborhoodsIter<'graph> {
    type Item = (Distance, Vec<usize>);
    fn next(&mut self) -> Option<Self::Item> {
        if self.parking.is_some() {
            return self.parking.take();
        }
        let exclusion_zone = self.exclusion_zone;

        while let Some(&(d, a, b, new_edge)) = self.edges.next() {
            let neighs_a = self.update_and_get(new_edge, a, b, d, exclusion_zone);
            let neighs_b = self.update_and_get(new_edge, b, a, d, exclusion_zone);

            match (neighs_a, neighs_b) {
                (Some(na), Some(nb)) => {
                    self.cnt_emitted += 2;
                    self.parking.replace((d, nb));
                    return Some((d, na));
                }
                (Some(nn), None) | (None, Some(nn)) => {
                    self.cnt_emitted += 1;
                    return Some((d, nn));
                }
                (None, None) => (), // nothing to do in this case, we go to the next iteration
            }
        }
        None
    }
}

impl<'graph> Drop for NeighborhoodsIter<'graph> {
    fn drop(&mut self) {
        log::trace!("Emitted {} neighborhoods", self.cnt_emitted);
    }
}

mod test {
    use crate::{
        distance::zeucl,
        knn::{compute_extent, Distance},
        load::loadts,
        timeseries::{FFTData, Overlaps, WindowedTimeseries},
    };
    use std::collections::{BTreeMap, HashMap};

    use super::Graph;

    #[test]
    fn test_graph() {
        let ts: Vec<f64> = loadts("data/ecg-heartbeat-av.csv", None).unwrap();
        let w = 100;
        let exclusion_zone = w / 2;
        let ts = WindowedTimeseries::new(ts, w, false);
        dbg!(ts.num_subsequences());
        dbg!(ts.num_subsequence_pairs());
        let fft_data = FFTData::new(&ts);
        let mut dists = vec![0.0f64; ts.num_subsequences()];
        let mut buf = vec![0.0f64; w];
        let mut graph = Graph::new(ts.num_subsequences(), exclusion_zone);

        for i in 0..ts.num_subsequences() {
            ts.distance_profile(&fft_data, i, &mut dists, &mut buf);
            for j in i..ts.num_subsequences() {
                if !i.overlaps(j, exclusion_zone) {
                    graph.insert(Distance(dists[j]), i, j);
                }
            }
        }
        dbg!(graph.num_edges());

        let max_k = 25;
        let mut ground = crate::motiflets::brute_force_motiflets(&ts, max_k, exclusion_zone);
        for tup in &mut ground {
            tup.1.sort();
        }

        let mut motiflets = BTreeMap::<usize, (Distance, Vec<usize>)>::new();
        for (dist, ids) in graph.neighborhoods() {
            let extent = compute_extent(&ts, &ids);
            let k = ids.len();
            if k <= max_k {
                motiflets
                    .entry(k)
                    .and_modify(|m| {
                        if extent < m.0 {
                            *m = (extent, ids.clone());
                        }
                    })
                    .or_insert_with(|| (extent, ids.clone()));
            }

            if let Some(max_extent) = motiflets.values().map(|pair| pair.0).max() {
                if max_extent < dist && motiflets.len() == max_k - 1 {
                    break;
                }
            }
        }
        let motiflets: Vec<(Distance, Vec<usize>)> = motiflets.values().cloned().collect();

        for (actual, base) in motiflets.iter().zip(&ground) {
            assert_eq!(actual.1.len(), base.1.len());
            dbg!(actual.1.len());
            dbg!(actual
                .1
                .iter()
                .zip(&base.1)
                .collect::<Vec<(&usize, &usize)>>());
            assert!(actual.0 <= base.0, "{} > {}", actual.0, base.0);
        }
        dbg!(&motiflets);
        assert_eq!(motiflets.len(), max_k - 1);
    }
}

pub struct AdjacencyGraph {
    exclusion_zone: usize,
    neighborhoods: Vec<Vec<(Distance, usize)>>,
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
        GraphStats {
            num_nodes,
            num_edges,
            max_neighborhood_size,
        }
    }

    pub fn insert(&mut self, d: Distance, a: usize, b: usize) {
        // duplicates will be handled later
        self.neighborhoods[a].push((d, b));
        self.neighborhoods[b].push((d, a));
        self.updated.set(a, true);
        self.updated.set(b, true);
    }

    pub fn reset_flags(&mut self) {
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
                    nn.dedup();
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
                let mut j = 0;
                while indices.len() < k && j < nn.len() {
                    // find the non-overlapping subsequences
                    let (jd, jj) = nn[j];
                    if !jj.overlaps(indices.as_slice(), exclusion_zone) {
                        indices.push(jj);
                        distances.push(jd);
                    }
                    j += 1;
                }
                Some((indices, distances))
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
