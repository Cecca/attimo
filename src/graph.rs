use std::collections::{BTreeMap, BTreeSet, HashMap};

use crate::{knn::Distance, timeseries::Overlaps};

#[derive(Clone, Copy, Debug)]
pub struct GraphStats {
    num_edges: usize,
}

/// This graph data structure maintains the edges in increasing order
pub struct Graph {
    exclusion_zone: usize,
    edges: Vec<(Distance, usize, usize)>,
}

impl Graph {
    pub fn new(exclusion_zone: usize) -> Self {
        Self {
            exclusion_zone,
            edges: Default::default(),
        }
    }

    pub fn stats(&self) -> GraphStats {
        GraphStats {
            num_edges: self.num_edges(),
        }
    }

    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// inserts an edge in the graph, if it does not exist already
    pub fn insert(&mut self, d: Distance, a: usize, b: usize) {
        assert!(a < b, "{} >= {}", a, b);
        assert!(d.is_finite());
        assert!(!a.overlaps(b, self.exclusion_zone));
        self.edges.push((d, a, b));
    }

    pub fn remove_larger_than(&mut self, dist: Distance) {
        if dist.is_finite() {
            let pos = match self.edges.binary_search(&(dist, 0, 0)) {
                Ok(p) | Err(p) => p,
            };
            let removed = self.edges.len() - pos;
            self.edges.truncate(pos);
            log::debug!("Removed {} edges larger than {}", removed, dist);
        }
    }

    pub fn neighborhoods(&mut self) -> NeighborhoodsIter {
        self.edges.sort_unstable();
        self.edges.dedup();
        NeighborhoodsIter::from_graph(self)
    }
}

pub struct NeighborhoodsIter<'graph> {
    exclusion_zone: usize,
    edges: std::slice::Iter<'graph, (Distance, usize, usize)>,
    neighborhoods: HashMap<usize, BTreeSet<usize>>,
    parking: Option<(Distance, Vec<usize>)>,
}
impl<'graph> NeighborhoodsIter<'graph> {
    fn from_graph(graph: &'graph Graph) -> Self {
        Self {
            exclusion_zone: graph.exclusion_zone,
            edges: graph.edges.iter(),
            neighborhoods: HashMap::new(),
            parking: None,
        }
    }

    /// updates the neighborhood of `src` and returns it if it was updated, otherwise
    /// it returns None.
    fn update_and_get(
        &mut self,
        src: usize,
        dst: usize,
        exclusion_zone: usize,
    ) -> Option<Vec<usize>> {
        let start = if dst < exclusion_zone {
            0
        } else {
            dst - exclusion_zone
        };
        let end = dst + exclusion_zone;
        let neighborhood = self.neighborhoods.entry(src).or_insert_with(|| {
            let mut set = BTreeSet::new();
            set.insert(src);
            set
        });
        if neighborhood.range(start..=end).next().is_none() {
            // no overlap
            neighborhood.insert(dst);
            Some(neighborhood.iter().copied().collect())
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

        while let Some(&(d, a, b)) = self.edges.next() {
            let neighs_a = self.update_and_get(a, b, exclusion_zone);
            let neighs_b = self.update_and_get(b, a, exclusion_zone);

            match (neighs_a, neighs_b) {
                (Some(na), Some(nb)) => {
                    self.parking.replace((d, nb));
                    return Some((d, na));
                }
                (Some(nn), None) | (None, Some(nn)) => {
                    return Some((d, nn));
                }
                (None, None) => (), // nothing to do in this case, we go to the next iteration
            }
        }
        None
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
        let mut graph = Graph::new(exclusion_zone);

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
