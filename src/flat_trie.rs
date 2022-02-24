use std::{cmp::Ordering, ops::Range};

use crate::lsh::K;

/// A trie leveraging the property that all the strings it contains have the
/// same length, thus avoiding pointers and using only flat arrays to retrieve
/// the groups sharing the same prefix
pub struct FlatTrie {
    /// the length of the common prefix of each element and the previous one
    common: Vec<u8>,
    /// the indices of the elements (which should be stored elsewhere)
    indices: Vec<u32>,
}

impl FlatTrie {
    pub fn new<K: LexiCmp>(kv_pairs: &mut [(K, u32)]) -> Self {
        // arrange the keys lexicographically
        kv_pairs.sort_by(|a, b| a.0.lexi_cmp(&b.0));
        let mut common = Vec::with_capacity(kv_pairs.len());
        let mut indices = Vec::with_capacity(kv_pairs.len());

        // push the first element, which has no preceding with which to compare the prefix
        common.push(0u8);
        indices.push(kv_pairs[0].1);
        let mut prev = &kv_pairs[0].0;

        for (k, v) in &kv_pairs[1..] {
            let comm = k.common_prefix_len(prev);
            common.push(comm);
            indices.push(*v);
            prev = k;
        }

        Self {
            common,
            indices,
        }
    }

    /// trieve the groups sharing a prefix of the given length, returning the
    /// range of indices into the `self.indices` array
    pub fn groups_at(&self, prefix_length: u8, output: &mut Vec<Range<usize>>) {
        output.clear();
        let mut s = 0;
        let mut e = s + 1;
        loop {
            if e == self.common.len() || self.common[e] < prefix_length {
                output.push(s .. e);
                if e >= self.common.len() {
                    return;
                }
                s = e;
                e = s + 1;
            } else {
                e += 1;
            }
        }
    }

    /// return the slice of indices in the given range
    pub fn slice(&self, range: Range<usize>) -> &[u32] {
        &self.indices[range]
    }
}

pub trait LexiCmp {
    fn key_len(&self) -> usize;
    fn lexi_cmp(&self, other: &Self) -> Ordering;
    fn common_prefix_len(&self, other: &Self) -> u8;
}

impl<O: Ord> LexiCmp for [O; K] {
    fn key_len(&self) -> usize {
        self.len()
    }
    fn lexi_cmp(&self, other: &Self) -> Ordering {
        for (s, o) in self.iter().zip(other.iter()) {
            if s < o {
                return Ordering::Less;
            } else if s > o {
                return Ordering::Greater;
            }
        }
        Ordering::Equal
    }
    fn common_prefix_len(&self, other: &Self) -> u8 {
        self.iter()
            .zip(other.iter())
            .take_while(|(s, o)| s == o)
            .count() as u8
    }
}

impl<O: Ord> LexiCmp for &[O] {
    fn key_len(&self) -> usize {
        self.len()
    }
    fn lexi_cmp(&self, other: &Self) -> Ordering {
        for (s, o) in self.iter().zip(other.iter()) {
            if s < o {
                return Ordering::Less;
            } else if s > o {
                return Ordering::Greater;
            }
        }
        Ordering::Equal
    }
    fn common_prefix_len(&self, other: &Self) -> u8 {
        self.iter()
            .zip(other.iter())
            .take_while(|(s, o)| s == o)
            .count() as u8
    }
}

#[cfg(test)]
mod test {
    use super::FlatTrie;

    #[test]
    fn test_prefix_tree() {
        let mut data = [
            ("abkd".as_bytes(), 10),
            ("abad".as_bytes(), 0),
            ("abke".as_bytes(), 30),
            ("bbsd".as_bytes(), 40),
            ("abkd".as_bytes(), 20),
        ];
        let trie = FlatTrie::new(&mut data);
        let mut out = Vec::new();
        trie.groups_at(2, &mut out);
        assert_eq!(out, [0..4, 4..5]);

        trie.groups_at(3, &mut out);
        assert_eq!(out, [0..1, 1..4, 4..5]);

        assert_eq!(trie.slice(out[1].clone()), [10,20,30]);
    }
}
