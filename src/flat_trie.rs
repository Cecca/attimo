use std::{cmp::Ordering, ops::Range};

use rayon::prelude::*;
use crate::lsh::{K, HashCollection};

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
    pub fn new(n: u32, hash_coll: &HashCollection, rep: usize) -> Self {
        let mut common = vec![0u8; n as usize];
        let mut indices: Vec<u32> = (0..n).collect();

        hash_coll.lexi_sort(rep, &mut indices);

        for i in 1..n as usize {
            common[i] = hash_coll.common_prefix(indices[i] as usize, indices[i-1] as usize, rep) as u8;
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
    // fn lexi_cmp(&self, other: &Self) -> Ordering {
    //     for (s, o) in self.iter().zip(other.iter()) {
    //         if s != o {
    //             if s < o {
    //                 return Ordering::Less;
    //             } else if s > o {
    //                 return Ordering::Greater;
    //             }
    //         }
    //     }
    //     Ordering::Equal
    // }
    fn lexi_cmp(&self, other: &Self) -> Ordering {
        let mut i = 0;
        while i < K {
            let s0 = &self[i];
            let s1 = &self[i+1];
            let s2 = &self[i+2];
            let s3 = &self[i+3];

            let o0 = &other[i];
            let o1 = &other[i+1];
            let o2 = &other[i+2];
            let o3 = &other[i+3];

            if s0 < o0 { return Ordering::Less; }
            if s0 > o0 { return Ordering::Greater; }

            if s1 < o1 { return Ordering::Less; }
            if s1 > o1 { return Ordering::Greater; }

            if s2 < o2 { return Ordering::Less; }
            if s2 > o2 { return Ordering::Greater; }

            if s3 < o3 { return Ordering::Less; }
            if s3 > o3 { return Ordering::Greater; }
            
            i += 4;
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
