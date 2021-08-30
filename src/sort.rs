use std::{fmt::Debug, ops::Range};

pub trait GetByte {
    fn num_bytes(&self) -> usize;
    fn get_byte(&self, i: usize) -> u8;
}

impl GetByte for u8 {
    fn num_bytes(&self) -> usize {
        1
    }

    fn get_byte(&self, i: usize) -> u8 {
        assert!(i == 0);
        *self
    }
}

impl GetByte for usize {
    fn num_bytes(&self) -> usize {
        std::mem::size_of::<usize>()
    }

    #[inline(always)]
    fn get_byte(&self, i: usize) -> u8 {
        (self >> (8*(std::mem::size_of::<usize>() - i - 1)) & 0xFF) as u8
    }
}

pub trait RadixSort {
    fn radix_sort(&mut self);
}

impl<T: GetByte + Debug + Ord> RadixSort for Vec<T> {
    fn radix_sort(&mut self) {
        radix_sort_impl(self, 0);
    }
}

pub fn insertion_sort<T: Ord>(arr: &mut [T]) {
    for i in 1..arr.len() {
        let mut j = i;
        while j > 0 && arr[j - 1] > arr[j] {
            arr.swap(j - 1, j);
            j -= 1;
        }
    }
}

const THRESHOLD: usize = 128;

fn radix_sort_impl<T: GetByte + Debug + Ord>(v: &mut [T], byte_index: usize) {
    if v.len() <= THRESHOLD {
        insertion_sort(v);
        return;
    }
    let nbytes = v[0].num_bytes();
    if byte_index == nbytes {
        return;
    }

    //// First, compute the histogram of byte values and build the offsets of the bucket starts
    let mut counts = [0usize; 256];
    for x in v.iter() {
        unsafe {
            *counts.get_unchecked_mut(x.get_byte(byte_index) as usize) += 1;
        }
    }

    let mut buckets_by_size = Vec::with_capacity(256);
    let mut offsets = [0usize; 256];
    let mut write_heads = [0usize; 256];
    let mut ends = [0usize; 256];
    let mut sum = 0;
    for i in 0..256 {
        unsafe {
            *offsets.get_unchecked_mut(i) = sum;
            *write_heads.get_unchecked_mut(i) = sum;
            sum += *counts.get_unchecked(i);
            ends[i] = sum;
            let span = sum - offsets.get_unchecked(i);
            if span > 0 {
                buckets_by_size.push((i, span));
            }
        }
    }

    //// Sort the buckets by decreasing size, to be able to pop from this stack in increasing order
    //// of bucket size. This sort is cheap to do, since it's always at most 256 elements.
    buckets_by_size.sort_unstable_by_key(|p| p.1);
    //// Remove the largest bucket. By construction, when all the other buckets are sorted, it
    //// must also be sorted, so we can avoid iterating over it.
    //// This optimization should help a great deal for very unbalanced byte distributions.
    buckets_by_size.pop();

    while !buckets_by_size.is_empty() {
        for &(current_bucket, _) in buckets_by_size.iter() {
            unsafe {
                let offset = *write_heads.get_unchecked(current_bucket);
                if offset < ends[current_bucket] {
                    let span = ends[current_bucket] - offset;
                    let quotient = span / 4;
                    let remainder = span % 4;
                    for q in 0..quotient {
                        let q = offset + q * 4;
                        let b1 = v.get_unchecked(q).get_byte(byte_index) as usize;
                        let b2 = v.get_unchecked(q + 1).get_byte(byte_index) as usize;
                        let b3 = v.get_unchecked(q + 2).get_byte(byte_index) as usize;
                        let b4 = v.get_unchecked(q + 3).get_byte(byte_index) as usize;

                        let t1 = *write_heads.get_unchecked(b1);
                        *write_heads.get_unchecked_mut(b1) += 1;
                        let t2 = *write_heads.get_unchecked(b2);
                        *write_heads.get_unchecked_mut(b2) += 1;
                        let t3 = *write_heads.get_unchecked(b3);
                        *write_heads.get_unchecked_mut(b3) += 1;
                        let t4 = *write_heads.get_unchecked(b4);
                        *write_heads.get_unchecked_mut(b4) += 1;

                        v.swap(q, t1);
                        v.swap(q+1, t2);
                        v.swap(q+2, t3);
                        v.swap(q+3, t4);
                    }

                    let rem_offset = offset + quotient * 4;
                    for r in 0..remainder {
                        let r = rem_offset + r;
                        let b = v.get_unchecked(r).get_byte(byte_index) as usize;
                        let t = *write_heads.get_unchecked(b);
                        v.swap(r, t);
                        *write_heads.get_unchecked_mut(b) += 1;
                    }
                }
            }
        }
        buckets_by_size.retain(|&(b, _)| write_heads[b] < ends[b]);
    }

    //// Finally, we recur into each bucket to sort it independently from the others
    for i in 0..256 {
        let r = offsets[i]..ends[i];
        if counts[i] > THRESHOLD {
            radix_sort_impl(&mut v[r], byte_index + 1);
        } else if counts[i] > 1 {
            insertion_sort(&mut v[r]);
        }
    }
}

#[test]
fn test_radix_sort_u8() {
    use rand::prelude::*;
    use rand_distr::Uniform;
    use rand_xoshiro::Xoshiro256PlusPlus;

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(12435);
    let unif = Uniform::new_inclusive(usize::MIN, usize::MAX);
    let v: Vec<usize> = unif.sample_iter(&mut rng).take(100000).collect();
    let mut expected = v.clone();
    let mut actual = v.clone();

    expected.sort_unstable();
    actual.radix_sort();
    assert_eq!(
        expected, actual,
        "expected: {:#x?}\nactual: {:#x?}",
        expected, actual
    );
}
