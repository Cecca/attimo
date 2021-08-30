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
    let mut sum = 0;
    for i in 0..256 {
        unsafe {
            *offsets.get_unchecked_mut(i) = sum;
            *write_heads.get_unchecked_mut(i) = sum;
            sum += *counts.get_unchecked(i);
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

    for &(current_bucket, current_bucket_span) in buckets_by_size.iter() {
        unsafe {
            let mut i = *write_heads.get_unchecked(current_bucket);
            let end = current_bucket_span + *offsets.get_unchecked(current_bucket);
            while i < end {
                let byte = v.get_unchecked(i).get_byte(byte_index) as usize;
                let target = *write_heads.get_unchecked(byte);
                //// Do the swapping. Note that we are swapping even in the case where the key is already in the
                //// correct bucket (so we are copying it in place). While this might seem wasteful, this allows
                //// us to avoid doing a branching on `byte == current_bucket`.
                v.swap(i, target);
                *write_heads.get_unchecked_mut(byte) += 1;
                //// The current read position is also the current write position in the current bucket
                i = *write_heads.get_unchecked(current_bucket);
            }
        }
    }

    //// Finally, we recur into each bucket to sort it independently from the others
    for i in 0..255 {
        let r = offsets[i]..offsets[i + 1];
        if counts[i] > THRESHOLD {
            radix_sort_impl(&mut v[r], byte_index + 1);
        } else if counts[i] > 1 {
            insertion_sort(&mut v[r]);
        }
    }
    let r = offsets[255]..v.len();
    if counts[255] > THRESHOLD {
        radix_sort_impl(&mut v[r], byte_index + 1);
    } else if counts[255] > 1 {
        insertion_sort(&mut v[r]);
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
