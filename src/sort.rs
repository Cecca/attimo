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

    let mut offsets = [0usize; 256];
    let mut write_heads = [0usize; 256];
    let mut sum = 0;
    for i in 0..256 {
        unsafe {
            *offsets.get_unchecked_mut(i) = sum;
            *write_heads.get_unchecked_mut(i) = sum;
            sum += *counts.get_unchecked(i);
        }
    }

    //// Then, start swapping values to the right position
    let mut current_bucket = 0;
    let mut i = 0;
    //// We loop on the first 255 blocks. The last one will be sorted, by construction, when all
    //// the previous ones are settled
    while current_bucket < 255 {
        //// If we move to the next bucket, we settled the previous one, and we increment the
        //// `current_bucket` index.
        unsafe {
            if i >= *offsets.get_unchecked(current_bucket + 1) {
                current_bucket += 1;
                //// Furthermore, we avoid iterating over all the elements which have already
                //// been sorted in the current bucket.
                i = *write_heads.get_unchecked(current_bucket);
                continue;
            }
        }
        let byte = unsafe { v.get_unchecked(i).get_byte(byte_index) as usize };
        //// If the current byte is in already in its place in the current bucket, then
        //// we can move on without moving it.
        if byte == current_bucket {
            i += 1;
            continue;
        }
        //// Once we get here, the byte it's not in the correct bucket, hence we swap
        //// with an element at the correct byte, and move the corresponding write head.
        unsafe {
            v.swap(i, *write_heads.get_unchecked(byte));
            *write_heads.get_unchecked_mut(byte) += 1;
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
    // let unif = Uniform::new_inclusive(0, 1000);
    let v: Vec<usize> = unif.sample_iter(&mut rng).take(10000).collect();
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
