use std::fmt::Debug;

use crate::lsh::HashValue;

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
        (self >> (8 * (std::mem::size_of::<usize>() - i - 1)) & 0xFF) as u8
    }
}

impl<T1: GetByte, T2: GetByte> GetByte for (T1, T2) {
    fn num_bytes(&self) -> usize {
        self.0.num_bytes() + self.1.num_bytes()
    }

    #[inline(always)]
    fn get_byte(&self, i: usize) -> u8 {
        if i < self.0.num_bytes() {
            self.0.get_byte(i)
        } else {
            self.1.get_byte(i)
        }
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

const TINY_THRESHOLD: usize = 16;
const SMALL_THRESHOLD: usize = 1024;
const RECURSION_THRESHOLD: usize = 16;

fn radix_sort_impl<T: GetByte + Debug + Ord>(v: &mut [T], byte_index: usize) {
    // use std::time::Instant;
    // let start = Instant::now();
    if v.len() <= TINY_THRESHOLD {
        // println!(
        //     "{}sorting {} elements with insertion sort",
        //     "  ".repeat(byte_index),
        //     v.len()
        // );
        insertion_sort(v);
        return;
    }
    if v.len() <= SMALL_THRESHOLD {
        v.sort_unstable();
        return;
    }
    let nbytes = v[0].num_bytes();
    if byte_index == nbytes {
        return;
    }

    if byte_index >= RECURSION_THRESHOLD {
        // println!("{}sorting {} elements with PDQ", "  ".repeat(byte_index), v.len());
        v.sort_unstable();
        return;
    }
    // println!("{}sorting {} elements", "  ".repeat(byte_index), v.len());

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

    //// Sort the buckets by decreasing size.
    //// This sort is cheap to do, since it's always at most 256 elements.
    //// Benchmarks revealed that it's cheaper to first fix the second largest bucket,
    //// and the move to smaller buckets.
    buckets_by_size.sort_unstable_by(|p1, p2| p1.1.cmp(&p2.1).reverse());
    //// Remove the largest bucket. By construction, when all the other buckets are sorted, it
    //// must also be sorted, so we can avoid iterating over it.
    buckets_by_size.remove(0);

    // let mut cnt_swaps = 0;
    while !buckets_by_size.is_empty() {
        for &(current_bucket, _) in buckets_by_size.iter() {
            unsafe {
                loop {
                    let offset = *write_heads.get_unchecked(current_bucket);
                    if offset + 4 < ends[current_bucket] {
                        let q = offset;
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
                        v.swap(q + 1, t2);
                        v.swap(q + 2, t3);
                        v.swap(q + 3, t4);
                    } else if offset < ends[current_bucket] {
                        let r = offset;
                        let b = v.get_unchecked(r).get_byte(byte_index) as usize;
                        let t = *write_heads.get_unchecked(b);
                        v.swap(r, t);
                        *write_heads.get_unchecked_mut(b) += 1;
                    } else {
                        break;
                    }
                }
            }
        }
        buckets_by_size.retain(|&(b, _)| write_heads[b] < ends[b]);
    }

    //// Finally, we recur into each bucket to sort it independently from the others
    // let mut rec = 0;
    // let mut insert_cnt = 0;
    // let mut dur_rec = std::time::Duration::from_secs(0);
    // let mut dur_insert = std::time::Duration::from_secs(0);
    for i in 0..256 {
        let r = offsets[i]..ends[i];
        if counts[i] > SMALL_THRESHOLD {
            // let s = Instant::now();
            radix_sort_impl(&mut v[r], byte_index + 1);
            // dur_rec += s.elapsed();
        } else if counts[i] > TINY_THRESHOLD {
            v[r].sort_unstable();
        } else if counts[i] > 1 {
            // println!(
            //     "{}sorting {} elements with insertion sort",
            //     "  ".repeat(byte_index),
            //     v.len()
            // );
            // let s = Instant::now();
            insertion_sort(&mut v[r]);
            // dur_insert += s.elapsed();
        }
    }
    // println!("{}completed in {:?}, byte {}, {} recursive calls ({:?}), {} insertion sorts ({:?}), {} elements, {} swaps, {} swaps/elem",
    //     "  ".repeat(byte_index),
    //     start.elapsed(),
    //     byte_index,
    //     rec,
    //     dur_rec,
    //     insert_cnt,
    //     dur_insert,
    //     v.len(),
    //     cnt_swaps,
    //     cnt_swaps as f32 / v.len() as f32
    // );
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

macro_rules! getbyte {
    ($x: expr, $b: literal) => {
        (($x >> 8*$b) & 0xff) as usize
    };
}

pub fn sort_hash_pairs(
    data: &mut [(HashValue, u32)],
    scratch: &mut [(HashValue, u32)]
) {
    assert!(data.len() == scratch.len());

    // build histograms in a first pass over the data
    let mut b0 = [0usize; 256];
    let mut b1 = [0usize; 256];
    let mut b2 = [0usize; 256];
    let mut b3 = [0usize; 256];

    for (h, _) in data.iter() {
        let h = h.0;
        b0[getbyte!(h, 0)] += 1;
        b1[getbyte!(h, 1)] += 1;
        b2[getbyte!(h, 2)] += 1;
        b3[getbyte!(h, 3)] += 1;
    }

    // set write heads
    let mut sum0 = 0;
    let mut sum1 = 0;
    let mut sum2 = 0;
    let mut sum3 = 0;
    for i in 0..256 {
        let mut tmp = sum0 + b0[i];
        b0[i] = sum0;
        sum0 = tmp;
        
        tmp = sum1 + b1[i];
        b1[i] = sum1;
        sum1 = tmp;

        tmp = sum2 + b2[i];
        b2[i] = sum2;
        sum2 = tmp;

        tmp = sum3 + b3[i];
        b3[i] = sum3;
        sum3 = tmp;
    }

    // Sort the data in four more passes
    for pair in data.iter() {
        let b = getbyte!((pair.0).0, 0);
        let t = b0[b];
        b0[b] += 1;
        scratch[t] = *pair;
    }
    for pair in scratch.iter() {
        let b = getbyte!((pair.0).0, 1);
        let t = b1[b];
        b1[b] += 1;
        data[t] = *pair;
    }
    for pair in data.iter() {
        let b = getbyte!((pair.0).0, 2);
        let t = b2[b];
        b2[b] += 1;
        scratch[t] = *pair;
    }
    for pair in scratch.iter() {
        let b = getbyte!((pair.0).0, 3);
        let t = b3[b];
        b3[b] += 1;
        data[t] = *pair;
    }
}

#[test]
fn test_radix_sort_hash_pairs() {
    use rand::prelude::*;
    use rand_distr::Uniform;
    use rand_xoshiro::Xoshiro256PlusPlus;

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(12435);
    let unif = Uniform::new_inclusive(u32::MIN, u32::MAX);
    let v: Vec<(HashValue, u32)> = unif.sample_iter(&mut rng).take(100000).enumerate().map(|(i, h)| (HashValue(h), i as u32)).collect();
    let mut expected = v.clone();
    let mut actual = v.clone();
    let mut scratch: Vec<(HashValue, u32)> = Vec::new();
    scratch.resize_with(v.len(), Default::default);

    expected.sort_unstable();
    sort_hash_pairs(&mut actual, &mut scratch);
    assert_eq!(
        expected, actual,
        "expected: {:#x?}\nactual: {:#x?}",
        expected, actual
    );
}

