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
        self.to_le_bytes()[std::mem::size_of::<usize>() - i - 1]
    }
}

pub trait RadixSort {
    fn radix_sort(&mut self);
}

impl<T: GetByte + Debug + Ord> RadixSort for Vec<T> {
    // fn radix_sort(&mut self) {
    //     let nbytes = self[0].num_bytes();
    //     let mut stash: Vec<Vec<T>> = Vec::new();
    //     let mut bufs = Vec::new();
    //     bufs.resize_with(256, || Vec::with_capacity(1024));
    //     let mut output = Vec::new();
    //     output.resize_with(256, Vec::new);

    //     // run the first one, which also allocates the small buffers on the stash
    //     for x in self.drain(..) {
    //         let byte = x.get_byte(0) as usize;
    //         bufs[byte].push(x);
    //         if bufs[byte].len() == 1024 {
    //             let free = stash.pop().unwrap_or_else(|| Vec::with_capacity(1024));
    //             let full = std::mem::replace(&mut bufs[byte], free);
    //             output[byte].push(full);
    //         }
    //     }
    //     for (byte, buf) in bufs.iter_mut().enumerate() {
    //         let free = stash.pop().unwrap_or_else(|| Vec::with_capacity(1024));
    //         let full = std::mem::replace(buf, free);
    //         output[byte].push(full);
    //     }

    //     for byte_idx in 0..nbytes {
    //         let mut input = Vec::new();
    //         input.resize_with(256, Vec::new);
    //         std::mem::swap(&mut input, &mut output);

    //         for mut list in input.into_iter().flatten() {
    //             for x in list.drain(..) {
    //                 let byte = x.get_byte(byte_idx) as usize;
    //                 bufs[byte].push(x);
    //                 if bufs[byte].len() == 1024 {
    //                     let free = stash.pop().unwrap_or_else(|| Vec::with_capacity(1024));
    //                     let full = std::mem::replace(&mut bufs[byte], free);
    //                     output[byte].push(full);
    //                 }
    //             }
    //             stash.push(list);
    //         }
    //         for (byte, buf) in bufs.iter_mut().enumerate() {
    //             let free = stash.pop().unwrap_or_else(|| Vec::with_capacity(1024));
    //             let full = std::mem::replace(buf, free);
    //             output[byte].push(full);
    //         }
    //     }
    //
    //     // Move the output into the input vector
    //     for x in output.into_iter().flatten().flatten() {
    //         self.push(x);
    //     }
    // }

    // fn radix_sort(&mut self) {
    //     let n = self.len();
    //     let nbytes = self[0].num_bytes();
    //     let mut counts = [0usize; 256];
    //     let mut tmp = Vec::with_capacity(n);

    //     for byte_idx in 0..nbytes {
    //         counts.fill(0);
    //         assert!(tmp.is_empty());
    //         // Count the occurrences of the byte, and move data to the temporary vector
    //         for x in self.drain(..) {
    //             counts[x.get_byte(byte_idx) as usize] += 1;
    //             tmp.push(x);
    //         }
    //         // Do the cumulative sum
    //         for i in 1..counts.len() {
    //             counts[i] += counts[i - 1];
    //         }

    //         unsafe { self.set_len(n) };
    //         for x in tmp.drain(..).rev() {
    //             let byte = x.get_byte(byte_idx) as usize;
    //             self[counts[byte] - 1] = x;
    //             counts[byte] -= 1;
    //         }
    //     }
    // }

    fn radix_sort(&mut self) {
        // let n = self.len();
        // let nbytes = self[0].num_bytes();
        // let mut counts = [0usize; 256];
        // let mut tmp = Vec::with_capacity(n);
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

fn radix_sort_impl<T: GetByte + Debug + Ord>(v: &mut [T], byte_index: usize) {
    if v.len() <= 32 {
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
        if counts[i] > 0 {
            let r = offsets[i]..offsets[i + 1];
            radix_sort_impl(&mut v[r], byte_index + 1);
        }
    }
    if counts[255] > 0 {
        let r = offsets[255]..v.len();
        radix_sort_impl(&mut v[r], byte_index + 1);
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
