use std::{fmt::Debug, mem::MaybeUninit};

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

    fn get_byte(&self, i: usize) -> u8 {
        self.to_le_bytes()[i]
    }
}

pub trait RadixSort {
    fn radix_sort(&mut self);
}

impl<T: GetByte + Debug> RadixSort for Vec<T> {
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

    fn radix_sort(&mut self) {
        let n = self.len();
        let nbytes = self[0].num_bytes();
        let mut counts = [0usize; 256];
        let mut tmp = Vec::with_capacity(n);

        for byte_idx in 0..nbytes {
            counts.fill(0);
            assert!(tmp.is_empty());
            // Count the occurrences of the byte, and move data to the temporary vector
            for x in self.drain(..) {
                counts[x.get_byte(byte_idx) as usize] += 1;
                tmp.push(x);
            }
            // Do the cumulative sum
            for i in 1..counts.len() {
                counts[i] += counts[i - 1];
            }

            unsafe {self.set_len(n)};
            for x in tmp.drain(..).rev() {
                let byte = x.get_byte(byte_idx) as usize;
                self[counts[byte] - 1] = x;
                counts[byte] -= 1;
            }
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
    assert_eq!(expected, actual);
}
