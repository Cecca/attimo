use crate::index::HashValue;

macro_rules! getbyte {
    ($x: expr, $b: ident) => {
        (($x >> 8 * $b) & 0xff) as usize
    };
    ($x: expr, $b: literal) => {
        (($x >> 8 * $b) & 0xff) as usize
    };
}

pub fn sort_hash_pairs(data: &mut [(HashValue, u32)], scratch: &mut [(HashValue, u32)]) {
    assert!(data.len() == scratch.len());
    const BYTES: usize = 8;

    // build histograms in a first pass over the data
    let mut b = [[0usize; 256]; BYTES];

    for (h, _) in data.iter() {
        let h: u64 = h.into();
        for i in 0..BYTES {
            b[i][getbyte!(h, i)] += 1;
        }
    }

    // set write heads
    let mut sum = [0usize; BYTES];
    for i in 0..256 {
        for j in 0..BYTES {
            let tmp = sum[j] + b[j][i];
            b[j][i] = sum[j];
            sum[j] = tmp;
        }
    }

    #[inline]
    fn pass(
        data: &mut [(HashValue, u32)],
        scratch: &mut [(HashValue, u32)],
        pass_idx: usize,
        hist: &mut [usize; 256],
    ) {
        for pair in data.iter() {
            let h: u64 = (pair.0).into();
            let b = getbyte!(h, pass_idx);
            let t = hist[b];
            hist[b] += 1;
            scratch[t] = *pair;
        }
    }

    // Sort the data in eight more passes
    // pass(data, scratch, 0, &mut b[0]);
    // pass(scratch, data, 1, &mut b[1]);
    // pass(data, scratch, 2, &mut b[2]);
    // pass(scratch, data, 3, &mut b[3]);
    // pass(data, scratch, 4, &mut b[4]);
    // pass(scratch, data, 5, &mut b[5]);
    // pass(data, scratch, 6, &mut b[6]);
    // pass(scratch, data, 7, &mut b[7]);
    for i in 0..BYTES {
        pass(data, scratch, i, &mut b[i]);
        data.swap_with_slice(scratch);
    }
}

fn merge<T: Ord + Copy>(in1: &[T], in2: &[T], out: &mut [T]) {
    assert_eq!(in1.len() + in2.len(), out.len());
    debug_assert!(in1.is_sorted());
    debug_assert!(in2.is_sorted());

    let n1 = in1.len();
    let n2 = in2.len();
    let mut i1 = 0;
    let mut i2 = 0;
    let mut iout = 0;
    while i1 < n1 && i2 < n2 {
        if in1[i1] < in2[i2] {
            out[iout] = in1[i1];
            i1 += 1;
        } else {
            out[iout] = in2[i2];
            i2 += 1;
        }
        iout += 1;
    }

    while i1 < n1 {
        out[iout] = in1[i1];
        i1 += 1;
        iout += 1;
    }
    while i2 < n2 {
        out[iout] = in2[i2];
        i2 += 1;
        iout += 1;
    }
    debug_assert!(out.is_sorted());
}

fn merge_chunks(
    data: &mut [(HashValue, u32)],
    scratch: &mut [(HashValue, u32)],
    chunk_size: usize,
) {
    use rayon::prelude::*;
    assert_eq!(data.len(), scratch.len());

    data.par_chunks_mut(2 * chunk_size)
        .zip(scratch.par_chunks_mut(2 * chunk_size))
        .for_each(|(data_chunk, scratch_chunk)| {
            if data_chunk.len() < chunk_size {
                // last chunk, all by itself
                scratch_chunk.copy_from_slice(data_chunk);
            } else {
                merge(
                    &data_chunk[..chunk_size],
                    &data_chunk[chunk_size..],
                    scratch_chunk,
                )
            }
        });

    data.swap_with_slice(scratch);
}

pub fn par_sort_hash_pairs(data: &mut [(HashValue, u32)], scratch: &mut [(HashValue, u32)]) {
    use rayon::prelude::*;
    assert_eq!(data.len(), scratch.len());

    let threads = rayon::current_num_threads();
    let chunk_size = (data.len() as f64 / threads as f64).ceil() as usize;

    data.par_chunks_mut(chunk_size)
        .zip(scratch.par_chunks_mut(chunk_size))
        .for_each(|(data_chunk, scratch_chunk)| sort_hash_pairs(data_chunk, scratch_chunk));

    let mut chunk_size = chunk_size;
    while chunk_size < data.len() {
        merge_chunks(data, scratch, chunk_size);
        chunk_size *= 2;
    }
    debug_assert!(data.is_sorted());
}

#[test]
fn test_radix_sort_hash_pairs() {
    use rand::prelude::*;
    use rand_distr::Uniform;
    use rand_xoshiro::Xoshiro256PlusPlus;

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(12435);
    let unif = Uniform::new_inclusive(u64::MIN, u64::MAX);
    let v: Vec<(HashValue, u32)> = unif
        .sample_iter(&mut rng)
        .take(100000)
        .enumerate()
        // .map(|(i, h)| (HashValue(h), i as u32))
        .map(|(i, h)| (h.into(), i as u32))
        .collect();
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

#[test]
fn test_par_radix_sort_hash_pairs() {
    use rand::prelude::*;
    use rand_distr::Uniform;
    use rand_xoshiro::Xoshiro256PlusPlus;

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(12435);
    let unif = Uniform::new_inclusive(u64::MIN, u64::MAX);
    for n in [16, 17, 200, 1000, 100000] {
        let v: Vec<(HashValue, u32)> = unif
            .sample_iter(&mut rng)
            .take(n)
            .enumerate()
            .map(|(i, h)| (h.into(), i as u32))
            .collect();
        let mut expected = v.clone();
        let mut actual = v.clone();
        let mut scratch: Vec<(HashValue, u32)> = Vec::new();
        scratch.resize_with(v.len(), Default::default);

        expected.sort_unstable();
        par_sort_hash_pairs(&mut actual, &mut scratch);
        assert_eq!(
            expected, actual,
            "expected: {:#x?}\nactual: {:#x?}",
            expected, actual
        );
    }
}

#[test]
fn test_merge() {
    let in1 = [1, 3, 5, 7, 9];
    let in2 = [2, 4, 6, 8, 10];
    let mut out = [0; 10];

    merge(&in1, &in2, &mut out);
    assert_eq!(out, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
}
