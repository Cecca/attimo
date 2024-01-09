use crate::lsh::HashValue;

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
        let h = h.0;
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
            let b = getbyte!((pair.0).0, pass_idx);
            let t = hist[b];
            hist[b] += 1;
            scratch[t] = *pair;
        }
    }

    // Sort the data in eight more passes
    pass(data, scratch, 0, &mut b[0]);
    pass(scratch, data, 1, &mut b[1]);
    pass(data, scratch, 2, &mut b[2]);
    pass(scratch, data, 3, &mut b[3]);
    pass(data, scratch, 4, &mut b[4]);
    pass(scratch, data, 5, &mut b[5]);
    pass(data, scratch, 6, &mut b[6]);
    pass(scratch, data, 7, &mut b[7]);
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
        .map(|(i, h)| (HashValue(h), i as u32))
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
