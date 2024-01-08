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

    // Sort the data in four more passes
    pass(data, scratch, 0, &mut b0);
    pass(scratch, data, 1, &mut b1);
    pass(data, scratch, 2, &mut b2);
    pass(scratch, data, 3, &mut b3);
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
