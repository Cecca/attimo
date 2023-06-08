use rand::Rng;

use crate::{
    load::loadts,
    timeseries::{FFTData, WindowedTimeseries},
};

const K: usize = 32;
const K_HALF: usize = K / 2;

pub struct SimHash {
    dimension: usize,
    direction: Vec<f64>,
}

impl SimHash {
    pub fn new<R: Rng>(dimension: usize, rng: &mut R) -> Self {
        use rand_distr::StandardNormal;
        let direction = rng.sample_iter(StandardNormal).take(dimension).collect();
        Self {
            dimension,
            direction,
        }
    }

    pub fn hash(&self, ts: &WindowedTimeseries, fft_data: &FFTData, out: &mut [u16]) {
        ts.znormalized_sliding_dot_product_for_each(fft_data, &self.direction, |i, dotp| {
            out[i] <<= 1;
            if dotp > 0.0 {
                out[i] |= 1;
            }
        });
    }
}

fn get_minimal_index_pair(idx: usize) -> (usize, usize) {
    let sqrt = (idx as f64).sqrt() as usize;
    if idx == sqrt * sqrt + 2 * sqrt {
        (sqrt, sqrt)
    } else if idx >= sqrt * sqrt + sqrt {
        (sqrt, idx - (sqrt * sqrt + sqrt))
    } else {
        (idx - sqrt * sqrt, sqrt)
    }
}

fn interleave_bits(mut a: u16, mut b: u16) -> u32 {
    let mut r = 0u32;

    for i in 0..16 {
        r <<= 1;
        r |= (a & 1) as u32;
        a >>= 1;

        r <<= 1;
        r |= (b & 1) as u32;
        b >>= 1;
    }

    r
}

struct LSHTablesBuilder<'ts> {
    ts: &'ts WindowedTimeseries,
    dimension: usize,
    repetitions: usize,
    tensor_reps: usize,
    hashers: Vec<SimHash>,
    half_hashes: Vec<u16>,
}

impl<'ts> LSHTablesBuilder<'ts> {
    fn new<R: Rng>(
        ts: &'ts WindowedTimeseries,
        dimension: usize,
        repetitions: usize,
        rng: &mut R,
    ) -> Self {
        let n = ts.num_subsequences();
        let tensor_reps = (repetitions as f64).sqrt().ceil() as usize;
        let hashers = (0..tensor_reps * K)
            .map(|_| SimHash::new(dimension, rng))
            .collect();
        let half_hashes = vec![0u16; tensor_reps * n];
        Self {
            ts,
            dimension,
            repetitions,
            tensor_reps,
            hashers,
            half_hashes,
        }
    }

    fn build(mut self) -> LSHTables {
        let n = self.ts.num_subsequences();
        let fft_data = FFTData::new(&self.ts);

        let mut hashers = self.hashers.chunks_exact(K_HALF);
        assert!(hashers.remainder().is_empty());
        let mut half_hashes = self.half_hashes.chunks_exact_mut(n);

        // TODO do this in parallel
        for (hashes, hashers) in half_hashes.zip(hashers) {
            for h in hashers {
                h.hash(&self.ts, &fft_data, hashes);
            }
        }

        let mut tables = Vec::new();

        for r in 0..self.repetitions {
            let (l_idx, r_idx) = get_minimal_index_pair(r);
            let l_hashes = &self.half_hashes[l_idx * n..(l_idx + 1) * n];
            let r_hashes = &self.half_hashes[r_idx * n..(r_idx + 1) * n];
            // TODO loop unrolling
            let mut table = Vec::new();
            for i in 0..n {
                let h = interleave_bits(l_hashes[i], r_hashes[i]);
                table.push((h, i));
            }
            table.sort();
            tables.push(table);
        }

        LSHTables {
            dimension: self.ts.w,
            tensor_repetitions: self.tensor_reps,
            repetitions: self.repetitions,
            tables,
        }
    }
}

pub struct LSHTables {
    dimension: usize,
    repetitions: usize,
    tensor_repetitions: usize,
    tables: Vec<Vec<(u32, usize)>>,
}

impl LSHTables {
    pub fn from_ts<R: Rng>(ts: &WindowedTimeseries, repetitions: usize, rng: &mut R) -> Self {
        let builder = LSHTablesBuilder::new(ts, ts.w, repetitions, rng);
        builder.build()
    }

    /// The collision probability of a single hash function at the given dot product
    pub fn collision_probability_at(&self, dotp: f64) -> f64 {
        let theta = (dotp / self.dimension as f64).acos();
        1.0 - theta / std::f64::consts::PI
    }

    /// What would be the failure probability if iterations were independent
    pub fn independent_failure_probability(&self, dotp: f64, reps: usize, bits: usize) -> f64 {
        let p = self.collision_probability_at(dotp);

        let cur_failure = (1.0 - p.powi(bits as i32)).powi(reps as i32 + 1);
        let prev_failure =
            (1.0 - p.powi(bits as i32 + 1)).powi((self.repetitions - reps + 1) as i32);
        return cur_failure * prev_failure;
    }

    pub fn failure_probability(&self, dotp: f64, reps: usize, bits: usize) -> f64 {
        let p = self.collision_probability_at(dotp);

        // TODO: precompute all these numbers
        let cur_left_bits = (bits as f64 / 2.0).floor() as i32;
        let cur_right_bits = (bits as f64 / 2.0).ceil() as i32;
        assert_eq!(cur_left_bits + cur_right_bits, bits as i32);

        let prev_left_bits = ((bits + 1) as f64 / 2.0).floor() as i32;
        let prev_right_bits = ((bits + 1) as f64 / 2.0).ceil() as i32;
        assert_eq!(prev_left_bits + prev_right_bits, bits as i32 + 1);
        assert!(prev_left_bits >= cur_left_bits);
        assert!(prev_right_bits >= cur_right_bits);

        let up_treps = (reps as f64 + 1.0).sqrt().floor() as i32;
        let low_treps = self.tensor_repetitions as i32 - up_treps;

        // Probabilities of *not* colliding on a *single* repetition with a given number of bits
        let cur_left_fail = 1.0 - p.powi(cur_left_bits);
        let cur_right_fail = 1.0 - p.powi(cur_right_bits);

        let prev_left_fail = 1.0 - p.powi(prev_left_bits);
        let prev_right_fail = 1.0 - p.powi(prev_right_bits);

        // Probabilities of collising in *at least* on repetition
        let collide_up_left_cur = 1.0 - cur_left_fail.powi(up_treps);
        let collide_up_right_cur = 1.0 - cur_right_fail.powi(up_treps);

        let collide_up_left_prev = 1.0 - prev_left_fail.powi(up_treps);
        let collide_up_right_prev = 1.0 - prev_right_fail.powi(up_treps);

        let collide_low_left_prev = 1.0 - prev_left_fail.powi(low_treps);
        let collide_low_right_prev = 1.0 - prev_right_fail.powi(low_treps);

        (1.0 - collide_up_left_cur * collide_up_right_cur)
            * (1.0 - collide_low_left_prev * collide_low_right_prev)
            * (1.0 - collide_up_left_prev * collide_low_right_prev)
            * (1.0 - collide_low_left_prev * collide_up_right_prev)
    }
}

#[test]
fn test_failure_probability_tensor() {
    use crate::distance::zeucl;
    use std::io::Write;

    let w = 1000;
    let ts = loadts("data/ECG.csv.gz", Some(3000)).unwrap();
    let ts = WindowedTimeseries::new(ts, w, false);
    let mut rng = rand::thread_rng();
    let repetitions = 128;
    let tables = LSHTables::from_ts(&ts, repetitions, &mut rng);
    let zdist = zeucl(&ts, 0, 10);
    let dotp = (2.0 * (w as f64) - zdist.powi(2)) / 2.0;
    let p = tables.collision_probability_at(dotp);
    let mut printed = false;
    for bits in (31..=K).rev() {
        for rep in 0..repetitions {
            let fp = tables.failure_probability(dotp, rep, bits);
            let fp_independent = tables.independent_failure_probability(dotp, rep, bits);
            dbg!(fp);
            assert!(fp >= 0.0);
            assert!(fp >= fp_independent);
            if fp < 0.01 && !printed {
                eprintln!("bits {} rep {}", bits, rep);
                printed = true;
            }
        }
    }
}
