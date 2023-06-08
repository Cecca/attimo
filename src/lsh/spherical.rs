use crate::timeseries::{FFTData, WindowedTimeseries};
use rand::Rng;
use std::time::Instant;

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

fn interleave_zeros(input: u16) -> u32 {
    let mut res = input as u32;

    res = (res ^ (res << 8)) & 0x00ff00ff;
    res = (res ^ (res << 4)) & 0x0f0f0f0f;
    res = (res ^ (res << 2)) & 0x33333333;
    res = (res ^ (res << 1)) & 0x55555555;

    res
}

fn interleave_bits(a: u16, b: u16) -> u32 {
    (interleave_zeros(a) << 1) | interleave_zeros(b)
}

#[test]
fn test_interleave_bits() {
    // Exhaustively check all pairs of u16 (takes about 40 seconds)
    for a in 0..std::u16::MAX {
        for b in 0..std::u16::MAX {
            assert_eq!(interleave_bits_slow(a, b), interleave_bits(a, b));
        }
    }
}

#[cfg(test)]
fn interleave_bits_slow(mut a: u16, mut b: u16) -> u32 {
    let mut r = 0u32;

    for i in (0..16).rev() {
        r <<= 1;
        r |= ((a >> i) & 1) as u32;

        r <<= 1;
        r |= ((b >> i) & 1) as u32;
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
        eprintln!("Allocated hashers and space");
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

        let hashers = self.hashers.chunks_exact(K_HALF);
        assert!(hashers.remainder().is_empty());
        let half_hashes = self.half_hashes.chunks_exact_mut(n);

        eprintln!("Compute half-width hashes");
        let start = Instant::now();
        // TODO do this in parallel
        for (hashes, hashers) in half_hashes.zip(hashers) {
            for h in hashers {
                h.hash(&self.ts, &fft_data, hashes);
            }
        }
        let end = Instant::now();
        eprintln!("elapsed: {:?}", end - start);

        let mut tables = Vec::with_capacity(self.repetitions);

        let mut scratch = Vec::with_capacity(n);
        eprintln!("setup sorted tables");
        let start = Instant::now();
        // TODO also parallelize
        for r in 0..self.repetitions {
            let (l_idx, r_idx) = get_minimal_index_pair(r);
            let l_hashes = &self.half_hashes[l_idx * n..(l_idx + 1) * n];
            let r_hashes = &self.half_hashes[r_idx * n..(r_idx + 1) * n];

            let table = Table::new(
                n,
                (0..n).map(|i| {
                    let h = interleave_bits(l_hashes[i], r_hashes[i]);
                    (h, i)
                }),
                &mut scratch,
            );

            tables.push(table);
        }
        let end = Instant::now();
        eprintln!("elapsed: {:?}", end - start);

        LSHTables {
            dimension: self.ts.w,
            tensor_repetitions: self.tensor_reps,
            repetitions: self.repetitions,
            tables,
        }
    }
}

struct Table {
    hashes: Vec<u32>,
    indices: Vec<usize>,
}

impl Table {
    fn new<I: IntoIterator<Item = (u32, usize)>>(
        n: usize,
        iter_elements: I,
        scratch: &mut Vec<(u32, usize)>,
    ) -> Self {
        scratch.clear();
        scratch.reserve_exact(n);
        for pair in iter_elements {
            scratch.push(pair);
        }
        scratch.sort_unstable_by_key(|pair| pair.0);
        let (hashes, indices): (Vec<u32>, Vec<usize>) = scratch.drain(..).unzip();
        debug_assert!(hashes.is_sorted());
        Self { hashes, indices }
    }

    fn segments<'slf>(&'slf self) -> TableSegments<'slf> {
        TableSegments::new(self)
    }
}

/// A collection of segments of lexicograpyically sorted elements, with the invariant
/// that all elements in the same range have the same prefix of the given number of bits.
pub struct TableSegments<'table> {
    table: &'table Table,
    prefix_length: usize,
    /// The breakpoints between one segment and the other
    breakpoints: Vec<usize>,
}

impl<'table> TableSegments<'table> {
    fn new(table: &'table Table) -> Self {
        let prefix_length = 32;
        let mut breakpoints = Vec::new();

        let mut b = 0;
        breakpoints.push(b);
        while b < table.hashes.len() {
            let needle = table.hashes[b];
            b += table.hashes[b..].partition_point(|h| *h == needle);
            breakpoints.push(b);
        }
        assert_eq!(b, table.hashes.len());

        Self {
            table,
            prefix_length,
            breakpoints,
        }
    }

    fn current_mask(&self) -> u32 {
        if self.prefix_length == 0 {
            return 0;
        }
        0xFFFFFFFF << (32 - self.prefix_length)
    }

    /// Iterate through contiguous runs of indices sharing the same prefix at the
    /// current prefix length.
    pub fn iter(&self) -> impl Iterator<Item = &'table [usize]> + '_ {
        #[cfg(test)]
        let mask = self.current_mask();

        self.breakpoints
            .iter()
            .zip(&self.breakpoints[1..])
            .map(move |(&start, &end)| {
                #[cfg(test)]
                {
                    let hs = &self.table.hashes[start..end];
                    for &h in hs {
                        assert_eq!(h & mask, hs[0] & mask);
                    }
                }
                &self.table.indices[start..end]
            })
    }

    pub fn shorten_prefix(&mut self, new_prefix: usize) {
        assert!(new_prefix < self.prefix_length);
        self.prefix_length = new_prefix;
        let mask = self.current_mask();

        assert_eq!(self.breakpoints[0], 0);
        assert_eq!(*self.breakpoints.last().unwrap(), self.table.hashes.len());

        let mut last_hash = self.table.hashes[0];

        let hashes = &self.table.hashes;

        self.breakpoints.retain(|b| {
            if *b == 0 || *b == hashes.len() {
                // keep the first and the last
                return true;
            }
            let keep = (hashes[*b] & mask) != (last_hash & mask);
            last_hash = hashes[*b];
            keep
        });

        assert_eq!(self.breakpoints[0], 0);
        assert_eq!(*self.breakpoints.last().unwrap(), self.table.hashes.len());
    }
}

pub struct LSHTables {
    dimension: usize,
    repetitions: usize,
    tensor_repetitions: usize,
    tables: Vec<Table>,
}

impl LSHTables {
    pub fn from_ts<R: Rng>(ts: &WindowedTimeseries, repetitions: usize, rng: &mut R) -> Self {
        let builder = LSHTablesBuilder::new(ts, ts.w, repetitions, rng);
        builder.build()
    }

    pub fn segments<'tables>(&'tables self) -> Vec<TableSegments<'tables>> {
        self.tables.iter().map(|t| t.segments()).collect()
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
fn test_segments() {
    let w = 1000;
    let ts = crate::load::loadts("data/ECG.csv.gz", Some(3000)).unwrap();
    let ts = WindowedTimeseries::new(ts, w, false);
    let mut rng = rand::thread_rng();
    let repetitions = 4;
    let tables = LSHTables::from_ts(&ts, repetitions, &mut rng);
    let t = &tables.tables[0];
    let mut segments = t.segments();
    let _ = segments.iter().count();
    segments.shorten_prefix(31);
    let _ = segments.iter().count();
    segments.shorten_prefix(29);
    let _ = segments.iter().count();
    segments.shorten_prefix(1);
    assert_eq!(segments.iter().count(), 2);
    segments.shorten_prefix(0);
    assert_eq!(segments.iter().count(), 1);
}

#[test]
fn test_failure_probability_tensor() {
    use crate::distance::zeucl;

    let w = 1000;
    let ts = crate::load::loadts("data/ECG.csv.gz", Some(3000)).unwrap();
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
            assert!(fp >= 0.0);
            assert!(fp >= fp_independent);
            if fp < 0.01 && !printed {
                eprintln!("bits {} rep {}", bits, rep);
                printed = true;
            }
        }
    }
}
