use anyhow::Result;
use argh::FromArgs;
use attimo::{
    distance::zeucl,
    load::loadts,
    motifs::{Motif, TopK},
    timeseries::WindowedTimeseries,
};
use rand::prelude::*;
use rand_xoshiro::Xoshiro256PlusPlus;
/// This is an implementation of [Probabilistic discovery of time series motifs](https://www.cs.ucr.edu/~eamonn/SIGKDD_Motif.pdf).
use std::{collections::BTreeMap, time::Instant};

struct SaxTransformer {
    /// lenght of the PAA window
    paa_window: usize,
    /// the length of the transformed subsequence
    word_length: usize,
    /// the breakpoints to transform numbers in symbols
    breakpoints: Vec<f64>,
}

impl SaxTransformer {
    fn new(ts: &WindowedTimeseries, alphabet: usize, paa_window: usize) -> Self {
        assert!(3 <= alphabet && alphabet <= 6);
        let breakpoints = match alphabet {
            3 => vec![-0.43, 0.43],
            4 => vec![-0.67, 0.0, 0.67],
            5 => vec![-0.84, -0.25, 0.25, 0.84],
            6 => vec![-0.97, -0.43, 0.0, 0.43, 0.97],
            _ => panic!("Unsupported alphabet size"),
        };
        let transformed_length = ts.w / paa_window;
        Self {
            paa_window,
            word_length: transformed_length,
            breakpoints,
        }
    }

    fn transform_single(&self, ts: &WindowedTimeseries, j: usize, out: &mut [u8], buf: &mut [f64]) {
        assert!(buf.len() == ts.w);
        assert!(out.len() == ts.w / self.paa_window);
        ts.znormalized(j, buf);
        for (i, chunk) in buf.chunks_exact(self.paa_window).enumerate() {
            let c = chunk.iter().sum::<f64>() / chunk.len() as f64;
            let mut s = 0;
            while s < self.breakpoints.len() && self.breakpoints[s] < c {
                s += 1;
            }
            out[i] = s as u8;
        }
    }

    fn transform(&self, ts: &WindowedTimeseries) -> SaxTimeseries {
        let ns = ts.num_subsequences();
        let mut indices = Vec::new();
        let mut sax_words = vec![0; self.word_length];
        let mut buf = vec![0.0; ts.w];
        self.transform_single(ts, 0, &mut sax_words[0..self.word_length], &mut buf);
        indices.push(0usize);
        let mut saxw = vec![0u8; self.word_length];
        for i in 1..ns {
            self.transform_single(ts, i, &mut saxw, &mut buf);
            if saxw != sax_words[sax_words.len() - self.word_length..] {
                // If the word is different from the previous one, then we skip it
                sax_words.extend_from_slice(&saxw);
                indices.push(i);
            }
        }
        SaxTimeseries {
            word_length: self.word_length,
            indices,
            sax_words,
        }
    }
}

struct SaxTimeseries {
    /// the length of a single SAX word
    word_length: usize,
    /// indices into the original timeseries. This allows for
    /// run-length encoding by skipping redundant entries
    indices: Vec<usize>,
    /// the words corresponding the the indices
    sax_words: Vec<u8>,
}

impl SaxTimeseries {
    fn iter(&self) -> impl Iterator<Item = (usize, &[u8])> + '_ {
        self.indices
            .iter()
            .copied()
            .zip(self.sax_words.chunks_exact(self.word_length))
    }
}

fn projection_motifs(
    ts: &WindowedTimeseries,
    saxts: &SaxTimeseries,
    motifs: usize,
    k: usize,
    repetitions: usize,
    seed: u64,
) -> Vec<Motif> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let start = Instant::now();

    let rngs: Vec<(usize, Xoshiro256PlusPlus)> = (0..repetitions)
        .map(|rep| {
            rng.jump();
            let r = rng.clone();
            (rep, r)
        })
        .collect();

    // benchmarking shows that BTreeMap works better than HashMap in this use case
    let mut collision_matrix = BTreeMap::new();

    let indices: Vec<usize> = (0..saxts.word_length).collect();
    println!("rep,comp,time");
    rngs.into_iter().for_each(|(rep, mut rng)| {
        let rep_start = Instant::now();
        eprintln!("Repetition {rep}");
        let proj: Vec<usize> = indices.choose_multiple(&mut rng, k).cloned().collect();
        let mut hashed: Vec<(Vec<u8>, usize)> = saxts
            .iter()
            .map(|(i, word)| {
                let hash: Vec<u8> = proj.iter().map(move |h| word[*h]).collect();
                (hash, i)
            })
            .collect();
        println!("{},hashing,{}", rep, rep_start.elapsed().as_secs_f64());
        let sort_start = Instant::now();
        hashed.sort_unstable();
        println!("{},sorting,{}", rep, sort_start.elapsed().as_secs_f64());

        let counting_start = Instant::now();
        // define the buckets and count collisions
        let mut start = 0;
        let mut end = start + 1;
        while end < hashed.len() {
            // find the end of the bucket
            while end < hashed.len() && hashed[end].0 == hashed[start].0 {
                end += 1;
            }

            for i in start..end {
                for j in (i + 1)..end {
                    let a = std::cmp::min(i, j);
                    let b = std::cmp::max(i, j);
                    if b - a > ts.w {
                        collision_matrix
                            .entry((a, b))
                            .and_modify(|c| *c += 1)
                            .or_insert(1);
                    }
                }
            }

            start = end;
            end = start + 1;
        }
        println!(
            "{},counting,{:?}",
            rep,
            counting_start.elapsed().as_secs_f64()
        );
        eprintln!("rep {} took {:?}", rep, rep_start.elapsed());
    });

    let mut collision_matrix: Vec<(usize, (usize, usize))> = collision_matrix
        .into_iter()
        .map(|pair| (pair.1, pair.0))
        .collect();
    collision_matrix.sort_unstable_by(|a, b| a.cmp(b).reverse());

    let mut topk = TopK::new(motifs, ts.w);

    eprintln!("collision matrix has {} entries", collision_matrix.len());
    let mut cur_count = collision_matrix[0].0;
    for (count, (a, b)) in collision_matrix.iter() {
        if *count == cur_count {
            if b - a > ts.w {
                let d = zeucl(ts, *a, *b);
                let m = Motif {
                    idx_a: *a,
                    idx_b: *b,
                    distance: d,
                    elapsed: None,
                    discovered: start.elapsed(),
                };
                topk.insert(m);
            }
        } else {
            // if we have enough elements in the topk, we stop
            if topk.len() == motifs {
                topk.for_each(|m| {
                    dbg!(&m);
                    m.elapsed = Some(start.elapsed());
                });
                return topk.to_vec();
            }
            cur_count = *count;
        }
    }
    panic!("Could not find enough motifs")
}

/// Compute the approximate matrix profile using SCRIMP++
#[derive(FromArgs)]
struct Args {
    #[argh(option, short = 'w')]
    /// subsequcence length
    pub window: usize,

    #[argh(option, default = "1")]
    /// the number of motifs to look for
    pub motifs: usize,

    #[argh(option)]
    /// consider only the given number of points from the input
    pub prefix: Option<usize>,

    #[argh(option)]
    /// the size of the PAA window
    pub paa: usize,

    #[argh(option)]
    /// the size of the alphabet
    pub alphabet: usize,

    #[argh(option)]
    /// the number of repetitions
    pub repetitions: usize,

    #[argh(option)]
    /// the length of the projection
    pub k: usize,

    #[argh(option, default = "12453")]
    /// seed for the psudorandom number generator
    pub seed: u64,

    #[argh(switch)]
    /// wether meand and std computations should be at the best precision, at the expense of running time
    pub precise: bool,

    #[argh(option, default = "default_output()")]
    /// path to the output file
    pub output: String,

    #[argh(positional)]
    /// path to the data file
    pub path: String,
}

fn default_output() -> String {
    "ChiuKL.csv".to_owned()
}

fn output_csv(path: &str, motifs: &[Motif]) -> Result<()> {
    use std::io::prelude::*;
    let mut f = std::fs::File::create(path)?;
    for m in motifs {
        if let Some(confirmation_time) = m.elapsed {
            writeln!(
                f,
                "{}, {}, {}, {}",
                m.idx_a,
                m.idx_b,
                m.distance,
                confirmation_time.as_secs_f64()
            )?;
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let args: Args = argh::from_env();

    let w = args.window;
    let ts = WindowedTimeseries::new(loadts(args.path, args.prefix)?, w, args.precise);

    let transformer = SaxTransformer::new(&ts, args.alphabet, args.paa);
    let saxts = transformer.transform(&ts);
    eprintln!("The sax words are {}", saxts.indices.len());

    let motifs = projection_motifs(
        &ts,
        &saxts,
        args.motifs,
        args.k,
        args.repetitions,
        args.seed,
    );
    output_csv(&args.output, &motifs)?;
    Ok(())
}
