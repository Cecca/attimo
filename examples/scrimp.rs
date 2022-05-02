use argh::FromArgs;
use indicatif::{ProgressBar, ProgressStyle};
use anyhow::Result;
use attimo::load::loadts;
use attimo::timeseries::*;
use attimo::distance::*;
use attimo::motifs::{Motif, TopK};
use std::time::Instant;
use rand::prelude::*;
use rand_xoshiro::Xoshiro256StarStar;

fn rand_seq<R: Rng>(min: usize, max: usize, by: usize, rng: &mut R) -> Vec<usize> {
    eprint!("Computing shuffled sequence...");
    let mut i = min;
    let mut res: Vec<usize> = if by == 1 {
        (min..max).collect()
    } else {
        std::iter::from_fn(|| {
            if i >= max {
                return None;
            }
            let res = Some(i);
            i += by;
            res
        })
        .collect()
    };
    res.shuffle(rng);
    eprintln!(" done");
    res
}

fn zeucl_to_dotp(w: usize, d: f64, ma: f64, mb: f64, sa: f64, sb: f64) -> f64 {
    let w = w as f64;
    (1.0 - d*d / (2.0*w)) * w * sa * sb + w * ma * mb
}

fn dotp_to_zeucl(w: usize, q: f64, ma: f64, mb: f64, sa: f64, sb: f64) -> f64 {
    let w = w as f64;
    (2.0 * w * (1.0 - (q - w*ma*mb) / (w*sa*sb))).sqrt()
}

fn pre_scrimp<R: Rng>(
    ts: &WindowedTimeseries,
    s: usize,
    exclusion_zone: usize,
    mp: &mut [f64],
    indices: &mut [usize],
    fft_data: &FFTData,
    rng: &mut R,
) {
    // note that here we are considering sequences which are s > ts.w apart, so
    // we have no trivial matches
    let ns = ts.num_subsequences();
    let w = ts.w;

    let pbar = ProgressBar::new((ns / s) as u64);
    pbar.set_style(ProgressStyle::default_bar()
        .template("[{elapsed_precise}] {bar:40.cyan/blue} ETA: {eta} {pos:>7}/{len:7} {msg}"));
    for i in rand_seq(0, ns, s, rng) {
        let dists = ts.distance_profile(fft_data, i);

        // update the running matrix profile
        for (j, ((mp_val, index_val), dp_val)) in mp.iter_mut().zip(indices.iter_mut()).zip(dists.iter()).enumerate() {
            if (i as isize - j as isize).abs() >= exclusion_zone as isize && *dp_val < *mp_val {
                *mp_val = *dp_val;
                *index_val = i;
            }
        }

        // nearest neighbor index, excluding trivial matches
        let (j, d) = dists
            .iter()
            .enumerate()
            .filter(|(j, _)| (*j as isize - i as isize).abs() >= exclusion_zone as isize)
            .min_by(|p1, p2| p1.1.partial_cmp(p2.1).unwrap())
            .unwrap();

        let mut q = zeucl_to_dotp(w, *d, ts.mean(i), ts.mean(j), ts.sd(i), ts.sd(j));
        debug_assert!((q - dot(ts.subsequence(i), ts.subsequence(j))).abs() < 0.00001,
            "actual {} expectd {}",
            q,
            dot(ts.subsequence(i), ts.subsequence(j))
        );
        let q_prime = q;

        for k in 1..(s-1) {
            if i + k >= ns || j + k >= ns {
                break;
            }
            q = q - ts.data[i + k] * ts.data[j+k]
                  + ts.data[i + k + w] * ts.data[j + k + w];
            debug_assert!((q - dot(ts.subsequence(i+k), ts.subsequence(j+k))).abs() < 0.1,
                "actual {} expectd {}",
                q,
                dot(ts.subsequence(i+k), ts.subsequence(j+k))
            );
            let d = dotp_to_zeucl(w, q, ts.mean(i+k), ts.mean(j+k), ts.sd(i+k), ts.sd(j+k));
            // debug_assert!((d - zeucl(ts, i, j)).abs() <= 0.001,
            //     "actual {} expected {}",
            //     d, zeucl(ts, i, j)
            // );
            if d < mp[i+k] {
                mp[i+k] = d;
                indices[i+k] = j+k;
            }
            if d < mp[j+k] {
                mp[j+k] = d;
                indices[j+k] = i+k;
            }
        }

        q = q_prime;
        for k in 1..(s-1) {
            if k > i || k > j {
                break;
            }
            q = q - ts.data[i - k + w] * ts.data[j - k + w]
                  + ts.data[j - k] * ts.data[i - k];
            let d = dotp_to_zeucl(w, q, ts.mean(i - k), ts.mean(j-k), ts.sd(i-k), ts.sd(j-k));
            // debug_assert!((d - zeucl(ts, i, j)).abs() <= 0.00000001);
            if d < mp[i-k] {
                mp[i-k] = d;
                indices[i-k] = j-k;
            }
            if d < mp[j-k] {
                mp[j-k] = d;
                indices[j-k] = i-k;
            }
        }
        pbar.inc(1);
    }
    pbar.finish_and_clear();
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

    #[argh(option, default="0.25")]
    /// the skip for pre-scrimp, as a fraction of the window size
    pub skip: f64,

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
    "scrimp.csv".to_owned()
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
    debug_assert!(false, "This software should run only in Release mode, times are important");
    let args: Args = argh::from_env();

    eprintln!("Reading input");
    let w = args.window;
    let ts = WindowedTimeseries::new(loadts(args.path, args.prefix)?, w, args.precise);
    eprintln!("Computing FFT");
    let fft_data = FFTData::new(&ts);
    let s = (args.skip * (w as f64)) as usize;
    let exclusion_zone = w;

    let mut rng = Xoshiro256StarStar::seed_from_u64(args.seed);

    let timer = Instant::now();
    let mut mp = vec![f64::INFINITY; ts.num_subsequences()];
    let mut indices = vec![0; ts.num_subsequences()];

    eprintln!("Running PreSCRIMP");
    pre_scrimp(&ts, s, exclusion_zone, &mut mp, &mut indices, &fft_data, &mut rng);

    eprintln!("Getting motifs");
    let mut topk = TopK::new(args.motifs, exclusion_zone);
    for (i, (j, d)) in indices.iter().zip(mp.iter()).enumerate() {
        let m = Motif {
            idx_a: std::cmp::min(i, *j),
            idx_b: std::cmp::max(i, *j),
            distance: *d,
            elapsed: None,
        };
        topk.insert(m);
    }
    eprintln!("Writing output");
    let mut motifs = topk.to_vec();
    for m in motifs.iter_mut() {
        m.elapsed.replace(timer.elapsed());
    }
    output_csv(&args.output, &motifs)?;
    eprintln!("Done");

    Ok(())
}
