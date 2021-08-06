// # Embedding
//
// To be able to use simple LSH functions like HyperplaneLSH,
// we need to embed vectors in the Euclidean space into a kernel
// space whose inner product is related to the euclidean distance.
//
//  - [Rahimi and Recht paper](https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf) on random Fourier features
//  - [Sutherland and Schneider paper](https://arxiv.org/pdf/1506.02785.pdf) on more accurate analysis of the above
//  - [Christiani paper](https://arxiv.org/pdf/1605.02687v1.pdf) on LSH framework, including embeddings


use crate::types::WindowedTimeseries;
use crate::distance::*;
use rand::prelude::*;
use rand_distr::{StandardNormal, Uniform};
use rand_xoshiro::Xoshiro256Plus;

pub fn scaling_factor<D: Fn(&WindowedTimeseries, usize, usize) -> f64>(
    ts: &WindowedTimeseries,
    d: D,
    target_similarity: f64,
) -> f64 {
    let dp = ts.distance_profile(0, d);
    let mean: f64 = dp.iter().sum::<f64>() / dp.len() as f64;
    mean / (-2.0 * target_similarity.ln())
}

pub struct Embedder {
    pub dim_in: usize,
    pub dim_out: usize,
    pub input_scaling: f64,
    pub scale_factor: f64,
    vectors: Vec<Vec<f64>>,
    offsets: Vec<f64>,
}

impl Embedder {
    pub fn new(dim_in: usize, dim_out: usize, input_scaling: f64, seed: u64) -> Self {
        let scale_factor = (2.0/dim_out as f64).sqrt();
        let mut rng = Xoshiro256Plus::seed_from_u64(seed);
        let mut vectors = Vec::with_capacity(dim_out);
        let mut offsets: Vec<f64> = Vec::with_capacity(dim_out);
        let unif = Uniform::new_inclusive(0.0, 2.0*std::f64::consts::PI);
        for _ in 0..dim_out {
            let mut v: Vec<f64> = Vec::with_capacity(dim_in);
            for _ in 0..dim_in {
                v.push(rng.sample(StandardNormal));
            }
            vectors.push(v);
            offsets.push(rng.sample(unif));
        }
        Self {
            dim_in, dim_out, input_scaling, scale_factor, vectors, offsets
        }
    }

    // This is where we do the embedding

    pub fn embed(&self, v: &[f64], output: &mut [f64]) {
        assert!(v.len() == self.dim_in);
        assert!(output.len() == self.dim_out);
        for i in 0..self.dim_out {
            let p = dot(v, &self.vectors[i]) / self.input_scaling;
            output[i] = self.scale_factor * (p + self.offsets[i]).cos();
        }
        let n = norm(&output);
        for i in 0..self.dim_out {
            output[i] /= n;
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{distance::{dot, euclidean, normalize}, embedding::Embedder};

    fn gaussian_kernel(a: &[f64], b: &[f64]) -> f64 {
        (-euclidean(a, b).powi(2) / 2.0).exp()
    }

    #[test]
    fn test_embedding_distance() {
        use rand::prelude::*;
        use rand_distr::StandardNormal;
        use rand_xoshiro::Xoroshiro128PlusPlus;
        let mut rng = Xoroshiro128PlusPlus::seed_from_u64(3463992);
        let dim = 100;

        let embedder = Embedder::new(dim, dim, 1.0, 13462);

        let a: Vec<f64> = StandardNormal.sample_iter(&mut rng).take(dim).collect();
        let b: Vec<f64> = a.iter().map(|x| x + 1.1).collect();
        let a = normalize(&a);
        let b = normalize(&b);
        let mut ea = vec![0.0; dim];
        let mut eb = vec![0.0; dim];
        embedder.embed(&a, &mut ea);
        embedder.embed(&b, &mut eb);

        let dotp = dot(&ea, &eb);
        let kernd = gaussian_kernel(&a, &b);
        assert_eq!(dotp, kernd);
    }
}
