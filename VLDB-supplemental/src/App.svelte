<script>
  import { writable } from 'svelte/store'
  import { text as d3text } from "d3-fetch";
  import { randomInt, randomLcg } from "d3-random";
  import RandomPointsExample from "./RandomPointsExample.svelte";
  import RandomPointsExampleGrid from "./RandomPointsExampleGrid.svelte";
  import ProbabilityPlot from './ProbabilityPlot.svelte';

  const limit = 10000;
  const sublength = 400;

  let global_highlight = undefined;

  $: random_points = [];

  $: insect_ts = [];

  $: subsequences = (function() {
    let subs = [];
    for (var start=0; start < insect_ts.length - sublength; start += sublength/2) {
      subs.push(insect_ts.slice(start, start+sublength));
    }
    return subs;
  })();

  function sample_subs(list, sample_size, sublength, seed) {
    const n = list.length;
    if (n == 0) {
      return [];
    }
    const rng = randomInt.source(randomLcg(seed))(0, n);
    let subs = [];
    while(subs.length < sample_size) {
      let i = rng();
      console.log(i);
      subs.push(list.slice(i, i+sublength));
    }
    return subs;
  }
  
  async function load_randpoints() {
    let dat = await d3text("uniform.csv").then((txt) => {
      return txt
        .split("\n")
        .map((line, i) => {
          let tokens = line.split(",");
          return { x: parseFloat(tokens[0]), y: parseFloat(tokens[1]) };
        });
    });
    random_points = dat;
  }
  
  async function load_data() {
    let ts = await d3text("insect_b.txt").then((txt) => {
      return txt
        .split("\n")
        .map((line, i) => {
          return { x: i, y: +line };
        })
        .filter((d) => d.x < limit);
    });
    insect_ts = ts;
  }

  let insect_ts_promise = load_data();
  load_randpoints();

  // let subsequences_promise = insect_ts_promise.then((insect_ts) => {
  //   return sample_subs(insect_ts, 8, sublength, 1234);
  // });

  function handleHighlight(sub) {
    global_highlight = sub[0].x;
    console.log(global_highlight);
  }
</script>

<style>
  .subts-outer-container {
    width: 100%;
    /* height: 400px; */
  }

  .subts-container {
    position: relative;
    display: inline-block;
    width: 140px;
    height: 30px;
  }
</style>

<main>
  <d-front-matter>
    <script id="distill-front-matter" type="text/json">
{
        "title": "Supplemental material: Fast and Scalable Mining of Time Series Motifs with Probabilistic Guarantees",
        "description": "",
        "published": "",
        "authors": [
          {
            "author":"Matteo Ceccarello",
            "authorURL":"https://www.inf.unibz.it/~ceccarello/",
            "affiliations": [{"name": "Free University of Bozen/Bolzano"}]
          },
          {
            "author":"Johann Gamper",
            "authorURL":"https://www.inf.unibz.it/~gamper/",
            "affiliations": [{"name": "Free University of Bozen/Bolzano"}]
          }
        ],
        "katex": {
          "delimiters": [
            {"left": "$$", "right": "$$", "display": false}
          ]
        }
      }
    </script>
  </d-front-matter>
  <d-title>
    <h1>
      Supplemental material:<br/>
      <em>
      Fast and Scalable Mining of Time Series Motifs with Probabilistic
      Guarantees
      </em>
    </h1>
    <p>
        This is supplemental material to the paper "Fast and
        Scalable Mining of Time Series Motifs with Probabilistic Guarantees", 
        mainly to provide some background on Locality Sensitive Hashing.
        The goal is to give an intuition rather than to present a theoretically
        rigorous analysis, for which we refer the interested reader to the paper 
        by Datar et al.<d-cite key="DBLP:conf/compgeom/DatarIIM04" />.
    </p>
  </d-title>
  <d-byline />

  <d-article>
    
    <h2>A brief LSH primer</h2>

    <p>Locality Sensitive Hashing (LSH for short) is a technique often used in
    the context of <em>similarity search</em>.</p>
    <p>The basic idea of LSH is to map each input element to a <em>hash
    value</em> such that similar points have the same hash value with a higher
    probability than dissimilar points</p>
    <p>There are LSH families for several metric spaces (Cosine similarity,
    Jaccard similarity, Hamming distance...): we focus here on a
    popular LSH family for the Euclidean distance, first proposed in a seminal
    work by Datar et al.<d-cite key="DBLP:conf/compgeom/DatarIIM04" />.
    </p>
    <aside>Note that LSH families for the Euclidean distance with better
    properties have been proposed since<d-cite key="DBLP:conf/focs/AndoniI06"
    /><d-cite key="DBLP:conf/stoc/AndoniR15"/>.
    We focus here on the Datar et al. scheme because of its simplicity.
    </aside>

    <p>In this first part we will focus on interactive examples, deferring the details 
      of the inner workings of these functions to the following sections.</p>

    <p>The LSH function for a point <d-math>{` \\vec{x} \\in\\mathbb{R}^d `}</d-math> is defined as</p>
    
    <d-math block>{` h(x) = \\left\\lfloor\\frac{\\vec{a} \\cdot \\vec{x} + b}{r}\\right\\rfloor `}</d-math>
    <p>where <d-math>{` \\vec{a} \\in \\mathbb{R}^d `}</d-math> is a random vector with coordinates sampled from <d-math>{` \\mathcal{N}(0, 1) `}</d-math>, $$b$$ is a random scalar distributed uniformly in $$[0, r]$$, and $$r$$ is a parameter.</p>
    <p>In other words, this LSH function projects input points on a random direction, shifts them, and then quantizes the shifted projections according to the parameter $$r$$.</p>    
    <p>The picture below depicts a set of points on the Euclidean plane. 
      The slider allows to tune the parameter $$r$$, and clicking the <em>Sample function</em> 
      button will sample a new random function.</p>

    <RandomPointsExample data={random_points} />

    <p>First of all, not that the randomness is in the sampling of the function.
      Once the hash function has been sampled, it is applied to <em>all</em> points.
    </p>
    <p>Furthermore, note that the fact that <d-math>{` \\vec{a} `}</d-math> is sampled from
    the Standard Normal distribution has nothing to do with the distribution of the data (which
    in the example above is distributed uniformly at random). Rather, it has to do with the 2-stable property of the Gaussian distribution, as we shall see later.</p>
    <p>
      The key property of LSH is that points which are close to each other happen to have the
      same hash value (color in the figure above) in most cases, whereas far away points are
      less likely to be assigned the same hash value.
      We refer to the probability of getting the same hash value as the
      <em>collision probability</em>.
    </p>
    <p>This collision probability can be controlled by means of the parameter $$r$$: the larger the value, the higher the probability of any two points to get the same color.</p>

    <p>
      Another way to control the collision probability is to consider several hash functions
      <em>at the same time</em>: fix a parameter $$K$$ and sample $$K$$ functions at random.
      The new hash values will simply be the concatenation of the outputs of each single
      hash function.
    </p>
    <p>
      Continuing our example, increasing this parameter $$K$$ is equivalent to using a more complex random partition of the input set.
    </p>
    <RandomPointsExampleGrid data={random_points} />
    
    <p>
      Clearly, increasing $$K$$ allows to decrease the probability that points are assigned the same hash value.
      This is very useful in order to make it less likely for far away points to collide on the same hash value.
      Of course, increasing $$K$$ also makes it less likely for <em>close</em> points to collide.
      To obviate this fact, the most straighforward solution is to repeat the above process $$L$$ times, where in each
      repetition a new collection of $$K$$ hash function is randomly sampled.
    </p>
    <p>
      Denoting with $$p(d)$$ the probability that two points at distance $$d$$ collide, we have that using $$K$$ concatenations
      with $$L$$ repetitions, the probability that two points at distance $$d$$ collide in at least one repetition is
    </p>
    <d-math block>
      1 - \left(1 - p(d)^K\right)^L
    </d-math>

    <p>
      To build an intuition of what this means in the case of the hash function described above, the plot below
      reports the probability of colliding as a function of the distance.
      The three sliders allow to tweak the parameters that we have seen so far.
    </p>

    <ProbabilityPlot />

    <p>We can see several effects playing out at the same time with the plot above:</p>
    <ul>
      <li>Increasing $$K$$ makes the curve <em>steeper</em>, as does making $$r$$ smaller;</li>
      <li>$$r$$ is less effective at controlling the shape of the curve, compared to $$K$$;</li>
      <li>Increasing $$L$$ increases the collision probability for points at any given distance, as expected;</li>
      <li>Changing the parameters changes the gap in collision probability of pairs of points that are at similar distances.
        For instance, with $$K=1$$, $$L=10$$, and $$r=0.1$$, points at distance 0.2 collide with probability $$\approx 0.9$$, and points
        at distance 0.3 collide with probability $$\approx 0.75$$.
        On the other hand, setting $$K=3$$ and $$L=200$$ makes these two probabilities 0.8 and 0.4, respectively.
      </li>
    </ul>
    
    <p>An effective setting of parameters would assign high probabilities to pairs of points at distances that are 
      <em>interesting</em> (for some definition of interesting) and low probabilities to the other pairs.
      In the context studied in the paper to which this supplemental material is companion interesting pairs are pairs of time series subsequences with smallest z-normalized Euclidean distances.
      In this setting, setting the parameters optimally would require the knowledge of the distribution of distances, which is not available.
      The algorithm we present automatically sets all the parameters.
    </p>
    
    <h3>Random projections and the Euclidean distance</h3>

    <p>The LSH family proposed by Datar et al. is based on a nice relationship between the Gaussian
    distribution and the Euclidean distance. Consider two points <d-math>{`x, y
    \\in \\mathbb{R}^d`}</d-math>, and let <d-math>||x - y||_2</d-math> be their
    Euclidean distance.</p>
    <p>Let {`$$\\vec{a} \\in \\mathbb{R}^d$$`} be a vector whose coordinates are independently
    distributed according to <d-math> {`\\mathcal{N}(0, 1) `}</d-math>, i.e. the
    standard Normal distribution.
    For a vector 
    <d-math>{` \\vec{x} \\in \\mathbb{R}^d `}</d-math>, let 
    <d-math>{` f_{\\vec{a}}(\\vec{x}) = \\vec{a} \\cdot \\vec{x} `}</d-math>, i.e. the dot product of 
    <d-math>{` \\vec{x} `}</d-math> and <d-math>{` \\vec{a} `}</d-math>.
    Computing this dot product amounts to computing the <em>projection</em> of 
    <d-math>{` \\vec{x} `}</d-math> on the line defined by
    <d-math>{` \\vec{a} `}</d-math>.
    </p>
    <p>Consider now two vectors
      <d-math>{` \\vec{x}, \\vec{y} \\in \\mathbb{R}^d `}</d-math>.
      The difference of their projections
      <d-math block>{` f_{\\vec{a}}(\\vec{x}) - f_{\\vec{a}}(\\vec{y}) `}</d-math>
      is a random variable: how is it distributed?
      Expanding the expression with the definition of 
      <d-math>{` f_{\\vec{a}}(\\vec{x}) `}</d-math>
      we have
    </p>
    <d-math block>{` 
      f_{\\vec{a}}(\\vec{x}) - f_{\\vec{a}}(\\vec{y}) =
      \\sum_{j=1}^d x_j a_j - \\sum_{j=1}^d y_j a_j =
      \\sum_{j=1}^d (x_j - y_j) a_j
    `}</d-math>
    <p>We now have that since each <d-math>{` a_j `}</d-math> is a Gaussian with mean 0 and variance 1, then each
      <d-math>{` (x_j - y_j) a_j `}</d-math> is a Gaussian with mean zero and variance <d-math>{` (x_j - y_j)^2 `}</d-math>.
      Then, given that the sum of independent Gaussian random variables is itself a Gaussian random variable, 
      we have that the difference of the random projections of the two points
      <d-math>{` f_{\\vec{a}}(\\vec{x}) - f_{\\vec{a}}(\\vec{y}) `}</d-math>
      is distributed as a Gaussian with mean 0 and variance
    </p>
    <d-math block>{` \\sum_{j=1}^d (x_j - y_j)^2 = || \\vec{x} - \\vec{y} ||_2^2 `}</d-math>
    <p>in other words, the variance of the difference of the projections is the squared Euclidean distance of the original points!</p>
    
    <p>
      To see what happens when we apply random projections to points at different distances, consider the 
      the following two pairs of points in <d-math>{` \\mathbb{R}^2 `}</d-math>.
    </p>
    <figure style="width: 60%; margin: auto">
      <img src="four-points.svg" alt="Four points on the plane, at different distances">
    </figure>

    <p> As we have seen before, under a random projection the distribution of
    distances follows a Gaussian distribution with variance equal to 
    the distance between the two points.
    Below we report the distribution of the distances of 
    random projections between the two blue points and the two red points.
    </p>
    <figure>
      <img src="projection-distribution.svg" alt="The probability distribution of the difference of projections of pairs of points at different distances">
    </figure>
    <p>From this figure, it is clear that the probability of the two blue points 
      to be projected close to each other is much higher than for the two orange points.
    </p>

    <h2>Conclusions</h2>
    <p>We have presented a short summary of Locality Sensitive Hashing applied to points in the Euclidean space. In our paper <em>Fast and Scalable Mining of Time Series Motifs with Probabilistic Guarantees</em> we employ these techniques to efficiently prune distance computations, automatically setting the parameters.
    </p>
    <p>
      In particular, the "points" to which we apply LSH are (z-normalized) subsequences of length $w$ of the input time series, which can be seen as points <d-math>{`\\in\\mathbb{R}^w`}</d-math>.
    </p>
    <p>
      For a more comprehensive treatment of LSH, we refer the interested reader to the original paper by Datar et al.<d-cite key="DBLP:conf/compgeom/DatarIIM04"></d-cite>, to the paper by Christiani<d-cite key="DBLP:conf/sisap/Christiani19"></d-cite> for an overview of techniques to reduce the number of hash computations, and to Wang et al.
      <d-cite key="DBLP:journals/corr/WangSSJ14"></d-cite> for a survey of LSH function families for many different metric spaces.
    </p>
    
  </d-article>

  <d-appendix>
    <d-bibliography src="references.bib" />
  </d-appendix>
</main>

