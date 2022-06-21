<script>
  import Points from "./Points.svelte";
  import Projections from "./Projections.svelte";
  import SlopedLine from "./SlopedLine.svelte";
  import { LayerCake, Svg } from 'layercake';
  import { randomNormal, randomUniform, randomLcg, randomInt } from "d3-random";
  import { scaleOrdinal } from "d3-scale";
  import { format } from "d3-format";
  import { extent } from "d3";
  import { schemeTableau10, schemeSet1 } from "d3-scale-chromatic";

  export let data;
  export let seed = 1234;
  export let k = 1;
  export let w = 0.5;

  $: functions = sampleFunctions(seed, k, w);

  $: rng = randomNormal.source(randomLcg(seed))(0, 1);
  $: rngUnif = randomUniform.source(randomLcg(seed))(0, w);

  function sampleFunctions(seed, k, w) {
    let rng = randomNormal.source(randomLcg(seed))(0, 1);
    let rngUnif = randomUniform.source(randomLcg(seed))(0, w);
    var funcs = [];
    for (var j=0; j<k; j++) {
      let x = rng();
      let y = rng();
      let b = rngUnif();
      let norm = Math.sqrt(x*x + y*y);
      let a = {
        x: x / norm,
        y: y / norm
      };
      let ticks = [];
      for (var i=-10; i<10; i++) {
        let off = (w * i - b);
        if (Math.abs(off) <= 0.5) {
          let tick = {
            x: a.x * off,
            y: a.y * off,
          }
          ticks.push(tick);
        }
      }
      let func = {
        a: a,
        b: b,
        bounds: ticks
      };
      funcs.push(func);
    }
    console.log("Sampled " + funcs.length + " functions out of " + k);
    return funcs;
  }

  function project(v, a) {
    return v.x * a.x + v.y * a.y;
  }

  function lsh(v, a, b, w) {
    return Math.floor((project(v, a) + b) / w);
  }

  // compute the ticks
  // $: boundaries = (function (){
  //   console.log("w " + w + " anorm " + anorm);
  //   let ticks = [];
  //   for (var i=-10; i<10; i++) {
  //     let off = (w * i - b);
  //     console.log("off " + off);
  //     if (Math.abs(off) <= 0.5) {
  //       let tick = {
  //         x: a.x * off,
  //         y: a.y * off,
  //       }
  //       ticks.push(tick);
  //     }
  //   }
  //   console.log(ticks);
  //   return ticks;
  // })();

  // Compute and store the LSH values
  $: {
    console.log("Now we have " + functions.length + " hash functions");
    data.forEach((v) => {
      let hash = [];
      for (var i=0; i<functions.length; i++) {
        let f = functions[i];
        let h = lsh(v, f.a, f.b, w);
        hash.push(h);
      }
      v.h = hash.join();
    });
    data = data; // trigger re-rendering
  }

  $: colorScale = scaleOrdinal(schemeTableau10)
    .domain(extent(data, d => d.h));

  function handleClick() {
    if (seed) {
      seed += 100;
    } else {
      seed = 1234;
    }
  }

</script>

<style>
  /* see https://spin.atomicobject.com/2015/07/14/css-responsive-square/ */
  .container {
    margin-left: auto;
    margin-right: auto;
    width: 50%;
  }
  .container:after {
    content: "";
    display: block;
    padding-bottom: 100%;
  }
  .content {
    position:absolute;
    width: 100%;
    height: 100%;
  }
</style>

<div>
<aside>Note that points in different cells might be assigned the same color: this is just due to the limited number of colors.</aside>
  <div>
    <label for="volume">Quantization width <d-math>r = {w}</d-math></label>
    <input type=range bind:value={w} min=0.1 max=2 step=0.1>
  </div>
  <div>
    <label for="volume">Number of hash functions <d-math>K = {k}</d-math></label>
    <input type=range bind:value={k} min=1 max=5 step=1>
  </div>

  <button on:click={handleClick}>
    Sample function
  </button>
  <figure class="l-body container">
    <div class="content">
      <LayerCake data={data} x="x" y="y">
        <Svg>
          {#if seed}
            <Points {colorScale} />
            {#each functions as f}
              {#each f.bounds as b}
                <SlopedLine 
                  slope={- f.a.x / f.a.y} 
                  scale=1.9
                  stroke_width=0.3
                  referenceX={b.x} 
                  referenceY={b.y}/>
              {/each}
            {/each}
          {:else}  
            <Points />
          {/if}
        </Svg>
      </LayerCake>
    </div>
  </figure>
</div>