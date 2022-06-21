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
  export let seed = undefined;
  export let w = 0.1;

  $: rng = randomNormal.source(randomLcg(seed))(0, 1);
  $: rngUnif = randomUniform.source(randomLcg(seed))(0, w);

  $: aorig = { x: rng(), y: rng() };
  $: anorm = Math.sqrt(aorig.x*aorig.x + aorig.y*aorig.y);
  $: a = {
    x: aorig.x / anorm,
    y: aorig.y / anorm
  };
  $: b = rngUnif();
  
  function project(v, a) {
    return v.x * a.x + v.y * a.y;
  }

  function lsh(v, a, b, w) {
    return Math.floor((project(v, a) + b) / w);
  }

  // compute the ticks
  $: boundaries = (function (){
    let anorm = Math.sqrt(a.x*a.x + a.y*a.y);
    let base = {
      x: a.x / anorm,
      y: a.y / anorm
    };
    console.log("w " + w + " anorm " + anorm);
    let ticks = [];
    for (var i=-10; i<10; i++) {
      let off = (w * i - b);
      console.log("off " + off);
      if (Math.abs(off) <= 0.5) {
        let tick = {
          x: a.x * off,
          y: a.y * off,
        }
        ticks.push(tick);
      }
    }
    console.log(ticks);
    return ticks;
  })();

  // Compute and store the LSH values
  $: {
    data.forEach((v) => {
      v.h = lsh(v, a, b, w);
      v.p = project(v, a) + b;
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

  let numFormat = function(num) {
    // format(".4");
    return num;
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
  <div>
    <label for="volume">Quantization width <d-math>r = {w}</d-math></label>
    <input type=range bind:value={w} min=0.1 max=2 step=0.1>
  </div>

  <button on:click={handleClick}>
    Sample function
  </button>
  <d-math>a = ({numFormat(a.x.toFixed(4))}, {numFormat(a.y.toFixed(4))})</d-math>
  <d-math>b = {numFormat(b.toFixed(4))}</d-math>
  <figure class="l-body container">
    <div class="content">
      <LayerCake data={data} x="x" y="y">
        <Svg>
          {#if seed}
            <Points {colorScale} />
            <Projections direction={a} {colorScale} />
            <SlopedLine slope={a.y / a.x} stroke_width=2 />
            {#each boundaries as b}
              <SlopedLine 
                slope={- a.x / a.y} 
                scale=50
                stroke_width=2
                referenceX={b.x} 
                referenceY={b.y}/>
            {/each}
          {:else}  
            <Points />
          {/if}
        </Svg>
      </LayerCake>
    </div>
  </figure>
</div>