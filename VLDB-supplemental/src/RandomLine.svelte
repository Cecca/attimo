<script>
  import { LayerCake, Svg } from 'layercake';
  import Line from './Line.svelte';
  import { randomLcg, randomNormal } from 'd3-random';

  export let length;
  export let seed;
  export let stroke;

  $: data = (function() {
    let rng = randomNormal.source(randomLcg(seed))(0, 1);
    let randVec = [];
    for (var i=0; i<length; i++) {
      randVec.push({x: i, y: rng()});
    }
    return randVec;
  })();
</script>

<style>
  .ts-container {
    width: 25%;
    height: 100px;
  }
</style>

<div class="ts-container">
<LayerCake x="x" y="y" data={data}>
  <Svg>
    <Line stroke={stroke}/>
  </Svg>
</LayerCake>
</div>

