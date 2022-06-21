<script>
  import Line from './Line.svelte';
  import { LayerCake, Svg } from 'layercake';

  export let data;
  export let sublength;
  export let highlighted;
  
  $: start = highlighted
</script>

<style>
  .ts-container {
    width: 100%;
    height: 100px;
  }
</style>

<figure class="l-page ts-container">
   <LayerCake x="x" y="y" data={data}>
     <Svg>
       <Line stroke_width=0.4/>
     </Svg>
     <Svg>
       <Line stroke={"#F00"} minX={start} maxX={start + sublength} stroke_width=2/>
     </Svg>
   </LayerCake>
</figure>
