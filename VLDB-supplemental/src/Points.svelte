<script>
  import { getContext } from "svelte";

  const { data, xGet, yGet, xScale, yScale } = getContext("LayerCake");

  export let w = undefined;

  // $: console.log($data)

  /** @type {Number} [r=3] – The circle's radius. */
  export let r = 3;

  export let colorScale = undefined;

  /** @type {Number} [strokeWidth=0] – The circle's stroke width. */
  export let strokeWidth = 0;
</script>

<g>
<g class="scatter-group">
  {#each $data as d}
    <circle
      cx={$xGet(d)}
      cy={$yGet(d)}
      {r}
      fill={(colorScale)? colorScale(d.h) : "#000"}
      stroke={(colorScale)? colorScale(d.h) : "#000"}
      stroke-width={strokeWidth}
    />
  {/each}
</g>
{#if w}
<path 
  d={`M${$xScale(0)},${$yScale(0)}L${$xScale(w)},${$yScale(0)}`}
  stroke="#000"
  stroke-width=5
/>
{/if}
</g>