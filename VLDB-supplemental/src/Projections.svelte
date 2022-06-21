<script>
  import { getContext } from "svelte";
  import { draw } from 'svelte/transition';

  const { data, xGet, yGet, xScale, yScale } = getContext("LayerCake");

  /** @type {Number} - The direction of the projection*/
  export let direction;

  export let colorScale;

  /** @type {Number} [strokeWidth=0.3] – The line stroke width. */
  export let strokeWidth = 0.3;

  function buildPath(point, direction) {
    let norm = direction.x*direction.x + direction.y*direction.y;
    let proj = point.x*direction.x + point.y*direction.y;
    let end = {
      x: proj * (direction.x / norm),
      y: proj * (direction.y / norm)
    };
    return `M${$xGet(point)},${$yGet(point)}L${$xGet(end)},${$yGet(end)}`;
  }
</script>

<g class="projections-group">
  {#each $data as d}
    {#key d}
      <path
        in:draw="{{duration: 1000}}"
        d={buildPath(d, direction)}
        stroke={colorScale(d.h)}
        stroke-width={strokeWidth}
      />
    {/key}
  {/each}
</g>
