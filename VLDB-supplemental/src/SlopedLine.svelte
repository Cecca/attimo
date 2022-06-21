<script>
  import { getContext } from "svelte";

  const { xGet, yGet } = getContext("LayerCake");

  export let scale = 1.8;
  export let slope = 1;
  export let referenceX = 0.0;
  export let referenceY = 0.0;

  /** @type {String} [stroke='#000'] - The shape's fill color. This is technically optional because it comes with a default value but you'll likely want to replace it with your own color. */
  export let stroke = "#000";

  export let stroke_width = 0.5;

  function normalize(point, scale) {
    // return point;
    let norm = Math.sqrt(point.x * point.x + point.y * point.y) * scale;
    return {
      x: point.x / norm,
      y: point.y / norm
    };
  }

  $: projection_line_data = [-0.5, 0.5].map((x) => {
    let p = {x: x, y: slope * x };
    p = normalize(p, scale);
    return {
      x: p.x + referenceX,
      y: p.y + referenceY
    }
  });

  $: path =
    "M" +
    projection_line_data
      .map((d) => {
        return $xGet(d) + "," + $yGet(d);
      })
      .join("L");
</script>

<g>
  <path 
    class="path-line" 
    d={path} 
    stroke={stroke} 
    stroke-width={stroke_width}/>
</g>

<style>
  .path-line {
    fill: none;
    stroke-linejoin: round;
    stroke-linecap: round;
  }
</style>

