<script>
  import { getContext } from "svelte";

  const { data, xGet, yGet, x, y } = getContext("LayerCake");

  /** @type {String} [stroke='#000'] - The shape's fill color. This is technically optional because it comes with a default value but you'll likely want to replace it with your own color. */
  export let stroke = "#000";

  export let stroke_width = 0.5;

  export let minX = undefined;
  export let maxX = undefined;

  $: path =
    "M" +
    $data
      .filter(d => (minX == undefined || minX <= $x(d)) && (maxX == undefined ||$x(d) <= maxX))
      .map((d) => {
        return $xGet(d) + "," + $yGet(d);
      })
      .join("L");

</script>

<path class="path-line" d={path} stroke={stroke} stroke-width={stroke_width}/>

<style>
  .path-line {
    fill: none;
    stroke-linejoin: round;
    stroke-linecap: round;
  }
</style>
