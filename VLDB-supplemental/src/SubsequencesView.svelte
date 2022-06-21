<script>
  import Line from './Line.svelte';
  import { Svg, LayerCake } from 'layercake';

  export let subsequences;
</script>

<figure class="l-body subts-outer-container">
  {#each subsequences as sub}
    <div class="subts-container" 
      on:mouseover="{() => global_highlight = sub[0].x}"
      on:mouseout="{() => global_highlight = undefined}"
      on:focus="{() => global_highlight = sub[0].x}"
    >
      <LayerCake
        padding={{ top: 2, right: 16, bottom: 2, left: 16 }}
        x="x"
        y="y"
        data={sub}
      >
        <Svg>
          <Line></Line>
        </Svg>
      </LayerCake>
    </div>
  {/each}
</figure>

