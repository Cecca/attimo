test:
  cargo nextest run --release

release-build:
  cargo build --release

hyperfine: release-build
  hyperfine \
    'target/release/attimo data/ECG.csv.gz --failure-probability 0.01 --repetitions 8192 --window 1000 --motiflets 10'

profile: release-build
  perf record -F 999 --call-graph dwarf -g target/release/attimo \
    data/GAP.csv.gz \
    --failure-probability 0.01 \
    --repetitions 8192 \
    --window 600 \
    --motiflets 3

live-profile:
  perf record -F 999 -g -p $(pgrep attimo)

speedscope-perf:
  perf script -i perf.data | speedscope -

# sync data from the remote machine where experiments have been run
sync:
  rsync --progress gpucluster:attimo-rs/attimo-results.db .
  rm _targets/objects/data_{attimo,prescrimp,rproj,scamp,scamp_gpu,ll,scalability}

# run the R code for the analysis
analysis:
  R -e 'targets::tar_make()'

# copy the results of the analysis to the paper's figure directory
copy: analysis
  cp imgs/time-comparison.tex ../Attimo-paper/figs/
  cp imgs/repetitions.png ../Attimo-paper/figs/
  cp imgs/dataset-info.tex ../Attimo-paper/figs/
  cp imgs/10-motifs.png ../Attimo-paper/figs/
  cp imgs/scalability_n.png ../Attimo-paper/figs/
