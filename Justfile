all: copy

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