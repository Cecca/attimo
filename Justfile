all: copy

# sync data from the remote machine where experiments have been run
sync:
        rsync --progress gpucluster:attimo-rs/attimo-results.db .
        rm _targets/objects/data_{attimo,prescrimp,scamp,ll}

# run the R code for the analysis
analysis:
        R -e 'targets::tar_make()'

# copy the results of the analysis to the paper's figure directory
copy: analysis
        cp imgs/time-comparison.tex ../Attimo-paper/figs/
        cp imgs/repetitions.png ../Attimo-paper/figs/
        cp imgs/dataset-info.tex ../Attimo-paper/figs/
