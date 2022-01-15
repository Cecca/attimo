# This file is akin to a Makefile for the analysis in R.

library(targets)
source("analysis/functions.R")
options(tidyverse.quiet = TRUE)
tar_option_set(packages = c(
    "tidyverse",
    "tidyjson",
    "lubridate",
    "ggrepel",
    "ggbeeswarm",
    "patchwork",
    "kableExtra"
))

# Here we have the list of targets
list(
    # Data loading ------------------------------------------------------------
    tar_target(
        data_attimo,
        load_attimo()
    ),
    tar_target(
        data_scamp,
        load_scamp()
    ),
    tar_target(
        data_ll,
        load_ll()
    ),
    tar_target(
        data_gpucluster,
        read_csv("gpucluster.csv") %>% mutate(algorithm = "scamp", hostname = "gpucluster")
    ),
    tar_target(
        data_measures,
        dataset_measures(data_attimo)
    ),
    # The motif occurences
    tar_target(
        data_motif_occurences,
        get_motif_instances(data_attimo)
    ),

    # Figure motifs ------------------------------------------------------------
    tar_target(
        imgs_motifs,
        plot_motifs(data_motif_occurences)
    ),

    # Figure scalability -------------------------------------------------------
    tar_target(
        fig_scalability_n,
        bind_rows(
            select(
                data_attimo, algorithm, hostname, dataset, prefix, threads, window,
                repetitions, motifs, delta, time_s
            ) %>% filter(motifs == 1),
            select(
                data_scamp, algorithm, hostname, dataset, prefix,
                threads, window, time_s
            ),
            select(
                data_ll, algorithm, hostname, dataset, prefix, window,
                time_s
            )
        ) %>% plot_scalability_n()
    ),
    tar_target(
        img_scalability_n,
        ggsave("imgs/scalability_n.png",
            plot = fig_scalability_n,
            width = 8,
            height = 4,
            dpi = 300
        )
    ),

    # Time comparison -----------------------------------------------
    tar_target(
        tab_time_comparison,
        do_tab_time_comparison(data_attimo, data_scamp, data_ll, "imgs/time-comparison.tex")
    ),

    # Figure motifs 10 -----------------------------------------------------
    tar_target(
        img_motifs_10,
        ggsave(
            "imgs/10-motifs.png",
            plot = plot_motifs_10_alt(data_attimo, data_scamp, data_measures),
            width = 5,
            height = 4,
            dpi = 300
        )
    ),

    # Figure profile -------------------------------------------------------
    tar_target(
        fig_profile,
        data_attimo %>%
            filter(dataset == "HumanY", motifs == 1, repetitions == 100, window == 18000) %>%
            head(1) %>%
            plot_profile()
    ),
    tar_target(
        img_profile,
        ggsave(
            "imgs/profile.png",
            plot = fig_profile,
            width = 10,
            height = 1.5,
            dpi = 300
        )
    ),

    # Figure repetitions ----------------------------------------------------
    tar_target(
        fig_repetitions,
        plot_memory_time(data_attimo)
    ),
    tar_target(
        img_repetitions,
        ggsave(
            "imgs/repetitions.png",
            plot = fig_repetitions,
            width = 5,
            height = 5,
            dpi = 300
        )
    ),

    # Figure measures -------------------------------------------------------
    tar_target(
        table_info,
        latex_info(data_measures)
    )
)