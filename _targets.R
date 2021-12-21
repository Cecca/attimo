# This file is akin to a Makefile for the analysis in R.

library(targets)
source("analysis/functions.R")
options(tidyverse.quiet = TRUE)
tar_option_set(packages = c(
    "tidyverse",
    "tidyjson",
    "lubridate",
    "ggrepel"
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
        data_gpucluster,
        read_csv("gpucluster.csv") %>% mutate(algorithm = "scamp", hostname = "gpucluster")
    ),

    # Figure scalability -------------------------------------------------------
    tar_target(
        fig_scalability_n,
        bind_rows(
            select(
                data_attimo, algorithm, hostname, dataset, threads, window,
                repetitions, motifs, delta, time_s
            ) %>% filter(motifs == 1),
            select(
                data_scamp, algorithm, hostname, dataset,
                threads, window, time_s
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

    # Figure profile -------------------------------------------------------
    tar_target(
        fig_profile,
        data_attimo %>%
            filter(dataset == "data/HumanY.txt", motifs == 1) %>%
            plot_profile()
    ),
    tar_target(
        img_profile,
        ggsave(
            "imgs/profile.png",
            plot = fig_profile,
            width = 8,
            height = 4,
            dpi = 300
        )
    )
)