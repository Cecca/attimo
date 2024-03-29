# This file is akin to a Makefile for the analysis in R.

# the default delta value to be considered
delta_val = 0.01

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
    "latex2exp",
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
        data_scamp_gpu,
        load_scamp_gpu()
    ),
    tar_target(
        data_prescrimp,
        load_prescrimp()
    ),
    tar_target(
        data_ll,
        load_ll()
    ),
    tar_target(
        data_rproj,
        load_rproj()
    ),
    tar_target(
        data_scalability,
        load_scalability()
        # bind_rows(
        #     data_attimo %>%
        #         filter(!is.na(perc_size)) %>%
        #         filter(delta == 0.01) %>%
        #         select(algorithm, dataset, perc_size, time_s),
        #     data_scamp %>%
        #         filter(!is.na(perc_size)) %>%
        #         select(algorithm, dataset, perc_size, time_s),
        #     read_csv("scamp-gpu-scalability.csv", col_names = c("dataset", "window", "time_s")) %>%
        #         fix_names() %>%
        #         add_prefix_info() %>%
        #         mutate(algorithm = "scamp-gpu", hostname = "gpucluster") %>%
        #         select(algorithm, dataset, perc_size, time_s)
        # )
    ),
    tar_target(
        # A model for the performance of SCAMP, fitting a quadratic polynomial
        scamp_model,
        lm(time_s ~ poly(n, 2), drop_na(select(data_scamp, n, time_s)))
    ),
    tar_target(
        data_scamp_gpu_scalability,
        read_csv("scamp-gpu-scalability.csv", col_names = c("dataset", "window", "time_s")) %>%
            fix_names() %>%
            add_prefix_info() %>%
            mutate(algorithm = "scamp-gpu", hostname = "gpucluster") %>%
            select(algorithm, dataset, perc_size, time_s) %>%
            inner_join(dataset_info()) %>%
            mutate(scaled_n = perc_size * n)
    ),
    tar_target(
        # A model for the performance of SCAMP-gpu, fitting a quadratic polynomial
        scamp_gpu_model,
        lm(time_s ~ poly(scaled_n, 2), data_scamp_gpu_scalability)
    ),
    tar_target(
        # DEPRECATED
        data_gpucluster,
        read_csv("gpucluster.csv") %>%
            group_by(dataset, w) %>%
            summarise(time_s = mean(time_s)) %>%
            fix_names() %>%
            add_prefix_info() %>%
            mutate(algorithm = "scamp-gpu", hostname = "gpucluster") %>%
            select(dataset, w, algorithm, time_s)
    ),
    tar_target(
        data_recall,
        data_attimo %>% 
            filter((repetitions == 400) | (dataset == "Seismic"), motifs == 10) %>% 
            add_recall()
    ),
    tar_target(
        data_measures,
        dataset_measures(data_attimo)
    ),
    # tar_target(
    #     data_distances,
    #     compute_distance_distibution(data_attimo)
    # ),
    # The motif occurences
    tar_target(
        data_motif_occurences,
        get_motif_instances(data_attimo)
    ),
    tar_target(
        data_comparison,
        get_data_comparison(
            filter(data_attimo, delta == delta_val),
            data_scamp,
            data_ll,
            # data_gpucluster,
            data_scamp_gpu,
            data_prescrimp,
            data_rproj
        )
    ),

    # Figure motifs ------------------------------------------------------------
    # tar_target(
    #     imgs_motifs,
    #     plot_motifs(data_motif_occurences)
    # ),

    # Figure scalability -------------------------------------------------------
    tar_target(
        img_scalability_n,
        ggsave("imgs/scalability_n.png",
            plot = plot_scalability_n_alt(data_scalability),
            width = 5,
            height = 3,
            dpi = 300
        )
    ),
    tar_target(
        img_scalability_n_linear,
        ggsave("imgs/scalability_n_linear.png",
            plot = plot_scalability_n_linear(data_scalability),
            width = 5,
            height = 3,
            dpi = 300
        )
    ),

    # Time comparison -----------------------------------------------
    tar_target(
        tab_time_comparison,
        do_tab_time_comparison(data_comparison, "imgs/time-comparison.tex")
    ),
    # tar_target(
    #     tab_time_comparison_normalized,
    #     do_tab_time_comparison_normalized(data_comparison)
    # ),

    # Figure motifs 10 -----------------------------------------------------
    tar_target(
        img_motifs_10,
        ggsave(
            "imgs/10-motifs.png",
            plot = plot_motifs_10_alt3(filter(data_attimo, delta == delta_val), 
                                       data_scamp, 
                                       data_scamp_gpu),
            width = 5,
            height = 5,
            dpi = 300
        )
    ),
    # Figure motifs 10, simplified -----------------------------------------------
    tar_target(
        img_motifs_10_simple,
        ggsave(
            "imgs/10-motifs-simple.png",
            plot = plot_motifs_10_simple(filter(data_attimo, delta == delta_val), 
                                       data_scamp, 
                                       data_scamp_gpu),
            width = 5,
            height = 6,
            dpi = 300
        )
    ),

    # Figure memory -----------------------------------------------------
    tar_target(
        tab_mem,
        table_mem(filter(data_attimo, delta == delta_val))
    ),
    tar_target(
        img_mem,
        ggsave(
            "imgs/memory.png",
            plot = plot_mem(filter(data_attimo, delta == delta_val)),
            width = 5,
            height = 2,
            dpi = 300
        )
    ),

    # Figure repetitions ----------------------------------------------------
    tar_target(
        fig_repetitions,
        plot_memory_time(filter(data_attimo, delta == delta_val))
    ),
    tar_target(
        img_repetitions,
        ggsave(
            "imgs/repetitions.png",
            plot = fig_repetitions,
            # width = 10,
            width = 4.5,
            height = 3,
            dpi = 300
        )
    ),

    # Figure measures -------------------------------------------------------
    tar_target(
        table_info,
        latex_info(data_measures)
    )
)
