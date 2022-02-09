theme_paper <- function() {
    theme_classic() +
        theme(
            strip.background = element_blank(),
            text = element_text(family = "Helvetica")
        )
}

dataset_info <- function() {
    tribble(
        ~dataset, ~n,
        "ASTRO", 1151349,
        "ECG", 7871870,
        "EMG", 543893,
        "HumanY", 26415045,
        "GAP", 2049279,
        "freezer", 7430755,
        "Seismic", 100000000
    )
}

allowed_combinations <- tibble::tribble(
    ~dataset, ~window,
    "HumanY", 18000,
    "GAP", 600,
    "EMG", 500,
    "ECG", 1000,
    "freezer", 5000,
    "ASTRO", 100,
    "Seismic", 100
)

fix_names <- function(df) {
    df %>%
        mutate(
            path = dataset,
            dataset = if_else(str_detect(dataset, "VCAB"), "Seismic", dataset),
        )
}

reorder_datasets <- function(df) {
    df %>%
        mutate(
            dataset = factor(dataset, c("EMG", "freezer", "ASTRO", "GAP", "ECG", "HumanY", "Seismic"), ordered = T)
        )
}

add_prefix_info <- function(dat) {
    dat %>%
        mutate(
            path = if_else(is.na(path), dataset, path),
            dataset = str_remove(dataset, "data/") %>%
                str_remove("-\\d+") %>%
                str_remove(".(csv|txt)")
        ) %>%
        left_join(dataset_info()) %>%
        mutate(
            prefix = str_extract(path, "\\d+") %>% as.integer(),
            prefix = if_else(
                is.na(prefix),
                as.integer(n),
                prefix
            ),
            is_full_dataset = !str_detect(path, "-\\d+"),
            is_full_dataset = if_else(dataset == "Seismic", T, is_full_dataset)
        )
}

load_attimo <- function() {
    conn <- DBI::dbConnect(RSQLite::SQLite(), "attimo-results.db")
    table <- tbl(conn, "attimo") %>%
        filter(version == max(version, na.rm = T)) %>%
        collect() %>%
        fix_names() %>%
        mutate(
            algorithm = "attimo",
            expid = row_number()
        ) %>%
        add_prefix_info() %>%
        semi_join(allowed_combinations)
    mem <- table %>%
        as_tbl_json(json.column = "log") %>%
        gather_array() %>%
        spread_all() %>%
        filter() %>%
        group_by(expid, prefix, dataset) %>%
        summarise(max_mem_bytes = max(mem_bytes, na.rm = T)) %>%
        ungroup() %>%
        mutate(
            bytes_per_subsequence = max_mem_bytes / prefix,
            max_mem_gb = max_mem_bytes / (1024^3)
        ) %>%
        as_tibble() %>%
        select(expid, bytes_per_subsequence, max_mem_gb, max_mem_bytes)
    dists <- table %>%
        as_tbl_json(json.column = "log") %>%
        gather_array() %>%
        spread_all() %>%
        filter(msg == "motifs completed") %>%
        as_tibble() %>%
        select(expid, distances_fraction, total_distances, cnt_dist)

    phases <- table %>%
        as_tbl_json(json.column = "log") %>%
        gather_array() %>%
        spread_all() %>%
        filter(tag == "phase", msg != "fft computation") %>% # Consider the FFT as part of the input reading
        mutate(ts = lubridate::ymd_hms(ts)) %>%
        group_by(expid) %>%
        mutate(elapsed_ts = ts - min(ts)) %>%
        mutate(
            end = lead(elapsed_ts),
            phase_duration = end - elapsed_ts
        ) %>%
        filter(msg != "end") %>%
        mutate(
            phase = fct_collapse(msg,
                "preprocessing" = c("input reading", "quantization width estimation", "hash computation"),
                "motif_finding" = "tries exploration"
            )
        ) %>%
        group_by(expid, phase) %>%
        summarise(phase_duration = sum(phase_duration)) %>%
        select(expid, phase, phase_duration) %>%
        pivot_wider(names_from = phase, values_from = phase_duration)

    DBI::dbDisconnect(conn)
    inner_join(table, mem, by = "expid") %>%
        inner_join(dists, by = "expid") %>%
        inner_join(phases) %>%
        reorder_datasets()
}

load_scamp <- function() {
    conn <- DBI::dbConnect(RSQLite::SQLite(), "attimo-results.db")
    tbl <- tbl(conn, "scamp") %>%
        collect() %>%
        fix_names() %>%
        add_prefix_info() %>%
        right_join(allowed_combinations) %>%
        mutate(algorithm = "scamp") %>%
        reorder_datasets()
    DBI::dbDisconnect(conn)
    tbl
}

load_ll <- function() {
    conn <- DBI::dbConnect(RSQLite::SQLite(), "attimo-results.db")
    tbl <- tbl(conn, "ll") %>%
        collect() %>%
        fix_names() %>%
        add_prefix_info() %>%
        semi_join(allowed_combinations) %>%
        mutate(algorithm = "ll") %>%
        reorder_datasets()
    DBI::dbDisconnect(conn)
    tbl
}

get_motif_instances <- function(data_attimo) {
    data_attimo %>%
        filter(!str_detect(path, "-\\d+")) %>%
        as_tbl_json(json.column = "motif_pairs") %>%
        gather_array() %>%
        # filter(array.index == 1) %>%
        spread_all() %>%
        as_tibble() %>%
        rename(motif_idx = array.index) %>%
        distinct(path, dataset, window, motif_idx, a, b, dist)
}

get_data_depths <- function(data_attimo) {
    data_attimo %>%
        as_tbl_json(json.column = "log") %>%
        gather_array() %>%
        spread_all() %>%
        mutate(ts = lubridate::ymd_hms(ts)) %>%
        group_by(expid) %>%
        mutate(start_time = min(ts)) %>%
        filter(msg == "level completed") %>%
        as_tibble() %>%
        select(expid, dataset, window, depth, ts, start_time) %>%
        mutate(elapsed_s = (ts - start_time) / lubridate::dseconds(1)) %>%
        group_by(expid, dataset, window, depth) %>%
        slice_max(elapsed_s)
}

plot_scalability_n <- function(plotdata) {
    results_filter <- plotdata %>%
        filter(algorithm == "attimo") %>%
        select(dataset, window)

    all <- plotdata %>%
        left_join(dataset_info(), by = "dataset") %>%
        semi_join(results_filter) %>%
        mutate(
            dataset = str_remove(dataset, "data/") %>%
                str_remove("-\\d+") %>%
                str_remove(".(csv|txt)") %>%
                str_c(" (", window, ")")
        )
    last <- all %>%
        group_by(dataset, algorithm) %>%
        slice_max(prefix)
    ggplot(all, aes(prefix, time_s, color = algorithm)) +
        geom_line() +
        geom_point() +
        # geom_text_repel(
        #     mapping = aes(
        #         x = 1.1 * prefix,
        #         label = scales::number(time_s, accuracy = 1)
        #     ),
        #     data = last,
        #     hjust = 0,
        #     direction = "y"
        # ) +
        facet_wrap(vars(dataset), scales = "free") +
        scale_x_continuous(
            labels = scales::number_format(scale = 1 / 1000000),
            expand = expansion(mult = c(0, 2))
        ) +
        labs(x = "prefix (millions)", y = "time (s)") +
        theme_paper()
}

plot_profile <- function(data_attimo) {
    assertthat::assert_that(nrow(data_attimo) == 1)
    expanded <- data_attimo %>%
        tidyjson::as_tbl_json(json.column = "log") %>%
        gather_array() %>%
        spread_all() %>%
        mutate(
            elapsed_s = elapsed_ms / 1000,
            ts = lubridate::ymd_hms(ts)
        ) %>%
        mutate(elapsed_ts = ts - min(ts))

    events <- expanded %>%
        filter(tag == "phase") %>%
        filter(msg != "fft computation") %>% # Consider the FFT as part of the input reading
        mutate(
            end = lead(elapsed_ts),
            dur = end - elapsed_ts
        ) %>%
        filter(msg != "end") %>%
        mutate(labelpos = case_when(
            msg == "input reading" ~ 0,
            msg == "quantization width estimation" ~ 5,
            msg == "hash computation" ~ 15,
            msg == "tries exploration" ~ 49
        )) %>%
        mutate(msg = str_remove(msg, "quantization ")) %>%
        mutate(msg = if_else(msg == "input reading", "input", msg))

    outputs <- expanded %>%
        filter(tag == "output")

    memory <- expanded %>%
        filter(tag == "memory") %>%
        mutate(
            mem_gb = mem_bytes / (1024 * 1024 * 1024)
        )

    p <- ggplot(memory, aes(elapsed_ts, mem_gb)) +
        geom_rect(
            data = events,
            mapping = aes(xmin = elapsed_ts, xmax = end, fill = msg),
            ymin = -Inf,
            ymax = max(19, pull(summarise(memory, max(mem_gb)))),
            inherit.aes = F,
            alpha = 0.3
        ) +
        scale_fill_manual(
            values = c(
                "hash computation" = "#ffd607",
                "input" = "#bebebe",
                "width estimation" = "#e0712d",
                "tries exploration" = "#74caff"
            )
        ) +
        geom_vline(
            data = outputs,
            mapping = aes(xintercept = elapsed_ts),
            linetype = "dashed"
        ) +
        geom_step(color = "#d70303") +
        geom_text(
            data = events,
            mapping = aes(
                # x = labelpos,
                x = elapsed_ts,
                label = msg,
            ),
            y = -5.3,
            # y = 0.2,
            size = 3,
            angle = 0,
            vjust = 1,
            hjust = 0
        ) +
        scale_y_continuous(limits = c(0, NA)) +
        geom_text(
            data = events,
            mapping = aes(
                x = elapsed_ts + (end - elapsed_ts) / 2,
                y = 23,
                label = scales::number(as.double(dur), accuracy = 0.1, suffix = " s")
            ),
            size = 3,
            vjust = 1
        ) +
        geom_segment(
            data = events,
            mapping = aes(
                x = elapsed_ts + 0.3,
                xend = end - 0.3,
                y = 20,
                yend = 20
            ),
            size = 0.5,
            color = "gray"
        ) +
        scale_x_continuous(
            breaks = c(0, events %>% pull(end) %>% as.double()),
            labels = scales::number_format(accuracy = 0.1)
        ) +
        labs(
            x = "elapsed time (s)",
            y = "memory (Gb)"
        ) +
        coord_cartesian(clip = "off") +
        theme_classic() +
        theme(legend.position = "none")

    p
}

plot_measures <- function(data_measures) {
    data_measures <- data_measures %>%
        mutate(dataset = str_c(dataset, " (", window, ")"))
    motif <- data_measures %>%
        group_by(dataset, window) %>%
        slice(1)

    ggplot(data_measures) +
        geom_density(aes(rc1), fill = "lightgray", alpha = 0.5) +
        scale_x_log10(labels = scales::number_format(accuracy = 1)) +
        labs(
            x = "Nearest neighbor relative Contrast",
            y = "Density"
        ) +
        facet_wrap(vars(dataset), scales = "free", nrow = 1) +
        theme_paper()
}

plot_motifs <- function(data_motif_occurences) {
    dat <- data_motif_occurences
    # dat <- data_motif_occurences %>%
    #     arrange(dataset, window) %>%
    #     group_by(dataset, path, window) %>%
    #     slice(1) %>%
    #     ungroup()
    for (i in seq_len(nrow(dat))) {
        row <- as.list(dat[i, ])
        print(str(row))
        corr <- 1 - row$dist^2 / (2 * row$window)
        ts <- read_csv(row$path, col_names = "y") %>% pull()
        a <- ts[row$a:(row$a + row$window - 1)]
        b <- ts[row$b:(row$b + row$window - 1)]
        w <- row$window
        idx <- row$motif_idx
        plotdata <- tibble(
            a = a,
            b = b,
            xs = 1:row$window
        )

        ggplot(plotdata, aes(xs)) +
            geom_line(aes(y = a), color = "#f78a36") +
            geom_line(aes(y = b), color = "#1788f9") +
            labs(
                title = str_c(
                    row$dataset, " distance ", row$dist,
                    " correlation ", corr
                )
            ) +
            theme_paper() +
            theme(
                plot.title = element_text(size = 8),
                axis.line.y = element_blank(),
                axis.ticks.y = element_blank(),
                axis.text.y = element_blank(),
                axis.title = element_blank()
            )
        fname <- str_c("imgs/motifs-", row$dataset, "-w", w, "-m", idx, ".png")
        ggsave(fname, width = 9, height = 1.2, dpi = 300)
    }
}

latex_info <- function(data_motif_measures) {
    inner_join(ungroup(data_motif_measures), dataset_info()) %>%
        filter(motif_idx %in% c(1, 10)) %>%
        mutate(label = str_c("$RC_{", motif_idx, "}$")) %>%
        select(dataset, n, window, label, rc1) %>%
        pivot_wider(names_from = label, values_from = rc1) %>%
        arrange(`$RC_{10}$`) %>%
        mutate(
            across(matches("RC"), scales::number_format(accuracy = 0.01, big.mark = "\\\\,")),
            n = scales::number(n, big.mark = "\\\\,"),
            window = scales::number(window, big.mark = "\\\\,")
        ) %>%
        kbl(format = "latex", booktabs = T, linesep = "", align = "lrrrr", escape = F) %>%
        write_file("imgs/dataset-info.tex")
}

do_tab_time_comparison <- function(data_attimo, data_scamp, data_ll, data_gpucluster, file_out) {
    bind_rows(
        data_attimo %>%
            filter(repetitions == 200) %>%
            select(
                algorithm, hostname, dataset, is_full_dataset, threads, window,
                repetitions, motifs, delta, time_s, distances_fraction
            ) %>% filter(motifs == 1),
        select(
            data_scamp, algorithm, hostname, dataset, is_full_dataset,
            threads, window, time_s
        ),
        select(
            data_ll, algorithm, hostname, dataset, is_full_dataset, window,
            time_s
        ),
        data_gpucluster %>% mutate(is_full_dataset = T)
    ) %>%
        filter(is_full_dataset) %>%
        # mutate(dataset = str_c(dataset, " (", window, ")")) %>%
        mutate(
            time_s = scales::number(time_s, accuracy = 0.1),
            distances_fraction = scales::scientific(
                distances_fraction,
                prefix = "$",
                suffix = "}$",
                digits = 2
            ) %>% str_replace("e", "\\\\cdot 10^{"),
            time_s = if_else(algorithm == "attimo",
                str_c(time_s, " (", distances_fraction, ")"),
                time_s
            )
        ) %>%
        select(dataset, algorithm, time_s) %>%
        pivot_wider(names_from = algorithm, values_from = time_s) %>%
        reorder_datasets() %>%
        arrange(dataset) %>%
        kbl(
            format = "latex", booktabs = T, linesep = "", align = "lrrr",
            escape = F,
            caption = "Time to find the top motif at different window lengths. For \\our,
            the number in parentheses reports the fraction of distance computations over ${n \\choose 2}$
            performed to find the solution."
        ) %>%
        kable_styling() %>%
        add_header_above(c(" " = 1, "Time (s)" = 3)) %>%
        write_file(file_out)
}

compute_distance_distibution <- function(data_attimo) {
    do_compute <- function(dataset, path, window) {
        out <- paste0(path, ".", window, ".dists")
        if (!file.exists(out) || !file.exists(paste0(out, ".gz"))) {
            print(paste("Compute", out))
            system2(
                "target/release/examples/distances",
                c(
                    "--window", window,
                    "--path", path,
                    "--output", out,
                    "--samples", "10000000"
                )
            )
        }
        if (file.exists(paste0(out, ".gz"))) {
            out <- paste0(out, ".gz")
        }
        dat <- readr::read_csv(out) %>%
            mutate(dataset = dataset) %>%
            rename(window = w)
        dat
    }

    data_attimo %>%
        distinct(path, dataset, window) %>%
        rowwise() %>%
        summarise(do_compute(dataset, path, window))
}

compute_measures <- function(path, window, idxs) {
    out <- paste0(path, ".", window, ".measures")
    if (!file.exists(out) || !file.exists(paste0(out, ".gz"))) {
        system2(
            "target/release/examples/measures",
            c(
                "--window", window,
                "--path", path,
                "--output", out,
                idxs
            )
        )
    }
    if (file.exists(paste0(out, ".gz"))) {
        out <- paste0(out, ".gz")
    }
    dat <- readr::read_csv(out) %>%
        mutate(
            motif_idx = row_number(),
            nn_corr = 1 - nn^2 / (2 * window)
        )
    dat
}

dataset_measures <- function(data_attimo, data_distances) {
    avg_dists <- data_distances %>%
        group_by(dataset, window) %>%
        summarise(avg_distance = mean(distance))

    data_attimo %>%
        filter(motifs == 10) %>%
        group_by(path, dataset, window) %>%
        slice(1) %>%
        as_tbl_json(json.column = "motif_pairs") %>%
        select(path, dataset, window) %>%
        gather_array() %>%
        spread_all() %>%
        rename(motif_idx = array.index) %>%
        inner_join(avg_dists) %>%
        mutate(rc1 = avg_distance / dist)

    # data_attimo %>%
    #     filter(motifs == 10) %>%
    #     group_by(path, dataset, window) %>%
    #     slice(1) %>%
    #     as_tbl_json(json.column = "motif_pairs") %>%
    #     select(path, dataset, window) %>%
    #     gather_array() %>%
    #     spread_all() %>%
    #     group_by(path, dataset, window) %>%
    #     summarise(idxs = list(a)) %>%
    #     rowwise() %>%
    #     summarise(compute_measures(path, window, idxs))
}

plot_memory_time <- function(data_attimo) {
    data_attimo %>%
        filter(motifs == 10) %>%
        group_by(dataset, window) %>%
        mutate(
            labelpos = time_s + 0.1 * max(time_s),
            tickpos = -0.1 * max(time_s),
            mempos = -0.6 * max(time_s),
        ) %>%
        ggplot(aes(repetitions, time_s)) +
        geom_area(aes(y = preprocessing), fill = "#f78a36", alpha = 0.4) +
        geom_ribbon(aes(ymin = preprocessing, ymax = time_s), fill = "#74caff", alpha = 0.4) +
        geom_line() +
        geom_point() +
        labs(
            x = "repetitions",
            y = "total time (s)"
        ) +
        facet_wrap(vars(str_c(dataset, " (", window, ")")), ncol = 3, scales = "free") +
        coord_cartesian(clip = "off") +
        theme_paper() +
        theme(
            # axis.line.y = element_blank(),
            # axis.text.y = element_blank(),
            # axis.ticks.y = element_blank(),
            # axis.line.x = element_blank(),
            # axis.text.x = element_blank(),
            # axis.ticks.x = element_blank(),
            panel.spacing = unit(5, "mm")
        )
}

plot_motifs_10 <- function(data_attimo, data_scamp, data_measures) {
    data_attimo %>%
        inner_join(select(filter(data_scamp, is_full_dataset), dataset, window, time_scamp_s = time_s)) %>%
        filter(motifs == 10) %>%
        as_tbl_json(json.column = "motif_pairs") %>%
        gather_array() %>%
        spread_all() %>%
        rename(motif_idx = array.index) %>%
        as_tibble() %>%
        inner_join(select(data_measures, dataset, window, motif_idx, rc1, nn)) %>%
        ggplot(aes(motif_idx, confirmation_time)) +
        geom_point() +
        geom_segment(aes(xend = motif_idx, yend = as.double(preprocessing))) +
        geom_area(aes(y = preprocessing), fill = "#f78a36") +
        scale_x_continuous(breaks = c(2, 4, 6, 8, 10)) +
        facet_wrap(vars(dataset), scales = "free") +
        labs(
            x = "motif index",
            y = "time (s)"
        ) +
        theme_paper() +
        theme(
            panel.spacing = unit(5, "mm")
        )
}

plot_motifs_10_alt <- function(data_attimo, data_scamp, data_measures) {
    bars <- data_attimo %>%
        inner_join(select(filter(data_scamp, is_full_dataset), dataset, window, time_scamp_s = time_s)) %>%
        filter(motifs == 10, repetitions == 200) %>%
        as_tbl_json(json.column = "motif_pairs") %>%
        gather_array() %>%
        spread_all() %>%
        rename(motif_idx = array.index) %>%
        as_tibble() %>%
        inner_join(select(data_measures, dataset, window, motif_idx, rc1)) %>%
        mutate(
            confirmation_time = as.numeric(confirmation_time),
            preprocessing = as.numeric(preprocessing)
        ) %>% # select(dataset, confirmation_time, preprocessing) %>% view()
        ggplot(aes(x = factor(0), y = confirmation_time)) +
        geom_col(
            mapping = aes(y = preprocessing),
            data = function(d) {
                group_by(d, dataset) %>% slice(1)
            },
            fill = "#f78a36",
            width = 0.3
        ) +
        geom_quasirandom(size = 1, width = 0.3) +
        facet_wrap(vars(dataset), ncol = 1, scales = "free", strip.position = "left") +
        labs(
            # x = "motif index",
            y = "time (s)"
        ) +
        coord_flip() +
        theme_paper() +
        theme(
            axis.line.y = element_blank(),
            axis.text.y = element_blank(),
            axis.title.y = element_blank(),
            axis.ticks.y = element_blank(),
            panel.grid.major.y = element_line(color = "gray"),
            # axis.line.x = element_blank(),
            # axis.text.x = element_blank(),
            # axis.ticks.x = element_blank(),
            panel.spacing = unit(0, "mm")
        )

    scamp <- data_scamp %>%
        filter(is_full_dataset) %>%
        ggplot(aes(x = factor(0), y = factor(0))) +
        geom_text(
            aes(label = scales::number(time_s, accuracy = 0.1, suffix = " s")),
            size = 3,
            hjust = 0.5
        ) +
        facet_wrap(vars(dataset), ncol = 1) +
        labs(title = "SCAMP") +
        theme_void() +
        theme(
            strip.background = element_blank(),
            strip.text = element_blank(),
            plot.title = element_text(hjust = 0.5, size = 9)
            # plot.margin = unit(0, "mm")
        )

    bars + scamp + plot_layout(widths = c(5, 1))
}

plot_motifs_10_alt2 <- function(data_attimo, data_scamp, data_depths, data_measures) {
    data_depths <- semi_join(data_depths, data_attimo)
    bars <- data_attimo %>%
        left_join(select(filter(data_scamp, is_full_dataset), dataset, window, time_scamp_s = time_s)) %>%
        filter(motifs == 10) %>%
        filter((repetitions == 200) | (dataset == "Seismic")) %>%
        # print()
        as_tbl_json(json.column = "motif_pairs") %>%
        gather_array() %>%
        spread_all() %>%
        rename(motif_idx = array.index) %>%
        as_tibble() %>%
        mutate(
            time_scamp_s_hline = if_else(
                (time_scamp_s < 1000) & (motif_idx == 1),
                time_scamp_s,
                as.double(NA)
            ),
            time_scamp_s_label = if_else(
                time_scamp_s >= 1000 & (motif_idx == 1),
                time_scamp_s,
                as.double(NA)
            )
        ) %>%
        mutate(
            confirmation_time = as.numeric(confirmation_time),
            preprocessing = as.numeric(preprocessing)
        ) %>% # select(dataset, confirmation_time, preprocessing) %>% view()
        ggplot(aes(y = confirmation_time, x = dist)) +
        geom_rect(
            mapping = aes(ymax = preprocessing, ymin = 0),
            xmin = -Inf,
            xmax = Inf,
            fill = "#f78a36",
            alpha = 0.3,
            data = function(d) {
                group_by(d, dataset) %>% slice(1)
            },
        ) +
        geom_point() +
        geom_hline(
            aes(yintercept = time_scamp_s_hline),
            linetype = "dotted",
        ) +
        geom_text(
            aes(
                label = scales::number(
                    time_scamp_s_hline,
                    accuracy = 1,
                    prefix = "SCAMP: ",
                    suffix = " s"
                ),
                x = dist,
                y = time_scamp_s_hline + 10
            ),
            hjust = 0,
            vjust = 0
        ) +
        geom_text(
            aes(
                label = scales::number(
                    time_scamp_s_label,
                    accuracy = 1,
                    prefix = "SCAMP: ",
                    suffix = " s →"
                ),
                x = dist# * 1.2
            ),
            y = 500,
            hjust = 1,
            vjust = 0
        ) +
        scale_y_continuous(limits = c(0, NA)) +
        scale_x_continuous(limits = c(NA, NA), position = "top") +
        facet_wrap(vars(dataset), ncol = 1, scales = "free_y", strip.position = "left") +
        labs(
            x = "distance",
            y = "time (s)"
        ) +
        coord_flip() +
        theme_paper() +
        theme(
            # axis.line.y = element_blank(),
            # axis.text.y = element_blank(),
            # axis.title.y = element_blank(),
            # axis.ticks.y = element_blank(),
            # panel.grid.major.y = element_line(color = "gray"),
            # axis.line.x = element_blank(),
            # axis.text.x = element_blank(),
            # axis.ticks.x = element_blank(),
            panel.spacing = unit(5, "mm")
        )

    bars
}


plot_distributions <- function(data_measures, data_distances) {
    data_distances %>%
        group_by(dataset) %>%
        slice_min(distance, n = 100) %>%
        ggplot(aes(distance)) +
        geom_density() +
        geom_vline(
            data = data_measures,
            mapping = aes(xintercept = dist)
        ) +
        # geom_rug(length = unit(0.3, "npc"), color = "red") +
        facet_wrap(
            vars(dataset),
            scales = "free",
            ncol = 1
        ) +
        theme_paper() +
        theme(
            strip.text = element_text(hjust = 0)
        )
}