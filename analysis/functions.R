theme_paper <- function() {
    theme_classic() +
        theme(strip.background = element_blank())
}

dataset_info <- function() {
    tribble(
        ~dataset, ~n,
        "ASTRO", 1151349,
        "ECG", 7871870,
        "EMG", 543893,
        "HumanY", 26415045,
        "GAP", 2049279,
        "freezer", 7430755
    )
}

add_prefix_info <- function(dat) {
    dat %>%
        mutate(
            path = dataset,
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
            is_full_dataset = !str_detect(path, "-\\d+")
        )
}

load_attimo <- function() {
    conn <- DBI::dbConnect(RSQLite::SQLite(), "attimo-results.db")
    table <- tbl(conn, "attimo") %>%
        filter(version == max(version, na.rm = T)) %>%
        collect() %>%
        mutate(
            algorithm = "attimo",
            expid = row_number()
        ) %>%
        add_prefix_info()
    mem <- table %>%
        as_tbl_json(json.column = "log") %>%
        gather_array() %>%
        spread_all() %>%
        filter(tag == "memory") %>%
        group_by(expid, prefix, dataset) %>%
        summarise(max_mem_bytes = max(mem_bytes, na.rm = T)) %>%
        ungroup() %>%
        mutate(
            bytes_per_subsequence = max_mem_bytes / prefix,
            max_mem_gb = max_mem_bytes / (1024^3)
        ) %>%
        as_tibble() %>%
        select(expid, bytes_per_subsequence, max_mem_gb, max_mem_bytes)
    DBI::dbDisconnect(conn)
    inner_join(table, mem, by = "expid")
}

load_scamp <- function() {
    conn <- DBI::dbConnect(RSQLite::SQLite(), "attimo-results.db")
    tbl <- tbl(conn, "scamp") %>%
        collect() %>%
        add_prefix_info() %>%
        mutate(algorithm = "scamp")
    DBI::dbDisconnect(conn)
    tbl
}

load_ll <- function() {
    conn <- DBI::dbConnect(RSQLite::SQLite(), "attimo-results.db")
    tbl <- tbl(conn, "ll") %>%
        collect() %>%
        add_prefix_info() %>%
        mutate(algorithm = "ll")
    DBI::dbDisconnect(conn)
    tbl
}


load_measures <- function() {
    bind_rows(
        read_csv("data/ASTRO.csv.measures") %>% mutate(dataset = "ASTRO"),
        read_csv("data/ECG.csv.measures") %>% mutate(dataset = "ECG"),
        read_csv("data/EMG.csv.measures") %>% mutate(dataset = "EMG"),
        read_csv("data/GAP.csv.measures") %>% mutate(dataset = "GAP"),
        read_csv("data/HumanY.txt.measures") %>% mutate(dataset = "HumanY"),
        read_csv("data/freezer.txt.measures") %>% mutate(dataset = "freezer")
    ) %>% rename(window = w)
}

get_motif_instances <- function(data_attimo) {
    data_attimo %>%
        filter(!str_detect(path, "-\\d+")) %>%
        as_tbl_json(json.column = "motif_pairs") %>%
        gather_array() %>%
        filter(array.index == 1) %>%
        spread_all() %>%
        as_tibble() %>%
        distinct(path, dataset, window, a, b, dist)
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
        mutate(end = lead(elapsed_ts)) %>%
        filter(msg != "end") %>%
        mutate(labelpos = case_when(
            msg == "input reading" ~ -7,
            msg == "quantization width estimation" ~ 5,
            msg == "hash computation" ~ 16,
            msg == "tries exploration" ~ 48
        )) %>%
        mutate(msg = str_remove(msg, "quantization ")) %>%
        mutate(msg = str_replace(msg, " ", "\n"))

    outputs <- expanded %>%
        filter(tag == "output")

    outputs %>%
        print()

    memory <- expanded %>%
        filter(tag == "memory") %>%
        mutate(
            mem_gb = mem_bytes / (1024 * 1024 * 1024)
        )

    ggplot(memory, aes(elapsed_ts, mem_gb)) +
        geom_rect(
            data = events,
            mapping = aes(xmin = elapsed_ts, xmax = end, fill = msg),
            ymin = -Inf,
            ymax = Inf,
            inherit.aes = F,
            alpha = 0.3
        ) +
        scale_fill_manual(
            values = c(
                "hash\ncomputation" = "#ffd607",
                "input\nreading" = "#bebebe",
                "width\nestimation" = "#e0712d",
                "tries\nexploration" = "#74caff"
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
                x = labelpos,
                label = msg
            ),
            y = 0.2,
            angle = 90,
            vjust = 1,
            hjust = 0,
            direction = "x"
        ) +
        labs(
            x = "elapsed time (s)",
            y = "memory (Gb)"
        ) +
        theme_classic() +
        theme(legend.position = "none")
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
    dat <- data_motif_occurences %>%
        arrange(dataset, window) %>%
        group_by(dataset, path, window) %>%
        slice(1) %>%
        ungroup()
    for (i in seq_len(nrow(dat))) {
        # print(paste("Plotting motifs for", row$dataset, "with window", row$window))
        row <- as.list(dat[i, ])
        print(str(row))
        corr <- 1 - row$dist^2 / (2 * row$window)
        ts <- read_csv(row$path, col_names = "y") %>% pull()
        a <- ts[row$a:(row$a + row$window - 1)]
        b <- ts[row$b:(row$b + row$window - 1)]
        # a <- (a - mean(a)) / sd(a)
        # b <- (b - mean(b)) / sd(b)
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
        fname <- str_c("imgs/motifs-", row$dataset, "-", row$window, ".png")
        ggsave(fname, width = 9, height = 1.2, dpi = 300)
    }
}

latex_info <- function(data_motif_measures) {
    inner_join(data_motif_measures, dataset_info()) %>%
        select(dataset, n, window, rc1) %>%
        arrange(n) %>%
        mutate_if(is.numeric, scales::number_format(big.mark = "\\\\,")) %>%
        kbl(format = "latex", booktabs = T, align = "lrrr", escape = F) %>%
        write_file("imgs/dataset-info.tex")
}

do_tab_time_comparison <- function(data_attimo, data_scamp, data_ll, file_out) {
    bind_rows(
        select(
            data_attimo,
            algorithm, hostname, dataset, is_full_dataset, threads, window,
            repetitions, motifs, delta, time_s
        ) %>% filter(motifs == 1),
        select(
            data_scamp, algorithm, hostname, dataset, is_full_dataset,
            threads, window, time_s
        ),
        select(
            data_ll, algorithm, hostname, dataset, is_full_dataset, window,
            time_s
        )
    ) %>%
        filter(is_full_dataset) %>%
        mutate(dataset = str_c(dataset, " (", window, ")")) %>%
        select(dataset, algorithm, time_s) %>%
        pivot_wider(names_from = algorithm, values_from = time_s) %>%
        arrange(dataset) %>%
        # drop_na() %>%
        mutate_if(is.numeric, scales::number_format(accuracy = 0.1)) %T>%
        print() %>%
        kbl(format = "latex", booktabs = T, linesep = "", caption = "Time to find the top motif at different window lengths.") %>%
        kable_styling() %>%
        add_header_above(c("dataset (window) " = 1, "Time (s)" = 3)) %>%
        write_file(file_out)
}