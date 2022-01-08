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

load_attimo <- function() {
    conn <- DBI::dbConnect(RSQLite::SQLite(), "attimo-results.db")
    table <- tbl(conn, "attimo") %>%
        filter(version == max(version, na.rm = T)) %>%
        collect() %>%
        mutate(
            algorithm = "attimo",
            path = dataset,
            dataset = str_remove(dataset, "data/") %>%
                str_remove("-\\d+") %>%
                str_remove(".(csv|txt)"),
            expid = row_number()
        ) %>%
        left_join(dataset_info()) %>%
        mutate(
            prefix = str_extract(path, "\\d+") %>% as.integer(),
            prefix = if_else(
                is.na(prefix),
                as.integer(n),
                prefix
            )
        )
    mem <- table %>%
        as_tbl_json(json.column = "log") %>%
        gather_array() %>%
        spread_all() %>%
        filter(tag == "memory") %>%
        group_by(expid, prefix, dataset) %>%
        summarise(mem_bytes = max(mem_bytes, na.rm = T)) %>%
        ungroup() %>%
        mutate(
            bytes_per_subsequence = mem_bytes / prefix,
            mem_gb = mem_bytes / (1024^3)
        ) %>%
        as_tibble() %>%
        select(expid, bytes_per_subsequence, mem_gb, mem_bytes)
    DBI::dbDisconnect(conn)
    inner_join(table, mem, by = "expid")
}

load_scamp <- function() {
    conn <- DBI::dbConnect(RSQLite::SQLite(), "attimo-results.db")
    tbl <- tbl(conn, "scamp") %>%
        collect() %>%
        left_join(dataset_info()) %>%
        mutate(
            algorithm = "scamp",
            path = dataset,
            dataset = str_remove(dataset, "data/") %>%
                str_remove("-\\d+") %>%
                str_remove(".(csv|txt)"),
            prefix = str_extract(dataset, "\\d+") %>% as.integer(),
            prefix = if_else(
                is.na(prefix),
                as.integer(n),
                prefix
            )
        )
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
    plotdata %>%
        left_join(dataset_info(), by = "dataset") %>%
        mutate(
            dataset = str_remove(dataset, "data/") %>%
                str_remove("-\\d+") %>%
                str_remove(".(csv|txt)") %>%
                str_c(" (", window, ")")
        ) %>%
        ggplot(aes(prefix, time_s,
            color = algorithm
        )) +
        geom_line() +
        geom_point() +
        facet_wrap(vars(dataset), scales = "free") +
        scale_x_continuous(
            labels = scales::number_format(scale = 1 / 1000000)
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
        mutate(ts = lubridate::ymd_hms(ts))

    events <- expanded %>%
        filter(tag == "profiling") %>%
        group_by(msg) %>%
        slice_min(ts)
    memory <- expanded %>%
        filter(tag == "memory") %>%
        mutate(
            mem_gb = mem_bytes / (1024 * 1024 * 1024)
        )

    ggplot(memory, aes(ts, mem_gb)) +
        geom_vline(
            data = events,
            mapping = aes(xintercept = ts),
            linetype = "dashed"
        ) +
        geom_line(color = "#d70303") +
        geom_text_repel(
            data = events,
            mapping = aes(x = ts, label = msg),
            y = 80,
            angle = 90,
            direction = "x",
            linetype = "dashed"
        ) +
        theme_classic()
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