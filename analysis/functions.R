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
        "GAP", 2049279
    )
}

load_attimo <- function() {
    conn <- DBI::dbConnect(RSQLite::SQLite(), "attimo-results.db")
    tbl <- tbl(conn, "attimo") %>%
        filter(version == max(version)) %>%
        collect() %>%
        mutate(
            algorithm = "attimo",
            dataset = str_remove(dataset, "data/") %>%
                str_remove("-\\d+") %>%
                str_remove(".(csv|txt)")
        )
    DBI::dbDisconnect(conn)
    tbl
}

load_scamp <- function() {
    conn <- DBI::dbConnect(RSQLite::SQLite(), "attimo-results.db")
    tbl <- tbl(conn, "scamp") %>%
        collect() %>%
        mutate(
            algorithm = "scamp",
            dataset = str_remove(dataset, "data/") %>%
                str_remove("-\\d+") %>%
                str_remove(".(csv|txt)")
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
        read_csv("data/HumanY.txt.measures") %>% mutate(dataset = "HumanY")
    ) %>% rename(window = w)
}

plot_scalability_n <- function(plotdata) {
    plotdata %>%
        left_join(dataset_info(), by = "dataset") %>%
        mutate(
            prefix = str_extract(dataset, "\\d+") %>% as.integer(),
            prefix = if_else(
                is.na(prefix),
                as.integer(n),
                prefix
            ),
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

latex_info <- function(data_motif_measures) {
    inner_join(data_motif_measures, dataset_info()) %>%
        select(dataset, n, window, rc1) %>%
        arrange(n) %>%
        mutate_if(is.numeric, scales::number_format(big.mark = "\\\\,")) %>%
        kbl(format = "latex", booktabs = T, align = "lrrr", escape = F) %>%
        write_file("imgs/dataset-info.tex")
}