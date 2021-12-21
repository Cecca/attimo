theme_paper <- function() {
    theme_classic()
}

dataset_info <- function() {
    tribble(
        ~dataset, ~n,
        "data/ASTRO.csv", 1151349,
        "data/ECG.csv", 7871870,
        "data/EEG.csv", 543893,
        "data/HumanY.txt", 26415045,
        "data/GAP.csv", 2049279
    )
}

load_attimo <- function() {
    conn <- DBI::dbConnect(RSQLite::SQLite(), "attimo-results.db")
    tbl <- tbl(conn, "attimo") %>%
        filter(version == max(version)) %>%
        collect() %>%
        mutate(algorithm = "attimo")
    DBI::dbDisconnect(conn)
    tbl
}

load_scamp <- function() {
    conn <- DBI::dbConnect(RSQLite::SQLite(), "attimo-results.db")
    tbl <- tbl(conn, "scamp") %>%
        collect() %>%
        mutate(algorithm = "scamp")
    DBI::dbDisconnect(conn)
    tbl
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