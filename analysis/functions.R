baselines <- jsonlite::read_json("baselines.json")

theme_paper <- function() {
    theme_classic() +
        theme(
            strip.background = element_blank(),
            text = element_text(family = "Helvetica")
        )
}

fmt_duration <- function(d_secs) {
    inner <- function(d_secs) {
        scaling <- c(
            "s" = 1,
            "m" = 60,
            "h" = 60*60,
            "days" = 60*60*24,
            "years" = 60*60*24*365
        )
        scaled <- d_secs / scaling
        lastentry <- tail(scaled[scaled >= 1], 1)
        str_c(scales::number(lastentry, accuracy = 0.1), names(lastentry), sep=" ")
    }
    sapply(d_secs, inner)
}

dataset_info <- function() {
    tribble(
        ~dataset, ~n, ~avg_dist, ~window,
        "ASTRO", 1151349, 13.954815, 100,
        "ECG", 7871870, 44.087175, 1000,
        # "EMG", 543893, NA,
        "HumanY", 26415045, 176.138839, 18000,
        "GAP", 2049279, 34.235004, 600,
        "freezer", 7430755, 99.810919, 5000,
        "Seismic100M", 10^8, NA, 100,
        "Seismic", 10^9, 14.114333, 100,
        "Whales", 308941605, 16.715681, 140
    ) %>%
    mutate(
        size_gb = (n * 8) / (1024^3) # assuming a 64 bit representation of each value
    )
}

allowed_combinations <- tibble::tribble(
    ~dataset, ~window,
    "HumanY", 18000,
    "GAP", 600,
    # "EMG", 500,
    "ECG", 1000,
    "freezer", 5000,
    "ASTRO", 100,
    "Seismic", 100,
    "Whales", 140
)

fix_names <- function(df) {
    df %>%
        filter(!str_detect(dataset, "data/VCAB_BP2_580_days-100000000.txt")) %>% 
        mutate(
            path = dataset,
            dataset = if_else(str_detect(dataset, "Whales"), "Whales", dataset),
            dataset = if_else(str_detect(dataset, "VCAB.*100000000"), "Seismic100M", dataset),
            dataset = if_else(str_detect(dataset, "VCAB_noised"), "Seismic", dataset),
            dataset = str_remove(dataset, ".gz")
        )
}

reorder_datasets <- function(df) {
    df %>%
        mutate(
            dataset = factor(dataset, c(
                "ASTRO",
                "GAP",
                "freezer",
                "ECG",
                "HumanY",
                "Whales",
                "Seismic"
            ),
            ordered = T)
        )
}

add_prefix_info <- function(dat) {
    dat %>%
        mutate(
            path = if_else(is.na(path), dataset, path),
            dataset = str_remove(dataset, "data/") %>%
                str_remove("-(perc)?\\d+") %>%
                str_remove("scamp-") %>%
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
            perc_size = as.double(str_match(path, "-perc(\\d+)")[, 2]) / 100,
            is_full_dataset = !str_detect(path, "-(perc)?\\d+"),
            is_full_dataset = if_else(path == "data/VCAB_BP2_580_days-100000000.txt", T, is_full_dataset)
        )
}

load_attimo <- function() {
    conn <- DBI::dbConnect(RSQLite::SQLite(), "attimo-results.db")
    table <- tbl(conn, "attimo") %>%
        filter(version == max(version, na.rm = T)) %>%
        filter(hostname == "gpu02") %>%
        collect() %>%
        fix_names() %>%
        filter(!str_detect(dataset, "EMG")) %>%
        mutate(
            algorithm = "attimo",
            expid = row_number()
        ) %>%
        add_prefix_info() %>%
        semi_join(allowed_combinations)
    print("Computing memory usage")
    mem <- table %>%
        as_tbl_json(json.column = "log") %>%
        gather_array() %>%
        # spread_all() %>%
        spread_values(
            mem_bytes = jnumber("mem_bytes")
        ) %>%
        drop_na(mem_bytes) %>%
        group_by(expid, dataset) %>%
        summarise(max_mem_bytes = max(mem_bytes, na.rm = T)) %>%
        ungroup() %>%
        mutate(
            max_mem_gb = max_mem_bytes / (1024^3)
        ) %>%
        as_tibble() %>%
        select(expid, max_mem_gb, max_mem_bytes)
    print("Computing distance counts")
    dists <- table %>%
        as_tbl_json(json.column = "log") %>%
        gather_array() %>%
        spread_values(
            distances_fraction = jnumber("distances_fraction"),
            total_distances = jnumber("total_distances"),
            cnt_dist = jnumber("cnt_dist"),
            msg = jstring("msg")
        ) %>%
        filter(msg == "motifs completed") %>%
        drop_na(expid, distances_fraction, total_distances, cnt_dist) %>%
        as_tibble() %>%
        select(expid, distances_fraction, total_distances, cnt_dist)

    phases <- table %>%
        as_tbl_json(json.column = "log") %>%
        gather_array() %>%
        spread_values(
            msg = jstring("msg"),
            tag = jstring("tag"),
            phase = jstring("phase"),
            ts = jstring("ts"),
        ) %>%
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

load_scalability <- function() {
    conn <- DBI::dbConnect(RSQLite::SQLite(), "attimo-results.db")

    getinfo <- function(data) {
        data %>%
            filter(str_detect(dataset, "synth")) %>%
            mutate(
                n = as.integer(str_match(dataset, "n(\\d+)")[,2]),
                rc = as.integer(str_match(dataset, "rc(\\d+)")[,2]),
                difficulty = str_match(dataset, "easy|middle|difficult")[,1],
                dataset = 'synth'
            ) %>%
            drop_na()
    }

    attimo <- tbl(conn, "attimo") %>% 
        filter(repetitions == 400) %>%
        select(dataset, time_s) %>%
        mutate(algorithm = 'attimo') %>%
        collect() %>%
        getinfo()
    scamp <- tbl(conn, "scamp_gpu") %>%
        select(dataset, time_s) %>%
        mutate(algorithm = 'scamp-gpu') %>%
        collect() %>%
        getinfo()
    model <- lm(time_s ~ poly(n, 2), data=scamp)
    n_predict <- c(2**25, 2**27)
    estimates <- tibble(
        n = n_predict,
        time_s = predict(model, tibble(n = n_predict)),
        algorithm = "scamp-gpu",
        difficulty = "difficult"
    )
    scamp <- bind_rows(scamp, estimates)

    DBI::dbDisconnect(conn)
    bind_rows(attimo, scamp) %>%
        filter(difficulty != "middle")
}


load_rproj <- function() {
    conn <- DBI::dbConnect(RSQLite::SQLite(), "attimo-results.db")
    table <- conn %>%
        tbl(sql("select *, json_extract(motif_pairs, '$[0].dist') as motif_distance from projection")) %>%
        filter(
            hostname == "gpu03",
            outcome == "ok",
            motifs == 1
        ) %>%
        group_by(dataset, window) %>%
        slice_min(motif_distance) %>%
        collect() %>%
        fix_names() %>%
        add_prefix_info() %>%
        right_join(allowed_combinations) %>%
        mutate(algorithm = "rproj") %>%
        mutate(max_mem_gb = max_mem_bytes / (1024^3)) %>%
        reorder_datasets()
    DBI::dbDisconnect(conn)
    table
}

# The original prescrimp implementation takes 72 minutes on GAP, 
# with the same parameterization we use.
load_prescrimp <- function() {
    conn <- DBI::dbConnect(RSQLite::SQLite(), "attimo-results.db")
    tbl <- tbl(conn, "prescrimp") %>%
        collect() %>%
        filter(!str_detect(dataset, "EMG")) %>%
        filter(hostname == "gpu03") %>%
        fix_names() %>%
        add_prefix_info() %>%
        # right_join(allowed_combinations) %T>%
        mutate(algorithm = "prescrimp") %>%
        mutate(max_mem_gb = max_mem_bytes / (1024^3)) %>%
        reorder_datasets()
    
    tbl_measured <- tbl %>%
        filter(is_full_dataset) %>%
        mutate(is_estimate = F)

    smaller <- tbl %>%
        filter(!is_full_dataset)
    models <- smaller %>%
        group_by(dataset, window, stepsize) %>%
        do(
            model_time_s = lm(time_s ~ poly(prefix, 2), data=.),
            model_max_mem_gb = lm(max_mem_gb ~ prefix, data=.)
        )

    predictions <- tbl %>%
        # Select the combinations for which measurements are missing
        filter(is_full_dataset) %>%
        right_join(allowed_combinations) %>%
        filter(is.na(time_s)) %>%
        select(dataset, window,) %>%
        inner_join(models) %>%
        inner_join(dataset_info()) %>%
        group_by(dataset, window, stepsize, n) %>%
        do(tibble(
            time_s = predict(.$model_time_s[[1]], tibble(prefix=.$n)),
            max_mem_gb = predict(.$model_max_mem_gb[[1]], tibble(prefix=.$n)),
        )) %>%
        mutate(is_estimate = T, is_full_dataset = T, algorithm = 'prescrimp')

    DBI::dbDisconnect(conn)
    bind_rows(tbl_measured, predictions)
}

load_scamp_gpu <- function() {
    conn <- DBI::dbConnect(RSQLite::SQLite(), "attimo-results.db")
    tbl <- tbl(conn, "scamp_gpu") %>%
        filter(hostname == "gpu03") %>%
        collect() %>%
        filter(!str_detect(dataset, "EMG")) %>%
        fix_names() %>%
        add_prefix_info() %>%
        # right_join(allowed_combinations) %>%
        mutate(algorithm = "scamp-gpu") %>%
        mutate(max_mem_gb = max_mem_bytes / (1024^3)) %>%
        reorder_datasets()

    tbl_measured <- tbl %>%
        filter(is_full_dataset) %>%
        mutate(is_estimate = F)

    smaller <- tbl %>%
        filter(!is_full_dataset)
    models <- smaller %>%
        group_by(dataset, window) %>%
        do(
            model_time_s = lm(time_s ~ poly(prefix, 2), data=.),
            model_max_mem_gb = lm(max_mem_gb ~ poly(prefix, 1), data=.)
        )

    predictions <- tbl %>%
        # Select the combinations for which measurements are missing
        filter(is_full_dataset) %>%
        right_join(allowed_combinations) %>%
        filter(is.na(time_s)) %>%
        select(dataset) %>%
        inner_join(models) %>%
        inner_join(dataset_info()) %>%
        group_by(dataset, window, n) %>%
        do(tibble(
            time_s = predict(.$model_time_s[[1]], tibble(prefix=.$n)),
            max_mem_gb = predict(.$model_max_mem_gb[[1]], tibble(prefix=.$n)),
        )) %>%
        mutate(is_estimate = T, is_full_dataset = T, algorithm = 'scamp-gpu')


    DBI::dbDisconnect(conn)
    bind_rows(tbl_measured, predictions)
}


load_scamp <- function() {
    conn <- DBI::dbConnect(RSQLite::SQLite(), "attimo-results.db")
    tbl <- tbl(conn, "scamp") %>%
        filter(hostname == "gpu03") %>%
        collect() %>%
        filter(!str_detect(dataset, "EMG")) %>%
        fix_names() %>%
        add_prefix_info() %>%
        # right_join(allowed_combinations) %>%
        mutate(algorithm = "scamp") %>%
        mutate(max_mem_gb = max_mem_bytes / (1024^3)) %>%
        reorder_datasets()

    tbl_measured <- tbl %>%
        filter(is_full_dataset) %>%
        mutate(is_estimate = F)

    smaller <- tbl %>%
        filter(!is_full_dataset)
    models <- smaller %>%
        group_by(dataset, window, threads) %>%
        do(
            model_time_s = lm(time_s ~ poly(prefix, 2), data=.),
            model_max_mem_gb = lm(max_mem_gb ~ poly(prefix, 1), data=.)
        )

    predictions <- tbl %>%
        # Select the combinations for which measurements are missing
        filter(is_full_dataset) %>%
        right_join(allowed_combinations) %>%
        filter(is.na(time_s)) %>%
        select(dataset) %>%
        inner_join(models) %>%
        inner_join(dataset_info()) %>%
        group_by(dataset, window, threads, n) %>%
        do(tibble(
            time_s = predict(.$model_time_s[[1]], tibble(prefix=.$n)),
            max_mem_gb = predict(.$model_max_mem_gb[[1]], tibble(prefix=.$n)),
        )) %>%
        mutate(is_estimate = T, is_full_dataset = T, algorithm = 'scamp')


    DBI::dbDisconnect(conn)
    bind_rows(tbl_measured, predictions)
}

load_ll <- function() {
    conn <- DBI::dbConnect(RSQLite::SQLite(), "attimo-results.db")
    tbl <- tbl(conn, "ll") %>%
        collect() %>%
        filter(!str_detect(dataset, "EMG")) %>%
        fix_names() %>%
        add_prefix_info() %>%
        semi_join(allowed_combinations) %>%
        mutate(algorithm = "ll") %>%
        mutate(max_mem_gb = max_mem_bytes / (1024^3)) %>%
        reorder_datasets()
    DBI::dbDisconnect(conn)
    tbl
}

add_recall <- function(data_attimo, data_scamp) {
    compute_recall <- function(dataset_name, window_size, motifs_json) {
        mots <- tidyjson::gather_array(motifs_json) %>%
            tidyjson::spread_all()

        baseline <- baselines[[as.character(dataset_name)]]
        if (is.null(baseline)) {
            return(NA)
        }

        actual_a <- mots %>% pull(a)
        actual_b <- mots %>% pull(b)
        # actual_dist <- mots %>% pull(dist)
        # ground_dist <- purrr::map_dbl(baseline, ~ .[[3]])
        # print(ground_dist - actual_dist)

        cnt <- 0

        for (i in 1:length(baseline)) {
            ground_a <- baseline[[i]][[1]]
            ground_b <- baseline[[i]][[2]]
            found_a <- sum(abs(ground_a - actual_a) <= window_size) > 0
            found_b <- sum(abs(ground_b - actual_b) <= window_size) > 0
            found <- found_a || found_b
            if (found) {
                cnt <- cnt + 1
            }
        }

        cnt / nrow(mots)
    }

    data_attimo %>%
        filter(motifs == 10) %>%
        ungroup() %>%
        rowwise() %>%
        mutate(recall = compute_recall(dataset, window, motif_pairs))
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

get_data_comparison <- function(data_attimo, data_scamp, data_ll, data_scamp_gpu, data_prescrimp, data_rproj) {
    bind_rows(
        data_attimo %>%
            filter((repetitions == 200) | (dataset == "Seismic") | (dataset == "Whales")) %>%
            select(
                algorithm, hostname, dataset, is_full_dataset, threads, window,
                repetitions, motifs, delta, time_s, max_mem_gb, distances_fraction
            ) %>% 
            filter(motifs == 1) %T>%
            print() %>%
            group_by(algorithm, hostname, dataset, window) %>%
            slice_min(time_s)
            ,
        select(
            data_scamp, algorithm, hostname, dataset, is_full_dataset, is_estimate,
            threads, window, max_mem_gb, time_s
        ),
        select(
            data_prescrimp, algorithm, hostname, dataset, is_full_dataset, is_estimate,
            window, max_mem_gb, time_s
        ),
        select(
            data_ll, algorithm, hostname, dataset, is_full_dataset, window,
            max_mem_gb, time_s
        ),
        select(
            data_rproj, algorithm, hostname, dataset, is_full_dataset, window,
            max_mem_gb, time_s
        ),
        select(data_scamp_gpu, algorithm, dataset, window, time_s, max_mem_gb, is_full_dataset, is_estimate)
    ) %>%
        replace_na(list(is_estimate = F)) %>%
        semi_join(allowed_combinations) %>%
        filter(is_full_dataset) %>%
        group_by(dataset, algorithm) %>%
        slice_min(time_s) %>%
        ungroup() %>%
        inner_join(dataset_info()) %>%
        select(is_estimate, algorithm, dataset, n, size_gb, window, time_s, max_mem_gb, distances_fraction)
}

do_tab_speedup <- function(data_comparison) {
    data_comparison %>%
        select(algorithm, dataset, time_s) %>%
        pivot_wider(names_from = algorithm, values_from=time_s) %>%
        mutate(speedup = `scamp-gpu` / attimo) %>%
        select(dataset, speedup) %>%
        reorder_datasets() %>%
        arrange(dataset)
        kableExtra::kbl(booktabs = T, format = 'latex', linesep = "") %>%
        write_file("imgs/time-comparison-normalized.tex")
}


do_tab_time_comparison_normalized <- function(data_comparison) {
    data_comparison %>%
        mutate(ntime = time_s / n) %>%
        select(algorithm, dataset, ntime) %>%
        pivot_wider(names_from = algorithm, values_from=ntime) %>%
        mutate_if(is.numeric, scales::scientific_format()) %>%
        mutate(across(c("attimo", "ll", "scamp", "scamp-gpu"), ~ if_else(is.na(.), "-", .))) %>%
        reorder_datasets() %>%
        arrange(dataset) %>%
        select(dataset, attimo, `scamp-gpu`, scamp, ll) %>%
        kableExtra::kbl(booktabs = T, format = 'latex', linesep = "") %>%
        write_file("imgs/time-comparison-normalized.tex")
}

do_tab_time_comparison <- function(data_comparison, file_out) {
    data_comparison %>%
        group_by(dataset) %>%
        mutate(
            is_best = time_s == min(time_s),
            is_best_mem = max_mem_gb == min(max_mem_gb),
            # time_s = scales::number(time_s, accuracy = 0.1),
            time_s = if_else(time_s > 2*3600,
                fmt_duration(time_s),
                scales::number(time_s, accuracy=0.1)
            ),
            time_s = if_else(is_estimate,
                str_c("{\\small(", time_s,")}"),
                time_s
            ),
            time_s = if_else(is_best,
                str_c("\\underline{", time_s, "}"),
                time_s
            ),
            mem_overhead_gb = scales::number(max_mem_gb, accuracy = 0.1),
            mem_overhead_gb = if_else(is_estimate,
                str_c("{\\small(", mem_overhead_gb, ")}"),
                mem_overhead_gb
            ),
            mem_overhead_gb = if_else(is_best_mem,
                str_c("\\underline{", mem_overhead_gb, "}"),
                mem_overhead_gb
            ),
            distances_fraction = scales::scientific(
                distances_fraction,
                prefix = "$",
                suffix = "}$",
                digits = 2
            ) %>% str_replace("e", "\\\\cdot 10^{"),
        ) %>%
        select(dataset, algorithm, time_s, mem_overhead_gb, distances_fraction) %>% 
        pivot_wider(names_from = algorithm, values_from = c(time_s, mem_overhead_gb, distances_fraction)) %>%
        mutate(across(!matches("dataset"), ~if_else(is.na(.), "-", .))) %>%
        reorder_datasets() %>%
        arrange(dataset) %>%
        select(dataset,
            # time
            `\\attimo` = time_s_attimo,
            `\\scampgpu` = `time_s_scamp-gpu`,
            `\\scamp`=time_s_scamp,
            `\\prescrimp`=time_s_prescrimp,
            `\\LL`=time_s_ll,
            `\\rproj`=time_s_rproj,
            # memory
            `\\attimo ` = mem_overhead_gb_attimo,
            `\\scampgpu ` = `mem_overhead_gb_scamp-gpu`,
            `\\scamp `=mem_overhead_gb_scamp,
            `\\prescrimp `=mem_overhead_gb_prescrimp,
            `\\LL `=mem_overhead_gb_ll,
            `\\rproj `=mem_overhead_gb_rproj,
        ) %>%
        kbl(
            format = "latex", booktabs = T, linesep = "", align = "l rrrrr rrrrr",
            escape = F
        ) %>%
        add_header_above(c(" " = 1, "Time (s)" = 6, "Memory (Gb)" = 6), escape = F) %>%
        write_file(file_out)
}

compute_distance_distibution <- function(data_attimo) {
    do_compute <- function(dataset, path, window) {
        print(paste("dataset", dataset, "window", window))
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
        filter(is_full_dataset | (path == "data/VCAB_BP2_580_days-100000000.txt")) %>%
        distinct(path, dataset, window) %>%
        rowwise() %>%
        summarise(do_compute(dataset, path, window))
}

# compute_measures <- function(path, window, idxs) {
#     out <- paste0(path, ".", window, ".measures")
#     if (!file.exists(out) || !file.exists(paste0(out, ".gz"))) {
#         system2(
#             "target/release/examples/measures",
#             c(
#                 "--window", window,
#                 "--path", path,
#                 "--output", out,
#                 idxs
#             )
#         )
#     }
#     if (file.exists(paste0(out, ".gz"))) {
#         out <- paste0(out, ".gz")
#     }
#     dat <- readr::read_csv(out) %>%
#         mutate(
#             motif_idx = row_number(),
#             nn_corr = 1 - nn^2 / (2 * window)
#         )
#     dat
# }

dataset_measures <- function(data_attimo) {
    # avg_dists <- data_distances %>%
    #     group_by(dataset, window) %>%
    #     summarise(avg_distance = mean(distance))

    data_attimo %>%
        inner_join(dataset_info()) %>%
        filter(motifs == 10) %>%
        group_by(path, dataset, window) %>%
        slice(1) %>%
        as_tbl_json(json.column = "motif_pairs") %>%
        select(path, dataset, window, avg_dist) %>%
        gather_array() %>%
        spread_all() %>%
        rename(motif_idx = array.index) %>%
        mutate(rc1 = avg_dist / dist)

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
    labsfun <- function(breaks) {
        if (breaks[length(breaks)] == 1600) {
            breaks[length(breaks) - 1] = ""
        }
        breaks
    }
    data_attimo %>%
        filter(motifs == 10) %>%
        filter(dataset != 'Seismic') %>%
        filter((dataset != 'HumanY') | (repetitions < 800)) %>%
        group_by(dataset, window) %>%
        slice_min(time_s, n=5) %>%
        group_by(dataset, window, repetitions) %>%
        filter(time_s < 10000) %>%
        summarise(
            time_s = mean(time_s),
            preprocessing = mean(preprocessing)
        ) %>%
        inner_join(dataset_info()) %>%
        mutate(
            dataset = factor(dataset, levels=c("freezer", "ASTRO", "GAP", "Whales", "ECG", "HumanY"), ordered=T),
            labelpos = time_s + 0.1 * max(time_s),
            tickpos = -0.1 * max(time_s),
            mempos = -0.6 * max(time_s),
        ) %>%
        ggplot(aes(repetitions, time_s)) +
        geom_area(aes(y = preprocessing), fill = "#f78a36", alpha = 0.4) +
        geom_ribbon(aes(ymin = preprocessing, ymax = time_s), fill = "#74caff", alpha = 0.4) +
        geom_line() +
        geom_point() +
        scale_x_continuous(labels=labsfun) +
        labs(
            x = "Repetitions",
            y = "Total time (s)"
        ) +
        facet_wrap(vars(dataset), ncol = 3, scales = "free") +
        coord_cartesian(clip = "off") +
        theme_paper() +
        theme(
            panel.spacing = unit(0, "mm")
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

plot_motifs_10_alt2 <- function(data_attimo, data_scamp, data_scamp_gpu, data_measures) {
    pdata <- data_attimo %>%
        left_join(select(data_scamp_gpu, dataset, window = w, time_scamp_gpu_s = time_s)) %>%
        filter(motifs == 10) %>%
        filter((repetitions == 400) | (dataset == "Seismic") | (dataset == "Whales")) %>%
        group_by(dataset, window) %>%
        slice_min(time_s) %>%
        ungroup() %>%
        as_tbl_json(json.column = "motif_pairs") %>%
        gather_array() %>%
        spread_all() %>%
        rename(motif_idx = array.index) %>%
        as_tibble() %>%
        select(
            dataset, window, time_s, time_scamp_gpu_s, dist,
            motif_idx, confirmation_time, preprocessing
        ) %>%
        mutate(
            time_scamp_s_hline = if_else(
                (time_scamp_gpu_s < 1000) & (motif_idx == 1),
                time_scamp_gpu_s,
                as.double(NA)
            ),
            time_scamp_s_label = if_else(
                time_scamp_gpu_s >= 1000 & (motif_idx == 1),
                time_scamp_gpu_s,
                as.double(NA)
            )
        ) %T>%
        print(n=100) %>%
        mutate(
            confirmation_time = as.numeric(confirmation_time),
            preprocessing = as.numeric(preprocessing)
        )  %>%
        # reorder_datasets()
        inner_join(dataset_info()) %>%
        mutate(
            dataset = fct_reorder(dataset, n)
        )

    maxval <- pdata %>% summarise(max(time_s)) %>% pull()

    bars <- ggplot(pdata, aes(y = confirmation_time, x = dist)) +
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
                    prefix = "Scamp-gpu: ",
                    suffix = " s"
                ),
                x = dist,
                y = time_scamp_s_hline + 30
            ),
            hjust = 0,
            vjust = 0
        ) +
        geom_text(
            aes(
                label = scales::number(
                    time_scamp_s_label,
                    accuracy = 1,
                    prefix = "Scamp-gpu: ",
                    suffix = " s →"
                ),
                x = dist # * 1.2
            ),
            # y = 500,
            y = maxval,
            hjust = 1,
            vjust = 0
        ) +
        scale_y_continuous(limits = c(0, NA)) +
        # scale_y_continuous(limits = c(0, NA), breaks = 0:5 * 100) +
        scale_x_continuous(limits = c(NA, NA), position = "top") +
        facet_wrap(vars(dataset), ncol = 1, scales = "free_y", strip.position = "left") +
        labs(
            x = "distance",
            y = "time (s)"
        ) +
        coord_flip(clip='off') +
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
            panel.spacing = unit(8, "mm")
        )

    bars
}

plot_motifs_10_alt3 <- function(data_attimo, data_scamp, data_scamp_gpu) {
    scamp_thresh <- 2300
    textsize <- 4

    pdata <- data_attimo %>%
        left_join(select(data_scamp_gpu, dataset, window, time_scamp_gpu_s = time_s)) %>%
        filter(motifs == 10) %>%
        filter((repetitions == 400) | (dataset == "Seismic") | (dataset == "Whales")) %>%
        group_by(dataset, window) %>%
        slice_min(time_s) %>%
        ungroup() %>%
        as_tbl_json(json.column = "motif_pairs") %>%
        gather_array() %>%
        spread_all() %>%
        rename(motif_idx = array.index) %>%
        as_tibble() %>%
        select(dataset, window, time_s, time_scamp_gpu_s, dist, motif_idx, confirmation_time, preprocessing) %>%
        mutate(
            time_scamp_s_hline = if_else(
                (time_scamp_gpu_s < scamp_thresh) & (motif_idx == 1),
                time_scamp_gpu_s,
                as.double(NA)
            ),
            time_scamp_s_label = if_else(
                time_scamp_gpu_s >= scamp_thresh & (motif_idx == 1),
                time_scamp_gpu_s,
                as.double(NA)
            ),
            label_just = if_else(
                time_scamp_gpu_s >= 1000 & (motif_idx == 1),
                1,
                0
            ),
            segment_offset = if_else(
                time_scamp_gpu_s >= 1000 & (motif_idx == 1),
                -40,
                40
            )
        ) %>%
        mutate(
            confirmation_time = as.numeric(confirmation_time),
            preprocessing = as.numeric(preprocessing)
        )  %>%
        inner_join(dataset_info()) %>%
        mutate(
            dataset = fct_reorder(dataset, n)
        )

    maxval <- pdata %>% summarise(max(time_s)) %>% pull()
    maxval <- scamp_thresh

    bars <- ggplot(pdata, aes(y = confirmation_time, x = 0)) +
        geom_segment(
            mapping = aes(yend = time_scamp_gpu_s, y = 0, xend = 0),
            color = "gray80",
            size = 0.2,
            data = function(d) {
                d %>%
                    filter(time_scamp_gpu_s < scamp_thresh) %>%
                    # mutate(time_scamp_gpu_s = min(time_scamp_gpu_s, 2204)) %>%
                    group_by(dataset) %>% slice(1)
            },
        ) +
        geom_segment(
            mapping = aes(yend = time_s, y = 0, xend = 0),
            color = "gray30",
            size = 1,
            data = function(d) {
                group_by(d, dataset) %>% slice(1)
            },
        ) +
        geom_segment(
            mapping = aes(yend = preprocessing, y = 0, xend = 0),
            color = "#f78a36",
            size = 3,
            # alpha = 0.3,
            data = function(d) {
                group_by(d, dataset) %>% slice(1)
            },
        ) +
        geom_point(
            shape="|", 
            size = 2
        ) +
        geom_segment(
            aes(y = time_scamp_s_hline, yend = time_scamp_s_hline),
            x = 0,
            xend = -0.9,
            linetype = "solid",
            color = "gray80",
            size = 0.3,
        ) +
        geom_segment(
            aes(y = time_scamp_s_hline, yend = time_scamp_s_hline + segment_offset),
            x = -0.9,
            xend = -0.9,
            linetype = "solid",
            color = "gray80",
            size = 0.3,
        ) +
        geom_text(
            aes(
                label = scales::number(
                    time_scamp_s_hline,
                    accuracy = 1,
                    prefix = "Sᴄᴀᴍᴘ-ɢᴘᴜ: ",
                    suffix = " s",
                ),
                hjust = label_just,
                x = -0.8,
                y = time_scamp_s_hline + segment_offset
            ),
            color = "gray40",
            size = textsize,
            vjust = 0
        ) +
        geom_text(
            aes(
                label = scales::number(
                    time_scamp_s_label,
                    accuracy = 1,
                    prefix = "Sᴄᴀᴍᴘ-ɢᴘᴜ: (",
                    suffix = " s) →"
                ),
                x = -0.8 # * 1.2
            ),
            # y = 500,
            color = "gray40",
            y = maxval,
            size = textsize,
            hjust = 1,
            vjust = 0
        ) +
        geom_text(
            aes(label = dataset, x=0),
            y = 0,
            nudge_x = 0.8,
            size = textsize,
            hjust = 0
        ) +
        scale_y_continuous(limits = c(0, NA)) +
        scale_x_continuous(limits = c(-1, 1)) +
        facet_wrap(vars(dataset), ncol = 1, scales = "free_y", strip.position = "left") +
        labs(
            x = "",
            y = "time (s)"
        ) +
        coord_flip(clip='off') +
        theme_paper() +
        theme(
            axis.line.y = element_blank(),
            axis.text.y = element_blank(),
            axis.title.y = element_blank(),
            axis.ticks.y = element_blank(),
            # axis.line.x = element_blank(),
            # axis.text.x = element_blank(),
            # axis.ticks.x = element_blank(),
            strip.text = element_blank(),
            panel.spacing = unit(2, "mm")
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

plot_scalability_n_alt <- function(data_scalability) {
    data_scalability <- data_scalability %>% 
        filter(as.integer(log2(n)) == log2(n))
    attimo <- data_scalability %>% filter(algorithm == "attimo")
    baseline <- data_scalability %>% filter(algorithm != "attimo")
    labels <- baseline %>%
        filter(n == max(n))

    ggplot(attimo, aes(n, time_s, linetype=difficulty, color=difficulty)) +
        geom_point(data=baseline, color='black', linetype='solid', stat='summary') +
        geom_line(data=baseline, color='black', linetype='solid', stat='summary') +
        geom_point() +
        geom_line() +
        geom_text(
            data = labels,
            label = "Sᴄᴀᴍᴘ-ɢᴘᴜ",
            show.legend = F,
            color = 'black',
            nudge_x = -1.1
        ) +
        scale_y_continuous(
            labels = scales::number_format(accuracy=1), 
            trans='log2') +
        scale_x_continuous(
            # labels = scales::scientific_format(), 
            breaks = 2 ** c(19, 21, 23, 25, 27),
            labels = function(breaks) {
                TeX(str_c("$2^{", log2(breaks), "}$"))
            },
            trans='log2') +
        scale_color_manual(values = c("#5778a4", "#e49444", "#e15759")) +
        labs(
            x = "Size",
            y = "Total time (s)",
            color = "",
            linetype = ""
        ) +
        theme_paper() +
        theme(
            legend.position = c(0.2, 0.8)
        )
}

plot_mem <- function(data_attimo) {
    plotdata <- data_attimo %>%
        filter(motifs == 10) %>%
        mutate(bytes_per_subsequence = max_mem_bytes / n)

    ggplot(plotdata, aes(repetitions, bytes_per_subsequence)) +
        geom_point(size=0.5) +
        geom_point(stat="summary") +
        geom_line(stat="summary") +
        geom_text(
            aes(label=scales::number_bytes(stat(y), accuracy=0.01)),
            nudge_y = 300,
            size = 3,
            stat="summary"
        ) +
        scale_x_continuous(expand = expansion(mult=0.1)) +
        labs(
            x = "Repetitions",
            y = "Bytes per subsequence"
        ) +
        theme_paper()
}

table_mem <- function(data_attimo) {
    data_attimo %>%
        filter(motifs == 10, repetitions %in% c(50, 100, 200, 400, 800, 1600)) %>%
        mutate(bytes_per_subsequence = max_mem_bytes / n) %>%
        group_by(repetitions) %>%
        summarise(
            bytes_per_subsequence = scales::number_bytes(mean(bytes_per_subsequence), accuracy = 0.01)
        ) %>%
        pivot_wider(names_from=repetitions, values_from=bytes_per_subsequence) %>%
        kableExtra::kbl(format = "latex", booktabs = T, align = "rrrrrr") %>%
        write_file("imgs/memory.tex")
}
