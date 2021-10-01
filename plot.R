library(tidyverse)
library(ggrepel)

db <- DBI::dbConnect(RSQLite::SQLite(), "attimo-results.db")

attimo <- collect(tbl(db, "attimo")) %>%
    group_by(dataset, repetitions, window) %>%
    summarise(time_s = mean(time_s)) %>%
    ungroup() %>%
    group_by(dataset, window) %>%
    slice_min(time_s) %>%
    ungroup() %>%
    mutate(algorithm = "attimo")

scamp <- collect(tbl(db, "scamp")) %>%
    mutate(algorithm = str_c("scamp-", threads))

data <- bind_rows(attimo, scamp) %>%
mutate(
    n = as.integer(str_extract(dataset, "\\d+"))
)

speedups <- data %>%
    filter((algorithm == 'attimo') | (threads == 32)) %>%
    select(n, algorithm, time_s) %>%
    pivot_wider(names_from=algorithm, values_from=time_s) %>%
    mutate(speedup = `scamp-32` / attimo,
           label = scales::number(speedup, accuracy=0.1, prefix="x"))

p <- ggplot(data, aes(n, time_s, color=algorithm, shape=algorithm)) +
    geom_text(
        aes(label = label),
        data = ~group_by(.x, algorithm) %>% mutate(label = if_else(n == max(n), algorithm, "")),
        show.legend = F,
        hjust = 1,
        nudge_y = 200
    ) +
    geom_segment(
        aes(y = attimo, yend=`scamp-32`, x=n, xend=n),
        data=filter(speedups, n >= 3000000),
        inherit.aes = F,
        color="gray"
    ) +
    geom_label(
        aes(label = label, x=n, y=(`scamp-32` + attimo) / 2),
        data=filter(speedups, n >= 3000000),
        size=2,
        fill = "white",
        color="black",
        show.legend = F,
        hjust = 0.5,
        inherit.aes = F
    ) +
    geom_line(stat='summary') +
    geom_point() +
    scale_x_continuous(labels=scales::number_format()) +
    labs(
        x = "n",
        y = "time (s)",
        title = "ECG dataset",
        caption = "Running on Intel(R) Xeon(R) CPU E5-2667 v3 @ 3.20GHz"
    ) +
    theme_classic() +
    theme(legend.position="none")

ggsave("ecg.png", width=6, height=3)
