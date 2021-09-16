library(tidyverse)

db <- DBI::dbConnect(RSQLite::SQLite(), "attimo-results.db")

attimo <- collect(tbl(db, "attimo")) %>%
    group_by(dataset, memory, window) %>%
    summarise(time_s = mean(time_s)) %>%
    ungroup() %>%
    group_by(dataset, window) %>%
    slice_min(time_s) %>%
    ungroup() %>%
    mutate(algorithm = "attimo")

scamp <- collect(tbl(db, "scamp")) %>%
    mutate(algorithm = "scamp")

data <- bind_rows(attimo, scamp) %>%
mutate(
    n = as.integer(str_extract(dataset, "\\d+"))
)

ggplot(data, aes(n, time_s, color=algorithm, shape=algorithm)) +
    geom_line(stat='summary') +
    geom_point() +
    theme_classic()
