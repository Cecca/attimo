library(tidyverse)
library(tidyjson)
library(lubridate)

mem_df <- read_json(".trace.json", format="jsonl") %>% 
    spread_all() %>% 
    filter(tag == "memory") %>% 
    mutate(
        ts = ymd_hms(ts),
        mem_gb = mem_bytes / (1024 * 1024 * 1024)
    )

prof_df <- read_json(".trace.json", format="jsonl") %>%
    spread_all() %>%
    filter(tag == "profiling") %>%
    mutate(
        ts = ymd_hms(ts)
    )
prof_df %>%
    as_tibble() %>%
    select(ts, msg) %>%
    print(n = 1000)

ggplot(mem_df, aes(ts, mem_gb)) + 
    geom_line() +
    geom_vline(
        data=prof_df, 
        mapping=aes(xintercept = ts),
        linetype = "dashed"
    ) +
    theme_classic() 

ggsave("mem.png", width = 10, height = 4)
