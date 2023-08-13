library(ggplot2)
library(tibble)
library(dplyr)

data <- tribble(
  ~dataset, ~algorithm, ~mem,
  "astro", "attimo", 0.7,
  "GAP", "attimo", 1.1,
  "freezer", "attimo", 4.0,
  "ECG", "attimo", 3.9,
  "HumanY", "attimo", 14.4,
  "Whales", "attimo", 103.7,
  "Seismic", "attimo", 129.1,
  "astro", "SCAMP", 5.8,
  "GAP", "SCAMP", 5.9,
  "freezer", "SCAMP", 6.4,
  "ECG", "SCAMP", 6.5,
  "HumanY", "SCAMP", 8.5,
  "Whales", "SCAMP", 38.1,
  "Seismic", "SCAMP", 110.6
) |> mutate(
  dataset = factor(dataset, ordered=TRUE, levels = rev(c(
    "astro", "GAP", "freezer", "ECG", "HumanY", "Whales", "Seismic"
  )))
)


p <- ggplot(data, aes(dataset, mem, fill=algorithm)) +
  geom_col(position="dodge") +
  labs(x="dataset", y="memory (Gb)") +
  scale_fill_manual(values=c(
    "attimo" = "#f3762a",
    "SCAMP" = "#808080"
  )) +
  coord_flip() +
  theme_classic() +
  theme(legend.position = c(0.9, 0.9))
ggsave("imgs/memory.png", dpi=300, width=5, height=3)
