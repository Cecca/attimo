#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "polars",
#     "seaborn",
# ]
# ///

import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import sys

print("reading data", file=sys.stderr)
data = pl.read_csv(sys.argv[1])
data = data.filter(pl.col("name") == "graph/distance").select("value").unique()

print("creating histogram", file=sys.stderr)
sns.kdeplot(data, x="value")
sns.rugplot(data, x="value")
plt.axvline(44, color="red")

plt.tight_layout()
plt.savefig(sys.stdout, format="svg")
sys.stdout.flush()
