# /usr/bin/env python

from icecream import ic
import pandas as pd
import math
import pyattimo
import time
from pathlib import Path
from urllib.request import urlretrieve
from urllib.parse import urlparse
import os
import logging
import joblib

logging.basicConfig(level=logging.WARN)

MEM = joblib.Memory(".joblib")


def download(url):
    localdir = Path(".data")
    if not localdir.is_dir():
        os.mkdir(localdir)
    fname = Path(urlparse(url).path).name
    localname = localdir / fname
    if not localname.is_file():
        print(f"downloading {url} to {localname}")
        urlretrieve(url, localname)
    return localname


def read_penguin_1m(nhalf=10_000):
    path = "../datasets/experiments/"
    url = "https://raw.githubusercontent.com/patrickzib/motiflets/refs/heads/pyattimo/datasets/experiments/penguin.txt"
    series = pd.read_csv(
        download(url),
        names=(["X-Acc", "Y-Acc", "Z-Acc", "4", "5", "6", "7", "Pressure", "9"]),
        delimiter="\t",
        header=None,
    )
    ts = series.iloc[497699 - nhalf : 497699 + nhalf, 0].values
    return f"penguin-{ts.shape[0]}", ts


def read_astro(prefix):
    url = "https://github.com/patrickzib/motiflets/raw/refs/heads/pyattimo/datasets/original/ASTRO.csv"
    ds_name = f"ASTRO-{prefix}"
    series = pd.read_csv(download(url), header=None).squeeze("columns")
    return ds_name, series.values[:prefix]


def read_gap(prefix):
    url = "https://github.com/patrickzib/motiflets/raw/refs/heads/pyattimo/datasets/original/GAP.csv"
    ds_name = f"GAP-{prefix}"
    series = pd.read_csv(download(url), header=None).squeeze("columns")
    return ds_name, series.values[:prefix]


def read_dishwasher(prefix):
    url = "https://github.com/patrickzib/motiflets/raw/refs/heads/pyattimo/datasets/original/dishwasher.txt"
    ds_name = "Dishwasher"
    series = pd.read_csv(download(url), header=None).squeeze('columns')
    return ds_name, series.values[:prefix]

@MEM.cache
def run_baseline(ts, window, max_support):
    t_start = time.time()
    baseline = pyattimo.motiflet_brute_force(ts, window, max_support)
    t_baseline = time.time() - t_start
    return t_baseline


@MEM.cache
def run_attimo(ts, window, max_support, delta):
    n = ts.shape[0]
    t_start = time.time()
    iter = pyattimo.MotifletsIterator(
        ts,
        window,
        max_support,
        delta=delta,
        observability_file="/tmp/observe.csv",
        stop_on_threshold=True,
        fraction_threshold=math.log(n) / n,
    )
    approx = list(iter)
    t_attimo = time.time() - t_start
    return t_attimo


def do_bench():
    datasets = [
        # {"data": lambda: read_penguin_1m(), "max_support": 20, "window": 125},
        # {"data": lambda: read_penguin_1m(20000), "max_support": 20, "window": 125}
    ]
    for n in [50_000, 100_000, 200_000]:
        datasets.extend([
            # {"data": read_astro(n), "max_support": 10, "window": 70 * 38},
            # {"data": read_gap(n), "max_support": 10, "window": 50 * 68},
            {"data": read_dishwasher(n), "max_support": 20, "window": 125*8},
        ])
    results = []
    for conf in datasets:
        name, ts = conf["data"]

        t_baseline = run_baseline(ts, conf["window"], conf["max_support"])
        results.append(
            {
                "dataset": name,
                "algorithm": "baseline",
                "time_s": t_baseline,
                "n": ts.shape[0],
            }
        )

        for delta in [0.05, 0.5]:
            t_attimo = run_attimo(
                ts, conf["window"], conf["max_support"], delta
            )
            results.append(
                {
                    "dataset": name,
                    "algorithm": f"attimo-{delta}",
                    "time_s": t_attimo,
                    "n": ts.shape[0],
                }
            )

    results = pd.DataFrame(results)
    print(results)


if __name__ == "__main__":
    do_bench()
