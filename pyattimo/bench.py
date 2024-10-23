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


def read_arrhythmia(prefix):
    url = "https://raw.githubusercontent.com/patrickzib/motiflets/refs/heads/pyattimo/datasets/experiments/arrhythmia_subject231_channel0.csv"
    series = pd.read_csv(download(url))
    ds_name = "Arrhythmia"
    return ds_name, series.iloc[:, 0].T.values[:prefix]


def read_penguin_1m(nhalf=10_000):
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
    series = pd.read_csv(download(url), header=None).squeeze("columns")
    return ds_name, series.values[:prefix]


def read_eeg(prefix):
    url = "https://github.com/patrickzib/motiflets/raw/refs/heads/pyattimo/datasets/original/npo141.csv"  # Dataset Length n:  269286
    ds_name = "EEG-Sleep"
    series = pd.read_csv(download(url), header=None).squeeze("columns")
    return ds_name, series.values[:prefix]

def read_pamap(prefix, selection=None):
    import numpy as np

    desc_url = "https://raw.githubusercontent.com/patrickzib/motiflets/refs/heads/pyattimo/datasets/experiments/pamap_desc.txt"
    desc_filename = download(desc_url)
    desc_file = []

    with open(desc_filename, 'r') as file:
        for line in file.readlines(): desc_file.append(line.split(","))

    df = []
    for idx, row in enumerate(desc_file):
        if selection is not None and idx not in selection: continue

        (ts_name, window_size), change_points = row[:2], row[2:]
        if len(change_points) == 1 and change_points[0] == "\n": change_points = list()
        ts = np.load(file=".data/pamap_data.npz")[ts_name]

        df.append(
            (ts_name, int(window_size), np.array([int(_) for _ in change_points]), ts))

    return "PAMAP", pd.DataFrame.from_records(
        df, columns=["name", "window_size", "change_points", "time_series"]).time_series[0].values[:prefix]


@MEM.cache
def run_baseline(ts, window, max_support):
    t_start = time.time()
    baseline = pyattimo.motiflet_brute_force(ts, window, max_support)
    t_baseline = time.time() - t_start
    baseline.sort(key=lambda m: m.support)
    extents = [m.extent for m in baseline]
    supports = [m.support for m in baseline]
    return t_baseline, extents, supports


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
        fraction_threshold=math.sqrt(n) / n,
    )
    approx = list(iter)
    t_attimo = time.time() - t_start
    approx.sort(key=lambda m: m.support)
    extents = [m.extent for m in approx]
    supports = [m.support for m in approx]
    return t_attimo, extents, supports


def do_bench():
    datasets = [
        {"data": read_penguin_1m(), "max_support": 20, "window": 125},
        {"data": read_penguin_1m(50_000), "max_support": 20, "window": 125},
    ]
    for n in [50_000, 100_000, 150_000]:
        datasets.extend(
            [
                {"data": read_arrhythmia(n), "max_support": 20, "window": 200},
                # {"data": read_pamap(n), "max_support": 20, "window": 200},
                {"data": read_eeg(n), "max_support": 20, "window": 25*25},
                {"data": read_astro(n), "max_support": 10, "window": 70 * 38},
                {"data": read_gap(n), "max_support": 10, "window": 50 * 68},
                {"data": read_dishwasher(n), "max_support": 20, "window": 125 * 8},
            ]
        )
    results = []
    extents = []
    for conf in datasets:
        name, ts = conf["data"]
        name = name.split("-")[0]
        ic(name)

        t_baseline, extents_baseline, supports_baseline = run_baseline(
            ts, conf["window"], conf["max_support"]
        )
        results.append(
            {
                "dataset": name,
                "algorithm": "baseline",
                "time_s": t_baseline,
                "n": ts.shape[0],
            }
        )
        for e, s in zip(extents_baseline, supports_baseline):
            extents.append(
                {
                    "dataset": name,
                    "algorithm": "baseline",
                    "n": ts.shape[0],
                    "support": s,
                    "extent": e,
                }
            )

        for delta in [0.05, 0.5]:
            t_attimo, extents_attimo, supports_attimo = run_attimo(
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
            for e, s in zip(extents_attimo, supports_attimo):
                extents.append(
                    {
                        "dataset": name,
                        "algorithm": f"attimo-{delta}",
                        "n": ts.shape[0],
                        "support": s,
                        "extent": e,
                    }
                )

    results = pd.DataFrame(results)
    print(results)
    extents = pd.DataFrame(extents).pivot(
        columns="algorithm", values="extent", index=["dataset", "n", "support"]
    )
    for c in extents.columns:
        extents[c] = extents[c] / extents["baseline"]

    extents = extents.groupby(level=["dataset", "n"]).max()
    print(extents)


if __name__ == "__main__":
    do_bench()
