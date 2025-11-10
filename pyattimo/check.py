# A scratch Python file to try out the API
import pyattimo
import datetime
import time
import logging
import numpy as np
import scipy
import math
import requests
import csv
import pathlib
import itertools

logging.basicConfig(level=logging.DEBUG)


def get_datasets():
    datasets = []
    resp = requests.get(
        "https://api.github.com/repos/patrickzib/motiflets/git/trees/pyattimo_refactor?recursive=1"
    ).json()
    for f in resp["tree"]:
        path = f["path"]
        if "swtAttack7" not in path:
            continue
        if "momp" in path and path.endswith(".mat"):
            name = pathlib.Path(path).name.strip(".mat")
            url = f"https://github.com/patrickzib/motiflets/raw/refs/heads/pyattimo_refactor/datasets/momp/{name}.mat"
            fname = pathlib.Path("data") / (name + ".mat")
            if not fname.is_file():
                print("downloading", name)
                remote_file = requests.get(url, stream=True)
                with open(fname, "wb") as fp:
                    for chunk in remote_file.iter_content(chunk_size=1024):
                        fp.write(chunk)
            datasets.append(name)
    return datasets


datasets = get_datasets()
# datasets = ["Bird12-Week3_2018_1_10"]
# datasets = ["recorddata"]

support = 9
windows = [512, 1024, 2048, 4096, 8192]
windows = [16384]

with open("results.csv", "w") as fp:
    writer = csv.DictWriter(fp, ["timestamp", "dataset", "name", "w", "delta", "mem", "support", "time_s", "cnt_confirmed", "cnt_estimated"])
    writer.writeheader()
    for dataset, w in itertools.product(datasets, windows):
        print("============== dataset", dataset, "w", w)
        path = f"data/{dataset}.mat"
        data = scipy.io.loadmat(path)
        for name in data.keys():
            print("----------", name)
            if not hasattr(data[name], "shape"):
                continue
            ts = data[name].flatten()
            n = ts.shape[0]
            if n < 10:
                continue
            start = time.time()
            mem = "4GB"
            delta = 0.1
            m_iter = pyattimo.MotifletsIterator(
                ts,
                w,
                delta=delta,
                support=support,
                max_memory=mem,
                exclusion_zone=w // 2,
                stop_on_threshold=True,
                fraction_threshold=math.log(n) / n,
                observability_file=f"observe-dataset-{dataset}-{name}-support{support}-w{w}-mem{mem}-delta{delta}.csv",
            )

            cnt_confirmed = 0
            cnt_estimated = 0
            for m in m_iter:
                print(m, "support", m.support, "lower bound: ", m.lower_bound)
                if m.extent == m.lower_bound:
                    cnt_confirmed += 1
                else:
                    cnt_estimated += 1

            end = time.time()
            print("Discovered motiflets in", end - start, "seconds")
            writer.writerow(
                {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "dataset": dataset,
                    "name": name,
                    "w": w,
                    "delta": delta,
                    "mem": mem,
                    "support": support,
                    "time_s": end - start,
                    "cnt_confirmed": cnt_confirmed,
                    "cnt_estimated": cnt_estimated
                }
            )
