# A scratch Python file to try out the API
import pyattimo
import time
import logging
import numpy as np
import scipy
import math
import requests
import sys
import pathlib
import itertools

logging.basicConfig(level=logging.INFO)


def get_datasets():
    datasets = []
    resp = requests.get(
        "https://api.github.com/repos/patrickzib/motiflets/git/trees/pyattimo_refactor?recursive=1"
    ).json()
    for f in resp["tree"]:
        path = f["path"]
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
datasets = ["Bird12-Week3_2018_1_10"]

# dataset = "FingerFlexionECoG"


k = 9
windows = [512, 1024, 2048, 4096, 8192]
windows = [1024]

for dataset, w in itertools.product(datasets, windows):
    print("============== dataset", dataset, "w", w)
    path = f"data/{dataset}.mat"
    data = scipy.io.loadmat(path)
    for name in data.keys():
        if not hasattr(data[name], "shape"):
            continue
        ts = data[name].flatten()
        np.savetxt(f"data/{dataset}.csv", ts, fmt="%.8f")
        n = ts.shape[0]
        if n < 10:
            continue
        start = time.time()
        m_iter = pyattimo.MotifletsIterator(
            ts,
            w,
            delta=0.5,
            support=9,
            max_memory="20GB",
            exclusion_zone=w // 2,
            stop_on_threshold=True,
            fraction_threshold=math.log(n) / n,
            observability_file="obs.csv",
        )

        for m in m_iter:
            print(m, "support", m.support, "lower bound: ", m.lower_bound)

        end = time.time()
        print("Discovered motiflets in", end - start, "seconds")
