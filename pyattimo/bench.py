#/usr/bin/env python

import pandas as pd
import pyattimo
import time
from motiflets.plotting import *

k = 20
motif_length = 1000

res = []

for thousands in [30, 50, 100]:
    n = thousands * 1000
    ts = pyattimo.load_dataset("ecg", n)

    ml = Motiflets(ds_name="ECG", series=ts)
    start = time.time()
    dists, motiflets, elbow_points = ml.fit_k_elbow(k_max=k+1, motif_length=motif_length, 
                                                    plot_elbows=False, plot_motifs_as_grid=False)
    end = time.time()
    baseline = end - start

    start = time.time()
    m = pyattimo.motiflet_brute_force(ts, motif_length, support=k)
    end = time.time()
    rust_time = end - start
    res.append({
        "n": n,
        "rust_time": rust_time,
        "baseline": baseline
    })

res = pd.DataFrame(res)
print(res)


