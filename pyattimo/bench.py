#!/usr/bin/env python

import pyattimo
import time
from motiflets.plotting import *

k = 20
motif_length = 1000
# ts = pyattimo.load_dataset("ecg", 30000)
# ml = Motiflets(ds_name="ECG", series=ts)
# dists, motiflets, elbow_points = ml.fit_k_elbow(k_max=k+1, motif_length=motif_length)
# print(dists[-1])
# print(sorted( motiflets[-1] ))
#
# start = time.time()
# m = pyattimo.motiflet_brute_force(ts, motif_length, support=k)
# end = time.time()
# print("elapsed", end - start)
# print(sorted(m.indices))
# print(m.extent, m.extent * m.extent)

for thousands in [30, 50, 100, 300, 500]:
    n = thousands * 1000
    ts = pyattimo.load_dataset("ecg", n)
    start = time.time()
    m = pyattimo.motiflet_brute_force(ts, 1000, support=k)
    end = time.time()
    print("%5d | %.4f" % (n, (end - start)))

