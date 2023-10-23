#!/usr/bin/env python

import pyattimo
import time
from motiflets.plotting import *

ts = pyattimo.load_dataset("ecg", 30000)

k = 16
motif_length = 1000
ml = Motiflets(ds_name="ECG", series=ts)
dists, motiflets, elbow_points = ml.fit_k_elbow(k_max=k+1, motif_length=motif_length)
print(dists[-1])
print(sorted( motiflets[-1] ))

start = time.time()
m = pyattimo.motiflet_brute_force(ts, motif_length, support=k)
end = time.time()
print("elapsed", end - start)
print(sorted(m.indices))
print(m.extent, m.extent * m.extent)

# for k in [5, 10, 15, 20]:
#     start = time.time()
#     m = pyattimo.motiflet_brute_force(ts, 1000, support=k)
#     end = time.time()
#     print("%d | %.4f" % (k, (end - start)))


