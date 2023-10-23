#!/usr/bin/env python

import pyattimo
import time
from motiflets.plotting import *

ts = pyattimo.load_dataset("ecg", 30000)

k = 7
motif_length = 1000
ml = Motiflets(ds_name="ECG", series=ts)
dists, motiflets, elbow_points = ml.fit_k_elbow(k_max=k, motif_length=motif_length)
print(dists)
print(motiflets)
print(elbow_points)

m = pyattimo.motiflet(ts, motif_length, support=k)
print(sorted(m.indices))
print(m.extent)

# for k in [20]: #[5, 10, 15, 20]:
#     start = time.time()
#     m = pyattimo.motiflet(ts, 1000, support=k)
#     end = time.time()
#     print("%d | %.4f" % (k, (end - start)))
