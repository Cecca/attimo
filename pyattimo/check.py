# A scratch Python file to try out the API
import pyattimo
import time
import logging
import numpy as np
import scipy
import math

logging.basicConfig(level=logging.DEBUG)

dataset = "FingerFlexionECoG"
path = f"data/{dataset}.mat"

print("load_data")
ts = scipy.io.loadmat(path)[dataset].flatten()
print(ts.shape)
n = ts.shape[0]
w = 4096
k = 9


start = time.time()
m_iter = pyattimo.MotifletsIterator(
    ts,
    w,
    delta=0.5, 
    support=9, 
    max_memory="20GB", 
    exclusion_zone=w//2, 
    stop_on_threshold=False, 
    fraction_threshold=math.log(n)/n,
    observability_file="obs.csv",
)

for m in m_iter:
    print(m, "support", m.support, type(m))

end = time.time()
print("Discovered motiflets in", end - start, "seconds")
