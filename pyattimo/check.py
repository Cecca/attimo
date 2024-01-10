# A scratch Python file to try out the API
import pyattimo
import time
import logging
import numpy as np
from matplotlib import pyplot as plt

logging.basicConfig(level=logging.INFO)

# ts = pyattimo.load_dataset("ecg", 1500)
ts = np.loadtxt("../data/npo141.csv")
ts = ts[::26]
print(ts.shape)
w = 145
k = 20

ms = pyattimo.motiflet_brute_force(ts, w=w, support=20, exclusion_zone=w // 2)
print(ms)

plt.figure()
for i in ms.indices:
    plt.plot(ts[i : i + w])

plt.savefig("motiflets.png")


start = time.time()
m_iter = pyattimo.MotifletsIterator(ts, w=w, max_k=k, exclusion_zone=w // 2)

for m in m_iter:
    print(m, "support", m.support)
    plt.figure()
    for i in ms.indices:
        plt.plot(ts[i : i + w])
    plt.savefig(f"motiflets-{m.support}.png")
    plt.close()

end = time.time()
print("Discovered motiflets in", end - start, "seconds")
