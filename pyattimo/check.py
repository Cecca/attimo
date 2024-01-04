# A scratch Python file to try out the API
import pyattimo
import time
import logging

logging.basicConfig(level=logging.DEBUG)

ts = pyattimo.load_dataset("ecg", 1500)
w = 25
k = 30
start = time.time()
m_iter = pyattimo.MotifletsIterator(ts, w=w, max_k=k)

for m in m_iter:
    print(m)

end = time.time()
print("Discovered motiflets in", end - start, "seconds")
