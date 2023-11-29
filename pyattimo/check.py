# A scratch Python file to try out the API
import pyattimo
import time
import logging

logging.basicConfig(level=logging.INFO)

ts = pyattimo.load_dataset("ecg")

start = time.time()
m_iter = pyattimo.MotifletsIterator(
    ts, w=1000, max_k=30
)

for m in m_iter:
    print(m)

end = time.time()
print("Discovered motiflets in", end - start, "seconds")
