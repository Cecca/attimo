# A scratch Python file to try out the API
import pyattimo

ts = pyattimo.load_dataset("ecg")

m_iter = pyattimo.MotifletsIterator(
    ts, w=1000, max_k=10
)

for m in m_iter:
    print(m)
