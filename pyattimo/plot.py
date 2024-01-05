import numpy as np
import pyattimo
from matplotlib import pyplot as plt
import seaborn as sns

ts = np.loadtxt("data/npo141.csv")
dists = pyattimo.pairwise_distance_distribution(ts, 100)
sns.kdeplot(dists)
plt.savefig("distr.png")
