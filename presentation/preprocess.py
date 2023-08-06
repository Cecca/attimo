import polars as pl
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import TSNE
# from umap import UMAP
import matplotlib.pyplot as plt
import sys
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

w = 640
i, j = 643, 8724
motif_idxs = np.array([i, j], dtype=np.int64)

np.random.seed(1234)
steamgen = pl.read_csv("steamgen.csv")
steam = steamgen.select("steam flow").to_numpy()[:, 0]
np.savetxt("d3-imgs/steam.csv", steam, fmt="%f", header="y", comments="")
sample_idxs = np.random.choice(
    np.arange(steam.shape[0]-w, dtype=np.int64), 20)
p_idxs = np.concatenate([motif_idxs, sample_idxs])
np.savetxt("d3-imgs/indices.csv", p_idxs,
           fmt="%f", header="index", comments="")

windows = sliding_window_view(steam, w)
swindows = windows[p_idxs, :]
embedder = PCA(n_components=2)
prj = embedder.fit_transform(swindows)
np.random.seed(1234)
prj = prj + np.random.normal(scale=5, size=prj.shape)
prj = MinMaxScaler(feature_range=(-0.5, 0.5)).fit_transform(prj)
np.savetxt("d3-imgs/prj.csv", prj, delimiter=",",
           fmt="%f", header="x,y", comments="")

# for i in range(swindows.shape[0]):
#     plt.figure(figsize=(10, 2))
#     plt.plot(swindows[i, :])
#     plt.title(f"sub {i}")
#     plt.tight_layout()
#     plt.savefig(f"sub-{i}.png")
#
plt.figure(figsize=(10, 10))

plt.scatter(prj[:, 0], prj[:, 1], c="blue")
plt.scatter(prj[[0, 1], 0],
            prj[[0, 1], 1], c="red")

plt.tight_layout()
plt.savefig("embedding.png", format="png")
