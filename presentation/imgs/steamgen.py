import numpy as np
import stumpy
import pandas
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# https://zenodo.org/record/4273921/files/STUMPY_Basics_steamgen.csv?download=1
data = pandas.read_csv('figs/steamgen.csv')
flow = data['steam flow']

fig = plt.figure(figsize=(12, 3), dpi=300)
plt.suptitle('Steamgen Dataset', fontsize='14', x=0.1, ha='left')
flow.plot()
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(True)
ax.spines['left'].set_visible(False)
plt.savefig(
    "figs/steamgen.png",
    transparent=True,
    bbox_inches='tight'
)

m = 640
baseidx = 4000
window = flow.iloc[baseidx:baseidx+m]
window.plot()
plt.text(baseidx, 35, s="Subsequence", fontsize=12)
plt.savefig(
    "figs/steamgen-with-window.png",
    transparent=True,
    bbox_inches='tight'
)

# Motifs, taken from https://stumpy.readthedocs.io/en/latest/Tutorial_STUMPY_Basics.html
fig, axs = plt.subplots(2, figsize=(12, 6), dpi=300)
axs[0].plot(flow, alpha=0.5, linewidth=1)
axs[0].plot(flow.iloc[643:643+m])
axs[0].plot(flow.iloc[8724:8724+m])
axs[0].spines['top'].set_visible(False)
axs[0].spines['right'].set_visible(False)
axs[0].spines['bottom'].set_visible(True)
axs[0].spines['left'].set_visible(False)
rect = Rectangle((643, 0), m, 40, facecolor='lightgrey')
axs[0].add_patch(rect)
rect = Rectangle((8724, 0), m, 40, facecolor='lightgrey')
axs[0].add_patch(rect)
axs[1].plot(flow.values[643:643+m], color='C1')
axs[1].plot(flow.values[8724:8724+m], color='C2')
plt.savefig(
    "figs/steamgen-with-motifs.png",
    transparent=True,
    bbox_inches='tight'
)

# Matrix profile
mp = stumpy.stump(flow.values, m)
motif_idx = np.argsort(mp[:, 0])[0]
nearest_neighbor_idx = mp[motif_idx, 1]
fig, axs = plt.subplots(2, sharex=True, figsize=(12, 6), dpi=300)
for ax in axs:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(False)
axs[0].plot(flow.values)
rect = Rectangle((motif_idx, 0), m, 40, facecolor='lightgrey')
axs[0].add_patch(rect)
rect = Rectangle((nearest_neighbor_idx, 0), m, 40, facecolor='lightgrey')
axs[0].add_patch(rect)
axs[1].axvline(x=motif_idx, linestyle="dashed", color="C1")
axs[1].axvline(x=nearest_neighbor_idx, linestyle="dashed", color='C1')
axs[1].plot(mp[:, 0], color='C1')
plt.savefig(
    "figs/steamgen-with-matrix-profile.png",
    transparent=True,
    bbox_inches='tight'
)





