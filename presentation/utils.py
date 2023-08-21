import polars as pl
from scipy.stats import norm as normal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import sys
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def plot_ts(ts, w=None, highlight=[]):
    plt.figure(figsize=(12, 2))
    plt.plot(ts)
    for pos in highlight:
        assert w is not None
        plt.gca().add_patch(Rectangle((pos, ts.min()), w, ts.max(),
                                      facecolor='lightgrey'))
        plt.plot(np.arange(pos, pos+w), ts[pos:pos+w], c="red")
    plt.axis('off')
    plt.tight_layout()


def scale(ys):
    return (ys - ys.mean()) / ys.std()


def zeucl(x, y):
    return np.linalg.norm(scale(x) - scale(y))


def plot_subs(ts, w, idxs=[]):
    plt.figure(figsize=(6, 2))
    # plt.plot(ts)
    for pos in idxs:
        assert w is not None
        plt.plot(scale(ts[pos:pos+w]))
    # plt.axis('off')
    plt.tight_layout()


def plot_projection_circle(prj, highlight=[]):
    plt.figure(figsize=(4, 4))
    radius = 1.1 * \
        np.linalg.norm(prj[np.argmax(np.linalg.norm(prj, axis=1)), :])
    circ = Circle((0, 0), radius, fill=None, color="lightgray")
    plt.scatter(prj[:, 0], prj[:, 1], s=16)
    for i in highlight:
        plt.scatter(prj[i, 0], prj[i, 1], s=32, c="red", marker="D")
    plt.gca().add_patch(circ)
    plt.axis('off')
    plt.gca().set_xlim((-0.7, 0.7))
    plt.gca().set_ylim((-0.7, 0.7))
    plt.gca().set_aspect('equal')
    plt.tight_layout()


def cp_pstable(dist, dim, r):
    return 1.0 - 2.0 * normal.cdf(-r / dist) - (2.0 / (np.sqrt(np.pi * 2.0) * (r / dist))) * (1.0 - np.exp(-r * r / (2.0 * dist * dist)))


def plot_cp(w, r, ks=[1]):
    dists = np.arange(1000) / 100
    plt.figure(figsize=(6, 4))
    for k in ks:
        probs = cp_pstable(dists, w, r)**k
        plt.plot(dists, probs, label=f"k={k}")
    plt.xlabel("Distance")
    plt.ylabel("Collision probability")
    if len(ks) > 1:
        plt.legend()
    plt.gca().set_aspect('auto')
    plt.tight_layout()


def plot_success_p(w, r, k, ls=[10], p_threshold=None, dist=None, text=False):
    dists = np.arange(1000) / 100
    plt.figure(figsize=(6, 4))
    for L in ls:
        probs = cp_pstable(dists, w, r)**k
        success = 1 - (1 - probs)**L
        plt.plot(dists, success, label=f"repetitions={L}")
    plt.xlabel("Distance")
    plt.ylabel("Success probability")
    if len(ls) > 1:
        plt.legend()
    if text:
        plt.title(f"k={k}, repetitions={ls[0]}")
    if p_threshold is not None:
        plt.axhline(p_threshold, xmin=0, xmax=10,
                    color="gray", linestyle=":")
        plt.text(10, 0.91, r"1 - $\delta$", ha="right")
    if dist is not None:
        assert len(ls) == 1
        plt.axvline(dist, ymin=0, ymax=1,
                    color="firebrick", linestyle=":")
        plt.text(dist, 0.95, r"Current candidate distance", ha="left")
        L = ls[0]
        prob = cp_pstable(dist, w, r)**k
        success = 1 - (1 - prob)**L
        success_s = f"{success:.2e}" if success < 0.1 else f"{success:.2f}"
        plt.text(
            10, success, f"Current success probability = {success_s}",
            ha="right", va="bottom")
        plt.axhline(success, xmin=0, xmax=10,
                    color="gray", linestyle=":")
    plt.gca().set_aspect('auto')
    plt.tight_layout()


def plot_execution(w, r, k, rep, max_reps=100, p_threshold=None, dist=None, figsize=(8, 5)):
    def success_p(dists):
        probs_cur = cp_pstable(dists, w, r)**k
        probs_prev = cp_pstable(dists, w, r)**(k+1)
        fail_cur = (1 - probs_cur)**rep
        fail_prev = (1 - probs_prev)**(max_reps - rep)
        success = 1 - (fail_cur * fail_prev)
        return success

    dists = np.arange(1000) / 100
    success = success_p(dists)
    plt.figure(figsize=figsize)
    plt.plot(dists, success)
    plt.xlabel("Distance")
    plt.ylabel("Success probability")

    plt.title(f"k={k}, repetitions={rep}/{max_reps}")

    if p_threshold is not None:
        plt.axhline(p_threshold, xmin=0, xmax=10,
                    color="gray", linestyle=":")
        plt.text(10, 0.91, r"1 - $\delta$ = 0.9", ha="right", fontsize=14)

    if dist is not None:
        plt.axvline(dist, ymin=0, ymax=1,
                    color="firebrick", linestyle=":")
        plt.text(dist, 0.95, r"Current candidate distance", ha="left")
        success = success_p(dist)
        success_s = f"{success:.2e}" if success < 0.1 else f"{success:.4f}"
        if success > 0.9:
            color = "forestgreen"
        else:
            color = "black"
        if success > 0.8:
            va = "top"
            offset = -0.1
        else:
            va = "bottom"
            offset = 0
        plt.text(
            10, success + offset, f"Current success probability = {success_s}",
            fontsize=14,
            ha="right", va=va, c=color)
        plt.axhline(success, xmin=0, xmax=10,
                    color="gray", linestyle=":")
    plt.gca().set_aspect('auto')
    plt.tight_layout()


def hash(x, r, seed):
    n, w = x.shape
    np.random.seed(seed)
    if seed == 0:
        a = np.array([1, 0])
    else:
        a = np.random.normal(size=w)
    b = np.random.uniform(0, r)
    dots = []
    for i in range(n):
        dots.append(np.dot(a, x[i, :]))
    dots = np.array(dots)
    h = np.floor((dots + b) / r)
    return h, a, b


def hash_many(x, k, r, seed):
    hashes = []
    a_vals = []
    b_vals = []
    for kval in range(k):
        h, a, b = hash(x, r, seed + kval)
        a_vals.append(a)
        b_vals.append(b)
        hashes.append(h)
    hashes = np.array(hashes)
    hash_values = np.zeros(x.shape[0], dtype=np.int32)
    hmap = dict()
    assert hashes.shape[1] == x.shape[0]
    for i in range(hashes.shape[1]):
        h = tuple(hashes[:, i])
        if h not in hmap:
            hmap[h] = len(hmap)
        hash_values[i] = hmap[h]

    return hash_values, a_vals, b_vals


def plot_hashes(prj, seed, r, k, size=4):
    plt.figure(figsize=(size, size))
    radius = 1.1 * \
        np.linalg.norm(prj[np.argmax(np.linalg.norm(prj, axis=1)), :])
    circ = Circle((0, 0), radius, fill=None, color="lightgray")
    plt.gca().add_patch(circ)

    hashes, a_vals, b_vals = hash_many(prj, k, r, seed)

    for a, b in zip(a_vals, b_vals):
        m = (a[1] / a[0])
        clipper = Circle((0, 0), radius, transform=plt.gca().transData)

        if k == 1:
            plt.axline((0, 0), slope=m, c="black", clip_path=clipper)

        for i in range(-10, 10):
            x = (r*i - b) / (a[0]*(1+m*m))
            y = m*x
            p = np.array([x, y])
            norm_p = np.linalg.norm(p)
            if abs(norm_p) <= radius:
                plt.axline(p, slope=-1/m, c="lightgray",
                           clip_path=clipper)

    plt.scatter(prj[:, 0], prj[:, 1], s=16, c=hashes, cmap="tab10")

    plt.axis('off')
    plt.gca().set_xlim((-0.7, 0.7))
    plt.gca().set_ylim((-0.7, 0.7))
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.tight_layout()
    plt.tight_layout()

if __name__ == "__main__":
    figsize=(8,3)
    k=4
    dist = 1
    w = 640
    plot_execution(w, r=1, k=k, max_reps=100, rep=1, p_threshold=0.9, dist=dist + 1, figsize=figsize)
    plt.savefig("imgs/execution1.png", dpi=300)

    plot_execution(w, r=1, k=k, max_reps=100, rep=100, p_threshold=0.9, dist=dist, figsize=figsize)
    plt.savefig("imgs/execution2.png", dpi=300)

    k = 3
    plot_execution(w, r=1, k=k, max_reps=100, rep=1, p_threshold=0.9, dist=dist, figsize=figsize)
    plt.savefig("imgs/execution3.png", dpi=300)

    plot_execution(w, r=1, k=k, max_reps=100, rep=15, p_threshold=0.9, dist=dist, figsize=figsize)
    plt.savefig("imgs/execution4.png", dpi=300)
