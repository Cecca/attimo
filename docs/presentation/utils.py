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


def plot_ts(ts, w=None, highlight=[], axis=False, colors=["forestgreen", "steelblue", "orange"]):
    plt.figure(figsize=(12, 2))
    color = "black" if len(highlight) == 0 else "grey"
    plt.plot(ts, c=color)
    old_lim = plt.gca().get_ylim()
    for pos, color in zip(highlight, colors):
        assert w is not None
        plt.gca().add_patch(Rectangle((pos, ts.min()), w, ts.max(),
                                      facecolor='lightgrey'))
        plt.plot(np.arange(pos, pos+w), ts[pos:pos+w], c=color)
    plt.gca().set_ylim(old_lim)
    if not axis:
        plt.axis('off')
    plt.xlabel("time")
    plt.tight_layout()


def plot_catalog(ts, occs, w, height=1, spacing=1, colors=plt.colormaps.get("tab10").colors, labels=None):
    plt.figure(figsize=(3, height*2*len(occs)))
    ax = plt.gca()
    offset = 0
    for pair, c in zip(occs, colors):
        xline = w * 1.24
        xtext = w * 1.3
        plt.plot([xline, xline], [-offset, -offset - spacing], linewidth=0.5, color="black")
        i, j = pair
        d = zeucl(ts[i:i+w], ts[j:j+w])
        plt.text(xtext, -offset - spacing/2, "{:.3}".format(d), size=13, va='center')
        for i in pair[0:2]:
            vals = scale(ts[i:i+w])
            ax.plot(vals - offset, color=c)
            if labels is not None:
                lab = labels.get(i, "?")
                plt.text(int(len(vals)*1.1), -offset, lab, size=16, va="center")
            #offset += (vals.max() - vals.min()) * spacing
            offset += spacing
            ax.axis("off")


def plot_in_context(ts, occs, w, colors=plt.colormaps.get("tab10").colors, labels = None, labelpos = dict()):
    plt.figure(figsize=(10, 1.5))
    plt.plot(ts, color="grey")
    plt.gca().axis('off')
    labeltop = ts.max() * 1.1
    for pair, c in zip(occs, colors):
        for i in pair[0:2]:
            idx = np.arange(i, i+w)
            plt.plot(idx, ts[idx], color=c)
            if labels is not None:
                pos = labelpos.get(i, labeltop)
                lab = labels.get(i, "?")
                plt.text(i + w/2, pos, lab, size=16, ha='center')

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
        plt.plot(dists, probs, label=f"τ={k}")
    plt.xlabel("Distance")
    plt.ylabel("Collision probability")
    if len(ks) > 1:
        plt.legend()
    # plt.gca().set_aspect('auto')
    plt.tight_layout()


def plot_success_by_reps(w, r, k, dist, max_reps):
    reps = np.arange(max_reps)
    plt.figure(figsize=(6, 4))
    probs = cp_pstable(dist, w, r)**k
    success = 1 - (1 - probs)**reps
    plt.plot(reps, success)
    plt.xlabel("Number of repetitions")
    plt.ylabel("Success probability")
    plt.title("For fixed τ and candidate distance")


def plot_success_p(w, r, k, ls=[10], p_threshold=None, dist=None, text=False, title=None):
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
        plt.title(f"τ={k}, repetitions={ls[0]}")
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
    if title is not None:
        plt.title(title)
    plt.gca().set_aspect('auto')
    plt.tight_layout()


def plot_execution(w, r, k, rep, max_reps=100, p_threshold=None, dist=None, figsize=(8, 5), prevs=[]):
    def success_p(dists, k, rep):
        probs_cur = cp_pstable(dists, w, r)**k
        probs_prev = cp_pstable(dists, w, r)**(k+1)
        fail_cur = (1 - probs_cur)**rep
        fail_prev = (1 - probs_prev)**(max_reps - rep)
        success = 1 - (fail_cur * fail_prev)
        return success

    end = 3
    dists = np.linspace(0, end)
    success = success_p(dists, k, rep)
    plt.figure(figsize=figsize)
    plt.plot(dists, success)
    for kk, rrep in prevs:
        plt.plot(dists, success_p(dists, kk, rrep), color="gray", linestyle=":")
    plt.xlabel("Distance")
    plt.ylabel("Success probability")

    plt.title(f"k={k}, repetitions={rep}/{max_reps}")

    if p_threshold is not None:
        plt.axhline(p_threshold, xmin=0, xmax=10,
                    color="gray", linestyle=":")
        plt.text(end, 0.91, r"1 - $\delta$ = 0.9", ha="right", fontsize=14)

    if dist is not None:
        plt.axvline(dist, ymin=0, ymax=1,
                    color="firebrick", linestyle=":")
        plt.text(dist, 0.95, r"Candidate distance", ha="left" if dist <= 1.5 else "right")
        success = success_p(dist, k, rep)
        success_s = f"{success:.2e}" if success < 0.1 else f"{success:.4f}"
        if success > 0.9:
            color = "forestgreen"
        else:
            color = "black"
        if success > 0.8:
            va = "top"
            offset = -0.01
        else:
            va = "bottom"
            offset = 0
        plt.text(
            end, success + offset, f"success probability\n= {success_s}",
            fontsize=14,
            ha="right", va=va, c=color)
        plt.axhline(success, xmin=0, xmax=end,
                    color="firebrick", linestyle=":")
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


def plot_hashes(prj, seed, r, k, size=4, title=None):
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

    plt.scatter(prj[:, 0], prj[:, 1], s=16, c=hashes, cmap="tab20")

    if title is not None:
        plt.title(title)

    plt.axis('off')
    plt.gca().set_xlim((-0.7, 0.7))
    plt.gca().set_ylim((-0.7, 0.7))
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.tight_layout()
    plt.tight_layout()


def plot_eucl(ts, w, i, j, ci, cj):
    from matplotlib import collections  as mc
    plt.figure(figsize=(6,1.5))
    a = ts[i:i+w]
    b = ts[j:j+w]
    lines = zip(enumerate(a),enumerate(b))
    lc = mc.LineCollection(lines, colors="lightgray")
    plt.gca().add_collection(lc)
    plt.plot(a, c=ci)
    plt.plot(b, c=cj)
    plt.tight_layout()


def simulate_execution():
    figsize=(8,3)
    k=4
    dist = 1
    w = 640
    prevs = []
    plot_execution(w, r=1, k=k, max_reps=100, rep=1, p_threshold=0.9, dist=dist + 1, figsize=figsize)
    plt.savefig("imgs/execution1.png", dpi=300)
    prevs.append((k, 1))

    plot_execution(w, r=1, k=k, max_reps=100, rep=100, p_threshold=0.9, dist=dist, prevs=prevs, figsize=figsize)
    plt.savefig("imgs/execution2.png", dpi=300)
    prevs.append((k, 100))

    k = 3
    plot_execution(w, r=1, k=k, max_reps=100, rep=1, p_threshold=0.9, dist=dist, prevs=prevs, figsize=figsize)
    plt.savefig("imgs/execution3.png", dpi=300)
    prevs.append((k, 1))

    plot_execution(w, r=1, k=k, max_reps=100, rep=15, p_threshold=0.9, dist=dist, prevs=prevs, figsize=figsize)
    plt.savefig("imgs/execution4.png", dpi=300)


if __name__ == "__main__":
    # simulate_execution()
    w = 400
    ts = np.loadtxt("insect15.txt")
    plot_eucl(ts, w=w, i=6974, j=8490, ci="steelblue", cj="orange")
    plt.savefig("insect.png")
    plot_eucl(ts, w=w, i=6974, j=200, ci="steelblue", cj="forestgreen")
    plt.savefig("insect-far.png")
