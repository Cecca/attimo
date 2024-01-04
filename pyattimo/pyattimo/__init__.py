import numpy as np
from .pyattimo import *

DATASETS = {
    "astro": "https://figshare.com/ndownloader/files/36982360",
    "ecg": "https://figshare.com/ndownloader/files/36982384",
    "freezer": "https://figshare.com/ndownloader/files/36982390",
    "gap": "https://figshare.com/ndownloader/files/36982396",
}


def load_dataset(dataset, prefix=None):
    import numpy
    from urllib.request import urlretrieve
    import os

    outfname = dataset + ".csv.gz"
    if not os.path.isfile(outfname):
        print("Downloading dataset")
        urlretrieve(DATASETS[dataset], outfname)

    if prefix is not None:
        return numpy.loadtxt(outfname)[:prefix]
    else:
        return numpy.loadtxt(outfname)


def pairwise_distance_distribution(ts, w, samples=1000000, seed=1234):
    from numpy.lib.stride_tricks import sliding_window_view

    def znormalize(xs):
        means = xs.mean(axis=1)
        stds = xs.std(axis=1)
        return (xs - means[:, np.newaxis]) / stds[:, np.newaxis]

    dat = sliding_window_view(ts, w)

    gen = np.random.default_rng(seed)
    indices = gen.integers(0, ts.shape[0] - w, size=(2, samples))
    left = znormalize(dat[indices[0, :]])
    right = znormalize(dat[indices[1, :]])
    return np.linalg.norm(left - right, axis=1)
