from .pyattimo import *

DATASETS = {
    "astro": "https://figshare.com/ndownloader/files/36982360",
    "ecg": "https://figshare.com/ndownloader/files/36982384",
    "freezer": "https://figshare.com/ndownloader/files/36982390",
    "gap": "https://figshare.com/ndownloader/files/36982396"
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
