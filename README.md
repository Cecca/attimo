ATTIMO: AdapTive TImeseries MOtifs
=====================================

This is the implementation of the ATTIMO algorithm for fast mining
of timeseries motifs, with probabilistic guarantees.

The inner workings and guarantees of the algorithm are described in [this paper](https://www.vldb.org/pvldb/vol15/p3841-ceccarello.pdf>).

If you find this software useful for your research, please use the following citation:

```
@article{DBLP:journals/pvldb/CeccarelloG22,
  author    = {Matteo Ceccarello and
               Johann Gamper},
  title     = {Fast and Scalable Mining of Time Series Motifs with Probabilistic
               Guarantees},
  journal   = {Proc. {VLDB} Endow.},
  volume    = {15},
  number    = {13},
  pages     = {3841--3853},
  year      = {2022},
  url       = {https://www.vldb.org/pvldb/vol15/p3841-ceccarello.pdf},
  timestamp = {Wed, 11 Jan 2023 17:06:38 +0100},
  biburl    = {https://dblp.org/rec/journals/pvldb/CeccarelloG22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## Installation

First, you need to install Rust on your system. The simplest way is to visit
[https://rustup.rs/]() and follow the instructions there. You will need the
`nightly` toolchain:

    curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain nightly

### Python wrapper

To install the Python wrapper, issue the following commands (preferably in a virtual environment)

```
pip install maturin
cd pyattimo
maturin develop --release
```

### Rust CLI

To install the Rust cli, you can just run

    cargo install --locked --force --path .

At this point, you should have the `attimo` command available on your system.

## Using the Python wrapper

### Motiflets

There is an experimental implementation of the [k-motiflets definition](https://www.vldb.org/pvldb/vol16/p725-schafer.pdf)
that you can use as follows

```python
import pyattimo

# load a dataset, any list of numpy array of floats works fine
# The following call loads the first 100000 points of the ECG 
# dataset (which will be downloaded from the internet)
ts = pyattimo.load_dataset('ecg', 100000)

# Now we can find k-motiflets:
#  - w is the window length
#  - support is the number of subsequences in the motiflet (k in the motiflet paper)
#  - repetitions is the number of LSH repetitions
m = pyattimo.motiflet(ts, w=1000, support=5, repetitions=512)

# The motiflet object allows to:
#   - get the indices of the subsequences
m.indices
#   - get the extent of the motiflet
m.extent
#   - plot the motiflet, showing it in a window or returning the 
#     plot object (default) for embedding in a notebook
m.plot(show=False)
#   - get the values of the i-th subsequence in the motiflet
m.values(2) # for the second subsequence
#   - get the z-normalized values of the i-th subsequence in the motiflet
m.zvalues(2) # for the second subsequence
```

## Running the CLI

Executing the command with no arguments shows a short help message.

## Data format

`attimo` works with univariate time series in a very common format: text files
with one value per line:

    1.1292
    1.1096
    1.0986
    1.0925
    1.0889
    1.0815
    1.0767
    1.073
    1.0681
    1.0608

You can find some sample data files
[here](https://figshare.com/articles/dataset/Datasets/20747617).

