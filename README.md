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

After that, you can just run

    cargo install --locked --force --path .

At this point, you should have the `attimo` command available on your system.

## Running

Executing the command with no arguments shows a short help message.

    ❯ attimo
    Required positional arguments not provided:
        path
    Required options not provided:
        --window
        --motifs
        --memory

The flag `--help` gives a more comprehensive overview

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
[here](https://www.dropbox.com/sh/ookuitgxqc1z1op/AACw_kkI7Xcop76UBytEIoFsa?dl=0).

