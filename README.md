ATTIMO: ApproximaTe TImeseries MOtifs
=====================================

This is the implementation of the ATTIMO algorithm for fast approximate mining
of timeseries motifs, with probabilistic guarantees.

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

    $ attimo                                                                                                                                                            (base) 
    Required positional arguments not provided:
        path
    Required options not provided:
        --window
        --motifs
        --memory

The flag `--help` gives a more comprehensive overview

    ❯ attimo --help                                                                                                                                                     (base) 
    Usage: attimo <path> -w <window> --motifs <motifs> [--delta <delta>] -m <memory> [--plot] [--prefix <prefix>] [--seed <seed>]

    ATTIMO computes ApproximaTe TImeseries MOtifs.

    Options:
      -w, --window      subsequcence length
      --motifs          the number of motifs to look for
      --delta           failure probability of the LSH scheme
      -m, --memory      the amount of memory the algorithm can use
      --plot            open a browser window with a plot of the approximate matrix
                        profile
      --prefix          consider only the given number of points from the input
      --seed            seed for the psudorandom number generator
      --help            display usage information

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

