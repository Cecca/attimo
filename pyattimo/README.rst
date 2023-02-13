pyATTIMO
========

This is a python wrapper for `ATTIMO <https://cecca.github.io/attimo/>`_, a fast algorithm for mining time series motifs, with probabilistic guarantees.

The inner workings and guarantees of the algorithm are described in `this paper <https://www.vldb.org/pvldb/vol15/p3841-ceccarello.pdf>`_.

If you find this software useful for your research, please use the following citation::

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



Installation
------------

`pyATTIMO` is a Rust library wrapped in Python. Therefore, if a wheel is available for your platform, you can install it simply by invoking::

    pip install pyattimo

Otherwise, you need the Rust toolchain installed to be able to compile it.
The simplest way is to visit https://rustup.rs/ and follow the instructions there. You will need the
`nightly` toolchain::

    curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain nightly

After that, you can just run::

    pip install pyattimo

At this point, you should have the `pyattimo` library available in your interpreter.

Usage
-----

In essence, the library provides an iterator over the motifs of the given time series.
The following snippet illustrates the basic usage:

.. code-block:: python

    import pyattimo

    # Load an example time series
    ts = pyattimo.load_dataset("ecg", prefix=1000000)

    # Create the motifs iterator
    motifs = pyattimo.MotifsIterator(ts, w=1000, max_k=100)

    # Get the top motif via the iterator interface
    m = next(motifs)

    # Plot the motif just obtained
    m.plot()

Further information and examples can be found `here <https://cecca.github.io/attimo/pyattimo.html>`_

