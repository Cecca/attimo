import pandas as pd
import numpy as np
import subprocess as sp
import time
import sqlite3
import os
import socket
import multiprocessing

HOSTNAME = socket.gethostname()
NUM_CPUS = multiprocessing.cpu_count()


def install_scamp():
    import shlex
    try:
        if sp.run(["./SCAMP", "--version"]).returncode == 0:
            return
    except:
        print("Installing SCAMP....")
        commands = [
            ("git clone https://github.com/zpzim/SCAMP /tmp/SCAMP"  , "/tmp"),
            ("git checkout 70f1e63"                                 , "/tmp/SCAMP"),
            ("git submodule update --init --recursive"              , "/tmp/SCAMP"),
            ("mkdir build"                                          , "/tmp/SCAMP"),
            ("cmake .."                                             , "/tmp/SCAMP/build"),
            ("cmake --build . --config Release"                     , "/tmp/SCAMP/build"),
            ("cp /tmp/SCAMP/build/SCAMP ."                          , "."),
        ]
        for cmd, d in commands:
            sp.run(shlex.split(cmd), cwd=d).check_returncode()


def get_db():
    db = sqlite3.connect("attimo-results.db", isolation_level=None)

    db.execute("""
    CREATE TABLE IF NOT EXISTS scamp (
        hostname     TEXT,
        dataset      TEXT,
        threads      INT,
        window       INT,
        time_s       REAL,
        motif_pairs  TEXT
    );
    """)

    db.execute("""
    CREATE TABLE IF NOT EXISTS attimo (
        hostname     TEXT,
        dataset      TEXT,
        threads      INT,
        repetitions  INT,
        delta        REAL,
        seed         INT,
        window       INT,
        motifs       INT,
        time_s       REAL,
        log          TEXT,
        motif_pairs  TEXT
    );
    """)

    return db


def prefix(path, n):
    fname, ext = os.path.splitext(path)
    outpath = fname + "-{}{}".format(n, ext)
    if not os.path.isfile(outpath):
        with open(outpath, "w") as fp:
            sp.run(["head", "-n{}".format(n), path], stdout=fp)
    return outpath


def get_datasets():
    return [
        ("data/Steamgen.csv", 300)
        # (prefix("data/ECG.csv", 100000), 1500),
        # (prefix("data/ECG.csv", 200000), 1500)
    ]

def remove_trivial(df, w):
    def is_trivial(a1, b1, a2, b2):
        idxs = sorted([a1, b1, a2, b2])
        return idxs[0] + w > idxs[1] or idxs[1] + w > idxs[2] or idxs[2] + w > idxs[3]
    df = df[df['b'] - df['a'] > w]
    prev = df[['a', 'b']].iloc[0]
    prev = prev['a'], prev['b']
    trivial = [False]
    rows = df[['a', 'b']].iterrows()
    next(rows) # skip the first row
    for _, row in rows:
        ap, bp = prev
        a = row['a']
        b = row['b']
        t = is_trivial(a, b, ap, bp)
        trivial.append(t)
        if not t:
            prev = a, b
    df['trivial'] = trivial
    return df[df['trivial'] == False]



def run_scamp():
    db = get_db()
    datasets = get_datasets()
    threads = NUM_CPUS
    for dataset, window in datasets:
        print(f"running on {dataset} with w={window} and {threads} threads... ", end="")
        start = time.time()
        sp.run([
            "./SCAMP", 
            "--window={}".format(str(window)), 
            "--input_a_file_name={}".format(dataset),
            "--num_cpu_workers={}".format(threads)
        ])
        end = time.time()
        elapsed = end - start
        print(f"{elapsed} seconds")
        dists = np.loadtxt('mp_columns_out')
        idxs  = np.loadtxt('mp_columns_out_index')
        df = pd.DataFrame({
            "a": np.arange(len(idxs)),
            "b": idxs.astype('int'),
            "dist": dists
        }).sort_values('dist')
        df = df[df['a'] < df['b']]
        df = remove_trivial(df, window)
        motifs = df.head(100)[['a', 'b', 'dist']].to_json(orient='records')

        db.execute("""
            INSERT INTO scamp VALUES (:hostname,:dataset,:threads,:window,:elapsed,:motifs);
            """,
            {
                "hostname": HOSTNAME,
                "dataset": dataset,
                "threads": threads,
                "window": window,
                "elapsed": elapsed,
                "motifs": motifs
            }
        )



if __name__ == "__main__":
    install_scamp()
    run_scamp()
    
