import json
import shlex
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

    dbver = db.execute("PRAGMA user_version").fetchone()[0]
    if dbver is None:
        dbver = 0
    print("Database version", dbver)
    # ------ Version 1 ---------
    if dbver < 1:
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
            git_sha      TEXT,
            version      INT,
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
        db.execute("PRAGMA user_version = 1;")
        print("  Bump version to 1")
    
    print("Database initialized")

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
        (prefix("data/GAP.csv",  250000), 60),
        (prefix("data/GAP.csv",  500000), 60),
        (prefix("data/GAP.csv", 1000000), 60),
        (prefix("data/GAP.csv", 1500000), 60),
        (prefix("data/GAP.csv", 2000000), 60),
        ("data/GAP.csv", 60),

        (prefix("data/EMG.csv",  500000), 100),
        (prefix("data/EMG.csv", 1000000), 100),
        (prefix("data/EMG.csv", 1500000), 100),
        (prefix("data/EMG.csv", 2000000), 100),
        (prefix("data/EMG.csv", 2500000), 100),
        (prefix("data/EMG.csv", 3000000), 100),
        ("data/EMG.csv", 100),

        (prefix("data/ECG.csv", 1000000), 1000),
        (prefix("data/ECG.csv", 2000000), 1000),
        (prefix("data/ECG.csv", 3000000), 1000),
        (prefix("data/ECG.csv", 4000000), 1000),
        (prefix("data/ECG.csv", 5000000), 1000),
        (prefix("data/ECG.csv", 6000000), 1000),
        (prefix("data/ECG.csv", 7000000), 1000),
        ("data/ECG.csv", 1000),

        (prefix("data/ASTRO.csv", 100000), 100),
        (prefix("data/ASTRO.csv", 200000), 100),
        (prefix("data/ASTRO.csv", 300000), 100),
        (prefix("data/ASTRO.csv", 400000), 100),
        (prefix("data/ASTRO.csv", 500000), 100),
        (prefix("data/ASTRO.csv", 600000), 100),
        (prefix("data/ASTRO.csv", 700000), 100),
        (prefix("data/ASTRO.csv", 800000), 100),
        (prefix("data/ASTRO.csv", 900000), 100),
        (prefix("data/ASTRO.csv", 1000000), 100),
        ("data/ASTRO.csv", 100),

        ("data/HumanY.txt", 18000)
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


def run_attimo():
    gitsha = sp.check_output(shlex.split("git rev-parse HEAD"))
    sp.check_call(shlex.split("cargo install --force --locked --path ."))
    version = int(sp.check_output(["attimo", "--version"]))
    db = get_db()
    datasets = get_datasets()
    threads = NUM_CPUS
    repetitions = 50
    delta = 0.001
    for seed in [14514]: #, 1346, 2524]:
        for motifs in [1, 5, 10]:
            for dataset, window in datasets:
                # Check if already run
                execid = db.execute("""
                    select rowid from attimo
                    where hostname=:hostname
                    and version=:version
                    and dataset=:dataset
                    and threads=:threads
                    and repetitions=:repetitions
                    and delta=:delta
                    and seed=:seed
                    and window=:window
                    and motifs=:motifs
                    """,
                    {
                        "hostname": HOSTNAME,
                        "version": version,
                        "dataset": dataset,
                        "threads": threads,
                        "repetitions": repetitions,
                        "delta": delta,
                        "seed": seed,
                        "window": window,
                        "motifs":motifs,
                    }
                ).fetchone()
                if execid is not None:
                    print("experiment already executed (attimo id={})".format(execid[0]))
                    continue

                start = time.time()
                sp.run([
                    "attimo",
                    "--window", str(window),
                    "--motifs", str(motifs),
                    "--repetitions", str(repetitions),
                    "--delta", str(delta),
                    "--seed", str(seed),
                    "--log-path", "/tmp/attimo.json",
                    "--output", "/tmp/motifs.csv",
                    dataset
                ]).check_returncode()
                end = time.time()
                elapsed = end - start
                motif_pairs = pd.read_csv('/tmp/motifs.csv', names=['a', 'b','dist']).to_json(orient='records')
                with open("/tmp/attimo.json") as fp:
                    log = json.dumps([json.loads(l) for l in fp.readlines()])

                os.remove("/tmp/attimo.json")
                os.remove("/tmp/motifs.csv")

                db.execute("""
                    INSERT INTO attimo VALUES (
                        :hostname,
                        :gitsha,
                        :version,
                        :dataset,
                        :threads,
                        :repetitions,
                        :delta,
                        :seed,
                        :window,
                        :motifs,
                        :time_s,
                        :log,
                        :motif_pairs
                    );
                    """,
                    {
                        "hostname": HOSTNAME,
                        "gitsha": gitsha,
                        "version": version,
                        "dataset": dataset,
                        "threads": threads,
                        "repetitions": repetitions,
                        "delta": delta,
                        "seed": seed,
                        "window": window,
                        "motifs":motifs,
                        "time_s": elapsed,
                        "log": log,
                        "motif_pairs": motif_pairs
                    }
                )


def run_scamp():
    install_scamp()
    db = get_db()
    datasets = get_datasets()
    threads = NUM_CPUS
    for dataset, window in datasets:
        execid = db.execute("""
            SELECT rowid from scamp
            where hostname=:hostname
              and dataset=:dataset
              and threads=:threads
              and window=:window
            """,
            {
                "hostname": HOSTNAME,
                "dataset": dataset,
                "threads": threads,
                "window": window,
            }
        ).fetchone()
        if execid is not None:
            print("Experiment already executed (scamp id={})".format(execid[0]))
            continue

        print(f"running on {dataset} with w={window} and {threads} threads... ")
        start = time.time()
        sp.run([
            "./SCAMP", 
            "--window={}".format(str(window)), 
            "--input_a_file_name={}".format(dataset),
            "--num_cpu_workers={}".format(threads)
        ]).check_returncode()
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

        os.remove("mp_columns_out")
        os.remove("mp_columns_out_index")

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
    run_scamp()
    run_attimo()
    
