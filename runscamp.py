#!/usr/bin/env python3

import sqlite3
import time
import subprocess as sp

db = sqlite3.connect("attimo-results.db", isolation_level=None)

db.execute("""
CREATE TABLE IF NOT EXISTS scamp (
    dataset      TEXT,
    threads      INT,
    window       INT,
    time_s       REAL
);
""")

db.execute("""
CREATE TABLE IF NOT EXISTS attimo (
    dataset      TEXT,
    memory       TEXT,
    window       INT,
    time_s       REAL
);
""")

datasets = [
    "data/ECG-100000.csv",
    "data/ECG-200000.csv",
    "data/ECG-300000.csv",
    "data/ECG-400000.csv",
    "data/ECG-500000.csv",
    "data/ECG-600000.csv",
    "data/ECG-700000.csv",
]
window = 200

###############################################################################
#
# Run the SCAMP baseline
threads = 4
for dataset in datasets:
    print(f"running on {dataset} with w={window} and {threads} threads... ", end="")
    start = time.time()
    sp.run([
        "SCAMP", 
        "--window={}".format(str(window)), 
        "--input_a_file_name={}".format(dataset),
        "--num_cpu_workers={}".format(threads)
    ])
    end = time.time()
    elapsed = end - start
    print(f"{elapsed} seconds")
    db.execute("""
    INSERT INTO scamp VALUES (?,?,?,?);
    """, (dataset, threads, window, elapsed))

###############################################################################
#
# Run Attimo

for dataset in datasets:
    for memory in ["1Gb"]:
        print(f"running on {dataset} with w={window} and {memory} memory... ", end="")
        start = time.time()
        sp.run([
            "attimo", 
            "--window", str(window),
            "--memory", memory,
            dataset
        ])
        end = time.time()
        elapsed = end - start
        print(f"{elapsed} seconds")
        db.execute("""
        INSERT INTO attimo VALUES (?,?,?,?);
        """, (dataset, memory, window, elapsed))

