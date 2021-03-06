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
import psutil

HOSTNAME = socket.gethostname()
NUM_CPUS = multiprocessing.cpu_count()

SCAMP_EXE = './SCAMP-' + HOSTNAME
LL_EXE = './LL-' + HOSTNAME

def install_scamp():
    try:
        if sp.run([SCAMP_EXE, "--version"]).returncode == 0:
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
            ("cp /tmp/SCAMP/build/SCAMP {}".format(SCAMP_EXE)       , "."),
            ("rm -rf /tmp/SCAMP"                                    , ".")
        ]
        for cmd, d in commands:
            sp.run(shlex.split(cmd), cwd=d).check_returncode()

def install_prescrimp():
    try:
        if os.path.isfile("./prescrimp"):
            return
    except:
        print("Installing PreSCRIMP....")
        commands = [
            ("wget https://sites.google.com/site/scrimpplusplus/home/prescrimp.cpp",
            "."),
            ("g++ -O3 -march=native -std=c++11 -o prescrimp prescrimp.cpp -lm -lfftw3", ".")
        ]
        for cmd, d in commands:
            sp.run(shlex.split(cmd), cwd=d).check_returncode()

def install_ll():
    if os.path.isfile(LL_EXE):
        return
    else:
        print("Installing LL....")
        commands = [
            ("git clone https://github.com/xiaoshengli/LL.git /tmp/LL"  , "/tmp"),
            ("git checkout 6a20ec1"                                     , "/tmp/LL"),
            ("g++ -O3 -o LL LL.cpp -std=c++11"                          , "/tmp/LL"),
            ("cp /tmp/LL/LL {}".format(LL_EXE)                          , "."),
            ("rm -rf /tmp/LL"                                           , ".")
        ]
        for cmd, d in commands:
            sp.run(shlex.split(cmd), cwd=d).check_returncode()

def install_mk():
    if os.path.isfile("./mk_l"):
        return
    else:
        print("Installing mk_l ....")
        commands = [
            ("wget http://www.cs.ucr.edu/%7Emueen/zip/MK_code.zip"   , "/tmp"),
            ("unzip MK_code.zip"                                        , "/tmp"),
            ("g++ -O3 -o mk_l mk_l.cpp -std=c++11"                      , os.path.join("/tmp", "MK", "Subsequence version")),
            ("cp /tmp/MK/Subsequence version/mk_l ."                    , "."),
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
    if dbver < 2:
        db.execute("""
        CREATE TABLE IF NOT EXISTS ll (
            hostname     TEXT,
            dataset      TEXT,
            threads      INT,
            window       INT,
            grids        INT,
            time_s       REAL
        );
        """)
        db.execute("PRAGMA user_version = 2;")
        print("  Bump version to 2")
    if dbver < 3:
        db.execute("""
        CREATE TABLE IF NOT EXISTS mk (
            hostname     TEXT,
            dataset      TEXT,
            threads      INT,
            window       INT,
            reference_points  INT,
            time_s       REAL
        );
        """)
        db.execute("PRAGMA user_version = 3;")
        print("  Bump version to 3")
    if dbver < 4:
        db.execute("""
        CREATE TABLE IF NOT EXISTS prescrimp (
            hostname     TEXT,
            dataset      TEXT,
            window       INT,
            motifs       INT,
            stepsize     REAL,   -- As a fraction of window
            time_s       REAL,
            motif_pairs  TEXT
        )
        """)
        db.execute("PRAGMA user_version = 4;")
        print("  Bump version to 4")
    if dbver < 5:
        db.execute("""
        CREATE TABLE IF NOT EXISTS projection (
            hostname     TEXT,
            dataset      TEXT,
            window       INT,
            motifs       INT,
            paa          INT,
            alphabet     INT,
            repetitions  INT,
            k            INT,
            seed         INT,
            time_s       REAL,
            outcome      TEXT,  -- one of 'ok', 'timeout', 'fail'
            motif_pairs  TEXT
        )
        """)
        db.execute("PRAGMA user_version = 5;")
        print("  Bump version to 5")
    if dbver < 6:
        db.executescript("""
        ALTER TABLE scamp ADD COLUMN max_mem_bytes;
        ALTER TABLE prescrimp ADD COLUMN max_mem_bytes;
        ALTER TABLE ll ADD COLUMN max_mem_bytes;
        ALTER TABLE projection ADD COLUMN max_mem_bytes;
        PRAGMA user_version = 6;
        """)
        print("  Bump version to 6")
    if dbver < 7:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS scamp_gpu (
            hostname     TEXT,
            dataset      TEXT,
            window       INT,
            time_s       REAL,
            motif_pairs  TEXT,
            max_mem_bytes  INT
        );
        PRAGMA user_version = 7;
        """)
    
    print("Database initialized")

    return db


def _get_gpu_memory_bytes():
    suffix = {
        'KiB': 1000,
        'MiB': 1000*1000,
        'GiB': 1000*1000*1000
    }
    output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    ACCEPTABLE_AVAILABLE_MEMORY = 1024
    COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
    try:
        memory_use_info = output_to_list(sp.check_output(COMMAND.split(),stderr=sp.STDOUT))[1:]
    except sp.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    tot = 0
    for mem in memory_use_info:
        m, unit = mem.split()
        tot += int(m) * suffix[unit]
    # print("memory of GPU:", tot)
    return tot


def run(cmdline, stdout=None, timeout=None, measure_mem_gpu=False):
    """Run the given command, and return the maximum memory in bytes used by the child process, along with the outcome (one of "ok", "timeout", or "crash")"""
    mem = 0
    mem_gpu = 0
    start = time.time()
    try:
        child = sp.Popen(cmdline, stdout=stdout)
        p = psutil.Process(child.pid)
        while child.poll() is None:
            time.sleep(0.5)
            m = p.memory_info().vms
            mem = max(mem, m)
            if measure_mem_gpu:
                mem_gpu = max(mem_gpu, _get_gpu_memory_bytes())
            if timeout is not None and time.time() > start + timeout:
                child.kill()
                return mem, "timeout"
            retcode = child.poll()
        if child.returncode == 0:
            outcome = "ok"
        else:
            outcome = "crash"
    except sp.CalledProcessError:
        outcome = "crash"
    return mem + mem_gpu, outcome


def prefix(path, n):
    fname, ext = os.path.splitext(path)
    outpath = fname + "-{}{}".format(n, ext)
    if not os.path.isfile(outpath):
        with open(outpath, "w") as fp:
            sp.run(["head", "-n{}".format(n), path], stdout=fp)
    return outpath


def get_datasets():
    synth = [
        ("data/synth-w100-n{}-rc{}.txt.gz".format(n, rc), 100)
        for n in 2 ** np.arange(19, 25, step=2)
        # for n in 2 ** np.arange(19, 28, step=2)
        for rc in [
            '100-easy',
            '50-middle',
            '20-difficult'
        ]
    ]
    return [
        ("data/ASTRO.csv.gz", 100),
        ("data/GAP.csv.gz", 600),
        ("data/freezer.txt.gz", 5000),
        # ("data/ECG.csv.gz", 1000),
        # ("data/HumanY.txt.gz", 18000),
        # ("data/Whales-amplitude-noised.txt.gz", 140)
        # ("data/VCAB_noised.txt.gz", 100)
        #### Prefix datasets for runtime estimation
        # ("data/Whales-amplitude-noised-1000000.txt.gz", 140),
        # ("data/Whales-amplitude-noised-2000000.txt.gz", 140),
        # ("data/Whales-amplitude-noised-4000000.txt.gz", 140),
        # ("data/VCAB_noised-1000000.txt.gz", 100),
        # ("data/VCAB_noised-2000000.txt.gz", 100),
        # ("data/VCAB_noised-4000000.txt.gz", 100),
        # ("data/HumanY-1000000.txt.gz", 18000),
        # ("data/HumanY-2000000.txt.gz", 18000),
        # ("data/HumanY-4000000.txt.gz", 18000),
    ] #+ synth

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


def run_attimo_recall():
    gitsha = sp.check_output(shlex.split("git rev-parse HEAD"))
    sp.check_call(shlex.split("cargo install --force --locked --path ."))
    version = int(sp.check_output(["attimo", "--version"]))
    db = get_db()
    datasets = get_datasets()
    threads = NUM_CPUS
    repetitions = 400
    motifs = 10
    for seed in range(1, 30):
        for delta in [0.1, 0.2, 0.3, 0.4, 0.5]:
            for dataset, window in datasets:
                print("==== Looking for", motifs, "in", dataset,
                      "window",window)
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
                    "--min-correlation", "0.9",
                    "--log-path", "/tmp/attimo.json",
                    "--output", "/tmp/motifs.csv",
                    dataset
                ]).check_returncode()
                end = time.time()
                elapsed = end - start
                motif_pairs = pd.read_csv('/tmp/motifs.csv', names=['a', 'b','dist', 'confirmation_time']).to_json(orient='records')
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


def run_attimo():
    gitsha = sp.check_output(shlex.split("git rev-parse HEAD"))
    sp.check_call(shlex.split("cargo install --force --locked --path ."))
    version = int(sp.check_output(["attimo", "--version"]))
    db = get_db()
    datasets = get_datasets()
    threads = NUM_CPUS
    delta = 0.01
    for seed in [14514]:#, 1346, 2524]:
        # for repetitions in [r*r for r in [9, 8]]:#, 200, 400, 800, 1600]:
        for repetitions in [9]:
            for motifs in [1]:
                for dataset, window in datasets:
                    print("==== Looking for", motifs, "in", dataset,
                          "window",window, "with repetitions", repetitions)
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
                    motif_pairs = pd.read_csv('/tmp/motifs.csv', names=['a', 'b','dist', 'confirmation_time']).to_json(orient='records')
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

def wc(path):
    with open(path) as fp:
        return len(fp.readlines())


def head(n, input, output):
    with open(input, "r") as ifp:
        with open(output, "w") as ofp:
            for line in ifp.readlines():
                if n == 0:
                    return
                ofp.write(line)
                n -= 1


def run_mk():
    install_mk()
    db = get_db()
    datasets = get_datasets()
    threads = NUM_CPUS
    reference_points = 2
    for dataset, window in datasets:
        execid = db.execute("""
            SELECT rowid from mk
            where hostname=:hostname
              and dataset=:dataset
              and threads=:threads
              and window=:window
              and reference_points=:reference_points
            """,
            {
                "hostname": HOSTNAME,
                "dataset": dataset,
                "threads": threads,
                "window": window,
                "reference_points": reference_points
            }
        ).fetchone()
        if execid is not None:
            print("Experiment already executed (mk id={})".format(execid[0]))
            continue

        print(f"running on {dataset} with w={window} and {threads} threads... ")
        start = time.time()
        sp.run([
            "./mk_l", 
            dataset,
            str(wc(dataset)),
            str(window),
            str(window),
            str(reference_points),
            "1"
        ]).check_returncode()
        end = time.time()
        elapsed = end - start
        print(f"{elapsed} seconds")

        db.execute("""
            INSERT INTO mk VALUES (:hostname,:dataset,:threads,:window,:reference_points,:elapsed);
            """,
            {
                "hostname": HOSTNAME,
                "dataset": dataset,
                "threads": threads,
                "window": window,
                "reference_points": reference_points,
                "elapsed": elapsed
            }
        )


def run_ll():
    install_ll()
    db = get_db()
    datasets = get_datasets()
    threads = NUM_CPUS
    grids = 4
    for dataset, window in datasets:
        if dataset.endswith('.gz'):
            if not os.path.isfile(dataset.replace('.gz', '')):
                sp.run(['gunzip', '--keep', dataset])
            dataset = dataset.replace('.gz','')

        execid = db.execute("""
            SELECT rowid from ll
            where hostname=:hostname
              and dataset=:dataset
              and threads=:threads
              and window=:window
              and grids=:grids
            """,
            {
                "hostname": HOSTNAME,
                "dataset": dataset,
                "threads": threads,
                "window": window,
                "grids": grids
            }
        ).fetchone()
        execid = None
        if execid is not None:
            print("Experiment already executed (ll id={})".format(execid[0]))
            continue


        print(f"running on {dataset} with w={window} and {threads} threads... ")
        start = time.time()
        mem, outcome = run([
            LL_EXE,
            dataset,
            str(wc(dataset)),
            str(window),
            str(grids)
        ], stdout=sp.DEVNULL)
        end = time.time()
        elapsed = end - start
        print(f"{elapsed} seconds")

        db.execute("""
        INSERT INTO ll VALUES (:hostname,:dataset,:threads,:window,:grids,:elapsed,:max_mem_bytes);
            """,
            {
                "hostname": HOSTNAME,
                "dataset": dataset,
                "threads": threads,
                "window": window,
                "grids": grids,
                "elapsed": elapsed,
                "max_mem_bytes": mem
            }
        )


def run_prescrimp():
    db = get_db()
    datasets = get_datasets()
    stepsize = 0.25
    motifs = 10
    for dataset, window in datasets:
        execid = db.execute("""
            SELECT rowid from prescrimp
            where hostname=:hostname
              and dataset=:dataset
              and stepsize=:stepsize
              and window=:window
              and motifs=:motifs
            """,
            {
                "hostname": HOSTNAME,
                "dataset": dataset,
                "stepsize": stepsize,
                "window": window,
                "motifs": motifs
            }
        ).fetchone()
        if execid is not None:
            print("Experiment already executed (scrimp id={})".format(execid[0]))
            continue

        print(f"running PreSCRIMP on {dataset} with w={window} and stepsize={stepsize}... ")
        start = time.time()
        mem_bytes, outcome = run([
            "cargo", 
            "run",
            "--release",
            "--example",
            "prescrimp",
            "--",
            dataset,
            "--window", str(window),
            "--skip", str(stepsize),
            "--motifs", str(motifs),
            "--output", "/tmp/prescrimp.csv"
        ])
        end = time.time()
        elapsed = end - start
        print(f"{elapsed} seconds")
        motif_pairs = pd.read_csv('/tmp/prescrimp.csv', names=['a', 'b','dist', 'confirmation_time']).to_json(orient='records')
        os.remove("/tmp/prescrimp.csv")

        db.execute("""
        INSERT INTO prescrimp VALUES (:hostname,:dataset,:window,:motifs,:stepsize,:elapsed,:motif_pairs,:max_mem_bytes);
            """,
            {
                "hostname": HOSTNAME,
                "dataset": dataset,
                "window": window,
                "motifs": motifs,
                "stepsize": stepsize,
                "elapsed": elapsed,
                "motif_pairs": motif_pairs,
                "max_mem_bytes": mem_bytes
            }
        )


def run_projection():
    sp.run(["cargo", "build", "--release", "--example", "ChiuKL"])
    db = get_db()
    datasets = get_datasets()
    # stepsize = 0.25
    best_params = {
        'data/ASTRO.csv.gz': {'alphabet': 3, 'k': 6},
        'data/GAP.csv.gz': {'alphabet': 4, 'k': 4},
        'data/freezer.txt.gz': {'alphabet': 4, 'k': 3}
    }
    motifs = 1
    alphabet = 3 # 6
    repetitions = 10
    seed = 1234
    for k in [6]:
        for dataset, window in datasets:
            bestconf = best_params[dataset]
            k = bestconf['k']
            alphabet = bestconf['alphabet']
            paa = window // 10
            execid = db.execute("""
                SELECT rowid from projection
                where hostname=:hostname
                  and dataset=:dataset
                  and window=:window
                  and paa=:paa
                  and repetitions=:repetitions
                  and alphabet=:alphabet
                  and seed=:seed
                  and k=:k
                  and motifs=:motifs
                """,
                {
                    "hostname": HOSTNAME,
                    "dataset": dataset,
                    "window": window,
                    "motifs": motifs,
                    "repetitions": repetitions,
                    "paa": paa,
                    "alphabet": alphabet,
                    "k": k,
                    "seed": seed
                }
            ).fetchone()
            if execid is not None:
                print("Experiment already executed (projection id={})".format(execid[0]))
                continue

            print(f"running Projection on {dataset} with w={window}... ")
            start = time.time()
            mem_bytes, outcome = run([
                "cargo", 
                "run",
                "--release",
                "--example",
                "ChiuKL",
                "--",
                dataset,
                "--window", str(window),
                "--paa", str(paa),
                "--k", str(k),
                "--seed", str(seed),
                "--repetitions", str(repetitions),
                "--alphabet", str(alphabet),
                "--motifs", str(motifs),
                "--output", "/tmp/projection.csv"
            ], timeout=3600)
            end = time.time()
            elapsed = end - start
            print(f"{elapsed} seconds, outcome {outcome}")
            if outcome == "ok":
                motif_pairs = pd.read_csv('/tmp/projection.csv', names=['a', 'b','dist', 'confirmation_time']).to_json(orient='records')
            else:
                motif_pairs = None
            if os.path.isfile("/tmp/projection.csv"):
                os.remove("/tmp/projection.csv")


            db.execute("""
            INSERT INTO projection VALUES (:hostname,:dataset,:window,:motifs,:paa,:alphabet,:repetitions,:k,:seed,:elapsed,:outcome,:motif_pairs,:max_mem_bytes);
                """,
                {
                    "hostname": HOSTNAME,
                    "dataset": dataset,
                    "window": window,
                    "motifs": motifs,
                    "repetitions": repetitions,
                    "paa": paa,
                    "alphabet": alphabet,
                    "k": k,
                    "seed": seed,
                    "elapsed": elapsed,
                    "outcome": outcome,
                    "motif_pairs": motif_pairs,
                    "max_mem_bytes": mem_bytes
                }
            )



def run_scamp(gpu=False):
    install_scamp()
    db = get_db()
    datasets = get_datasets()
    threads = NUM_CPUS
    for dataset, window in datasets:
        if dataset.endswith('.gz'):
            if not os.path.isfile(dataset.replace('.gz', '')):
                sp.run(['gunzip', '--keep', dataset])
            dataset = dataset.replace('.gz','')

        if gpu:
            execid = db.execute("""
                SELECT rowid from scamp_gpu
                where hostname=:hostname
                  and dataset=:dataset
                  and window=:window
                """,
                {
                    "hostname": HOSTNAME,
                    "dataset": dataset,
                    "window": window,
                }
            ).fetchone()
        else:
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

        print(f"running on {dataset} with w={window} and {threads} threads, GPU={gpu}... ")
        start = time.time()
        if gpu:
            mem_bytes, outcome = run([
                SCAMP_EXE,
                "--window={}".format(str(window)), 
                "--input_a_file_name={}".format(dataset)
            ], measure_mem_gpu=True)
        else:
            mem_bytes, outcome = run([
                SCAMP_EXE,
                "--window={}".format(str(window)), 
                "--input_a_file_name={}".format(dataset),
                "--no_gpu", # we want to measure the time without the GPU, only using the CPU
                "--num_cpu_workers={}".format(threads)
            ])
        print('memory is', mem_bytes)
        assert outcome == 'ok'
        end = time.time()
        elapsed = end - start
        print(f"{elapsed} seconds")
        # dists = np.loadtxt('mp_columns_out')
        # idxs  = np.loadtxt('mp_columns_out_index')
        # df = pd.DataFrame({
        #     "a": np.arange(len(idxs)),
        #     "b": idxs.astype('int'),
        #     "dist": dists
        # }).sort_values('dist')
        # df = df[df['a'] < df['b']]
        # df = remove_trivial(df, window)
        # motifs = df.head(100)[['a', 'b', 'dist']].to_json(orient='records')
        motifs = '{}' # don't collect the motifs, we don't use this information

        # os.remove("mp_columns_out")
        # os.remove("mp_columns_out_index")

        if gpu:
            db.execute("""
            INSERT INTO scamp_gpu VALUES (:hostname,:dataset,:window,:elapsed,:motifs,:max_mem_bytes);
                """,
                {
                    "hostname": HOSTNAME,
                    "dataset": dataset,
                    "window": window,
                    "elapsed": elapsed,
                    "motifs": motifs,
                    "max_mem_bytes": mem_bytes
                }
            )
        else:
            db.execute("""
            INSERT INTO scamp VALUES (:hostname,:dataset,:threads,:window,:elapsed,:motifs,:max_mem_bytes);
                """,
                {
                    "hostname": HOSTNAME,
                    "dataset": dataset,
                    "threads": threads,
                    "window": window,
                    "elapsed": elapsed,
                    "motifs": motifs,
                    "max_mem_bytes": mem_bytes
                }
            )


def scalability_attimo():
    gitsha = sp.check_output(shlex.split("git rev-parse HEAD"))
    sp.check_call(shlex.split("cargo install --force --locked --path ."))
    version = int(sp.check_output(["attimo", "--version"]))
    db = get_db()
    datasets = get_datasets()
    threads = NUM_CPUS
    repetitions = 100
    delta = 0.001
    for seed in [14514]: #, 1346, 2524]:
        for motifs in [1]:
            for dataset, window in datasets:
                for percent in [20, 40, 60, 80, 100]:
                    lines = int(wc(dataset) * (percent / 100))
                    pre, ext = os.path.splitext(dataset)
                    fname = "".join([pre, "-perc" + str(percent), ext])
                    if not os.path.isfile(fname):
                        head(lines, dataset, fname)
                    print("==== Looking for", motifs, "in", fname,
                          "window",window)
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
                            "dataset": fname,
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
                        "--min-correlation", "0.9",
                        "--log-path", "/tmp/attimo.json",
                        "--output", "/tmp/motifs.csv",
                        fname
                    ]).check_returncode()
                    end = time.time()
                    elapsed = end - start
                    motif_pairs = pd.read_csv('/tmp/motifs.csv', names=['a', 'b','dist', 'confirmation_time']).to_json(orient='records')
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
                            "dataset": fname,
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



if __name__ == "__main__":
    # scalability_attimo()
    # run_attimo()
    # run_attimo_recall()
    # run_scamp()
    # run_scamp(gpu=True)
    run_projection()
    # run_prescrimp()
    # run_ll()
    # run_mk()
    
