#!/usr/bin/env python

import subprocess
import json
import sys
import sqlite3
import datetime

bench_date = datetime.datetime.now().isoformat()

git_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True).stdout.decode('ascii').strip()
assert len(git_sha) > 0
git_commitdate = subprocess.run(["git", "log", "-1", "--format=%ci", "HEAD"], capture_output=True).stdout.decode('ascii').strip()
assert len(git_commitdate) > 0
git_branch = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True).stdout.decode('ascii').strip() 
assert len(git_branch) > 0
git_diff = subprocess.run(["git", "diff"], capture_output=True).stdout.decode('utf-8')
git_msg = subprocess.run(["git", "log", "-1", "--pretty=%B"], capture_output=True).stdout.decode("utf-8").strip()

res = subprocess.run(
    ["cargo", "criterion", "--message-format=json"],
    stdout=subprocess.PIPE
)

if res.returncode != 0:
    print(res.stderr)
    sys.exit(res.returncode)

with sqlite3.connect("bench.sqlite") as db:

    db.execute("""
    CREATE TABLE IF NOT EXISTS bench_results (
        date    TIMESTAMP,
        git_sha   TEXT,
        git_commit_date TIMESTAMP,
        git_branch  TEXT,
        git_diff    TEXT,
        git_msg    TEXT,
        stats    TEXT
    );
    """)

    for line in res.stdout.decode('ascii').splitlines():
        obj = json.loads(line)
        if obj['reason'] == 'benchmark-complete':
            db.execute("""
                INSERT INTO bench_results (
                    date, git_sha, git_commit_date, git_branch, git_diff, git_msg, stats
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?
                )
            """, (bench_date, git_sha, git_commitdate, git_branch, git_diff, git_msg, json.dumps(obj)))

