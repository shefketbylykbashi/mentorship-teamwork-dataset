#!/usr/bin/env python3
"""Build the MATILDA metadata file: instance features plus solver performance.

    python3 isa/build_metadata.py [--features FILE] [--results FILE] [--out FILE]

Reads the features written by isa/extract_features.py and the per-run records of
the solver campaign (results/results.jsonl), and writes isa/metadata.csv in the
layout MATILDA expects:

    Instances,Source,feature_<name>,...,algo_<solver>,...

Performance is the relative expected score R_ia = U_ia / max_a U_ia, where U_ia
is the sum of the validated scores of solver a on instance i divided by its
number of planned runs (10 seeds for the randomised solvers, 1 run for the
deterministic ones). A run that is not "ok" contributes zero. Only the Python
standard library is used.
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

SEEDED = ("original_b", "original_d", "evolved_b", "evolved_d", "evolved_e")
DETERMINISTIC = ("original_c", "original_e", "original_f", "evolved_c", "evolved_f")
SOLVERS = sorted(SEEDED + DETERMINISTIC)
SEEDS = 10


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--features", default=os.path.join(here, "features.csv"))
    parser.add_argument("--results", default=os.path.join(repo, "results", "results.jsonl"))
    parser.add_argument("--out", default=os.path.join(here, "metadata.csv"))
    args = parser.parse_args()

    planned = {s: (SEEDS if s in SEEDED else 1) for s in SOLVERS}
    total = defaultdict(float)
    runs = defaultdict(int)
    with open(args.results, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            key = (f"{r['family']}/{r['instance']}", r["solver"])
            runs[key] += 1
            if r["status"] == "ok":
                total[key] += r["score"]

    with open(args.features, newline="") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        rows = list(reader)

    out_rows = []
    for row in rows:
        inst = row["Instances"]
        for s in SOLVERS:
            if runs[(inst, s)] != planned[s]:
                sys.exit(f"ERROR: {inst}: {runs[(inst, s)]} runs of {s}, expected {planned[s]}")
        u = {s: total[(inst, s)] / planned[s] for s in SOLVERS}
        best = max(u.values())
        if best <= 0:
            sys.exit(f"ERROR: {inst}: every solver has expected score 0, relative performance undefined")
        out_rows.append([row[c] for c in columns] + [round(u[s] / best, 6) for s in SOLVERS])

    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(columns + [f"algo_{s}" for s in SOLVERS])
        writer.writerows(out_rows)
    print(f"wrote {len(out_rows)} instances, {len(columns) - 2} features and {len(SOLVERS)} solvers to {args.out}")


if __name__ == "__main__":
    main()
