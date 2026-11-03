#!/usr/bin/env python3
"""Summarize a campaign: python campaign/summarize.py [results_dir]

Writes into the results dir:
  runs.csv                  one row per run (latest record per run key)
  solver_instance.csv       per solver x instance: runs, ok, best/mean/sd score, mean/max wall
  status_counts.csv         per solver x status
and prints totals (including total recorded wall-clock time). Stdlib only.
"""

from __future__ import annotations

import csv
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUN_COLS = ["solver", "family", "instance", "seed", "status", "score", "strict_score", "valid", "strict_valid",
            "projects", "exit_code", "wall_s", "cpu_s", "max_rss_kib", "validate_ms", "host",
            "started_at", "finished_at", "solution_sha256", "solution_file"]


def load(out: Path) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    with open(out / "results.jsonl", encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            prev = latest.get(r["key"])
            if prev is None or prev.get("status") != "ok":
                latest[r["key"]] = r
    return latest


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "campaign_results" / "main"
    runs = load(out)
    rows = sorted(runs.values(), key=lambda r: (r["solver"], r["family"], r["instance"], r["seed"]))
    with open(out / "runs.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, RUN_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["solver"], r["family"], r["instance"])].append(r)
    with open(out / "solver_instance.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["solver", "family", "instance", "runs", "ok", "best_score", "mean_score", "sd_score",
                    "mean_wall_s", "max_wall_s"])
        for (s, f, i), rs in sorted(groups.items()):
            sc = [r["score"] for r in rs if r["status"] == "ok"]
            wall = [r["wall_s"] for r in rs if r.get("wall_s") is not None]
            w.writerow([s, f, i, len(rs), len(sc), max(sc) if sc else "", round(st.mean(sc), 3) if sc else "",
                        round(st.stdev(sc), 3) if len(sc) > 1 else "", round(st.mean(wall), 3) if wall else "",
                        max(wall) if wall else ""])

    counts = Counter((r["solver"], r["status"]) for r in rows)
    with open(out / "status_counts.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["solver", "status", "runs"])
        for (s, k), n in sorted(counts.items()):
            w.writerow([s, k, n])

    total_wall = sum(r.get("wall_s") or 0 for r in rows)
    print(f"runs recorded: {len(rows)}")
    print("status:", dict(Counter(r["status"] for r in rows)))
    print(f"total solver wall-clock: {total_wall / 3600:.2f} h (sum over runs)")
    print(f"wrote {out / 'runs.csv'}, solver_instance.csv, status_counts.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
