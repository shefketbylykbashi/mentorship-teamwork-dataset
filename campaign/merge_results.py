#!/usr/bin/env python3
"""Merge the per-server shard results of a campaign into one results.jsonl.

    python campaign/merge_results.py campaign_results/SERVER1 campaign_results/SERVER2 ... \
        --out campaign_results/merged

Steps
  1. Read results.jsonl of every shard directory (latest record per run key; an
     ok record wins). A run key recorded in two different shards is an error:
     the shards must be disjoint.
  2. Check coverage against the full campaign plan (same defaults as
     run_campaign.py). Missing runs stop the merge unless --allow-missing.
  3. Write <out>/results.jsonl (sorted by run key) and <out>/coverage.json.

Solution files are not copied: move every <shard>/solutions/ tree into
<out>/solutions/ (the shards are disjoint, so no file is overwritten).
Afterwards campaign/summarize.py and isa/build_metadata.py work on <out>.
Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_campaign as rc  # noqa: E402


def read_shard(d: Path) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    p = d / "results.jsonl"
    if not p.exists():
        raise SystemExit(f"ERROR: {p} not found")
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue  # torn last line after a hard kill
            prev = latest.get(r["key"])
            if prev is None or prev.get("status") != "ok":
                latest[r["key"]] = r
    return latest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shards", nargs="+", type=Path, help="result directories, one per server")
    ap.add_argument("--out", type=Path, default=rc.REPO / "campaign_results" / "merged")
    ap.add_argument("--seeds", default="1-10")
    ap.add_argument("--allow-missing", action="store_true", help="merge even if some planned runs are missing")
    ap.add_argument("--overwrite", action="store_true")
    cfg = ap.parse_args()

    merged: dict[str, dict] = {}
    origin: dict[str, str] = {}
    clashes = []
    for d in cfg.shards:
        for key, rec in read_shard(d).items():
            if key in origin:
                clashes.append((key, origin[key], str(d)))
                continue
            origin[key] = str(d)
            merged[key] = rec
        print(f"read {d}: total keys so far {len(merged)}")
    if clashes:
        for c in clashes[:10]:
            print("DUPLICATE run in two shards:", c, file=sys.stderr)
        raise SystemExit(f"ERROR: {len(clashes)} run keys appear in more than one shard")

    plan_cfg = argparse.Namespace(families=[], instances_per_family=0, solvers=list(rc.ALL_SOLVERS),
                                  seeds=rc.parse_seeds(cfg.seeds))
    plan_keys = {t["key"] for t in rc.build_plan(plan_cfg)}
    missing = sorted(plan_keys - merged.keys())
    extra = sorted(merged.keys() - plan_keys)
    coverage = {"planned_runs": len(plan_keys), "recorded_runs": len(merged), "missing_runs": len(missing),
                "unplanned_runs": len(extra),
                "status_counts": dict(Counter(r.get("status", "?") for r in merged.values())),
                "shards": [str(d) for d in cfg.shards], "missing_first_50": missing[:50]}
    print(json.dumps({k: v for k, v in coverage.items() if k != "missing_first_50"}, indent=1))
    if missing and not cfg.allow_missing:
        print("first missing:", missing[:10], file=sys.stderr)
        raise SystemExit(f"ERROR: {len(missing)} planned runs missing (a server not finished, or use --retry-failed "
                         "there). Use --allow-missing to merge anyway.")

    out = cfg.out.resolve()
    if (out / "results.jsonl").exists() and not cfg.overwrite:
        raise SystemExit(f"ERROR: {out} already has results; pass --overwrite to replace them")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.jsonl", "w", encoding="utf-8") as fh:
        for key in sorted(merged):
            fh.write(json.dumps(merged[key], sort_keys=True) + "\n")
    (out / "coverage.json").write_text(json.dumps(coverage, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out / 'results.jsonl'} ({len(merged)} runs) and coverage.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
