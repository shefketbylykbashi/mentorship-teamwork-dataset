#!/usr/bin/env python3
"""Merge the per-server shard results and build a simple ISA dataset.

    python campaign/build_isa_dataset.py campaign_results/PARTINA campaign_results/DASTILA \
        campaign_results/DIRISTA campaign_results/MANDETA campaign_results/LYDRA \
        campaign_results/MIRGETA --out campaign_results/merged

Steps
  1. Read results.jsonl of every shard directory. A run key recorded in two
     different shards is an error (the shards must be disjoint).
  2. Check coverage against the full campaign plan (same defaults as
     run_campaign.py). Missing runs stop the build unless --allow-missing.
  3. Write <out>/results.jsonl (merged, latest record per run key, an ok record
     wins) so campaign/summarize.py works on the merged directory, plus
     coverage.json.
  4. Write the ISA inputs into <out>/isa/:
       metadata.csv       Instances, Source, feature_* (--feature-set pool of
                          isa/prepare_isa.py, default 'extended' = 31 candidates,
                          minus constant / |Spearman| >= threshold columns),
                          algo_* = R_ia = U_ia / max_a U_ia
       metadata_raw.csv   same, algo_* = U_ia (absolute expected score)
       all_features.csv   every raw feature of the extractor
       solver_instance.csv  U_ia, ok/timeout/failed counts, best score
       excluded_instances.csv  instances with max_a U_ia <= 0 (R undefined)
       dataset_info.json  definitions, counts, input hashes

U_ia = (1 / planned runs) * sum of verified scores of the runs of solver a on
instance i; any non-ok run (timeout, crash, invalid, ...) contributes 0.
Planned runs are 10 for seeded and 1 for deterministic solvers. Features are
computed from the instance files only; solver results never influence them.
Stdlib only (plus the repository's own feature extractor).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import run_campaign as rc  # noqa: E402
from benchmark_generation.core.features import FEATURE_NAMES, extract_features  # noqa: E402
from benchmark_generation.core.parser import parse_file  # noqa: E402
from isa.prepare_isa import (  # noqa: E402
    DEFAULT_REDUNDANCY_THRESHOLD, EXCLUDED_FEATURES, FEATURE_POLICIES, NOT_IN_EXTENDED_POOL, select_features,
)


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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shards", nargs="+", type=Path, help="result directories, one per server")
    ap.add_argument("--out", type=Path, default=rc.REPO / "campaign_results" / "merged")
    ap.add_argument("--seeds", default="1-10")
    ap.add_argument("--allow-missing", action="store_true", help="build even if some planned runs are missing")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--feature-set", choices=sorted(FEATURE_POLICIES), default="extended",
                    help="candidate feature pool: 'core' = original 15-feature shortlist, "
                         "'extended' = core + 16 (default); see isa/prepare_isa.py")
    ap.add_argument("--redundancy-threshold", type=float, default=DEFAULT_REDUNDANCY_THRESHOLD,
                    help="drop a candidate whose |Spearman rho| with an earlier kept one reaches this "
                         "(features only, never solver results)")
    cfg = ap.parse_args()
    policy = FEATURE_POLICIES[cfg.feature_set]

    # ---------------------------------------------------------------- merge
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
    plan = rc.build_plan(plan_cfg)
    plan_keys = {t["key"] for t in plan}
    missing = sorted(plan_keys - merged.keys())
    extra = sorted(merged.keys() - plan_keys)
    statuses: dict[str, int] = defaultdict(int)
    for r in merged.values():
        statuses[r.get("status", "?")] += 1
    coverage = {"planned_runs": len(plan), "recorded_runs": len(merged), "missing_runs": len(missing),
                "unplanned_runs": len(extra), "status_counts": dict(statuses),
                "shards": [str(d) for d in cfg.shards], "missing_first_50": missing[:50]}
    print(json.dumps({k: v for k, v in coverage.items() if k != "missing_first_50"}, indent=1))
    if missing and not cfg.allow_missing:
        print("first missing:", missing[:10], file=sys.stderr)
        raise SystemExit(f"ERROR: {len(missing)} planned runs missing (a server not finished, or use --retry-failed "
                         "there). Use --allow-missing to build anyway (missing runs count as score 0).")

    out = cfg.out.resolve()
    isa_dir = out / "isa"
    if (out / "results.jsonl").exists() and not cfg.overwrite:
        raise SystemExit(f"ERROR: {out} already has results; pass --overwrite to replace the merged files")
    isa_dir.mkdir(parents=True, exist_ok=True)
    with open(out / "results.jsonl", "w", encoding="utf-8") as fh:
        for key in sorted(merged):
            fh.write(json.dumps(merged[key], sort_keys=True) + "\n")
    (out / "coverage.json").write_text(json.dumps(coverage, indent=1), encoding="utf-8")

    # ---------------------------------------------------------------- performance U_ia
    solvers = list(rc.ALL_SOLVERS)
    insts = {(t["family"], t["instance"]): t["path"] for t in plan}
    planned_n = {s: (len(plan_cfg.seeds) if s in rc.SEEDED else 1) for s in solvers}
    agg = defaultdict(lambda: {"sum": 0.0, "ok": 0, "timeout": 0, "failed": 0, "best": 0})
    for t in plan:
        a = agg[(t["family"], t["instance"], t["solver"])]
        r = merged.get(t["key"])
        if r is not None and r.get("status") == "ok":
            a["sum"] += r["score"]
            a["ok"] += 1
            a["best"] = max(a["best"], r["score"])
        elif r is not None and r.get("status") == "timeout":
            a["timeout"] += 1
        else:
            a["failed"] += 1  # crash/invalid/runner_error/validator_disagree, or missing

    si_rows = []
    U: dict[tuple, dict[str, float]] = {}
    for (fam, iid) in sorted(insts):
        U[(fam, iid)] = {}
        for s in solvers:
            a = agg[(fam, iid, s)]
            u = a["sum"] / planned_n[s]
            U[(fam, iid)][s] = u
            si_rows.append([f"{fam}/{iid}", fam, s, planned_n[s], a["ok"], a["timeout"], a["failed"],
                            round(u, 4), a["best"]])
    write_csv(isa_dir / "solver_instance.csv",
              ["instance", "family", "solver", "planned_runs", "ok_runs", "timeout_runs", "failed_runs",
               "expected_score_U", "best_score"], si_rows)

    # ---------------------------------------------------------------- features
    order = sorted(insts)
    feat_rows = []
    for n, (fam, iid) in enumerate(order, 1):
        feat_rows.append(extract_features(parse_file(insts[(fam, iid)])))
        if n % 50 == 0:
            print(f"features {n}/{len(insts)}")
    # Candidate pool -> drop constant / near-duplicate columns (instance features only).
    selected, decisions, transformed = select_features(feat_rows, cfg.redundancy_threshold, policy)
    for d in decisions:
        if d["status"] != "selected":
            print(f"feature {d['column']}: {d['status']} {d.get('with_column', '')} {d.get('spearman', '')}")
    print(f"feature set '{cfg.feature_set}': {len(selected)} of {len(policy)} candidates kept")

    feat_cols = [f"feature_{name}" for name in selected]
    algo_cols = [f"algo_{s}" for s in solvers]
    rel_rows, raw_rows, all_rows, excluded = [], [], [], []
    for n, (fam, iid) in enumerate(order):
        feats = feat_rows[n]
        name = f"{fam}/{iid}"
        all_rows.append([name, fam] + [feats.get(f) for f in FEATURE_NAMES])
        fvals = [transformed[col][n] for col in selected]
        u = U[(fam, iid)]
        best = max(u.values())
        raw_rows.append([name, fam] + fvals + [round(u[s], 4) for s in solvers])
        if best <= 0:
            excluded.append([name, fam, "max expected score is 0 for every solver (relative performance undefined)"])
        else:
            rel_rows.append([name, fam] + fvals + [round(u[s] / best, 6) for s in solvers])
    write_csv(isa_dir / "metadata.csv", ["Instances", "Source"] + feat_cols + algo_cols, rel_rows)
    write_csv(isa_dir / "metadata_raw.csv", ["Instances", "Source"] + feat_cols + algo_cols, raw_rows)
    write_csv(isa_dir / "all_features.csv", ["instance", "family"] + list(FEATURE_NAMES), all_rows)
    write_csv(isa_dir / "excluded_instances.csv", ["instance", "family", "reason"], excluded)

    info = {
        "performance_definition": "algo_* in metadata.csv = U_ia / max_a U_ia; U_ia = sum of ok scores / planned runs "
                                  "(non-ok and missing runs count 0); metadata_raw.csv holds U_ia",
        "planned_runs_per_solver": planned_n,
        "seeded_solvers": list(rc.SEEDED), "deterministic_solvers": list(rc.DETERMINISTIC),
        "feature_set": cfg.feature_set,
        "redundancy_threshold": cfg.redundancy_threshold,
        "features": [{"column": f"feature_{suf}", "raw": raw, "transform": kind, "group": grp}
                     for raw, suf, kind, grp in policy if suf in selected],
        "feature_selection_decisions": decisions,
        "features_not_in_pool": {**EXCLUDED_FEATURES,
                                 **(NOT_IN_EXTENDED_POOL if cfg.feature_set == "extended" else {})},
        "feature_note": "computed and selected from instance files only; solver outcomes never used. "
                        "InstanceSpace SIFTED makes the final selection from these columns.",
        "instances_total": len(insts), "instances_in_metadata": len(rel_rows), "instances_excluded": len(excluded),
        "coverage": {k: v for k, v in coverage.items() if k != "missing_first_50"},
        "results_jsonl_sha256": sha256(out / "results.jsonl"),
        "metadata_sha256": sha256(isa_dir / "metadata.csv"),
    }
    (isa_dir / "dataset_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(f"wrote {isa_dir / 'metadata.csv'} ({len(rel_rows)} instances, {len(excluded)} excluded) "
          f"and metadata_raw.csv, all_features.csv, solver_instance.csv, dataset_info.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
