"""Validate and merge independently executed MTP campaign shards."""

from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path
from typing import Iterable

from .common import atomic_write_json, now_utc, read_json, repo_root
from .reporting import collect_run_records, rebuild_reports
from .sharding import SEED_DERIVATION_VERSION, verify_run_id_partition
from .state import TERMINAL_STATES, run_id as make_run_id


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")

    root = repo_root()
    input_root = Path(args.input_root)
    if not input_root.is_absolute():
        input_root = root / input_root
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir

    shard_campaigns = discover_shard_campaigns(input_root, args.num_shards)
    manifests = [manifest for _campaign_dir, manifest in shard_campaigns]
    compatibility = validate_compatible_shards(manifests, args.num_shards)
    expected_ids = expected_run_ids_from_campaign_manifest(manifests[0])
    partition = verify_run_id_partition((m["shard"]["run_ids"] for m in manifests), expected_ids)
    if not partition["passed"]:
        raise SystemExit(f"Shard planned-run partition failed validation: {partition}")

    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"Output directory is not empty; use --overwrite: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    copy_shard_runs(shard_campaigns, output_dir)
    record_validation = validate_merged_records(output_dir, expected_ids)
    merged_manifest = build_merged_manifest(
        output_dir=output_dir,
        manifests=manifests,
        shard_campaigns=shard_campaigns,
        compatibility=compatibility,
        partition=partition,
        record_validation=record_validation,
    )
    atomic_write_json(output_dir / "campaign_manifest.json", merged_manifest)
    atomic_write_json(output_dir / "benchmark_manifest.json", merged_manifest["benchmark_manifest"])
    atomic_write_json(output_dir / "solver_manifest.json", merged_manifest["solver_manifest"])

    summary = rebuild_reports(output_dir)
    print(f"Merged {args.num_shards} shards into {output_dir}")
    print(f"Global planned runs: {len(expected_ids)}")
    print(f"Recorded terminal runs: {record_validation['recorded_run_count']}")
    print(f"Report summary: {summary}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def discover_shard_campaigns(input_root: Path, num_shards: int) -> list[tuple[Path, dict]]:
    if not input_root.exists():
        raise SystemExit(f"Input root does not exist: {input_root}")
    paths = sorted({path.resolve() for path in input_root.glob("**/campaign_manifest.json")})
    by_index: dict[int, tuple[Path, dict]] = {}
    for manifest_path in paths:
        manifest = read_json(manifest_path, default={}) or {}
        shard = manifest.get("shard") or {}
        if not isinstance(shard, dict) or "shard_index" not in shard:
            continue
        if int(shard.get("num_shards", -1)) != num_shards:
            continue
        index = int(shard["shard_index"])
        if index in by_index:
            raise SystemExit(f"Multiple campaign manifests found for shard {index:03d}")
        by_index[index] = (manifest_path.parent, manifest)

    missing = [index for index in range(num_shards) if index not in by_index]
    if missing:
        raise SystemExit(f"Missing shard campaign(s): {missing[:10]}")
    return [by_index[index] for index in range(num_shards)]


def validate_compatible_shards(manifests: list[dict], expected_shards: int) -> dict:
    if len(manifests) != expected_shards:
        raise SystemExit(f"Expected {expected_shards} shards, discovered {len(manifests)}")
    first = manifests[0]
    checks = (
        ("benchmark_manifest", "benchmark manifest"),
        ("solver_manifest", "solver manifest"),
        ("root_seed", "root seed"),
        ("seed_derivation_version", "seed derivation version"),
        ("runs", "runs per solver-instance"),
        ("time_limit_seconds", "time limit"),
        ("timeout_grace_seconds", "timeout grace"),
        ("evaluator_version", "evaluator version"),
        ("experiment_schema_version", "experiment schema version"),
    )
    for manifest in manifests[1:]:
        for key, label in checks:
            if manifest.get(key) != first.get(key):
                raise SystemExit(f"Incompatible shard {label}")
        if manifest.get("seed_derivation_version") != SEED_DERIVATION_VERSION:
            raise SystemExit("Unsupported seed derivation version")

    global_counts = {int(m.get("global_planned_run_count", -1)) for m in manifests}
    shard_global_counts = {int((m.get("shard") or {}).get("global_run_count", -1)) for m in manifests}
    if len(global_counts) != 1 or len(shard_global_counts) != 1 or global_counts != shard_global_counts:
        raise SystemExit("Shard global run counts disagree")

    discovered_indices = sorted(int(m["shard"]["shard_index"]) for m in manifests)
    expected_indices = list(range(expected_shards))
    if discovered_indices != expected_indices:
        raise SystemExit("Shard indices are incomplete or duplicated")

    return {
        "expected_shards": expected_shards,
        "discovered_shards": len(manifests),
        "global_planned_runs": next(iter(global_counts)),
        "runs_per_pair": int(first["runs"]),
        "time_limit_seconds": float(first["time_limit_seconds"]),
        "objective_direction": sorted(
            {
                str(row.get("objective", "maximize"))
                for row in (first.get("solver_manifest", {}) or {}).get("solvers", [])
            }
        ),
    }


def expected_run_ids_from_campaign_manifest(manifest: dict) -> set[str]:
    runs = int(manifest["runs"])
    solvers = sorted(
        (manifest.get("solver_manifest", {}) or {}).get("solvers", []),
        key=lambda row: str(row["solver_id"]),
    )
    instances = sorted(
        (manifest.get("benchmark_manifest", {}) or {}).get("instances", []),
        key=lambda row: (str(row["family"]), str(row["instance_id"]), str(row["path"])),
    )
    return {
        make_run_id(str(solver["solver_id"]), str(instance["instance_id"]), execution_index)
        for solver in solvers
        for instance in instances
        for execution_index in range(1, runs + 1)
    }


def copy_shard_runs(shard_campaigns: Iterable[tuple[Path, dict]], output_dir: Path) -> None:
    for campaign_dir, _manifest in shard_campaigns:
        runs_root = campaign_dir / "runs"
        if not runs_root.exists():
            raise SystemExit(f"Shard has no runs directory: {campaign_dir}")
        for run_dir in sorted(runs_root.glob("*/*/exec_*")):
            rel = run_dir.relative_to(campaign_dir)
            target = output_dir / rel
            if target.exists():
                raise SystemExit(f"Duplicate run artifact path while merging: {rel}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(run_dir, target)


def validate_merged_records(output_dir: Path, expected_run_ids: set[str]) -> dict:
    records = collect_run_records(output_dir)
    run_ids = [str(record.get("run_id")) for record in records if record.get("run_id")]
    counts = Counter(run_ids)
    duplicates = sorted(run_id for run_id, count in counts.items() if count > 1)
    observed = set(run_ids)
    missing = sorted(expected_run_ids - observed)
    unexpected = sorted(observed - expected_run_ids)
    nonterminal = sorted(
        str(record.get("run_id"))
        for record in records
        if record.get("run_id") and record.get("status") not in TERMINAL_STATES
    )
    if duplicates:
        raise SystemExit(f"Duplicate executed run IDs across shards: {duplicates[:10]}")
    if missing:
        raise SystemExit(f"Missing executed run IDs after merge: {missing[:10]}")
    if unexpected:
        raise SystemExit(f"Unexpected executed run IDs after merge: {unexpected[:10]}")
    if nonterminal:
        raise SystemExit(f"Non-terminal run statuses cannot be merged as complete: {nonterminal[:10]}")
    return {
        "recorded_run_count": len(records),
        "unique_recorded_run_ids": len(observed),
        "duplicate_run_ids_across_shards": 0,
        "missing_run_ids": 0,
        "unexpected_run_ids": 0,
        "nonterminal_run_ids": 0,
        "passed": True,
    }


def build_merged_manifest(
    *,
    output_dir: Path,
    manifests: list[dict],
    shard_campaigns: list[tuple[Path, dict]],
    compatibility: dict,
    partition: dict,
    record_validation: dict,
) -> dict:
    first = manifests[0]
    merged = {
        "experiment_schema_version": first.get("experiment_schema_version"),
        "evaluator_version": first.get("evaluator_version"),
        "seed_derivation_version": first.get("seed_derivation_version"),
        "campaign_id": output_dir.name,
        "created_at": now_utc(),
        "merged_at": now_utc(),
        "merged_from_shards": [
            {
                "campaign_dir": str(campaign_dir),
                "campaign_id": manifest.get("campaign_id"),
                "shard_index": int(manifest["shard"]["shard_index"]),
                "shard_run_count": int(manifest["shard"]["shard_run_count"]),
            }
            for campaign_dir, manifest in shard_campaigns
        ],
        "config_hash": first.get("config_hash"),
        "config": first.get("config"),
        "root_seed": first.get("root_seed"),
        "runs": first.get("runs"),
        "time_limit_seconds": first.get("time_limit_seconds"),
        "timeout_grace_seconds": first.get("timeout_grace_seconds"),
        "threads": first.get("threads"),
        "global_planned_run_count": compatibility["global_planned_runs"],
        "planned_run_count": compatibility["global_planned_runs"],
        "instance_filter": first.get("instance_filter"),
        "benchmark_manifest_hash": first.get("benchmark_manifest_hash"),
        "solver_manifest_hash": first.get("solver_manifest_hash"),
        "benchmark_manifest": first.get("benchmark_manifest"),
        "solver_manifest": first.get("solver_manifest"),
        "solver_audit": first.get("solver_audit"),
        "merge_validation": {
            **compatibility,
            "unique_planned_run_ids": partition["union_run_count"],
            "duplicate_run_ids_across_shards": partition["duplicate_run_count"],
            "missing_planned_run_ids": partition["missing_run_count"],
            "unexpected_planned_run_ids": partition["unexpected_run_count"],
            "record_validation": record_validation,
            "passed": partition["passed"] and record_validation["passed"],
        },
    }
    return merged


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
