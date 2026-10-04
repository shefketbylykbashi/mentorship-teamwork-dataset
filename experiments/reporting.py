"""Rebuild and update campaign reports from authoritative per-run JSON."""

from __future__ import annotations

import csv
import shutil
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .common import atomic_write_json, atomic_write_text, now_utc, read_json, safe_id
from .state import (
    ERROR,
    INTERRUPTED,
    INVALID_OUTPUT,
    INVALID_SOLUTION,
    SUCCESS,
    TIMEOUT,
    result_path,
)


def update_best_for_pair(campaign_dir: str | Path, solver_id: str, instance_id: str) -> dict:
    campaign_dir = Path(campaign_dir)
    records = [
        record
        for record in _pair_records(campaign_dir, solver_id, instance_id)
        if record.get("status") == SUCCESS and record.get("verified_score") is not None
    ]
    out_dir = campaign_dir / "best_results" / safe_id(solver_id) / safe_id(instance_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not records:
        best = {
            "solver_id": solver_id,
            "instance_id": instance_id,
            "status": "NO_VALID_RUN",
            "best_score": None,
            "updated_at": now_utc(),
        }
        atomic_write_json(out_dir / "best_result.json", best)
        return best

    best_record = sorted(
        records,
        key=lambda r: (
            -int(r["verified_score"]),
            float(r.get("wall_time_seconds") or 1e100),
            int(r.get("execution_index") or 10**9),
        ),
    )[0]
    best = {
        "solver_id": solver_id,
        "instance_id": instance_id,
        "status": SUCCESS,
        "best_execution_index": best_record.get("execution_index"),
        "best_seed": best_record.get("seed"),
        "best_score": best_record.get("verified_score"),
        "best_runtime": best_record.get("wall_time_seconds"),
        "best_solution_path": best_record.get("solution_path"),
        "source_run_id": best_record.get("run_id"),
        "updated_at": now_utc(),
    }
    atomic_write_json(out_dir / "best_result.json", best)
    solution = best_record.get("solution_path")
    if solution:
        src = campaign_dir / solution if not Path(solution).is_absolute() else Path(solution)
        if src.exists():
            shutil.copyfile(src, out_dir / "best_solution.txt")
    return best


def rebuild_reports(campaign_dir: str | Path) -> dict:
    campaign_dir = Path(campaign_dir)
    summaries_dir = campaign_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_json(campaign_dir / "campaign_manifest.json", default={}) or {}
    planned_runs = int(manifest.get("planned_run_count") or 0)

    records = collect_run_records(campaign_dir)
    _write_csv(summaries_dir / "per_run_results.csv", records)
    _write_csv(campaign_dir / "per_run_results.csv", records)

    pair_rows = solver_instance_summary(records, manifest)
    _write_csv(summaries_dir / "solver_instance_summary.csv", pair_rows)
    _write_csv(campaign_dir / "solver_instance_summary.csv", pair_rows)
    best_rows = _best_rows(pair_rows)
    _write_csv(summaries_dir / "best_result_per_solver_instance.csv", best_rows)
    _write_csv(campaign_dir / "best_result_per_solver_instance.csv", best_rows)

    solver_rows = solver_summary(records, manifest)
    _write_csv(summaries_dir / "solver_summary.csv", solver_rows)
    _write_csv(campaign_dir / "solver_summary.csv", solver_rows)

    instance_rows = instance_summary(records, manifest)
    _write_csv(summaries_dir / "instance_summary.csv", instance_rows)
    _write_csv(campaign_dir / "instance_summary.csv", instance_rows)

    _write_csv(summaries_dir / "failure_report.csv", [r for r in records if r.get("status") == ERROR])
    _write_csv(campaign_dir / "failure_report.csv", [r for r in records if r.get("status") == ERROR])
    _write_csv(summaries_dir / "timeout_report.csv", [r for r in records if r.get("status") == TIMEOUT])
    _write_csv(campaign_dir / "timeout_report.csv", [r for r in records if r.get("status") == TIMEOUT])
    _write_csv(
        summaries_dir / "invalid_output_report.csv",
        [r for r in records if r.get("status") in {INVALID_OUTPUT, INVALID_SOLUTION}],
    )
    _write_csv(
        campaign_dir / "invalid_output_report.csv",
        [r for r in records if r.get("status") in {INVALID_OUTPUT, INVALID_SOLUTION}],
    )

    counts = Counter(record.get("status", "UNKNOWN") for record in records)
    progress = {
        "planned_runs": planned_runs,
        "recorded_runs": len(records),
        "completed_runs": sum(counts.values()) - counts.get("RUNNING", 0),
        "pending_runs": max(0, planned_runs - len(records)),
        "status_counts": dict(sorted(counts.items())),
        "updated_at": now_utc(),
    }
    atomic_write_json(campaign_dir / "progress.json", progress)

    campaign_summary = {
        "planned_runs": planned_runs,
        "recorded_runs": len(records),
        "status_counts": dict(sorted(counts.items())),
        "successful_runs": counts.get(SUCCESS, 0),
        "error_runs": counts.get(ERROR, 0),
        "timeout_runs": counts.get(TIMEOUT, 0),
        "invalid_output_runs": counts.get(INVALID_OUTPUT, 0),
        "invalid_solution_runs": counts.get(INVALID_SOLUTION, 0),
        "interrupted_runs": counts.get(INTERRUPTED, 0),
        "updated_at": now_utc(),
    }
    atomic_write_json(summaries_dir / "campaign_summary.json", campaign_summary)
    atomic_write_json(campaign_dir / "campaign_summary.json", campaign_summary)
    _write_csv(summaries_dir / "campaign_summary.csv", [campaign_summary])
    _write_csv(campaign_dir / "campaign_summary.csv", [campaign_summary])
    return campaign_summary


def collect_run_records(campaign_dir: str | Path) -> List[dict]:
    campaign_dir = Path(campaign_dir)
    rows: List[dict] = []
    for run_dir in sorted(campaign_dir.glob("runs/*/*/exec_*")):
        result = read_json(result_path(run_dir), default=None)
        status = read_json(run_dir / "status.json", default={}) or {}
        row = dict(status)
        if isinstance(result, dict):
            row.update(result)
        row.setdefault("run_dir", str(run_dir.relative_to(campaign_dir)))
        rows.append(_flatten(row))
    return rows


def solver_instance_summary(records: List[dict], manifest: dict) -> List[dict]:
    groups: dict[tuple[str, str], List[dict]] = defaultdict(list)
    for record in records:
        groups[(record.get("solver_id"), record.get("instance_id"))].append(record)
    rows = []
    planned_runs = int(manifest.get("runs") or 0)
    for (solver_id, instance_id), recs in sorted(groups.items()):
        status_counts = Counter(r.get("status") for r in recs)
        successes = [r for r in recs if r.get("status") == SUCCESS and r.get("verified_score") is not None]
        scores = [int(r["verified_score"]) for r in successes]
        runtimes = [float(r.get("wall_time_seconds") or 0.0) for r in successes]
        best = _best_success(successes)
        rows.append(
            {
                "solver_id": solver_id,
                "instance_id": instance_id,
                "planned_runs": planned_runs,
                "successful_runs": len(successes),
                "error_runs": status_counts.get(ERROR, 0),
                "timeout_runs": status_counts.get(TIMEOUT, 0),
                "invalid_output_runs": status_counts.get(INVALID_OUTPUT, 0),
                "invalid_solution_runs": status_counts.get(INVALID_SOLUTION, 0),
                "interrupted_runs": status_counts.get(INTERRUPTED, 0),
                "successful_execution_ids": " ".join(str(r.get("run_id")) for r in successes),
                "best_execution_index": best.get("execution_index") if best else None,
                "best_seed": best.get("seed") if best else None,
                "best_verified_score": best.get("verified_score") if best else None,
                "best_runtime": best.get("wall_time_seconds") if best else None,
                "best_solution_path": best.get("solution_path") if best else None,
                "minimum_successful_score": min(scores) if scores else None,
                "maximum_successful_score": max(scores) if scores else None,
                "mean_score": _mean(scores),
                "median_score": _median(scores),
                "score_standard_deviation": _stdev(scores),
                "mean_runtime": _mean(runtimes),
                "median_runtime": _median(runtimes),
                "minimum_runtime": min(runtimes) if runtimes else None,
                "maximum_runtime": max(runtimes) if runtimes else None,
                "status": "OK" if successes else "NO_VALID_RUN",
            }
        )
    return rows


def solver_summary(records: List[dict], manifest: dict) -> List[dict]:
    expected = _expected_counts(manifest, by="solver_id")
    groups: dict[str, List[dict]] = defaultdict(list)
    for record in records:
        groups[str(record.get("solver_id"))].append(record)
    rows = []
    for solver_id in sorted(set(expected) | set(groups)):
        recs = groups.get(solver_id, [])
        counts = Counter(r.get("status") for r in recs)
        successes = [r for r in recs if r.get("status") == SUCCESS and r.get("verified_score") is not None]
        by_instance: dict[str, List[dict]] = defaultdict(list)
        for record in recs:
            by_instance[str(record.get("instance_id"))].append(record)
        rows.append(
            {
                "solver_id": solver_id,
                "expected_runs": expected.get(solver_id, 0),
                "completed_runs": len(recs),
                "successful_runs": len(successes),
                "errors": counts.get(ERROR, 0),
                "timeouts": counts.get(TIMEOUT, 0),
                "invalid_outputs": counts.get(INVALID_OUTPUT, 0),
                "invalid_solutions": counts.get(INVALID_SOLUTION, 0),
                "instances_10_of_10_successful": sum(
                    1 for items in by_instance.values() if sum(r.get("status") == SUCCESS for r in items) == 10
                ),
                "instances_fewer_than_10_successful": sum(
                    1 for items in by_instance.values() if 0 < sum(r.get("status") == SUCCESS for r in items) < 10
                ),
                "instances_0_successful": sum(
                    1 for items in by_instance.values() if sum(r.get("status") == SUCCESS for r in items) == 0
                ),
                "mean_runtime": _mean([float(r.get("wall_time_seconds") or 0.0) for r in successes]),
                "median_runtime": _median([float(r.get("wall_time_seconds") or 0.0) for r in successes]),
                "mean_verified_score": _mean([int(r["verified_score"]) for r in successes]),
                "median_verified_score": _median([int(r["verified_score"]) for r in successes]),
            }
        )
    return rows


def instance_summary(records: List[dict], manifest: dict) -> List[dict]:
    instance_meta = {
        row["instance_id"]: row
        for row in (manifest.get("benchmark_manifest", {}) or {}).get("instances", [])
    }
    groups: dict[tuple[str, str], List[dict]] = defaultdict(list)
    for record in records:
        groups[(str(record.get("instance_id")), str(record.get("solver_id")))].append(record)
    rows = []
    for (instance_id, solver_id), recs in sorted(groups.items()):
        successes = [r for r in recs if r.get("status") == SUCCESS and r.get("verified_score") is not None]
        scores = [int(r["verified_score"]) for r in successes]
        best = _best_success(successes)
        meta = instance_meta.get(instance_id, {})
        rows.append(
            {
                "instance_id": instance_id,
                "family": meta.get("family"),
                "contributors": meta.get("contributors"),
                "projects": meta.get("projects"),
                "skills": meta.get("skills"),
                "roles": meta.get("roles"),
                "solver_id": solver_id,
                "success_count": len(successes),
                "best_verified_score": best.get("verified_score") if best else None,
                "mean_verified_score": _mean(scores),
                "median_score": _median(scores),
                "score_standard_deviation": _stdev(scores),
                "best_execution": best.get("execution_index") if best else None,
                "best_seed": best.get("seed") if best else None,
                "best_runtime": best.get("wall_time_seconds") if best else None,
            }
        )
    return rows


def _pair_records(campaign_dir: Path, solver_id: str, instance_id: str) -> Iterable[dict]:
    pair_dir = campaign_dir / "runs" / safe_id(solver_id) / safe_id(instance_id)
    for result_file in sorted(pair_dir.glob("exec_*/result.json")):
        result = read_json(result_file, default=None)
        if isinstance(result, dict):
            yield result


def _best_success(records: List[dict]) -> dict | None:
    if not records:
        return None
    return sorted(
        records,
        key=lambda r: (
            -int(r["verified_score"]),
            float(r.get("wall_time_seconds") or 1e100),
            int(r.get("execution_index") or 10**9),
        ),
    )[0]


def _best_rows(pair_rows: List[dict]) -> List[dict]:
    return [
        {
            "solver_id": row.get("solver_id"),
            "instance_id": row.get("instance_id"),
            "best_execution_index": row.get("best_execution_index"),
            "best_seed": row.get("best_seed"),
            "best_verified_score": row.get("best_verified_score"),
            "best_runtime": row.get("best_runtime"),
            "best_solution_path": row.get("best_solution_path"),
            "status": row.get("status"),
        }
        for row in pair_rows
    ]


def _expected_counts(manifest: dict, by: str) -> Dict[str, int]:
    runs = int(manifest.get("runs") or 0)
    solvers = (manifest.get("solver_manifest", {}) or {}).get("solvers", [])
    instances = (manifest.get("benchmark_manifest", {}) or {}).get("instances", [])
    if by == "solver_id":
        return {row["solver_id"]: len(instances) * runs for row in solvers}
    return {row["instance_id"]: len(solvers) * runs for row in instances}


def _flatten(data: dict, prefix: str = "") -> dict:
    row = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            row.update(_flatten(value, name))
        elif isinstance(value, list):
            row[name] = " ".join(str(item) for item in value)
        else:
            row[name] = value
    return row


def _write_csv(path: str | Path, rows: List[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        atomic_write_text(path, "")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    atomic_write_text(path, buffer.getvalue())


def _mean(values: List[float] | List[int]) -> float | None:
    return float(statistics.mean(values)) if values else None


def _median(values: List[float] | List[int]) -> float | None:
    return float(statistics.median(values)) if values else None


def _stdev(values: List[float] | List[int]) -> float | None:
    return float(statistics.stdev(values)) if len(values) > 1 else (0.0 if values else None)

