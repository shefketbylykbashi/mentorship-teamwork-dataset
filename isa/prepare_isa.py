"""Audit a completed solver campaign and export MATLAB InstanceSpace metadata.

The primary export is descriptive.  Feature filtering never inspects solver
outcomes; predictive claims require a separate, genuinely nested evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from benchmark_generation.core.features import FEATURE_NAMES, extract_features
from benchmark_generation.core.parser import parse_file


# Each entry is (raw feature, ISA column suffix, transform, conceptual group).
# This is a pre-declared, solver-independent shortlist.  The only automatic
# exclusions are constant and almost perfectly rank-correlated columns.
FEATURE_POLICY = (
    ("n_contributors", "log1p_n_contributors", "log1p", "size"),
    ("n_projects", "log1p_n_projects", "log1p", "size"),
    ("roles_per_project_mean", "roles_per_project_mean", "identity", "size"),
    ("contention", "log1p_contention", "log1p", "supply_demand"),
    ("demand_supply_alignment", "demand_supply_alignment", "identity", "supply_demand"),
    ("unmet_skill_share", "unmet_skill_share", "identity", "supply_demand"),
    ("contributor_level_mean", "contributor_level_mean", "identity", "levels"),
    ("level_gap_mean", "level_gap_mean", "identity", "levels"),
    ("direct_fillability", "direct_fillability", "identity", "mentorship"),
    ("learning_reachable_demand", "learning_reachable_demand", "identity", "mentorship"),
    ("deadline_tightness_mean", "deadline_tightness_mean", "identity", "time"),
    ("schedule_pressure", "log1p_schedule_pressure", "log1p", "time"),
    ("score_concentration", "score_concentration", "identity", "scores"),
    ("score_density", "log1p_score_density", "log1p", "scores"),
    ("score_effort_correlation", "score_effort_correlation", "identity", "scores"),
)
EXCLUDED_FEATURES = {
    "mentor_availability": "exact duplicate of mentorship_fillability in the current extractor",
    "mentorship_fillability": "legacy role-level proxy misses zero-skill mentees for level-1 roles",
    "within_project_mentorship_feasibility": "legacy proxy does not verify a distinct feasible team assignment",
}
DEFAULT_GOOD_THRESHOLD = 0.95
DEFAULT_REDUNDANCY_THRESHOLD = 0.98


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_campaign_hash(path: Path) -> tuple[str, bool]:
    """Account only for Git's LF/CRLF checkout conversion, never content edits."""
    data = path.read_bytes()
    canonical = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(canonical).hexdigest(), canonical != data


def _as_int(row: dict[str, str], field: str) -> int:
    return int(row[field])


def expected_score(row: dict[str, str]) -> float:
    """Mean verified score over *planned* runs; a timeout contributes zero.

    Campaign `mean_score` is conditional on success and otherwise overstates
    algorithms with timeout risk. This is a one-attempt utility under the
    campaign's nominal 180-s solver limit and 15-s external grace, not a
    reconstructed strict 180-s deadline utility.
    """
    planned = _as_int(row, "planned_runs")
    success = _as_int(row, "successful_runs")
    timeout = _as_int(row, "timeout_runs")
    error = _as_int(row, "error_runs")
    interrupted = _as_int(row, "interrupted_runs")
    invalid = _as_int(row, "invalid_output_runs") + _as_int(row, "invalid_solution_runs")
    if planned <= 0 or min(success, timeout, error, interrupted, invalid) < 0:
        raise ValueError(f"Invalid run counts: {row['instance_id']} / {row['solver_id']}")
    if success + timeout + error + interrupted + invalid != planned:
        raise ValueError(f"Run counts do not sum to planned: {row['instance_id']} / {row['solver_id']}")
    if error or interrupted or invalid:
        raise ValueError(f"Campaign contains non-timeout failures: {row['instance_id']} / {row['solver_id']}")
    score = float(row["mean_score"]) if success else 0.0
    if not math.isfinite(score) or score < 0:
        raise ValueError(f"Invalid verified mean score: {row['instance_id']} / {row['solver_id']}")
    return score * success / planned


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    pos = 0
    while pos < len(order):
        end = pos + 1
        while end < len(order) and values[order[end]] == values[order[pos]]:
            end += 1
        average_rank = (pos + end - 1) / 2.0
        for j in range(pos, end):
            result[order[j]] = average_rank
        pos = end
    return result


def _pearson(left: list[float], right: list[float]) -> float:
    lm, rm = mean(left), mean(right)
    a = [x - lm for x in left]
    b = [x - rm for x in right]
    denom = math.sqrt(sum(x * x for x in a) * sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / denom if denom else 0.0


def _transform(value: float, kind: str) -> float:
    if kind == "identity":
        return value
    if kind == "log1p" and value >= 0:
        return math.log1p(value)
    raise ValueError(f"Invalid {kind} transform input: {value}")


def select_features(
    feature_rows: list[dict[str, float]], threshold: float = DEFAULT_REDUNDANCY_THRESHOLD
) -> tuple[list[str], list[dict[str, Any]], dict[str, list[float]]]:
    """Domain shortlist, then variance and Spearman redundancy filtering.

    This sees only static instance features.  Original order sets priority;
    high rank correlation does not remove the last feature of a domain group.
    """
    if not feature_rows:
        raise ValueError("No instances to select features from")
    if not 0 < threshold <= 1:
        raise ValueError("Redundancy threshold must lie in (0, 1]")
    values: dict[str, list[float]] = {}
    decisions: list[dict[str, Any]] = []
    selected: list[str] = []
    selected_groups: Counter[str] = Counter()
    for raw, name, transform, group in FEATURE_POLICY:
        column = [_transform(float(row[raw]), transform) for row in feature_rows]
        if not all(math.isfinite(x) for x in column):
            raise ValueError(f"Nonfinite feature: {raw}")
        values[name] = column
        decision: dict[str, Any] = {
            "raw_feature": raw,
            "column": f"feature_{name}",
            "transform": transform,
            "group": group,
            "status": "selected",
        }
        if max(column) - min(column) <= 1e-12:
            decision["status"] = "excluded_constant"
        else:
            ranked = _rank(column)
            redundant = [
                (prior, _pearson(ranked, _rank(values[prior])))
                for prior in selected
            ]
            close = [(prior, corr) for prior, corr in redundant if abs(corr) >= threshold]
            # Keep at least one member of each substantive feature group.
            if close and selected_groups[group] > 0:
                prior, corr = max(close, key=lambda pair: abs(pair[1]))
                decision.update(status="excluded_redundant", with_column=f"feature_{prior}", spearman=corr)
        if decision["status"] == "selected":
            selected.append(name)
            selected_groups[group] += 1
        decisions.append(decision)
    if len(selected) < 3:
        raise ValueError("Fewer than three non-redundant features remain")
    return selected, decisions, values


def _solver_selection(manifest: dict[str, Any]) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    solvers = manifest["solvers"]
    if manifest["count"] != len(solvers):
        raise ValueError("Solver manifest count mismatch")
    all_ids = [item["solver_id"] for item in solvers]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("Duplicate solver ID")
    groups: dict[str, list[str]] = defaultdict(list)
    for item in solvers:
        if item["objective"] != "maximize" or not item.get("source_sha256"):
            raise ValueError(f"Unsupported solver manifest entry: {item['solver_id']}")
        groups[item["source_sha256"]].append(item["solver_id"])
    representatives = [ids[0] for ids in groups.values()]
    source_groups = [
        {"source_sha256": source_hash, "representative": ids[0], "solver_ids": ids}
        for source_hash, ids in groups.items()
    ]
    return all_ids, representatives, source_groups


def prepare(
    repo_root: Path,
    campaign: Path,
    output: Path,
    *,
    good_threshold: float = DEFAULT_GOOD_THRESHOLD,
    redundancy_threshold: float = DEFAULT_REDUNDANCY_THRESHOLD,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create an audited ISA snapshot without modifying inputs or campaign data."""
    repo_root = repo_root.resolve()
    campaign = (repo_root / campaign).resolve() if not campaign.is_absolute() else campaign.resolve()
    output = (repo_root / output).resolve() if not output.is_absolute() else output.resolve()
    if not 0 < good_threshold <= 1:
        raise ValueError("Good threshold must lie in (0, 1]")
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"Refusing to overwrite nonempty ISA output directory: {output}")
    benchmark_path = campaign / "benchmark_manifest.json"
    solver_path = campaign / "solver_manifest.json"
    summary_path = campaign / "solver_instance_summary.csv"
    benchmark = _read_json(benchmark_path)
    solver_manifest = _read_json(solver_path)
    summary_rows = _read_csv(summary_path)
    instances = benchmark["instances"]
    if benchmark["count"] != len(instances):
        raise ValueError("Benchmark manifest count mismatch")
    ids = [item["instance_id"] for item in instances]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate instance ID in benchmark manifest")
    all_solvers, representatives, source_groups = _solver_selection(solver_manifest)
    pairs: dict[tuple[str, str], dict[str, str]] = {}
    for row in summary_rows:
        key = (row["instance_id"], row["solver_id"])
        if key in pairs:
            raise ValueError(f"Duplicate solver-instance summary: {key}")
        pairs[key] = row
    expected_pairs = {(iid, solver) for iid in ids for solver in all_solvers}
    if set(pairs) != expected_pairs:
        missing = expected_pairs - set(pairs)
        extra = set(pairs) - expected_pairs
        raise ValueError(f"Incomplete solver-instance matrix: {len(missing)} missing, {len(extra)} extra")

    features: list[dict[str, float]] = []
    normalized_line_endings = 0
    hash_set: set[str] = set()
    for index, item in enumerate(instances, 1):
        path = (repo_root / item["path"]).resolve()
        if not path.is_relative_to(repo_root) or not path.is_file():
            raise ValueError(f"Benchmark path escapes repository or is missing: {item['path']}")
        current_hash, was_normalized = _canonical_campaign_hash(path)
        if current_hash != item["sha256"]:
            raise ValueError(f"Input hash mismatch for {item['instance_id']}: {item['path']}")
        if current_hash in hash_set:
            raise ValueError(f"Duplicate input content: {item['instance_id']}")
        hash_set.add(current_hash)
        normalized_line_endings += int(was_normalized)
        parsed = parse_file(path)
        counts = (len(parsed.contributors), len(parsed.projects), parsed.total_roles, len(parsed.skill_universe))
        recorded = (item["contributors"], item["projects"], item["roles"], item["skills"])
        if counts != recorded:
            raise ValueError(f"Input counts mismatch for {item['instance_id']}: {counts} != {recorded}")
        extracted = extract_features(parsed)
        if set(extracted) != set(FEATURE_NAMES) or not all(math.isfinite(x) for x in extracted.values()):
            raise ValueError(f"Invalid feature vector for {item['instance_id']}")
        features.append(extracted)
        if index % 100 == 0 or index == len(instances):
            print(f"Audited and featurized {index}/{len(instances)} inputs")

    selected, decisions, transformed = select_features(features, redundancy_threshold)
    utility: dict[tuple[str, str], float] = {key: expected_score(row) for key, row in pairs.items()}
    all_solver_rows: list[dict[str, Any]] = []
    instance_audit_rows: list[dict[str, Any]] = []
    primary_rows: list[dict[str, Any]] = []
    all_solver_metadata_rows: list[dict[str, Any]] = []
    best_of_ten_rows: list[dict[str, Any]] = []
    ties = 0
    all_equal = 0
    near_equal = 0
    zero_success_pairs = 0
    total_timeouts = 0
    for i, item in enumerate(instances):
        iid = item["instance_id"]
        best = max(utility[(iid, solver)] for solver in representatives)
        if best <= 0:
            raise ValueError(f"No positive verified utility for {iid}")
        rep_ratios = {solver: utility[(iid, solver)] / best for solver in representatives}
        co_best = [solver for solver, ratio in rep_ratios.items() if math.isclose(ratio, 1.0, abs_tol=1e-12)]
        good = [solver for solver, ratio in rep_ratios.items() if ratio >= good_threshold]
        ties += len(co_best) > 1
        all_equal += len(co_best) == len(representatives)
        near_equal += min(rep_ratios.values()) >= 0.99
        base: dict[str, Any] = {"instances": iid, "source": item["family"]}
        base.update({f"feature_{name}": transformed[name][i] for name in selected})
        primary_rows.append({**base, **{f"algo_{solver}": rep_ratios[solver] for solver in representatives}})
        all_best = max(utility[(iid, solver)] for solver in all_solvers)
        all_solver_metadata_rows.append({
            **base, **{f"algo_{solver}": utility[(iid, solver)] / all_best for solver in all_solvers}
        })
        best_ten = {
            solver: float(pairs[(iid, solver)]["maximum_successful_score"] or 0)
            for solver in representatives
        }
        best_ten_max = max(best_ten.values())
        best_of_ten_rows.append({
            **base, **{f"algo_{solver}": best_ten[solver] / best_ten_max for solver in representatives}
        })
        instance_audit_rows.append({
            "instances": iid,
            "source": item["family"],
            "best_expected_score": best,
            "co_best_solver_ids": " ".join(co_best),
            "co_best_count": len(co_best),
            "good_solver_ids": " ".join(good),
            "good_solver_count": len(good),
            "worst_to_best_ratio": min(rep_ratios.values()),
        })
        for solver in all_solvers:
            raw = pairs[(iid, solver)]
            success = _as_int(raw, "successful_runs")
            timeout = _as_int(raw, "timeout_runs")
            zero_success_pairs += success == 0
            total_timeouts += timeout
            all_solver_rows.append({
                "instances": iid,
                "source": item["family"],
                "solver_id": solver,
                "source_representative": next(group["representative"] for group in source_groups if solver in group["solver_ids"]),
                "planned_runs": _as_int(raw, "planned_runs"),
                "successful_runs": success,
                "timeout_runs": timeout,
                "mean_success_score": float(raw["mean_score"]) if success else 0.0,
                "expected_score": utility[(iid, solver)],
                "maximum_successful_score": float(raw["maximum_successful_score"] or 0),
                "ratio_to_primary_portfolio_best": utility[(iid, solver)] / best,
            })

    output.mkdir(parents=True, exist_ok=True)
    feature_fields = ["instances", "source", *FEATURE_NAMES]
    _write_csv(output / "all_features.csv", feature_fields, [
        {"instances": item["instance_id"], "source": item["family"], **f}
        for item, f in zip(instances, features)
    ])
    base_fields = ["instances", "source", *(f"feature_{name}" for name in selected)]
    _write_csv(output / "metadata.csv", [*base_fields, *(f"algo_{s}" for s in representatives)], primary_rows)
    _write_csv(output / "metadata_all_solvers.csv", [*base_fields, *(f"algo_{s}" for s in all_solvers)], all_solver_metadata_rows)
    _write_csv(output / "metadata_best_of_ten.csv", [*base_fields, *(f"algo_{s}" for s in representatives)], best_of_ten_rows)
    _write_csv(output / "all_solvers.csv", list(all_solver_rows[0]), all_solver_rows)
    _write_csv(output / "instance_audit.csv", list(instance_audit_rows[0]), instance_audit_rows)
    feature_selection = {
        "policy": "prespecified solver-independent domain shortlist; remove constants and |Spearman rho| >= threshold unless final group member",
        "redundancy_threshold": redundancy_threshold,
        "candidate_feature_count": len(FEATURE_NAMES),
        "selected_count": len(selected),
        "selected_columns": [f"feature_{name}" for name in selected],
        "decisions": decisions,
        "not_shortlisted": [
            name for name in FEATURE_NAMES if name not in {item[0] for item in FEATURE_POLICY}
        ],
        "excluded_legacy_features": EXCLUDED_FEATURES,
        "scaling": "Only explicit log1p transforms here; InstanceSpace PRELIM performs its configured normalization.",
    }
    (output / "feature_selection.json").write_text(json.dumps(feature_selection, indent=2) + "\n", encoding="utf-8")
    audit: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "exploratory first ISA; descriptive footprints, not validated prediction",
        "campaign": str(campaign.relative_to(repo_root)) if campaign.is_relative_to(repo_root) else str(campaign),
        "feature_extractor_sha256": _sha256(repo_root / "benchmark_generation/core/features.py"),
        "isa_exporter_sha256": _sha256(Path(__file__)),
        "benchmark_manifest_sha256": _sha256(benchmark_path),
        "solver_manifest_sha256": _sha256(solver_path),
        "solver_instance_summary_sha256": _sha256(summary_path),
        "n_instances": len(instances),
        "family_counts": dict(Counter(item["family"] for item in instances)),
        "n_solver_labels": len(all_solvers),
        "n_source_distinct_solvers": len(representatives),
        "primary_solver_ids": representatives,
        "source_groups": source_groups,
        "n_solver_instance_pairs": len(pairs),
        "n_zero_success_pairs": zero_success_pairs,
        "total_timeouts": total_timeouts,
        "inputs_with_lf_normalized_for_manifest_hash": normalized_line_endings,
        "primary_utility": "mean_success_score * successful_runs / planned_runs; timeout is zero",
        "primary_normalization": "divide each solver utility by the maximum utility among seven source-distinct representatives for the same instance",
        "good_threshold": good_threshold,
        "n_instances_with_cobest_tie": ties,
        "n_instances_all_primary_solvers_equal": all_equal,
        "n_instances_all_primary_solvers_within_one_percent": near_equal,
        "feature_selection_file": "feature_selection.json",
        "type3_caution": "98 campaign Type 3 inputs lack a finalized family metadata/report; do not call this a finalized benchmark",
        "run_level_caution": "per_run_results.csv is not in this checkout; run-level uncertainty and seed-paired tests cannot be reconstructed from summary alone",
        "toolkit_tie_caution": "InstanceSpace PRELIM randomly selects one of exact best solvers for P; use tie-aware co-best/good labels, not raw P alone",
    }
    audit["output_sha256"] = {
        name: _sha256(output / name)
        for name in ("metadata.csv", "metadata_all_solvers.csv", "metadata_best_of_ten.csv", "all_features.csv", "all_solvers.csv", "instance_audit.csv", "feature_selection.json")
    }
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--campaign", type=Path, default=Path("experiment_results/rizana_trial_01"))
    parser.add_argument("--output", type=Path, default=Path("isa/outputs/rizana_trial_01"))
    parser.add_argument("--good-threshold", type=float, default=DEFAULT_GOOD_THRESHOLD)
    parser.add_argument("--redundancy-threshold", type=float, default=DEFAULT_REDUNDANCY_THRESHOLD)
    parser.add_argument("--overwrite", action="store_true", help="Replace only named ISA export files in an existing output directory")
    args = parser.parse_args()
    audit = prepare(
        args.repo_root,
        args.campaign,
        args.output,
        good_threshold=args.good_threshold,
        redundancy_threshold=args.redundancy_threshold,
        overwrite=args.overwrite,
    )
    print(json.dumps({key: audit[key] for key in (
        "n_instances", "n_source_distinct_solvers", "n_solver_instance_pairs", "total_timeouts",
        "n_instances_with_cobest_tie", "n_instances_all_primary_solvers_equal",
    )}, indent=2))


if __name__ == "__main__":
    main()
