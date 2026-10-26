"""Write descriptive, tie-aware portfolio diagnostics for an ISA export.

These are benchmark summaries, not out-of-sample selector estimates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def summarize(directory: Path, *, overwrite: bool = False) -> dict[str, Any]:
    """Summarize observed good labels and attainable oracle headroom."""
    directory = directory.resolve()
    output = directory / "preflight_report.json"
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {output}")
    audit = json.loads((directory / "audit.json").read_text(encoding="utf-8"))
    for name in ("metadata.csv", "instance_audit.csv"):
        expected = audit.get("output_sha256", {}).get(name)
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if expected != actual:
            raise ValueError(f"ISA export checksum mismatch: {name}")
    rows = _load_csv(directory / "metadata.csv")
    instance_audit = _load_csv(directory / "instance_audit.csv")
    if len(rows) != audit["n_instances"] or len(instance_audit) != len(rows):
        raise ValueError("ISA export row count does not match audit")
    solver_ids = audit["primary_solver_ids"]
    threshold = float(audit["good_threshold"])
    if not solver_ids:
        raise ValueError("No primary algorithms")
    by_id = {row["instances"]: row for row in instance_audit}
    if len(by_id) != len(rows) or {row["instances"] for row in rows} != set(by_id):
        raise ValueError("ISA audit identifiers are incomplete or duplicated")
    by_family: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["source"] != by_id[row["instances"]]["source"]:
            raise ValueError(f"Family mismatch for {row['instances']}")
        for solver in solver_ids:
            value = float(row[f"algo_{solver}"])
            if not math.isfinite(value) or not 0 <= value <= 1 + 1e-10:
                raise ValueError(f"Invalid normalized utility for {row['instances']} / {solver}")
        by_family[row["source"]].append(row)

    def group_summary(group: list[dict[str, str]]) -> dict[str, Any]:
        good_counts = [sum(float(row[f"algo_{s}"]) >= threshold for s in solver_ids) for row in group]
        ties = [int(by_id[row["instances"]]["co_best_count"]) for row in group]
        solver_metrics = {}
        for solver in solver_ids:
            ratios = [float(row[f"algo_{solver}"]) for row in group]
            solver_metrics[solver] = {
                "mean_ratio": mean(ratios),
                "median_ratio": median(ratios),
                "mean_normalized_regret": mean(1 - value for value in ratios),
                "good_count": sum(value >= threshold for value in ratios),
                "good_fraction": sum(value >= threshold for value in ratios) / len(group),
            }
        return {
            "count": len(group),
            "number_good_distribution": dict(sorted(Counter(good_counts).items())),
            "cobest_tie_count": sum(count > 1 for count in ties),
            "all_solvers_cobest_count": sum(count == len(solver_ids) for count in ties),
            "easy_count_beta_0_55": sum(count / len(solver_ids) > 0.55 for count in good_counts),
            "solver_metrics": solver_metrics,
        }

    pooled = group_summary(rows)
    families = {family: group_summary(group) for family, group in sorted(by_family.items())}
    pooled_best = max(solver_ids, key=lambda s: pooled["solver_metrics"][s]["mean_ratio"])
    family_balanced = {
        solver: mean(families[family]["solver_metrics"][solver]["mean_ratio"] for family in families)
        for solver in solver_ids
    }
    balanced_best = max(solver_ids, key=lambda s: family_balanced[s])
    report: dict[str, Any] = {
        "interpretation": "descriptive benchmark diagnostics; oracle uses observed same-instance outcomes and is unattainable at deployment",
        "good_threshold": threshold,
        "portfolio_size": len(solver_ids),
        "pooled": pooled,
        "by_family": families,
        "best_fixed_solver_pooled": pooled_best,
        "best_fixed_mean_ratio_pooled": pooled["solver_metrics"][pooled_best]["mean_ratio"],
        "pooled_oracle_ratio": 1.0,
        "pooled_oracle_headroom_vs_best_fixed": 1 - pooled["solver_metrics"][pooled_best]["mean_ratio"],
        "family_balanced_mean_ratio": family_balanced,
        "best_fixed_solver_family_balanced": balanced_best,
        "family_balanced_oracle_headroom_vs_best_fixed": 1 - family_balanced[balanced_best],
        "caution": "The 611 designed instances are not a random population sample; these are descriptive scores, not confidence intervals or cross-validated selector accuracy.",
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("isa/outputs/rizana_trial_01"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = summarize(args.input, overwrite=args.overwrite)
    print(json.dumps({
        "best_fixed_solver_pooled": result["best_fixed_solver_pooled"],
        "best_fixed_mean_ratio_pooled": result["best_fixed_mean_ratio_pooled"],
        "pooled_oracle_headroom_vs_best_fixed": result["pooled_oracle_headroom_vs_best_fixed"],
        "family_counts": {key: value["count"] for key, value in result["by_family"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
