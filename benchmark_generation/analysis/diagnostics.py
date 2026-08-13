"""Post-generation diagnostics for measured MTP benchmark features.

This script is diagnostic only. It does not participate in instance generation,
does not implement ISA, and must not be used as a generation mechanism.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from benchmark_generation.core.distance import FeatureScaler, pairwise_distances, weighted_distance
from benchmark_generation.core.features import FEATURE_NAMES
from benchmark_generation.core.metadata import summarize_numeric, write_manifest


def read_metadata(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def feature_dict(row: Dict[str, str]) -> Dict[str, float]:
    return {
        name: float(row.get(f"feature.{name}") or 0.0)
        for name in FEATURE_NAMES
    }


def build_diagnostics(metadata_dir: str | Path, output_dir: str | Path) -> Dict[str, object]:
    metadata_dir = Path(metadata_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_family = {
        family: read_metadata(metadata_dir / f"{family}_metadata.csv")
        for family in ("type2", "type3", "type4")
    }
    all_rows = [row for rows in rows_by_family.values() for row in rows]
    all_features = [feature_dict(row) for row in all_rows]
    report: Dict[str, object] = {
        "note": (
            "This visualization/report is diagnostic only and is not used as "
            "the instance-generation mechanism or final ISA projection."
        ),
        "counts": {family: len(rows) for family, rows in rows_by_family.items()},
    }
    if not all_features:
        report["warning"] = "No generated metadata rows found."
        write_manifest(output_dir / "diagnostic_report.json", report)
        return report

    scaler = FeatureScaler.fit(all_features)
    normalized_by_family = {
        family: [scaler.normalize(feature_dict(row)) for row in rows]
        for family, rows in rows_by_family.items()
    }
    for family, vectors in normalized_by_family.items():
        distances = pairwise_distances(vectors, scaler.group_weights)
        report[f"{family}_pairwise_distances"] = summarize_numeric(distances)
    for family, rows in rows_by_family.items():
        feature_report = {}
        for feature_name in FEATURE_NAMES:
            values = [float(row.get(f"feature.{feature_name}") or 0.0) for row in rows]
            if values:
                feature_report[feature_name] = summarize_numeric(values)
        report[f"{family}_feature_distributions"] = feature_report

    _write_feature_distribution_csv(rows_by_family, output_dir)
    _try_write_plots(rows_by_family, output_dir)
    write_manifest(output_dir / "diagnostic_report.json", report)
    return report


def _write_feature_distribution_csv(
    rows_by_family: Dict[str, List[Dict[str, str]]],
    output_dir: Path,
) -> None:
    path = output_dir / "feature_distributions_by_family.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["family", "feature", "min", "median", "mean", "max"])
        for family, rows in rows_by_family.items():
            for feature_name in FEATURE_NAMES:
                values = [float(row.get(f"feature.{feature_name}") or 0.0) for row in rows]
                if not values:
                    continue
                summary = summarize_numeric(values)
                writer.writerow(
                    [
                        family,
                        feature_name,
                        summary["min"],
                        summary["median"],
                        summary["mean"],
                        summary["max"],
                    ]
                )


def _try_write_plots(rows_by_family: Dict[str, List[Dict[str, str]]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        return

    watched = [
        "n_contributors",
        "n_projects",
        "total_roles",
        "contention",
        "direct_fillability",
        "mentorship_fillability",
        "deadline_tightness_mean",
        "score_density",
    ]
    for feature_name in watched:
        plt.figure(figsize=(8, 5))
        for family, rows in rows_by_family.items():
            values = [float(row.get(f"feature.{feature_name}") or 0.0) for row in rows]
            if values:
                plt.hist(values, bins=30, alpha=0.45, label=family)
        plt.title(f"{feature_name} by family")
        plt.xlabel(feature_name)
        plt.ylabel("count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{feature_name}_by_family.png", dpi=150)
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", default="benchmark_metadata")
    parser.add_argument("--output-dir", default="benchmark_metadata/diagnostics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_diagnostics(args.metadata_dir, args.output_dir)


if __name__ == "__main__":
    main()
