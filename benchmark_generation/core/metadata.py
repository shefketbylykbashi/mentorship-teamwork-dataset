"""Metadata, manifest and report helpers."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .distance import pairwise_distances, weighted_distance
from .features import FEATURE_NAMES


def config_hash(config: Mapping[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def flatten_mapping(
    mapping: Mapping[str, Any],
    prefix: str = "",
    result: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if result is None:
        result = {}
    for key, value in mapping.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flatten_mapping(value, full_key, result)
        elif isinstance(value, (list, tuple)):
            result[full_key] = json.dumps(value, sort_keys=True)
        else:
            result[full_key] = value
    return result


def build_metadata_row(
    filename: str,
    family: str,
    family_instance_id: int,
    seed: int,
    generator_version: str,
    parameters: Mapping[str, Any],
    features: Mapping[str, float],
    validation: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "filename": filename,
        "family": family,
        "family_instance_id": family_instance_id,
        "seed": seed,
        "generator_version": generator_version,
    }
    row.update(flatten_mapping({"parameters": parameters}))
    row.update({f"feature.{name}": float(features.get(name, 0.0)) for name in FEATURE_NAMES})
    row.update(flatten_mapping({"validation": validation}))
    if extra:
        row.update(flatten_mapping(extra))
    return row


def write_metadata(rows: Sequence[Mapping[str, Any]], csv_path: str | Path, jsonl_path: str | Path) -> None:
    csv_path = Path(csv_path)
    jsonl_path = Path(jsonl_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def write_manifest(
    path: str | Path,
    manifest: Mapping[str, Any],
    overwrite: bool = True,
) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def update_master_manifest(path: str | Path, family: str, entry: Mapping[str, Any]) -> None:
    path = Path(path)
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
    else:
        manifest = {"families": {}}
    manifest.setdefault("families", {})[family] = dict(entry)
    write_manifest(path, manifest, overwrite=True)


def summarize_numeric(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"min": 0.0, "median": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": float(min(values)),
        "median": float(median(values)),
        "mean": float(mean(values)),
        "max": float(max(values)),
    }


def family_statistical_report(
    rows: Sequence[Mapping[str, Any]],
    normalized_vectors: Sequence[Mapping[str, float]],
    group_weights: Mapping[str, float] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {"count": len(rows)}
    for feature_name in FEATURE_NAMES:
        key = f"feature.{feature_name}"
        values = [float(row[key]) for row in rows if key in row and row[key] != ""]
        if values:
            report[feature_name] = summarize_numeric(values)
    distances = pairwise_distances(list(normalized_vectors), group_weights)
    if distances:
        report["within_family_pairwise_distance"] = summarize_numeric(distances)
        report["within_family_min_pairwise_distance"] = min(distances)
    else:
        report["within_family_pairwise_distance"] = summarize_numeric([])
    report["degeneracy_flags"] = degeneracy_flags(rows)
    if extra:
        report.update(extra)
    return report


def nearest_distance(
    vector: Mapping[str, float],
    others: Iterable[Mapping[str, float]],
    group_weights: Mapping[str, float] | None = None,
) -> float:
    distances = [weighted_distance(vector, other, group_weights) for other in others]
    return min(distances) if distances else float("inf")


def degeneracy_flags(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    flags = []
    if not rows:
        return ["empty_family"]
    watched = [
        "feature.n_contributors",
        "feature.n_projects",
        "feature.total_roles",
        "feature.contention",
        "feature.direct_fillability",
        "feature.mentorship_fillability",
        "feature.deadline_tightness_mean",
        "feature.score_density",
    ]
    for key in watched:
        values = [float(row[key]) for row in rows if key in row and row[key] != ""]
        if len(values) > 1 and max(values) - min(values) < 1e-9:
            flags.append(f"constant_{key}")
    fillability = [
        max(float(row["feature.direct_fillability"]), float(row["feature.mentorship_fillability"]))
        for row in rows
        if "feature.direct_fillability" in row and "feature.mentorship_fillability" in row
    ]
    if fillability and median(fillability) < 0.30:
        flags.append("median_initial_fillability_below_default_threshold")
    return flags
