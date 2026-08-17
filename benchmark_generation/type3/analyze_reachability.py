"""Empirical Type 3 anchor reachability and radius calibration.

This diagnostic is required before final Type 3 generation. It measures the
local resolution of the forward generator around each immutable Google anchor
and writes calibrated, anchor-specific radius bands for hard Type 3 acceptance.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping

from benchmark_generation.core.anchors import load_google_anchors
from benchmark_generation.core.config import load_config
from benchmark_generation.core.distance import FeatureScaler, feature_weights, weighted_distance
from benchmark_generation.core.features import FEATURE_GROUPS, FEATURE_NAMES, extract_features
from benchmark_generation.core.generator import derive_seed, generate_valid_instance
from benchmark_generation.core.metadata import summarize_numeric, write_manifest
from benchmark_generation.core.parameter_sampling import (
    blend_difficulty_mix,
    bounds_from_anchor_estimates,
    estimate_parameters_from_instance,
    perturb_parameters,
)
from benchmark_generation.type3.generate_type3 import DEFAULT_CONFIG


DEFAULT_REACHABILITY_CONFIG: Dict[str, Any] = {
    "samples_per_anchor": 200,
    "max_attempt_factor": 8,
    "max_generation_attempts": 10,
    "perturbation_scales": [0.0, 0.02, 0.05, 0.10, 0.20, 0.30],
    "viability_difficulty_prior": [0.35, 0.35, 0.20, 0.10],
    "viability_blends": [1.0, 0.75, 0.50, 0.25, 0.0],
    "calibration_quantiles": [0.05, 0.25, 0.50, 0.75, 0.90],
    "band_names": ["inner", "near", "intermediate", "outer"],
    "minimum_band_width": 1e-6,
}


def analyze_reachability(
    google_dir: str | Path,
    metadata_dir: str | Path,
    seed: int,
    type3_config: Mapping[str, Any],
    reachability_config: Mapping[str, Any],
) -> Dict[str, Any]:
    google_dir = Path(google_dir)
    metadata_dir = Path(metadata_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    anchors = load_google_anchors(google_dir)
    anchor_names = [str(anchor.source_name) for anchor in anchors]
    anchor_features = [extract_features(anchor) for anchor in anchors]
    scaler = FeatureScaler.fit(
        anchor_features,
        group_weights=type3_config.get("distance_group_weights"),
    )
    anchor_vectors = [scaler.normalize(features) for features in anchor_features]
    anchor_params = [estimate_parameters_from_instance(anchor) for anchor in anchors]
    anchor_bounds = [
        bounds_from_anchor_estimates([params], type3_config["parameter_search"])
        for params in anchor_params
    ]

    rows = []
    anchors_report: Dict[str, Any] = {}
    calibrated = {
        "source": "empirical_forward_generator_reachability",
        "root_seed": seed,
        "samples_per_anchor": int(reachability_config["samples_per_anchor"]),
        "calibration_quantiles": list(reachability_config["calibration_quantiles"]),
        "anchors": {},
    }

    for anchor_idx, anchor_name in enumerate(anchor_names):
        logging.info("Reachability anchor %s", anchor_name)
        result = analyze_anchor(
            anchor_name=anchor_name,
            anchor_idx=anchor_idx,
            anchor_vector=anchor_vectors[anchor_idx],
            center_params=anchor_params[anchor_idx],
            bounds=anchor_bounds[anchor_idx],
            scaler=scaler,
            seed=derive_seed(seed, anchor_idx + 1, salt=271_828),
            reachability_config=reachability_config,
        )
        anchors_report[anchor_name] = result
        rows.append(flatten_anchor_report(anchor_name, result))
        calibrated["anchors"][anchor_name] = {
            "bands": calibrate_bands(
                result["distances"],
                band_names=reachability_config["band_names"],
                quantiles=reachability_config["calibration_quantiles"],
                minimum_width=float(reachability_config["minimum_band_width"]),
            )
        }

    report = {
        "note": (
            "Type 3 reachability diagnostic. Distances are measured in the same "
            "normalized, group-balanced feature space used by Type 3 generation."
        ),
        "anchors": anchors_report,
        "calibrated_radii": calibrated,
    }
    write_manifest(metadata_dir / "type3_reachability.json", report)
    write_reachability_csv(metadata_dir / "type3_reachability.csv", rows)
    write_manifest(metadata_dir / "type3_calibrated_radii.json", calibrated)
    write_human_report(metadata_dir / "type3_reachability_report.md", report)
    return report


def analyze_anchor(
    anchor_name: str,
    anchor_idx: int,
    anchor_vector: Mapping[str, float],
    center_params,
    bounds: Mapping[str, tuple[float, float]],
    scaler: FeatureScaler,
    seed: int,
    reachability_config: Mapping[str, Any],
) -> Dict[str, Any]:
    samples_per_anchor = int(reachability_config["samples_per_anchor"])
    max_attempts = samples_per_anchor * int(reachability_config["max_attempt_factor"])
    generation_attempts = int(reachability_config["max_generation_attempts"])
    scales = [float(value) for value in reachability_config["perturbation_scales"]]
    centers = difficulty_centers(center_params, reachability_config)
    rng = random.Random(seed)

    exact_distance = None
    exact_candidate = None
    rejected_generations = 0
    try:
        exact = generate_valid_instance(
            center_params,
            derive_seed(seed, 1, salt=31_337),
            max_attempts=generation_attempts,
            source_name=f"reachability_center_{anchor_idx}",
        )
        exact_features = extract_features(exact.instance)
        exact_norm = scaler.normalize(exact_features)
        exact_distance = weighted_distance(exact_norm, anchor_vector, scaler.group_weights)
        exact_candidate = {
            "seed": exact.seed,
            "attempts": exact.attempts,
            "distance": exact_distance,
            "parameters": exact.parameters.to_dict(),
        }
        rejected_generations += max(0, exact.attempts - 1)
    except RuntimeError as exc:
        exact_candidate = {"error": str(exc)}

    candidates = []
    attempts = 0
    while len(candidates) < samples_per_anchor and attempts < max_attempts:
        attempts += 1
        scale = scales[(attempts - 1) % len(scales)]
        center = centers[((attempts - 1) // len(scales)) % len(centers)]
        params = center if scale == 0.0 else perturb_parameters(center, bounds, rng, scale)
        try:
            outcome = generate_valid_instance(
                params,
                derive_seed(seed, attempts + 10_000, salt=97_409),
                max_attempts=generation_attempts,
                source_name=f"reachability_{anchor_idx}_{attempts}",
            )
        except RuntimeError:
            rejected_generations += generation_attempts
            continue
        rejected_generations += max(0, outcome.attempts - 1)
        features = extract_features(outcome.instance)
        normalized = scaler.normalize(features)
        distance = weighted_distance(normalized, anchor_vector, scaler.group_weights)
        candidates.append(
            {
                "distance": distance,
                "scale": scale,
                "center_blend_index": ((attempts - 1) // len(scales)) % len(centers),
                "seed": outcome.seed,
                "attempts": outcome.attempts,
                "parameters": outcome.parameters.to_dict(),
                "features": features,
                "normalized_features": normalized,
            }
        )

    if len(candidates) < samples_per_anchor:
        raise RuntimeError(
            f"Reachability for {anchor_name} produced {len(candidates)} valid "
            f"candidates; required {samples_per_anchor}."
        )

    candidates.sort(key=lambda item: item["distance"])
    distances = [float(candidate["distance"]) for candidate in candidates]
    best = candidates[0]
    contributions = feature_contributions(anchor_vector, best["normalized_features"], scaler.group_weights)
    return {
        "anchor_name": anchor_name,
        "exact_center_reconstruction": exact_candidate,
        "exact_center_reconstruction_distance": exact_distance,
        "minimum_reachable_distance": min(distances),
        "p05": percentile(distances, 0.05),
        "p10": percentile(distances, 0.10),
        "p25": percentile(distances, 0.25),
        "median": percentile(distances, 0.50),
        "mean": sum(distances) / len(distances),
        "p75": percentile(distances, 0.75),
        "p90": percentile(distances, 0.90),
        "maximum": max(distances),
        "valid_candidates": len(candidates),
        "sampling_attempts": attempts,
        "rejected_generations": rejected_generations,
        "distances": distances,
        "best_candidate_parameter_set": best["parameters"],
        "best_candidate_feature_vector": best["features"],
        "best_candidate_normalized_feature_vector": best["normalized_features"],
        "feature_differences_to_best": {
            name: best["normalized_features"][name] - anchor_vector[name]
            for name in FEATURE_NAMES
        },
        "feature_distance_contributions": contributions["features"],
        "group_distance_contributions": contributions["groups"],
    }


def difficulty_centers(center_params, config: Mapping[str, Any]):
    target_mix = config.get("viability_difficulty_prior", [0.35, 0.35, 0.20, 0.10])
    blends = [float(value) for value in config.get("viability_blends", [0.0])]
    centers = []
    for blend in blends:
        centers.append(blend_difficulty_mix(center_params, target_mix, blend))
    return centers or [center_params]


def feature_contributions(
    anchor_vector: Mapping[str, float],
    candidate_vector: Mapping[str, float],
    group_weights: Mapping[str, float],
) -> Dict[str, Dict[str, float]]:
    weights = feature_weights(group_weights)
    feature_items = {}
    group_items = {group: 0.0 for group in FEATURE_GROUPS}
    for name in FEATURE_NAMES:
        diff = float(candidate_vector[name]) - float(anchor_vector[name])
        contribution = weights[name] * diff * diff
        feature_items[name] = contribution
        for group, names in FEATURE_GROUPS.items():
            if name in names:
                group_items[group] += contribution
                break
    return {
        "features": dict(sorted(feature_items.items(), key=lambda item: item[1], reverse=True)),
        "groups": dict(sorted(group_items.items(), key=lambda item: item[1], reverse=True)),
    }


def calibrate_bands(
    distances: List[float],
    band_names: List[str],
    quantiles: List[float],
    minimum_width: float,
) -> List[Dict[str, float | str]]:
    sorted_distances = sorted(distances)
    cut_points = [percentile(sorted_distances, q) for q in quantiles]
    bands = []
    for idx, name in enumerate(band_names):
        low = float(cut_points[idx])
        high = float(cut_points[idx + 1])
        if high <= low:
            high = low + minimum_width
        bands.append({"name": str(name), "min": low, "max": high})
    return bands


def flatten_anchor_report(anchor_name: str, report: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "anchor_name": anchor_name,
        "exact_center_reconstruction_distance": report["exact_center_reconstruction_distance"],
        "minimum_reachable_distance": report["minimum_reachable_distance"],
        "p05": report["p05"],
        "p10": report["p10"],
        "p25": report["p25"],
        "median": report["median"],
        "mean": report["mean"],
        "p75": report["p75"],
        "p90": report["p90"],
        "maximum": report["maximum"],
        "valid_candidates": report["valid_candidates"],
        "sampling_attempts": report["sampling_attempts"],
        "rejected_generations": report["rejected_generations"],
        "best_candidate_parameter_set": json.dumps(report["best_candidate_parameter_set"], sort_keys=True),
        "best_candidate_feature_vector": json.dumps(report["best_candidate_feature_vector"], sort_keys=True),
        "feature_differences_to_best": json.dumps(report["feature_differences_to_best"], sort_keys=True),
        "group_distance_contributions": json.dumps(report["group_distance_contributions"], sort_keys=True),
    }


def write_reachability_csv(path: Path, rows: List[Mapping[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def write_human_report(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# Type 3 Reachability Diagnostic",
        "",
        "Distances are measured in the normalized group-balanced Type 3 feature space.",
        "",
    ]
    for anchor_name, anchor_report in report["anchors"].items():
        lines.extend(
            [
                f"## {anchor_name}",
                "",
                f"- exact-center distance: {anchor_report['exact_center_reconstruction_distance']}",
                f"- minimum reachable distance: {anchor_report['minimum_reachable_distance']}",
                f"- P05/P25/median/P75/P90: {anchor_report['p05']}, {anchor_report['p25']}, {anchor_report['median']}, {anchor_report['p75']}, {anchor_report['p90']}",
                f"- valid candidates: {anchor_report['valid_candidates']}",
                f"- rejected generations: {anchor_report['rejected_generations']}",
                f"- group contributions of closest candidate: {json.dumps(anchor_report['group_distance_contributions'], sort_keys=True)}",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def percentile(sorted_values: List[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    values = sorted(sorted_values)
    if len(values) == 1:
        return float(values[0])
    position = max(0.0, min(1.0, q)) * (len(values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(values[lower])
    fraction = position - lower
    return float(values[lower] * (1.0 - fraction) + values[upper] * fraction)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--google-dir", default="google_instances")
    parser.add_argument("--metadata-dir", default="benchmark_metadata")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--type3-config", default="benchmark_generation/configs/type3.yaml")
    parser.add_argument("--samples-per-anchor", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    type3_config = load_config(args.type3_config, DEFAULT_CONFIG)
    reachability_config = dict(DEFAULT_REACHABILITY_CONFIG)
    if args.samples_per_anchor is not None:
        reachability_config["samples_per_anchor"] = args.samples_per_anchor
    analyze_reachability(
        google_dir=args.google_dir,
        metadata_dir=args.metadata_dir,
        seed=args.seed,
        type3_config=type3_config,
        reachability_config=reachability_config,
    )


if __name__ == "__main__":
    main()
