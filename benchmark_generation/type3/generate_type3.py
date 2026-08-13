"""CLI for Type 3 neighborhood instances around Google anchors."""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping

from benchmark_generation.core.anchors import load_google_anchors
from benchmark_generation.core.config import load_config
from benchmark_generation.core.distance import FEATURE_NAMES, FeatureScaler, feature_weights, weighted_distance
from benchmark_generation.core.duplicates import existing_generated_hashes
from benchmark_generation.core.features import extract_features
from benchmark_generation.core.google_metadata import write_google_anchor_metadata
from benchmark_generation.core.generator import derive_seed
from benchmark_generation.core.metadata import (
    build_metadata_row,
    config_hash,
    family_statistical_report,
    summarize_numeric,
    update_master_manifest,
    write_manifest,
    write_metadata,
)
from benchmark_generation.core.parameter_sampling import (
    bounds_from_anchor_estimates,
    estimate_parameters_from_instance,
)
from benchmark_generation.core.parser import write_instance
from benchmark_generation.core.search import adaptive_parameter_search
from benchmark_generation.core.validator import structural_hash, validate_file


DEFAULT_CONFIG: Dict[str, Any] = {
    "family": "type3",
    "count": 100,
    "root_seed": 2026,
    "fillable_fraction": 0.30,
    "distance_group_weights": {
        "size": 1,
        "supply_demand": 1,
        "levels": 1,
        "mentorship": 1,
        "time": 1,
        "scores": 1,
    },
    "radius_bands": [
        {"name": "small", "min": 0.03, "max": 0.08},
        {"name": "medium", "min": 0.08, "max": 0.18},
        {"name": "outer", "min": 0.18, "max": 0.35},
        {"name": "wide", "min": 0.35, "max": 0.55},
    ],
    "parameter_search": {
        "anchor_margin": 0.50,
        "max_profiles": 4,
        "hard_bounds": {
            "n_skills": [5, 2000],
            "skill_skew": [0.0, 2.0],
            "n_contributors": [5, 12000],
            "skills_per_contributor_min": [1, 20],
            "skills_per_contributor_max": [1, 20],
            "level_mean": [1.0, 10.0],
            "level_sd": [0.2, 4.0],
            "level_max": [10, 10],
            "n_profiles": [1, 4],
            "n_projects": [5, 22000],
            "duration_min": [1, 100000],
            "duration_max": [1, 100000],
            "roles_min": [1, 50],
            "roles_max": [1, 50],
            "score_base": [0.1, 10.0],
            "score_perturbation": [1.0, 5.0],
            "tightness_min": [0.0, 10.0],
            "tightness_max": [0.0, 10.0],
            "role_skill_alignment": [0.0, 1.0],
            "supply_quantile": [0.5, 0.9],
            "role_level_max": [100, 100],
            "future_offset_min": [2, 4],
            "future_offset_max": [2, 8],
            "profile_bias": [1.0, 6.0],
        },
        "fillable_fraction": 0.30,
    },
    "search": {
        "population_size": 48,
        "iterations": 4,
        "elite_fraction": 0.25,
        "novelty_penalty": 0.20,
        "duplicate_penalty": 10.0,
        "minimum_accepted_distance": 0.015,
        "radius_penalty": 1.5,
        "initial_perturbation_scale": 0.30,
        "random_restart_rate": 0.20,
        "max_generation_attempts": 30,
    },
}


def make_type3_targets(
    anchor_names: List[str],
    anchor_vectors: List[Mapping[str, float]],
    count: int,
    config: Dict[str, Any],
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    anchor_allocations = _balanced_allocation(count, len(anchor_vectors))
    weights = feature_weights(config.get("distance_group_weights"))
    targets: List[Dict[str, Any]] = []
    for anchor_idx, allocation in enumerate(anchor_allocations):
        band_allocations = _balanced_allocation(allocation, len(config["radius_bands"]))
        for band, band_count in zip(config["radius_bands"], band_allocations):
            for _ in range(band_count):
                direction = random_weighted_direction(rng, weights)
                radius = rng.uniform(float(band["min"]), float(band["max"]))
                target_vector = {
                    name: anchor_vectors[anchor_idx][name] + radius * direction[name]
                    for name in FEATURE_NAMES
                }
                targets.append(
                    {
                        "anchor": anchor_names[anchor_idx],
                        "anchor_index": anchor_idx,
                        "radius_band": band["name"],
                        "radius_min": float(band["min"]),
                        "radius_max": float(band["max"]),
                        "desired_radius": radius,
                        "target_vector": target_vector,
                    }
                )
    rng.shuffle(targets)
    return targets[:count]


def random_weighted_direction(
    rng: random.Random, weights: Mapping[str, float]
) -> Dict[str, float]:
    raw = {name: rng.gauss(0.0, 1.0) for name in FEATURE_NAMES}
    norm = math.sqrt(sum(weights[name] * raw[name] * raw[name] for name in FEATURE_NAMES))
    if norm <= 1e-12:
        raw[FEATURE_NAMES[0]] = 1.0
        norm = math.sqrt(weights[FEATURE_NAMES[0]])
    return {name: raw[name] / norm for name in FEATURE_NAMES}


def _balanced_allocation(count: int, n_bins: int) -> List[int]:
    base = count // n_bins
    remainder = count % n_bins
    return [base + (1 if idx < remainder else 0) for idx in range(n_bins)]


def generate_type3(
    google_dir: str | Path,
    output_dir: str | Path,
    metadata_dir: str | Path,
    count: int,
    seed: int,
    config: Dict[str, Any],
    overwrite: bool = False,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    google_dir = Path(google_dir)
    output_dir = Path(output_dir)
    metadata_dir = Path(metadata_dir)
    if not dry_run:
        write_google_anchor_metadata(
            google_dir,
            metadata_dir,
            group_weights=config.get("distance_group_weights"),
        )
    anchors = load_google_anchors(google_dir)
    anchor_names = [str(anchor.source_name) for anchor in anchors]
    anchor_features = [extract_features(anchor) for anchor in anchors]
    scaler = FeatureScaler.fit(anchor_features, group_weights=config.get("distance_group_weights"))
    anchor_vectors = [scaler.normalize(features) for features in anchor_features]
    anchor_params = [estimate_parameters_from_instance(anchor) for anchor in anchors]
    per_anchor_bounds = [
        bounds_from_anchor_estimates([params], config["parameter_search"])
        for params in anchor_params
    ]
    targets = make_type3_targets(
        anchor_names,
        anchor_vectors,
        count,
        config,
        seed=seed * 130_363 + 79,
    )

    existing_hashes = {structural_hash(anchor) for anchor in anchors}
    existing_hashes.update(existing_generated_hashes(output_dir.parent))
    accepted_vectors: List[Mapping[str, float]] = list(anchor_vectors)
    generated_vectors: List[Mapping[str, float]] = []
    rows: List[Dict[str, Any]] = []
    anchor_distances: List[float] = []
    target_distances: List[float] = []

    for idx, target in enumerate(targets, start=1):
        filename = f"type3_{idx:03d}.txt"
        anchor_idx = int(target["anchor_index"])
        logging.info(
            "Type3 target %03d/%03d anchor=%s band=%s radius=%.3f",
            idx,
            count,
            target["anchor"],
            target["radius_band"],
            target["desired_radius"],
        )
        result = adaptive_parameter_search(
            target_vector=target["target_vector"],
            scaler=scaler,
            bounds=per_anchor_bounds[anchor_idx],
            config=config["search"],
            seed=derive_seed(seed, idx),
            accepted_vectors=accepted_vectors,
            existing_hashes=existing_hashes,
            anchor_vector=anchor_vectors[anchor_idx],
            radius_min=target["radius_min"],
            radius_max=target["radius_max"],
            source_name=filename,
        )
        candidate = result.candidate
        existing_hashes.add(candidate.structural_hash)
        accepted_vectors.append(candidate.normalized_features)
        generated_vectors.append(candidate.normalized_features)
        actual_radius = weighted_distance(
            candidate.normalized_features,
            anchor_vectors[anchor_idx],
            scaler.group_weights,
        )
        anchor_distances.append(actual_radius)
        target_distances.append(candidate.target_distance)

        if not dry_run:
            path = output_dir / filename
            write_instance(candidate.outcome.instance, path, overwrite=overwrite)
            validation = validate_file(
                path,
                fillable_fraction=candidate.outcome.parameters.fillable_fraction,
            )
            if not validation.ok:
                raise RuntimeError(f"Written instance failed validation: {path}: {validation.issues}")
        else:
            validation = candidate.outcome.validation

        row = build_metadata_row(
            filename=filename,
            family="type3",
            family_instance_id=idx,
            seed=candidate.outcome.seed,
            generator_version=candidate.outcome.generator_version,
            parameters=candidate.outcome.parameters.to_dict(),
            features=candidate.features,
            validation={
                "ok": validation.ok,
                "issues": validation.issues,
                "initial_fillable_fraction": validation.metrics.get("initial_fillable_fraction", ""),
                "generation_attempts": candidate.outcome.attempts,
                "search_evaluations": result.evaluations,
            },
            extra={
                "type3": {
                    "anchor": target["anchor"],
                    "radius_band": target["radius_band"],
                    "radius_min": target["radius_min"],
                    "radius_max": target["radius_max"],
                    "desired_radius": target["desired_radius"],
                    "actual_distance_to_anchor": actual_radius,
                    "distance_to_target": candidate.target_distance,
                    "nearest_existing_distance": candidate.nearest_existing_distance,
                    "structural_hash": candidate.structural_hash,
                    "search_log": result.log,
                },
                "target": target["target_vector"],
            },
        )
        rows.append(row)
        logging.info(
            "accepted type3_%03d distance_to_target=%.4f radius=%.4f",
            idx,
            candidate.target_distance,
            actual_radius,
        )

    report = family_statistical_report(
        rows,
        generated_vectors,
        group_weights=scaler.group_weights,
        extra={
            "distance_to_target": summarize_numeric(target_distances),
            "distance_to_anchor": summarize_numeric(anchor_distances),
            "anchor_counts": {
                anchor: sum(1 for row in rows if row.get("type3.anchor") == anchor)
                for anchor in anchor_names
            },
            "radius_band_counts": {
                band["name"]: sum(
                    1 for row in rows if row.get("type3.radius_band") == band["name"]
                )
                for band in config["radius_bands"]
            },
            "within_anchor_diversity": _within_anchor_diversity(rows, generated_vectors, scaler.group_weights),
        },
    )

    if not dry_run:
        write_metadata(
            rows,
            metadata_dir / "type3_metadata.csv",
            metadata_dir / "type3_metadata.jsonl",
        )
        write_manifest(metadata_dir / "type3_statistical_report.json", report)
        update_master_manifest(
            metadata_dir / "benchmark_manifest.json",
            "type3",
            {
                "count": len(rows),
                "seed": seed,
                "config_hash": config_hash(config),
                "output_dir": str(output_dir),
                "metadata_csv": str(metadata_dir / "type3_metadata.csv"),
                "statistical_report": str(metadata_dir / "type3_statistical_report.json"),
            },
        )
        (metadata_dir / "type3_config_used.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return rows


def _within_anchor_diversity(
    rows: List[Mapping[str, Any]],
    vectors: List[Mapping[str, float]],
    group_weights: Mapping[str, float],
) -> Dict[str, Dict[str, float]]:
    by_anchor: Dict[str, List[Mapping[str, float]]] = {}
    for row, vector in zip(rows, vectors):
        by_anchor.setdefault(str(row.get("type3.anchor")), []).append(vector)
    result = {}
    for anchor, anchor_vectors in by_anchor.items():
        distances = []
        for i in range(len(anchor_vectors)):
            for j in range(i + 1, len(anchor_vectors)):
                distances.append(weighted_distance(anchor_vectors[i], anchor_vectors[j], group_weights))
        result[anchor] = summarize_numeric(distances)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--google-dir", default="google_instances")
    parser.add_argument("--output-dir", default="generated_instances/type3")
    parser.add_argument("--metadata-dir", default="benchmark_metadata")
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--config", default="benchmark_generation/configs/type3.yaml")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="Reserved for future parallel search.")
    parser.add_argument("--max-attempts", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config = load_config(args.config, DEFAULT_CONFIG)
    count = args.count if args.count is not None else int(config["count"])
    seed = args.seed if args.seed is not None else int(config["root_seed"])
    if args.max_attempts is not None:
        config["search"]["max_generation_attempts"] = args.max_attempts
    generate_type3(
        google_dir=args.google_dir,
        output_dir=args.output_dir,
        metadata_dir=args.metadata_dir,
        count=count,
        seed=seed,
        config=config,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
