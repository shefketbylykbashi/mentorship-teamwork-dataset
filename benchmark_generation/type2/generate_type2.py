"""CLI for Type 2 gap-filling instances between Google anchors."""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

from benchmark_generation.core.anchors import load_google_anchors
from benchmark_generation.core.config import load_config
from benchmark_generation.core.distance import FEATURE_NAMES, FeatureScaler, weighted_distance
from benchmark_generation.core.duplicates import existing_generated_hashes
from benchmark_generation.core.features import extract_features
from benchmark_generation.core.generator import derive_seed
from benchmark_generation.core.google_metadata import write_google_anchor_metadata
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
    "family": "type2",
    "count": 100,
    "root_seed": 2026,
    "alpha_range": [0.2, 0.8],
    "fillable_fraction": 0.30,
    "distance_group_weights": {
        "size": 1,
        "supply_demand": 1,
        "levels": 1,
        "mentorship": 1,
        "time": 1,
        "scores": 1,
    },
    "parameter_search": {
        "anchor_margin": 0.35,
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
        "minimum_accepted_distance": 0.02,
        "initial_perturbation_scale": 0.35,
        "random_restart_rate": 0.20,
        "max_generation_attempts": 30,
    },
}


def make_type2_targets(
    anchor_names: List[str],
    anchor_vectors: List[Mapping[str, float]],
    count: int,
    group_weights: Mapping[str, float],
    seed: int,
    alpha_range: Tuple[float, float],
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    pairs = list(itertools.combinations(range(len(anchor_vectors)), 2))
    distances = [
        weighted_distance(anchor_vectors[i], anchor_vectors[j], group_weights)
        for i, j in pairs
    ]
    allocations = _allocate_by_weight(count, distances)
    targets: List[Dict[str, Any]] = []
    for pair_idx, ((i, j), allocation) in enumerate(zip(pairs, allocations)):
        low, high = alpha_range
        for local_idx in range(allocation):
            stratum_low = low + (high - low) * local_idx / allocation
            stratum_high = low + (high - low) * (local_idx + 1) / allocation
            alpha = rng.uniform(stratum_low, stratum_high)
            vector = {
                name: (1.0 - alpha) * anchor_vectors[i][name] + alpha * anchor_vectors[j][name]
                for name in FEATURE_NAMES
            }
            targets.append(
                {
                    "pair_index": pair_idx,
                    "anchor_a": anchor_names[i],
                    "anchor_b": anchor_names[j],
                    "anchor_a_index": i,
                    "anchor_b_index": j,
                    "alpha": alpha,
                    "target_vector": vector,
                    "pair_distance": distances[pair_idx],
                }
            )
    rng.shuffle(targets)
    return targets[:count]


def _allocate_by_weight(count: int, weights: List[float]) -> List[int]:
    if not weights:
        return []
    if count < len(weights):
        allocations = [0] * len(weights)
        for idx in sorted(range(len(weights)), key=lambda i: weights[i], reverse=True)[:count]:
            allocations[idx] = 1
        return allocations
    total = sum(weights)
    if total <= 0:
        base = count // len(weights)
        allocations = [base] * len(weights)
        for idx in range(count - sum(allocations)):
            allocations[idx] += 1
        return allocations
    raw = [count * weight / total for weight in weights]
    allocations = [max(1, int(value)) for value in raw]
    while sum(allocations) > count:
        idx = max(range(len(allocations)), key=lambda i: allocations[i] - raw[i])
        if allocations[idx] > 1:
            allocations[idx] -= 1
        else:
            break
    while sum(allocations) < count:
        idx = max(range(len(allocations)), key=lambda i: raw[i] - allocations[i])
        allocations[idx] += 1
    return allocations


def generate_type2(
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
    bounds = bounds_from_anchor_estimates(anchor_params, config["parameter_search"])
    targets = make_type2_targets(
        anchor_names,
        anchor_vectors,
        count,
        scaler.group_weights,
        seed=derive_target_seed(seed),
        alpha_range=tuple(config["alpha_range"]),
    )

    existing_hashes = {structural_hash(anchor) for anchor in anchors}
    existing_hashes.update(existing_generated_hashes(output_dir.parent))
    accepted_vectors: List[Mapping[str, float]] = list(anchor_vectors)
    generated_vectors: List[Mapping[str, float]] = []
    rows: List[Dict[str, Any]] = []
    distance_to_targets: List[float] = []
    nearest_google_distances: List[float] = []

    for idx, target in enumerate(targets, start=1):
        filename = f"type2_{idx:03d}.txt"
        logging.info(
            "Type2 target %03d/%03d pair=%s,%s alpha=%.3f",
            idx,
            count,
            target["anchor_a"],
            target["anchor_b"],
            target["alpha"],
        )
        result = adaptive_parameter_search(
            target_vector=target["target_vector"],
            scaler=scaler,
            bounds=bounds,
            config=config["search"],
            seed=derive_seed(seed, idx),
            accepted_vectors=accepted_vectors,
            existing_hashes=existing_hashes,
            source_name=filename,
        )
        candidate = result.candidate
        existing_hashes.add(candidate.structural_hash)
        accepted_vectors.append(candidate.normalized_features)
        generated_vectors.append(candidate.normalized_features)
        distance_to_targets.append(candidate.target_distance)
        nearest_google = min(
            weighted_distance(candidate.normalized_features, anchor, scaler.group_weights)
            for anchor in anchor_vectors
        )
        nearest_google_distances.append(nearest_google)

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
            family="type2",
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
                "type2": {
                    "anchor_a": target["anchor_a"],
                    "anchor_b": target["anchor_b"],
                    "alpha": target["alpha"],
                    "pair_distance": target["pair_distance"],
                    "distance_to_target": candidate.target_distance,
                    "nearest_existing_distance": candidate.nearest_existing_distance,
                    "distance_to_nearest_google_anchor": nearest_google,
                    "structural_hash": candidate.structural_hash,
                    "search_log": result.log,
                },
                "target": target["target_vector"],
            },
        )
        rows.append(row)
        logging.info(
            "accepted type2_%03d distance_to_target=%.4f nearest_google=%.4f",
            idx,
            candidate.target_distance,
            nearest_google,
        )

    report = family_statistical_report(
        rows,
        generated_vectors,
        group_weights=scaler.group_weights,
        extra={
            "distance_to_target": summarize_numeric(distance_to_targets),
            "distance_to_nearest_google_anchor": summarize_numeric(nearest_google_distances),
            "pair_coverage": {
                f"{a}|{b}": sum(
                    1
                    for row in rows
                    if row.get("type2.anchor_a") == a and row.get("type2.anchor_b") == b
                )
                for a, b in itertools.combinations(anchor_names, 2)
            },
        },
    )

    if not dry_run:
        write_metadata(
            rows,
            metadata_dir / "type2_metadata.csv",
            metadata_dir / "type2_metadata.jsonl",
        )
        write_manifest(metadata_dir / "type2_statistical_report.json", report)
        update_master_manifest(
            metadata_dir / "benchmark_manifest.json",
            "type2",
            {
                "count": len(rows),
                "seed": seed,
                "config_hash": config_hash(config),
                "output_dir": str(output_dir),
                "metadata_csv": str(metadata_dir / "type2_metadata.csv"),
                "statistical_report": str(metadata_dir / "type2_statistical_report.json"),
            },
        )
        (metadata_dir / "type2_config_used.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return rows


def derive_target_seed(seed: int) -> int:
    return seed * 104_729 + 53


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--google-dir", default="google_instances")
    parser.add_argument("--output-dir", default="generated_instances/type2")
    parser.add_argument("--metadata-dir", default="benchmark_metadata")
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--config", default="benchmark_generation/configs/type2.yaml")
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
    generate_type2(
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
