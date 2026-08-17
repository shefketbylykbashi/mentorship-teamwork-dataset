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
    blend_difficulty_mix,
    bounds_from_anchor_estimates,
    estimate_parameters_from_instance,
    perturb_parameters,
)
from benchmark_generation.core.parser import write_instance
from benchmark_generation.core.search import adaptive_parameter_search
from benchmark_generation.core.search import radius_in_bounds
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
    "radius_bands_path": "benchmark_metadata/type3_calibrated_radii.json",
    "allow_uncalibrated_radius_bands": False,
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
        "population_size": 10,
        "iterations": 2,
        "elite_fraction": 0.25,
        "novelty_penalty": 0.20,
        "duplicate_penalty": 10.0,
        "minimum_accepted_distance": 0.015,
        "radius_penalty": 1.5,
        "initial_perturbation_scale": 0.25,
        "random_restart_rate": 0.15,
        "max_generation_attempts": 10,
        "centered_initial_samples": 6,
        "initial_perturbation_scales": [0.0, 0.02, 0.05, 0.10, 0.20],
        "hard_radius": True,
        "hard_radius_violation_penalty": 100.0,
        "radius_tolerance_absolute": 1e-9,
        "radius_tolerance_relative": 1e-9,
        "max_target_retries": 5,
        "max_failed_targets": 500,
        "viability_difficulty_prior": [0.35, 0.35, 0.20, 0.10],
        "viability_blends": [1.0, 0.75, 0.50, 0.25, 0.0],
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
    anchor_radius_bands = resolve_anchor_radius_bands(anchor_names, config)
    targets: List[Dict[str, Any]] = []
    for anchor_idx, allocation in enumerate(anchor_allocations):
        bands = anchor_radius_bands[anchor_names[anchor_idx]]
        band_allocations = _balanced_allocation(allocation, len(bands))
        for band, band_count in zip(bands, band_allocations):
            for _ in range(band_count):
                targets.append(
                    make_single_type3_target(
                        anchor_name=anchor_names[anchor_idx],
                        anchor_idx=anchor_idx,
                        anchor_vector=anchor_vectors[anchor_idx],
                        band=band,
                        rng=rng,
                        weights=weights,
                    )
                )
    rng.shuffle(targets)
    return targets[:count]


def make_single_type3_target(
    anchor_name: str,
    anchor_idx: int,
    anchor_vector: Mapping[str, float],
    band: Mapping[str, float | str],
    rng: random.Random,
    weights: Mapping[str, float],
) -> Dict[str, Any]:
    direction = random_weighted_direction(rng, weights)
    radius = rng.uniform(float(band["min"]), float(band["max"]))
    target_vector = {
        name: anchor_vector[name] + radius * direction[name]
        for name in FEATURE_NAMES
    }
    return {
        "anchor": anchor_name,
        "anchor_index": anchor_idx,
        "radius_band": str(band["name"]),
        "radius_min": float(band["min"]),
        "radius_max": float(band["max"]),
        "desired_radius": radius,
        "target_vector": target_vector,
        "retry_count": 0,
    }


def resolve_anchor_radius_bands(
    anchor_names: List[str],
    config: Mapping[str, Any],
) -> Dict[str, List[Dict[str, float | str]]]:
    allow_uncalibrated = bool(config.get("allow_uncalibrated_radius_bands", False))
    path_value = config.get("radius_bands_path")
    if path_value:
        path = Path(str(path_value))
        if not path.exists():
            if allow_uncalibrated:
                return _uncalibrated_radius_bands(anchor_names, config)
            raise FileNotFoundError(
                f"Type 3 calibrated radius file is required but missing: {path}"
            )
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            anchors = data["anchors"]
            resolved: Dict[str, List[Dict[str, float | str]]] = {}
            for anchor in anchor_names:
                if anchor not in anchors:
                    raise ValueError(f"Calibrated radius file is missing anchor {anchor}")
                bands = anchors[anchor].get("bands")
                if not isinstance(bands, list) or not bands:
                    raise ValueError(f"Calibrated radius file has no bands for {anchor}")
                resolved[anchor] = [_parse_radius_band(anchor, band) for band in bands]
            return resolved
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            if allow_uncalibrated:
                return _uncalibrated_radius_bands(anchor_names, config)
            raise ValueError(
                f"Type 3 calibrated radius file is malformed or incomplete: {path}"
            ) from exc
    if not allow_uncalibrated:
        raise ValueError(
            "Type 3 final generation requires radius_bands_path. Set "
            "allow_uncalibrated_radius_bands=true only for development tests."
        )
    return _uncalibrated_radius_bands(anchor_names, config)


def _parse_radius_band(anchor: str, band: Mapping[str, Any]) -> Dict[str, float | str]:
    name = str(band["name"])
    radius_min = float(band["min"])
    radius_max = float(band["max"])
    if not name:
        raise ValueError(f"Empty calibrated radius band name for {anchor}")
    if not math.isfinite(radius_min) or not math.isfinite(radius_max):
        raise ValueError(f"Non-finite calibrated radius band for {anchor}:{name}")
    if radius_max <= radius_min:
        raise ValueError(f"Invalid calibrated radius band for {anchor}:{name}")
    return {"name": name, "min": radius_min, "max": radius_max}


def _uncalibrated_radius_bands(
    anchor_names: List[str],
    config: Mapping[str, Any],
) -> Dict[str, List[Dict[str, float | str]]]:
    common = [
        _parse_radius_band("__development_uncalibrated__", band)
        for band in config["radius_bands"]
    ]
    return {anchor: list(common) for anchor in anchor_names}


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
    anchor_radius_bands = resolve_anchor_radius_bands(anchor_names, config)
    target_weights = feature_weights(config.get("distance_group_weights"))
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
    radius_errors: List[float] = []
    failed_targets: List[Dict[str, Any]] = []
    target_queue = list(targets)
    target_cursor = 0
    max_failed_targets = int(config["search"].get("max_failed_targets", 500))
    max_target_retries = int(config["search"].get("max_target_retries", 5))

    while len(rows) < count and target_cursor < len(target_queue):
        target = target_queue[target_cursor]
        target_cursor += 1
        idx = len(rows) + 1
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
        initial_population = _type3_initial_population(
            center=anchor_params[anchor_idx],
            bounds=per_anchor_bounds[anchor_idx],
            config=config["search"],
            seed=derive_seed(seed, idx, salt=314_159),
        )
        try:
            result = adaptive_parameter_search(
                target_vector=target["target_vector"],
                scaler=scaler,
                bounds=per_anchor_bounds[anchor_idx],
                config=config["search"],
                seed=derive_seed(seed, target_cursor),
                accepted_vectors=accepted_vectors,
                existing_hashes=existing_hashes,
                anchor_vector=anchor_vectors[anchor_idx],
                radius_min=target["radius_min"],
                radius_max=target["radius_max"],
                source_name=filename,
                initial_population=initial_population,
                hard_radius=bool(config["search"].get("hard_radius", True)),
                radius_tolerance_absolute=float(
                    config["search"].get("radius_tolerance_absolute", 1e-9)
                ),
                radius_tolerance_relative=float(
                    config["search"].get("radius_tolerance_relative", 1e-9)
                ),
            )
        except RuntimeError as exc:
            failure = {
                "target_id": target_cursor,
                "anchor": target["anchor"],
                "radius_band": target["radius_band"],
                "desired_radius": target["desired_radius"],
                "radius_min": target["radius_min"],
                "radius_max": target["radius_max"],
                "failure_reason": str(exc),
                "retry_count": int(target.get("retry_count", 0)),
            }
            failed_targets.append(failure)
            if len(failed_targets) > max_failed_targets:
                raise RuntimeError(
                    f"Exceeded max_failed_targets={max_failed_targets} before "
                    f"accepting {count} Type 3 instances"
                ) from exc
            retry_count = int(target.get("retry_count", 0))
            if retry_count < max_target_retries:
                band = next(
                    band
                    for band in anchor_radius_bands[str(target["anchor"])]
                    if str(band["name"]) == str(target["radius_band"])
                )
                replacement_rng = random.Random(
                    derive_seed(seed, target_cursor + retry_count + 1, salt=1_618_033)
                )
                replacement = make_single_type3_target(
                    anchor_name=str(target["anchor"]),
                    anchor_idx=anchor_idx,
                    anchor_vector=anchor_vectors[anchor_idx],
                    band=band,
                    rng=replacement_rng,
                    weights=target_weights,
                )
                replacement["retry_count"] = retry_count + 1
                target_queue.append(replacement)
            logging.warning(
                "failed Type3 target anchor=%s band=%s retry=%s reason=%s",
                target["anchor"],
                target["radius_band"],
                target.get("retry_count", 0),
                exc,
            )
            continue
        candidate = result.candidate
        actual_radius = weighted_distance(
            candidate.normalized_features,
            anchor_vectors[anchor_idx],
            scaler.group_weights,
        )
        radius_pass = radius_in_bounds(
            actual_radius,
            float(target["radius_min"]),
            float(target["radius_max"]),
            radius_tolerance_absolute=float(
                config["search"].get("radius_tolerance_absolute", 1e-9)
            ),
            radius_tolerance_relative=float(
                config["search"].get("radius_tolerance_relative", 1e-9)
            ),
        )
        if not radius_pass:
            raise RuntimeError(
                "Internal Type 3 error: hard-radius search returned an "
                f"out-of-band candidate radius={actual_radius}, "
                f"band=[{target['radius_min']}, {target['radius_max']}]"
            )
        existing_hashes.add(candidate.structural_hash)
        accepted_vectors.append(candidate.normalized_features)
        generated_vectors.append(candidate.normalized_features)
        anchor_distances.append(actual_radius)
        target_distances.append(candidate.target_distance)
        radius_error = actual_radius - float(target["desired_radius"])
        radius_errors.append(radius_error)

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
                "generation_failures": result.generation_failures,
            },
            extra={
                "type3": {
                    "anchor": target["anchor"],
                    "anchor_index": anchor_idx,
                    "radius_band": target["radius_band"],
                    "radius_min": target["radius_min"],
                    "radius_max": target["radius_max"],
                    "desired_radius": target["desired_radius"],
                    "actual_distance_to_anchor": actual_radius,
                    "radius_error": radius_error,
                    "absolute_radius_error": abs(radius_error),
                    "distance_to_target": candidate.target_distance,
                    "nearest_existing_distance": candidate.nearest_existing_distance,
                    "radius_constraint_pass": radius_pass,
                    "structural_validation_pass": validation.ok,
                    "structural_hash": candidate.structural_hash,
                    "search_log": result.log,
                    "retry_count": int(target.get("retry_count", 0)),
                    "fallback_retry_mode": "replacement_target_same_band"
                    if int(target.get("retry_count", 0)) > 0
                    else "",
                },
                "target": target["target_vector"],
                "normalized": candidate.normalized_features,
            },
        )
        rows.append(row)
        logging.info(
            "accepted type3_%03d distance_to_target=%.4f radius=%.4f",
            idx,
            candidate.target_distance,
            actual_radius,
        )

    if len(rows) != count:
        raise RuntimeError(
            f"Type 3 accepted {len(rows)} instances, expected {count}; "
            f"failed_targets={len(failed_targets)}"
        )

    quality_gate = _type3_quality_gate(rows, anchor_names, anchor_radius_bands, count)
    report = family_statistical_report(
        rows,
        generated_vectors,
        group_weights=scaler.group_weights,
        extra={
            "failed_target_attempts": len(failed_targets),
            "failed_targets": failed_targets,
            "distance_to_target": summarize_numeric(target_distances),
            "distance_to_anchor": summarize_numeric(anchor_distances),
            "absolute_radius_error": summarize_numeric([abs(value) for value in radius_errors]),
            "anchor_counts": {
                anchor: sum(1 for row in rows if row.get("type3.anchor") == anchor)
                for anchor in anchor_names
            },
            "radius_band_counts": {
                band_name: sum(
                    1 for row in rows if row.get("type3.radius_band") == band_name
                )
                for band_name in sorted({str(row.get("type3.radius_band")) for row in rows})
            },
            "within_anchor_diversity": _within_anchor_diversity(rows, generated_vectors, scaler.group_weights),
            "radius_band_actuals": _radius_band_actuals(rows),
            "quality_gate": quality_gate,
        },
    )

    if not quality_gate["passed"]:
        raise RuntimeError(
            "Type 3 final quality gate failed: "
            + json.dumps(quality_gate, sort_keys=True)
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


def _radius_band_actuals(rows: List[Mapping[str, Any]]) -> Dict[str, Dict[str, float]]:
    by_band: Dict[str, List[float]] = {}
    outside_by_band: Dict[str, int] = {}
    for row in rows:
        band = str(row.get("type3.radius_band"))
        actual = float(row.get("type3.actual_distance_to_anchor", 0.0))
        radius_min = float(row.get("type3.radius_min", 0.0))
        radius_max = float(row.get("type3.radius_max", 0.0))
        by_band.setdefault(band, []).append(actual)
        if not (radius_min <= actual <= radius_max):
            outside_by_band[band] = outside_by_band.get(band, 0) + 1
    report = {}
    for band, values in by_band.items():
        sorted_values = sorted(values)
        report[band] = {
            **summarize_numeric(values),
            "p10": _percentile(sorted_values, 0.10),
            "p90": _percentile(sorted_values, 0.90),
            "inside_band": len(values) - outside_by_band.get(band, 0),
            "outside_band": outside_by_band.get(band, 0),
        }
    return report


def _type3_quality_gate(
    rows: List[Mapping[str, Any]],
    anchor_names: List[str],
    anchor_radius_bands: Mapping[str, List[Mapping[str, float | str]]],
    expected_count: int,
) -> Dict[str, Any]:
    required_columns = {
        "filename",
        "family",
        "family_instance_id",
        "seed",
        "generator_version",
        "type3.anchor",
        "type3.anchor_index",
        "type3.radius_band",
        "type3.radius_min",
        "type3.radius_max",
        "type3.desired_radius",
        "type3.actual_distance_to_anchor",
        "type3.radius_error",
        "type3.absolute_radius_error",
        "type3.distance_to_target",
        "type3.nearest_existing_distance",
        "type3.structural_hash",
        "type3.radius_constraint_pass",
        "type3.structural_validation_pass",
        "validation.generation_attempts",
        "validation.search_evaluations",
    }
    structural_hashes = [str(row.get("type3.structural_hash")) for row in rows]
    represented_anchors = {str(row.get("type3.anchor")) for row in rows}
    expected_bands_by_anchor = {
        anchor: {str(band["name"]) for band in bands}
        for anchor, bands in anchor_radius_bands.items()
    }
    represented_bands_by_anchor = {
        anchor: {
            str(row.get("type3.radius_band"))
            for row in rows
            if str(row.get("type3.anchor")) == anchor
        }
        for anchor in anchor_names
    }
    missing_bands_by_anchor = {
        anchor: sorted(expected_bands_by_anchor.get(anchor, set()) - represented_bands_by_anchor.get(anchor, set()))
        for anchor in anchor_names
    }
    unexpected_bands = []
    for row in rows:
        anchor = str(row.get("type3.anchor"))
        band = str(row.get("type3.radius_band"))
        if band not in expected_bands_by_anchor.get(anchor, set()):
            unexpected_bands.append({"anchor": anchor, "band": band, "filename": row.get("filename")})
    missing_metadata_rows = [
        str(row.get("filename", f"row_{idx}"))
        for idx, row in enumerate(rows, start=1)
        if any(column not in row or row.get(column) in ("", None) for column in required_columns)
    ]
    gate = {
        "accepted_count": len(rows),
        "expected_count": expected_count,
        "accepted_count_exact": len(rows) == expected_count,
        "unique_structural_hashes": len(set(structural_hashes)),
        "unique_structural_hash_count_exact": len(set(structural_hashes)) == expected_count,
        "all_structurally_valid": all(_truthy(row.get("type3.structural_validation_pass")) for row in rows),
        "all_radius_constraints_pass": all(_truthy(row.get("type3.radius_constraint_pass")) for row in rows),
        "all_metadata_rows_present": len(rows) == expected_count and not missing_metadata_rows,
        "missing_metadata_rows": missing_metadata_rows,
        "all_five_anchors_represented": set(anchor_names).issubset(represented_anchors),
        "all_expected_anchor_bands_represented": all(
            not missing for missing in missing_bands_by_anchor.values()
        ),
        "missing_bands_by_anchor": missing_bands_by_anchor,
        "unexpected_anchor_bands": unexpected_bands,
        "all_rows_use_expected_anchor_band": not unexpected_bands,
        "duplicate_count": len(structural_hashes) - len(set(structural_hashes)),
    }
    gate["passed"] = all(
        [
            gate["accepted_count_exact"],
            gate["unique_structural_hash_count_exact"],
            gate["all_structurally_valid"],
            gate["all_radius_constraints_pass"],
            gate["all_metadata_rows_present"],
            gate["all_five_anchors_represented"],
            gate["all_expected_anchor_bands_represented"],
            gate["all_rows_use_expected_anchor_band"],
            gate["duplicate_count"] == 0,
        ]
    )
    return gate


def _truthy(value: Any) -> bool:
    return value is True or str(value).lower() == "true"


def _type3_initial_population(
    center,
    bounds: Mapping[str, tuple[float, float]],
    config: Mapping[str, Any],
    seed: int,
):
    rng = random.Random(seed)
    count = max(1, int(config.get("centered_initial_samples", 1)))
    scales = [float(value) for value in config.get("initial_perturbation_scales", [])]
    if not scales:
        scales = [float(config.get("initial_perturbation_scale", 0.25))]
    positive_scales = [scale for scale in scales if scale > 0.0]
    population = [center]

    for scale in positive_scales:
        if len(population) >= count:
            return population
        population.append(perturb_parameters(center, bounds, rng, scale=scale))

    adjusted_centers = [
        adjusted
        for adjusted in _difficulty_centers(center, config)
        if adjusted.to_dict() != center.to_dict()
    ]
    for adjusted in adjusted_centers:
        if len(population) >= count:
            return population
        population.append(adjusted)

    center_cycle = adjusted_centers or [center]
    scale_cycle = positive_scales or [0.0]
    cursor = 0
    while len(population) < count:
        base = center_cycle[cursor % len(center_cycle)]
        scale = scale_cycle[cursor % len(scale_cycle)]
        cursor += 1
        population.append(perturb_parameters(base, bounds, rng, scale=scale))
    return population


def _difficulty_centers(center, config: Mapping[str, Any]):
    target_mix = config.get("viability_difficulty_prior", [0.35, 0.35, 0.20, 0.10])
    blends = [float(value) for value in config.get("viability_blends", [0.0])]
    centers = []
    for blend in blends:
        centers.append(blend_difficulty_mix(center, target_mix, blend))
    return centers or [center]


def _percentile(sorted_values: List[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = max(0.0, min(1.0, q)) * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


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
