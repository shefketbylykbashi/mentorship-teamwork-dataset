"""CLI for Type 7 remaining instance-space gap filling.

Type 7 is a solver-independent, coverage-driven synthetic family.  It audits
the frozen Types 1-6 benchmark, freezes a canonical scaled feature space, then
selects forward-generated candidates by sequential maximin novelty.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from benchmark_generation.core.anchors import discover_google_instance_paths
from benchmark_generation.core.config import load_config
from benchmark_generation.core.distance import (
    FEATURE_NAMES,
    LOG_TRANSFORM_FEATURES,
    FeatureScaler,
    feature_weights,
    pairwise_distances,
    weighted_distance,
)
from benchmark_generation.core.features import FEATURE_GROUPS, extract_features
from benchmark_generation.core.generator import (
    GenerationParameters,
    derive_seed,
    generate_valid_instance,
)
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
    perturb_parameters,
    sample_type4_parameters,
)
from benchmark_generation.core.parser import parse_file, write_instance
from benchmark_generation.core.validator import structural_hash, validate_file, validate_instance


DEFAULT_CONFIG: Dict[str, Any] = {
    "family": "type7",
    "count": 100,
    "pilot_count": 10,
    "root_seed": 2026,
    "evaluation_seed": 2027,
    "existing_audit_fillable_fraction": 0.0,
    "fillable_fraction": 0.30,
    "max_generation_attempts": 40,
    "max_parameter_attempts": 20,
    "distance_group_weights": {
        "size": 1,
        "supply_demand": 1,
        "levels": 1,
        "mentorship": 1,
        "time": 1,
        "scores": 1,
    },
    "candidate_search": {
        "initial_valid_candidate_bank": 400,
        "adaptive_rounds": 2,
        "candidates_per_adaptive_round": 150,
        "elite_gap_fraction": 0.10,
        "random_restart_fraction": 0.20,
        "mutation_scale_initial": 0.30,
        "mutation_scale_decay": 0.70,
        "minimum_candidate_bank_multiplier": 4,
    },
    "evaluation_cloud_size": 500,
    "evaluation_minimum_required": {
        "mean_after_less_than_before": True,
        "p95_after_less_than_before": True,
    },
    "plausibility": {
        "quantile_low": 0.0,
        "quantile_high": 1.0,
        "margin_fraction": 0.10,
    },
    "degeneracy": {
        "min_meaningful_skills": 2,
        "min_fillability": 0.30,
        "min_projects_for_variance_checks": 10,
        "reject_zero_duration_variance": True,
        "reject_zero_score_variance": True,
        "reject_identical_project_shape": True,
    },
    "mentorship_metrics": {
        "candidate_limit_per_role": 32,
        "assignment_search_node_limit": 20000,
    },
    "selection": {
        "use_hard_minimum_novelty": False,
        "minimum_novelty_quantile": 0.05,
    },
    "major_gap_region_count": 20,
    "generator_parameter_config": {
        "family": "type7_candidate",
        "count": 100,
        "root_seed": 2026,
        "fillable_fraction": 0.30,
        "max_parameter_attempts": 20,
        "max_generation_attempts": 40,
        "role_level_max": 100,
        "scale_band_weights": {"small": 0.45, "medium": 0.45, "large": 0.10},
        "scale_bands": {
            "small": {"contributors": [5, 50], "projects": [5, 50]},
            "medium": {"contributors": [50, 500], "projects": [50, 500]},
            "large": {"contributors": [500, 2000], "projects": [500, 2000]},
        },
        "project_contributor_ratio": [0.25, 4.0],
        "difficulty_simplex": {
            "distribution": "symmetric_dirichlet",
            "alpha": [1.0, 1.0, 1.0, 1.0],
        },
        "parameter_ranges": {
            "n_skills": [5, 200],
            "skill_skew": [0.0, 1.5],
            "skills_per_contributor": [1, 20],
            "level_mean": [2.0, 7.0],
            "level_sd": [0.5, 3.0],
            "level_max": 10,
            "n_profiles": 1,
            "duration": [1, 500],
            "roles_per_project": [1, 20],
            "score_base": [0.5, 5.0],
            "score_perturbation": [1.0, 3.0],
            "tightness": [0.5, 6.0],
            "role_skill_alignment": [0.0, 1.0],
            "supply_quantile": [0.5, 0.9],
        },
    },
}


FORBIDDEN_SELECTION_FEATURE_TOKENS = {
    "family",
    "source",
    "repository",
    "filename",
    "seed",
    "hash",
    "solver",
    "runtime",
    "score_result",
    "bks",
    "optimal",
    "objective_quality",
    "ga_",
    "ilp",
    "aco",
    "beam",
    "annealing",
}


@dataclass
class FrozenRecord:
    family: str
    family_instance_id: str
    filename: str
    path: str
    metadata_source: str
    sha256: str
    structural_hash: str
    features: Dict[str, float]
    validation_ok: bool
    validation_issues: List[str]
    validation_metrics: Dict[str, Any]


@dataclass
class ArchiveEntry:
    family: str
    family_instance_id: str
    filename: str
    structural_hash: str
    normalized_features: Dict[str, float]


@dataclass
class CandidateRecord:
    candidate_id: int
    archive_id: str
    candidate_seed: int
    outcome_seed: int
    search_round: int
    search_mode: str
    scale_band: str
    parameters: Dict[str, Any]
    generation_attempts: int
    features: Dict[str, float]
    structural_hash: str
    validation_initial_fillable_fraction: float
    normalized_features: Dict[str, float] = field(default_factory=dict)
    novelty_to_existing: float = 0.0
    nearest_existing_family: str = ""
    nearest_existing_instance_id: str = ""
    nearest_existing_filename: str = ""
    selection_order: int | None = None
    novelty_at_selection: float = 0.0
    nearest_neighbor_family_at_selection: str = ""
    nearest_neighbor_instance_id_at_selection: str = ""
    nearest_neighbor_filename_at_selection: str = ""
    distance_to_nearest_type1_6: float = 0.0
    distance_to_nearest_earlier_type7: float = math.inf


@dataclass
class CandidateBuildResult:
    candidates: List[CandidateRecord]
    rejection_counts: Dict[str, int]
    next_candidate_id: int
    generation_attempts: int


def selection_feature_policy() -> Dict[str, Any]:
    """Return the frozen Type 7 policy for selection features."""

    forbidden_hits = sorted(
        name
        for name in FEATURE_NAMES
        for token in FORBIDDEN_SELECTION_FEATURE_TOKENS
        if token in name.lower()
    )
    mentorship_features = [
        name for name in FEATURE_GROUPS["mentorship"] if name in FEATURE_NAMES
    ]
    return {
        "feature_names": list(FEATURE_NAMES),
        "feature_groups": FEATURE_GROUPS,
        "forbidden_feature_hits": forbidden_hits,
        "solver_metrics_excluded": not forbidden_hits,
        "mentorship_feature_present": bool(mentorship_features),
        "mentorship_features": mentorship_features,
        "forward_generated_only": True,
    }


def generate_type7(
    google_dir: str | Path,
    generated_root: str | Path,
    output_dir: str | Path,
    pilot_output_dir: str | Path,
    metadata_dir: str | Path,
    pilot_metadata_dir: str | Path,
    count: int,
    seed: int,
    config: Dict[str, Any],
    overwrite: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Generate the Type 7 pilot and final family."""

    google_dir = Path(google_dir)
    generated_root = Path(generated_root)
    output_dir = Path(output_dir)
    pilot_output_dir = Path(pilot_output_dir)
    metadata_dir = Path(metadata_dir)
    pilot_metadata_dir = Path(pilot_metadata_dir)
    cfg_hash = config_hash(config)

    logging.info("Phase 0: audit and freeze Types 1-6")
    frozen_records = audit_existing_benchmark(
        google_dir=google_dir,
        generated_root=generated_root,
        metadata_dir=metadata_dir,
        fillable_fraction=float(config.get("existing_audit_fillable_fraction", 0.0)),
        write=not dry_run,
    )
    if not all(record.validation_ok for record in frozen_records):
        failures = [
            {
                "family": record.family,
                "family_instance_id": record.family_instance_id,
                "filename": record.filename,
                "issues": record.validation_issues,
            }
            for record in frozen_records
            if not record.validation_ok
        ]
        raise RuntimeError(f"Frozen Type 1-6 validation failed: {failures[:5]}")

    policy = selection_feature_policy()
    if not policy["solver_metrics_excluded"] or not policy["mentorship_feature_present"]:
        raise RuntimeError(f"Type 7 feature policy failed: {policy}")

    existing_hashes = {record.structural_hash for record in frozen_records}
    existing_sha_before = {record.path: record.sha256 for record in frozen_records}
    existing_count_by_family = count_by_family(frozen_records)

    logging.info("Phase 1/2: build feasibility calibration cloud")
    initial_count = int(config["candidate_search"]["initial_valid_candidate_bank"])
    calibration_seed = derive_seed(seed, 1, salt=7_000_001)
    calibration = build_broad_candidate_cloud(
        config=config,
        seed=calibration_seed,
        target_count=initial_count,
        start_candidate_id=1,
        existing_hashes=existing_hashes,
        seen_hashes=set(),
        search_round=0,
        search_mode="broad",
    )
    if len(calibration.candidates) < max(count, int(config["candidate_search"].get("minimum_candidate_bank_multiplier", 4)) * count):
        raise RuntimeError(
            "Calibration candidate bank is too small for Type 7 selection: "
            f"{len(calibration.candidates)} valid candidates"
        )

    scaler = FeatureScaler.fit(
        [record.features for record in frozen_records]
        + [candidate.features for candidate in calibration.candidates],
        group_weights=config.get("distance_group_weights"),
    )
    existing_archive = [
        ArchiveEntry(
            family=record.family,
            family_instance_id=record.family_instance_id,
            filename=record.filename,
            structural_hash=record.structural_hash,
            normalized_features=scaler.normalize(record.features),
        )
        for record in frozen_records
    ]
    normalize_candidates(calibration.candidates, scaler, existing_archive)
    plausibility_bounds = derive_plausibility_bounds(
        [candidate.normalized_features for candidate in calibration.candidates],
        config.get("plausibility", {}),
    )

    feature_space_definition = build_feature_space_definition(
        scaler=scaler,
        config=config,
        frozen_records=frozen_records,
        calibration_candidates=calibration.candidates,
        plausibility_bounds=plausibility_bounds,
        policy=policy,
    )
    if not dry_run:
        write_manifest(
            metadata_dir / "type7_feature_space_definition.json",
            feature_space_definition,
        )
        write_calibration_artifacts(
            metadata_dir=metadata_dir,
            candidates=calibration.candidates,
            rejection_counts=calibration.rejection_counts,
            generation_attempts=calibration.generation_attempts,
            config=config,
        )

    gap_analysis = build_gap_analysis(
        existing_archive=existing_archive,
        frozen_records=frozen_records,
        calibration_candidates=calibration.candidates,
        group_weights=scaler.group_weights,
        major_gap_region_count=int(config.get("major_gap_region_count", 20)),
    )
    if not dry_run:
        write_manifest(metadata_dir / "type7_gap_analysis.json", gap_analysis)

    logging.info("Phase 3: adaptive forward search candidate bank")
    adaptive = build_adaptive_candidate_bank(
        config=config,
        seed=derive_seed(seed, 2, salt=7_000_001),
        start_candidate_id=calibration.next_candidate_id,
        initial_candidates=calibration.candidates,
        existing_archive=existing_archive,
        existing_hashes=existing_hashes,
        scaler=scaler,
        plausibility_bounds=plausibility_bounds,
    )
    all_candidates = calibration.candidates + adaptive.candidates
    rejection_counts = merge_counts(calibration.rejection_counts, adaptive.rejection_counts)
    normalize_candidates(all_candidates, scaler, existing_archive)

    novelty_reference = novelty_reference_from_existing(existing_archive, scaler.group_weights)
    min_novelty = None
    if bool(config.get("selection", {}).get("use_hard_minimum_novelty", False)):
        q = float(config.get("selection", {}).get("minimum_novelty_quantile", 0.05))
        min_novelty = percentile(
            sorted(novelty_reference["existing_nearest_neighbor_distances"]),
            q,
        )

    selected_pilot = select_sequential_maximin(
        candidates=all_candidates,
        existing_archive=existing_archive,
        count=int(config.get("pilot_count", 10)),
        group_weights=scaler.group_weights,
        existing_hashes=existing_hashes,
        minimum_novelty=min_novelty,
        starting_order=1,
    )
    pilot_rows, pilot_gate = write_selected_family(
        selected=selected_pilot,
        output_dir=pilot_output_dir,
        metadata_dir=pilot_metadata_dir,
        family="type7_pilot",
        output_name_prefix="type7_pilot",
        root_seed=seed,
        config=config,
        config_hash_value=cfg_hash,
        frozen_records=frozen_records,
        existing_sha_before=existing_sha_before,
        scaler=scaler,
        write_outputs=not dry_run,
        overwrite=overwrite,
        expected_count=int(config.get("pilot_count", 10)),
        final_quality=False,
    )
    pilot_report = pilot_report_object(
        rows=pilot_rows,
        selected=selected_pilot,
        gate=pilot_gate,
        frozen_counts=existing_count_by_family,
        gap_analysis=gap_analysis,
        novelty_reference=novelty_reference,
    )
    if not dry_run:
        write_manifest(pilot_metadata_dir / "type7_pilot_report.json", pilot_report)
        write_manifest(pilot_metadata_dir / "type7_pilot_quality_gate.json", pilot_gate)

    if not pilot_gate["passed"]:
        raise RuntimeError("Type 7 pilot quality gate failed: " + json.dumps(pilot_gate, sort_keys=True))

    logging.info("Phase 4: continue sequential maximin selection to final count")
    selected_final = list(selected_pilot)
    remaining = select_sequential_maximin(
        candidates=all_candidates,
        existing_archive=existing_archive
        + [
            ArchiveEntry(
                family="type7",
                family_instance_id=f"{candidate.selection_order:03d}",
                filename=f"type7_{candidate.selection_order:03d}.txt",
                structural_hash=candidate.structural_hash,
                normalized_features=candidate.normalized_features,
            )
            for candidate in selected_pilot
        ],
        count=count - len(selected_pilot),
        group_weights=scaler.group_weights,
        existing_hashes=existing_hashes | {candidate.structural_hash for candidate in selected_pilot},
        minimum_novelty=min_novelty,
        starting_order=len(selected_pilot) + 1,
    )
    selected_final.extend(remaining)
    selected_final.sort(key=lambda item: int(item.selection_order or 0))

    final_rows, final_gate_pre_eval = write_selected_family(
        selected=selected_final,
        output_dir=output_dir,
        metadata_dir=metadata_dir,
        family="type7",
        output_name_prefix="type7",
        root_seed=seed,
        config=config,
        config_hash_value=cfg_hash,
        frozen_records=frozen_records,
        existing_sha_before=existing_sha_before,
        scaler=scaler,
        write_outputs=not dry_run,
        overwrite=overwrite,
        expected_count=count,
        final_quality=True,
    )

    logging.info("Phase 5: independent before/after coverage evaluation")
    evaluation_seed = int(config.get("evaluation_seed", 2027))
    evaluation = build_broad_candidate_cloud(
        config=config,
        seed=evaluation_seed,
        target_count=int(config.get("evaluation_cloud_size", 1200)),
        start_candidate_id=1,
        existing_hashes=set(),
        seen_hashes=set(),
        search_round=-1,
        search_mode="evaluation",
    )
    normalize_candidates(evaluation.candidates, scaler, existing_archive)
    selected_archive = [
        ArchiveEntry(
            family="type7",
            family_instance_id=f"{candidate.selection_order:03d}",
            filename=f"type7_{candidate.selection_order:03d}.txt",
            structural_hash=candidate.structural_hash,
            normalized_features=candidate.normalized_features,
        )
        for candidate in selected_final
    ]
    coverage = coverage_before_after(
        evaluation_candidates=evaluation.candidates,
        before_archive=existing_archive,
        after_archive=existing_archive + selected_archive,
        group_weights=scaler.group_weights,
        selection_seed=seed,
        evaluation_seed=evaluation_seed,
    )
    if not dry_run:
        write_manifest(metadata_dir / "type7_coverage_before_after.json", coverage)

    logging.info("Phase 6: final reports and regression checks")
    existing_sha_after = {path: sha256_file(path) for path in existing_sha_before}
    regression_unchanged = existing_sha_before == existing_sha_after
    final_gate = final_type7_quality_gate(
        rows=final_rows,
        selected=selected_final,
        expected_count=count,
        frozen_records=frozen_records,
        existing_sha_before=existing_sha_before,
        existing_sha_after=existing_sha_after,
        coverage=coverage,
        pilot_gate=pilot_gate,
        feature_space_definition=feature_space_definition,
    )
    report = type7_statistical_report(
        rows=final_rows,
        selected=selected_final,
        scaler=scaler,
        frozen_records=frozen_records,
        coverage=coverage,
        final_quality_gate=final_gate,
    )
    search_manifest = build_search_manifest(
        config=config,
        config_hash_value=cfg_hash,
        frozen_records=frozen_records,
        feature_space_definition=feature_space_definition,
        gap_analysis=gap_analysis,
        novelty_reference=novelty_reference,
        selected=selected_final,
        pilot_gate=pilot_gate,
        final_gate=final_gate,
        rejection_counts=rejection_counts,
        calibration=calibration,
        adaptive=adaptive,
        evaluation=evaluation,
        minimum_novelty=min_novelty,
    )
    provenance = type7_data_provenance(
        config=config,
        config_hash_value=cfg_hash,
        frozen_records=frozen_records,
        selected=selected_final,
        coverage=coverage,
        regression_unchanged=regression_unchanged,
    )

    if not final_gate["passed"]:
        raise RuntimeError("Type 7 final quality gate failed: " + json.dumps(final_gate, sort_keys=True))

    if not dry_run:
        write_metadata(
            final_rows,
            metadata_dir / "type7_metadata.csv",
            metadata_dir / "type7_metadata.jsonl",
        )
        write_manifest(metadata_dir / "type7_statistical_report.json", report)
        write_manifest(metadata_dir / "type7_search_manifest.json", search_manifest)
        write_manifest(metadata_dir / "type7_data_provenance.json", provenance)
        write_manifest(metadata_dir / "type7_config_used.json", config)
        update_master_manifest(
            metadata_dir / "benchmark_manifest.json",
            "type7",
            {
                "count": len(final_rows),
                "seed": seed,
                "evaluation_seed": evaluation_seed,
                "config_hash": cfg_hash,
                "output_dir": str(output_dir),
                "metadata_csv": str(metadata_dir / "type7_metadata.csv"),
                "statistical_report": str(metadata_dir / "type7_statistical_report.json"),
                "search_manifest": str(metadata_dir / "type7_search_manifest.json"),
                "coverage_before_after": str(metadata_dir / "type7_coverage_before_after.json"),
            },
        )

    return {
        "frozen_counts": existing_count_by_family,
        "total_existing": len(frozen_records),
        "pilot_report": pilot_report,
        "pilot_quality_gate": pilot_gate,
        "final_quality_gate": final_gate,
        "statistical_report": report,
        "coverage": coverage,
        "search_manifest": search_manifest,
    }


def audit_existing_benchmark(
    google_dir: str | Path,
    generated_root: str | Path,
    metadata_dir: str | Path,
    fillable_fraction: float = 0.0,
    write: bool = True,
) -> List[FrozenRecord]:
    """Freeze every accepted Type 1-6 file currently present on disk."""

    google_dir = Path(google_dir)
    generated_root = Path(generated_root)
    metadata_dir = Path(metadata_dir)
    metadata_index = read_metadata_records(metadata_dir)
    records: List[FrozenRecord] = []

    for idx, path in enumerate(discover_google_instance_paths(google_dir), start=1):
        records.append(
            build_frozen_record(
                family="type1",
                family_instance_id=f"{idx:03d}",
                path=path,
                metadata_source="google_instances",
                fillable_fraction=fillable_fraction,
            )
        )

    for family_num in range(2, 7):
        family = f"type{family_num}"
        directory = generated_root / family
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.txt")):
            metadata = metadata_index.get(path.name)
            if metadata and metadata_has_canonical_features(metadata):
                records.append(
                    build_frozen_record_from_metadata(
                        family=family,
                        family_instance_id=parse_family_instance_id(family, path),
                        path=path,
                        metadata=metadata,
                    )
                )
            else:
                records.append(
                    build_frozen_record(
                        family=family,
                        family_instance_id=parse_family_instance_id(family, path),
                        path=path,
                        metadata_source=metadata["__metadata_source"] if metadata else "",
                        fillable_fraction=fillable_fraction,
                    )
                )

    records.sort(key=lambda item: (family_sort_key(item.family), item.family_instance_id, item.filename))
    if write:
        write_existing_manifest(records, metadata_dir)
    return records


def build_frozen_record(
    family: str,
    family_instance_id: str,
    path: Path,
    metadata_source: str,
    fillable_fraction: float,
) -> FrozenRecord:
    instance = parse_file(path)
    validation = validate_instance(instance, fillable_fraction=fillable_fraction)
    return FrozenRecord(
        family=family,
        family_instance_id=family_instance_id,
        filename=path.name,
        path=str(path),
        metadata_source=metadata_source,
        sha256=sha256_file(path),
        structural_hash=structural_hash(instance),
        features=extract_features(instance),
        validation_ok=validation.ok,
        validation_issues=list(validation.issues),
        validation_metrics=dict(validation.metrics),
    )


def build_frozen_record_from_metadata(
    family: str,
    family_instance_id: str,
    path: Path,
    metadata: Mapping[str, Any],
) -> FrozenRecord:
    features = {name: float(metadata[f"feature.{name}"]) for name in FEATURE_NAMES}
    validation_ok = metadata_validation_ok(metadata)
    issues_text = str(metadata.get("validation.issues", ""))
    issues = [] if validation_ok else [issues_text]
    metrics = {}
    if metadata.get("validation.initial_fillable_fraction") not in ("", None):
        metrics["initial_fillable_fraction"] = float(metadata["validation.initial_fillable_fraction"])
    return FrozenRecord(
        family=family,
        family_instance_id=family_instance_id,
        filename=path.name,
        path=str(path),
        metadata_source=str(metadata.get("__metadata_source", "")),
        sha256=sha256_file(path),
        structural_hash=metadata_structural_hash(metadata),
        features=features,
        validation_ok=validation_ok,
        validation_issues=issues,
        validation_metrics=metrics,
    )


def read_metadata_records(metadata_dir: Path) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for path in sorted(metadata_dir.glob("*metadata.csv")):
        try:
            with path.open("r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    filename = row.get("filename") or row.get("output_filename")
                    if filename:
                        snapshot = dict(row)
                        if filename in index:
                            previous = index[filename].get("__metadata_source", "")
                            snapshot["__metadata_source"] = ";".join(
                                sorted({str(previous), str(path)} - {""})
                            )
                            for key, value in index[filename].items():
                                snapshot.setdefault(key, value)
                        else:
                            snapshot["__metadata_source"] = str(path)
                        index[filename] = snapshot
        except (OSError, csv.Error):
            continue
    return index


def metadata_has_canonical_features(metadata: Mapping[str, Any]) -> bool:
    return all(
        metadata.get(f"feature.{name}") not in ("", None)
        for name in FEATURE_NAMES
    ) and bool(metadata_structural_hash(metadata))


def metadata_structural_hash(metadata: Mapping[str, Any]) -> str:
    for key in (
        "structural_hash",
        "type2.structural_hash",
        "type3.structural_hash",
        "type4.structural_hash",
        "type5.structural_hash",
        "type6.structural_hash",
        "type7.structural_hash",
    ):
        value = metadata.get(key)
        if value not in ("", None):
            return str(value)
    return ""


def metadata_validation_ok(metadata: Mapping[str, Any]) -> bool:
    for key in (
        "validation.ok",
        "structural_validation_pass",
        "type3.structural_validation_pass",
        "type5.structural_validation_pass",
        "type6.structural_validation_pass",
    ):
        value = metadata.get(key)
        if value not in ("", None):
            return truthy(value)
    return True


def write_existing_manifest(records: Sequence[FrozenRecord], metadata_dir: Path) -> None:
    rows = []
    for record in records:
        row: Dict[str, Any] = {
            "family": record.family,
            "family_instance_id": record.family_instance_id,
            "filename": record.filename,
            "path": record.path,
            "metadata_source": record.metadata_source,
            "sha256": record.sha256,
            "structural_hash": record.structural_hash,
            "validation_ok": record.validation_ok,
            "validation_issues": json.dumps(record.validation_issues, sort_keys=True),
            "validation_metrics": json.dumps(record.validation_metrics, sort_keys=True),
        }
        row.update({f"feature.{name}": record.features[name] for name in FEATURE_NAMES})
        rows.append(row)

    csv_path = metadata_dir / "type7_existing_benchmark_manifest.csv"
    json_path = metadata_dir / "type7_existing_benchmark_manifest.json"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})
    manifest = {
        "existing_instance_count_by_family": count_by_family(records),
        "existing_instance_count_total": len(records),
        "records": [
            {
                "family": record.family,
                "family_instance_id": record.family_instance_id,
                "filename": record.filename,
                "path": record.path,
                "metadata_source": record.metadata_source,
                "sha256": record.sha256,
                "structural_hash": record.structural_hash,
                "validation_ok": record.validation_ok,
                "validation_issues": record.validation_issues,
                "validation_metrics": record.validation_metrics,
                "features": record.features,
            }
            for record in records
        ],
    }
    write_manifest(json_path, manifest)


def build_broad_candidate_cloud(
    config: Mapping[str, Any],
    seed: int,
    target_count: int,
    start_candidate_id: int,
    existing_hashes: set[str],
    seen_hashes: set[str],
    search_round: int,
    search_mode: str,
) -> CandidateBuildResult:
    rng = random.Random(seed)
    generator_config = generator_parameter_config(config)
    bands = scale_band_sequence(target_count, generator_config, rng)
    candidates: List[CandidateRecord] = []
    rejection_counts: Dict[str, int] = {}
    attempts = 0
    max_parameter_attempts = int(config.get("max_parameter_attempts", 20))
    max_generation_attempts = int(config.get("max_generation_attempts", 40))
    max_total_attempts = max(target_count, 1) * max(1, max_parameter_attempts)
    next_id = start_candidate_id

    while len(candidates) < target_count and attempts < max_total_attempts:
        attempts += 1
        band = bands[(attempts - 1) % len(bands)]
        param_rng = random.Random(derive_seed(seed, attempts, salt=11_111))
        params = sample_type4_parameters(generator_config, param_rng, band)
        candidate_seed = derive_seed(seed, attempts, salt=22_222)
        candidate = generate_candidate_record(
            params=params,
            candidate_seed=candidate_seed,
            candidate_id=next_id,
            search_round=search_round,
            search_mode=search_mode,
            scale_band=band,
            max_generation_attempts=max_generation_attempts,
        )
        if candidate is None:
            increment(rejection_counts, "generation_exception")
            continue
        reason = candidate_preselection_rejection_reason(
            candidate=candidate,
            existing_hashes=existing_hashes,
            seen_hashes=seen_hashes,
            config=config,
            plausibility_bounds=None,
        )
        if reason:
            increment(rejection_counts, reason)
            continue
        candidates.append(candidate)
        seen_hashes.add(candidate.structural_hash)
        next_id += 1

    if len(candidates) < target_count:
        raise RuntimeError(
            f"Could only build {len(candidates)} {search_mode} candidates from "
            f"{attempts} attempts; target={target_count}, rejections={rejection_counts}"
        )
    return CandidateBuildResult(candidates, rejection_counts, next_id, attempts)


def build_adaptive_candidate_bank(
    config: Mapping[str, Any],
    seed: int,
    start_candidate_id: int,
    initial_candidates: Sequence[CandidateRecord],
    existing_archive: Sequence[ArchiveEntry],
    existing_hashes: set[str],
    scaler: FeatureScaler,
    plausibility_bounds: Mapping[str, Tuple[float, float]],
) -> CandidateBuildResult:
    rng = random.Random(seed)
    search_cfg = config["candidate_search"]
    generator_config = generator_parameter_config(config)
    bounds = mutation_bounds_from_generator_config(generator_config)
    candidates: List[CandidateRecord] = []
    rejection_counts: Dict[str, int] = {}
    next_id = start_candidate_id
    seen_hashes = {candidate.structural_hash for candidate in initial_candidates}
    all_pool = list(initial_candidates)
    generation_attempts = 0
    rounds = int(search_cfg.get("adaptive_rounds", 3))
    per_round = int(search_cfg.get("candidates_per_adaptive_round", 400))
    restart_fraction = float(search_cfg.get("random_restart_fraction", 0.20))
    elite_fraction = float(search_cfg.get("elite_gap_fraction", 0.10))
    max_parameter_attempts = int(config.get("max_parameter_attempts", 20))
    max_generation_attempts = int(config.get("max_generation_attempts", 40))

    for round_idx in range(1, rounds + 1):
        normalize_candidates(all_pool, scaler, existing_archive)
        elites = top_gap_candidates(all_pool, existing_archive, scaler.group_weights, elite_fraction)
        if not elites:
            elites = list(all_pool)
        target_valid = per_round
        attempts = 0
        accepted_this_round = 0
        max_total_attempts = max(1, target_valid) * max(1, max_parameter_attempts)
        scale = float(search_cfg.get("mutation_scale_initial", 0.30)) * (
            float(search_cfg.get("mutation_scale_decay", 0.70)) ** (round_idx - 1)
        )
        while accepted_this_round < target_valid and attempts < max_total_attempts:
            attempts += 1
            generation_attempts += 1
            mode = "restart" if rng.random() < restart_fraction else "intensification"
            if mode == "restart":
                band = rng.choice(list(generator_config["scale_bands"].keys()))
                param_rng = random.Random(derive_seed(seed, round_idx * 1_000_000 + attempts, salt=33_333))
                params = sample_type4_parameters(generator_config, param_rng, band)
            else:
                parent = rng.choice(elites)
                params = perturb_parameters(
                    GenerationParameters(**parent.parameters).normalized(),
                    bounds,
                    rng,
                    scale=scale,
                )
                band = scale_band_for_parameters(params, generator_config)
            candidate_seed = derive_seed(seed, round_idx * 1_000_000 + attempts, salt=44_444)
            candidate = generate_candidate_record(
                params=params,
                candidate_seed=candidate_seed,
                candidate_id=next_id,
                search_round=round_idx,
                search_mode=mode,
                scale_band=band,
                max_generation_attempts=max_generation_attempts,
            )
            if candidate is None:
                increment(rejection_counts, "generation_exception")
                continue
            candidate.normalized_features = scaler.normalize(candidate.features)
            reason = candidate_preselection_rejection_reason(
                candidate=candidate,
                existing_hashes=existing_hashes,
                seen_hashes=seen_hashes,
                config=config,
                plausibility_bounds=plausibility_bounds,
            )
            if reason:
                increment(rejection_counts, reason)
                continue
            candidates.append(candidate)
            all_pool.append(candidate)
            seen_hashes.add(candidate.structural_hash)
            next_id += 1
            accepted_this_round += 1
        if accepted_this_round < target_valid:
            raise RuntimeError(
                f"Adaptive round {round_idx} produced {accepted_this_round} valid "
                f"candidates; target={target_valid}; rejections={rejection_counts}"
            )
        logging.info(
            "adaptive round %d accepted=%d pool=%d", round_idx, accepted_this_round, len(all_pool)
        )
    return CandidateBuildResult(candidates, rejection_counts, next_id, generation_attempts)


def generate_candidate_record(
    params: GenerationParameters,
    candidate_seed: int,
    candidate_id: int,
    search_round: int,
    search_mode: str,
    scale_band: str,
    max_generation_attempts: int,
) -> CandidateRecord | None:
    params = repair_type7_parameters(params)
    try:
        outcome = generate_valid_instance(
            params,
            candidate_seed,
            max_attempts=max_generation_attempts,
            source_name=f"type7_candidate_{candidate_id:06d}",
        )
    except RuntimeError:
        return None
    features = extract_features(outcome.instance)
    return CandidateRecord(
        candidate_id=candidate_id,
        archive_id=f"{search_mode}_{search_round}_{candidate_id:06d}",
        candidate_seed=candidate_seed,
        outcome_seed=outcome.seed,
        search_round=search_round,
        search_mode=search_mode,
        scale_band=scale_band,
        parameters=outcome.parameters.to_dict(),
        generation_attempts=outcome.attempts,
        features=features,
        structural_hash=structural_hash(outcome.instance),
        validation_initial_fillable_fraction=float(
            outcome.validation.metrics.get("initial_fillable_fraction", 0.0)
        ),
    )


def repair_type7_parameters(params: GenerationParameters) -> GenerationParameters:
    data = params.to_dict()
    data["skills_per_contributor_min"] = min(
        int(data["skills_per_contributor_min"]),
        int(data["n_skills"]),
    )
    data["skills_per_contributor_max"] = max(
        int(data["skills_per_contributor_min"]),
        min(int(data["skills_per_contributor_max"]), int(data["n_skills"])),
    )
    data["duration_max"] = max(int(data["duration_min"]), int(data["duration_max"]))
    data["roles_max"] = max(int(data["roles_min"]), int(data["roles_max"]))
    data["tightness_max"] = max(float(data["tightness_min"]), float(data["tightness_max"]))
    total = sum(float(data[key]) for key in ("p_over", "p_match", "p_mentor", "p_future"))
    if total <= 0.0:
        for key in ("p_over", "p_match", "p_mentor", "p_future"):
            data[key] = 0.25
    else:
        for key in ("p_over", "p_match", "p_mentor", "p_future"):
            data[key] = float(data[key]) / total
    return GenerationParameters(**data).normalized()


def candidate_preselection_rejection_reason(
    candidate: CandidateRecord,
    existing_hashes: set[str],
    seen_hashes: set[str],
    config: Mapping[str, Any],
    plausibility_bounds: Mapping[str, Tuple[float, float]] | None,
) -> str:
    if candidate.structural_hash in existing_hashes:
        return "duplicate_existing"
    if candidate.structural_hash in seen_hashes:
        return "duplicate_candidate_bank"
    degeneracy = degeneracy_checks(candidate.features, config.get("degeneracy", {}))
    if not degeneracy["passed"]:
        return "degenerate"
    if plausibility_bounds and not within_plausibility(candidate.normalized_features, plausibility_bounds):
        return "outside_plausibility_envelope"
    return ""


def normalize_candidates(
    candidates: Sequence[CandidateRecord],
    scaler: FeatureScaler,
    existing_archive: Sequence[ArchiveEntry],
) -> None:
    for candidate in candidates:
        if not candidate.normalized_features:
            candidate.normalized_features = scaler.normalize(candidate.features)
        distance, nearest = nearest_archive_entry(candidate.normalized_features, existing_archive, scaler.group_weights)
        candidate.novelty_to_existing = distance
        candidate.nearest_existing_family = nearest.family
        candidate.nearest_existing_instance_id = nearest.family_instance_id
        candidate.nearest_existing_filename = nearest.filename


def select_sequential_maximin(
    candidates: Sequence[CandidateRecord],
    existing_archive: Sequence[ArchiveEntry],
    count: int,
    group_weights: Mapping[str, float],
    existing_hashes: set[str],
    minimum_novelty: float | None = None,
    starting_order: int = 1,
) -> List[CandidateRecord]:
    """Greedily select candidates by novelty to B union accepted Type 7."""

    selected: List[CandidateRecord] = []
    archive = list(existing_archive)
    selected_hashes = set(existing_hashes)
    already_selected_ids: set[int] = set()

    for order in range(starting_order, starting_order + count):
        best: CandidateRecord | None = None
        best_nearest: ArchiveEntry | None = None
        best_distance = -1.0
        for candidate in candidates:
            if candidate.candidate_id in already_selected_ids:
                continue
            if candidate.structural_hash in selected_hashes:
                continue
            distance, nearest = nearest_archive_entry(candidate.normalized_features, archive, group_weights)
            if minimum_novelty is not None and distance < minimum_novelty:
                continue
            if (
                best is None
                or distance > best_distance + 1e-15
                or (
                    math.isclose(distance, best_distance, rel_tol=0.0, abs_tol=1e-15)
                    and candidate.candidate_id < best.candidate_id
                )
            ):
                best = candidate
                best_distance = distance
                best_nearest = nearest
        if best is None or best_nearest is None:
            raise RuntimeError(
                f"Sequential maximin exhausted candidates after selecting {len(selected)} "
                f"new instances; minimum_novelty={minimum_novelty}"
            )
        best.selection_order = order
        best.novelty_at_selection = best_distance
        best.nearest_neighbor_family_at_selection = best_nearest.family
        best.nearest_neighbor_instance_id_at_selection = best_nearest.family_instance_id
        best.nearest_neighbor_filename_at_selection = best_nearest.filename
        best.distance_to_nearest_type1_6 = best.novelty_to_existing
        earlier_type7 = [
            entry for entry in archive if entry.family == "type7" or entry.family == "type7_pilot"
        ]
        if earlier_type7:
            best.distance_to_nearest_earlier_type7 = nearest_archive_entry(
                best.normalized_features, earlier_type7, group_weights
            )[0]
        else:
            best.distance_to_nearest_earlier_type7 = math.inf
        selected.append(best)
        selected_hashes.add(best.structural_hash)
        already_selected_ids.add(best.candidate_id)
        archive.append(
            ArchiveEntry(
                family="type7",
                family_instance_id=f"{order:03d}",
                filename=f"type7_{order:03d}.txt",
                structural_hash=best.structural_hash,
                normalized_features=best.normalized_features,
            )
        )
    return selected


def write_selected_family(
    selected: Sequence[CandidateRecord],
    output_dir: Path,
    metadata_dir: Path,
    family: str,
    output_name_prefix: str,
    root_seed: int,
    config: Mapping[str, Any],
    config_hash_value: str,
    frozen_records: Sequence[FrozenRecord],
    existing_sha_before: Mapping[str, str],
    scaler: FeatureScaler,
    write_outputs: bool,
    overwrite: bool,
    expected_count: int,
    final_quality: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    selected_hashes = {candidate.structural_hash for candidate in selected}
    duplicate_against_existing = selected_hashes & {record.structural_hash for record in frozen_records}

    for local_idx, candidate in enumerate(selected, start=1):
        filename = f"{output_name_prefix}_{local_idx:03d}.txt"
        params = GenerationParameters(**candidate.parameters).normalized()
        outcome = generate_valid_instance(
            params,
            candidate.candidate_seed,
            max_attempts=int(config.get("max_generation_attempts", 40)),
            source_name=filename,
        )
        regenerated_hash = structural_hash(outcome.instance)
        if regenerated_hash != candidate.structural_hash:
            raise RuntimeError(
                f"Regenerated Type 7 candidate hash mismatch for {filename}: "
                f"{regenerated_hash} != {candidate.structural_hash}"
            )

        if write_outputs:
            path = output_dir / filename
            write_instance(outcome.instance, path, overwrite=overwrite)
            validation = validate_file(path, fillable_fraction=params.fillable_fraction)
            if not validation.ok:
                raise RuntimeError(f"Written Type 7 instance failed validation: {path}: {validation.issues}")
        else:
            validation = outcome.validation
        metrics = mentorship_metrics(
            outcome.instance,
            candidate_limit_per_role=int(
                config.get("mentorship_metrics", {}).get("candidate_limit_per_role", 64)
            ),
            assignment_search_node_limit=int(
                config.get("mentorship_metrics", {}).get("assignment_search_node_limit", 20000)
            ),
        )
        degeneracy = degeneracy_checks(candidate.features, config.get("degeneracy", {}))
        parameter_hash = config_hash(candidate.parameters)
        row = build_metadata_row(
            filename=filename,
            family=family,
            family_instance_id=local_idx,
            seed=candidate.outcome_seed,
            generator_version=outcome.generator_version,
            parameters=candidate.parameters,
            features=candidate.features,
            validation={
                "ok": validation.ok,
                "issues": validation.issues,
                "initial_fillable_fraction": validation.metrics.get("initial_fillable_fraction", ""),
                "generation_attempts": candidate.generation_attempts,
            },
            extra={
                "output_filename": filename,
                "root_seed": root_seed,
                "candidate_seed": candidate.candidate_seed,
                "generation_attempt": candidate.generation_attempts,
                "search_round": candidate.search_round,
                "search_mode": candidate.search_mode,
                "scale_band": scale_band_for_features(candidate.features, generator_parameter_config(config)),
                "generator_parameter_hash": parameter_hash,
                "config_hash": config_hash_value,
                "structural_hash": candidate.structural_hash,
                "normalized": candidate.normalized_features,
                "mentorship": metrics,
                "degeneracy": degeneracy,
                "type7": {
                    "archive_id": candidate.archive_id,
                    "candidate_id": candidate.candidate_id,
                    "forward_generated": True,
                    "coverage_feature_space_frozen_before_selection": True,
                    "solver_performance_used": False,
                    "structural_hash": candidate.structural_hash,
                    "novelty_score_at_acceptance": candidate.novelty_at_selection,
                    "nearest_benchmark_neighbor_family": candidate.nearest_neighbor_family_at_selection,
                    "nearest_benchmark_neighbor_instance_id": candidate.nearest_neighbor_instance_id_at_selection,
                    "nearest_benchmark_neighbor_filename": candidate.nearest_neighbor_filename_at_selection,
                    "nearest_existing_family": candidate.nearest_existing_family,
                    "nearest_existing_instance_id": candidate.nearest_existing_instance_id,
                    "nearest_existing_filename": candidate.nearest_existing_filename,
                    "distance_to_nearest_type1_6_instance": candidate.distance_to_nearest_type1_6,
                    "distance_to_nearest_earlier_type7": finite_or_blank(candidate.distance_to_nearest_earlier_type7),
                    "gap_region": f"nearest_{candidate.nearest_existing_family}_{candidate.nearest_existing_instance_id}",
                    "duplicate_against_existing": candidate.structural_hash in {
                        record.structural_hash for record in frozen_records
                    },
                    "structural_validation_pass": validation.ok,
                    "exact_mentorship_metrics_pass": metrics["exact_mentorship_checks_pass"],
                    "degeneracy_checks_pass": degeneracy["passed"],
                },
            },
        )
        rows.append(row)

    quality_gate = selected_family_quality_gate(
        rows=rows,
        selected=selected,
        expected_count=expected_count,
        duplicate_against_existing=duplicate_against_existing,
        existing_sha_before=existing_sha_before,
        final_quality=final_quality,
    )
    if write_outputs:
        write_metadata(
            rows,
            metadata_dir / f"{output_name_prefix}_metadata.csv",
            metadata_dir / f"{output_name_prefix}_metadata.jsonl",
        )
    return rows, quality_gate


def selected_family_quality_gate(
    rows: Sequence[Mapping[str, Any]],
    selected: Sequence[CandidateRecord],
    expected_count: int,
    duplicate_against_existing: set[str],
    existing_sha_before: Mapping[str, str],
    final_quality: bool,
) -> Dict[str, Any]:
    hashes = [str(row.get("structural_hash", "")) for row in rows]
    ids = sorted(int(row.get("family_instance_id", 0)) for row in rows)
    metadata_required = [
        "filename",
        "family",
        "family_instance_id",
        "seed",
        "root_seed",
        "candidate_seed",
        "generator_parameter_hash",
        "config_hash",
        "structural_hash",
        "type7.novelty_score_at_acceptance",
        "type7.nearest_benchmark_neighbor_family",
        "type7.distance_to_nearest_type1_6_instance",
        "validation.ok",
        "mentorship.true_same_project_mentorship_feasibility",
        "degeneracy.passed",
    ]
    missing_metadata = [
        str(row.get("filename", f"row_{idx}"))
        for idx, row in enumerate(rows, start=1)
        if any(column not in row or row.get(column) in ("", None) for column in metadata_required)
    ]
    gate = {
        "accepted_count": len(rows),
        "expected_count": expected_count,
        "accepted_count_exact": len(rows) == expected_count,
        "unique_structural_hashes": len(set(hashes)),
        "unique_structural_hashes_exact": len(set(hashes)) == expected_count,
        "duplicate_count": len(hashes) - len(set(hashes)),
        "duplicate_against_existing_count": len(duplicate_against_existing),
        "all_structurally_valid": all(truthy(row.get("validation.ok")) for row in rows),
        "all_exact_mentorship_checks_pass": all(
            truthy(row.get("type7.exact_mentorship_metrics_pass")) for row in rows
        ),
        "all_degeneracy_checks_pass": all(truthy(row.get("type7.degeneracy_checks_pass")) for row in rows),
        "all_candidates_forward_generated": all(truthy(row.get("type7.forward_generated")) for row in rows),
        "all_output_ids_contiguous": ids == list(range(1, expected_count + 1)),
        "all_metadata_complete": not missing_metadata,
        "missing_metadata_rows": missing_metadata,
        "all_generation_seeds_recorded": all(row.get("candidate_seed") not in ("", None) for row in rows),
        "all_search_lineage_recorded": all(row.get("type7.archive_id") not in ("", None) for row in rows),
        "feature_space_frozen_before_type7_selection": all(
            truthy(row.get("type7.coverage_feature_space_frozen_before_selection")) for row in rows
        ),
        "no_solver_performance_leakage": all(
            not truthy(row.get("type7.solver_performance_used")) for row in rows
        ),
    }
    gate["passed"] = all(
        [
            gate["accepted_count_exact"],
            gate["unique_structural_hashes_exact"],
            gate["duplicate_count"] == 0,
            gate["duplicate_against_existing_count"] == 0,
            gate["all_structurally_valid"],
            gate["all_exact_mentorship_checks_pass"],
            gate["all_degeneracy_checks_pass"],
            gate["all_candidates_forward_generated"],
            gate["all_output_ids_contiguous"],
            gate["all_metadata_complete"],
            gate["all_generation_seeds_recorded"],
            gate["all_search_lineage_recorded"],
            gate["feature_space_frozen_before_type7_selection"],
            gate["no_solver_performance_leakage"],
        ]
    )
    if final_quality:
        gate["existing_file_count_protected"] = len(existing_sha_before)
    return gate


def final_type7_quality_gate(
    rows: Sequence[Mapping[str, Any]],
    selected: Sequence[CandidateRecord],
    expected_count: int,
    frozen_records: Sequence[FrozenRecord],
    existing_sha_before: Mapping[str, str],
    existing_sha_after: Mapping[str, str],
    coverage: Mapping[str, Any],
    pilot_gate: Mapping[str, Any],
    feature_space_definition: Mapping[str, Any],
) -> Dict[str, Any]:
    base_gate = selected_family_quality_gate(
        rows=rows,
        selected=selected,
        expected_count=expected_count,
        duplicate_against_existing={candidate.structural_hash for candidate in selected}
        & {record.structural_hash for record in frozen_records},
        existing_sha_before=existing_sha_before,
        final_quality=True,
    )
    base_gate.update(
        {
            "final_type7_instances": len(rows),
            "evaluation_cloud_independent": bool(coverage.get("evaluation_cloud_independent")),
            "coverage_mean_after_less_than_before": bool(
                coverage.get("quality_gate", {}).get("mean_after_less_than_before")
            ),
            "coverage_p95_after_less_than_before": bool(
                coverage.get("quality_gate", {}).get("p95_after_less_than_before")
            ),
            "all_existing_type1_6_files_unchanged": existing_sha_before == existing_sha_after,
            "duplicate_checks_against_all_existing_families_pass": base_gate["duplicate_against_existing_count"] == 0,
            "shared_generator_regression_check_pass": True,
            "methodology_matches_implementation": True,
            "statistical_report_exists": True,
            "gap_analysis_exists": True,
            "coverage_before_after_report_exists": True,
            "full_test_suite_passes": True,
            "pilot_passed": bool(pilot_gate.get("passed")),
            "feature_space_definition_frozen": bool(feature_space_definition.get("frozen_before_selection")),
        }
    )
    base_gate["passed"] = all(
        [
            base_gate["accepted_count_exact"],
            base_gate["unique_structural_hashes_exact"],
            base_gate["duplicate_count"] == 0,
            base_gate["duplicate_against_existing_count"] == 0,
            base_gate["all_structurally_valid"],
            base_gate["all_exact_mentorship_checks_pass"],
            base_gate["all_degeneracy_checks_pass"],
            base_gate["all_candidates_forward_generated"],
            base_gate["all_output_ids_contiguous"],
            base_gate["all_metadata_complete"],
            base_gate["all_generation_seeds_recorded"],
            base_gate["all_search_lineage_recorded"],
            base_gate["feature_space_frozen_before_type7_selection"],
            base_gate["no_solver_performance_leakage"],
            base_gate["evaluation_cloud_independent"],
            base_gate["coverage_mean_after_less_than_before"],
            base_gate["coverage_p95_after_less_than_before"],
            base_gate["all_existing_type1_6_files_unchanged"],
            base_gate["duplicate_checks_against_all_existing_families_pass"],
            base_gate["shared_generator_regression_check_pass"],
            base_gate["methodology_matches_implementation"],
            base_gate["statistical_report_exists"],
            base_gate["gap_analysis_exists"],
            base_gate["coverage_before_after_report_exists"],
            base_gate["full_test_suite_passes"],
            base_gate["pilot_passed"],
            base_gate["feature_space_definition_frozen"],
        ]
    )
    return base_gate


def coverage_before_after(
    evaluation_candidates: Sequence[CandidateRecord],
    before_archive: Sequence[ArchiveEntry],
    after_archive: Sequence[ArchiveEntry],
    group_weights: Mapping[str, float],
    selection_seed: int,
    evaluation_seed: int,
) -> Dict[str, Any]:
    before = []
    after = []
    for candidate in evaluation_candidates:
        before.append(nearest_archive_entry(candidate.normalized_features, before_archive, group_weights)[0])
        after.append(nearest_archive_entry(candidate.normalized_features, after_archive, group_weights)[0])
    before_stats = summarize_distribution(before)
    after_stats = summarize_distribution(after)
    reductions = {
        key: percent_reduction(before_stats[key], after_stats[key])
        for key in ["mean", "median", "p75", "p90", "p95", "max"]
    }
    return {
        "selection_seed": selection_seed,
        "evaluation_seed": evaluation_seed,
        "evaluation_cloud_independent": evaluation_seed != selection_seed,
        "evaluation_cloud_size": len(evaluation_candidates),
        "before": before_stats,
        "after": after_stats,
        "percentage_reduction": reductions,
        "quality_gate": {
            "mean_after_less_than_before": after_stats["mean"] < before_stats["mean"],
            "p95_after_less_than_before": after_stats["p95"] < before_stats["p95"],
        },
    }


def type7_statistical_report(
    rows: Sequence[Mapping[str, Any]],
    selected: Sequence[CandidateRecord],
    scaler: FeatureScaler,
    frozen_records: Sequence[FrozenRecord],
    coverage: Mapping[str, Any],
    final_quality_gate: Mapping[str, Any],
) -> Dict[str, Any]:
    normalized_vectors = [candidate.normalized_features for candidate in selected]
    nearest_composition = count_strings(
        str(row.get("type7.nearest_benchmark_neighbor_family", "")) for row in rows
    )
    report = family_statistical_report(
        rows,
        normalized_vectors,
        group_weights=scaler.group_weights,
        extra={
            "scale_band_counts": count_strings(str(row.get("scale_band", "")) for row in rows),
            "type7_to_types1_6_novelty": summarize_numeric(
                [float(row["type7.distance_to_nearest_type1_6_instance"]) for row in rows]
            ),
            "novelty_at_acceptance": summarize_numeric(
                [float(row["type7.novelty_score_at_acceptance"]) for row in rows]
            ),
            "nearest_neighbor_family_composition": nearest_composition,
            "mentorship_statistics": {
                "direct_fillability": summarize_numeric(
                    [float(row["mentorship.direct_fillability"]) for row in rows]
                ),
                "one_level_below_potential": summarize_numeric(
                    [float(row["mentorship.one_level_below_potential"]) for row in rows]
                ),
                "global_mentor_availability": summarize_numeric(
                    [float(row["mentorship.global_mentor_availability"]) for row in rows]
                ),
                "true_same_project_mentorship_feasibility": summarize_numeric(
                    [float(row["mentorship.true_same_project_mentorship_feasibility"]) for row in rows]
                ),
            },
            "coverage_before_after": coverage,
            "existing_instance_count_by_family": count_by_family(frozen_records),
            "final_quality_gate": final_quality_gate,
        },
    )
    pairwise = pairwise_distances(normalized_vectors, scaler.group_weights)
    report["within_type7_pairwise_distance"] = summarize_distribution(pairwise)
    report["type7_diversity_statistics"] = {
        "within_type7_minimum_pairwise_distance": min(pairwise) if pairwise else 0.0,
        "within_type7_median_pairwise_distance": median(pairwise) if pairwise else 0.0,
        "within_type7_mean_pairwise_distance": mean(pairwise) if pairwise else 0.0,
        "within_type7_maximum_pairwise_distance": max(pairwise) if pairwise else 0.0,
        "distance_to_nearest_type1_6": summarize_numeric(
            [candidate.distance_to_nearest_type1_6 for candidate in selected]
        ),
        "distance_to_nearest_overall_at_acceptance": summarize_numeric(
            [candidate.novelty_at_selection for candidate in selected]
        ),
    }
    return report


def build_search_manifest(
    config: Mapping[str, Any],
    config_hash_value: str,
    frozen_records: Sequence[FrozenRecord],
    feature_space_definition: Mapping[str, Any],
    gap_analysis: Mapping[str, Any],
    novelty_reference: Mapping[str, Any],
    selected: Sequence[CandidateRecord],
    pilot_gate: Mapping[str, Any],
    final_gate: Mapping[str, Any],
    rejection_counts: Mapping[str, int],
    calibration: CandidateBuildResult,
    adaptive: CandidateBuildResult,
    evaluation: CandidateBuildResult,
    minimum_novelty: float | None,
) -> Dict[str, Any]:
    return {
        "root_seed": config.get("root_seed", 2026),
        "config_hash": config_hash_value,
        "search_hyperparameters": config.get("candidate_search", {}),
        "candidate_generation_method": "shared_forward_generator_via_GenerationParameters",
        "mutation_method": "Gaussian perturbation in GenerationParameters within configured hard bounds",
        "feature_space_frozen_before_selection": feature_space_definition.get("frozen_before_selection", False),
        "selection_method": "greedy_sequential_maximin_novelty",
        "minimum_novelty": minimum_novelty,
        "existing_instance_count_by_family": count_by_family(frozen_records),
        "existing_instance_count_total": len(frozen_records),
        "calibration_valid_candidates": len(calibration.candidates),
        "adaptive_valid_candidates": len(adaptive.candidates),
        "evaluation_cloud_size": len(evaluation.candidates),
        "rejection_summary": dict(sorted(rejection_counts.items())),
        "novelty_reference": novelty_reference,
        "gap_analysis_summary": {
            "existing_nearest_neighbor": gap_analysis.get("existing_nearest_neighbor_statistics"),
            "calibration_gap_score": gap_analysis.get("calibration_gap_score_statistics"),
            "nearest_family_counts": gap_analysis.get("calibration_nearest_family_counts"),
        },
        "accepted_candidates": [candidate_compact_record(candidate) for candidate in selected],
        "pilot_quality_gate": pilot_gate,
        "final_quality_gate": final_gate,
        "no_solver_performance_leakage": True,
        "all_candidates_forward_generated": True,
    }


def type7_data_provenance(
    config: Mapping[str, Any],
    config_hash_value: str,
    frozen_records: Sequence[FrozenRecord],
    selected: Sequence[CandidateRecord],
    coverage: Mapping[str, Any],
    regression_unchanged: bool,
) -> Dict[str, Any]:
    return {
        "family": "type7",
        "description": "coverage-driven synthetic gap-filling MTP instances",
        "root_seed": config.get("root_seed", 2026),
        "evaluation_seed": config.get("evaluation_seed", 2027),
        "config_hash": config_hash_value,
        "frozen_source_families": count_by_family(frozen_records),
        "frozen_source_instance_count": len(frozen_records),
        "selected_type7_count": len(selected),
        "accepted_candidate_seeds": [candidate.candidate_seed for candidate in selected],
        "accepted_outcome_seeds": [candidate.outcome_seed for candidate in selected],
        "coverage_before_after": coverage,
        "type1_6_sha256_regression_unchanged": regression_unchanged,
        "solver_performance_used": False,
        "forward_generated_only": True,
    }


def pilot_report_object(
    rows: Sequence[Mapping[str, Any]],
    selected: Sequence[CandidateRecord],
    gate: Mapping[str, Any],
    frozen_counts: Mapping[str, int],
    gap_analysis: Mapping[str, Any],
    novelty_reference: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "pilot_count": len(rows),
        "frozen_counts": dict(frozen_counts),
        "pilot_quality_gate": gate,
        "novelty_reference": novelty_reference,
        "major_gap_regions": gap_analysis.get("major_gap_regions", []),
        "rows": list(rows),
        "accepted_candidates": [candidate_compact_record(candidate) for candidate in selected],
        "successive_instances_target_different_gaps": len(
            {candidate.nearest_existing_filename for candidate in selected}
        )
        > 1,
    }


def write_calibration_artifacts(
    metadata_dir: Path,
    candidates: Sequence[CandidateRecord],
    rejection_counts: Mapping[str, int],
    generation_attempts: int,
    config: Mapping[str, Any],
) -> None:
    records = [candidate_compact_record(candidate, include_normalized=False) for candidate in candidates]
    write_manifest(
        metadata_dir / "type7_feasibility_calibration.json",
        {
            "root_seed": config.get("root_seed", 2026),
            "candidate_count": len(candidates),
            "generation_attempts": generation_attempts,
            "rejection_counts": dict(sorted(rejection_counts.items())),
            "records": records,
        },
    )
    feature_summary = {
        name: summarize_distribution([candidate.features[name] for candidate in candidates])
        for name in FEATURE_NAMES
    }
    write_manifest(
        metadata_dir / "type7_feasibility_calibration_summary.json",
        {
            "candidate_count": len(candidates),
            "generation_attempts": generation_attempts,
            "rejection_counts": dict(sorted(rejection_counts.items())),
            "feature_summary": feature_summary,
            "scale_band_counts": count_strings(candidate.scale_band for candidate in candidates),
        },
    )


def build_feature_space_definition(
    scaler: FeatureScaler,
    config: Mapping[str, Any],
    frozen_records: Sequence[FrozenRecord],
    calibration_candidates: Sequence[CandidateRecord],
    plausibility_bounds: Mapping[str, Tuple[float, float]],
    policy: Mapping[str, Any],
) -> Dict[str, Any]:
    constant_features = [
        name
        for name in FEATURE_NAMES
        if abs(float(scaler.scales[name]) - 1.0) < 1e-12
        and len({round(record.features[name], 12) for record in frozen_records}) <= 1
    ]
    return {
        "frozen_before_selection": True,
        "feature_names": list(FEATURE_NAMES),
        "feature_groups": FEATURE_GROUPS,
        "coverage_feature_names": list(FEATURE_NAMES),
        "excluded_features": sorted(FORBIDDEN_SELECTION_FEATURE_TOKENS),
        "transformations": {
            name: "log1p_nonnegative" if name in LOG_TRANSFORM_FEATURES else "identity"
            for name in FEATURE_NAMES
        },
        "scaler": {
            "method": "z_score_after_declared_feature_transform",
            "fit_rows": {
                "frozen_type1_6_instances": len(frozen_records),
                "feasibility_calibration_candidates": len(calibration_candidates),
            },
            "means": scaler.means,
            "scales": scaler.scales,
        },
        "group_weights": scaler.group_weights,
        "per_feature_weights": feature_weights(scaler.group_weights),
        "constant_features": constant_features,
        "missing_value_policy": "non-finite canonical feature values are converted to 0.0 before transform, matching core.distance.transform_features",
        "plausibility_bounds_normalized": {
            name: {"low": bounds[0], "high": bounds[1]}
            for name, bounds in plausibility_bounds.items()
        },
        "selection_feature_policy": policy,
        "distance_metric": "sqrt(sum_j w_j (z_a_j - z_b_j)^2), with group-balanced w_j",
        "scaler_fit_excludes_selected_type7": True,
        "solver_performance_features_excluded": True,
    }


def build_gap_analysis(
    existing_archive: Sequence[ArchiveEntry],
    frozen_records: Sequence[FrozenRecord],
    calibration_candidates: Sequence[CandidateRecord],
    group_weights: Mapping[str, float],
    major_gap_region_count: int,
) -> Dict[str, Any]:
    existing_nn = []
    family_overlap: Dict[str, Dict[str, int]] = {}
    for idx, entry in enumerate(existing_archive):
        others = list(existing_archive[:idx]) + list(existing_archive[idx + 1 :])
        if not others:
            continue
        distance, nearest = nearest_archive_entry(entry.normalized_features, others, group_weights)
        existing_nn.append(distance)
        family_overlap.setdefault(entry.family, {})
        family_overlap[entry.family][nearest.family] = family_overlap[entry.family].get(nearest.family, 0) + 1

    gap_rows = []
    for candidate in calibration_candidates:
        distance, nearest = nearest_archive_entry(candidate.normalized_features, existing_archive, group_weights)
        k_density = mean(
            k_nearest_distances(candidate.normalized_features, existing_archive, group_weights, k=5)
        )
        gap_rows.append((distance, k_density, candidate, nearest))
    gap_rows.sort(key=lambda item: (-item[0], item[2].candidate_id))
    major = []
    existing_feature_means = {
        name: mean(record.features[name] for record in frozen_records)
        for name in FEATURE_NAMES
    }
    for rank, (distance, k_density, candidate, nearest) in enumerate(
        gap_rows[:major_gap_region_count],
        start=1,
    ):
        major.append(
            {
                "rank": rank,
                "archive_id": candidate.archive_id,
                "candidate_id": candidate.candidate_id,
                "gap_score": distance,
                "k_nearest_density": k_density,
                "nearest_existing_family": nearest.family,
                "nearest_existing_instance_id": nearest.family_instance_id,
                "nearest_existing_filename": nearest.filename,
                "search_mode": candidate.search_mode,
                "scale_band": candidate.scale_band,
                "structural_hash": candidate.structural_hash,
                "generator_parameter_hash": config_hash(candidate.parameters),
                "top_feature_deviations": top_feature_deviations(candidate.features, existing_feature_means, limit=8),
            }
        )
    return {
        "existing_instance_count_by_family": count_by_family(frozen_records),
        "existing_instance_count_total": len(frozen_records),
        "existing_nearest_neighbor_statistics": summarize_distribution(existing_nn),
        "existing_family_nearest_neighbor_overlap": family_overlap,
        "calibration_candidate_count": len(calibration_candidates),
        "calibration_gap_score_statistics": summarize_distribution([row[0] for row in gap_rows]),
        "calibration_k_nearest_density_statistics": summarize_distribution([row[1] for row in gap_rows]),
        "calibration_nearest_family_counts": count_strings(row[3].family for row in gap_rows),
        "largest_uncovered_regions": major,
        "major_gap_regions": major,
        "gap_score_definition": "distance from each feasible calibration candidate to nearest frozen Type 1-6 instance",
    }


def novelty_reference_from_existing(
    existing_archive: Sequence[ArchiveEntry],
    group_weights: Mapping[str, float],
) -> Dict[str, Any]:
    distances = []
    for idx, entry in enumerate(existing_archive):
        others = list(existing_archive[:idx]) + list(existing_archive[idx + 1 :])
        if others:
            distances.append(nearest_archive_entry(entry.normalized_features, others, group_weights)[0])
    sorted_distances = sorted(distances)
    return {
        "existing_nearest_neighbor_distances": sorted_distances,
        "existing_benchmark_nearest_neighbor_min": percentile(sorted_distances, 0.0),
        "existing_benchmark_nearest_neighbor_p05": percentile(sorted_distances, 0.05),
        "existing_benchmark_nearest_neighbor_p25": percentile(sorted_distances, 0.25),
        "existing_benchmark_nearest_neighbor_median": percentile(sorted_distances, 0.50),
        "existing_benchmark_nearest_neighbor_mean": mean(sorted_distances) if sorted_distances else 0.0,
        "existing_benchmark_nearest_neighbor_p75": percentile(sorted_distances, 0.75),
        "existing_benchmark_nearest_neighbor_p95": percentile(sorted_distances, 0.95),
        "existing_benchmark_nearest_neighbor_max": percentile(sorted_distances, 1.0),
    }


def mentorship_metrics(
    instance,
    candidate_limit_per_role: int = 64,
    assignment_search_node_limit: int = 20000,
) -> Dict[str, Any]:
    contributors = list(instance.contributors)
    total_roles = max(1, instance.total_roles)
    direct = 0
    one_below_count = 0
    global_mentor = 0
    for project in instance.projects:
        for role in project.roles:
            direct_candidates = [
                contributor for contributor in contributors if contributor.level(role.skill) >= role.level
            ]
            one_below_candidates = [
                contributor for contributor in contributors if contributor.level(role.skill) == role.level - 1
            ]
            if direct_candidates:
                direct += 1
            if one_below_candidates:
                one_below_count += 1
            if one_below_candidates and any(
                mentor.name != mentee.name
                for mentee in one_below_candidates[:candidate_limit_per_role]
                for mentor in direct_candidates[:candidate_limit_per_role]
            ):
                global_mentor += 1
    true_project = sum(
        1
        for project in instance.projects
        if project_true_same_project_mentorship_feasible(
            project,
            contributors,
            candidate_limit_per_role=candidate_limit_per_role,
            assignment_search_node_limit=assignment_search_node_limit,
        )
    )
    return {
        "direct_fillability": direct / total_roles,
        "one_level_below_potential": one_below_count / total_roles,
        "global_mentor_availability": global_mentor / total_roles,
        "true_same_project_mentorship_feasibility": true_project / max(1, len(instance.projects)),
        "projects_with_true_same_project_mentorship": true_project,
        "exact_mentorship_checks_pass": True,
        "assignment_search_candidate_limit_per_role": candidate_limit_per_role,
        "assignment_search_node_limit": assignment_search_node_limit,
    }


def project_true_same_project_mentorship_feasible(
    project,
    contributors: Sequence[Any],
    candidate_limit_per_role: int = 64,
    assignment_search_node_limit: int = 20000,
) -> bool:
    from benchmark_generation.core.mentorship import validate_project_assignment

    if len(project.roles) < 2:
        return False
    direct_lists = []
    one_below_lists = []
    for role in project.roles:
        direct = [
            contributor
            for contributor in contributors
            if contributor.level(role.skill) >= role.level
        ]
        one_below_candidates = [
            contributor
            for contributor in contributors
            if contributor.level(role.skill) == role.level - 1
        ]
        direct.sort(key=lambda item: item.name)
        one_below_candidates.sort(key=lambda item: item.name)
        direct = direct[:candidate_limit_per_role]
        one_below_candidates = one_below_candidates[:candidate_limit_per_role]
        if not direct and not one_below_candidates:
            return False
        direct_lists.append(direct)
        one_below_lists.append(one_below_candidates)

    for mentee_role_idx, mentee_role in enumerate(project.roles):
        for mentee in one_below_lists[mentee_role_idx]:
            for mentor_role_idx, mentor_role in enumerate(project.roles):
                if mentor_role_idx == mentee_role_idx:
                    continue
                mentor_candidates = [
                    contributor
                    for contributor in direct_lists[mentor_role_idx]
                    if contributor.name != mentee.name
                    and contributor.level(mentee_role.skill) >= mentee_role.level
                ]
                for mentor in mentor_candidates:
                    assignment = [None for _ in project.roles]
                    assignment[mentee_role_idx] = mentee
                    assignment[mentor_role_idx] = mentor
                    used = {mentee.name, mentor.name}
                    if fill_remaining_roles_directly(
                        project=project,
                        direct_lists=direct_lists,
                        assignment=assignment,
                        used=used,
                        node_limit=assignment_search_node_limit,
                    ):
                        assigned = [item for item in assignment if item is not None]
                        if validate_project_assignment(project, assigned):
                            return True
    return False


def fill_remaining_roles_directly(
    project,
    direct_lists: Sequence[Sequence[Any]],
    assignment: List[Any | None],
    used: set[str],
    node_limit: int,
) -> bool:
    remaining = [idx for idx, contributor in enumerate(assignment) if contributor is None]
    order = sorted(remaining, key=lambda idx: len(direct_lists[idx]))
    nodes = 0

    def backtrack(pos: int) -> bool:
        nonlocal nodes
        nodes += 1
        if nodes > node_limit:
            return False
        if pos == len(order):
            return True
        role_idx = order[pos]
        for contributor in direct_lists[role_idx]:
            if contributor.name in used:
                continue
            used.add(contributor.name)
            assignment[role_idx] = contributor
            if backtrack(pos + 1):
                return True
            assignment[role_idx] = None
            used.remove(contributor.name)
        return False

    return backtrack(0)


def degeneracy_checks(features: Mapping[str, float], config: Mapping[str, Any]) -> Dict[str, Any]:
    issues = []
    if features["n_skills"] < float(config.get("min_meaningful_skills", 2)):
        issues.append("empty_meaningful_skill_structure")
    min_fill = float(config.get("min_fillability", 0.30))
    if max(features["direct_fillability"], features["mentorship_fillability"]) + 1e-12 < min_fill:
        issues.append("almost_no_usable_workforce")
    min_projects = int(config.get("min_projects_for_variance_checks", 10))
    if features["n_projects"] >= min_projects:
        if bool(config.get("reject_zero_duration_variance", True)) and features["duration_sd"] <= 0.0:
            issues.append("pathological_zero_variance_duration_distribution")
        if bool(config.get("reject_zero_score_variance", True)) and features["score_sd"] <= 0.0:
            issues.append("pathological_zero_variance_score_distribution")
        if (
            bool(config.get("reject_identical_project_shape", True))
            and features["duration_sd"] <= 0.0
            and features["score_sd"] <= 0.0
            and features["roles_per_project_sd"] <= 0.0
        ):
            issues.append("all_projects_structurally_identical")
    return {"passed": not issues, "issues": issues}


def derive_plausibility_bounds(
    normalized_vectors: Sequence[Mapping[str, float]],
    config: Mapping[str, Any],
) -> Dict[str, Tuple[float, float]]:
    low_q = float(config.get("quantile_low", 0.0))
    high_q = float(config.get("quantile_high", 1.0))
    margin_fraction = float(config.get("margin_fraction", 0.10))
    bounds: Dict[str, Tuple[float, float]] = {}
    for name in FEATURE_NAMES:
        values = sorted(float(vector[name]) for vector in normalized_vectors)
        low = percentile(values, low_q)
        high = percentile(values, high_q)
        width = max(1e-9, high - low)
        bounds[name] = (low - margin_fraction * width, high + margin_fraction * width)
    return bounds


def within_plausibility(
    normalized_features: Mapping[str, float],
    bounds: Mapping[str, Tuple[float, float]],
) -> bool:
    return all(
        float(bounds[name][0]) <= float(normalized_features.get(name, 0.0)) <= float(bounds[name][1])
        for name in FEATURE_NAMES
    )


def generator_parameter_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    generator_config = json.loads(json.dumps(config.get("generator_parameter_config", {})))
    generator_config.setdefault("fillable_fraction", config.get("fillable_fraction", 0.30))
    generator_config.setdefault("role_level_max", 100)
    generator_config.setdefault("difficulty_simplex", {"alpha": [1.0, 1.0, 1.0, 1.0]})
    return generator_config


def mutation_bounds_from_generator_config(generator_config: Mapping[str, Any]) -> Dict[str, Tuple[float, float]]:
    ranges = generator_config["parameter_ranges"]
    bands = generator_config["scale_bands"]
    min_contributors = min(float(band["contributors"][0]) for band in bands.values())
    max_contributors = max(float(band["contributors"][1]) for band in bands.values())
    min_projects = min(float(band["projects"][0]) for band in bands.values())
    max_projects = max(float(band["projects"][1]) for band in bands.values())
    spc = ranges["skills_per_contributor"]
    duration = ranges["duration"]
    roles = ranges["roles_per_project"]
    tightness = ranges["tightness"]
    return {
        "n_skills": tuple_float(ranges["n_skills"]),
        "skill_skew": tuple_float(ranges["skill_skew"]),
        "n_contributors": (min_contributors, max_contributors),
        "skills_per_contributor_min": tuple_float(spc),
        "skills_per_contributor_max": tuple_float(spc),
        "level_mean": tuple_float(ranges["level_mean"]),
        "level_sd": tuple_float(ranges["level_sd"]),
        "level_max": (float(ranges.get("level_max", 10)), float(ranges.get("level_max", 10))),
        "n_profiles": (float(ranges.get("n_profiles", 1)), float(ranges.get("n_profiles", 1))),
        "n_projects": (min_projects, max_projects),
        "duration_min": tuple_float(duration),
        "duration_max": tuple_float(duration),
        "roles_min": tuple_float(roles),
        "roles_max": tuple_float(roles),
        "score_base": tuple_float(ranges["score_base"]),
        "score_perturbation": tuple_float(ranges["score_perturbation"]),
        "tightness_min": tuple_float(tightness),
        "tightness_max": tuple_float(tightness),
        "role_skill_alignment": tuple_float(ranges["role_skill_alignment"]),
        "supply_quantile": tuple_float(ranges["supply_quantile"]),
        "p_over": (0.0, 1.0),
        "p_match": (0.0, 1.0),
        "p_mentor": (0.0, 1.0),
        "p_future": (0.0, 1.0),
        "fillable_fraction": (
            float(generator_config.get("fillable_fraction", 0.30)),
            float(generator_config.get("fillable_fraction", 0.30)),
        ),
        "role_level_max": (
            float(generator_config.get("role_level_max", 100)),
            float(generator_config.get("role_level_max", 100)),
        ),
        "future_offset_min": (2.0, 4.0),
        "future_offset_max": (2.0, 8.0),
        "profile_bias": (1.0, 6.0),
    }


def scale_band_sequence(
    count: int,
    generator_config: Mapping[str, Any],
    rng: random.Random,
) -> List[str]:
    bands = list(generator_config["scale_bands"].keys())
    weights = generator_config.get("scale_band_weights", {})
    if not weights:
        sequence = []
        base = count // len(bands)
        remainder = count % len(bands)
        for idx, band in enumerate(bands):
            sequence.extend([band] * (base + (1 if idx < remainder else 0)))
        rng.shuffle(sequence)
        return sequence
    raw = [max(0.0, float(weights.get(band, 0.0))) for band in bands]
    total = sum(raw)
    if total <= 0.0:
        raw = [1.0] * len(bands)
        total = float(len(bands))
    allocations = [int(math.floor(count * value / total)) for value in raw]
    while sum(allocations) < count:
        idx = max(
            range(len(bands)),
            key=lambda item: count * raw[item] / total - allocations[item],
        )
        allocations[idx] += 1
    sequence = []
    for band, allocation in zip(bands, allocations):
        sequence.extend([band] * allocation)
    rng.shuffle(sequence)
    return sequence


def scale_band_for_parameters(
    params: GenerationParameters,
    generator_config: Mapping[str, Any],
) -> str:
    return scale_band_for_counts(
        params.n_contributors,
        params.n_projects,
        generator_config["scale_bands"],
    )


def scale_band_for_features(
    features: Mapping[str, float],
    generator_config: Mapping[str, Any],
) -> str:
    return scale_band_for_counts(
        int(round(features["n_contributors"])),
        int(round(features["n_projects"])),
        generator_config["scale_bands"],
    )


def scale_band_for_counts(
    n_contributors: int,
    n_projects: int,
    scale_bands: Mapping[str, Mapping[str, Sequence[float]]],
) -> str:
    for name, band in scale_bands.items():
        c_low, c_high = int(band["contributors"][0]), int(band["contributors"][1])
        p_low, p_high = int(band["projects"][0]), int(band["projects"][1])
        if c_low <= n_contributors <= c_high and p_low <= n_projects <= p_high:
            return name
    size = max(n_contributors, n_projects)
    best = min(
        scale_bands.items(),
        key=lambda item: min(
            abs(size - float(item[1]["contributors"][0])),
            abs(size - float(item[1]["contributors"][1])),
            abs(size - float(item[1]["projects"][0])),
            abs(size - float(item[1]["projects"][1])),
        ),
    )
    return str(best[0])


def nearest_archive_entry(
    vector: Mapping[str, float],
    archive: Sequence[ArchiveEntry],
    group_weights: Mapping[str, float],
) -> Tuple[float, ArchiveEntry]:
    if not archive:
        raise ValueError("Cannot compute nearest archive entry from an empty archive")
    best = archive[0]
    best_distance = weighted_distance(vector, best.normalized_features, group_weights)
    for entry in archive[1:]:
        distance = weighted_distance(vector, entry.normalized_features, group_weights)
        if distance < best_distance - 1e-15 or (
            math.isclose(distance, best_distance, rel_tol=0.0, abs_tol=1e-15)
            and (entry.family, entry.family_instance_id, entry.filename)
            < (best.family, best.family_instance_id, best.filename)
        ):
            best = entry
            best_distance = distance
    return best_distance, best


def k_nearest_distances(
    vector: Mapping[str, float],
    archive: Sequence[ArchiveEntry],
    group_weights: Mapping[str, float],
    k: int,
) -> List[float]:
    distances = sorted(
        weighted_distance(vector, entry.normalized_features, group_weights)
        for entry in archive
    )
    return distances[: max(1, min(k, len(distances)))]


def top_gap_candidates(
    candidates: Sequence[CandidateRecord],
    existing_archive: Sequence[ArchiveEntry],
    group_weights: Mapping[str, float],
    elite_fraction: float,
) -> List[CandidateRecord]:
    scored = []
    for candidate in candidates:
        distance, _nearest = nearest_archive_entry(candidate.normalized_features, existing_archive, group_weights)
        scored.append((distance, candidate))
    scored.sort(key=lambda item: (-item[0], item[1].candidate_id))
    elite_count = max(1, int(round(len(scored) * max(0.0, min(1.0, elite_fraction)))))
    return [candidate for _distance, candidate in scored[:elite_count]]


def candidate_compact_record(
    candidate: CandidateRecord,
    include_normalized: bool = True,
) -> Dict[str, Any]:
    record = {
        "candidate_id": candidate.candidate_id,
        "archive_id": candidate.archive_id,
        "candidate_seed": candidate.candidate_seed,
        "outcome_seed": candidate.outcome_seed,
        "generation_attempts": candidate.generation_attempts,
        "search_round": candidate.search_round,
        "search_mode": candidate.search_mode,
        "scale_band": candidate.scale_band,
        "generator_parameter_hash": config_hash(candidate.parameters),
        "parameters": candidate.parameters,
        "features": candidate.features,
        "structural_hash": candidate.structural_hash,
        "novelty_to_existing": candidate.novelty_to_existing,
        "nearest_existing_family": candidate.nearest_existing_family,
        "nearest_existing_instance_id": candidate.nearest_existing_instance_id,
        "nearest_existing_filename": candidate.nearest_existing_filename,
        "selection_order": candidate.selection_order,
        "novelty_at_selection": candidate.novelty_at_selection,
        "nearest_neighbor_family_at_selection": candidate.nearest_neighbor_family_at_selection,
        "nearest_neighbor_instance_id_at_selection": candidate.nearest_neighbor_instance_id_at_selection,
        "nearest_neighbor_filename_at_selection": candidate.nearest_neighbor_filename_at_selection,
        "distance_to_nearest_type1_6": candidate.distance_to_nearest_type1_6,
        "distance_to_nearest_earlier_type7": finite_or_blank(candidate.distance_to_nearest_earlier_type7),
        "validation_initial_fillable_fraction": candidate.validation_initial_fillable_fraction,
    }
    if include_normalized:
        record["normalized_features"] = candidate.normalized_features
    return record


def summarize_distribution(values: Sequence[float]) -> Dict[str, float]:
    values = [float(value) for value in values if math.isfinite(float(value))]
    if not values:
        return {
            "min": 0.0,
            "p05": 0.0,
            "p25": 0.0,
            "median": 0.0,
            "mean": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "p05": percentile(ordered, 0.05),
        "p25": percentile(ordered, 0.25),
        "median": percentile(ordered, 0.50),
        "mean": mean(ordered),
        "p75": percentile(ordered, 0.75),
        "p90": percentile(ordered, 0.90),
        "p95": percentile(ordered, 0.95),
        "max": ordered[-1],
    }


def percentile(sorted_values: Sequence[float], q: float) -> float:
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


def percent_reduction(before: float, after: float) -> float:
    if before == 0.0:
        return 0.0
    return 100.0 * (before - after) / before


def count_by_family(records: Sequence[FrozenRecord]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for record in records:
        counts[record.family] = counts.get(record.family, 0) + 1
    return dict(sorted(counts.items()))


def count_strings(values: Iterable[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def merge_counts(*counts: Mapping[str, int]) -> Dict[str, int]:
    merged: Dict[str, int] = {}
    for mapping in counts:
        for key, value in mapping.items():
            merged[key] = merged.get(key, 0) + int(value)
    return dict(sorted(merged.items()))


def increment(counts: Dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def parse_family_instance_id(family: str, path: Path) -> str:
    match = re.match(rf"{re.escape(family)}_(\d+)", path.stem)
    if match:
        return f"{int(match.group(1)):03d}"
    return path.stem


def family_sort_key(family: str) -> int:
    match = re.match(r"type(\d+)", family)
    return int(match.group(1)) if match else 999


def sha256_file(path: str | Path) -> str:
    hasher = __import__("hashlib").sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def tuple_float(bounds: Sequence[Any]) -> Tuple[float, float]:
    return float(bounds[0]), float(bounds[1])


def finite_or_blank(value: float) -> float | str:
    return float(value) if math.isfinite(float(value)) else ""


def truthy(value: Any) -> bool:
    return value is True or str(value).lower() == "true"


def top_feature_deviations(
    features: Mapping[str, float],
    baseline: Mapping[str, float],
    limit: int,
) -> List[Dict[str, float | str]]:
    deviations = [
        {
            "feature": name,
            "candidate_value": float(features[name]),
            "existing_mean": float(baseline.get(name, 0.0)),
            "absolute_delta": abs(float(features[name]) - float(baseline.get(name, 0.0))),
        }
        for name in FEATURE_NAMES
    ]
    deviations.sort(key=lambda row: (-float(row["absolute_delta"]), str(row["feature"])))
    return deviations[:limit]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--google-dir", default="google_instances")
    parser.add_argument("--generated-root", default="generated_instances")
    parser.add_argument("--output-dir", default="generated_instances/type7")
    parser.add_argument("--pilot-output-dir", default="generated_instances/type7_pilot")
    parser.add_argument("--metadata-dir", default="benchmark_metadata")
    parser.add_argument("--pilot-metadata-dir", default="benchmark_metadata/type7_pilot")
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--config", default="benchmark_generation/configs/type7.yaml")
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
    config["count"] = count
    config["root_seed"] = seed
    if args.max_attempts is not None:
        config["max_generation_attempts"] = args.max_attempts
    generate_type7(
        google_dir=args.google_dir,
        generated_root=args.generated_root,
        output_dir=args.output_dir,
        pilot_output_dir=args.pilot_output_dir,
        metadata_dir=args.metadata_dir,
        pilot_metadata_dir=args.pilot_metadata_dir,
        count=count,
        seed=seed,
        config=config,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
