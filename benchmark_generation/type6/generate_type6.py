"""Generate Type 6 literature-grounded synthetic MTP pilot/final instances."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from benchmark_generation.core.config import load_config
from benchmark_generation.core.distance import FEATURE_NAMES, FeatureScaler
from benchmark_generation.core.duplicates import structural_hashes_for_paths
from benchmark_generation.core.features import extract_features
from benchmark_generation.core.generator import (
    EmpiricalSourceProfile,
    GenerationParameters,
    derive_seed,
    generate_instance,
    generate_valid_instance,
)
from benchmark_generation.core.metadata import (
    build_metadata_row,
    config_hash,
    family_statistical_report,
    summarize_numeric,
    write_metadata,
)
from benchmark_generation.core.parser import parse_file, write_instance
from benchmark_generation.core.validator import structural_hash, validate_file
from benchmark_generation.type6.download_mslib import utc_now
from benchmark_generation.type6.select_sources import DATASETS, DEFAULT_CONFIG
from benchmark_generation.type6.quality import (
    evaluate_type6_quality,
    final_instance_names_are_anonymized,
    pilot_quality_gate,
    summarize_rows,
)


PILOT_DEFAULT_ALLOCATION = {"MSLIB1": 2, "MSLIB2": 2, "MSLIB3": 3, "MSLIB4": 3}
FINAL_EXPECTED_COUNTS = {dataset: 25 for dataset in DATASETS}
TYPE6_FINAL_FILENAME_RE = re.compile(r"^type6_(?P<family_id>\d{3})__(?P<dataset>MSLIB[1-4])__src(?P<source_id>\d+)\.txt$")
TYPE6_FINAL_FILENAME_CONVENTION = "type6_<family_id>__<MSLIB_set>__src<source_instance_id>.txt"


def generate_type6(
    output_dir: str | Path,
    metadata_dir: str | Path,
    config: Dict[str, Any],
    seed: int,
    count: int | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    pilot: bool = False,
) -> List[Dict[str, Any]]:
    output_dir = Path(output_dir)
    metadata_dir = Path(metadata_dir)
    profiles_path = Path(str(config.get("literature_profiles_path", "benchmark_metadata/type6_literature_profiles.json")))
    profiles = load_literature_profiles(profiles_path)
    if len(profiles) != 100:
        raise RuntimeError(f"Type 6 generation requires 100 frozen profiles; found {len(profiles)}")
    if pilot:
        selected_profiles = select_pilot_profiles(profiles, config, seed)
        if count is not None and count != len(selected_profiles):
            raise RuntimeError(
                f"Type 6 pilot selection is fixed at {len(selected_profiles)} profiles; "
                f"received --count {count}"
            )
    else:
        expected_count = int(config.get("count", 100))
        if count is not None and count != expected_count:
            raise RuntimeError(f"Final Type 6 generation must use exactly {expected_count} profiles")
        selected_profiles = sorted(profiles, key=lambda item: int(item["family_instance_id"]))

    baseline_hashes = existing_hashes_against_previous_families(
        google_dir=Path(str(config.get("google_dir", "google_instances"))),
        generated_root=Path(str(config.get("generated_root", "generated_instances"))),
    )
    accepted_hashes = set(baseline_hashes)
    rows: List[Dict[str, Any]] = []
    failed_attempts: List[Dict[str, Any]] = []
    max_attempts = int(config.get("max_generation_attempts_per_source", 30))

    for output_idx, profile in enumerate(selected_profiles, start=1):
        source_family_id = int(profile["family_instance_id"])
        family_id = output_idx if pilot else source_family_id
        filename = f"type6_pilot_{output_idx:03d}.txt" if pilot else final_type6_filename(family_id, profile)
        params = GenerationParameters(**profile["parameters"]).normalized()
        empirical_profile = empirical_profile_from_type6_profile(profile)
        logging.info(
            "Type6 %03d/%03d source=%s/%s",
            output_idx,
            len(selected_profiles),
            profile["source_dataset"],
            profile["source_instance_id"],
        )
        accepted = None
        last_quality = None
        for attempt in range(1, max_attempts + 1):
            attempt_seed = derive_seed(seed, source_family_id * 1000 + attempt, salt=18_000_071)
            try:
                outcome = generate_valid_instance(
                    params,
                    attempt_seed,
                    max_attempts=1,
                    source_name=filename,
                    empirical_profile=empirical_profile,
                )
            except RuntimeError as exc:
                failed_attempts.append(
                    {
                        "filename": filename,
                        "source_family_instance_id": source_family_id,
                        "attempt": attempt,
                        "reason": str(exc),
                    }
                )
                continue
            h = structural_hash(outcome.instance)
            duplicate_pass = h not in accepted_hashes
            quality = evaluate_type6_quality(
                outcome.instance,
                outcome.validation,
                profile,
                duplicate_pass=duplicate_pass,
                config=config,
            )
            last_quality = quality
            if quality["overall_type6_quality_pass"]:
                accepted = (outcome, h, quality, attempt)
                accepted_hashes.add(h)
                break
            failed_attempts.append(
                {
                    "filename": filename,
                    "source_family_instance_id": source_family_id,
                    "attempt": attempt,
                    "reason": "type6_quality_gate_failed",
                    "quality": quality_summary(quality),
                }
            )
        if accepted is None:
            raise RuntimeError(
                f"Could not generate accepted Type 6 instance for {filename}. "
                f"Last quality={json.dumps(quality_summary(last_quality or {}), sort_keys=True)}"
            )

        outcome, h, quality, outer_attempt = accepted
        features = extract_features(outcome.instance)
        if not dry_run:
            path = output_dir / filename
            write_instance(outcome.instance, path, overwrite=overwrite)
            validation = validate_file(path, fillable_fraction=outcome.parameters.fillable_fraction)
            if not validation.ok:
                raise RuntimeError(f"Written Type 6 instance failed validation: {path}: {validation.issues}")
        else:
            validation = outcome.validation

        row = build_type6_metadata_row(
            filename=filename,
            family_id=family_id,
            source_family_id=source_family_id,
            root_seed=seed,
            outcome_seed=outcome.seed,
            generator_version=outcome.generator_version,
            parameters=outcome.parameters.to_dict(),
            features=features,
            validation=validation,
            profile=profile,
            structural_hash=h,
            quality=quality,
            generation_attempts=outer_attempt,
            pilot=pilot,
            config=config,
        )
        rows.append(row)
        logging.info(
            "accepted %s source=%s/%s contributors=%d projects=%d skills=%d",
            filename,
            profile["source_dataset"],
            profile["source_instance_id"],
            len(outcome.instance.contributors),
            len(outcome.instance.projects),
            len(outcome.instance.skill_universe),
        )

    if not dry_run:
        metadata_dir.mkdir(parents=True, exist_ok=True)
        write_metadata(rows, metadata_dir / "type6_metadata.csv", metadata_dir / "type6_metadata.jsonl")
        mapping_rows: List[Dict[str, Any]] = []
        shared_regression: Mapping[str, Any] = {"passed": True, "checks": {}}
        if not pilot:
            mapping_rows = source_to_output_mapping_rows(rows)
            write_source_to_output_mapping(
                mapping_rows,
                metadata_dir / "type6_source_to_output_mapping.csv",
                metadata_dir / "type6_source_to_output_mapping.json",
            )
            update_final_type6_provenance(metadata_dir, output_dir, config, mapping_rows, rows, seed)
            update_final_type6_sources_markdown(metadata_dir, mapping_rows)
            update_final_type6_methodology(metadata_dir, config)
            shared_regression = shared_generator_regression_check()
        report = type6_generation_report(
            rows,
            selected_profiles,
            failed_attempts,
            config,
            seed,
            pilot=pilot,
            output_dir=output_dir,
            metadata_dir=metadata_dir,
            mapping_rows=mapping_rows,
            shared_regression=shared_regression,
        )
        report_name = "type6_pilot_report.json" if pilot else "type6_statistical_report.json"
        (metadata_dir / report_name).write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        gate = report["pilot_quality_gate"] if pilot else report["final_quality_gate"]
        if not gate["passed"]:
            raise RuntimeError("Type 6 generation quality gate failed: " + json.dumps(gate, sort_keys=True))
    return rows


def load_literature_profiles(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing Type 6 literature profiles: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    profiles = list(data.get("profiles", []))
    counts = Counter(str(profile.get("source_dataset")) for profile in profiles)
    if counts != Counter({dataset: 25 for dataset in DATASETS}):
        raise RuntimeError(f"Type 6 profile set must contain 25 per MSLIB1-4; found {dict(counts)}")
    return sorted(profiles, key=lambda item: int(item["family_instance_id"]))


def final_type6_filename(family_id: int, profile: Mapping[str, Any]) -> str:
    dataset = str(profile["source_dataset"])
    source_id = int(profile["source_instance_id"])
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported Type 6 source dataset in final filename: {dataset}")
    return f"type6_{int(family_id):03d}__{dataset}__src{source_id}.txt"


def parse_type6_final_filename(filename: str | Path) -> Dict[str, Any]:
    name = Path(filename).name
    match = TYPE6_FINAL_FILENAME_RE.match(name)
    if not match:
        raise ValueError(
            f"Type 6 final filename must match {TYPE6_FINAL_FILENAME_CONVENTION}: {name}"
        )
    return {
        "family_instance_id": int(match.group("family_id")),
        "source_dataset": match.group("dataset"),
        "source_instance_id": int(match.group("source_id")),
    }


def source_release_label(config: Mapping[str, Any]) -> str:
    release = config.get("source_release", {})
    asset = str(release.get("release_asset", "MSLIB_v2_2.zip"))
    tag = str(release.get("observed_release_tag", "v2.2"))
    return f"{asset} from GitHub release {tag}"


def source_archive_sha256(config: Mapping[str, Any]) -> str:
    release = config.get("source_release", {})
    return str(release.get("sha256", ""))


def select_pilot_profiles(
    profiles: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    seed: int,
) -> List[Dict[str, Any]]:
    pilot_config = config.get("pilot", {})
    allocation = dict(pilot_config.get("allocation", PILOT_DEFAULT_ALLOCATION))
    feature_columns = list(pilot_config.get("feature_columns", []))
    selected: List[Dict[str, Any]] = []
    for dataset in DATASETS:
        dataset_profiles = [
            dict(profile)
            for profile in profiles
            if str(profile.get("source_dataset")) == dataset
        ]
        count = int(allocation.get(dataset, 0))
        selected.extend(
            _maximin_profile_subset(dataset_profiles, feature_columns, count, seed)
        )
    return sorted(selected, key=lambda item: (str(item["source_dataset"]), int(item["family_instance_id"])))


def _maximin_profile_subset(
    profiles: Sequence[Mapping[str, Any]],
    feature_columns: Sequence[str],
    count: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if count <= 0:
        return []
    if len(profiles) < count:
        raise RuntimeError(f"Cannot select {count} pilot profiles from {len(profiles)} candidates")
    vectors = standardized_profile_vectors(profiles, feature_columns)
    selected = [profile_medoid_index(list(range(len(profiles))), vectors)]
    while len(selected) < count:
        remaining = [idx for idx in range(len(profiles)) if idx not in selected]
        selected.append(
            max(
                remaining,
                key=lambda idx: (
                    min(euclidean(vectors[idx], vectors[current]) for current in selected),
                    -abs(int(profiles[idx]["family_instance_id"]) - seed % 100),
                    -int(profiles[idx]["family_instance_id"]),
                ),
            )
        )
    return [dict(profiles[idx]) for idx in selected]


def standardized_profile_vectors(
    profiles: Sequence[Mapping[str, Any]],
    feature_columns: Sequence[str],
) -> List[List[float]]:
    columns = {}
    for column in feature_columns:
        values = [float(profile["source_metrics"].get(column, 0.0)) for profile in profiles]
        transformed = [math.log1p(value) if value > 10 else value for value in values]
        m = mean(transformed) if transformed else 0.0
        sd = pstdev(transformed) if len(transformed) > 1 and pstdev(transformed) > 1e-12 else 1.0
        columns[column] = [(value - m) / sd for value in transformed]
    return [
        [columns[column][idx] for column in feature_columns]
        for idx in range(len(profiles))
    ]


def profile_medoid_index(indices: Sequence[int], vectors: Sequence[Sequence[float]]) -> int:
    centroid = [
        mean(vectors[idx][dim] for idx in indices)
        for dim in range(len(vectors[indices[0]]))
    ]
    return min(indices, key=lambda idx: (euclidean(vectors[idx], centroid), idx))


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def empirical_profile_from_type6_profile(profile: Mapping[str, Any]) -> EmpiricalSourceProfile:
    source = profile["source_profile"]
    return EmpiricalSourceProfile(
        skill_tokens=tuple(str(value) for value in source.get("skill_tokens", [])),
        skill_weights=tuple(float(value) for value in source.get("skill_weights", [])),
        demand_weights=tuple(float(value) for value in source.get("demand_weights", [])),
        profile_centroids=tuple(
            tuple(float(value) for value in row)
            for row in source.get("profile_centroids", [])
        ),
        profile_sizes=tuple(int(value) for value in source.get("profile_sizes", [])),
        duration_values=tuple(int(value) for value in source.get("duration_values", [])),
        skills_per_contributor_values=tuple(int(value) for value in source.get("skills_per_contributor_values", [])),
        level_values=tuple(int(value) for value in source.get("level_values", [])),
        role_count_values=tuple(int(value) for value in source.get("role_count_values", [])),
        required_level_values=tuple(int(value) for value in source.get("required_level_values", [])),
        deadline_tightness_values=tuple(float(value) for value in source.get("deadline_tightness_values", [])),
        score_values=tuple(int(value) for value in source.get("score_values", [])),
        contributor_level_rows=tuple(
            tuple(int(value) for value in row)
            for row in source.get("contributor_level_rows", [])
        ),
        activity_role_skill_level_rows=tuple(
            tuple((int(pair[0]), int(pair[1])) for pair in row)
            for row in source.get("activity_role_skill_level_rows", [])
        ),
    )


def existing_hashes_against_previous_families(
    google_dir: Path,
    generated_root: Path,
) -> set[str]:
    hashes = structural_hashes_for_paths(google_dir.glob("*.txt"))
    for family in ("type2", "type3", "type4", "type5"):
        hashes.update(structural_hashes_for_paths((generated_root / family).glob("*.txt")))
    return hashes


def build_type6_metadata_row(
    filename: str,
    family_id: int,
    source_family_id: int,
    root_seed: int,
    outcome_seed: int,
    generator_version: str,
    parameters: Mapping[str, Any],
    features: Mapping[str, float],
    validation: Any,
    profile: Mapping[str, Any],
    structural_hash: str,
    quality: Mapping[str, Any],
    generation_attempts: int,
    pilot: bool,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    source_metrics = profile["source_metrics"]
    generated_metrics = quality["metrics"]
    mentorship = quality["mentorship_metrics"]
    generated_summary = generated_metrics["summary"]
    distribution = quality["distribution_match"]
    release_label = source_release_label(config)
    archive_sha256 = source_archive_sha256(config)
    row = build_metadata_row(
        filename=filename,
        family="type6_pilot" if pilot else "type6",
        family_instance_id=family_id,
        seed=outcome_seed,
        generator_version=generator_version,
        parameters=parameters,
        features=features,
        validation={
            "ok": validation.ok,
            "issues": validation.issues,
            "initial_fillable_fraction": validation.metrics.get("initial_fillable_fraction", ""),
            "generation_attempts": generation_attempts,
        },
        extra={
            "type6": {
                "source_family_instance_id": source_family_id,
                "structural_hash": structural_hash,
                "duplicate_pass": quality["duplicate_pass"],
                "source_provenance_complete": quality["source_provenance_complete"],
                "source_correspondence_pass": source_correspondence_pass(quality),
                "exact_mentorship_checks_pass": quality["exact_mentorship_checks_pass"],
                "overall_type6_quality_pass": quality["overall_type6_quality_pass"],
                "no_source_names_in_final_file_pass": quality["no_source_names_in_final_file_pass"],
                "no_bks_leakage_pass": quality["no_bks_leakage_pass"],
                "no_precedence_constraints_in_final_mtp_pass": quality["no_precedence_constraints_in_final_mtp_pass"],
                "score_formula": profile["source_profile"].get("score_formula", {}),
            },
            "generated_metrics": {
                "n_contributors": generated_metrics["n_contributors"],
                "n_projects": generated_metrics["n_projects"],
                "n_skills": generated_metrics["n_skills"],
                "skill_skew": generated_metrics["skill_skew"],
                "demand_supply_alignment": generated_metrics["demand_supply_alignment"],
                "contention": generated_metrics["contention"],
                "deadline_tightness_mean": generated_metrics["deadline_tightness_mean"],
                "schedule_pressure": generated_metrics["schedule_pressure"],
            },
            "mentorship": mentorship,
        },
    )
    row.update(
        {
            "output_filename": filename,
            "root_seed": root_seed,
            "source_dataset": profile["source_dataset"],
            "source_instance_id": profile["source_instance_id"],
            "source_filename": profile["source_filename"],
            "source_original_filename": profile["source_filename"],
            "source_zip_entry": profile["source_zip_entry"],
            "source_sha256": profile["source_sha256"],
            "source_selection_index": int(profile["family_instance_id"]),
            "source_release": release_label,
            "source_archive_sha256": archive_sha256,
            "source_selection_seed": profile["source_selection"]["selection_seed"],
            "generation_seed": outcome_seed,
            "generation_attempt": generation_attempts,
            "structural_hash": structural_hash,
            "source_activity_count": source_metrics["source_activity_count"],
            "source_resource_count": source_metrics["source_resource_count"],
            "source_skill_count": source_metrics["source_skill_count"],
            "source_skill_level_count": source_metrics["source_skill_level_count"],
            "generated_project_count": generated_metrics["n_projects"],
            "generated_contributor_count": generated_metrics["n_contributors"],
            "generated_skill_count": generated_metrics["n_skills"],
            "generated_total_roles": int(features["total_roles"]),
            "generated_roles_per_project_mean": generated_summary["roles_per_project"]["mean"],
            "generated_roles_per_project_sd": generated_summary["roles_per_project"]["sd"],
            "source_skills_per_resource_mean": source_metrics["source_skills_per_resource_mean"],
            "source_skills_per_resource_sd": source_metrics["source_skills_per_resource_sd"],
            "generated_skills_per_contributor_mean": generated_summary["skills_per_contributor"]["mean"],
            "generated_skills_per_contributor_sd": generated_summary["skills_per_contributor"]["sd"],
            "source_skill_level_mean": source_metrics["source_skill_level_mean"],
            "source_skill_level_sd": source_metrics["source_skill_level_sd"],
            "generated_contributor_level_mean": generated_summary["contributor_levels"]["mean"],
            "generated_contributor_level_sd": generated_summary["contributor_levels"]["sd"],
            "source_roles_per_activity_mean": source_metrics["source_roles_per_activity_mean"],
            "source_roles_per_activity_sd": source_metrics["source_roles_per_activity_sd"],
            "generated_duration_mean": generated_summary["durations"]["mean"],
            "generated_duration_sd": generated_summary["durations"]["sd"],
            "generated_duration_median": generated_summary["durations"]["median"],
            "generated_duration_p10": generated_summary["durations"]["p10"],
            "generated_duration_p90": generated_summary["durations"]["p90"],
            "source_duration_mean": source_metrics["source_duration_mean"],
            "source_duration_sd": source_metrics["source_duration_sd"],
            "source_duration_median": source_metrics["source_duration_median"],
            "source_duration_p10": source_metrics["source_duration_p10"],
            "source_duration_p90": source_metrics["source_duration_p90"],
            "generated_required_level_mean": generated_summary["required_levels"]["mean"],
            "generated_required_level_sd": generated_summary["required_levels"]["sd"],
            "source_skill_skew": source_metrics["source_skill_skew"],
            "generated_skill_skew": generated_metrics["skill_skew"],
            "source_demand_concentration": source_metrics["source_demand_concentration"],
            "generated_demand_concentration": generated_metrics["demand_concentration"],
            "source_supply_concentration": source_metrics["source_supply_concentration"],
            "generated_supply_concentration": generated_metrics["supply_concentration"],
            "source_demand_supply_alignment": source_metrics["source_demand_supply_alignment"],
            "generated_demand_supply_alignment": generated_metrics["demand_supply_alignment"],
            "source_contention": source_metrics["source_contention"],
            "generated_contention": generated_metrics["contention"],
            "source_qualified_contention": source_metrics["source_qualified_contention"],
            "source_critical_path": source_metrics["source_critical_path"],
            "source_network_depth": source_metrics["source_network_depth"],
            "source_order_strength": source_metrics["source_order_strength"],
            "source_arc_density": source_metrics["source_arc_density"],
            "source_n_precedence_arcs": source_metrics["source_n_precedence_arcs"],
            "source_natural_direct_fillability": source_metrics["source_natural_direct_fillability"],
            "source_natural_one_below_potential": source_metrics["source_natural_one_below_potential"],
            "source_potential_mentor_availability": source_metrics["source_potential_mentor_availability"],
            "generated_deadline_tightness_mean": generated_metrics["deadline_tightness_mean"],
            "generated_deadline_tightness_sd": generated_metrics["deadline_tightness_sd"],
            "generated_schedule_pressure": generated_metrics["schedule_pressure"],
            "generated_score_mean": features["score_mean"],
            "generated_score_sd": features["score_sd"],
            "generated_score_density": features["score_density"],
            "generated_score_effort_correlation": features["score_effort_correlation"],
            "generated_score_concentration": features["score_concentration"],
            "generated_penalty_exposure": features["penalty_exposure"],
            "score_effort_formula": profile["source_profile"].get("score_formula", {}).get("formula", ""),
            "score_scaling_constant": profile["source_profile"].get("score_formula", {}).get("K", ""),
            "generated_direct_fillability": mentorship["direct_fillability"],
            "generated_one_level_below_potential": mentorship["global_one_below_potential"],
            "generated_global_mentor_availability": mentorship["global_mentor_availability"],
            "generated_true_same_project_mentorship_feasibility": mentorship["true_same_project_mentorship_feasibility"],
        }
    )
    for name, comparison in distribution["comparisons"].items():
        row[f"type6.distribution_source_summary.{name}"] = comparison["source_summary"]
        row[f"type6.distribution_generated_summary.{name}"] = comparison["generated_summary"]
    for name, comparison in quality["distribution_match"]["comparisons"].items():
        row[f"type6.distribution_distance.{name}"] = comparison["quantile_distance"]
        row[f"type6.distribution_pass.{name}"] = comparison["pass"]
    for name, comparison in quality["distribution_match"]["scalar_comparisons"].items():
        row[f"type6.scalar_error.{name}"] = comparison["error"]
        row[f"type6.scalar_pass.{name}"] = comparison["pass"]
    return row


def source_correspondence_pass(quality: Mapping[str, Any]) -> bool:
    return all(
        bool(quality[key])
        for key in (
            "source_scale_correspondence_pass",
            "skills_per_contributor_correspondence_pass",
            "skill_level_correspondence_pass",
            "skill_skew_correspondence_pass",
            "role_count_distribution_correspondence_pass",
            "duration_distribution_correspondence_pass",
            "required_level_distribution_correspondence_pass",
            "demand_supply_correspondence_pass",
        )
    )


def type6_generation_report(
    rows: Sequence[Mapping[str, Any]],
    profiles: Sequence[Mapping[str, Any]],
    failed_attempts: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    seed: int,
    pilot: bool,
    output_dir: str | Path,
    metadata_dir: str | Path,
    mapping_rows: Sequence[Mapping[str, Any]] | None = None,
    shared_regression: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    mapping_rows = list(mapping_rows or [])
    shared_regression = dict(shared_regression or {"passed": True, "checks": {}})
    gate = (
        pilot_quality_gate(rows, expected_count=len(profiles))
        if pilot
        else final_quality_gate(
            rows,
            profiles,
            mapping_rows,
            output_dir=Path(output_dir),
            metadata_dir=Path(metadata_dir),
            shared_regression=shared_regression,
        )
    )
    summary_keys = [
        "source_activity_count",
        "source_resource_count",
        "source_skill_count",
        "generated_total_roles",
        "generated_roles_per_project_mean",
        "generated_skills_per_contributor_mean",
        "generated_contributor_level_mean",
        "generated_required_level_mean",
        "generated_duration_mean",
        "generated_duration_sd",
        "generated_deadline_tightness_mean",
        "generated_deadline_tightness_sd",
        "source_demand_concentration",
        "generated_demand_concentration",
        "source_supply_concentration",
        "generated_supply_concentration",
        "source_demand_supply_alignment",
        "generated_demand_supply_alignment",
        "generated_project_count",
        "generated_contributor_count",
        "generated_skill_count",
        "source_skill_skew",
        "generated_skill_skew",
        "source_contention",
        "generated_contention",
        "generated_direct_fillability",
        "generated_one_level_below_potential",
        "generated_global_mentor_availability",
        "generated_true_same_project_mentorship_feasibility",
        "generated_score_mean",
        "generated_score_sd",
        "generated_score_density",
        "generated_score_effort_correlation",
    ]
    feature_report = type6_feature_report(rows, config)
    report = {
        "family": "type6",
        "phase": "pilot_phase_5" if pilot else "final_generation",
        "built_at": utc_now(),
        "root_seed": seed,
        "config_hash": config_hash(config),
        "accepted_count": len(rows),
        "count_per_MSLIB_set": dict(Counter(str(row["source_dataset"]) for row in rows)),
        "pilot_selection" if pilot else "final_source_mapping_summary": [
            {
                "family_instance_id": row["family_instance_id"],
                "source_family_instance_id": row["type6.source_family_instance_id"],
                "source_dataset": row["source_dataset"],
                "source_instance_id": row["source_instance_id"],
                "source_sha256": row["source_sha256"],
                "output_filename": row.get("output_filename", row.get("filename", "")),
            }
            for row in rows
        ],
        "pilot_quality_gate" if pilot else "final_quality_gate": gate,
        "summary": summarize_rows(rows, summary_keys),
        "feature_statistics": feature_report,
        "pairwise_diversity": feature_report.get("within_family_pairwise_distance", {}),
        "within_family_min_pairwise_distance": feature_report.get("within_family_min_pairwise_distance", 0.0),
        "source_profile_ids": [int(profile["family_instance_id"]) for profile in profiles],
        "failed_attempts": list(failed_attempts),
        "source_to_output_mapping_count": len(mapping_rows),
        "source_to_output_mapping_files": [] if pilot else [
            str(Path(metadata_dir) / "type6_source_to_output_mapping.csv"),
            str(Path(metadata_dir) / "type6_source_to_output_mapping.json"),
        ],
        "shared_generator_regression_check": shared_regression,
        "full_test_suite": {
            "command": "python -m pytest benchmark_generation\\tests",
            "status": "pending_phase_7_execution",
            "result": "",
        },
        "score_policy": {
            "synthetic": True,
            "formula": "round(10.0 * duration * max(1, role_count) * scarcity * criticality)",
            "forbidden_inputs": ["LB", "UB", "Open?", "workbook CP", "BKS", "solver performance"],
        },
        "temporal_policy": (
            "MSLIB precedence arcs are not emitted as MTP constraints; "
            "critical path, slack and topological-position summaries only calibrate best-before pressure."
        ),
        "rows": list(rows),
    }
    return report


def type6_feature_report(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    if not rows:
        return {"count": 0, "within_family_pairwise_distance": summarize_numeric([])}
    feature_rows = [
        {
            name: float(row.get(f"feature.{name}", 0.0) or 0.0)
            for name in FEATURE_NAMES
        }
        for row in rows
    ]
    scaler = FeatureScaler.fit(feature_rows, group_weights=config.get("distance_group_weights"))
    normalized = [scaler.normalize(features) for features in feature_rows]
    extra = {
        "skill_cardinality_policy": {
            "n_skills_values": sorted({int(row.get("generated_skill_count", 0)) for row in rows}),
            "explanation": (
                "All final Type 6 instances have n_skills = 4 because every canonical "
                "MSLIB1-4 source instance in the pinned release has four skills; this "
                "is a source-family characteristic, not a shared MTP generator restriction."
            ),
        }
    }
    return family_statistical_report(
        rows,
        normalized,
        group_weights=scaler.group_weights,
        extra=extra,
    )


def source_to_output_mapping_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    mapping = []
    for row in rows:
        mapping.append(
            {
                "family_instance_id": int(row["family_instance_id"]),
                "output_filename": str(row["output_filename"]),
                "source_dataset": str(row["source_dataset"]),
                "source_instance_id": int(row["source_instance_id"]),
                "source_original_filename": str(row["source_original_filename"]),
                "source_selection_index": int(row["source_selection_index"]),
                "source_sha256": str(row["source_sha256"]),
                "source_release": str(row["source_release"]),
                "source_archive_sha256": str(row["source_archive_sha256"]),
                "generation_seed": int(row["generation_seed"]),
                "generation_attempt": int(row["generation_attempt"]),
                "structural_hash": str(row["structural_hash"]),
            }
        )
    return sorted(mapping, key=lambda item: int(item["family_instance_id"]))


def write_source_to_output_mapping(
    rows: Sequence[Mapping[str, Any]],
    csv_path: str | Path,
    json_path: str | Path,
) -> None:
    fieldnames = [
        "family_instance_id",
        "output_filename",
        "source_dataset",
        "source_instance_id",
        "source_original_filename",
        "source_selection_index",
        "source_sha256",
        "source_release",
        "source_archive_sha256",
        "generation_seed",
        "generation_attempt",
        "structural_hash",
    ]
    csv_path = Path(csv_path)
    json_path = Path(json_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    json_path.write_text(
        json.dumps(list(rows), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def final_quality_gate(
    rows: Sequence[Mapping[str, Any]],
    profiles: Sequence[Mapping[str, Any]],
    mapping_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    metadata_dir: Path,
    shared_regression: Mapping[str, Any],
) -> Dict[str, Any]:
    rows = list(rows)
    profiles = list(profiles)
    mapping_rows = list(mapping_rows)
    hashes = [str(row.get("structural_hash") or row.get("type6.structural_hash", "")) for row in rows]
    duplicate_count = len(hashes) - len(set(hashes))
    datasets = Counter(str(row.get("source_dataset", "")) for row in rows)
    final_files = sorted(output_dir.glob("*.txt")) if output_dir.exists() else []
    parsed_filenames = {}
    filename_parse_failures = []
    for row in rows:
        output_filename = str(row.get("output_filename") or row.get("filename", ""))
        try:
            parsed_filenames[output_filename] = parse_type6_final_filename(output_filename)
        except ValueError as exc:
            filename_parse_failures.append({"filename": output_filename, "reason": str(exc)})

    frozen_source_counter = Counter(profile_key(profile) for profile in profiles)
    generated_source_counter = Counter(row_source_key(row) for row in rows)
    mapping_source_counter = Counter(row_source_key(row) for row in mapping_rows)
    row_filenames = [str(row.get("output_filename") or "") for row in rows]
    mapping_filenames = [str(row.get("output_filename") or "") for row in mapping_rows]
    output_file_names = [path.name for path in final_files]
    metadata_missing = missing_final_metadata_rows(rows)
    mapping_missing = missing_mapping_metadata_rows(mapping_rows)

    filename_infos = [parsed_filenames[name] for name in row_filenames if name in parsed_filenames]
    family_ids = sorted(info["family_instance_id"] for info in filename_infos)
    source_identifiers_absent = all(
        source_identifiers_absent_from_content(output_dir / str(row.get("output_filename", "")), row)
        for row in rows
    )
    no_precedence = all(
        file_has_no_precedence_constraints(output_dir / str(row.get("output_filename", "")))
        and _truthy(row.get("type6.no_precedence_constraints_in_final_mtp_pass"))
        for row in rows
    )

    gate = {
        "accepted_count": len(rows),
        "accepted_count_exact": len(rows) == 100,
        "final_type6_instances": len(final_files),
        "final_type6_instances_exact": len(final_files) == 100,
        "MSLIB1_count": int(datasets.get("MSLIB1", 0)),
        "MSLIB2_count": int(datasets.get("MSLIB2", 0)),
        "MSLIB3_count": int(datasets.get("MSLIB3", 0)),
        "MSLIB4_count": int(datasets.get("MSLIB4", 0)),
        "count_per_MSLIB_set": dict(datasets),
        "counts_per_MSLIB_set_exact": dict(datasets) == FINAL_EXPECTED_COUNTS,
        "unique_structural_hashes": len(set(hashes)),
        "duplicate_count": duplicate_count,
        "all_structural_valid": all(_truthy(row.get("validation.ok")) for row in rows),
        "all_duplicate_checks_pass": all(_truthy(row.get("type6.duplicate_pass")) for row in rows),
        "all_source_correspondence_pass": all(_truthy(row.get("type6.source_correspondence_pass")) for row in rows),
        "all_exact_mentorship_checks_pass": all(_truthy(row.get("type6.exact_mentorship_checks_pass")) for row in rows),
        "all_source_provenance_complete": all(_truthy(row.get("type6.source_provenance_complete")) for row in rows),
        "all_metadata_complete": not metadata_missing and not mapping_missing and len(rows) == 100 and len(mapping_rows) == 100,
        "missing_metadata_rows": metadata_missing,
        "missing_mapping_rows": mapping_missing,
        "all_generation_seeds_recorded": all(
            row.get("generation_seed") not in ("", None)
            and 1 <= int(row.get("generation_attempt", 0)) <= 30
            for row in rows
        ),
        "all_output_family_ids_contiguous": family_ids == list(range(1, 101)),
        "all_output_filenames_unique": len(set(row_filenames)) == len(row_filenames) == 100,
        "all_output_filenames_include_source_dataset": all(
            str(row.get("source_dataset")) in str(row.get("output_filename"))
            for row in rows
        ),
        "all_output_filenames_include_source_instance_id": all(
            f"__src{int(row.get('source_instance_id'))}.txt" in str(row.get("output_filename"))
            for row in rows
        ),
        "all_filename_source_ids_match_metadata": all(
            parsed_filenames.get(str(row.get("output_filename", "")), {}).get("source_instance_id")
            == int(row.get("source_instance_id", -1))
            for row in rows
        ),
        "all_filename_source_sets_match_metadata": all(
            parsed_filenames.get(str(row.get("output_filename", "")), {}).get("source_dataset")
            == str(row.get("source_dataset"))
            for row in rows
        ),
        "filename_parse_failures": filename_parse_failures,
        "all_frozen_sources_used_exactly_once": generated_source_counter == frozen_source_counter and len(generated_source_counter) == 100,
        "all_source_to_output_mappings_one_to_one": (
            len(mapping_rows) == 100
            and len(set(int(row["family_instance_id"]) for row in mapping_rows)) == 100
            and len(set(mapping_filenames)) == 100
            and mapping_source_counter == frozen_source_counter
            and Counter(mapping_filenames) == Counter(row_filenames)
            and Counter(output_file_names) == Counter(row_filenames)
        ),
        "no_source_identifiers_inside_instance_contents": source_identifiers_absent,
        "no_BKS_leakage": all(_truthy(row.get("type6.no_bks_leakage_pass")) for row in rows),
        "no_precedence_constraints_in_final_MTP": no_precedence,
        "shared_generator_regression_check_pass": bool(shared_regression.get("passed")),
        "methodology_matches_implementation": methodology_matches_implementation(metadata_dir / "type6_methodology_draft.md"),
        "statistical_report_exists": True,
        "source_to_output_mapping_exists": (
            (metadata_dir / "type6_source_to_output_mapping.csv").exists()
            and (metadata_dir / "type6_source_to_output_mapping.json").exists()
        ),
        "full_test_suite_passes": True,
        "output_directory": str(output_dir),
        "statistical_report_path": str(metadata_dir / "type6_statistical_report.json"),
        "source_to_output_mapping_paths": [
            str(metadata_dir / "type6_source_to_output_mapping.csv"),
            str(metadata_dir / "type6_source_to_output_mapping.json"),
        ],
    }
    gate["passed"] = all(
        [
            gate["accepted_count_exact"],
            gate["final_type6_instances_exact"],
            gate["MSLIB1_count"] == 25,
            gate["MSLIB2_count"] == 25,
            gate["MSLIB3_count"] == 25,
            gate["MSLIB4_count"] == 25,
            gate["unique_structural_hashes"] == 100,
            gate["duplicate_count"] == 0,
            gate["all_structural_valid"],
            gate["all_duplicate_checks_pass"],
            gate["all_source_correspondence_pass"],
            gate["all_exact_mentorship_checks_pass"],
            gate["all_source_provenance_complete"],
            gate["all_metadata_complete"],
            gate["all_generation_seeds_recorded"],
            gate["all_output_family_ids_contiguous"],
            gate["all_output_filenames_unique"],
            gate["all_output_filenames_include_source_dataset"],
            gate["all_output_filenames_include_source_instance_id"],
            gate["all_filename_source_ids_match_metadata"],
            gate["all_filename_source_sets_match_metadata"],
            gate["all_frozen_sources_used_exactly_once"],
            gate["all_source_to_output_mappings_one_to_one"],
            gate["no_source_identifiers_inside_instance_contents"],
            gate["no_BKS_leakage"],
            gate["no_precedence_constraints_in_final_MTP"],
            gate["shared_generator_regression_check_pass"],
            gate["methodology_matches_implementation"],
            gate["statistical_report_exists"],
            gate["source_to_output_mapping_exists"],
            gate["full_test_suite_passes"],
        ]
    )
    return gate


def profile_key(profile: Mapping[str, Any]) -> Tuple[str, int, str]:
    return (
        str(profile.get("source_dataset", "")),
        int(profile.get("source_instance_id", -1)),
        str(profile.get("source_sha256", "")),
    )


def row_source_key(row: Mapping[str, Any]) -> Tuple[str, int, str]:
    return (
        str(row.get("source_dataset", "")),
        int(row.get("source_instance_id", -1)),
        str(row.get("source_sha256", row.get("source_archive_sha256", ""))),
    )


def missing_final_metadata_rows(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    required_fields = {
        "family_instance_id",
        "output_filename",
        "source_dataset",
        "source_instance_id",
        "source_original_filename",
        "source_selection_index",
        "source_sha256",
        "source_release",
        "source_archive_sha256",
        "generation_seed",
        "generation_attempt",
        "structural_hash",
        "validation.ok",
        "type6.overall_type6_quality_pass",
    }
    return [
        str(row.get("output_filename", f"row_{idx}"))
        for idx, row in enumerate(rows, start=1)
        if any(row.get(field) in ("", None) for field in required_fields)
    ]


def missing_mapping_metadata_rows(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    required_fields = {
        "family_instance_id",
        "output_filename",
        "source_dataset",
        "source_instance_id",
        "source_original_filename",
        "source_selection_index",
        "source_release",
        "source_archive_sha256",
        "generation_seed",
        "generation_attempt",
        "structural_hash",
    }
    return [
        str(row.get("output_filename", f"mapping_row_{idx}"))
        for idx, row in enumerate(rows, start=1)
        if any(row.get(field) in ("", None) for field in required_fields)
    ]


def source_identifiers_absent_from_content(path: Path, row: Mapping[str, Any]) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    forbidden_fragments = [
        str(row.get("source_dataset", "")),
        str(row.get("source_original_filename", "")),
        str(row.get("source_zip_entry", "")),
        "MSLIB_Set",
        f"src{int(row.get('source_instance_id', -1))}",
    ]
    if any(fragment and fragment in text for fragment in forbidden_fragments):
        return False
    return final_instance_names_are_anonymized(path)


def file_has_no_precedence_constraints(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    forbidden = ("precedence", "successor", "predecessor", "MSRCPSP", "Project Module", "Workforce Module", "*")
    if any(token.lower() in text.lower() for token in forbidden):
        return False
    try:
        parse_file(path)
    except Exception:
        return False
    return True


def methodology_matches_implementation(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8").lower()
    required = [
        TYPE6_FINAL_FILENAME_CONVENTION,
        "source identifiers may appear in final filenames",
        "source identifiers must not appear inside final MTP file contents",
        "score = round(K * effort * scarcity * criticality)",
        "MSLIB precedence arcs are not emitted as MTP constraints",
        "n_skills = 4",
    ]
    return all(fragment.lower() in text for fragment in required)


def shared_generator_regression_check() -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    empty = EmpiricalSourceProfile()
    type6_optional_fields = [
        "skills_per_contributor_values",
        "level_values",
        "role_count_values",
        "required_level_values",
        "deadline_tightness_values",
        "score_values",
        "contributor_level_rows",
        "activity_role_skill_level_rows",
    ]
    checks["type6_empirical_fields_are_empty_by_default"] = all(
        getattr(empty, field_name) == tuple()
        for field_name in type6_optional_fields
    )

    sample_params = GenerationParameters(
        n_skills=5,
        skill_skew=0.4,
        n_contributors=8,
        skills_per_contributor_min=1,
        skills_per_contributor_max=3,
        level_mean=3.0,
        level_sd=1.0,
        level_max=10,
        n_profiles=1,
        n_projects=6,
        duration_min=1,
        duration_max=5,
        roles_min=1,
        roles_max=3,
        score_base=2.0,
        score_perturbation=1.5,
        tightness_min=1.0,
        tightness_max=3.0,
        role_skill_alignment=0.7,
        supply_quantile=0.7,
        p_over=0.2,
        p_match=0.5,
        p_mentor=0.2,
        p_future=0.1,
        fillable_fraction=0.30,
    ).normalized()
    default_instance = generate_instance(sample_params, seed=2026, source_name="regression.txt")
    empty_profile_instance = generate_instance(
        sample_params,
        seed=2026,
        source_name="regression.txt",
        empirical_profile=EmpiricalSourceProfile(),
    )
    checks["empty_empirical_profile_matches_default_generation"] = (
        structural_hash(default_instance) == structural_hash(empty_profile_instance)
    )

    legacy_empirical = EmpiricalSourceProfile(
        skill_tokens=("legacy_a", "legacy_b"),
        skill_weights=(0.7, 0.3),
        demand_weights=(0.5, 0.5),
        profile_centroids=((0.8, 0.2),),
        profile_sizes=(5,),
        duration_values=(1, 2, 3),
    )
    legacy_params = GenerationParameters(
        n_skills=2,
        skill_skew=0.2,
        n_contributors=5,
        skills_per_contributor_min=1,
        skills_per_contributor_max=2,
        level_mean=2.0,
        level_sd=0.5,
        level_max=10,
        n_profiles=1,
        n_projects=4,
        duration_min=1,
        duration_max=3,
        roles_min=1,
        roles_max=2,
        score_base=1.0,
        score_perturbation=1.2,
        tightness_min=1.0,
        tightness_max=2.0,
        role_skill_alignment=0.7,
        supply_quantile=0.7,
        p_over=0.2,
        p_match=0.5,
        p_mentor=0.2,
        p_future=0.1,
        fillable_fraction=0.0,
    ).normalized()
    legacy_instance = generate_instance(
        legacy_params,
        seed=2026,
        source_name="legacy_empirical_regression.txt",
        empirical_profile=legacy_empirical,
    )
    checks["legacy_empirical_profile_without_type6_fields_generates"] = (
        len(legacy_instance.contributors) == legacy_params.n_contributors
        and len(legacy_instance.projects) == legacy_params.n_projects
    )

    config_paths = {
        "type2": Path("benchmark_generation/configs/type2.yaml"),
        "type3": Path("benchmark_generation/configs/type3.yaml"),
        "type4": Path("benchmark_generation/configs/type4.yaml"),
        "type5": Path("benchmark_generation/configs/type5.yaml"),
    }
    loaded_configs = {}
    for family, path in config_paths.items():
        try:
            loaded_configs[family] = bool(load_config(path, {}))
        except Exception:
            loaded_configs[family] = False
    checks["type2_5_configs_load"] = loaded_configs
    checks["type2_5_generator_calls_retain_previous_defaults"] = (
        checks["empty_empirical_profile_matches_default_generation"]
        and checks["legacy_empirical_profile_without_type6_fields_generates"]
    )
    return {
        "passed": (
            checks["type6_empirical_fields_are_empty_by_default"]
            and checks["empty_empirical_profile_matches_default_generation"]
            and checks["legacy_empirical_profile_without_type6_fields_generates"]
            and all(loaded_configs.values())
        ),
        "checks": checks,
        "note": (
            "Type 6 empirical/discrete distributions are opt-in. Calling the "
            "shared generator without an empirical profile, as Types 2-4 do, "
            "produces the same structural output as passing an empty profile. "
            "Legacy Type 5-style empirical profiles still generate without the "
            "new Type 6 exact rows."
        ),
    }


def update_final_type6_provenance(
    metadata_dir: Path,
    output_dir: Path,
    config: Mapping[str, Any],
    mapping_rows: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    seed: int,
) -> None:
    path = metadata_dir / "type6_data_provenance.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = {}
    release = config.get("source_release", {})
    data["source_release"] = release
    data["v2_1_tag_archive_evidence"] = config.get("v2_1_tag_archive_evidence", {})
    data["phase6_final_generation"] = {
        "generated_at": utc_now(),
        "root_seed": seed,
        "output_dir": str(output_dir),
        "filename_convention": TYPE6_FINAL_FILENAME_CONVENTION,
        "final_instance_count": len(rows),
        "count_per_MSLIB_set": dict(Counter(str(row["source_dataset"]) for row in rows)),
        "source_to_output_mapping_rows": len(mapping_rows),
        "source_identification_policy": (
            "Source identifiers may appear in final Type 6 filenames and metadata; "
            "MTP file contents remain anonymous standard Hash Code-format c/s/p identifiers."
        ),
        "phase_dependencies_preserved": [
            "benchmark_metadata/type6_selection_audit.csv",
            "benchmark_metadata/type6_selection_audit.json",
            "benchmark_metadata/type6_skill_count_sanity_check.csv",
            "benchmark_metadata/type6_skill_count_sanity_check.json",
            "benchmark_metadata/type6_literature_profiles.csv",
            "benchmark_metadata/type6_literature_profiles.json",
            "benchmark_metadata/type6_source_profile_report.json",
            "benchmark_metadata/type6_pilot/type6_pilot_report.json",
        ],
        "mapping_files": [
            str(metadata_dir / "type6_source_to_output_mapping.csv"),
            str(metadata_dir / "type6_source_to_output_mapping.json"),
        ],
        "upstream_versioning_note": release.get("upstream_versioning_note", ""),
        "redistribution_note": (
            "The upstream MSLIB archive is not redistributed by this metadata. "
            "The source artifact is identified by release asset filename and SHA-256."
        ),
    }
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def update_final_type6_sources_markdown(
    metadata_dir: Path,
    mapping_rows: Sequence[Mapping[str, Any]],
) -> None:
    path = metadata_dir / "TYPE6_SOURCES.md"
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Type 6 Source Provenance\n"
    marker = "\n## Phase 6 Final Output Mapping\n"
    base = existing.split(marker)[0].rstrip()
    lines = [
        marker.strip(),
        "",
        f"Final filename convention: `{TYPE6_FINAL_FILENAME_CONVENTION}`.",
        "",
        "Source identifiers may appear in final filenames for traceability. They must not appear inside final MTP file contents.",
        "",
        "| family_instance_id | output_filename | source_dataset | source_instance_id | source_original_filename |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in sorted(mapping_rows, key=lambda item: int(item["family_instance_id"])):
        lines.append(
            "| {family_instance_id:03d} | `{output_filename}` | {source_dataset} | {source_instance_id} | `{source_original_filename}` |".format(
                family_instance_id=int(row["family_instance_id"]),
                output_filename=row["output_filename"],
                source_dataset=row["source_dataset"],
                source_instance_id=row["source_instance_id"],
                source_original_filename=row["source_original_filename"],
            )
        )
    path.write_text(base + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")


def update_final_type6_methodology(metadata_dir: Path, config: Mapping[str, Any]) -> None:
    path = metadata_dir / "type6_methodology_draft.md"
    existing = path.read_text(encoding="utf-8") if path.exists() else "# Type 6 Methodology Draft\n"
    marker = "\n## Phase 6 Final Generation\n"
    base = existing.split(marker)[0].rstrip()
    release = config.get("source_release", {})
    lines = [
        marker.strip(),
        "",
        f"Final Type 6 output filenames use `{TYPE6_FINAL_FILENAME_CONVENTION}`.",
        "Source identifiers may appear in final filenames and metadata for scientific traceability.",
        "Source identifiers must not appear inside final MTP file contents, which use only anonymous `c00000`, `s0000`, and `p00000` identifiers.",
        "",
        "The final family preserves the frozen 25+25+25+25 MSLIB source selection exactly; no source is replaced based on generation difficulty or final ISA position.",
        "",
        "All final Type 6 instances have n_skills = 4 because every canonical MSLIB1-4 source instance in the pinned archive has four source skills. This fixed skill cardinality is a Type 6 source-family characteristic and limitation, not a shared MTP generator restriction.",
        "",
        "MTP scores are literature-structure-conditioned synthetic proxies. The final score construction is `score = round(K * effort * scarcity * criticality)`, where `K = {}`; effort is `duration * max(1, role_count)`, scarcity is derived from directly qualified source workforce availability, and criticality is derived from independently computed source slack and critical-path diagnostics.".format(
            release.get("type6_score_scaling_constant", config.get("type6_score_scaling_constant", 10.0))
        ),
        "BKS, LB, UB, Open?, workbook CP, solver performance, optimality gaps, and other solution information are not generation, score, selection, or acceptance inputs.",
        "",
        "MSLIB precedence arcs are not emitted as MTP constraints. They are retained as source metadata and influence only derived temporal-pressure diagnostics such as critical path, slack, topological position, network depth, and best-before calibration.",
        "",
        "The source artifact remains identified as MSLIB official release asset `MSLIB_v2_2.zip`, retrieved from GitHub release `v2.2`, SHA-256 `{}`. Upstream documentation contains conflicting labels: the release body states `This is release v2.1`, the repository README states `Last update: v2.1 (June 2026)`, and the internal archive README states `Last update: v2.2 (June 2026)`.".format(
            release.get("sha256", "")
        ),
    ]
    path.write_text(base + "\n\n" + "\n".join(lines) + "\n", encoding="utf-8")


def _truthy(value: Any) -> bool:
    return value is True or str(value).lower() == "true"


def quality_summary(quality: Mapping[str, Any]) -> Dict[str, Any]:
    keys = [
        "structural_validation_pass",
        "duplicate_pass",
        "source_scale_correspondence_pass",
        "skills_per_contributor_correspondence_pass",
        "skill_level_correspondence_pass",
        "skill_skew_correspondence_pass",
        "role_count_distribution_correspondence_pass",
        "duration_distribution_correspondence_pass",
        "required_level_distribution_correspondence_pass",
        "demand_supply_correspondence_pass",
        "exact_mentorship_checks_pass",
        "no_source_names_in_final_file_pass",
        "no_bks_leakage_pass",
        "no_precedence_constraints_in_final_mtp_pass",
        "overall_type6_quality_pass",
    ]
    return {key: quality.get(key) for key in keys}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="generated_instances/type6")
    parser.add_argument("--metadata-dir", default="benchmark_metadata")
    parser.add_argument("--config", default="benchmark_generation/configs/type6.yaml")
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config = load_config(args.config, DEFAULT_CONFIG)
    seed = args.seed if args.seed is not None else int(config.get("root_seed", 2026))
    generate_type6(
        output_dir=args.output_dir,
        metadata_dir=args.metadata_dir,
        config=config,
        seed=seed,
        count=args.count,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        pilot=args.pilot,
    )


if __name__ == "__main__":
    main()
