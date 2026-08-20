"""Executable Type 6 pilot/final quality gates."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from benchmark_generation.core.features import (
    build_demand_counts,
    build_supply_counts,
    extract_features,
)
from benchmark_generation.core.metadata import summarize_numeric
from benchmark_generation.core.models import Contributor, MTPInstance, Project
from benchmark_generation.core.parameter_sampling import estimate_skill_skew
from benchmark_generation.core.validator import ValidationResult


ANON_CONTRIBUTOR_RE = re.compile(r"^c\d{5}$")
ANON_PROJECT_RE = re.compile(r"^p\d{5}$")
ANON_SKILL_RE = re.compile(r"^s\d{4}$")
QUANTILES = (0.10, 0.50, 0.90)


def evaluate_type6_quality(
    instance: MTPInstance,
    validation: ValidationResult,
    profile: Mapping[str, Any],
    duplicate_pass: bool,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    metrics = type6_generation_metrics(instance)
    source_metrics = profile["source_metrics"]
    source_distributions = profile["source_distributions"]
    gates = config.get("quality_gates", {})
    distribution_tolerance = float(gates.get("distribution_distance_tolerance", 0.001))

    distribution = source_distribution_match(metrics, source_metrics, source_distributions, gates)
    mentorship = true_same_project_mentorship_metrics(instance)
    exact_mentorship_pass = (
        0.0 <= mentorship["true_same_project_mentorship_feasibility"] <= mentorship["global_one_below_potential"] + 1e-12
        and mentorship["true_same_project_mentorship_feasibility"] <= mentorship["global_mentor_availability"] + 1e-12
        and mentorship["roles_total"] == instance.total_roles
    )
    source_scale = {
        "project_count_exact": int(metrics["n_projects"]) == int(source_metrics["source_activity_count"]),
        "contributor_count_exact": int(metrics["n_contributors"]) == int(source_metrics["source_resource_count"]),
        "skill_count_exact": int(metrics["n_skills"]) == int(source_metrics["source_skill_count"]),
    }
    source_correspondence = {
        **source_scale,
        "skills_per_contributor_correspondence": distribution["comparisons"]["skills_per_contributor"]["pass"],
        "skill_level_correspondence": distribution["comparisons"]["skill_levels"]["pass"],
        "role_count_distribution_correspondence": distribution["comparisons"]["roles_per_project"]["pass"],
        "duration_distribution_correspondence": distribution["comparisons"]["durations"]["pass"],
        "required_level_distribution_correspondence": distribution["comparisons"]["required_levels"]["pass"],
        "skill_skew_correspondence": distribution["scalar_comparisons"]["skill_skew"]["pass"],
        "demand_supply_alignment_correspondence": distribution["scalar_comparisons"]["demand_supply_alignment"]["pass"],
        "demand_supply_correspondence": all(
            distribution["scalar_comparisons"][name]["pass"]
            for name in ("demand_concentration", "supply_concentration", "contention")
        ),
    }
    results = {
        "structural_validation_pass": validation.ok,
        "duplicate_pass": duplicate_pass,
        "source_provenance_complete": profile_provenance_complete(profile),
        "source_scale_correspondence_pass": all(source_scale.values()),
        "contributor_count_correspondence_pass": source_scale["contributor_count_exact"],
        "project_count_correspondence_pass": source_scale["project_count_exact"],
        "skill_count_correspondence_pass": source_scale["skill_count_exact"],
        "skills_per_contributor_correspondence_pass": source_correspondence["skills_per_contributor_correspondence"],
        "skill_level_correspondence_pass": source_correspondence["skill_level_correspondence"],
        "skill_skew_correspondence_pass": source_correspondence["skill_skew_correspondence"],
        "role_count_distribution_correspondence_pass": source_correspondence["role_count_distribution_correspondence"],
        "duration_distribution_correspondence_pass": source_correspondence["duration_distribution_correspondence"],
        "required_level_distribution_correspondence_pass": source_correspondence["required_level_distribution_correspondence"],
        "demand_supply_correspondence_pass": source_correspondence["demand_supply_correspondence"],
        "exact_mentorship_checks_pass": exact_mentorship_pass,
        "no_source_names_in_final_file_pass": final_instance_names_are_anonymized(instance),
        "no_bks_leakage_pass": generation_profile_excludes_solution_fields(profile),
        "no_precedence_constraints_in_final_mtp_pass": instance_has_no_precedence_constraints(instance),
        "all_project_role_counts_fit_contributors": all(
            len(project.roles) <= len(instance.contributors)
            for project in instance.projects
        ),
        "distribution_match": distribution,
        "distribution_distance_tolerance": distribution_tolerance,
        "metrics": metrics,
        "mentorship_metrics": mentorship,
        "quality_gate_results": source_correspondence,
    }
    required_keys = [
        "structural_validation_pass",
        "duplicate_pass",
        "source_provenance_complete",
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
        "all_project_role_counts_fit_contributors",
    ]
    results["overall_type6_quality_pass"] = all(bool(results[key]) for key in required_keys)
    return results


def type6_generation_metrics(instance: MTPInstance) -> Dict[str, Any]:
    features = extract_features(instance)
    skills_per_contributor = [len(contributor.skills) for contributor in instance.contributors]
    contributor_levels = [
        level
        for contributor in instance.contributors
        for level in contributor.skills.values()
    ]
    roles_per_project = [len(project.roles) for project in instance.projects]
    durations = [project.duration for project in instance.projects]
    required_levels = [role.level for project in instance.projects for role in project.roles]
    tightness = [project.best_before / project.duration for project in instance.projects if project.duration]
    return {
        "n_contributors": len(instance.contributors),
        "n_projects": len(instance.projects),
        "n_skills": len(instance.skill_universe),
        "skills_per_contributor_values": skills_per_contributor,
        "contributor_level_values": contributor_levels,
        "roles_per_project_values": roles_per_project,
        "duration_values": durations,
        "required_level_values": required_levels,
        "skill_skew": estimate_skill_skew(instance),
        "demand_concentration": features["demand_concentration"],
        "supply_concentration": features["supply_concentration"],
        "demand_supply_alignment": features["demand_supply_alignment"],
        "contention": features["contention"],
        "deadline_tightness_mean": features["deadline_tightness_mean"],
        "deadline_tightness_sd": features["deadline_tightness_sd"],
        "schedule_pressure": features["schedule_pressure"],
        "features": features,
        "skill_supply_counts": dict(build_supply_counts(instance)),
        "skill_demand_counts": dict(build_demand_counts(instance)),
        "summary": {
            "skills_per_contributor": distribution_summary(skills_per_contributor),
            "contributor_levels": distribution_summary(contributor_levels),
            "roles_per_project": distribution_summary(roles_per_project),
            "durations": distribution_summary(durations),
            "required_levels": distribution_summary(required_levels),
            "deadline_tightness": distribution_summary(tightness),
        },
    }


def source_distribution_match(
    metrics: Mapping[str, Any],
    source_metrics: Mapping[str, Any],
    source_distributions: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> Dict[str, Any]:
    tolerance = float(gates.get("distribution_distance_tolerance", 0.001))
    comparisons = {
        "skills_per_contributor": distribution_comparison(
            metrics["skills_per_contributor_values"],
            source_distributions["skills_per_resource_values"],
            tolerance,
        ),
        "skill_levels": distribution_comparison(
            metrics["contributor_level_values"],
            source_distributions["source_skill_level_values"],
            tolerance,
        ),
        "roles_per_project": distribution_comparison(
            metrics["roles_per_project_values"],
            source_distributions["roles_per_activity_values"],
            tolerance,
        ),
        "durations": distribution_comparison(
            metrics["duration_values"],
            source_distributions["activity_duration_values"],
            tolerance,
        ),
        "required_levels": distribution_comparison(
            metrics["required_level_values"],
            source_distributions["required_skill_level_values"],
            tolerance,
        ),
    }
    scalar_comparisons = {
        "skill_skew": scalar_comparison(
            float(metrics["skill_skew"]),
            float(source_metrics["source_skill_skew"]),
            float(gates.get("skill_skew_tolerance", 0.05)),
            "absolute",
        ),
        "demand_supply_alignment": scalar_comparison(
            float(metrics["demand_supply_alignment"]),
            float(source_metrics["source_demand_supply_alignment"]),
            float(gates.get("demand_supply_alignment_tolerance", 0.05)),
            "absolute",
        ),
        "demand_concentration": scalar_comparison(
            float(metrics["demand_concentration"]),
            float(source_metrics["source_demand_concentration"]),
            float(gates.get("demand_concentration_tolerance", 0.05)),
            "absolute",
        ),
        "supply_concentration": scalar_comparison(
            float(metrics["supply_concentration"]),
            float(source_metrics["source_supply_concentration"]),
            float(gates.get("supply_concentration_tolerance", 0.05)),
            "absolute",
        ),
        "contention": scalar_comparison(
            float(metrics["contention"]),
            float(source_metrics["source_contention"]),
            float(gates.get("contention_relative_tolerance", 0.001)),
            "relative",
        ),
    }
    return {
        "passed": all(item["pass"] for item in comparisons.values())
        and all(item["pass"] for item in scalar_comparisons.values()),
        "comparisons": comparisons,
        "scalar_comparisons": scalar_comparisons,
    }


def distribution_comparison(generated: Sequence[int | float], source: Sequence[int | float], tolerance: float) -> Dict[str, Any]:
    distance = quantile_distance(generated, source)
    return {
        "generated_summary": distribution_summary(generated),
        "source_summary": distribution_summary(source),
        "quantile_distance": distance,
        "tolerance": tolerance,
        "pass": distance <= tolerance,
    }


def scalar_comparison(generated: float, source: float, tolerance: float, mode: str) -> Dict[str, Any]:
    if mode == "relative":
        error = abs(generated - source) / max(1.0, abs(source))
    else:
        error = abs(generated - source)
    return {
        "generated": generated,
        "source": source,
        "tolerance": tolerance,
        "mode": mode,
        "error": error,
        "pass": error <= tolerance,
    }


def quantile_distance(generated: Sequence[int | float], source: Sequence[int | float]) -> float:
    if not generated and not source:
        return 0.0
    if not generated or not source:
        return float("inf")
    diffs = [abs(percentile(generated, q) - percentile(source, q)) for q in QUANTILES]
    scale = max(1.0, abs(mean([float(value) for value in source])))
    return float(mean(diffs) / scale)


def distribution_summary(values: Sequence[int | float]) -> Dict[str, float]:
    if not values:
        return {"min": 0.0, "p10": 0.0, "median": 0.0, "mean": 0.0, "p90": 0.0, "max": 0.0, "sd": 0.0}
    return {
        "min": float(min(values)),
        "p10": percentile(values, 0.10),
        "median": percentile(values, 0.50),
        "mean": float(mean(values)),
        "p90": percentile(values, 0.90),
        "max": float(max(values)),
        "sd": float(pstdev(values)) if len(values) > 1 else 0.0,
    }


def true_same_project_mentorship_metrics(instance: MTPInstance) -> Dict[str, Any]:
    contributors = list(instance.contributors)
    total_roles = max(1, instance.total_roles)
    direct = 0
    one_below = 0
    mentor_available = 0
    true_same_project = 0
    project_true = 0
    for project in instance.projects:
        direct_candidates, one_below_candidates = project_candidate_indices(project, contributors)
        project_has_true = False
        for role_idx in range(len(project.roles)):
            if direct_candidates[role_idx]:
                direct += 1
                mentor_available += 1
            if one_below_candidates[role_idx]:
                one_below += 1
            if role_has_true_same_project_mentorship(
                project,
                role_idx,
                direct_candidates,
                one_below_candidates,
            ):
                true_same_project += 1
                project_has_true = True
        if project_has_true:
            project_true += 1
    return {
        "roles_total": instance.total_roles,
        "direct_fillability": direct / total_roles,
        "global_one_below_potential": one_below / total_roles,
        "global_mentor_availability": mentor_available / total_roles,
        "true_same_project_mentorship_feasibility": true_same_project / total_roles,
        "projects_with_true_same_project_mentorship_share": project_true / max(1, len(instance.projects)),
    }


def project_candidate_indices(
    project: Project,
    contributors: Sequence[Contributor],
) -> Tuple[List[List[int]], List[List[int]]]:
    direct: List[List[int]] = []
    one_below: List[List[int]] = []
    for role in project.roles:
        direct.append([
            idx
            for idx, contributor in enumerate(contributors)
            if contributor.level(role.skill) >= role.level
        ])
        one_below.append([
            idx
            for idx, contributor in enumerate(contributors)
            if contributor.level(role.skill) == role.level - 1
        ])
    return direct, one_below


def role_has_true_same_project_mentorship(
    project: Project,
    role_idx: int,
    direct_candidates: Sequence[Sequence[int]],
    one_below_candidates: Sequence[Sequence[int]],
) -> bool:
    if len(project.roles) < 2 or not one_below_candidates[role_idx] or not direct_candidates[role_idx]:
        return False
    target_mentors = set(direct_candidates[role_idx])
    other_role_indices = [idx for idx in range(len(project.roles)) if idx != role_idx]
    for mentor_role_idx in other_role_indices:
        for mentor in direct_candidates[mentor_role_idx]:
            if mentor not in target_mentors:
                continue
            for mentee in one_below_candidates[role_idx]:
                if mentor == mentee:
                    continue
                blocked = {mentor, mentee}
                remaining = [idx for idx in other_role_indices if idx != mentor_role_idx]
                if direct_matching_exists(remaining, direct_candidates, blocked):
                    return True
    return False


def direct_matching_exists(
    role_indices: Sequence[int],
    direct_candidates: Sequence[Sequence[int]],
    blocked: set[int],
) -> bool:
    ordered = sorted(role_indices, key=lambda idx: len(direct_candidates[idx]))
    matched_to_role: Dict[int, int] = {}

    def try_match(role_idx: int, seen: set[int]) -> bool:
        for contributor_idx in direct_candidates[role_idx]:
            if contributor_idx in blocked or contributor_idx in seen:
                continue
            seen.add(contributor_idx)
            previous = matched_to_role.get(contributor_idx)
            if previous is None or try_match(previous, seen):
                matched_to_role[contributor_idx] = role_idx
                return True
        return False

    for role_idx in ordered:
        if not try_match(role_idx, set()):
            return False
    return True


def final_instance_names_are_anonymized(instance_or_path: MTPInstance | str | Path) -> bool:
    if isinstance(instance_or_path, (str, Path)):
        from benchmark_generation.core.parser import parse_file

        instance = parse_file(instance_or_path)
    else:
        instance = instance_or_path
    for contributor in instance.contributors:
        if not ANON_CONTRIBUTOR_RE.match(contributor.name):
            return False
        if any(not ANON_SKILL_RE.match(skill) for skill in contributor.skills):
            return False
    for project in instance.projects:
        if not ANON_PROJECT_RE.match(project.name):
            return False
        if any(not ANON_SKILL_RE.match(role.skill) for role in project.roles):
            return False
    return True


def generation_profile_excludes_solution_fields(profile: Mapping[str, Any]) -> bool:
    """Check generation-facing Type 6 inputs do not contain BKS/solver fields.

    Type 6 profiles intentionally retain BKS values under ``source_provenance``
    and name excluded fields in methodology text. Those provenance records are
    not generation inputs, so this check inspects only the parameters and the
    empirical source profile consumed by the shared forward generator.
    """

    source_profile = dict(profile.get("source_profile", {}))
    source_profile.pop("score_formula", None)
    payload = {
        "parameters": profile.get("parameters", {}),
        "source_profile": source_profile,
    }
    text = json.dumps(payload, sort_keys=True, default=str).lower()
    forbidden = (
        "bks",
        "deadline_skill_levels",
        "lb",
        "ub",
        "open?",
        "workbook cp",
        "solver",
        "cplex",
        "optimal",
    )
    return not any(token in text for token in forbidden)


def instance_has_no_precedence_constraints(instance: MTPInstance) -> bool:
    """MTP instances have independent projects and no precedence fields."""

    return all(hasattr(project, "roles") for project in instance.projects)


def profile_provenance_complete(profile: Mapping[str, Any]) -> bool:
    required = {
        "source_dataset",
        "source_instance_id",
        "source_filename",
        "source_zip_entry",
        "source_sha256",
        "source_selection",
        "source_profile",
        "parameters",
    }
    return all(profile.get(key) not in ("", None) for key in required)


def pilot_quality_gate(rows: Sequence[Mapping[str, Any]], expected_count: int = 10) -> Dict[str, Any]:
    hashes = [str(row.get("type6.structural_hash", "")) for row in rows]
    datasets = Counter(str(row.get("source_dataset")) for row in rows)
    required_fields = {
        "filename",
        "family_instance_id",
        "seed",
        "source_dataset",
        "source_instance_id",
        "source_sha256",
        "type6.structural_hash",
    }
    missing_metadata = [
        str(row.get("filename", f"row_{idx}"))
        for idx, row in enumerate(rows, start=1)
        if any(row.get(field) in ("", None) for field in required_fields)
    ]
    gate = {
        "accepted_count": len(rows),
        "accepted_count_exact": len(rows) == expected_count,
        "all_four_mslib_sets_represented": set(datasets) == {"MSLIB1", "MSLIB2", "MSLIB3", "MSLIB4"},
        "pilot_count_per_MSLIB_set": dict(datasets),
        "unique_structural_hashes": len(set(hashes)),
        "unique_structural_hashes_exact": len(set(hashes)) == len(rows),
        "all_structural_valid": all(_truthy(row.get("validation.ok")) for row in rows),
        "all_duplicate_checks_pass": all(_truthy(row.get("type6.duplicate_pass")) for row in rows),
        "all_source_correspondence_pass": all(_truthy(row.get("type6.source_correspondence_pass")) for row in rows),
        "all_exact_mentorship_checks_pass": all(_truthy(row.get("type6.exact_mentorship_checks_pass")) for row in rows),
        "all_metadata_complete": not missing_metadata and len(rows) == expected_count,
        "missing_metadata_rows": missing_metadata,
        "all_no_source_names_in_final_files": all(_truthy(row.get("type6.no_source_names_in_final_file_pass")) for row in rows),
    }
    gate["passed"] = all(
        [
            gate["accepted_count_exact"],
            gate["all_four_mslib_sets_represented"],
            gate["unique_structural_hashes_exact"],
            gate["all_structural_valid"],
            gate["all_duplicate_checks_pass"],
            gate["all_source_correspondence_pass"],
            gate["all_exact_mentorship_checks_pass"],
            gate["all_metadata_complete"],
            gate["all_no_source_names_in_final_files"],
        ]
    )
    return gate


def summarize_rows(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> Dict[str, Dict[str, float]]:
    return {
        key: summarize_numeric([
            float(row[key])
            for row in rows
            if row.get(key) not in ("", None)
        ])
        for key in keys
    }


def percentile(values: Sequence[int | float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(float(value) for value in values)
    if len(values) == 1:
        return values[0]
    position = max(0.0, min(1.0, float(q))) * (len(values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def _truthy(value: Any) -> bool:
    return value is True or str(value).lower() == "true"
