"""Executable Type 5 realism and final-family quality gates."""

from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from benchmark_generation.core.features import extract_features
from benchmark_generation.core.models import Contributor, MTPInstance, Project, Role, iter_roles
from benchmark_generation.core.parameter_sampling import estimate_difficulty_mix, estimate_skill_skew
from benchmark_generation.core.validator import ValidationResult


ANON_CONTRIBUTOR_RE = re.compile(r"^c\d{5}$")
ANON_PROJECT_RE = re.compile(r"^p\d{5}$")
ANON_SKILL_RE = re.compile(r"^s\d{4}$")


def type5_realism_metrics(instance: MTPInstance, source_record: Mapping[str, Any]) -> Dict[str, float]:
    features = extract_features(instance)
    p_over, p_match, p_mentor, p_future = estimate_difficulty_mix(
        instance,
        supply_quantile=float(source_record["parameters"].get("supply_quantile", 0.75)),
    )
    direct_share, true_mentorship_share = realistic_initial_feasibility(instance)
    profile_assignments = list(instance.metadata.get("profile_assignments", []))
    profile_count, largest_profile_share = generated_profile_summary(profile_assignments)
    durations = [project.duration for project in instance.projects]
    tightness = [project.best_before / project.duration for project in instance.projects if project.duration]
    return {
        "generated_profile_count": float(profile_count),
        "generated_largest_profile_share": largest_profile_share,
        "generated_skill_skew": estimate_skill_skew(instance),
        "generated_alignment": features["demand_supply_alignment"],
        "measured_p_over": p_over,
        "measured_p_match": p_match,
        "measured_p_mentor": p_mentor,
        "measured_p_future": p_future,
        "directly_fillable_share": direct_share,
        "truly_valid_initial_mentorship_feasible_share": true_mentorship_share,
        "realistic_initial_feasibility": direct_share + true_mentorship_share,
        "skills_per_contributor_mean": features["skills_per_contributor_mean"],
        "roles_per_project_mean": features["roles_per_project_mean"],
        "contributor_level_mean": features["contributor_level_mean"],
        "required_level_mean": features["required_level_mean"],
        "duration_median": float(median(durations)) if durations else 0.0,
        "deadline_tightness_mean": sum(tightness) / len(tightness) if tightness else 0.0,
        "n_contributors": features["n_contributors"],
        "n_projects": features["n_projects"],
        "n_skills": features["n_skills"],
    }


def evaluate_type5_quality(
    instance: MTPInstance,
    validation: ValidationResult,
    source_record: Mapping[str, Any],
    duplicate_pass: bool,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    metrics = type5_realism_metrics(instance, source_record)
    gates = config.get("quality_gates", {})
    source_metrics = source_record["source_metrics"]
    params = source_record["parameters"]
    profile_low, profile_high = config.get("profile_count_range", [3, 8])
    mentor_low, mentor_high = gates.get("mentor_share_bounds", [0.10, 0.30])
    future_low, future_high = gates.get("future_share_bounds", [0.03, 0.18])
    largest_profile_limit = float(gates.get("largest_profile_share_max", 0.70))
    profile_pass = (
        int(profile_low) <= int(metrics["generated_profile_count"]) <= int(profile_high)
        and (
            metrics["generated_largest_profile_share"] <= largest_profile_limit
            or float(source_metrics.get("source_largest_profile_share", 0.0)) > largest_profile_limit
        )
    )
    skill_skew_delta = abs(metrics["generated_skill_skew"] - float(source_metrics["source_skill_skew"]))
    alignment_delta = abs(metrics["generated_alignment"] - float(params["role_skill_alignment"]))
    difficulty_pass = (
        float(mentor_low) <= metrics["measured_p_mentor"] <= float(mentor_high)
        and float(future_low) <= metrics["measured_p_future"] <= float(future_high)
    )
    feasibility_pass = (
        metrics["realistic_initial_feasibility"]
        >= float(gates.get("realistic_initial_feasibility_min", 0.70))
    )
    scale_pass = (
        int(metrics["n_contributors"]) == int(params["n_contributors"])
        and int(metrics["n_projects"]) == int(params["n_projects"])
        and int(metrics["n_skills"]) == int(params["n_skills"])
    )
    distribution = source_distribution_match(metrics, source_record, config)
    results = {
        "structural_validation_pass": validation.ok,
        "duplicate_pass": duplicate_pass,
        "profiles_pass": profile_pass,
        "skill_skew_pass": skill_skew_delta <= float(gates.get("skill_skew_tolerance", 0.20)),
        "alignment_pass": (
            metrics["generated_alignment"] >= float(gates.get("alignment_min", 0.55))
            and alignment_delta <= float(gates.get("alignment_tolerance", 0.15))
        ),
        "difficulty_mix_pass": difficulty_pass,
        "initial_realistic_feasibility_pass": feasibility_pass,
        "scale_pass": scale_pass,
        "source_distribution_match_pass": distribution["passed"],
        "distribution_match": distribution,
        "metrics": metrics,
        "skill_skew_delta": skill_skew_delta,
        "alignment_delta": alignment_delta,
    }
    results["overall_type5_quality_pass"] = all(
        bool(results[key])
        for key in (
            "structural_validation_pass",
            "duplicate_pass",
            "profiles_pass",
            "skill_skew_pass",
            "alignment_pass",
            "difficulty_mix_pass",
            "initial_realistic_feasibility_pass",
            "scale_pass",
            "source_distribution_match_pass",
        )
    )
    return results


def source_distribution_match(
    metrics: Mapping[str, float],
    source_record: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    tolerances = config.get("quality_gates", {}).get("distribution_tolerances", {})
    source = source_record["source_metrics"]
    params = source_record["parameters"]
    comparisons = {
        "skills_per_contributor_mean": (
            metrics["skills_per_contributor_mean"],
            float(source["source_skills_per_contributor_mean"]),
            float(tolerances.get("skills_per_contributor_mean", 2.0)),
            "absolute",
        ),
        "roles_per_project_mean": (
            metrics["roles_per_project_mean"],
            float(source["source_roles_per_work_unit_mean"]),
            float(tolerances.get("roles_per_project_mean", 2.0)),
            "absolute",
        ),
        "contributor_level_mean": (
            metrics["contributor_level_mean"],
            float(params["level_mean"]),
            float(tolerances.get("contributor_level_mean", 1.25)),
            "absolute",
        ),
        "required_level_mean": (
            metrics["required_level_mean"],
            float(params["level_mean"]),
            float(tolerances.get("required_level_mean", 2.0)),
            "absolute",
        ),
        "skill_skew": (
            metrics["generated_skill_skew"],
            float(source["source_skill_skew"]),
            float(tolerances.get("skill_skew", 0.20)),
            "absolute",
        ),
        "role_skill_alignment": (
            metrics["generated_alignment"],
            float(params["role_skill_alignment"]),
            float(tolerances.get("role_skill_alignment", 0.15)),
            "absolute",
        ),
        "duration_median": (
            metrics["duration_median"],
            float(source["source_duration_median"]),
            float(tolerances.get("duration_median_relative", 0.60)),
            "relative",
        ),
        "deadline_tightness_mean": (
            metrics["deadline_tightness_mean"],
            (float(params["tightness_min"]) + float(params["tightness_max"])) / 2.0,
            float(tolerances.get("deadline_tightness_mean", 1.25)),
            "absolute",
        ),
        "mentor_share": (
            metrics["measured_p_mentor"],
            float(source_record["nominal_difficulty_mix"]["p_mentor"]),
            float(tolerances.get("mentor_share", 0.15)),
            "absolute",
        ),
        "future_share": (
            metrics["measured_p_future"],
            float(source_record["nominal_difficulty_mix"]["p_future"]),
            float(tolerances.get("future_share", 0.10)),
            "absolute",
        ),
    }
    rows = {}
    passed = True
    for name, (generated, target, tolerance, mode) in comparisons.items():
        if mode == "relative":
            error = abs(generated - target) / max(1.0, abs(target))
        else:
            error = abs(generated - target)
        ok = error <= tolerance
        rows[name] = {
            "generated": generated,
            "target": target,
            "tolerance": tolerance,
            "mode": mode,
            "error": error,
            "pass": ok,
        }
        passed = passed and ok
    return {"passed": passed, "comparisons": rows}


def generated_profile_summary(assignments: Sequence[int]) -> Tuple[int, float]:
    if not assignments:
        return 0, 0.0
    counts: Dict[int, int] = {}
    for assignment in assignments:
        counts[int(assignment)] = counts.get(int(assignment), 0) + 1
    return len(counts), max(counts.values()) / len(assignments)


def realistic_initial_feasibility(instance: MTPInstance) -> Tuple[float, float]:
    contributors = list(instance.contributors)
    total_roles = max(1, instance.total_roles)
    direct_count = 0
    mentorship_count = 0
    for project in instance.projects:
        direct_candidates, one_below_candidates = _project_candidate_indices(project, contributors)
        for role_idx, _role in enumerate(project.roles):
            if direct_candidates[role_idx]:
                direct_count += 1
            elif _role_has_true_same_project_mentorship(
                project,
                role_idx,
                contributors,
                direct_candidates,
                one_below_candidates,
            ):
                mentorship_count += 1
    return direct_count / total_roles, mentorship_count / total_roles


def _project_candidate_indices(
    project: Project,
    contributors: Sequence[Contributor],
) -> Tuple[List[List[int]], List[List[int]]]:
    direct: List[List[int]] = []
    one_below: List[List[int]] = []
    for role in project.roles:
        direct.append(
            [
                idx
                for idx, contributor in enumerate(contributors)
                if contributor.level(role.skill) >= role.level
            ]
        )
        one_below.append(
            [
                idx
                for idx, contributor in enumerate(contributors)
                if contributor.level(role.skill) == role.level - 1
            ]
        )
    return direct, one_below


def _role_has_true_same_project_mentorship(
    project: Project,
    role_idx: int,
    contributors: Sequence[Contributor],
    direct_candidates: Sequence[Sequence[int]],
    one_below_candidates: Sequence[Sequence[int]],
) -> bool:
    if len(project.roles) < 2 or not one_below_candidates[role_idx] or not direct_candidates[role_idx]:
        return False
    mentee = one_below_candidates[role_idx][0]
    if all(mentor == mentee for mentor in direct_candidates[role_idx]):
        alternate_mentees = [
            candidate
            for candidate in one_below_candidates[role_idx]
            if any(mentor != candidate for mentor in direct_candidates[role_idx])
        ]
        if not alternate_mentees:
            return False
        mentee = alternate_mentees[0]

    remaining_role_indices = [idx for idx in range(len(project.roles)) if idx != role_idx]
    for mentor_role_idx in remaining_role_indices:
        mentor_options = [
            idx
            for idx in direct_candidates[mentor_role_idx]
            if idx in set(direct_candidates[role_idx]) and idx != mentee
        ]
        for mentor in mentor_options[:25]:
            blocked = {mentee, mentor}
            roles_to_match = [
                idx
                for idx in remaining_role_indices
                if idx != mentor_role_idx
            ]
            if _direct_matching_exists(roles_to_match, direct_candidates, blocked):
                return True
    return False


def _direct_matching_exists(
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


def final_type5_quality_gate(
    rows: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    expected_count: int,
    corpus: Mapping[str, Any],
    report_path: str | Path,
    provenance_path: str | Path,
    methodology_path: str | Path,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    hashes = [str(row.get("type5.structural_hash", "")) for row in rows]
    windows = {str(row.get("source_window_name")) for row in rows}
    repos = {f"{row.get('source_owner')}/{row.get('source_repo')}" for row in rows}
    family_ids = sorted(_safe_int(row.get("family_instance_id")) for row in rows)
    source_slot_ids = [_safe_int(row.get("source_slot_id")) for row in rows]
    expected_family_ids = list(range(1, expected_count + 1))
    filenames = {str(row.get("filename", "")) for row in rows}
    expected_filenames = {f"type5_{idx:03d}.txt" for idx in expected_family_ids}
    required_source_fields = {
        "source_slot_id",
        "source_owner",
        "source_repo",
        "source_url",
        "source_domain",
        "source_repository_id",
        "source_default_branch",
        "source_snapshot_sha",
        "source_window_name",
        "source_window_start",
        "source_window_end",
        "source_collection_timestamp",
    }
    missing_metadata = [
        row.get("filename", f"row_{idx}")
        for idx, row in enumerate(rows, start=1)
        if any(row.get(field) in ("", None) for field in required_source_fields)
    ]
    anonymized = all(final_instance_names_are_anonymized(output_dir / str(row.get("filename"))) for row in rows)
    gate = {
        "retained_empirical_sources": int(corpus.get("retained_valid_source_configurations", expected_count) or 0),
        "retained_empirical_sources_exact": int(corpus.get("retained_valid_source_configurations", expected_count) or 0) == expected_count,
        "accepted_count": len(rows),
        "accepted_count_exact": len(rows) == expected_count,
        "family_instance_ids_contiguous": family_ids == expected_family_ids,
        "source_slot_ids_unique": len(set(source_slot_ids)) == expected_count and all(value > 0 for value in source_slot_ids),
        "output_filenames_contiguous": filenames == expected_filenames,
        "unique_structural_hashes": len(set(hashes)),
        "unique_structural_hashes_exact": len(set(hashes)) == expected_count,
        "all_structurally_valid": all(_truthy(row.get("type5.structural_validation_pass")) for row in rows),
        "all_duplicate_checks_pass": all(_truthy(row.get("type5.duplicate_pass")) for row in rows),
        "all_overall_type5_quality_pass": all(_truthy(row.get("type5.overall_type5_quality_pass")) for row in rows),
        "all_source_provenance_complete": not missing_metadata and len(rows) == expected_count,
        "all_metadata_complete": not missing_metadata and all(row.get("seed") not in ("", None) for row in rows),
        "missing_metadata_rows": missing_metadata,
        "all_four_windows_represented": windows == {"W1", "W2", "W3", "W4"},
        "source_repository_coverage": len(repos),
        "no_real_usernames_in_final_instances": anonymized,
        "all_seeds_recorded": all(row.get("seed") not in ("", None) and row.get("root_seed") not in ("", None) for row in rows),
        "statistical_report_exists": Path(report_path).exists(),
        "provenance_report_exists": Path(provenance_path).exists(),
        "methodology_documented": Path(methodology_path).exists()
        and "Type 5" in Path(methodology_path).read_text(encoding="utf-8"),
        "replacement_count": len(corpus.get("replacements", [])),
        "failed_source_windows": len(corpus.get("failed_source_windows", [])),
    }
    gate["passed"] = all(
        [
            gate["accepted_count_exact"],
            gate["retained_empirical_sources_exact"],
            gate["family_instance_ids_contiguous"],
            gate["source_slot_ids_unique"],
            gate["output_filenames_contiguous"],
            gate["unique_structural_hashes_exact"],
            gate["all_structurally_valid"],
            gate["all_duplicate_checks_pass"],
            gate["all_overall_type5_quality_pass"],
            gate["all_source_provenance_complete"],
            gate["all_metadata_complete"],
            gate["all_four_windows_represented"],
            gate["no_real_usernames_in_final_instances"],
            gate["all_seeds_recorded"],
            gate["statistical_report_exists"],
            gate["provenance_report_exists"],
            gate["methodology_documented"],
        ]
    )
    return gate


def final_instance_names_are_anonymized(path: str | Path) -> bool:
    from benchmark_generation.core.parser import parse_file

    if not Path(path).exists():
        return False
    instance = parse_file(path)
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


def _truthy(value: Any) -> bool:
    return value is True or str(value).lower() == "true"


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1
