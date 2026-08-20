"""Build Type 6 literature-derived profiles from the frozen source selection."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Dict, List, Mapping, Sequence, Tuple
import zipfile

from benchmark_generation import __version__
from benchmark_generation.core.config import load_config
from benchmark_generation.core.generator import GenerationParameters, derive_seed
from benchmark_generation.core.metadata import config_hash, flatten_mapping, summarize_numeric
from benchmark_generation.type6.download_mslib import utc_now
from benchmark_generation.type6.parse_msrcp import MSRCPSPInstance, parse_msrcp_text, topological_order
from benchmark_generation.type6.select_sources import DATASETS, DEFAULT_CONFIG, find_release_root
from benchmark_generation.type6.source_features import extract_source_features, percentile


TYPE6_SCORE_SCALING_CONSTANT = 10.0


def build_literature_profiles(
    source_dir: str | Path,
    metadata_dir: str | Path,
    config: Dict[str, Any],
    seed: int,
    selection_csv: str | Path | None = None,
    selection_json: str | Path | None = None,
) -> Dict[str, Any]:
    metadata_dir = Path(metadata_dir)
    source_dir = Path(source_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    selection_csv = Path(selection_csv or metadata_dir / "type6_source_selection.csv")
    selection_json = Path(selection_json or metadata_dir / "type6_source_selection.json")
    validate_selection_inputs(selection_csv, selection_json)
    selection_rows = read_selection_csv(selection_csv)
    release_root = find_release_root(source_dir)
    selection_report = json.loads(selection_json.read_text(encoding="utf-8"))
    profiles = []
    for idx, row in enumerate(selection_rows, start=1):
        if idx == 1 or idx % 25 == 0:
            logging.info("Building Type6 literature profile %03d/%03d", idx, len(selection_rows))
        instance = load_selected_instance(release_root, row)
        profile_seed = derive_seed(seed, int(row["type6_source_index"]), salt=16_000_057)
        profiles.append(
            profile_from_instance(
                instance,
                row,
                config,
                seed=profile_seed,
                artifact_statement=str(selection_report.get("artifact_statement", "")),
            )
        )

    report = profile_report(profiles, config, selection_report, seed)
    if not report["quality_gate"]["passed"]:
        raise RuntimeError(
            "Type 6 literature profile quality gate failed: "
            + json.dumps(report["quality_gate"], sort_keys=True)
        )
    write_profile_outputs(profiles, report, metadata_dir)
    return report


def validate_selection_inputs(selection_csv: Path, selection_json: Path) -> None:
    if not selection_csv.exists():
        raise FileNotFoundError(f"Missing frozen Type 6 source selection CSV: {selection_csv}")
    if not selection_json.exists():
        raise FileNotFoundError(f"Missing frozen Type 6 source selection JSON: {selection_json}")
    report = json.loads(selection_json.read_text(encoding="utf-8"))
    gate = report.get("selection_gate", {})
    if not gate.get("passed"):
        raise RuntimeError("Frozen Type 6 source selection has not passed its quality gate")
    audit = report.get("selection_audit", {})
    if audit and not audit.get("policy_passed"):
        raise RuntimeError("Frozen Type 6 source selection has not passed the BKS/performance audit")


def read_selection_csv(path: str | Path) -> List[Dict[str, Any]]:
    with Path(path).open(encoding="utf-8", newline="") as fh:
        rows = [dict(row) for row in csv.DictReader(fh)]
    if len(rows) != 100:
        raise RuntimeError(f"Type 6 Phase 4 requires exactly 100 selected sources; found {len(rows)}")
    return sorted(rows, key=lambda row: int(row["type6_source_index"]))


def load_selected_instance(release_root: Path, row: Mapping[str, Any]) -> MSRCPSPInstance:
    dataset = str(row["MSLIB_set"])
    entry = str(row["source_zip_entry"])
    zip_path = release_root / f"{dataset}.zip"
    with zipfile.ZipFile(zip_path) as zf:
        payload = zf.read(entry)
    instance = parse_msrcp_text(
        payload.decode("utf-8", errors="strict"),
        dataset=dataset,
        source_filename=str(row["source_filename"]),
        source_zip_entry=entry,
    )
    if instance.source_sha256 != str(row["source_sha256"]):
        raise RuntimeError(
            f"Source checksum mismatch for {dataset}/{row['source_filename']}: "
            f"{instance.source_sha256} != {row['source_sha256']}"
        )
    return instance


def profile_from_instance(
    instance: MSRCPSPInstance,
    selection_row: Mapping[str, Any],
    config: Mapping[str, Any],
    seed: int,
    artifact_statement: str = "",
) -> Dict[str, Any]:
    source_features = extract_source_features(instance, selection_row)
    distributions = source_distributions(instance)
    parameters, parameter_derivation = generation_parameters_from_source(
        instance,
        source_features,
        distributions,
        config,
    )
    source_profile = source_profile_for_generator(instance, distributions, config)
    return {
        "family": "type6",
        "profile_id": int(selection_row["type6_source_index"]),
        "family_instance_id": int(selection_row["type6_source_index"]),
        "profile_seed": seed,
        "generator_version": __version__,
        "source_dataset": instance.dataset,
        "source_instance_id": instance.source_instance_id,
        "source_filename": instance.source_filename,
        "source_zip_entry": instance.source_zip_entry,
        "source_sha256": instance.source_sha256,
        "source_artifact": artifact_statement,
        "source_selection": {
            "selection_seed": int(selection_row["selection_seed"]),
            "selection_rank_within_set": int(selection_row["selection_rank_within_set"]),
            "selection_reason": str(selection_row["selection_reason"]),
        },
        "source_metrics": source_metrics(instance, source_features, distributions),
        "source_distributions": distributions,
        "source_profile": source_profile,
        "parameters": parameters.to_dict(),
        "parameter_derivation": parameter_derivation,
        "proxy_flags": {
            "precedence_constraints": "source_metadata_only_not_emitted_as_mtp_constraints",
            "score_proxy": "literature_structure_conditioned_synthetic_no_bks",
            "bks_usage": "provenance_only_not_generation_parameter_input",
            "skill_level_semantics": "lower_bound_hierarchical_eligibility_level_ge_requirement",
        },
        "source_provenance": {
            "bks_basic_provenance_only": instance.bks_basic,
            "deadline_skill_levels_provenance_only": instance.deadline_skill_levels,
            "workbook_solution_fields_excluded_from_parameters": ["LB", "UB", "Open?", "CP"],
        },
    }


def source_distributions(instance: MSRCPSPInstance) -> Dict[str, Any]:
    real_ids = set(instance.real_activity_ids)
    real_activities = [
        activity for activity in instance.activities if activity.activity_id in real_ids
    ]
    roles_by_activity: Dict[int, List[Dict[str, int]]] = {
        activity_id: [] for activity_id in real_ids
    }
    for role in instance.role_requirements:
        if role.activity_id in roles_by_activity:
            roles_by_activity[role.activity_id].append(
                {
                    "skill_id": role.skill_id,
                    "required_level": role.required_level,
                }
            )
    skill_supply_by_level: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    supply_counts: Counter[int] = Counter()
    positive_levels = []
    skills_per_resource = []
    for row in instance.workforce_levels:
        skill_count = 0
        for skill_idx, level in enumerate(row, start=1):
            if level > 0:
                skill_count += 1
                positive_levels.append(level)
                supply_counts[skill_idx] += 1
                skill_supply_by_level[f"skill_{skill_idx:03d}"][str(level)] += 1
        skills_per_resource.append(skill_count)

    demand_counts: Counter[int] = Counter()
    required_levels = []
    qualified_counts = []
    one_below_counts = []
    role_class_counts = Counter()
    potential_by_skill_level: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {
            "roles": 0,
            "directly_qualified_roles": 0,
            "one_below_roles": 0,
            "mentor_available_roles": 0,
        }
    )
    levels_by_skill = levels_by_skill_id(instance)
    for role in instance.role_requirements:
        if role.activity_id not in real_ids:
            continue
        demand_counts[role.skill_id] += 1
        required_levels.append(role.required_level)
        levels = levels_by_skill.get(role.skill_id, [])
        qualified = sum(1 for level in levels if level >= role.required_level)
        one_below = sum(1 for level in levels if level == role.required_level - 1)
        qualified_counts.append(qualified)
        one_below_counts.append(one_below)
        role_class_counts[classify_role(role.required_level, levels)] += 1
        key = f"skill_{role.skill_id:03d}_level_{role.required_level:02d}"
        potential_by_skill_level[key]["roles"] += 1
        if qualified > 0:
            potential_by_skill_level[key]["directly_qualified_roles"] += 1
        if one_below > 0:
            potential_by_skill_level[key]["one_below_roles"] += 1
        if qualified > 0 and one_below > 0:
            potential_by_skill_level[key]["mentor_available_roles"] += 1

    network = detailed_network_distributions(instance)
    durations = [activity.duration for activity in real_activities]
    role_counts = [len(roles_by_activity[activity.activity_id]) for activity in real_activities]
    difficulty_mix = normalized_mix(role_class_counts)
    return {
        "skills_per_resource_values": skills_per_resource,
        "source_skill_level_values": positive_levels,
        "skill_supply_by_level": {
            skill: dict(sorted(levels.items(), key=lambda item: int(item[0])))
            for skill, levels in sorted(skill_supply_by_level.items())
        },
        "skill_supply_counts": {f"skill_{idx:03d}": count for idx, count in sorted(supply_counts.items())},
        "skill_demand_counts": {f"skill_{idx:03d}": count for idx, count in sorted(demand_counts.items())},
        "roles_per_activity_values": role_counts,
        "required_skill_level_values": required_levels,
        "activity_duration_values": durations,
        "activity_role_requirements": {
            str(activity_id): roles
            for activity_id, roles in sorted(roles_by_activity.items())
        },
        "qualified_resource_counts_per_role": qualified_counts,
        "one_below_resource_counts_per_role": one_below_counts,
        "natural_difficulty_mix": difficulty_mix,
        "potential_mentor_availability_by_skill_level": dict(sorted(potential_by_skill_level.items())),
        "network": network,
    }


def generation_parameters_from_source(
    instance: MSRCPSPInstance,
    source_features: Mapping[str, Any],
    distributions: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Tuple[GenerationParameters, Dict[str, Any]]:
    durations = [int(value) for value in distributions["activity_duration_values"]]
    role_counts = [int(value) for value in distributions["roles_per_activity_values"]]
    skills_per_resource = [int(value) for value in distributions["skills_per_resource_values"]]
    level_values = [int(value) for value in distributions["source_skill_level_values"]]
    required_levels = [int(value) for value in distributions["required_skill_level_values"]]
    difficulty_mix = distributions["natural_difficulty_mix"]
    tightness_values = temporal_tightness_values(distributions["network"])
    score_base, score_perturbation = synthetic_score_parameters(distributions)
    params = GenerationParameters(
        n_skills=instance.n_skills,
        skill_skew=float(source_features["source_skill_skew"]),
        n_contributors=instance.n_resources,
        skills_per_contributor_min=max(1, min(skills_per_resource) if skills_per_resource else 1),
        skills_per_contributor_max=max(1, max(skills_per_resource) if skills_per_resource else 1),
        level_mean=_mean(level_values, default=1.0),
        level_sd=max(0.5, _sd(level_values)),
        level_max=max(1, min(10, instance.n_skill_levels)),
        n_profiles=len(resource_profile_structure(instance)["profile_sizes"]),
        n_projects=instance.n_real_activities,
        duration_min=max(1, min(durations) if durations else 1),
        duration_max=max(1, max(durations) if durations else 1),
        roles_min=max(1, min(role_counts) if role_counts else 1),
        roles_max=max(1, max(role_counts) if role_counts else 1),
        score_base=score_base,
        score_perturbation=score_perturbation,
        tightness_min=max(0.5, min(tightness_values) if tightness_values else 1.0),
        tightness_max=min(6.0, max(tightness_values) if tightness_values else 4.0),
        role_skill_alignment=max(0.0, min(1.0, float(source_features["demand_supply_alignment"]))),
        supply_quantile=source_supply_quantile(instance),
        p_over=float(difficulty_mix["p_over"]),
        p_match=float(difficulty_mix["p_match"]),
        p_mentor=float(difficulty_mix["p_mentor"]),
        p_future=float(difficulty_mix["p_future"]),
        fillable_fraction=float(config.get("fillable_fraction", 0.30)),
        role_level_max=max(max(required_levels) if required_levels else instance.n_skill_levels, instance.n_skill_levels),
        future_offset_min=int(config.get("future_offset_min", 2)),
        future_offset_max=int(config.get("future_offset_max", 4)),
    ).normalized()
    return params, {
        "n_projects": "source non-dummy activity count",
        "n_contributors": "source renewable resource count",
        "n_skills": "source skill type count",
        "duration_range": "source non-dummy activity duration min/max",
        "roles_range": "source expanded activity requirement min/max",
        "difficulty_mix": "natural source direct/match/one-below/future role classification",
        "deadline_tightness": "derived from source topological position and slack, not precedence constraints",
        "score_base": "diagnostic summary of source-conditioned synthetic score density; BKS not used",
        "score_perturbation": "diagnostic spread of source-conditioned synthetic score density; BKS not used",
        "score_formula": (
            "score = round(10.0 * duration * max(1, role_count) "
            "* (1 + mean_role_inverse_qualified_supply) "
            "* (1 + 1 - slack / critical_path_length)); "
            "BKS, LB, UB, Open?, workbook CP and solver performance are not used"
        ),
    }


def source_profile_for_generator(
    instance: MSRCPSPInstance,
    distributions: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    profile_structure = resource_profile_structure(instance)
    supply_counts = distributions["skill_supply_counts"]
    demand_counts = distributions["skill_demand_counts"]
    skill_tokens = tuple(f"mslib_skill_{idx:03d}" for idx in range(1, instance.n_skills + 1))
    score_scale = float(config.get("type6_score_scaling_constant", TYPE6_SCORE_SCALING_CONSTANT))
    return {
        "skill_tokens": list(skill_tokens),
        "skill_weights": normalize_counts(supply_counts, skill_tokens),
        "demand_weights": normalize_counts(demand_counts, skill_tokens),
        "profile_centroids": profile_structure["profile_centroids"],
        "profile_sizes": profile_structure["profile_sizes"],
        "contributor_level_rows": [list(row) for row in instance.workforce_levels],
        "skills_per_contributor_values": list(distributions["skills_per_resource_values"]),
        "level_values": list(distributions["source_skill_level_values"]),
        "duration_values": [int(value) for value in distributions["activity_duration_values"]],
        "role_count_values": [int(value) for value in distributions["roles_per_activity_values"]],
        "required_level_values": [int(value) for value in distributions["required_skill_level_values"]],
        "activity_role_skill_level_rows": [
            [
                [int(role["skill_id"]), int(role["required_level"])]
                for role in roles
            ]
            for _activity_id, roles in sorted(
                distributions["activity_role_requirements"].items(),
                key=lambda item: int(item[0]),
            )
        ],
        "deadline_tightness_values": temporal_tightness_values(distributions["network"]),
        "score_values": synthetic_project_score_values(distributions, scaling_constant=score_scale),
        "score_formula": {
            "formula": "round(K * effort * scarcity * criticality)",
            "K": score_scale,
            "effort": "duration * max(1, role_count)",
            "scarcity": "1 + mean(1 / max(1, directly_qualified_resources_for_role)) over source activity roles",
            "criticality": "1 + (1 - source_activity_slack / source_critical_path_length)",
            "forbidden_inputs": ["BKS", "LB", "UB", "Open?", "workbook CP", "solver performance"],
        },
    }


def source_metrics(
    instance: MSRCPSPInstance,
    source_features: Mapping[str, Any],
    distributions: Mapping[str, Any],
) -> Dict[str, Any]:
    network = distributions["network"]
    return {
        "source_activity_count": instance.n_real_activities,
        "source_resource_count": instance.n_resources,
        "source_skill_count": instance.n_skills,
        "source_skill_level_count": instance.n_skill_levels,
        "source_roles_total": int(source_features["total_roles"]),
        "source_skills_per_resource_mean": float(source_features["skills_per_resource_mean"]),
        "source_skills_per_resource_sd": float(source_features["skills_per_resource_sd"]),
        "source_skill_level_mean": float(source_features["skill_level_mean"]),
        "source_skill_level_sd": float(source_features["skill_level_sd"]),
        "source_roles_per_activity_mean": float(source_features["roles_per_activity_mean"]),
        "source_roles_per_activity_sd": float(source_features["roles_per_activity_sd"]),
        "source_duration_mean": float(source_features["duration_mean"]),
        "source_duration_sd": float(source_features["duration_sd"]),
        "source_duration_median": float(source_features["duration_median"]),
        "source_duration_p10": float(source_features["duration_p10"]),
        "source_duration_p90": float(source_features["duration_p90"]),
        "source_skill_skew": float(source_features["source_skill_skew"]),
        "source_demand_concentration": float(source_features["demand_concentration"]),
        "source_supply_concentration": float(source_features["supply_concentration"]),
        "source_demand_supply_alignment": float(source_features["demand_supply_alignment"]),
        "source_contention": float(source_features["source_contention"]),
        "source_qualified_contention": float(source_features["qualified_contention"]),
        "source_critical_path": int(network["critical_path_length"]),
        "source_network_depth": int(network["network_depth"]),
        "source_order_strength": float(network["order_strength"]),
        "source_arc_density": float(network["arc_density"]),
        "source_n_precedence_arcs": int(source_features["n_precedence_arcs"]),
        "source_natural_direct_fillability": float(source_features["natural_direct_share"]),
        "source_natural_one_below_potential": float(source_features["natural_one_below_share"]),
        "source_potential_mentor_availability": float(source_features["natural_mentor_available_share"]),
    }


def detailed_network_distributions(instance: MSRCPSPInstance) -> Dict[str, Any]:
    order = topological_order(instance)
    activities = {activity.activity_id: activity for activity in instance.activities}
    predecessors: Dict[int, List[int]] = {activity.activity_id: [] for activity in instance.activities}
    for activity in instance.activities:
        for successor in activity.successors:
            predecessors[successor].append(activity.activity_id)
    earliest_start: Dict[int, int] = {}
    earliest_finish: Dict[int, int] = {}
    edge_depth: Dict[int, int] = {}
    for activity_id in order:
        earliest_start[activity_id] = max(
            (earliest_finish[pred] for pred in predecessors[activity_id]),
            default=0,
        )
        earliest_finish[activity_id] = earliest_start[activity_id] + activities[activity_id].duration
        edge_depth[activity_id] = max((edge_depth[pred] + 1 for pred in predecessors[activity_id]), default=0)
    horizon = earliest_finish.get(instance.dummy_end_id, max(earliest_finish.values(), default=0))
    latest_finish = {activity_id: horizon for activity_id in activities}
    latest_start: Dict[int, int] = {}
    for activity_id in reversed(order):
        successors = activities[activity_id].successors
        if successors:
            latest_finish[activity_id] = min(latest_start[successor] for successor in successors)
        latest_start[activity_id] = latest_finish[activity_id] - activities[activity_id].duration
    real_ids = list(instance.real_activity_ids)
    topo_rank = {activity_id: idx for idx, activity_id in enumerate(order)}
    denominator = max(1, len(order) - 1)
    slack_values = [max(0, latest_start[idx] - earliest_start[idx]) for idx in real_ids]
    topo_positions = [topo_rank[idx] / denominator for idx in real_ids]
    earliest_positions = [_safe_div(earliest_start[idx], horizon) for idx in real_ids]
    reachable_real_pairs = 0
    real_set = set(real_ids)
    successors_by_id = {activity.activity_id: activity.successors for activity in instance.activities}
    for activity_id in real_ids:
        reachable_real_pairs += len(_reachable_successors(activity_id, successors_by_id) & real_set)
    possible_pairs = len(real_ids) * (len(real_ids) - 1) / 2
    arcs = len(instance.precedence_arcs)
    possible_arcs = instance.n_activities_total * (instance.n_activities_total - 1) / 2
    return {
        "topological_order": list(order),
        "critical_path_length": horizon,
        "network_depth": edge_depth.get(instance.dummy_end_id, max(edge_depth.values(), default=0)),
        "activity_slack_values": slack_values,
        "topological_position_values": topo_positions,
        "earliest_position_values": earliest_positions,
        "zero_slack_share": _safe_div(sum(1 for value in slack_values if value == 0), len(slack_values)),
        "order_strength": _safe_div(reachable_real_pairs, possible_pairs),
        "arc_density": _safe_div(arcs, possible_arcs),
    }


def temporal_tightness_values(network: Mapping[str, Any]) -> List[float]:
    horizon = max(1.0, float(network.get("critical_path_length", 1)))
    slacks = [float(value) for value in network.get("activity_slack_values", [])]
    positions = [float(value) for value in network.get("topological_position_values", [])]
    values = []
    for slack, position in zip(slacks, positions):
        values.append(max(0.5, min(6.0, 1.0 + position + slack / horizon)))
    return values or [1.0, 4.0]


def synthetic_score_parameters(distributions: Mapping[str, Any]) -> Tuple[float, float]:
    scores = synthetic_project_score_values(distributions, scaling_constant=1.0)
    durations = [int(value) for value in distributions["activity_duration_values"]]
    roles = [int(value) for value in distributions["roles_per_activity_values"]]
    density = [value / max(1, durations[idx] * max(1, roles[idx])) for idx, value in enumerate(scores)]
    median_density = percentile(density, 0.50)
    p10 = max(1e-9, percentile(density, 0.10))
    p90 = max(p10, percentile(density, 0.90))
    return max(0.5, min(5.0, median_density)), max(1.0, min(3.0, p90 / p10))


def synthetic_project_score_values(
    distributions: Mapping[str, Any],
    scaling_constant: float = TYPE6_SCORE_SCALING_CONSTANT,
) -> List[int]:
    durations = [int(value) for value in distributions["activity_duration_values"]]
    roles = [int(value) for value in distributions["roles_per_activity_values"]]
    activity_requirements = [
        activity_roles
        for _activity_id, activity_roles in sorted(
            distributions["activity_role_requirements"].items(),
            key=lambda item: int(item[0]),
        )
    ]
    supply_by_level = distributions["skill_supply_by_level"]
    network = distributions["network"]
    slack_values = [float(value) for value in network.get("activity_slack_values", [])]
    horizon = max(1.0, float(network.get("critical_path_length", 1)))
    scores = []
    for idx, (duration, role_count) in enumerate(zip(durations, roles)):
        effort = max(1, duration * max(1, role_count))
        scarcity_terms = []
        for role in activity_requirements[idx] if idx < len(activity_requirements) else []:
            skill_key = f"skill_{int(role['skill_id']):03d}"
            required = int(role["required_level"])
            qualified = sum(
                int(count)
                for level, count in supply_by_level.get(skill_key, {}).items()
                if int(level) >= required
            )
            scarcity_terms.append(1.0 / max(1, qualified))
        scarcity = 1.0 + _mean(scarcity_terms, default=1.0)
        criticality = 1.0 + (1.0 - _safe_div(slack_values[idx] if idx < len(slack_values) else 0.0, horizon))
        scores.append(max(1, int(round(float(scaling_constant) * effort * scarcity * criticality))))
    return scores


def resource_profile_structure(instance: MSRCPSPInstance) -> Dict[str, Any]:
    grouped: Dict[Tuple[int, ...], int] = Counter(tuple(row) for row in instance.workforce_levels)
    rows = sorted(grouped.items(), key=lambda item: (-item[1], item[0]))
    denominator = max(1, instance.n_skill_levels)
    return {
        "profile_count": len(rows),
        "profile_sizes": [count for _levels, count in rows],
        "profile_level_vectors": [list(levels) for levels, _count in rows],
        "profile_centroids": [
            [float(level) / denominator for level in levels]
            for levels, _count in rows
        ],
    }


def levels_by_skill_id(instance: MSRCPSPInstance) -> Dict[int, List[int]]:
    result: Dict[int, List[int]] = defaultdict(list)
    for row in instance.workforce_levels:
        for skill_idx, level in enumerate(row, start=1):
            if level > 0:
                result[skill_idx].append(level)
    for values in result.values():
        values.sort()
    return result


def classify_role(required_level: int, levels: Sequence[int]) -> str:
    if not levels:
        return "future"
    qualified = [level for level in levels if level >= required_level]
    one_below = [level for level in levels if level == required_level - 1]
    if not qualified:
        return "future"
    if one_below:
        return "mentor"
    reference = percentile(levels, 0.50)
    if required_level < reference:
        return "over"
    return "match"


def normalized_mix(counts: Mapping[str, int]) -> Dict[str, float]:
    total = sum(int(counts.get(key, 0)) for key in ("over", "match", "mentor", "future"))
    if total <= 0:
        return {"p_over": 0.25, "p_match": 0.25, "p_mentor": 0.25, "p_future": 0.25}
    return {
        "p_over": int(counts.get("over", 0)) / total,
        "p_match": int(counts.get("match", 0)) / total,
        "p_mentor": int(counts.get("mentor", 0)) / total,
        "p_future": int(counts.get("future", 0)) / total,
    }


def source_supply_quantile(instance: MSRCPSPInstance) -> float:
    levels_by_skill = levels_by_skill_id(instance)
    observed = []
    for role in instance.role_requirements:
        if role.activity_id not in set(instance.real_activity_ids):
            continue
        levels = levels_by_skill.get(role.skill_id, [])
        if not levels:
            continue
        below_or_equal = sum(1 for level in levels if level <= role.required_level)
        observed.append(below_or_equal / len(levels))
    return max(0.50, min(0.90, percentile(observed, 0.50) if observed else 0.75))


def normalize_counts(counts: Mapping[str, Any], skill_tokens: Sequence[str]) -> List[float]:
    values = []
    for idx, _token in enumerate(skill_tokens, start=1):
        values.append(float(counts.get(f"skill_{idx:03d}", 0.0)))
    total = sum(values)
    if total <= 0:
        return [1.0 / len(skill_tokens) for _ in skill_tokens]
    return [value / total for value in values]


def profile_report(
    profiles: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    selection_report: Mapping[str, Any],
    seed: int,
) -> Dict[str, Any]:
    counts = Counter(str(profile["source_dataset"]) for profile in profiles)
    expected_counts = {dataset: 25 for dataset in DATASETS}
    quality_gate = {
        "profile_count": len(profiles),
        "profile_count_exact": len(profiles) == 100,
        "count_per_MSLIB_set": dict(counts),
        "exactly_25_per_mslib1_4": all(counts[dataset] == expected_counts[dataset] for dataset in DATASETS),
        "no_mslib5_profiles": "MSLIB5" not in counts,
        "all_source_provenance_complete": all(profile_provenance_complete(profile) for profile in profiles),
        "all_generation_parameters_without_bks": all(parameters_without_bks(profile) for profile in profiles),
        "dummy_activities_excluded_from_project_count": all(
            int(profile["parameters"]["n_projects"]) == int(profile["source_metrics"]["source_activity_count"])
            for profile in profiles
        ),
        "source_resource_count_preserved": all(
            int(profile["parameters"]["n_contributors"]) == int(profile["source_metrics"]["source_resource_count"])
            for profile in profiles
        ),
        "source_skill_count_preserved": all(
            int(profile["parameters"]["n_skills"]) == int(profile["source_metrics"]["source_skill_count"])
            for profile in profiles
        ),
        "precedence_not_generation_constraint": all(
            "precedence" not in json.dumps(profile["parameters"], sort_keys=True).lower()
            and "precedence_arcs" not in json.dumps(profile["source_profile"], sort_keys=True).lower()
            for profile in profiles
        ),
    }
    quality_gate["passed"] = all(
        bool(value)
        for key, value in quality_gate.items()
        if key not in {"profile_count", "count_per_MSLIB_set"}
    )
    dimensions = [
        "source_activity_count",
        "source_resource_count",
        "source_skill_count",
        "source_roles_total",
        "source_skills_per_resource_mean",
        "source_skill_level_mean",
        "source_roles_per_activity_mean",
        "source_duration_mean",
        "source_duration_median",
        "source_skill_skew",
        "source_demand_supply_alignment",
        "source_contention",
        "source_critical_path",
        "source_network_depth",
        "source_natural_direct_fillability",
        "source_natural_one_below_potential",
        "source_potential_mentor_availability",
    ]
    summaries = {
        dimension: summarize_numeric([
            float(profile["source_metrics"][dimension])
            for profile in profiles
        ])
        for dimension in dimensions
    }
    return {
        "family": "type6",
        "phase": "literature_profiles_phase_4",
        "built_at": utc_now(),
        "root_seed": seed,
        "config_hash": config_hash(config),
        "profile_count": len(profiles),
        "count_per_MSLIB_set": dict(counts),
        "quality_gate": quality_gate,
        "source_selection_audit": selection_report.get("selection_audit", {}),
        "major_dimension_summaries": summaries,
        "representation_notes": [
            "MSLIB precedence arcs are retained only as network summary/distribution metadata.",
            "MTP project count equals non-dummy activity count.",
            "MTP score parameters are synthetic source-structure proxies and do not use BKS/LB/UB/open status.",
            "All canonical MSLIB1-4 source instances contain exactly four skill types; Type 6 preserves this source characteristic and does not artificially diversify n_skills.",
            "No MTP instances are generated in Phase 4.",
        ],
        "profiles": list(profiles),
    }


def write_profile_outputs(
    profiles: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
    metadata_dir: Path,
) -> None:
    (metadata_dir / "type6_literature_profiles.json").write_text(
        json.dumps({"profiles": list(profiles)}, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    write_profiles_csv(profiles, metadata_dir / "type6_literature_profiles.csv")
    (metadata_dir / "type6_source_profile_report.json").write_text(
        json.dumps({key: value for key, value in report.items() if key != "profiles"}, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    update_methodology_draft(report, metadata_dir / "type6_methodology_draft.md")


def write_profiles_csv(profiles: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    rows = []
    for profile in profiles:
        row = {
            "profile_id": profile["profile_id"],
            "family_instance_id": profile["family_instance_id"],
            "source_dataset": profile["source_dataset"],
            "source_instance_id": profile["source_instance_id"],
            "source_filename": profile["source_filename"],
            "source_sha256": profile["source_sha256"],
            "profile_seed": profile["profile_seed"],
        }
        row.update(flatten_mapping({"source_metrics": profile["source_metrics"]}))
        row.update(flatten_mapping({"parameters": profile["parameters"]}))
        rows.append(row)
    fieldnames = sorted({key for row in rows for key in row})
    with Path(path).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def update_methodology_draft(report: Mapping[str, Any], path: str | Path) -> None:
    existing = Path(path).read_text(encoding="utf-8") if Path(path).exists() else "# Type 6 Methodology Draft\n"
    marker = "\n## Phase 4 Literature Profiles\n"
    phase_text = marker + "\n".join(
        [
            "",
            "Phase 4 reparses each frozen selected MSLIB source file and derives one source profile per selected instance.",
            "Generation parameters preserve source resource, skill, and non-dummy activity counts exactly.",
            "Empirical distributions are retained for skills per resource, source skill levels, role multiplicities, required levels, activity durations, temporal pressure, and natural mentorship potential.",
            "Precedence arcs are not emitted as MTP constraints; only critical path, depth, slack, and topological-position summaries are retained for later temporal-pressure modeling.",
            "MTP score fields remain synthetic and are parameterized from source effort/scarcity/criticality summaries, never from BKS, LB, UB, Open?, or algorithmic results.",
            "A source sanity check confirms that all canonical MSLIB1-4 instances use exactly four skill types in both the workbook # Skills field and parsed .msrcp headers; Type 6 treats this as a literature-source characteristic and limitation.",
            "",
            "Profile quality gate:",
            "",
            json.dumps(report["quality_gate"], indent=2, sort_keys=True, default=str),
            "",
        ]
    )
    if marker in existing:
        existing = existing.split(marker)[0]
    Path(path).write_text(existing.rstrip() + phase_text, encoding="utf-8")


def profile_provenance_complete(profile: Mapping[str, Any]) -> bool:
    required = [
        "source_dataset",
        "source_instance_id",
        "source_filename",
        "source_zip_entry",
        "source_sha256",
        "source_selection",
        "parameters",
        "source_profile",
    ]
    return all(profile.get(key) not in ("", None) for key in required)


def parameters_without_bks(profile: Mapping[str, Any]) -> bool:
    text = json.dumps(profile.get("parameters", {}), sort_keys=True).lower()
    forbidden = ["bks", "lb", "ub", "open", "cplex", "ga", "sat", "simulated"]
    return not any(token in text for token in forbidden)


def _reachable_successors(start: int, successors_by_id: Mapping[int, Sequence[int]]) -> set[int]:
    seen: set[int] = set()
    stack = list(successors_by_id.get(start, ()))
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(successors_by_id.get(current, ()))
    return seen


def _mean(values: Sequence[int | float], default: float = 0.0) -> float:
    return float(mean(values)) if values else default


def _sd(values: Sequence[int | float]) -> float:
    return float(pstdev(values)) if len(values) > 1 else 0.0


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default="type6_source_data")
    parser.add_argument("--metadata-dir", default="benchmark_metadata")
    parser.add_argument("--config", default="benchmark_generation/configs/type6.yaml")
    parser.add_argument("--selection-csv", default=None)
    parser.add_argument("--selection-json", default=None)
    parser.add_argument("--seed", type=int, default=None)
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
    report = build_literature_profiles(
        args.source_dir,
        args.metadata_dir,
        config,
        seed=seed,
        selection_csv=args.selection_csv,
        selection_json=args.selection_json,
    )
    print(json.dumps(report["quality_gate"], indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
