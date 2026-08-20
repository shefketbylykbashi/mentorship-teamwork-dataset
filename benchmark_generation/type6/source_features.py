"""Source-native feature extraction for parsed MSLIB MSRCPSP instances."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from statistics import mean, median, pstdev
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from benchmark_generation.type6.parse_msrcp import (
    MSRCPSPInstance,
    topological_order,
)


def extract_source_features(
    instance: MSRCPSPInstance,
    excel_indicators: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    real_ids = set(instance.real_activity_ids)
    real_activities = [
        activity for activity in instance.activities if activity.activity_id in real_ids
    ]
    durations = [activity.duration for activity in real_activities]
    roles_by_activity: Dict[int, int] = {activity_id: 0 for activity_id in real_ids}
    demand_counts: Counter[int] = Counter()
    requirement_levels = []
    qualified_counts = []
    one_below_counts = []
    direct_roles = 0
    one_below_roles = 0
    mentor_available_roles = 0

    workforce_levels_by_skill: Dict[int, List[int]] = defaultdict(list)
    for row in instance.workforce_levels:
        for skill_idx, level in enumerate(row, start=1):
            if level > 0:
                workforce_levels_by_skill[skill_idx].append(level)
    supply_counts = Counter({skill: len(levels) for skill, levels in workforce_levels_by_skill.items()})

    for role in instance.role_requirements:
        if role.activity_id not in real_ids:
            continue
        roles_by_activity[role.activity_id] += 1
        demand_counts[role.skill_id] += 1
        requirement_levels.append(role.required_level)
        levels = workforce_levels_by_skill.get(role.skill_id, [])
        qualified = sum(1 for level in levels if level >= role.required_level)
        one_below = sum(1 for level in levels if level == role.required_level - 1)
        qualified_counts.append(qualified)
        one_below_counts.append(one_below)
        if qualified > 0:
            direct_roles += 1
        if one_below > 0:
            one_below_roles += 1
        if qualified > 0 and one_below > 0:
            mentor_available_roles += 1

    skills_per_resource = [sum(1 for level in row if level > 0) for row in instance.workforce_levels]
    skill_levels = [level for row in instance.workforce_levels for level in row if level > 0]
    role_counts = [roles_by_activity[activity.activity_id] for activity in real_activities]
    network = network_statistics(instance)
    total_roles = sum(role_counts)
    indicators = sanitize_excel_indicators(excel_indicators or {})
    features: Dict[str, Any] = {
        "MSLIB_set": instance.dataset,
        "source_instance_id": instance.source_instance_id,
        "source_filename": instance.source_filename,
        "source_zip_entry": instance.source_zip_entry,
        "source_sha256": instance.source_sha256,
        "n_activities_total": instance.n_activities_total,
        "n_real_activities": instance.n_real_activities,
        "n_resources": instance.n_resources,
        "n_skills": instance.n_skills,
        "n_skill_levels": instance.n_skill_levels,
        "n_precedence_arcs": len(instance.precedence_arcs),
        "bks_basic": instance.bks_basic,
        "deadline_skill_levels": instance.deadline_skill_levels,
        "total_roles": total_roles,
        "roles_per_activity_mean": _mean(role_counts),
        "roles_per_activity_sd": _sd(role_counts),
        "roles_per_activity_min": min(role_counts) if role_counts else 0,
        "roles_per_activity_max": max(role_counts) if role_counts else 0,
        "duration_mean": _mean(durations),
        "duration_sd": _sd(durations),
        "duration_median": _median(durations),
        "duration_p10": percentile(durations, 0.10),
        "duration_p90": percentile(durations, 0.90),
        "skills_per_resource_mean": _mean(skills_per_resource),
        "skills_per_resource_sd": _sd(skills_per_resource),
        "skills_per_resource_min": min(skills_per_resource) if skills_per_resource else 0,
        "skills_per_resource_max": max(skills_per_resource) if skills_per_resource else 0,
        "skill_level_mean": _mean(skill_levels),
        "skill_level_sd": _sd(skill_levels),
        "skill_level_max": max(skill_levels) if skill_levels else 0,
        "requirement_level_mean": _mean(requirement_levels),
        "requirement_level_sd": _sd(requirement_levels),
        "requirement_level_max": max(requirement_levels) if requirement_levels else 0,
        "demand_concentration": entropy_concentration(demand_counts),
        "supply_concentration": entropy_concentration(supply_counts),
        "demand_supply_alignment": cosine_similarity(demand_counts, supply_counts),
        "source_skill_skew": skill_skew_from_counts(supply_counts),
        "source_contention": _safe_div(total_roles, instance.n_resources),
        "qualified_contention": _mean(
            [_safe_div(1.0, count) if count > 0 else 1.0 for count in qualified_counts]
        ),
        "natural_direct_share": _safe_div(direct_roles, total_roles),
        "natural_one_below_share": _safe_div(one_below_roles, total_roles),
        "natural_mentor_available_share": _safe_div(mentor_available_roles, total_roles),
        "mean_directly_qualified_resources_per_role": _mean(qualified_counts),
        "mean_one_below_resources_per_role": _mean(one_below_counts),
        "hierarchical_skill_levels_available": True,
        "hierarchical_skill_interpretation": (
            "Workforce Module with Skill Levels gives resource skill levels 1..L, "
            "Skill Level Requirements Module gives one required level per expanded "
            "activity skill requirement; lower-bound eligibility is level >= requirement."
        ),
    }
    features.update(network)
    features.update(indicators)
    return {key: _finite_value(value) for key, value in features.items()}


def network_statistics(instance: MSRCPSPInstance) -> Dict[str, Any]:
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
        edge_depth[activity_id] = max(
            (edge_depth[pred] + 1 for pred in predecessors[activity_id]),
            default=0,
        )

    horizon = earliest_finish.get(instance.dummy_end_id, max(earliest_finish.values(), default=0))
    latest_finish: Dict[int, int] = {activity_id: horizon for activity_id in activities}
    latest_start: Dict[int, int] = {}
    for activity_id in reversed(order):
        successors = activities[activity_id].successors
        if successors:
            latest_finish[activity_id] = min(latest_start[successor] for successor in successors)
        latest_start[activity_id] = latest_finish[activity_id] - activities[activity_id].duration

    real_ids = list(instance.real_activity_ids)
    slacks = [max(0, latest_start[idx] - earliest_start[idx]) for idx in real_ids]
    topo_rank = {activity_id: idx for idx, activity_id in enumerate(order)}
    denominator = max(1, len(order) - 1)
    topo_positions = [topo_rank[idx] / denominator for idx in real_ids]
    earliest_positions = [
        _safe_div(earliest_start[idx], horizon)
        for idx in real_ids
    ]
    successors_by_id = {activity.activity_id: activity.successors for activity in instance.activities}
    reachable_real_pairs = 0
    real_set = set(real_ids)
    for activity_id in real_ids:
        reachable = _reachable_successors(activity_id, successors_by_id)
        reachable_real_pairs += len(reachable & real_set)
    possible_pairs = len(real_ids) * (len(real_ids) - 1) / 2
    arcs = len(instance.precedence_arcs)
    possible_arcs = instance.n_activities_total * (instance.n_activities_total - 1) / 2
    return {
        "critical_path_length": horizon,
        "network_depth": edge_depth.get(instance.dummy_end_id, max(edge_depth.values(), default=0)),
        "slack_mean": _mean(slacks),
        "slack_sd": _sd(slacks),
        "slack_median": _median(slacks),
        "zero_slack_share": _safe_div(sum(1 for value in slacks if value == 0), len(slacks)),
        "topological_position_mean": _mean(topo_positions),
        "topological_position_sd": _sd(topo_positions),
        "earliest_position_mean": _mean(earliest_positions),
        "earliest_position_sd": _sd(earliest_positions),
        "order_strength": _safe_div(reachable_real_pairs, possible_pairs),
        "arc_density": _safe_div(arcs, possible_arcs),
    }


def sanitize_excel_indicators(indicators: Mapping[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    aliases = {
        "# Act": "excel_n_activities",
        "CNC": "excel_CNC",
        "OS": "excel_OS",
        "SP": "excel_SP",
        "# Skills": "excel_n_skills",
        "SF": "excel_SF",
        "SFalpha": "excel_SF_alpha",
        "SS": "excel_SS",
        "SSalpha": "excel_SS_alpha",
        "# R": "excel_n_resources",
        "RA": "excel_RA",
        "RAalpha": "excel_RA_alpha",
        "LB": "excel_basic_LB",
        "UB": "excel_basic_UB",
        "Open?": "excel_basic_status",
        "CP": "excel_basic_CP",
    }
    for key, value in indicators.items():
        clean = str(key).replace("\u03b1", "alpha")
        output = aliases.get(clean)
        if output:
            result[output] = value
    return result


def percentile(values: Sequence[int | float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(float(value) for value in values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = max(0.0, min(1.0, float(q))) * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def entropy_concentration(counts: Mapping[int | str, int | float]) -> float:
    values = [float(value) for value in counts.values() if float(value) > 0]
    total = sum(values)
    if total <= 0 or len(values) <= 1:
        return 0.0
    entropy = -sum((value / total) * math.log(value / total) for value in values)
    return max(0.0, min(1.0, 1.0 - entropy / math.log(len(values))))


def cosine_similarity(
    demand: Mapping[int | str, int | float],
    supply: Mapping[int | str, int | float],
) -> float:
    keys = set(demand) | set(supply)
    numerator = sum(float(demand.get(key, 0.0)) * float(supply.get(key, 0.0)) for key in keys)
    norm_a = math.sqrt(sum(float(demand.get(key, 0.0)) ** 2 for key in keys))
    norm_b = math.sqrt(sum(float(supply.get(key, 0.0)) ** 2 for key in keys))
    return _safe_div(numerator, norm_a * norm_b)


def skill_skew_from_counts(counts: Mapping[int | str, int | float]) -> float:
    values = sorted([max(1.0, float(value)) for value in counts.values() if value], reverse=True)
    if len(values) <= 1:
        return 0.0
    log_ranks = [math.log(idx + 1) for idx in range(len(values))]
    log_values = [math.log(value) for value in values]
    mean_x = _mean(log_ranks)
    mean_y = _mean(log_values)
    denom = sum((value - mean_x) ** 2 for value in log_ranks)
    if denom <= 0:
        return 0.0
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_ranks, log_values)) / denom
    return max(0.0, min(2.0, -slope))


def _reachable_successors(
    start: int,
    successors_by_id: Mapping[int, Sequence[int]],
) -> set[int]:
    seen: set[int] = set()
    stack = list(successors_by_id.get(start, ()))
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(successors_by_id.get(current, ()))
    return seen


def _mean(values: Sequence[int | float]) -> float:
    return float(mean(values)) if values else 0.0


def _median(values: Sequence[int | float]) -> float:
    return float(median(values)) if values else 0.0


def _sd(values: Sequence[int | float]) -> float:
    return float(pstdev(values)) if len(values) > 1 else 0.0


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _finite_value(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return 0.0
    return value

