"""Deterministic, solver-independent feature extraction for MTP instances."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Mapping, Sequence

from .models import MTPInstance, Project, Role, iter_roles


FEATURE_GROUPS: Dict[str, List[str]] = {
    "size": [
        "n_contributors",
        "n_projects",
        "n_skills",
        "total_roles",
        "roles_per_project_mean",
        "roles_per_project_sd",
        "skills_per_contributor_mean",
        "skills_per_contributor_sd",
    ],
    "supply_demand": [
        "contention",
        "workload_contention",
        "demand_concentration",
        "supply_concentration",
        "demand_supply_alignment",
        "unmet_skill_share",
    ],
    "levels": [
        "contributor_level_mean",
        "contributor_level_sd",
        "required_level_mean",
        "required_level_sd",
        "level_gap_mean",
        "level_gap_sd",
    ],
    "mentorship": [
        "direct_fillability",
        "mentorship_fillability",
        "learning_reachable_demand",
        "mentor_availability",
        "within_project_mentorship_feasibility",
    ],
    "time": [
        "duration_mean",
        "duration_sd",
        "deadline_tightness_mean",
        "deadline_tightness_sd",
        "infeasible_deadline_share",
        "schedule_pressure",
    ],
    "scores": [
        "score_mean",
        "score_sd",
        "score_concentration",
        "score_density",
        "score_effort_correlation",
        "penalty_exposure",
    ],
}

FEATURE_NAMES: List[str] = [
    feature for group_features in FEATURE_GROUPS.values() for feature in group_features
]


def _safe_mean(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def _safe_sd(values: Sequence[float]) -> float:
    return float(pstdev(values)) if len(values) > 1 else 0.0


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _entropy_concentration(counts: Mapping[str, int | float]) -> float:
    """Return 1-normalized entropy, where 0 is even and 1 is concentrated."""

    positives = [float(value) for value in counts.values() if value > 0]
    total = sum(positives)
    if total <= 0 or len(positives) <= 1:
        return 0.0
    entropy = -sum((value / total) * math.log(value / total) for value in positives)
    return float(max(0.0, min(1.0, 1.0 - entropy / math.log(len(positives)))))


def _cosine_similarity(a: Mapping[str, int | float], b: Mapping[str, int | float]) -> float:
    keys = set(a) | set(b)
    numerator = sum(float(a.get(key, 0.0)) * float(b.get(key, 0.0)) for key in keys)
    norm_a = math.sqrt(sum(float(a.get(key, 0.0)) ** 2 for key in keys))
    norm_b = math.sqrt(sum(float(b.get(key, 0.0)) ** 2 for key in keys))
    return _safe_div(numerator, norm_a * norm_b)


def _pearson(x_values: Sequence[float], y_values: Sequence[float]) -> float:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return 0.0
    mean_x = mean(x_values)
    mean_y = mean(y_values)
    centered_x = [x - mean_x for x in x_values]
    centered_y = [y - mean_y for y in y_values]
    denom = math.sqrt(sum(x * x for x in centered_x) * sum(y * y for y in centered_y))
    if denom == 0:
        return 0.0
    return float(sum(x * y for x, y in zip(centered_x, centered_y)) / denom)


def build_supply_index(instance: MTPInstance) -> Dict[str, List[int]]:
    index: Dict[str, List[int]] = defaultdict(list)
    for contributor in instance.contributors:
        for skill, level in contributor.skills.items():
            index[skill].append(level)
    for levels in index.values():
        levels.sort()
    return dict(index)


def build_supply_counts(instance: MTPInstance) -> Counter:
    counts: Counter = Counter()
    for contributor in instance.contributors:
        counts.update(contributor.skills.keys())
    return counts


def build_demand_counts(instance: MTPInstance) -> Counter:
    return Counter(role.skill for role in iter_roles(instance))


def role_directly_fillable(role: Role, supply_index: Mapping[str, Sequence[int]]) -> bool:
    levels = supply_index.get(role.skill, ())
    return bool(levels and levels[-1] >= role.level)


def role_mentor_fillable(role: Role, supply_index: Mapping[str, Sequence[int]]) -> bool:
    levels = supply_index.get(role.skill, ())
    if not levels:
        return False
    has_mentee = any(level == role.level - 1 for level in levels)
    has_mentor = levels[-1] >= role.level
    return bool(has_mentee and has_mentor)


def role_learning_reachable(role: Role, supply_index: Mapping[str, Sequence[int]]) -> bool:
    levels = supply_index.get(role.skill, ())
    return bool(levels and role.level <= levels[-1] + 2 and not role_directly_fillable(role, supply_index))


def project_has_potential_mentorship(
    project: Project, supply_index: Mapping[str, Sequence[int]]
) -> bool:
    """Cheap project-level proxy for within-project mentorship opportunity."""

    if len(project.roles) < 2:
        return False
    return any(role_mentor_fillable(role, supply_index) for role in project.roles)


def extract_features(instance: MTPInstance) -> Dict[str, float]:
    """Extract the full deterministic candidate feature set.

    The extractor does not solve the instance and does not use algorithm
    performance. All features are computed from static contributor/project data.
    """

    n_contributors = len(instance.contributors)
    n_projects = len(instance.projects)
    total_roles = instance.total_roles
    skill_universe = instance.skill_universe

    roles_per_project = [len(project.roles) for project in instance.projects]
    skills_per_contributor = [len(contributor.skills) for contributor in instance.contributors]
    contributor_levels = [
        level for contributor in instance.contributors for level in contributor.skills.values()
    ]
    all_roles = list(iter_roles(instance))
    required_levels = [role.level for role in all_roles]

    supply_index = build_supply_index(instance)
    supply_counts = build_supply_counts(instance)
    demand_counts = build_demand_counts(instance)

    total_effort = sum(project.duration * len(project.roles) for project in instance.projects)
    durations = [project.duration for project in instance.projects]
    scores = [project.score for project in instance.projects]
    efforts = [project.duration * len(project.roles) for project in instance.projects]
    tightness = [_safe_div(project.best_before, project.duration) for project in instance.projects]

    demand_levels_by_skill: Dict[str, List[int]] = defaultdict(list)
    for role in all_roles:
        demand_levels_by_skill[role.skill].append(role.level)

    level_gaps: List[float] = []
    for skill, role_levels in demand_levels_by_skill.items():
        supply_levels = supply_index.get(skill, [])
        if not supply_levels:
            level_gaps.extend(role_levels)
            continue
        gap = _safe_mean(role_levels) - _safe_mean(supply_levels)
        level_gaps.extend([gap] * len(role_levels))

    direct_roles = [role_directly_fillable(role, supply_index) for role in all_roles]
    mentor_roles = [role_mentor_fillable(role, supply_index) for role in all_roles]
    learning_roles = [role_learning_reachable(role, supply_index) for role in all_roles]
    mentor_available = [role_mentor_fillable(role, supply_index) for role in all_roles]
    project_mentorship = [
        project_has_potential_mentorship(project, supply_index) for project in instance.projects
    ]

    score_density_values = [
        _safe_div(project.score, project.duration * len(project.roles))
        for project in instance.projects
    ]
    penalty_exposures = [
        _safe_div(max(0, project.duration - project.best_before), project.duration)
        for project in instance.projects
    ]

    features: Dict[str, float] = {
        "n_contributors": float(n_contributors),
        "n_projects": float(n_projects),
        "n_skills": float(len(skill_universe)),
        "total_roles": float(total_roles),
        "roles_per_project_mean": _safe_mean(roles_per_project),
        "roles_per_project_sd": _safe_sd(roles_per_project),
        "skills_per_contributor_mean": _safe_mean(skills_per_contributor),
        "skills_per_contributor_sd": _safe_sd(skills_per_contributor),
        "contention": _safe_div(total_roles, n_contributors),
        "workload_contention": _safe_div(total_effort, n_contributors),
        "demand_concentration": _entropy_concentration(demand_counts),
        "supply_concentration": _entropy_concentration(supply_counts),
        "demand_supply_alignment": _cosine_similarity(demand_counts, supply_counts),
        "unmet_skill_share": _safe_div(
            sum(count for skill, count in demand_counts.items() if skill not in supply_counts),
            total_roles,
        ),
        "contributor_level_mean": _safe_mean(contributor_levels),
        "contributor_level_sd": _safe_sd(contributor_levels),
        "required_level_mean": _safe_mean(required_levels),
        "required_level_sd": _safe_sd(required_levels),
        "level_gap_mean": _safe_mean(level_gaps),
        "level_gap_sd": _safe_sd(level_gaps),
        "direct_fillability": _safe_div(sum(direct_roles), total_roles),
        "mentorship_fillability": _safe_div(sum(mentor_roles), total_roles),
        "learning_reachable_demand": _safe_div(sum(learning_roles), total_roles),
        "mentor_availability": _safe_div(sum(mentor_available), total_roles),
        "within_project_mentorship_feasibility": _safe_div(
            sum(project_mentorship), n_projects
        ),
        "duration_mean": _safe_mean(durations),
        "duration_sd": _safe_sd(durations),
        "deadline_tightness_mean": _safe_mean(tightness),
        "deadline_tightness_sd": _safe_sd(tightness),
        "infeasible_deadline_share": _safe_div(
            sum(1 for project in instance.projects if project.duration > project.best_before),
            n_projects,
        ),
        "schedule_pressure": _safe_div(total_effort, sum(project.best_before for project in instance.projects)),
        "score_mean": _safe_mean(scores),
        "score_sd": _safe_sd(scores),
        "score_concentration": _entropy_concentration(
            {project.name: project.score for project in instance.projects}
        ),
        "score_density": _safe_mean(score_density_values),
        "score_effort_correlation": _pearson(scores, efforts),
        "penalty_exposure": _safe_mean(penalty_exposures),
    }

    return {name: _finite(features.get(name, 0.0)) for name in FEATURE_NAMES}


def _finite(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return float(value)
