"""Parameter sampling and anchor-derived search priors."""

from __future__ import annotations

import math
import random
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .features import build_demand_counts, build_supply_counts, build_supply_index, extract_features
from .generator import GenerationParameters
from .models import MTPInstance, iter_roles


INT_PARAMETERS = {
    "n_skills",
    "n_contributors",
    "skills_per_contributor_min",
    "skills_per_contributor_max",
    "level_max",
    "n_profiles",
    "n_projects",
    "duration_min",
    "duration_max",
    "roles_min",
    "roles_max",
    "role_level_max",
    "future_offset_min",
    "future_offset_max",
}

PROBABILITY_PARAMETERS = ["p_over", "p_match", "p_mentor", "p_future"]


def band_allocation(count: int, bands: Sequence[str]) -> List[str]:
    """Return an approximately balanced deterministic band sequence."""

    if not bands:
        raise ValueError("At least one scale band is required")
    base = count // len(bands)
    remainder = count % len(bands)
    sequence = []
    for idx, band in enumerate(bands):
        sequence.extend([band] * (base + (1 if idx < remainder else 0)))
    return sequence


def sample_type4_parameters(
    config: Mapping[str, Any],
    rng: random.Random,
    band_name: str,
) -> GenerationParameters:
    ranges = config["parameter_ranges"]
    bands = config["scale_bands"]
    band = bands[band_name]

    n_contributors = log_uniform_int(
        int(band["contributors"][0]), int(band["contributors"][1]), rng
    )
    ratio = log_uniform(float(config["project_contributor_ratio"][0]), float(config["project_contributor_ratio"][1]), rng)
    project_min, project_max = int(band["projects"][0]), int(band["projects"][1])
    n_projects = max(project_min, min(project_max, int(round(n_contributors * ratio))))
    if rng.random() < 0.35:
        n_projects = log_uniform_int(project_min, project_max, rng)

    n_skills = randint_range(ranges["n_skills"], rng)
    spc_low = int(ranges["skills_per_contributor"][0])
    spc_high = int(ranges["skills_per_contributor"][1])
    skills_per_contributor_min, skills_per_contributor_max = sample_ordered_int_range(
        [min(spc_low, n_skills), min(spc_high, n_skills)], rng
    )
    duration_min, duration_max = sample_ordered_int_range(ranges["duration"], rng)
    roles_min, roles_max = sample_ordered_int_range(ranges["roles_per_project"], rng)
    tightness_min, tightness_max = sample_ordered_float_range(ranges["tightness"], rng)
    p_over, p_match, p_mentor, p_future = sample_simplex(
        config.get("difficulty_simplex", {}), rng
    )

    return GenerationParameters(
        n_skills=n_skills,
        skill_skew=uniform_range(ranges["skill_skew"], rng),
        n_contributors=n_contributors,
        skills_per_contributor_min=skills_per_contributor_min,
        skills_per_contributor_max=skills_per_contributor_max,
        level_mean=uniform_range(ranges["level_mean"], rng),
        level_sd=uniform_range(ranges["level_sd"], rng),
        level_max=int(ranges.get("level_max", 10)),
        n_profiles=int(ranges.get("n_profiles", 1)),
        n_projects=n_projects,
        duration_min=duration_min,
        duration_max=duration_max,
        roles_min=roles_min,
        roles_max=roles_max,
        score_base=uniform_range(ranges["score_base"], rng),
        score_perturbation=uniform_range(ranges["score_perturbation"], rng),
        tightness_min=tightness_min,
        tightness_max=tightness_max,
        role_skill_alignment=uniform_range(ranges["role_skill_alignment"], rng),
        supply_quantile=uniform_range(ranges["supply_quantile"], rng),
        p_over=p_over,
        p_match=p_match,
        p_mentor=p_mentor,
        p_future=p_future,
        fillable_fraction=float(config.get("fillable_fraction", 0.30)),
        role_level_max=int(config.get("role_level_max", 100)),
    ).normalized()


def estimate_parameters_from_instance(instance: MTPInstance) -> GenerationParameters:
    features = extract_features(instance)
    contributor_skill_counts = [len(contributor.skills) for contributor in instance.contributors]
    contributor_levels = [
        level for contributor in instance.contributors for level in contributor.skills.values()
    ]
    durations = [project.duration for project in instance.projects]
    role_counts = [len(project.roles) for project in instance.projects]
    tightness = [
        project.best_before / project.duration if project.duration else 0.0
        for project in instance.projects
    ]
    scores = [
        project.score / max(1, project.duration * len(project.roles))
        for project in instance.projects
    ]
    p_over, p_match, p_mentor, p_future = estimate_difficulty_mix(instance)
    return GenerationParameters(
        n_skills=max(1, int(features["n_skills"])),
        skill_skew=estimate_skill_skew(instance),
        n_contributors=len(instance.contributors),
        skills_per_contributor_min=max(1, min(contributor_skill_counts)),
        skills_per_contributor_max=max(contributor_skill_counts),
        level_mean=_mean(contributor_levels, 3.0),
        level_sd=max(0.5, _sd(contributor_levels)),
        level_max=min(10, max(contributor_levels) if contributor_levels else 10),
        n_profiles=1,
        n_projects=len(instance.projects),
        duration_min=max(1, min(durations)),
        duration_max=max(durations),
        roles_min=max(1, min(role_counts)),
        roles_max=max(role_counts),
        score_base=max(0.1, _mean(scores, 1.0)),
        score_perturbation=max(1.0, min(5.0, (max(scores) / max(1e-9, min(scores))) if scores else 1.0)),
        tightness_min=max(0.0, min(tightness)),
        tightness_max=max(tightness),
        role_skill_alignment=min(1.0, max(0.0, features["demand_supply_alignment"])),
        supply_quantile=0.75,
        p_over=p_over,
        p_match=p_match,
        p_mentor=p_mentor,
        p_future=p_future,
        fillable_fraction=0.30,
    ).normalized()


def estimate_difficulty_mix(instance: MTPInstance, supply_quantile: float = 0.75) -> Tuple[float, float, float, float]:
    supply_index = build_supply_index(instance)
    counts = {"over": 0, "match": 0, "mentor": 0, "future": 0}
    for role in iter_roles(instance):
        levels = supply_index.get(role.skill, [])
        reference = quantile(levels, supply_quantile) if levels else 1
        if role.level < reference:
            counts["over"] += 1
        elif role.level == reference:
            counts["match"] += 1
        elif role.level == reference + 1:
            counts["mentor"] += 1
        else:
            counts["future"] += 1
    total = sum(counts.values()) or 1
    return (
        counts["over"] / total,
        counts["match"] / total,
        counts["mentor"] / total,
        counts["future"] / total,
    )


def estimate_skill_skew(instance: MTPInstance) -> float:
    """Estimate Zipf-like skew from supply concentration as a bounded proxy."""

    supply_counts = build_supply_counts(instance)
    if len(supply_counts) <= 1:
        return 0.0
    values = sorted(supply_counts.values(), reverse=True)
    ranks = list(range(1, len(values) + 1))
    log_ranks = [math.log(rank) for rank in ranks if rank > 0]
    log_values = [math.log(max(1, value)) for value in values]
    mean_x = sum(log_ranks) / len(log_ranks)
    mean_y = sum(log_values) / len(log_values)
    denom = sum((x - mean_x) ** 2 for x in log_ranks)
    if denom <= 0:
        return 0.0
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(log_ranks, log_values)) / denom
    return min(2.0, max(0.0, -slope))


def bounds_from_anchor_estimates(
    anchors: Sequence[GenerationParameters],
    config: Mapping[str, Any],
) -> Dict[str, Tuple[float, float]]:
    """Create defensible search bounds from immutable anchor measurements."""

    margin = float(config.get("anchor_margin", 0.35))
    hard = config.get("hard_bounds", {})
    bounds: Dict[str, Tuple[float, float]] = {}
    keys = asdict(anchors[0]).keys()
    for key in keys:
        values = [float(asdict(anchor)[key]) for anchor in anchors]
        low, high = min(values), max(values)
        width = max(1.0 if key in INT_PARAMETERS else 0.05, high - low)
        low -= width * margin
        high += width * margin
        if key in hard:
            hard_low = float(hard[key][0])
            hard_high = float(hard[key][1])
            if hard_low > hard_high:
                hard_low, hard_high = hard_high, hard_low
            low = max(hard_low, low)
            high = min(hard_high, high)
            if low > high:
                low, high = hard_low, hard_high
        bounds[key] = (low, max(low, high))
    bounds["fillable_fraction"] = (
        float(config.get("fillable_fraction", 0.30)),
        float(config.get("fillable_fraction", 0.30)),
    )
    bounds["level_max"] = (10, 10)
    bounds["n_profiles"] = (1, max(1, int(config.get("max_profiles", 4))))
    return bounds


def sample_from_bounds(
    bounds: Mapping[str, Tuple[float, float]],
    rng: random.Random,
) -> GenerationParameters:
    data: Dict[str, float | int] = {}
    for key, (low, high) in bounds.items():
        if key in PROBABILITY_PARAMETERS:
            continue
        if key in INT_PARAMETERS:
            if key in {"n_contributors", "n_projects", "n_skills"}:
                data[key] = log_uniform_int(max(1, int(round(low))), max(1, int(round(high))), rng)
            else:
                data[key] = rng.randint(max(1, int(round(low))), max(1, int(round(high))))
        else:
            data[key] = rng.uniform(float(low), float(high))
    p_over, p_match, p_mentor, p_future = sample_simplex({"alpha": [1, 1, 1, 1]}, rng)
    data.update(
        {
            "p_over": p_over,
            "p_match": p_match,
            "p_mentor": p_mentor,
            "p_future": p_future,
        }
    )
    data["skills_per_contributor_max"] = max(
        int(data["skills_per_contributor_min"]),
        min(int(data["skills_per_contributor_max"]), int(data["n_skills"])),
    )
    data["duration_max"] = max(int(data["duration_min"]), int(data["duration_max"]))
    data["roles_max"] = max(int(data["roles_min"]), int(data["roles_max"]))
    data["tightness_max"] = max(float(data["tightness_min"]), float(data["tightness_max"]))
    return GenerationParameters(**data).normalized()


def perturb_parameters(
    params: GenerationParameters,
    bounds: Mapping[str, Tuple[float, float]],
    rng: random.Random,
    scale: float,
) -> GenerationParameters:
    base = params.to_dict()
    data: Dict[str, float | int] = {}
    for key, value in base.items():
        if key in PROBABILITY_PARAMETERS:
            continue
        low, high = bounds.get(key, (float(value), float(value)))
        width = max(1e-9, high - low)
        if key in {"n_contributors", "n_projects", "n_skills"}:
            log_value = math.log(max(1.0, float(value)))
            log_width = max(1e-9, math.log(max(1.0, high)) - math.log(max(1.0, low)))
            proposal = math.exp(log_value + rng.gauss(0.0, scale * log_width))
        else:
            proposal = float(value) + rng.gauss(0.0, scale * width)
        proposal = max(low, min(high, proposal))
        data[key] = int(round(proposal)) if key in INT_PARAMETERS else proposal
    simplex = [max(1e-6, float(base[key]) + rng.gauss(0.0, scale * 0.35)) for key in PROBABILITY_PARAMETERS]
    total = sum(simplex)
    for key, value in zip(PROBABILITY_PARAMETERS, simplex):
        data[key] = value / total
    data["skills_per_contributor_max"] = max(
        int(data["skills_per_contributor_min"]),
        min(int(data["skills_per_contributor_max"]), int(data["n_skills"])),
    )
    data["duration_max"] = max(int(data["duration_min"]), int(data["duration_max"]))
    data["roles_max"] = max(int(data["roles_min"]), int(data["roles_max"]))
    data["tightness_max"] = max(float(data["tightness_min"]), float(data["tightness_max"]))
    return GenerationParameters(**data).normalized()


def interpolate_parameters(
    a: GenerationParameters,
    b: GenerationParameters,
    alpha: float,
    bounds: Mapping[str, Tuple[float, float]],
) -> GenerationParameters:
    """Interpolate anchor-estimated parameters as a search prior.

    This does not create or accept an instance directly. It only seeds the
    derivative-free search with plausible parameters; candidates are still
    generated forward and judged by measured feature distance.
    """

    alpha = max(0.0, min(1.0, float(alpha)))
    a_data = a.to_dict()
    b_data = b.to_dict()
    data: Dict[str, float | int] = {}
    for key, a_value in a_data.items():
        b_value = b_data[key]
        value = (1.0 - alpha) * float(a_value) + alpha * float(b_value)
        low, high = bounds.get(key, (value, value))
        value = max(float(low), min(float(high), value))
        data[key] = int(round(value)) if key in INT_PARAMETERS else value
    total = sum(float(data[key]) for key in PROBABILITY_PARAMETERS)
    if total <= 0:
        for key in PROBABILITY_PARAMETERS:
            data[key] = 0.25
    else:
        for key in PROBABILITY_PARAMETERS:
            data[key] = float(data[key]) / total
    data["skills_per_contributor_max"] = max(
        int(data["skills_per_contributor_min"]),
        min(int(data["skills_per_contributor_max"]), int(data["n_skills"])),
    )
    data["duration_max"] = max(int(data["duration_min"]), int(data["duration_max"]))
    data["roles_max"] = max(int(data["roles_min"]), int(data["roles_max"]))
    data["tightness_max"] = max(float(data["tightness_min"]), float(data["tightness_max"]))
    return GenerationParameters(**data).normalized()


def blend_difficulty_mix(
    params: GenerationParameters,
    target_mix: Sequence[float],
    blend: float,
) -> GenerationParameters:
    """Blend only p_over/p_match/p_mentor/p_future toward a target mixture."""

    if len(target_mix) != 4:
        raise ValueError("target_mix must contain four probabilities")
    blend = max(0.0, min(1.0, float(blend)))
    target_total = sum(max(0.0, float(value)) for value in target_mix)
    if target_total <= 0:
        target = [0.25, 0.25, 0.25, 0.25]
    else:
        target = [max(0.0, float(value)) / target_total for value in target_mix]
    data = params.to_dict()
    for key, target_value in zip(PROBABILITY_PARAMETERS, target):
        data[key] = (1.0 - blend) * float(data[key]) + blend * target_value
    total = sum(float(data[key]) for key in PROBABILITY_PARAMETERS)
    for key in PROBABILITY_PARAMETERS:
        data[key] = float(data[key]) / total
    return GenerationParameters(**data).normalized()


def sample_simplex(config: Mapping[str, Any], rng: random.Random) -> Tuple[float, float, float, float]:
    alpha = config.get("alpha", [1.0, 1.0, 1.0, 1.0])
    if isinstance(alpha, (int, float)):
        alpha = [float(alpha)] * 4
    if len(alpha) != 4:
        raise ValueError("difficulty_simplex.alpha must contain four weights")
    draws = [rng.gammavariate(max(1e-9, float(a)), 1.0) for a in alpha]
    total = sum(draws)
    return tuple(draw / total for draw in draws)  # type: ignore[return-value]


def uniform_range(bounds: Sequence[float], rng: random.Random) -> float:
    return rng.uniform(float(bounds[0]), float(bounds[1]))


def randint_range(bounds: Sequence[int | float], rng: random.Random) -> int:
    low, high = int(round(bounds[0])), int(round(bounds[1]))
    return rng.randint(min(low, high), max(low, high))


def sample_ordered_int_range(bounds: Sequence[int | float], rng: random.Random) -> Tuple[int, int]:
    low, high = int(round(bounds[0])), int(round(bounds[1]))
    first = rng.randint(min(low, high), max(low, high))
    second = rng.randint(min(low, high), max(low, high))
    return min(first, second), max(first, second)


def sample_ordered_float_range(bounds: Sequence[float], rng: random.Random) -> Tuple[float, float]:
    first = uniform_range(bounds, rng)
    second = uniform_range(bounds, rng)
    return min(first, second), max(first, second)


def log_uniform(low: float, high: float, rng: random.Random) -> float:
    low = max(1e-9, low)
    high = max(low, high)
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def log_uniform_int(low: int, high: int, rng: random.Random) -> int:
    low = max(1, int(low))
    high = max(low, int(high))
    return max(low, min(high, int(round(log_uniform(low, high, rng)))))


def quantile(sorted_values: Sequence[int], q: float) -> int:
    if not sorted_values:
        return 1
    values = sorted(sorted_values)
    if len(values) == 1:
        return int(values[0])
    position = min(1.0, max(0.0, q)) * (len(values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return int(values[lower])
    value = values[lower] + (values[upper] - values[lower]) * (position - lower)
    return max(1, int(round(value)))


def _mean(values: Sequence[float], default: float = 0.0) -> float:
    return float(sum(values) / len(values)) if values else default


def _sd(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((value - m) ** 2 for value in values) / len(values))
