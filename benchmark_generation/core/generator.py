"""Single parameterized forward generator for all synthetic MTP families."""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Sequence, Tuple

from benchmark_generation import __version__

from .features import build_supply_counts, build_supply_index
from .models import Contributor, MTPInstance, Project, Role
from .validator import ValidationResult, validate_instance


@dataclass
class GenerationParameters:
    n_skills: int
    skill_skew: float
    n_contributors: int
    skills_per_contributor_min: int
    skills_per_contributor_max: int
    level_mean: float
    level_sd: float
    level_max: int
    n_profiles: int
    n_projects: int
    duration_min: int
    duration_max: int
    roles_min: int
    roles_max: int
    score_base: float
    score_perturbation: float
    tightness_min: float
    tightness_max: float
    role_skill_alignment: float
    supply_quantile: float
    p_over: float
    p_match: float
    p_mentor: float
    p_future: float
    fillable_fraction: float = 0.30
    role_level_max: int = 100
    future_offset_min: int = 2
    future_offset_max: int = 4
    profile_bias: float = 4.0

    def normalized(self) -> "GenerationParameters":
        data = asdict(self)
        data["n_skills"] = max(1, int(data["n_skills"]))
        data["n_contributors"] = max(1, int(data["n_contributors"]))
        data["n_projects"] = max(1, int(data["n_projects"]))
        data["skills_per_contributor_min"] = max(1, int(data["skills_per_contributor_min"]))
        data["skills_per_contributor_max"] = max(
            data["skills_per_contributor_min"], int(data["skills_per_contributor_max"])
        )
        data["skills_per_contributor_max"] = min(
            data["skills_per_contributor_max"], data["n_skills"]
        )
        data["level_max"] = max(1, min(10, int(data["level_max"])))
        data["level_sd"] = max(1e-6, float(data["level_sd"]))
        data["n_profiles"] = max(1, int(data["n_profiles"]))
        data["duration_min"] = max(1, int(data["duration_min"]))
        data["duration_max"] = max(data["duration_min"], int(data["duration_max"]))
        data["roles_min"] = max(1, int(data["roles_min"]))
        data["roles_max"] = max(data["roles_min"], int(data["roles_max"]))
        data["score_base"] = max(1e-6, float(data["score_base"]))
        data["score_perturbation"] = max(1.0, float(data["score_perturbation"]))
        data["tightness_min"] = max(0.0, float(data["tightness_min"]))
        data["tightness_max"] = max(data["tightness_min"], float(data["tightness_max"]))
        data["role_skill_alignment"] = min(1.0, max(0.0, float(data["role_skill_alignment"])))
        data["supply_quantile"] = min(1.0, max(0.0, float(data["supply_quantile"])))
        data["fillable_fraction"] = min(1.0, max(0.0, float(data["fillable_fraction"])))
        data["role_level_max"] = max(1, min(100, int(data["role_level_max"])))
        data["future_offset_min"] = max(2, int(data["future_offset_min"]))
        data["future_offset_max"] = max(data["future_offset_min"], int(data["future_offset_max"]))
        probs = [float(data[key]) for key in ("p_over", "p_match", "p_mentor", "p_future")]
        total = sum(max(0.0, prob) for prob in probs)
        if total <= 0:
            probs = [0.25, 0.25, 0.25, 0.25]
            total = 1.0
        for key, prob in zip(("p_over", "p_match", "p_mentor", "p_future"), probs):
            data[key] = max(0.0, prob) / total
        return GenerationParameters(**data)

    def to_dict(self) -> Dict[str, float | int]:
        return asdict(self.normalized())


@dataclass(frozen=True)
class EmpiricalSourceProfile:
    """Optional empirical distributions for the shared forward generator.

    Type 5 uses this hook to ground contributor specialization and skill demand
    in measured repository/window activity. The generated Hash Code file still
    uses anonymized ``sXXXX`` skill names; semantic skill tokens are stored only
    in Type 5 metadata.
    """

    skill_tokens: Tuple[str, ...] = tuple()
    skill_weights: Tuple[float, ...] = tuple()
    demand_weights: Tuple[float, ...] = tuple()
    profile_centroids: Tuple[Tuple[float, ...], ...] = tuple()
    profile_sizes: Tuple[int, ...] = tuple()
    duration_values: Tuple[int, ...] = tuple()
    skills_per_contributor_values: Tuple[int, ...] = tuple()
    level_values: Tuple[int, ...] = tuple()
    role_count_values: Tuple[int, ...] = tuple()
    required_level_values: Tuple[int, ...] = tuple()
    deadline_tightness_values: Tuple[float, ...] = tuple()
    score_values: Tuple[int, ...] = tuple()
    contributor_level_rows: Tuple[Tuple[int, ...], ...] = tuple()
    activity_role_skill_level_rows: Tuple[Tuple[Tuple[int, int], ...], ...] = tuple()


@dataclass
class GenerationOutcome:
    instance: MTPInstance
    parameters: GenerationParameters
    seed: int
    attempts: int
    validation: ValidationResult
    rejection_reasons: List[str] = field(default_factory=list)
    generator_version: str = __version__


def generate_valid_instance(
    params: GenerationParameters,
    seed: int,
    max_attempts: int = 50,
    source_name: str | None = None,
    empirical_profile: EmpiricalSourceProfile | None = None,
) -> GenerationOutcome:
    """Generate and validate an instance, retrying only by fresh child seed."""

    params = params.normalized()
    rejection_reasons: List[str] = []
    for attempt in range(1, max_attempts + 1):
        child_seed = derive_seed(seed, attempt)
        instance = generate_instance(
            params,
            child_seed,
            source_name=source_name,
            empirical_profile=empirical_profile,
        )
        validation = validate_instance(
            instance,
            fillable_fraction=params.fillable_fraction,
            require_positive_project=True,
        )
        if validation.ok:
            return GenerationOutcome(
                instance=instance,
                parameters=params,
                seed=child_seed,
                attempts=attempt,
                validation=validation,
                rejection_reasons=rejection_reasons,
            )
        rejection_reasons.append(";".join(validation.issues))
    raise RuntimeError(
        f"Failed to generate valid instance after {max_attempts} attempts. "
        f"Last rejection: {rejection_reasons[-1] if rejection_reasons else 'none'}"
    )


def generate_instance(
    params: GenerationParameters,
    seed: int,
    source_name: str | None = None,
    empirical_profile: EmpiricalSourceProfile | None = None,
) -> MTPInstance:
    """Generate one complete instance forward from a parameter configuration."""

    params = params.normalized()
    rng = random.Random(seed)

    skills = [f"s{i:04d}" for i in range(params.n_skills)]
    base_skill_weights = _empirical_or_zipf_skill_weights(
        params.n_skills,
        params.skill_skew,
        empirical_profile,
    )

    profile_skill_weights = _make_profile_skill_weights(
        skills,
        base_skill_weights,
        params.n_profiles,
        params.profile_bias,
        rng,
        empirical_profile,
    )
    profile_assignments = _profile_assignments(
        params.n_contributors,
        len(profile_skill_weights),
        empirical_profile,
    )
    contributors = _generate_contributors(
        params,
        skills,
        base_skill_weights,
        profile_skill_weights,
        profile_assignments,
        rng,
        empirical_profile,
    )
    contributors = _ensure_skill_supply_coverage(contributors, skills, params, rng)
    supply_index = build_supply_index(
        MTPInstance(contributors=tuple(contributors), projects=tuple())
    )
    supply_counts = build_supply_counts(
        MTPInstance(contributors=tuple(contributors), projects=tuple())
    )
    role_weights = _role_skill_weights(
        skills,
        base_skill_weights,
        supply_counts,
        params,
        empirical_profile,
    )

    projects = []
    project_source_indices = _empirical_project_indices(empirical_profile, params.n_projects, rng)
    role_count_sequence = _empirical_int_sequence(
        empirical_profile.role_count_values if empirical_profile else (),
        params.n_projects,
        rng,
        params.roles_min,
        params.roles_max,
    )
    required_level_sequence = _empirical_int_sequence(
        empirical_profile.required_level_values if empirical_profile else (),
        sum(role_count_sequence) if role_count_sequence else 0,
        rng,
        1,
        params.role_level_max,
    )
    tightness_sequence = _empirical_float_sequence(
        empirical_profile.deadline_tightness_values if empirical_profile else (),
        params.n_projects,
        rng,
        params.tightness_min,
        params.tightness_max,
    )
    score_sequence = _empirical_int_sequence(
        empirical_profile.score_values if empirical_profile else (),
        params.n_projects,
        rng,
        1,
        None,
    )
    role_cursor = 0
    for project_idx in range(params.n_projects):
        source_idx = project_source_indices[project_idx] if project_source_indices else None
        duration = _project_duration(params, rng, empirical_profile, source_idx)
        empirical_roles = _empirical_project_roles(empirical_profile, source_idx, skills, params)
        n_roles = len(empirical_roles) if empirical_roles else (
            role_count_sequence[project_idx] if role_count_sequence else rng.randint(params.roles_min, params.roles_max)
        )
        if (
            empirical_profile
            and source_idx is not None
            and 0 <= source_idx < len(empirical_profile.score_values)
        ):
            score = max(1, int(empirical_profile.score_values[source_idx]))
        elif score_sequence:
            score = score_sequence[project_idx]
        else:
            perturb = rng.uniform(1.0, params.score_perturbation)
            score = max(1, int(round(params.score_base * duration * n_roles * perturb)))
        if (
            empirical_profile
            and source_idx is not None
            and 0 <= source_idx < len(empirical_profile.deadline_tightness_values)
        ):
            tightness = max(
                params.tightness_min,
                min(params.tightness_max, float(empirical_profile.deadline_tightness_values[source_idx])),
            )
        else:
            tightness = tightness_sequence[project_idx] if tightness_sequence else rng.uniform(params.tightness_min, params.tightness_max)
        best_before = max(0, int(round(duration * tightness)))
        roles = []
        if empirical_roles:
            roles.extend(empirical_roles)
        else:
            for _role_idx in range(n_roles):
                skill = weighted_choice(skills, role_weights, rng)
                if required_level_sequence and role_cursor < len(required_level_sequence):
                    level = required_level_sequence[role_cursor]
                else:
                    level = _required_level(skill, supply_index, params, rng)
                role_cursor += 1
                roles.append(Role(skill=skill, level=level))
        projects.append(
            Project(
                name=f"p{project_idx:05d}",
                duration=duration,
                score=score,
                best_before=best_before,
                roles=tuple(roles),
            )
        )

    return MTPInstance(
        contributors=tuple(contributors),
        projects=tuple(projects),
        source_name=source_name,
        metadata={
            "seed": seed,
            "generator_version": __version__,
            "profile_assignments": tuple(profile_assignments),
        },
    )


def derive_seed(root_seed: int, child_index: int, salt: int = 1_000_003) -> int:
    return int((root_seed * salt + child_index * 97_409 + 17) % (2**31 - 1))


def zipf_weights(n_items: int, skew: float) -> List[float]:
    if n_items <= 0:
        return []
    skew = max(0.0, float(skew))
    weights = [1.0 / ((idx + 1) ** skew) for idx in range(n_items)]
    total = sum(weights)
    return [weight / total for weight in weights]


def normalize_weights(weights: Sequence[float], n_items: int | None = None) -> List[float]:
    """Return non-negative weights normalized to sum to one.

    If the supplied sequence is empty, the wrong length, or all non-positive,
    a uniform distribution is returned. This is deliberately small and lives in
    the core generator so Type 5 can reuse it without adding dependencies.
    """

    if n_items is None:
        n_items = len(weights)
    values = [max(0.0, float(value)) for value in weights[:n_items]]
    if len(values) != n_items:
        values = []
    total = sum(values)
    if n_items <= 0:
        return []
    if total <= 0.0:
        return [1.0 / n_items for _ in range(n_items)]
    return [value / total for value in values]


def _empirical_int_sequence(
    values: Sequence[int],
    count: int,
    rng: random.Random,
    low: int,
    high: int | None,
) -> List[int]:
    if count <= 0 or not values:
        return []
    pool = []
    for value in values:
        item = max(low, int(round(value)))
        if high is not None:
            item = min(high, item)
        pool.append(item)
    return _shuffled_repeated_sequence(pool, count, rng)


def _empirical_float_sequence(
    values: Sequence[float],
    count: int,
    rng: random.Random,
    low: float,
    high: float,
) -> List[float]:
    if count <= 0 or not values:
        return []
    pool = [max(low, min(high, float(value))) for value in values]
    return _shuffled_repeated_sequence(pool, count, rng)


def _shuffled_repeated_sequence(
    values: Sequence[int] | Sequence[float],
    count: int,
    rng: random.Random,
) -> List:
    if not values:
        return []
    sequence = []
    while len(sequence) < count:
        chunk = list(values)
        rng.shuffle(chunk)
        sequence.extend(chunk)
    return sequence[:count]


def weighted_choice(items: Sequence[str], weights: Sequence[float], rng: random.Random) -> str:
    threshold = rng.random()
    cumulative = 0.0
    for item, weight in zip(items, weights):
        cumulative += weight
        if threshold <= cumulative:
            return item
    return items[-1]


def _empirical_or_zipf_skill_weights(
    n_skills: int,
    skill_skew: float,
    empirical_profile: EmpiricalSourceProfile | None,
) -> List[float]:
    if empirical_profile and len(empirical_profile.skill_weights) == n_skills:
        return normalize_weights(empirical_profile.skill_weights, n_skills)
    return zipf_weights(n_skills, skill_skew)


def _make_profile_skill_sets(
    skills: Sequence[str],
    weights: Sequence[float],
    n_profiles: int,
    rng: random.Random,
) -> List[set[str]]:
    if n_profiles <= 1:
        return [set(skills)]
    profile_size = max(1, int(round(len(skills) / math.sqrt(n_profiles))))
    profiles = []
    for _profile_idx in range(n_profiles):
        chosen = set()
        while len(chosen) < min(profile_size, len(skills)):
            chosen.add(weighted_choice(skills, weights, rng))
        profiles.append(chosen)
    return profiles


def _make_profile_skill_weights(
    skills: Sequence[str],
    base_weights: Sequence[float],
    n_profiles: int,
    profile_bias: float,
    rng: random.Random,
    empirical_profile: EmpiricalSourceProfile | None,
) -> List[List[float]]:
    if (
        empirical_profile
        and empirical_profile.profile_centroids
        and all(len(row) == len(skills) for row in empirical_profile.profile_centroids)
    ):
        rows = [
            normalize_weights(
                [
                    0.15 * base + 0.85 * max(0.0, float(value))
                    for base, value in zip(base_weights, row)
                ],
                len(skills),
            )
            for row in empirical_profile.profile_centroids
        ]
        return rows[: max(1, n_profiles)]

    profile_skill_sets = _make_profile_skill_sets(skills, base_weights, n_profiles, rng)
    return [
        _profile_adjusted_weights(
            skills,
            base_weights,
            profile,
            profile_bias if n_profiles > 1 else 1.0,
        )
        for profile in profile_skill_sets
    ]


def _profile_assignments(
    n_contributors: int,
    n_profiles: int,
    empirical_profile: EmpiricalSourceProfile | None,
) -> List[int]:
    n_profiles = max(1, n_profiles)
    if empirical_profile and empirical_profile.profile_sizes:
        weights = normalize_weights(empirical_profile.profile_sizes, len(empirical_profile.profile_sizes))
        raw = [weight * n_contributors for weight in weights]
        counts = [int(math.floor(value)) for value in raw]
        while sum(counts) < n_contributors:
            idx = max(range(len(counts)), key=lambda item: raw[item] - counts[item])
            counts[idx] += 1
        while sum(counts) > n_contributors:
            idx = max(range(len(counts)), key=lambda item: counts[item] - raw[item])
            if counts[idx] > 0:
                counts[idx] -= 1
            else:
                break
        assignments: List[int] = []
        for idx, count in enumerate(counts):
            assignments.extend([idx] * count)
        if assignments:
            return assignments[:n_contributors]
    return [idx % n_profiles for idx in range(n_contributors)]


def _generate_contributors(
    params: GenerationParameters,
    skills: Sequence[str],
    base_weights: Sequence[float],
    profile_skill_weights: Sequence[Sequence[float]],
    profile_assignments: Sequence[int],
    rng: random.Random,
    empirical_profile: EmpiricalSourceProfile | None = None,
) -> List[Contributor]:
    empirical_rows = _empirical_contributor_rows(empirical_profile, params.n_contributors, len(skills), rng)
    if empirical_rows:
        contributors = []
        for contributor_idx, row in enumerate(empirical_rows):
            levels = {
                skill: max(1, min(params.level_max, int(level)))
                for skill, level in zip(skills, row)
                if int(level) > 0
            }
            contributors.append(Contributor(name=f"c{contributor_idx:05d}", skills=dict(sorted(levels.items()))))
        return contributors

    skill_count_sequence = _empirical_int_sequence(
        empirical_profile.skills_per_contributor_values if empirical_profile else (),
        params.n_contributors,
        rng,
        1,
        params.n_skills,
    )
    total_skill_assignments = sum(skill_count_sequence) if skill_count_sequence else 0
    level_sequence = _empirical_int_sequence(
        empirical_profile.level_values if empirical_profile else (),
        total_skill_assignments,
        rng,
        1,
        params.level_max,
    )
    level_cursor = 0
    contributors = []
    for contributor_idx in range(params.n_contributors):
        n_skills = skill_count_sequence[contributor_idx] if skill_count_sequence else rng.randint(
            params.skills_per_contributor_min, params.skills_per_contributor_max
        )
        profile_idx = profile_assignments[contributor_idx] % len(profile_skill_weights)
        skill_weights = profile_skill_weights[profile_idx]
        chosen = set()
        while len(chosen) < n_skills:
            chosen.add(weighted_choice(skills, skill_weights, rng))
        levels = {}
        for skill in sorted(chosen):
            if level_sequence and level_cursor < len(level_sequence):
                level = level_sequence[level_cursor]
            else:
                level = _truncated_normal_level(
                    params.level_mean, params.level_sd, params.level_max, rng
                )
            level_cursor += 1
            levels[skill] = level
        contributors.append(Contributor(name=f"c{contributor_idx:05d}", skills=levels))
    return contributors


def _empirical_contributor_rows(
    empirical_profile: EmpiricalSourceProfile | None,
    n_contributors: int,
    n_skills: int,
    rng: random.Random,
) -> List[Tuple[int, ...]]:
    if (
        not empirical_profile
        or not empirical_profile.contributor_level_rows
        or any(len(row) != n_skills for row in empirical_profile.contributor_level_rows)
    ):
        return []
    rows = list(empirical_profile.contributor_level_rows)
    if len(rows) >= n_contributors:
        rng.shuffle(rows)
        return rows[:n_contributors]
    sequence = []
    while len(sequence) < n_contributors:
        chunk = list(rows)
        rng.shuffle(chunk)
        sequence.extend(chunk)
    return sequence[:n_contributors]


def _ensure_skill_supply_coverage(
    contributors: Sequence[Contributor],
    skills: Sequence[str],
    params: GenerationParameters,
    rng: random.Random,
) -> List[Contributor]:
    """Ensure the synthetic skill universe is observable in the text instance."""

    if not skills:
        return list(contributors)
    mutable = [dict(contributor.skills) for contributor in contributors]
    counts: Dict[str, int] = {skill: 0 for skill in skills}
    for skill_map in mutable:
        for skill in skill_map:
            counts[skill] = counts.get(skill, 0) + 1

    for missing in [skill for skill in skills if counts.get(skill, 0) <= 0]:
        placed = False
        under_capacity = [
            idx
            for idx, skill_map in enumerate(mutable)
            if len(skill_map) < params.skills_per_contributor_max
        ]
        if under_capacity:
            contributor_idx = rng.choice(under_capacity)
            mutable[contributor_idx][missing] = _truncated_normal_level(
                params.level_mean,
                params.level_sd,
                params.level_max,
                rng,
            )
            counts[missing] = 1
            placed = True
        else:
            replacement_options = [
                (idx, skill)
                for idx, skill_map in enumerate(mutable)
                for skill in skill_map
                if counts.get(skill, 0) > 1
            ]
            if replacement_options:
                contributor_idx, replaced = rng.choice(replacement_options)
                del mutable[contributor_idx][replaced]
                counts[replaced] -= 1
                mutable[contributor_idx][missing] = _truncated_normal_level(
                    params.level_mean,
                    params.level_sd,
                    params.level_max,
                    rng,
                )
                counts[missing] = 1
                placed = True
        if not placed:
            break

    return [
        Contributor(name=contributor.name, skills=dict(sorted(skill_map.items())))
        for contributor, skill_map in zip(contributors, mutable)
    ]


def _profile_adjusted_weights(
    skills: Sequence[str],
    base_weights: Sequence[float],
    preferred: set[str],
    profile_bias: float,
) -> List[float]:
    weights = [
        weight * (profile_bias if skill in preferred else 1.0)
        for skill, weight in zip(skills, base_weights)
    ]
    total = sum(weights)
    return [weight / total for weight in weights]


def _truncated_normal_level(
    level_mean: float, level_sd: float, level_max: int, rng: random.Random
) -> int:
    for _ in range(200):
        value = int(round(rng.gauss(level_mean, level_sd)))
        if 1 <= value <= level_max:
            return value
    return min(level_max, max(1, int(round(level_mean))))


def _role_skill_weights(
    skills: Sequence[str],
    base_weights: Sequence[float],
    supply_counts: Dict[str, int],
    params: GenerationParameters,
    empirical_profile: EmpiricalSourceProfile | None = None,
) -> List[float]:
    if not skills:
        return []
    total_supply = sum(supply_counts.values())
    if total_supply <= 0:
        supply_probs = {skill: 1.0 / len(skills) for skill in skills}
    else:
        supply_probs = {skill: supply_counts.get(skill, 0) / total_supply for skill in skills}
    uniform = 1.0 / len(skills)
    weights = []
    empirical_demand = None
    if empirical_profile and len(empirical_profile.demand_weights) == len(skills):
        empirical_demand = normalize_weights(empirical_profile.demand_weights, len(skills))
    for idx, (skill, base) in enumerate(zip(skills, base_weights)):
        aligned = supply_probs.get(skill, 0.0)
        if empirical_demand is None:
            unaligned = 0.5 * uniform + 0.5 * base
        else:
            unaligned = empirical_demand[idx]
        weights.append(
            params.role_skill_alignment * aligned
            + (1.0 - params.role_skill_alignment) * unaligned
        )
    total = sum(weights)
    return [weight / total for weight in weights]


def _project_duration(
    params: GenerationParameters,
    rng: random.Random,
    empirical_profile: EmpiricalSourceProfile | None = None,
    source_idx: int | None = None,
) -> int:
    if (
        empirical_profile
        and source_idx is not None
        and 0 <= source_idx < len(empirical_profile.duration_values)
    ):
        value = int(round(empirical_profile.duration_values[source_idx]))
        return max(params.duration_min, min(params.duration_max, max(1, value)))
    if empirical_profile and empirical_profile.duration_values:
        value = int(round(rng.choice(empirical_profile.duration_values)))
        return max(params.duration_min, min(params.duration_max, max(1, value)))
    return rng.randint(params.duration_min, params.duration_max)


def _empirical_project_indices(
    empirical_profile: EmpiricalSourceProfile | None,
    n_projects: int,
    rng: random.Random,
) -> List[int]:
    if not empirical_profile or not empirical_profile.activity_role_skill_level_rows:
        return []
    count = min(
        len(empirical_profile.activity_role_skill_level_rows),
        len(empirical_profile.duration_values) if empirical_profile.duration_values else n_projects,
    )
    if count <= 0:
        return []
    indices = list(range(count))
    sequence = []
    while len(sequence) < n_projects:
        chunk = list(indices)
        rng.shuffle(chunk)
        sequence.extend(chunk)
    return sequence[:n_projects]


def _empirical_project_roles(
    empirical_profile: EmpiricalSourceProfile | None,
    source_idx: int | None,
    skills: Sequence[str],
    params: GenerationParameters,
) -> List[Role]:
    if (
        not empirical_profile
        or source_idx is None
        or source_idx < 0
        or source_idx >= len(empirical_profile.activity_role_skill_level_rows)
    ):
        return []
    roles = []
    for skill_id, required_level in empirical_profile.activity_role_skill_level_rows[source_idx]:
        skill_index = int(skill_id) - 1
        if 0 <= skill_index < len(skills):
            roles.append(
                Role(
                    skill=skills[skill_index],
                    level=max(1, min(params.role_level_max, int(required_level))),
                )
            )
    return roles


def _required_level(
    skill: str,
    supply_index: Dict[str, List[int]],
    params: GenerationParameters,
    rng: random.Random,
) -> int:
    levels = supply_index.get(skill, [])
    if levels:
        reference = _quantile(levels, params.supply_quantile)
    else:
        reference = max(1, min(params.level_max, int(round(params.level_mean))))
    category = weighted_choice(
        ["over", "match", "mentor", "future"],
        [params.p_over, params.p_match, params.p_mentor, params.p_future],
        rng,
    )
    if category == "over":
        level = rng.randint(1, max(1, reference - 1))
    elif category == "match":
        level = reference
    elif category == "mentor":
        level = reference + 1
    else:
        level = reference + rng.randint(params.future_offset_min, params.future_offset_max)
    return max(1, min(params.role_level_max, int(level)))


def _quantile(sorted_values: Sequence[int], quantile: float) -> int:
    if not sorted_values:
        return 1
    if len(sorted_values) == 1:
        return int(sorted_values[0])
    position = min(1.0, max(0.0, quantile)) * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return int(sorted_values[lower])
    value = sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (
        position - lower
    )
    return max(1, int(round(value)))
