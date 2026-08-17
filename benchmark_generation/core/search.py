"""Adaptive forward-generation search used by coverage-driven families."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Set

from .distance import FeatureScaler, weighted_distance
from .features import extract_features
from .generator import GenerationOutcome, GenerationParameters, derive_seed, generate_valid_instance
from .parameter_sampling import perturb_parameters, sample_from_bounds
from .validator import structural_hash


@dataclass
class SearchCandidate:
    outcome: GenerationOutcome
    features: Dict[str, float]
    normalized_features: Dict[str, float]
    structural_hash: str
    objective: float
    target_distance: float
    nearest_existing_distance: float
    radius_distance: float | None = None
    radius_constraint_pass: bool = True


@dataclass
class SearchResult:
    candidate: SearchCandidate
    evaluations: int
    log: List[Dict[str, Any]]
    best_candidate_found: SearchCandidate | None = None
    generation_failures: int = 0


def adaptive_parameter_search(
    target_vector: Mapping[str, float],
    scaler: FeatureScaler,
    bounds: Mapping[str, tuple[float, float]],
    config: Mapping[str, Any],
    seed: int,
    accepted_vectors: Sequence[Mapping[str, float]],
    existing_hashes: Set[str],
    anchor_vector: Mapping[str, float] | None = None,
    radius_min: float | None = None,
    radius_max: float | None = None,
    source_name: str | None = None,
    initial_population: Sequence[GenerationParameters] | None = None,
    hard_radius: bool = False,
    radius_tolerance_absolute: float = 1e-9,
    radius_tolerance_relative: float = 0.0,
) -> SearchResult:
    """Search over generator parameters while generating every candidate forward."""

    rng = random.Random(seed)
    population_size = int(config.get("population_size", 48))
    iterations = int(config.get("iterations", 4))
    elite_fraction = float(config.get("elite_fraction", 0.25))
    elite_count = max(1, int(round(population_size * elite_fraction)))
    novelty_lambda = float(config.get("novelty_penalty", 0.15))
    duplicate_penalty = float(config.get("duplicate_penalty", 10.0))
    min_distance = float(config.get("minimum_accepted_distance", 0.02))
    radius_penalty = float(config.get("radius_penalty", 1.0))
    max_generation_attempts = int(config.get("max_generation_attempts", 30))

    population: List[GenerationParameters] = list(initial_population or [])[:population_size]
    while len(population) < population_size:
        population.append(sample_from_bounds(bounds, rng))
    best: SearchCandidate | None = None
    best_any: SearchCandidate | None = None
    evaluations = 0
    generation_failures = 0
    search_log: List[Dict[str, Any]] = []

    for iteration in range(iterations):
        candidates: List[SearchCandidate] = []
        for params_idx, params in enumerate(population):
            evaluations += 1
            candidate_seed = derive_seed(seed, evaluations + iteration * 100_000)
            try:
                outcome = generate_valid_instance(
                    params,
                    candidate_seed,
                    max_attempts=max_generation_attempts,
                    source_name=source_name,
                )
            except RuntimeError as exc:
                generation_failures += 1
                search_log.append(
                    {
                        "iteration": iteration,
                        "candidate": params_idx,
                        "status": "invalid",
                        "reason": str(exc),
                    }
                )
                continue

            features = extract_features(outcome.instance)
            normalized = scaler.normalize(features)
            h = structural_hash(outcome.instance)
            target_distance = weighted_distance(normalized, target_vector, scaler.group_weights)
            nearest_existing = _nearest_distance(normalized, accepted_vectors, scaler.group_weights)
            objective = target_distance
            if h in existing_hashes:
                objective += duplicate_penalty
            if nearest_existing < min_distance:
                objective += novelty_lambda * (min_distance - nearest_existing)
            radius_distance = None
            radius_constraint_pass = True
            if anchor_vector is not None:
                radius_distance = weighted_distance(normalized, anchor_vector, scaler.group_weights)
                radius_constraint_pass = radius_in_bounds(
                    radius_distance,
                    radius_min,
                    radius_max,
                    radius_tolerance_absolute=radius_tolerance_absolute,
                    radius_tolerance_relative=radius_tolerance_relative,
                )
                if radius_min is not None and radius_distance < radius_min:
                    objective += radius_penalty * (radius_min - radius_distance)
                if radius_max is not None and radius_distance > radius_max:
                    objective += radius_penalty * (radius_distance - radius_max)
                if hard_radius and not radius_constraint_pass:
                    objective += float(config.get("hard_radius_violation_penalty", 100.0))

            candidate = SearchCandidate(
                outcome=outcome,
                features=features,
                normalized_features=normalized,
                structural_hash=h,
                objective=objective,
                target_distance=target_distance,
                nearest_existing_distance=nearest_existing,
                radius_distance=radius_distance,
                radius_constraint_pass=radius_constraint_pass,
            )
            candidates.append(candidate)
            if h not in existing_hashes and (
                best_any is None or candidate.objective < best_any.objective
            ):
                best_any = candidate
            can_accept = h not in existing_hashes and (
                not hard_radius or candidate.radius_constraint_pass
            )
            if can_accept and (best is None or candidate.objective < best.objective):
                best = candidate

        if not candidates:
            population = [sample_from_bounds(bounds, rng) for _ in range(population_size)]
            continue

        candidates.sort(key=lambda item: item.objective)
        search_log.append(
            {
                "iteration": iteration,
                "status": "evaluated",
                "best_objective": candidates[0].objective,
                "best_target_distance": candidates[0].target_distance,
                "best_radius_distance": candidates[0].radius_distance,
                "best_radius_constraint_pass": candidates[0].radius_constraint_pass,
                "evaluated": len(candidates),
            }
        )
        elites = [candidate.outcome.parameters for candidate in candidates[:elite_count]]
        scale = max(0.05, float(config.get("initial_perturbation_scale", 0.35)) * (0.65 ** iteration))
        next_population: List[GenerationParameters] = list(elites)
        while len(next_population) < population_size:
            if rng.random() < float(config.get("random_restart_rate", 0.20)):
                next_population.append(sample_from_bounds(bounds, rng))
            else:
                parent = rng.choice(elites)
                next_population.append(perturb_parameters(parent, bounds, rng, scale=scale))
        population = next_population

    if best is None:
        if best_any is None:
            raise RuntimeError("Search failed to produce any non-duplicate valid candidate")
        raise RuntimeError(
            "Search produced candidates but none satisfied the hard radius constraint. "
            f"Closest accepted-style radius={best_any.radius_distance}, "
            f"target_distance={best_any.target_distance}, evaluations={evaluations}, "
            f"generation_failures={generation_failures}"
        )
    return SearchResult(
        candidate=best,
        evaluations=evaluations,
        log=search_log,
        best_candidate_found=best_any,
        generation_failures=generation_failures,
    )


def _nearest_distance(
    vector: Mapping[str, float],
    accepted_vectors: Sequence[Mapping[str, float]],
    group_weights: Mapping[str, float] | None,
) -> float:
    if not accepted_vectors:
        return float("inf")
    return min(weighted_distance(vector, other, group_weights) for other in accepted_vectors)


def radius_in_bounds(
    radius: float,
    radius_min: float | None,
    radius_max: float | None,
    radius_tolerance_absolute: float = 1e-9,
    radius_tolerance_relative: float = 0.0,
) -> bool:
    tolerance = max(
        float(radius_tolerance_absolute),
        abs(float(radius)) * float(radius_tolerance_relative),
    )
    if radius_min is not None and radius + tolerance < radius_min:
        return False
    if radius_max is not None and radius - tolerance > radius_max:
        return False
    return True
