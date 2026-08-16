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


@dataclass
class SearchResult:
    candidate: SearchCandidate
    evaluations: int
    log: List[Dict[str, Any]]


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
    evaluations = 0
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
            if anchor_vector is not None:
                radius_distance = weighted_distance(normalized, anchor_vector, scaler.group_weights)
                if radius_min is not None and radius_distance < radius_min:
                    objective += radius_penalty * (radius_min - radius_distance)
                if radius_max is not None and radius_distance > radius_max:
                    objective += radius_penalty * (radius_distance - radius_max)

            candidate = SearchCandidate(
                outcome=outcome,
                features=features,
                normalized_features=normalized,
                structural_hash=h,
                objective=objective,
                target_distance=target_distance,
                nearest_existing_distance=nearest_existing,
                radius_distance=radius_distance,
            )
            candidates.append(candidate)
            if h not in existing_hashes and (best is None or candidate.objective < best.objective):
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
        raise RuntimeError("Search failed to produce any non-duplicate valid candidate")
    return SearchResult(candidate=best, evaluations=evaluations, log=search_log)


def _nearest_distance(
    vector: Mapping[str, float],
    accepted_vectors: Sequence[Mapping[str, float]],
    group_weights: Mapping[str, float] | None,
) -> float:
    if not accepted_vectors:
        return float("inf")
    return min(weighted_distance(vector, other, group_weights) for other in accepted_vectors)
