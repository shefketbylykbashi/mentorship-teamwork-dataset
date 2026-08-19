"""Empirical source-profile derivation for Type 5."""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from benchmark_generation.core.generator import EmpiricalSourceProfile, GenerationParameters, derive_seed
from benchmark_generation.core.parameter_sampling import quantile
from benchmark_generation.type5.sources import ObservationWindow, SourceRepo, parse_github_time, repo_cache_dir
from benchmark_generation.type5.taxonomy import (
    compile_bot_patterns,
    is_bot_identity,
    is_docs_only_path,
    parent_language_token,
    skill_token_for_path,
    should_exclude_path,
)


def cache_body(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "body" in data:
        return data["body"]
    return data


def activity_score(lines_changed: float, files_touched: float, prs_touching_skill: float) -> float:
    return (
        0.60 * math.log1p(max(0.0, lines_changed))
        + 0.25 * math.log1p(max(0.0, files_touched))
        + 0.15 * math.log1p(max(0.0, prs_touching_skill))
    )


def percentile_level(rank: int, count: int) -> int:
    if count <= 1:
        percentile_rank = 1.0
    else:
        percentile_rank = rank / (count - 1)
    return max(1, min(10, 1 + int(math.floor(9 * percentile_rank))))


def sample_type5_difficulty_prior(seed: int, config: Mapping[str, Any]) -> Tuple[float, float, float, float]:
    prior = config.get("difficulty_prior", {})
    alpha = [float(value) for value in prior.get("dirichlet_alpha", [25, 45, 20, 10])]
    mentor_bounds = prior.get("mentor_bounds", [0.15, 0.30])
    future_bounds = prior.get("future_bounds", [0.05, 0.15])
    rng = random.Random(seed)
    draws = [rng.gammavariate(max(1e-9, value), 1.0) for value in alpha]
    total = sum(draws) or 1.0
    values = [draw / total for draw in draws]
    values[2] = _clamp(values[2], float(mentor_bounds[0]), float(mentor_bounds[1]))
    values[3] = _clamp(values[3], float(future_bounds[0]), float(future_bounds[1]))
    total = sum(values) or 1.0
    return tuple(value / total for value in values)  # type: ignore[return-value]


def derive_window_measurement(
    repo: SourceRepo,
    window: ObservationWindow,
    cache_dir: str | Path,
    config: Mapping[str, Any],
    bot_config: Mapping[str, Any],
    seed: int,
) -> Dict[str, Any]:
    """Measure one repository/window from cached GitHub responses.

    The returned record contains aggregate public activity only. Raw GitHub
    usernames are used transiently for counting and then replaced by local
    anonymized contributor IDs before returning.
    """

    repo_dir = repo_cache_dir(cache_dir, repo)
    window_dir = repo_dir / window.name
    repository = cache_body(repo_dir / "repository.json")
    branch = cache_body(repo_dir / "default_branch.json", {})
    languages = cache_body(repo_dir / "languages.json", {})
    releases = cache_body(repo_dir / "releases.json", [])
    milestones = cache_body(repo_dir / "milestones.json", [])
    merged_prs = cache_body(window_dir / "merged_pull_requests.json", [])
    sampled_prs = cache_body(window_dir / "sampled_pull_requests.json", [])
    commits = cache_body(window_dir / "commits.json", [])
    if repository is None:
        return _failed_measurement(repo, window, "missing_repository_cache")
    if not isinstance(merged_prs, list) or not isinstance(sampled_prs, list):
        return _failed_measurement(repo, window, "missing_window_pr_cache")

    bot_patterns = compile_bot_patterns(bot_config)
    observed_authors = _observed_authors(merged_prs, commits)
    observed_author_set = {author for author in observed_authors if author}
    null_author_count = sum(1 for author in observed_authors if not author)
    bot_authors = {author for author in observed_author_set if is_bot_identity(author, bot_patterns)}
    human_authors = sorted(
        author for author in observed_author_set if author and author not in bot_authors
    )
    anonymized = {author: f"u{idx:05d}" for idx, author in enumerate(human_authors)}

    file_events: List[Dict[str, Any]] = []
    excluded_files = 0
    docs_only_prs = 0
    sampled_human_prs = []
    for pr in sampled_prs:
        author = _pr_author(pr)
        if not author or author not in anonymized:
            continue
        number = int(pr.get("number", 0))
        files = cache_body(window_dir / "pr_files" / f"{number}.json", [])
        if not isinstance(files, list):
            files = []
        pr_events = []
        docs_or_excluded = True
        for file_info in files:
            path = str(file_info.get("filename") or "")
            if should_exclude_path(path):
                excluded_files += 1
                continue
            token = skill_token_for_path(path)
            if token is None:
                if not is_docs_only_path(path):
                    excluded_files += 1
                continue
            docs_or_excluded = False
            additions = int(file_info.get("additions") or 0)
            deletions = int(file_info.get("deletions") or 0)
            lines = max(0, additions + deletions)
            pr_events.append(
                {
                    "anon_author": anonymized[author],
                    "pr_number": number,
                    "raw_skill": token,
                    "lines": lines,
                    "files": 1,
                    "created_at": pr.get("created_at"),
                    "merged_at": pr.get("merged_at"),
                    "milestone": pr.get("milestone"),
                }
            )
        if pr_events:
            sampled_human_prs.append(pr)
            file_events.extend(pr_events)
        elif docs_or_excluded:
            docs_only_prs += 1

    if not file_events:
        return _failed_measurement(repo, window, "no_technical_file_activity")

    skill_map = _retained_skill_map(file_events, config)
    retained_events = [
        {**event, "skill": skill_map[event["raw_skill"]]}
        for event in file_events
        if skill_map.get(event["raw_skill"])
    ]
    skill_tokens = sorted({event["skill"] for event in retained_events})
    skill_min, skill_max = config.get("skill_count_range", [5, 200])
    if len(skill_tokens) > int(skill_max):
        skill_tokens = _top_skills(retained_events, int(skill_max))
        retained_events = [event for event in retained_events if event["skill"] in set(skill_tokens)]
    if len(skill_tokens) < int(skill_min):
        return _failed_measurement(repo, window, "fewer_than_five_retained_skills")

    contributor_skill_activity = _contributor_skill_activity(retained_events)
    levels = _activity_levels(contributor_skill_activity, skill_tokens)
    active_contributors = sorted(levels.keys())
    if len(active_contributors) < int(config.get("min_human_contributors", 5)):
        return _failed_measurement(repo, window, "fewer_than_five_skill_evidenced_human_contributors")

    pr_work_units = _work_units(sampled_human_prs, retained_events, skill_tokens, releases, config)
    if not pr_work_units:
        return _failed_measurement(repo, window, "no_technical_work_units")

    role_counts = [unit["role_count"] for unit in pr_work_units]
    durations = [unit["duration"] for unit in pr_work_units]
    deadline_observations = [unit["tightness"] for unit in pr_work_units if unit.get("tightness") is not None]
    effort_densities = [unit["effort_density"] for unit in pr_work_units if unit["effort_density"] > 0]
    demand_counts = Counter()
    for unit in pr_work_units:
        demand_counts.update(unit["skills"])
    supply_counts = Counter()
    for contributor_levels in levels.values():
        supply_counts.update(contributor_levels.keys())

    matrix = _activity_matrix(contributor_skill_activity, active_contributors, skill_tokens)
    profile_result = choose_profile_clusters(
        matrix,
        min_k=int(config.get("profile_count_range", [3, 8])[0]),
        max_k=int(config.get("profile_count_range", [3, 8])[1]),
        seed=seed,
    )
    if not profile_result["valid"]:
        return _failed_measurement(repo, window, str(profile_result["reason"]))

    skill_supply_weights = _weights_from_counts(supply_counts, skill_tokens)
    skill_demand_weights = _weights_from_counts(demand_counts, skill_tokens)
    contributor_skill_counts = [len(levels[contributor]) for contributor in active_contributors]
    level_values = [
        level
        for contributor in active_contributors
        for level in levels[contributor].values()
    ]
    empirical_alignment = cosine_similarity(supply_counts, demand_counts, skill_tokens)
    measurement = {
        "valid": True,
        "failure_reason": "",
        "source": asdict(repo),
        "window": asdict(window),
        "repository_id": repository.get("id"),
        "default_branch": repository.get("default_branch"),
        "snapshot_sha": (((branch.get("commit") or {}).get("sha")) if isinstance(branch, dict) else ""),
        "collection_timestamp": _cache_timestamp(repo_dir / "collection_summary.json"),
        "license": ((repository.get("license") or {}).get("spdx_id") if isinstance(repository.get("license"), dict) else ""),
        "languages": languages if isinstance(languages, dict) else {},
        "merged_pr_count": len(merged_prs),
        "sampled_pr_count": len(sampled_human_prs),
        "technical_work_unit_count": len(pr_work_units),
        "total_contributors_observed": len(observed_author_set),
        "human_contributor_count": len(active_contributors),
        "bot_count": len(bot_authors),
        "null_author_count": null_author_count,
        "excluded_file_count": excluded_files,
        "docs_only_pr_count": docs_only_prs,
        "skill_tokens": skill_tokens,
        "skill_count": len(skill_tokens),
        "skill_supply_weights": skill_supply_weights,
        "skill_demand_weights": skill_demand_weights,
        "profile_count": int(profile_result["k"]),
        "profile_silhouette": float(profile_result["silhouette"]),
        "profile_sizes": profile_result["profile_sizes"],
        "profile_centroids": profile_result["centroids"],
        "largest_profile_share": max(profile_result["profile_sizes"]) / max(1, len(active_contributors)),
        "source_skill_skew": skill_skew_from_counts(supply_counts),
        "source_alignment": empirical_alignment,
        "source_supply_concentration": entropy_concentration(supply_counts),
        "source_demand_concentration": entropy_concentration(demand_counts),
        "skills_per_contributor_values": contributor_skill_counts,
        "skills_per_contributor_mean": _mean(contributor_skill_counts),
        "roles_per_work_unit_values": role_counts,
        "roles_per_work_unit_mean": _mean(role_counts),
        "roles_per_work_unit_median": _median(role_counts),
        "duration_values": durations,
        "duration_p10": percentile(durations, 0.10),
        "duration_median": percentile(durations, 0.50),
        "duration_p90": percentile(durations, 0.90),
        "deadline_observations": deadline_observations,
        "deadline_proxy_coverage": len(deadline_observations) / max(1, len(pr_work_units)),
        "deadline_proxy_mode_raw": _deadline_mode(pr_work_units),
        "effort_densities": effort_densities,
        "level_mean": _mean(level_values),
        "level_sd": _sd(level_values),
        "levels": _level_summary(levels),
        "supply_quantile_estimate": supply_quantile_estimate(retained_events, contributor_skill_activity, skill_tokens),
    }
    quality = source_window_quality(measurement, config)
    measurement["source_quality"] = quality
    measurement["valid"] = bool(quality["passed"])
    measurement["failure_reason"] = "" if quality["passed"] else ";".join(quality["issues"])
    return measurement


def finalize_source_profile(
    measurement: Mapping[str, Any],
    config: Mapping[str, Any],
    seed: int,
    family_instance_id: int,
    source_replacement: bool = False,
    original_source: Mapping[str, Any] | None = None,
    replacement_reason: str = "",
    corpus_deadline_bounds: Tuple[float, float] = (0.8, 4.0),
    global_score_k: float = 1.0,
) -> Dict[str, Any]:
    params = parameters_from_measurement(
        measurement,
        config,
        seed=seed,
        corpus_deadline_bounds=corpus_deadline_bounds,
        global_score_k=global_score_k,
    )
    source = dict(measurement["source"])
    window = dict(measurement["window"])
    source_profile = {
        "skill_tokens": list(measurement["skill_tokens"]),
        "skill_weights": list(measurement["skill_supply_weights"]),
        "demand_weights": list(measurement["skill_demand_weights"]),
        "profile_centroids": measurement["profile_centroids"],
        "profile_sizes": measurement["profile_sizes"],
    }
    source_metrics = {
        "source_merged_pr_count": int(measurement["merged_pr_count"]),
        "source_sampled_pr_count": int(measurement["sampled_pr_count"]),
        "source_human_contributor_count": int(measurement["human_contributor_count"]),
        "source_bot_count": int(measurement["bot_count"]),
        "source_skill_count": int(measurement["skill_count"]),
        "source_profile_count": int(measurement["profile_count"]),
        "source_profile_silhouette": float(measurement["profile_silhouette"]),
        "source_skill_skew": float(measurement["source_skill_skew"]),
        "source_alignment": float(measurement["source_alignment"]),
        "source_skills_per_contributor_mean": float(measurement["skills_per_contributor_mean"]),
        "source_roles_per_work_unit_mean": float(measurement["roles_per_work_unit_mean"]),
        "source_duration_p10": float(measurement["duration_p10"]),
        "source_duration_median": float(measurement["duration_median"]),
        "source_duration_p90": float(measurement["duration_p90"]),
        "source_deadline_proxy_coverage": float(measurement["deadline_proxy_coverage"]),
        "source_deadline_proxy_mode": params["deadline_proxy_mode"],
        "source_supply_concentration": float(measurement["source_supply_concentration"]),
        "source_demand_concentration": float(measurement["source_demand_concentration"]),
        "source_largest_profile_share": float(measurement["largest_profile_share"]),
        "source_scale_compression_factor": float(params["scale_compression_factor"]),
    }
    return {
        "family_instance_id": family_instance_id,
        "source_owner": source["owner"],
        "source_repo": source["repo"],
        "source_url": source["url"],
        "source_domain": source["domain"],
        "source_repository_id": measurement.get("repository_id", ""),
        "source_default_branch": measurement.get("default_branch", ""),
        "source_snapshot_sha": measurement.get("snapshot_sha", ""),
        "source_window_name": window["name"],
        "source_window_start": window["start"],
        "source_window_end": window["end"],
        "source_collection_timestamp": measurement.get("collection_timestamp", ""),
        "source_license": measurement.get("license", ""),
        "source_replacement": bool(source_replacement),
        "original_source_if_replaced": (
            f"{original_source.get('owner')}/{original_source.get('repo')}" if original_source else ""
        ),
        "replacement_reason": replacement_reason,
        "source_metrics": source_metrics,
        "parameters": params["parameters"],
        "nominal_difficulty_mix": params["nominal_difficulty_mix"],
        "source_profile": source_profile,
        "proxy_flags": {
            "skill_level_proxy": "activity_percentile",
            "project_unit_proxy": "merged_pull_request",
            "deadline_proxy": params["deadline_proxy_mode"],
            "score_proxy": "effort_conditioned_synthetic",
        },
        "quality_source": measurement.get("source_quality", {}),
    }


def empirical_profile_from_record(record: Mapping[str, Any]) -> EmpiricalSourceProfile:
    profile = record["source_profile"]
    return EmpiricalSourceProfile(
        skill_tokens=tuple(str(value) for value in profile.get("skill_tokens", [])),
        skill_weights=tuple(float(value) for value in profile.get("skill_weights", [])),
        demand_weights=tuple(float(value) for value in profile.get("demand_weights", [])),
        profile_centroids=tuple(
            tuple(float(value) for value in row)
            for row in profile.get("profile_centroids", [])
        ),
        profile_sizes=tuple(int(value) for value in profile.get("profile_sizes", [])),
    )


def parameters_from_measurement(
    measurement: Mapping[str, Any],
    config: Mapping[str, Any],
    seed: int,
    corpus_deadline_bounds: Tuple[float, float],
    global_score_k: float,
) -> Dict[str, Any]:
    skill_count = int(measurement["skill_count"])
    contributors = int(measurement["human_contributor_count"])
    projects = int(measurement["merged_pr_count"])
    compression = scale_compression_factor(contributors, projects, config)
    n_contributors = max(5, int(round(contributors * compression)))
    n_projects = max(5, int(round(projects * compression)))
    skills_per_values = [int(value) for value in measurement["skills_per_contributor_values"]]
    roles_values = [int(value) for value in measurement["roles_per_work_unit_values"]]
    duration_values = [int(value) for value in measurement["duration_values"]]
    tightness_values = [
        float(value)
        for value in measurement.get("deadline_observations", [])
        if math.isfinite(float(value))
    ]
    deadline_mode = str(measurement.get("deadline_proxy_mode_raw") or "corpus_fallback")
    if len(tightness_values) >= int(config.get("deadline_min_observations", 10)):
        tightness_min = percentile(tightness_values, 0.10)
        tightness_max = percentile(tightness_values, 0.90)
    else:
        tightness_min, tightness_max = corpus_deadline_bounds
        deadline_mode = "corpus_fallback" if not tightness_values else "mixed"
    tight_low, tight_high = config.get("tightness_range", [0.5, 6.0])
    tightness_min = _clamp(tightness_min, float(tight_low), float(tight_high))
    tightness_max = max(tightness_min, _clamp(tightness_max, float(tight_low), float(tight_high)))

    effort_densities = [float(value) for value in measurement.get("effort_densities", []) if float(value) > 0]
    normalized_effort_density = _median(effort_densities) * global_score_k
    score_low, score_high = config.get("score_base_range", [0.5, 5.0])
    score_base = _clamp(normalized_effort_density, float(score_low), float(score_high))
    p10_effort = max(1e-9, percentile(effort_densities, 0.10))
    p90_effort = max(p10_effort, percentile(effort_densities, 0.90))
    pert_low, pert_high = config.get("score_perturbation_range", [1.0, 3.0])
    score_perturbation = _clamp(p90_effort / p10_effort, float(pert_low), float(pert_high))
    p_over, p_match, p_mentor, p_future = sample_type5_difficulty_prior(seed, config)
    duration_low, duration_high = config.get("duration_range", [1, 500])
    roles_low, roles_high = config.get("roles_per_project_range", [1, 20])
    target_alignment = _clamp(max(0.60, float(measurement["source_alignment"])), 0.60, 0.95)
    params = GenerationParameters(
        n_skills=skill_count,
        skill_skew=float(measurement["source_skill_skew"]),
        n_contributors=n_contributors,
        skills_per_contributor_min=max(1, min(skill_count, int(round(percentile(skills_per_values, 0.10))))),
        skills_per_contributor_max=max(1, min(20, skill_count, int(round(percentile(skills_per_values, 0.90))))),
        level_mean=float(measurement["level_mean"]),
        level_sd=max(0.5, float(measurement["level_sd"])),
        level_max=10,
        n_profiles=int(measurement["profile_count"]),
        n_projects=n_projects,
        duration_min=max(int(duration_low), int(round(percentile(duration_values, 0.10)))),
        duration_max=min(int(duration_high), max(1, int(round(percentile(duration_values, 0.90))))),
        roles_min=max(int(roles_low), int(round(percentile(roles_values, 0.10)))),
        roles_max=min(int(roles_high), max(1, int(round(percentile(roles_values, 0.90))))),
        score_base=score_base,
        score_perturbation=score_perturbation,
        tightness_min=tightness_min,
        tightness_max=tightness_max,
        role_skill_alignment=target_alignment,
        supply_quantile=float(measurement.get("supply_quantile_estimate", 0.75)),
        p_over=p_over,
        p_match=p_match,
        p_mentor=p_mentor,
        p_future=p_future,
        fillable_fraction=float(config.get("fillable_fraction", 0.30)),
        role_level_max=int(config.get("role_level_max", 100)),
        future_offset_min=int(config.get("future_offset_min", 2)),
        future_offset_max=int(config.get("future_offset_max", 4)),
    ).normalized()
    data = params.to_dict()
    data["duration_max"] = max(int(data["duration_min"]), int(data["duration_max"]))
    data["roles_max"] = max(int(data["roles_min"]), int(data["roles_max"]))
    return {
        "parameters": GenerationParameters(**data).normalized().to_dict(),
        "nominal_difficulty_mix": {
            "p_over": p_over,
            "p_match": p_match,
            "p_mentor": p_mentor,
            "p_future": p_future,
            "prior": "dirichlet_concentration_[25,45,20,10]_with_mentor_future_bounds",
        },
        "deadline_proxy_mode": deadline_mode,
        "scale_compression_factor": compression,
    }


def source_window_quality(measurement: Mapping[str, Any], config: Mapping[str, Any]) -> Dict[str, Any]:
    issues = []
    if int(measurement.get("human_contributor_count", 0)) < int(config.get("min_human_contributors", 5)):
        issues.append("human_contributors_below_minimum")
    if int(measurement.get("merged_pr_count", 0)) < int(config.get("min_merged_prs", 5)):
        issues.append("merged_prs_below_minimum")
    if int(measurement.get("skill_count", 0)) < int(config.get("min_retained_skills", 5)):
        issues.append("retained_skills_below_minimum")
    if int(measurement.get("technical_work_unit_count", 0)) <= 0:
        issues.append("no_technical_work_units")
    profile_count = int(measurement.get("profile_count", 0))
    low, high = config.get("profile_count_range", [3, 8])
    if profile_count < int(low) or profile_count > int(high):
        issues.append("profile_count_out_of_range")
    if float(measurement.get("source_alignment", 0.0)) <= 0.0:
        issues.append("nonzero_alignment_unavailable")
    return {"passed": not issues, "issues": issues}


def choose_profile_clusters(
    matrix: Sequence[Sequence[float]],
    min_k: int,
    max_k: int,
    seed: int,
) -> Dict[str, Any]:
    n = len(matrix)
    if n < min_k:
        return {"valid": False, "reason": "not_enough_contributors_for_three_profiles"}
    max_k = min(max_k, n)
    best = None
    for k in range(min_k, max_k + 1):
        result = kmeans(matrix, k, seed=derive_seed(seed, k, salt=97_003))
        if any(size == 0 for size in result["profile_sizes"]):
            continue
        silhouette = approximate_silhouette(matrix, result["assignments"], k, seed=seed)
        candidate = {**result, "k": k, "silhouette": silhouette}
        if best is None or silhouette > best["silhouette"]:
            best = candidate
    if best is None:
        return {"valid": False, "reason": "profile_clustering_empty_profile"}
    return {"valid": True, **best}


def kmeans(matrix: Sequence[Sequence[float]], k: int, seed: int, iterations: int = 25) -> Dict[str, Any]:
    rng = random.Random(seed)
    rows = [list(row) for row in matrix]
    order = list(range(len(rows)))
    rng.shuffle(order)
    centers = [rows[order[idx % len(order)]][:] for idx in range(k)]
    assignments = [0] * len(rows)
    for _ in range(iterations):
        changed = False
        for idx, row in enumerate(rows):
            nearest = min(range(k), key=lambda cidx: squared_distance(row, centers[cidx]))
            if assignments[idx] != nearest:
                assignments[idx] = nearest
                changed = True
        new_centers = [[0.0 for _ in rows[0]] for _ in range(k)]
        counts = [0] * k
        for assignment, row in zip(assignments, rows):
            counts[assignment] += 1
            for col, value in enumerate(row):
                new_centers[assignment][col] += value
        for cidx in range(k):
            if counts[cidx] == 0:
                new_centers[cidx] = rows[order[cidx % len(order)]][:]
            else:
                new_centers[cidx] = [value / counts[cidx] for value in new_centers[cidx]]
        centers = [_normalize_vector(center) for center in new_centers]
        if not changed:
            break
    counts = [0] * k
    for assignment in assignments:
        counts[assignment] += 1
    return {"assignments": assignments, "profile_sizes": counts, "centroids": centers}


def approximate_silhouette(
    matrix: Sequence[Sequence[float]],
    assignments: Sequence[int],
    k: int,
    seed: int,
    max_points: int = 500,
) -> float:
    if len(matrix) < 2 or k < 2:
        return 0.0
    indices = list(range(len(matrix)))
    if len(indices) > max_points:
        rng = random.Random(seed)
        rng.shuffle(indices)
        indices = sorted(indices[:max_points])
    clusters: Dict[int, List[int]] = defaultdict(list)
    for idx in indices:
        clusters[assignments[idx]].append(idx)
    values = []
    for idx in indices:
        own = assignments[idx]
        own_members = [other for other in clusters[own] if other != idx]
        a = _mean([math.sqrt(squared_distance(matrix[idx], matrix[other])) for other in own_members])
        b_values = []
        for cluster, members in clusters.items():
            if cluster == own or not members:
                continue
            b_values.append(_mean([math.sqrt(squared_distance(matrix[idx], matrix[other])) for other in members]))
        b = min(b_values) if b_values else 0.0
        denom = max(a, b)
        values.append(0.0 if denom <= 0 else (b - a) / denom)
    return _mean(values)


def squared_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return sum((float(x) - float(y)) ** 2 for x, y in zip(a, b))


def percentile(values: Sequence[float | int], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(float(value) for value in values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = _clamp(q, 0.0, 1.0) * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def entropy_concentration(counts: Mapping[str, int | float]) -> float:
    positives = [float(value) for value in counts.values() if value > 0]
    total = sum(positives)
    if total <= 0 or len(positives) <= 1:
        return 0.0
    entropy = -sum((value / total) * math.log(value / total) for value in positives)
    return max(0.0, min(1.0, 1.0 - entropy / math.log(len(positives))))


def cosine_similarity(
    supply: Mapping[str, int | float],
    demand: Mapping[str, int | float],
    skill_tokens: Sequence[str],
) -> float:
    numerator = sum(float(supply.get(skill, 0.0)) * float(demand.get(skill, 0.0)) for skill in skill_tokens)
    norm_a = math.sqrt(sum(float(supply.get(skill, 0.0)) ** 2 for skill in skill_tokens))
    norm_b = math.sqrt(sum(float(demand.get(skill, 0.0)) ** 2 for skill in skill_tokens))
    return numerator / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0


def skill_skew_from_counts(counts: Mapping[str, int | float]) -> float:
    values = sorted([max(1.0, float(value)) for value in counts.values() if value > 0], reverse=True)
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


def scale_compression_factor(contributors: int, projects: int, config: Mapping[str, Any]) -> float:
    compression = config.get("scale_compression", {})
    max_contributors = int(compression.get("max_contributors", contributors))
    max_projects = int(compression.get("max_projects", projects))
    factors = [1.0]
    if contributors > max_contributors:
        factors.append(max_contributors / contributors)
    if projects > max_projects:
        factors.append(max_projects / projects)
    return min(factors)


def _observed_authors(prs: Sequence[Mapping[str, Any]], commits: Sequence[Mapping[str, Any]]) -> List[str | None]:
    authors: List[str | None] = []
    for pr in prs:
        authors.append(_pr_author(pr))
    for commit in commits:
        author = commit.get("author")
        if isinstance(author, dict):
            authors.append(author.get("login"))
    return authors


def _pr_author(pr: Mapping[str, Any]) -> str | None:
    user = pr.get("user")
    if isinstance(user, dict):
        login = user.get("login")
        return str(login) if login else None
    value = pr.get("author_login")
    return str(value) if value else None


def _retained_skill_map(events: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> Dict[str, str]:
    by_skill_contributors: Dict[str, set[str]] = defaultdict(set)
    by_skill_prs: Dict[str, set[int]] = defaultdict(set)
    for event in events:
        skill = str(event["raw_skill"])
        by_skill_contributors[skill].add(str(event["anon_author"]))
        by_skill_prs[skill].add(int(event["pr_number"]))
    mapping = {}
    for skill in by_skill_contributors:
        if len(by_skill_contributors[skill]) >= 2 or len(by_skill_prs[skill]) >= 3:
            mapping[skill] = skill
        else:
            mapping[skill] = parent_language_token(skill)
    return mapping


def _top_skills(events: Sequence[Mapping[str, Any]], limit: int) -> List[str]:
    scores: Counter = Counter()
    for event in events:
        scores[str(event["skill"])] += int(event.get("lines", 0)) + 10
    return [skill for skill, _score in scores.most_common(limit)]


def _contributor_skill_activity(events: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    result: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(lambda: defaultdict(lambda: {"lines": 0, "files": 0, "prs": set()}))
    for event in events:
        author = str(event["anon_author"])
        skill = str(event["skill"])
        result[author][skill]["lines"] += int(event.get("lines", 0))
        result[author][skill]["files"] += int(event.get("files", 0))
        result[author][skill]["prs"].add(int(event["pr_number"]))
    return result


def _activity_levels(
    contributor_skill_activity: Mapping[str, Mapping[str, Mapping[str, Any]]],
    skill_tokens: Sequence[str],
) -> Dict[str, Dict[str, int]]:
    scores_by_skill: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for contributor, skills in contributor_skill_activity.items():
        for skill, values in skills.items():
            score = activity_score(values["lines"], values["files"], len(values["prs"]))
            scores_by_skill[skill].append((contributor, score))
    levels: Dict[str, Dict[str, int]] = defaultdict(dict)
    for skill in skill_tokens:
        ranked = sorted(scores_by_skill.get(skill, []), key=lambda item: (item[1], item[0]))
        for rank, (contributor, _score) in enumerate(ranked):
            levels[contributor][skill] = percentile_level(rank, len(ranked))
    return {contributor: dict(skills) for contributor, skills in levels.items() if skills}


def _activity_matrix(
    contributor_skill_activity: Mapping[str, Mapping[str, Mapping[str, Any]]],
    contributors: Sequence[str],
    skill_tokens: Sequence[str],
) -> List[List[float]]:
    rows = []
    for contributor in contributors:
        row = []
        for skill in skill_tokens:
            values = contributor_skill_activity.get(contributor, {}).get(skill)
            if not values:
                row.append(0.0)
            else:
                row.append(activity_score(values["lines"], values["files"], len(values["prs"])))
        rows.append(_normalize_vector(row))
    return rows


def _work_units(
    sampled_prs: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    skill_tokens: Sequence[str],
    releases: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    by_pr_skill_lines: Dict[int, Counter] = defaultdict(Counter)
    pr_metadata: Dict[int, Mapping[str, Any]] = {}
    for pr in sampled_prs:
        pr_metadata[int(pr.get("number", 0))] = pr
    for event in events:
        by_pr_skill_lines[int(event["pr_number"])][str(event["skill"])] += int(event.get("lines", 0))
    units = []
    retained_skill_set = set(skill_tokens)
    for number, counts in sorted(by_pr_skill_lines.items()):
        pr = pr_metadata.get(number, {})
        selected_skills = [
            skill
            for skill, _lines in counts.most_common(20)
            if skill in retained_skill_set
        ]
        if not selected_skills:
            continue
        duration = pr_cycle_duration(pr)
        if duration is None:
            continue
        total_lines = sum(counts[skill] for skill in selected_skills)
        role_count = len(selected_skills)
        effort = math.log1p(max(0, total_lines)) * math.sqrt(max(1, role_count))
        effort_density = effort / max(1, duration * role_count)
        tightness, mode = deadline_tightness(pr, duration, releases, config)
        units.append(
            {
                "pr_number": number,
                "skills": selected_skills,
                "role_count": role_count,
                "duration": duration,
                "lines": total_lines,
                "effort_density": effort_density,
                "tightness": tightness,
                "deadline_mode": mode,
            }
        )
    return units


def pr_cycle_duration(pr: Mapping[str, Any]) -> int | None:
    created = pr.get("created_at")
    merged = pr.get("merged_at")
    if not created or not merged:
        return None
    start = parse_github_time(str(created))
    end = parse_github_time(str(merged))
    delta_days = (end - start).total_seconds() / 86400.0
    if delta_days < 0:
        return None
    return max(1, int(math.ceil(delta_days)))


def deadline_tightness(
    pr: Mapping[str, Any],
    duration: int,
    releases: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> Tuple[float | None, str]:
    created = pr.get("created_at")
    if not created:
        return None, "corpus_fallback"
    created_dt = parse_github_time(str(created))
    milestone = pr.get("milestone")
    due_on = milestone.get("due_on") if isinstance(milestone, dict) else None
    if due_on:
        due_dt = parse_github_time(str(due_on))
        days = (due_dt - created_dt).total_seconds() / 86400.0
        if days > 0:
            return _clamp(days / max(1, duration), 0.5, 6.0), "milestone"
    max_days = float(config.get("max_release_deadline_days", 180))
    candidates = []
    for release in releases:
        if release.get("draft") or release.get("prerelease"):
            continue
        published = release.get("published_at")
        if not published:
            continue
        published_dt = parse_github_time(str(published))
        days = (published_dt - created_dt).total_seconds() / 86400.0
        if 0 < days <= max_days:
            candidates.append(days)
    if candidates:
        return _clamp(min(candidates) / max(1, duration), 0.5, 6.0), "release"
    return None, "corpus_fallback"


def supply_quantile_estimate(
    events: Sequence[Mapping[str, Any]],
    contributor_skill_activity: Mapping[str, Mapping[str, Mapping[str, Any]]],
    skill_tokens: Sequence[str],
) -> float:
    scores_by_skill: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    for contributor, skills in contributor_skill_activity.items():
        for skill, values in skills.items():
            scores_by_skill[skill].append(
                (contributor, activity_score(values["lines"], values["files"], len(values["prs"])))
            )
    rank_lookup: Dict[Tuple[str, str], float] = {}
    for skill, rows in scores_by_skill.items():
        ranked = sorted(rows, key=lambda item: (item[1], item[0]))
        for rank, (contributor, _score) in enumerate(ranked):
            percentile_rank = 1.0 if len(ranked) <= 1 else rank / (len(ranked) - 1)
            rank_lookup[(contributor, skill)] = percentile_rank
    observed = [
        rank_lookup[(str(event["anon_author"]), str(event["skill"]))]
        for event in events
        if str(event.get("skill")) in skill_tokens
        and (str(event["anon_author"]), str(event["skill"])) in rank_lookup
    ]
    if not observed:
        return 0.75
    return _clamp(_median(observed), 0.60, 0.85)


def _weights_from_counts(counts: Mapping[str, int | float], skill_tokens: Sequence[str]) -> List[float]:
    values = [float(counts.get(skill, 0.0)) for skill in skill_tokens]
    total = sum(values)
    if total <= 0:
        return [1.0 / len(skill_tokens) for _ in skill_tokens]
    return [value / total for value in values]


def _deadline_mode(units: Sequence[Mapping[str, Any]]) -> str:
    modes = [str(unit.get("deadline_mode")) for unit in units if unit.get("tightness") is not None]
    if not modes:
        return "corpus_fallback"
    unique = set(modes)
    if unique == {"milestone"}:
        return "milestone"
    if unique == {"release"}:
        return "release"
    return "mixed"


def _level_summary(levels: Mapping[str, Mapping[str, int]]) -> Dict[str, Any]:
    return {
        contributor: dict(sorted(skills.items()))
        for contributor, skills in sorted(levels.items())
    }


def _failed_measurement(repo: SourceRepo, window: ObservationWindow, reason: str) -> Dict[str, Any]:
    return {
        "valid": False,
        "failure_reason": reason,
        "source": asdict(repo),
        "window": asdict(window),
    }


def _cache_timestamp(path: Path) -> str:
    if not path.exists():
        return ""
    data = json.loads(path.read_text(encoding="utf-8"))
    return str(data.get("collection_timestamp") or data.get("fetched_at") or "")


def _normalize_vector(values: Sequence[float]) -> List[float]:
    total = math.sqrt(sum(float(value) ** 2 for value in values))
    if total <= 0:
        return [0.0 for _ in values]
    return [float(value) / total for value in values]


def _mean(values: Sequence[float | int]) -> float:
    return float(mean(values)) if values else 0.0


def _median(values: Sequence[float | int]) -> float:
    return float(median(values)) if values else 0.0


def _sd(values: Sequence[float | int]) -> float:
    return float(pstdev(values)) if len(values) > 1 else 0.0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))
