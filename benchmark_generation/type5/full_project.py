"""Utilities for the Type 5 complete-project empirical extension.

The fixed-window Type 5 corpus is intentionally left untouched. This module
contains only opt-in helpers for the additional full-project extension.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from benchmark_generation.core.generator import derive_seed
from benchmark_generation.core.metadata import summarize_numeric
from benchmark_generation.type5.empirical import finalize_source_profile, percentile
from benchmark_generation.type5.sources import ObservationWindow, SourceRepo, parse_github_time


FULL_PROJECT_WINDOW_NAME = "FULL"
FULL_PROJECT_OBSERVATION_MODE = "full_project"
FULL_PROJECT_WINDOW_TYPE = "complete_eligible_project_history"
FULL_PROJECT_SOURCE_ORIGIN = "GitHub_empirical"
FINAL_EXTENSION_START_ID = 92
FINAL_EXTENSION_COUNT = 20
PILOT_EXTENSION_COUNT = 5
DAYS_PER_MONTH = 30.4375

LIFETIME_BANDS: Tuple[Tuple[str, int, int], ...] = (
    ("90_180", 90, 180),
    ("181_270", 181, 270),
    ("271_365", 271, 365),
    ("366_456", 366, 456),
    ("457_548", 457, 548),
)

SOURCE_SELECTION_METRICS: Tuple[str, ...] = (
    "project_history_days",
    "human_contributor_count",
    "merged_pr_count",
    "skill_count",
    "skills_per_contributor_mean",
    "source_skill_skew",
    "source_alignment",
    "roles_per_work_unit_mean",
    "duration_median",
    "source_supply_concentration",
    "source_demand_concentration",
    "contention",
)


def canonical_repo(owner: str, repo: str) -> str:
    """Return a stable GitHub repository identity in owner/repo form."""

    owner_clean = str(owner).strip().strip("/")
    repo_clean = str(repo).strip().strip("/")
    if repo_clean.endswith(".git"):
        repo_clean = repo_clean[:-4]
    return f"{owner_clean.lower()}/{repo_clean.lower()}"


def canonical_repo_from_mapping(row: Mapping[str, Any]) -> str:
    owner = row.get("owner") or row.get("source_owner") or row.get("repository_owner") or ""
    repo = row.get("repo") or row.get("source_repo") or row.get("repository_name") or ""
    return canonical_repo(str(owner), str(repo))


def safe_repo_token(value: str) -> str:
    """Normalize an owner or repository name for traceable filenames."""

    token = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value).strip())
    token = token.strip(".-")
    return token or "repo"


def full_project_filename(family_instance_id: int, owner: str, repo: str) -> str:
    return (
        f"type5_{int(family_instance_id):03d}__GHFULL__"
        f"{safe_repo_token(owner)}__{safe_repo_token(repo)}.txt"
    )


def pilot_filename(pilot_index: int, owner: str, repo: str) -> str:
    return (
        f"type5_full_project_pilot_{int(pilot_index):03d}__GHFULL__"
        f"{safe_repo_token(owner)}__{safe_repo_token(repo)}.txt"
    )


def project_history_days(start: str, end: str) -> float:
    start_dt = parse_github_time(start)
    end_dt = parse_github_time(end)
    return (end_dt - start_dt).total_seconds() / 86400.0


def project_history_months(days: float) -> float:
    return float(days) / DAYS_PER_MONTH


def lifetime_band(days: float) -> str:
    for name, low, high in LIFETIME_BANDS:
        if low <= days <= high:
            return name
    return "outside_90_548"


def full_project_window(start: str, end: str) -> ObservationWindow:
    return ObservationWindow(FULL_PROJECT_WINDOW_NAME, start, end)


def source_repo_from_mapping(row: Mapping[str, Any], *, kind: str = "full_project_candidate", index: int = 0) -> SourceRepo:
    owner = str(row.get("owner") or row.get("source_owner") or "")
    repo = str(row.get("repo") or row.get("source_repo") or "")
    url = str(row.get("url") or row.get("source_url") or f"https://github.com/{owner}/{repo}")
    domain = str(row.get("domain") or row.get("source_domain") or "uncategorized")
    return SourceRepo(owner=owner, repo=repo, url=url, domain=domain, kind=kind, index=index)


def full_project_candidate_batch_from_source(source: Mapping[str, Any]) -> str:
    """Infer deterministic full-project candidate batch for legacy summaries."""

    if source.get("candidate_batch"):
        return str(source["candidate_batch"])
    try:
        index = int(source.get("index") or source.get("source_index") or 0)
    except (TypeError, ValueError):
        index = 0
    return "initial_001_060" if index <= 60 else "expansion_061_150"


def existing_type5_repository_exclusion_records(
    metadata_dir: str | Path = "benchmark_metadata",
    sources_path: str | Path = "benchmark_metadata/type5_sources.csv",
    metadata_csv_path: str | Path = "benchmark_metadata/type5_metadata.csv",
    corpus_path: str | Path = "benchmark_metadata/type5_empirical_corpus.json",
) -> List[Dict[str, str]]:
    """Build the frozen Type 5 repository exclusion list.

    The list contains every primary source, every successful reserve source,
    and every source that contributed to the accepted 91 fixed-window files.
    Unused reserve repositories are not excluded unless they appear in the
    accepted metadata or replacement records.
    """

    records: Dict[str, Dict[str, str]] = {}

    def add(owner: str, repo: str, reason: str) -> None:
        if not owner or not repo:
            return
        repository = canonical_repo(owner, repo)
        existing = records.setdefault(
            repository,
            {"repository": repository, "owner": owner, "repo": repo, "reasons": ""},
        )
        reasons = {item for item in existing["reasons"].split(";") if item}
        reasons.add(reason)
        existing["reasons"] = ";".join(sorted(reasons))

    sources_file = Path(sources_path)
    if sources_file.exists():
        for row in _read_csv(sources_file):
            if str(row.get("kind", "")).lower() == "primary":
                add(str(row.get("owner", "")), str(row.get("repo", "")), "existing_type5_primary_source")

    metadata_file = Path(metadata_csv_path)
    if metadata_file.exists():
        for row in _read_csv(metadata_file):
            add(
                str(row.get("source_owner", "")),
                str(row.get("source_repo", "")),
                "contributed_existing_type5_instance",
            )

    corpus_file = Path(corpus_path)
    if corpus_file.exists():
        corpus = json.loads(corpus_file.read_text(encoding="utf-8"))
        for record in corpus.get("records", []):
            add(
                str(record.get("source_owner", "")),
                str(record.get("source_repo", "")),
                "contributed_existing_type5_empirical_record",
            )
        for replacement in corpus.get("replacements", []):
            replacement_source = str(replacement.get("replacement_source", ""))
            if "/" in replacement_source:
                owner, repo = replacement_source.split("/", 1)
                add(owner, repo, "successful_existing_type5_reserve_source")

    return sorted(records.values(), key=lambda item: item["repository"])


def write_exclusion_list(
    records: Sequence[Mapping[str, str]],
    metadata_dir: str | Path = "benchmark_metadata",
) -> None:
    metadata_dir = Path(metadata_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    csv_path = metadata_dir / "existing_type5_repository_exclusion_list.csv"
    json_path = metadata_dir / "existing_type5_repository_exclusion_list.json"
    fields = ["repository", "owner", "repo", "reasons"]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fields})
    payload = {
        "count": len(records),
        "repositories": list(records),
        "note": (
            "Exclusion list for the Type 5 full-project extension. It is "
            "derived from frozen fixed-window Type 5 source metadata and does "
            "not alter the accepted 91 instances."
        ),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def repository_overlap(repositories: Iterable[str], exclusion_records: Sequence[Mapping[str, str]]) -> List[str]:
    excluded = {str(record["repository"]).lower() for record in exclusion_records}
    return sorted({repo.lower() for repo in repositories if repo.lower() in excluded})


def augment_full_project_measurement(
    measurement: Mapping[str, Any],
    collection_summary: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Attach complete-history metadata and source-quality checks."""

    result = dict(measurement)
    source = result.get("source", {})
    if not isinstance(source, Mapping):
        source = {}
    start = str(collection_summary.get("project_history_start") or "")
    end = str(collection_summary.get("project_history_end") or "")
    days = float(collection_summary.get("project_history_days") or 0.0)
    if not days and start and end:
        days = project_history_days(start, end)
    result.update(
        {
            "candidate_batch": collection_summary.get("candidate_batch") or full_project_candidate_batch_from_source(source),
            "cheap_prescreen_pass": collection_summary.get("cheap_prescreen_pass", True),
            "detailed_collection_attempted": collection_summary.get("detailed_collection_attempted", True),
            "project_history_start": start,
            "project_history_end": end,
            "project_history_days": days,
            "project_history_months": project_history_months(days),
            "project_lifetime_band": lifetime_band(days),
            "observation_mode": FULL_PROJECT_OBSERVATION_MODE,
            "source_window_type": FULL_PROJECT_WINDOW_TYPE,
            "source_origin": FULL_PROJECT_SOURCE_ORIGIN,
            "technical_detail_mode": collection_summary.get("technical_detail_mode", ""),
            "full_history_merged_pr_count": int(collection_summary.get("full_history_merged_pr_count", 0) or 0),
            "full_history_human_pr_count": int(collection_summary.get("full_history_human_pr_count", 0) or 0),
        }
    )
    issues: List[str] = []
    min_days, max_days = config.get("project_history_days_range", [90, 548])
    if days < float(min_days):
        issues.append("project_history_below_90_days")
    if days > float(max_days):
        issues.append("project_history_above_548_days")
    if str(collection_summary.get("repository_overlap_with_existing_type5", "0")) not in {"0", "False", "false"}:
        issues.append("repository_overlaps_existing_type5")
    if not result.get("valid"):
        reason = str(result.get("failure_reason") or "empirical_profile_invalid")
        if reason:
            issues.extend(reason.split(";"))
    result["full_project_quality"] = {"passed": not issues, "issues": sorted(set(issues))}
    result["valid"] = bool(result.get("valid")) and not issues
    result["failure_reason"] = "" if result["valid"] else ";".join(result["full_project_quality"]["issues"])
    result["profile_quality_pass"] = bool(result["valid"])
    result["rejection_reason"] = result["failure_reason"]
    return result


def finalize_full_project_source_profile(
    measurement: Mapping[str, Any],
    config: Mapping[str, Any],
    seed: int,
    family_instance_id: int,
    selection_rank: int,
    corpus_deadline_bounds: Tuple[float, float],
    global_score_k: float,
) -> Dict[str, Any]:
    record = finalize_source_profile(
        measurement,
        config,
        seed=seed,
        family_instance_id=family_instance_id,
        source_slot_id=100_000 + int(selection_rank),
        source_replacement=False,
        original_source=None,
        replacement_reason="",
        corpus_deadline_bounds=corpus_deadline_bounds,
        global_score_k=global_score_k,
    )
    days = float(measurement["project_history_days"])
    record.update(
        {
            "source_repository": f"{record['source_owner']}/{record['source_repo']}",
            "source_origin": FULL_PROJECT_SOURCE_ORIGIN,
            "observation_mode": FULL_PROJECT_OBSERVATION_MODE,
            "source_window_type": FULL_PROJECT_WINDOW_TYPE,
            "source_window_name": FULL_PROJECT_WINDOW_NAME,
            "source_window_start": measurement["project_history_start"],
            "source_window_end": measurement["project_history_end"],
            "project_history_start": measurement["project_history_start"],
            "project_history_end": measurement["project_history_end"],
            "project_history_days": days,
            "project_history_months": project_history_months(days),
            "project_lifetime_band": lifetime_band(days),
            "selection_rank": int(selection_rank),
            "technical_detail_mode": measurement.get("technical_detail_mode", ""),
            "full_history_merged_pr_count": int(measurement.get("full_history_merged_pr_count", 0) or 0),
            "full_history_human_pr_count": int(measurement.get("full_history_human_pr_count", 0) or 0),
        }
    )
    record["source_metrics"].update(
        {
            "project_history_days": days,
            "project_history_months": project_history_months(days),
            "full_history_merged_pr_count": int(measurement.get("full_history_merged_pr_count", 0) or 0),
            "full_history_human_pr_count": int(measurement.get("full_history_human_pr_count", 0) or 0),
            "technical_work_unit_count": int(measurement.get("technical_work_unit_count", 0) or 0),
            "source_observation_mode": FULL_PROJECT_OBSERVATION_MODE,
            "source_window_type": FULL_PROJECT_WINDOW_TYPE,
            "project_lifetime_band": lifetime_band(days),
        }
    )
    level_values = [
        int(level)
        for contributor_levels in measurement.get("levels", {}).values()
        for level in contributor_levels.values()
    ]
    record["source_profile"].update(
        {
            "skills_per_contributor_values": [
                int(value) for value in measurement.get("skills_per_contributor_values", [])
            ],
            "level_values": level_values,
            "role_count_values": [
                int(value) for value in measurement.get("roles_per_work_unit_values", [])
            ],
        }
    )
    record["proxy_flags"].update(
        {
            "source_origin": FULL_PROJECT_SOURCE_ORIGIN,
            "observation_mode": FULL_PROJECT_OBSERVATION_MODE,
            "source_window_type": FULL_PROJECT_WINDOW_TYPE,
        }
    )
    return record


def select_diverse_full_project_sources(
    candidates: Sequence[Mapping[str, Any]],
    count: int = FINAL_EXTENSION_COUNT,
    seed: int = 2026,
) -> List[Mapping[str, Any]]:
    """Select sources using deterministic source-native diversity only."""

    valid = [
        candidate
        for candidate in candidates
        if _candidate_valid(candidate) and _candidate_repo(candidate) != "/"
    ]
    unique: Dict[str, Mapping[str, Any]] = {}
    for candidate in valid:
        repo = _candidate_repo(candidate)
        current = unique.get(repo)
        if current is None or _source_quality_score(candidate) > _source_quality_score(current):
            unique[repo] = candidate
    pool = list(unique.values())
    if len(pool) < count:
        raise RuntimeError(f"Only {len(pool)} valid unique full-project sources are available; {count} required.")

    vectors = _normalized_source_vectors(pool)
    selected: List[Mapping[str, Any]] = []
    selected_indices: List[int] = []
    selected_repos: set[str] = set()
    seen_domains: set[str] = set()
    seen_bands: set[str] = set()

    while len(selected) < count:
        scored = []
        for idx, candidate in enumerate(pool):
            repo = _candidate_repo(candidate)
            if repo in selected_repos:
                continue
            domain = _candidate_field(candidate, "domain", "source_domain")
            band = lifetime_band(_candidate_float(candidate, "project_history_days"))
            if not selected:
                diversity = _source_quality_score(candidate)
            else:
                diversity = min(_euclidean(vectors[idx], vectors[other_idx]) for other_idx in selected_indices)
            score = diversity
            if domain not in seen_domains:
                score += 0.10
            if band not in seen_bands:
                score += 0.10
            score += _stable_fraction(f"{seed}:{repo}") * 1e-9
            scored.append((score, repo, candidate))
        if not scored:
            break
        scored.sort(key=lambda item: (-item[0], item[1]))
        chosen = scored[0][2]
        selected.append(chosen)
        chosen_idx = pool.index(chosen)
        selected_indices.append(chosen_idx)
        chosen_repo = _candidate_repo(chosen)
        selected_repos.add(chosen_repo)
        seen_domains.add(_candidate_field(chosen, "domain", "source_domain"))
        seen_bands.add(lifetime_band(_candidate_float(chosen, "project_history_days")))

    return selected


def select_pilot_full_project_records(
    records: Sequence[Mapping[str, Any]],
    count: int = PILOT_EXTENSION_COUNT,
    seed: int = 2026,
) -> List[Mapping[str, Any]]:
    if len(records) < count:
        raise RuntimeError(f"Pilot requested {count} records, but only {len(records)} full-project records exist")
    return list(select_diverse_full_project_sources(records, count=count, seed=derive_seed(seed, 503, salt=1_901_771)))


def assignment_family_ids(records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    assigned = []
    for offset, record in enumerate(records):
        item = dict(record)
        item["family_instance_id"] = FINAL_EXTENSION_START_ID + offset
        item["selection_rank"] = offset + 1
        assigned.append(item)
    return assigned


def source_candidate_row(measurement: Mapping[str, Any]) -> Dict[str, Any]:
    source = measurement.get("source", {})
    failure_reason = measurement.get("rejection_reason") or measurement.get("failure_reason", "")
    return {
        "repository": canonical_repo_from_mapping(source),
        "owner": source.get("owner", ""),
        "repo": source.get("repo", ""),
        "url": source.get("url", ""),
        "domain": source.get("domain", ""),
        "candidate_batch": measurement.get("candidate_batch") or full_project_candidate_batch_from_source(source),
        "cheap_prescreen_pass": measurement.get("cheap_prescreen_pass", ""),
        "detailed_collection_attempted": measurement.get("detailed_collection_attempted", ""),
        "profile_quality_pass": measurement.get("profile_quality_pass", bool(measurement.get("valid"))),
        "repository_id": measurement.get("repository_id", ""),
        "default_branch": measurement.get("default_branch", ""),
        "snapshot_sha": measurement.get("snapshot_sha", ""),
        "snapshot_timestamp": measurement.get("collection_timestamp", ""),
        "project_history_start": measurement.get("project_history_start", ""),
        "project_history_end": measurement.get("project_history_end", ""),
        "project_history_days": measurement.get("project_history_days", ""),
        "project_history_months": measurement.get("project_history_months", ""),
        "project_lifetime_band": measurement.get("project_lifetime_band", ""),
        "human_contributor_count": measurement.get("human_contributor_count", ""),
        "merged_technical_pr_count": measurement.get("merged_pr_count", ""),
        "sampled_pr_count": measurement.get("sampled_pr_count", ""),
        "retained_skill_count": measurement.get("skill_count", ""),
        "profile_count": measurement.get("profile_count", ""),
        "source_skill_skew": measurement.get("source_skill_skew", ""),
        "source_alignment": measurement.get("source_alignment", ""),
        "activity_intensity": _safe_div(
            float(measurement.get("merged_pr_count", 0) or 0),
            max(1.0, float(measurement.get("project_history_days", 0) or 0)),
        ),
        "technical_detail_mode": measurement.get("technical_detail_mode", ""),
        "quality_status": "accepted_candidate" if measurement.get("valid") else "rejected",
        "rejection_reason": failure_reason,
        "rejection_endpoint": measurement.get("rejection_endpoint", ""),
        "rejection_http_status": measurement.get("rejection_http_status", ""),
        "rejection_timestamp": measurement.get("rejection_timestamp", ""),
        "rejection_stage": measurement.get("rejection_stage", ""),
    }


def write_candidates_csv(measurements: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    rows = [source_candidate_row(item) for item in measurements]
    _write_dict_rows(rows, path)


def write_selection_files(
    records: Sequence[Mapping[str, Any]],
    exclusion_records: Sequence[Mapping[str, str]],
    metadata_dir: str | Path,
) -> None:
    metadata_dir = Path(metadata_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    selection_rows = [selection_row(record) for record in records]
    _write_dict_rows(selection_rows, metadata_dir / "type5_full_project_selection.csv")
    payload = {
        "extension": "type5_full_project",
        "selected_count": len(records),
        "family_instance_ids": [record["family_instance_id"] for record in records],
        "repository_overlap_with_existing_type5": repository_overlap(
            [canonical_repo_from_mapping(record) for record in records],
            exclusion_records,
        ),
        "records": list(records),
    }
    (metadata_dir / "type5_full_project_selection.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    write_full_project_sources_markdown(records, metadata_dir / "TYPE5_FULL_PROJECT_SOURCES.md")


def selection_row(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "family_instance_id": record.get("family_instance_id", ""),
        "selection_rank": record.get("selection_rank", ""),
        "source_repository": record.get("source_repository", f"{record.get('source_owner', '')}/{record.get('source_repo', '')}"),
        "source_repository_id": record.get("source_repository_id", ""),
        "source_url": record.get("source_url", ""),
        "source_domain": record.get("source_domain", ""),
        "source_snapshot_sha": record.get("source_snapshot_sha", ""),
        "source_snapshot_timestamp": record.get("source_collection_timestamp", ""),
        "project_history_start": record.get("project_history_start", ""),
        "project_history_end": record.get("project_history_end", ""),
        "project_history_days": record.get("project_history_days", ""),
        "project_history_months": record.get("project_history_months", ""),
        "project_lifetime_band": record.get("project_lifetime_band", ""),
        "human_contributor_count": record.get("source_metrics", {}).get("source_human_contributor_count", ""),
        "merged_work_unit_count": record.get("source_metrics", {}).get("source_merged_pr_count", ""),
        "sampled_pr_count": record.get("source_metrics", {}).get("source_sampled_pr_count", ""),
        "retained_skill_count": record.get("source_metrics", {}).get("source_skill_count", ""),
        "source_skill_skew": record.get("source_metrics", {}).get("source_skill_skew", ""),
        "source_alignment": record.get("source_metrics", {}).get("source_alignment", ""),
        "observation_mode": record.get("observation_mode", FULL_PROJECT_OBSERVATION_MODE),
        "source_window_type": record.get("source_window_type", FULL_PROJECT_WINDOW_TYPE),
    }


def write_full_project_sources_markdown(records: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    lines = [
        "# Type 5 Full-Project Extension Sources",
        "",
        "These sources are complete eligible GitHub project histories for the Type 5 extension.",
        "They do not replace or modify the accepted fixed-window Type 5 family.",
        "",
    ]
    for record in records:
        lines.extend(
            [
                f"## type5_{int(record['family_instance_id']):03d}: {record['source_owner']}/{record['source_repo']}",
                "",
                f"- URL: {record['source_url']}",
                f"- Domain: {record['source_domain']}",
                f"- Observation mode: {record.get('observation_mode', FULL_PROJECT_OBSERVATION_MODE)}",
                f"- Complete eligible history: {record['project_history_start']} to {record['project_history_end']}",
                f"- Project history days: {float(record['project_history_days']):.3f}",
                f"- Project history months: {float(record['project_history_months']):.3f}",
                f"- Snapshot SHA: {record.get('source_snapshot_sha', '')}",
                f"- Snapshot timestamp: {record.get('source_collection_timestamp', '')}",
                f"- Repository ID: {record.get('source_repository_id', '')}",
                "",
            ]
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_full_project_methodology_draft(path: str | Path, selected_count: int = 0) -> None:
    output_end = FINAL_EXTENSION_START_ID + int(selected_count or FINAL_EXTENSION_COUNT) - 1
    lines = [
        "# Type 5 Full-Project Extension Methodology Draft",
        "",
        "This document describes the opt-in extension of Type 5 with complete eligible GitHub project histories.",
        "The accepted 91 fixed six-month Type 5 instances remain immutable and are not regenerated.",
        "",
        "## Design",
        "",
        "Each new source repository contributes at most one empirical profile. The collector observes the repository's complete eligible public technical work history, from the first retained non-bot work unit through the latest retained work unit at the acquisition snapshot. No fixed six-month windows are used in this extension.",
        "",
        "Repositories are eligible only when the measured project history is between 90 and 548 days, inclusive. Mature repositories are rejected rather than truncated to fit this interval.",
        "",
        "## Source Selection",
        "",
        "Selection is source-driven and occurs before MTP generation. Candidate repositories are filtered by repository status, overlap with the frozen Type 5 source list, lifetime, non-bot activity, retained technical skills, and contributor-profile quality. When more valid sources are available than the configured extension count, deterministic farthest-first selection is applied over normalized source-native metrics such as lifetime, contributor count, work-unit count, skill count, skill skew, demand/supply alignment, contention, and duration. Solver scores, ISA positions, and generated MTP performance are not used.",
        "",
        "## Empirical Interpretation",
        "",
        "Human GitHub contributors are treated as anonymized contributor proxies. Merged pull requests are treated as work-unit proxies. Technical language/subsystem tokens derived from changed file paths are treated as skill proxies. Activity-percentile levels are relative empirical proxies, not literal human competence measurements. Scores and best-before values remain synthetic or derived proxies, using the existing Type 5 methodology.",
        "",
        "## Generation",
        "",
        "For each selected source, the empirical profile parameterizes the shared forward MTP generator already used by Type 5. The generated text file contains only anonymous cXXXXX contributors, sXXXX skills, and pXXXXX projects. Repository identifiers, usernames, PR numbers, source paths, and semantic skill tokens are excluded from the MTP text and stored only in metadata.",
        "",
        "## Quality Gates",
        "",
        f"The extension requires structural MTP validity, exact mentorship semantics, no structural duplicates against existing benchmark families, source/generation correspondence, complete source provenance, repository uniqueness, zero overlap with existing Type 5 repositories, contiguous output IDs type5_{FINAL_EXTENSION_START_ID:03d} through type5_{output_end:03d}, and byte-for-byte preservation of type5_001 through type5_091.",
        "",
        f"Selected full-project sources in the current selection file: {selected_count}",
        "",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def full_project_source_statistics(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    def values(key: str) -> List[float]:
        out = []
        for record in records:
            if key in record:
                out.append(float(record.get(key, 0) or 0))
            else:
                metrics = record.get("source_metrics", {})
                out.append(float(metrics.get(key, 0) or 0))
        return out

    return {
        "count": len(records),
        "project_history_days": summarize_numeric(values("project_history_days")),
        "project_history_months": summarize_numeric(values("project_history_months")),
        "contributors": summarize_numeric(values("source_human_contributor_count")),
        "projects_work_units": summarize_numeric(values("source_merged_pr_count")),
        "skills": summarize_numeric(values("source_skill_count")),
        "skill_skew": summarize_numeric(values("source_skill_skew")),
        "alignment": summarize_numeric(values("source_alignment")),
        "lifetime_band_counts": _counts(
            [str(record.get("project_lifetime_band") or lifetime_band(float(record.get("project_history_days", 0) or 0))) for record in records]
        ),
        "domain_counts": _counts([str(record.get("source_domain", "")) for record in records]),
    }


def _normalized_source_vectors(candidates: Sequence[Mapping[str, Any]]) -> List[List[float]]:
    raw_rows = [_source_vector(candidate) for candidate in candidates]
    cols = list(zip(*raw_rows)) if raw_rows else []
    mins = [min(col) for col in cols]
    maxes = [max(col) for col in cols]
    normalized = []
    for row in raw_rows:
        normalized.append(
            [
                0.0 if maxes[idx] - mins[idx] <= 1e-12 else (value - mins[idx]) / (maxes[idx] - mins[idx])
                for idx, value in enumerate(row)
            ]
        )
    return normalized


def _source_vector(candidate: Mapping[str, Any]) -> List[float]:
    metric_lookup = {
        "project_history_days": _candidate_float(candidate, "project_history_days"),
        "human_contributor_count": math.log1p(_candidate_float(candidate, "human_contributor_count", "source_human_contributor_count")),
        "merged_pr_count": math.log1p(_candidate_float(candidate, "merged_pr_count", "source_merged_pr_count")),
        "skill_count": math.log1p(_candidate_float(candidate, "skill_count", "source_skill_count")),
        "skills_per_contributor_mean": _candidate_float(candidate, "skills_per_contributor_mean", "source_skills_per_contributor_mean"),
        "source_skill_skew": _candidate_float(candidate, "source_skill_skew"),
        "source_alignment": _candidate_float(candidate, "source_alignment"),
        "roles_per_work_unit_mean": _candidate_float(candidate, "roles_per_work_unit_mean", "source_roles_per_work_unit_mean"),
        "duration_median": math.log1p(_candidate_float(candidate, "duration_median", "source_duration_median")),
        "source_supply_concentration": _candidate_float(candidate, "source_supply_concentration"),
        "source_demand_concentration": _candidate_float(candidate, "source_demand_concentration"),
        "contention": _safe_div(
            _candidate_float(candidate, "merged_pr_count", "source_merged_pr_count"),
            _candidate_float(candidate, "human_contributor_count", "source_human_contributor_count"),
        ),
    }
    return [metric_lookup[name] for name in SOURCE_SELECTION_METRICS]


def _source_quality_score(candidate: Mapping[str, Any]) -> float:
    return (
        math.log1p(_candidate_float(candidate, "human_contributor_count", "source_human_contributor_count"))
        + math.log1p(_candidate_float(candidate, "merged_pr_count", "source_merged_pr_count"))
        + math.log1p(_candidate_float(candidate, "skill_count", "source_skill_count"))
        + _candidate_float(candidate, "source_alignment")
    )


def _candidate_valid(candidate: Mapping[str, Any]) -> bool:
    if "valid" in candidate:
        return bool(candidate.get("valid"))
    if "overall_type5_quality_pass" in candidate:
        return bool(candidate.get("overall_type5_quality_pass"))
    return True


def _candidate_repo(candidate: Mapping[str, Any]) -> str:
    if isinstance(candidate.get("source"), Mapping):
        return canonical_repo_from_mapping(candidate["source"])
    return canonical_repo_from_mapping(candidate)


def _candidate_field(candidate: Mapping[str, Any], raw_key: str, record_key: str) -> str:
    if raw_key in candidate:
        return str(candidate.get(raw_key, ""))
    if record_key in candidate:
        return str(candidate.get(record_key, ""))
    if isinstance(candidate.get("source"), Mapping):
        return str(candidate["source"].get(raw_key, ""))
    return ""


def _candidate_float(candidate: Mapping[str, Any], raw_key: str, record_key: str | None = None) -> float:
    keys = [raw_key]
    if record_key:
        keys.append(record_key)
    for key in keys:
        if key in candidate and candidate.get(key) not in ("", None):
            try:
                return float(candidate.get(key, 0) or 0)
            except (TypeError, ValueError):
                return 0.0
    metrics = candidate.get("source_metrics")
    if isinstance(metrics, Mapping):
        for key in keys:
            if key in metrics and metrics.get(key) not in ("", None):
                try:
                    return float(metrics.get(key, 0) or 0)
                except (TypeError, ValueError):
                    return 0.0
    return 0.0


def _euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def _stable_fraction(text: str) -> float:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return int(digest, 16) / float(16**16 - 1)


def _write_dict_rows(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _counts(values: Sequence[str]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return dict(sorted(result.items()))


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
