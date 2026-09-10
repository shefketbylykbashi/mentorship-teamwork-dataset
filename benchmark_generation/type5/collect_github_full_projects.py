"""Collect complete eligible GitHub histories for the Type 5 extension."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import urllib.parse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from benchmark_generation.core.config import load_config
from benchmark_generation.core.generator import derive_seed
from benchmark_generation.type5.collect_github_corpus import (
    DEFAULT_COLLECT_CONFIG,
    GitHubClient,
    GitHubNotFoundError,
    cached_body,
    get_github_token,
    utc_now,
    write_cache,
)
from benchmark_generation.type5.full_project import (
    FULL_PROJECT_WINDOW_NAME,
    canonical_repo,
    existing_type5_repository_exclusion_records,
    project_history_days,
    project_history_months,
    repository_overlap,
    source_repo_from_mapping,
    write_exclusion_list,
)
from benchmark_generation.type5.sampling import deterministic_pr_sample
from benchmark_generation.type5.sources import SourceRepo, repo_cache_dir
from benchmark_generation.type5.taxonomy import compile_bot_patterns, is_bot_identity


DEFAULT_CONFIG: Dict[str, Any] = {
    **DEFAULT_COLLECT_CONFIG,
    "root_seed": 2026,
    "project_history_days_range": [90, 548],
    "candidate_target": 150,
    "candidate_minimum_before_selection": 120,
    "candidate_existing_batch": "initial_001_060",
    "candidate_expansion_batch": "expansion_061_150",
    "max_search_pages_per_query": 2,
    "full_detail_pr_limit": 200,
    "detailed_pr_sample_size": 60,
    "candidate_queries": [],
    "candidate_repositories": [],
    "cheap_prescreen": {
        "max_bot_pr_share": 0.80,
        "min_technical_language_bytes": 1,
        "nontechnical_languages": ["Markdown", "MDX", "Text", "TeX"],
        "personal_only_keywords": [
            "personal portfolio",
            "portfolio",
            "resume",
            "curriculum vitae",
        ],
        "low_activity_keywords": [
            "documentation",
            "docs",
            "tutorial",
            "course",
            "workshop",
            "example",
            "sample",
            "demo",
            "starter",
            "template",
            "boilerplate",
        ],
    },
}


@dataclass(frozen=True)
class CandidateSource:
    """A Type 5 full-project repository candidate plus deterministic batch."""

    repo: SourceRepo
    candidate_batch: str


def discover_candidate_repositories(
    client: GitHubClient,
    config: Mapping[str, Any],
    cache_dir: str | Path,
    exclusion_records: Sequence[Mapping[str, str]],
    metadata_dir: str | Path = "benchmark_metadata",
) -> List[CandidateSource]:
    """Return a deterministic, de-duplicated pool of candidate repositories.

    Previously inspected candidates are loaded first so a resumed run preserves
    their cache and rejection metadata before deterministic expansion begins.
    """

    target = int(config.get("candidate_target", 150))
    cache_dir = Path(cache_dir)
    metadata_dir = Path(metadata_dir)
    candidates: Dict[str, CandidateSource] = {}
    existing_batch = str(config.get("candidate_existing_batch", "initial_001_060"))
    expansion_batch = str(config.get("candidate_expansion_batch", "expansion_061_150"))

    for idx, summary in enumerate(_load_resume_candidate_summaries(cache_dir, metadata_dir), start=1):
        source = summary.get("source", {})
        if not isinstance(source, Mapping):
            continue
        repo = source_repo_from_mapping(source, kind=str(source.get("kind") or "full_project_candidate"), index=idx)
        key = canonical_repo(repo.owner, repo.repo)
        if key == "/" or key in candidates:
            continue
        batch = str(summary.get("candidate_batch") or source.get("candidate_batch") or existing_batch)
        candidates[key] = CandidateSource(repo, batch)
        if len(candidates) >= target:
            return _renumber_candidates(candidates.values())

    for idx, item in enumerate(config.get("candidate_repositories", []), start=1):
        repo = source_repo_from_mapping(item, kind="full_project_seed", index=idx)
        key = canonical_repo(repo.owner, repo.repo)
        if key != "/" and key not in candidates:
            batch = str(item.get("candidate_batch") or item.get("batch") or expansion_batch)
            candidates[key] = CandidateSource(repo, batch)
        if len(candidates) >= target:
            return _renumber_candidates(candidates.values())

    for query_idx, query in enumerate(config.get("candidate_queries", []), start=1):
        if len(candidates) >= target:
            break
        query_text = str(query.get("q") or query.get("query") or "")
        domain = str(query.get("domain") or "uncategorized")
        batch = str(query.get("candidate_batch") or query.get("batch") or expansion_batch)
        if not query_text:
            continue
        per_page = int(query.get("per_page", config.get("per_page", 100)))
        max_pages = int(query.get("pages", config.get("max_search_pages_per_query", 2)))
        for page in range(1, max_pages + 1):
            body = client.get_json(
                "/search/repositories",
                {
                    "q": query_text,
                    "sort": str(query.get("sort", "updated")),
                    "order": str(query.get("order", "desc")),
                    "per_page": per_page,
                    "page": page,
                },
                cache_dir / "search" / f"query_{query_idx:03d}_page_{page:03d}.json",
                prefer_cache=True,
            )
            for item in body.get("items", []) if isinstance(body, dict) else []:
                owner = ((item.get("owner") or {}).get("login")) if isinstance(item.get("owner"), dict) else ""
                repo_name = item.get("name") or ""
                key = canonical_repo(str(owner), str(repo_name))
                if not owner or not repo_name or key in candidates:
                    continue
                candidates[key] = CandidateSource(
                    SourceRepo(
                        owner=str(owner),
                        repo=str(repo_name),
                        url=str(item.get("html_url") or f"https://github.com/{owner}/{repo_name}"),
                        domain=domain,
                        kind="full_project_search",
                        index=len(candidates) + 1,
                    ),
                    batch,
                )
                if len(candidates) >= target:
                    break
            if len(candidates) >= target:
                break
    return _renumber_candidates(candidates.values())


def collect_full_project_repository(
    client: GitHubClient,
    repo: SourceRepo,
    cache_dir: str | Path,
    config: Mapping[str, Any],
    bot_config: Mapping[str, Any],
    seed: int,
    exclusion_records: Sequence[Mapping[str, str]],
    candidate_batch: str | None = None,
) -> Dict[str, Any]:
    """Collect one complete-history candidate into the Type 5 cache layout."""

    repo_dir = repo_cache_dir(cache_dir, repo)
    summary_path = repo_dir / "full_project_collection_summary.json"
    existing = _safe_read_json(summary_path)
    if is_complete_full_project_summary(existing, repo):
        logging.info("Type5 full-project %s already collected; skipping", repo.full_name)
        return existing

    overlap = repository_overlap([canonical_repo(repo.owner, repo.repo)], exclusion_records)
    if overlap:
        summary = _collection_failure(
            repo,
            "repository_overlaps_existing_type5",
            candidate_batch=candidate_batch,
            stage="existing_type5_overlap",
            cheap_prescreen_pass=False,
            detailed_collection_attempted=False,
        )
        summary["repository_overlap_with_existing_type5"] = len(overlap)
        write_cache(summary_path, summary)
        return summary

    stage = "repository_metadata"
    detailed_collection_attempted = False
    try:
        repository = client.get_json(
            f"/repos/{repo.owner}/{repo.repo}",
            {},
            repo_dir / "repository.json",
            prefer_cache=True,
        )
        if repository.get("private") or repository.get("archived") or repository.get("disabled") or repository.get("fork"):
            summary = _collection_failure(
                repo,
                "repository_private_archived_disabled_or_fork",
                candidate_batch=candidate_batch,
                stage=stage,
                cheap_prescreen_pass=False,
                detailed_collection_attempted=False,
            )
            write_cache(summary_path, summary)
            return summary
        if repository.get("mirror_url"):
            summary = _collection_failure(
                repo,
                "repository_mirror",
                candidate_batch=candidate_batch,
                stage=stage,
                cheap_prescreen_pass=False,
                detailed_collection_attempted=False,
            )
            write_cache(summary_path, summary)
            return summary

        default_branch = str(repository.get("default_branch") or "")
        branch = {}
        if default_branch:
            stage = "default_branch_metadata"
            branch = client.get_json(
                f"/repos/{repo.owner}/{repo.repo}/branches/{urllib.parse.quote(default_branch, safe='')}",
                {},
                repo_dir / "default_branch.json",
                prefer_cache=True,
            )
        stage = "repository_languages"
        languages = client.get_json(
            f"/repos/{repo.owner}/{repo.repo}/languages",
            {},
            repo_dir / "languages.json",
            prefer_cache=True,
        )
        repository_screen_reason = cheap_repository_metadata_rejection(repository, languages, config)
        if repository_screen_reason:
            summary = _collection_failure(
                repo,
                repository_screen_reason,
                candidate_batch=candidate_batch,
                stage="cheap_repository_prescreen",
                cheap_prescreen_pass=False,
                detailed_collection_attempted=False,
            )
            summary.update(
                {
                    "repository_id": repository.get("id"),
                    "default_branch": default_branch,
                    "default_branch_head_sha": (((branch.get("commit") or {}).get("sha")) if isinstance(branch, dict) else ""),
                    "languages": languages if isinstance(languages, dict) else {},
                }
            )
            write_cache(summary_path, summary)
            return summary

        stage = "pull_request_discovery"
        closed_prs, complete_pr_history = collect_all_closed_pull_requests(client, repo, repo_dir, config)
        if not complete_pr_history:
            summary = _collection_failure(
                repo,
                "closed_pr_history_page_limit_reached",
                candidate_batch=candidate_batch,
                stage=stage,
                cheap_prescreen_pass=False,
                detailed_collection_attempted=False,
            )
            summary["closed_pr_count_at_limit"] = len(closed_prs)
            write_cache(summary_path, summary)
            return summary
        merged_prs = [
            pr for pr in closed_prs
            if pr.get("merged_at") and pr.get("created_at")
        ]
        bot_patterns = compile_bot_patterns(bot_config)
        human_prs = [
            pr for pr in merged_prs
            if not is_bot_identity(_pr_author(pr), bot_patterns)
        ]
        if len(human_prs) < int(config.get("min_merged_prs", 5)):
            summary = _collection_failure(
                repo,
                "fewer_than_five_non_bot_merged_prs",
                candidate_batch=candidate_batch,
                stage=stage,
                cheap_prescreen_pass=False,
                detailed_collection_attempted=False,
            )
            summary["full_history_merged_pr_count"] = len(merged_prs)
            summary["full_history_human_pr_count"] = len(human_prs)
            write_cache(summary_path, summary)
            return summary
        plausible_human_contributors = sorted({author for author in (_pr_author(pr) for pr in human_prs) if author})
        if len(plausible_human_contributors) < int(config.get("min_human_contributors", 5)):
            summary = _collection_failure(
                repo,
                "fewer_than_five_plausible_human_contributors",
                candidate_batch=candidate_batch,
                stage=stage,
                cheap_prescreen_pass=False,
                detailed_collection_attempted=False,
            )
            summary["full_history_merged_pr_count"] = len(merged_prs)
            summary["full_history_human_pr_count"] = len(human_prs)
            summary["plausible_human_contributor_count"] = len(plausible_human_contributors)
            write_cache(summary_path, summary)
            return summary
        bot_share = _bot_pr_share(merged_prs, bot_patterns)
        max_bot_share = float((config.get("cheap_prescreen") or {}).get("max_bot_pr_share", 0.80))
        if bot_share > max_bot_share:
            summary = _collection_failure(
                repo,
                "bot_dominated_pull_request_history",
                candidate_batch=candidate_batch,
                stage=stage,
                cheap_prescreen_pass=False,
                detailed_collection_attempted=False,
            )
            summary["full_history_merged_pr_count"] = len(merged_prs)
            summary["full_history_human_pr_count"] = len(human_prs)
            summary["bot_pr_share"] = bot_share
            write_cache(summary_path, summary)
            return summary

        start, end = complete_history_bounds(human_prs)
        days = project_history_days(start, end)
        min_days, max_days = config.get("project_history_days_range", [90, 548])
        if days < float(min_days) or days > float(max_days):
            summary = _collection_failure(
                repo,
                "project_history_outside_90_548_days",
                candidate_batch=candidate_batch,
                stage=stage,
                cheap_prescreen_pass=False,
                detailed_collection_attempted=False,
            )
            summary.update(
                {
                    "project_history_start": start,
                    "project_history_end": end,
                    "project_history_days": days,
                    "project_history_months": project_history_months(days),
                    "full_history_merged_pr_count": len(merged_prs),
                    "full_history_human_pr_count": len(human_prs),
                }
            )
            write_cache(summary_path, summary)
            return summary

        stage = "repository_releases"
        releases = cached_body(repo_dir / "releases.json")
        if releases is None:
            releases = client.paginate(
                f"/repos/{repo.owner}/{repo.repo}/releases",
                {"per_page": int(config.get("per_page", 100))},
                repo_dir / "releases",
                max_pages=100,
                prefer_cache=True,
            )
            write_cache(repo_dir / "releases.json", {"body": releases, "fetched_at": utc_now()})
        stage = "repository_milestones"
        milestones = cached_body(repo_dir / "milestones.json")
        if milestones is None:
            milestones = client.paginate(
                f"/repos/{repo.owner}/{repo.repo}/milestones",
                {"state": "all", "per_page": int(config.get("per_page", 100))},
                repo_dir / "milestones",
                max_pages=100,
                prefer_cache=True,
            )
            write_cache(repo_dir / "milestones.json", {"body": milestones, "fetched_at": utc_now()})
    except GitHubNotFoundError as exc:
        summary = _github_not_found_failure(
            repo,
            exc,
            stage,
            candidate_batch=candidate_batch,
            cheap_prescreen_pass=stage in {"repository_releases", "repository_milestones"},
            detailed_collection_attempted=False,
        )
        write_cache(summary_path, summary)
        return summary

    window_dir = repo_dir / FULL_PROJECT_WINDOW_NAME
    detail_limit = int(config.get("full_detail_pr_limit", 200))
    if len(human_prs) <= detail_limit:
        detailed_prs = sorted(human_prs, key=lambda item: (str(item.get("merged_at") or ""), int(item.get("number", 0))))
        technical_detail_mode = "complete_pr_file_details"
    else:
        detailed_prs = deterministic_pr_sample(
            human_prs,
            seed=derive_seed(seed, repo.index, salt=8_800_171),
            sample_size=int(config.get("detailed_pr_sample_size", 60)),
            monthly_target=int(config.get("monthly_pr_sample_target", 10)),
            max_author_share=float(config.get("max_author_sample_share", 0.20)),
        )
        technical_detail_mode = "deterministic_pr_file_detail_sample"

    detailed_collection_attempted = True
    write_cache(window_dir / "merged_pull_requests.json", {"body": human_prs, "fetched_at": utc_now()})
    write_cache(window_dir / "sampled_pull_requests.json", {"body": detailed_prs, "fetched_at": utc_now()})
    write_cache(window_dir / "commits.json", {"body": [], "fetched_at": utc_now()})
    for idx, pr in enumerate(detailed_prs, start=1):
        number = int(pr["number"])
        if cached_body(window_dir / "pr_files" / f"{number}.json") is not None:
            continue
        logging.info(
            "Type5 full-project %s file details: %d/%d PR #%d",
            repo.full_name,
            idx,
            len(detailed_prs),
            number,
        )
        try:
            files = client.paginate(
                f"/repos/{repo.owner}/{repo.repo}/pulls/{number}/files",
                {"per_page": int(config.get("per_page", 100))},
                window_dir / "pr_files" / str(number),
                max_pages=20,
                prefer_cache=True,
            )
        except GitHubNotFoundError as exc:
            summary = _github_not_found_failure(
                repo,
                exc,
                "pr_file_details",
                candidate_batch=candidate_batch,
                cheap_prescreen_pass=True,
                detailed_collection_attempted=True,
            )
            write_cache(summary_path, summary)
            return summary
        write_cache(window_dir / "pr_files" / f"{number}.json", {"body": files, "fetched_at": utc_now()})

    collection_timestamp = utc_now()
    window_summary = {
        "window": {
            "name": FULL_PROJECT_WINDOW_NAME,
            "start": start,
            "end": end,
        },
        "source_pr_total_count": len(human_prs),
        "source_sampled_pr_count": len(detailed_prs),
        "source_commit_count": 0,
        "collected_at": collection_timestamp,
        "observation_mode": "full_project",
        "source_window_type": "complete_eligible_project_history",
    }
    write_cache(window_dir / "window_collection.json", window_summary)
    summary = {
        "valid": True,
        "failure_reason": "",
        "candidate_batch": _candidate_batch(repo, candidate_batch),
        "cheap_prescreen_pass": True,
        "detailed_collection_attempted": detailed_collection_attempted,
        "profile_quality_pass": "",
        "rejection_reason": "",
        "source": asdict(repo),
        "repository_id": repository.get("id"),
        "default_branch": default_branch,
        "default_branch_head_sha": (((branch.get("commit") or {}).get("sha")) if isinstance(branch, dict) else ""),
        "collection_timestamp": collection_timestamp,
        "project_history_start": start,
        "project_history_end": end,
        "project_history_days": days,
        "project_history_months": project_history_months(days),
        "full_history_closed_pr_count": len(closed_prs),
        "full_history_merged_pr_count": len(merged_prs),
        "full_history_human_pr_count": len(human_prs),
        "detailed_pr_count": len(detailed_prs),
        "technical_detail_mode": technical_detail_mode,
        "repository_overlap_with_existing_type5": 0,
        "windows": {FULL_PROJECT_WINDOW_NAME: window_summary},
        "languages": languages if isinstance(languages, dict) else {},
    }
    write_cache(summary_path, summary)
    write_cache(repo_dir / "collection_summary.json", summary)
    return summary


def collect_full_project_repositories(
    client: GitHubClient,
    repos: Sequence[SourceRepo | CandidateSource],
    cache_dir: str | Path,
    config: Mapping[str, Any],
    bot_config: Mapping[str, Any],
    seed: int,
    exclusion_records: Sequence[Mapping[str, str]],
) -> List[Dict[str, Any]]:
    """Collect candidates in order, treating per-source rejections as data."""

    summaries = []
    for idx, candidate in enumerate(repos, start=1):
        if isinstance(candidate, CandidateSource):
            repo = candidate.repo
            candidate_batch = candidate.candidate_batch
        else:
            repo = candidate
            candidate_batch = None
        repo = SourceRepo(repo.owner, repo.repo, repo.url, repo.domain, repo.kind, idx)
        logging.info("Type5 full-project candidate %02d/%02d %s", idx, len(repos), repo.full_name)
        summaries.append(
            collect_full_project_repository(
                client,
                repo,
                cache_dir,
                config,
                bot_config,
                seed=derive_seed(seed, idx, salt=7_777_901),
                exclusion_records=exclusion_records,
                candidate_batch=candidate_batch,
            )
        )
    return summaries


def collect_all_closed_pull_requests(
    client: GitHubClient,
    repo: SourceRepo,
    repo_dir: Path,
    config: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], bool]:
    cached_record = _safe_read_json(repo_dir / FULL_PROJECT_WINDOW_NAME / "closed_pull_requests.json")
    cached = cached_record.get("body") if isinstance(cached_record, dict) else None
    if isinstance(cached, list) and bool(cached_record.get("complete", True)):
        return cached, True
    if isinstance(cached, list) and not bool(cached_record.get("complete", True)):
        return cached, False

    records: List[Dict[str, Any]] = []
    per_page = int(config.get("per_page", 100))
    max_pages = int(config.get("max_full_pr_pages", 1000))
    complete = True
    for page in range(1, max_pages + 1):
        body = client.get_json(
            f"/repos/{repo.owner}/{repo.repo}/pulls",
            {
                "state": "closed",
                "sort": "created",
                "direction": "asc",
                "per_page": per_page,
                "page": page,
            },
            repo_dir / FULL_PROJECT_WINDOW_NAME / "pulls" / f"page_{page:04d}.json",
            prefer_cache=True,
        )
        if not body:
            complete = True
            break
        if not isinstance(body, list):
            raise RuntimeError("Pull request list endpoint did not return a list")
        records.extend(body)
        if len(body) < per_page:
            complete = True
            break
    else:
        complete = False

    write_cache(
        repo_dir / FULL_PROJECT_WINDOW_NAME / "closed_pull_requests.json",
        {"body": records, "fetched_at": utc_now(), "complete": complete},
    )
    return records, complete


def complete_history_bounds(prs: Sequence[Mapping[str, Any]]) -> Tuple[str, str]:
    starts = [str(pr.get("created_at")) for pr in prs if pr.get("created_at")]
    ends = [str(pr.get("merged_at")) for pr in prs if pr.get("merged_at")]
    if not starts or not ends:
        raise RuntimeError("Cannot compute complete project history bounds without created_at and merged_at")
    return min(starts), max(ends)


def write_collection_outputs(
    summaries: Sequence[Mapping[str, Any]],
    metadata_dir: str | Path,
    exclusion_records: Sequence[Mapping[str, str]],
) -> None:
    metadata_dir = Path(metadata_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    normalized_summaries = [_summary_with_metadata_defaults(summary) for summary in summaries]
    for summary in normalized_summaries:
        source = summary.get("source", {})
        candidate_batch = str(summary.get("candidate_batch") or _candidate_batch_from_source(source))
        rows.append(
            {
                "repository": canonical_repo(str(source.get("owner", "")), str(source.get("repo", ""))),
                "owner": source.get("owner", ""),
                "repo": source.get("repo", ""),
                "url": source.get("url", ""),
                "domain": source.get("domain", ""),
                "repository_id": summary.get("repository_id", ""),
                "default_branch": summary.get("default_branch", ""),
                "snapshot_sha": summary.get("default_branch_head_sha", ""),
                "snapshot_timestamp": summary.get("collection_timestamp", ""),
                "project_history_start": summary.get("project_history_start", ""),
                "project_history_end": summary.get("project_history_end", ""),
                "project_history_days": summary.get("project_history_days", ""),
                "project_history_months": summary.get("project_history_months", ""),
                "human_contributor_count": summary.get("full_history_human_pr_count", ""),
                "merged_pr_count": summary.get("full_history_merged_pr_count", ""),
                "sampled_pr_count": summary.get("detailed_pr_count", ""),
                "technical_detail_mode": summary.get("technical_detail_mode", ""),
                "candidate_batch": candidate_batch,
                "cheap_prescreen_pass": _summary_cheap_prescreen_pass(summary),
                "detailed_collection_attempted": _summary_detailed_collection_attempted(summary),
                "profile_quality_pass": summary.get("profile_quality_pass", "" if summary.get("valid") else False),
                "quality_status": "collected" if summary.get("valid") else "rejected",
                "rejection_reason": summary.get("rejection_reason") or summary.get("failure_reason", ""),
                "rejection_endpoint": summary.get("rejection_endpoint", ""),
                "rejection_http_status": summary.get("rejection_http_status", ""),
                "rejection_timestamp": summary.get("rejection_timestamp", ""),
                "rejection_stage": summary.get("rejection_stage", ""),
            }
        )
    fields = sorted({key for row in rows for key in row})
    with (metadata_dir / "type5_full_project_candidates.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    payload = {
        "extension": "type5_full_project",
        "collected_at": utc_now(),
        "candidate_count": len(normalized_summaries),
        "valid_collection_count": sum(1 for item in normalized_summaries if item.get("valid")),
        "existing_type5_repository_exclusion_list": list(exclusion_records),
        "repository_overlap_with_existing_type5": sum(
            int(item.get("repository_overlap_with_existing_type5", 0) or 0) for item in normalized_summaries
        ),
        "candidates": normalized_summaries,
    }
    (metadata_dir / "type5_full_project_collection.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def is_complete_full_project_summary(summary: Mapping[str, Any], repo: SourceRepo) -> bool:
    if not summary:
        return False
    source = summary.get("source", {})
    return (
        isinstance(source, Mapping)
        and source.get("owner") == repo.owner
        and source.get("repo") == repo.repo
        and "valid" in summary
        and "collection_timestamp" in summary
    )


def cheap_repository_metadata_rejection(
    repository: Mapping[str, Any],
    languages: Mapping[str, Any] | Sequence[Any] | None,
    config: Mapping[str, Any],
) -> str:
    """Return a cheap source-level rejection reason before PR detail fetches."""

    technical_bytes, total_bytes = _technical_language_bytes(languages, config)
    cheap_config = config.get("cheap_prescreen") or {}
    if total_bytes <= 0 or technical_bytes < int(cheap_config.get("min_technical_language_bytes", 1)):
        return "insufficient_technical_activity"

    normalized = _repository_text(repository)
    personal_keywords = [str(value).lower() for value in cheap_config.get("personal_only_keywords", [])]
    if any(keyword and keyword in normalized for keyword in personal_keywords):
        return "repository_documentation_tutorial_or_personal_only"

    low_activity_keywords = [str(value).lower() for value in cheap_config.get("low_activity_keywords", [])]
    technical_share = technical_bytes / max(1, total_bytes)
    if technical_share < 0.20 and any(keyword and keyword in normalized for keyword in low_activity_keywords):
        return "repository_documentation_tutorial_or_personal_only"
    return ""


def _collection_failure(
    repo: SourceRepo,
    reason: str,
    *,
    candidate_batch: str | None = None,
    stage: str = "",
    cheap_prescreen_pass: bool = False,
    detailed_collection_attempted: bool = False,
) -> Dict[str, Any]:
    return {
        "valid": False,
        "failure_reason": reason,
        "candidate_batch": _candidate_batch(repo, candidate_batch),
        "cheap_prescreen_pass": bool(cheap_prescreen_pass),
        "detailed_collection_attempted": bool(detailed_collection_attempted),
        "profile_quality_pass": False,
        "rejection_reason": reason,
        "rejection_stage": stage,
        "source": asdict(repo),
        "collection_timestamp": utc_now(),
    }


def _github_not_found_failure(
    repo: SourceRepo,
    exc: GitHubNotFoundError,
    stage: str,
    *,
    candidate_batch: str | None = None,
    cheap_prescreen_pass: bool = False,
    detailed_collection_attempted: bool = False,
) -> Dict[str, Any]:
    timestamp = utc_now()
    summary = _collection_failure(
        repo,
        "github_repository_not_found_or_inaccessible",
        candidate_batch=candidate_batch,
        stage=stage,
        cheap_prescreen_pass=cheap_prescreen_pass,
        detailed_collection_attempted=detailed_collection_attempted,
    )
    summary.update(
        {
            "repository": canonical_repo(repo.owner, repo.repo),
            "rejection_endpoint": exc.endpoint,
            "rejection_http_status": exc.status,
            "rejection_timestamp": timestamp,
            "rejection_stage": stage,
            "github_failure": {
                "repository": canonical_repo(repo.owner, repo.repo),
                "endpoint": exc.endpoint,
                "http_status": exc.status,
                "timestamp": timestamp,
                "stage": stage,
                "reason": "github_repository_not_found_or_inaccessible",
            },
        }
    )
    return summary


def _summary_with_metadata_defaults(summary: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(summary)
    source = result.get("source", {})
    result.setdefault("candidate_batch", _candidate_batch_from_source(source))
    result.setdefault("cheap_prescreen_pass", _summary_cheap_prescreen_pass(summary))
    result.setdefault("detailed_collection_attempted", _summary_detailed_collection_attempted(summary))
    result.setdefault("profile_quality_pass", "" if result.get("valid") else False)
    result.setdefault("rejection_reason", "" if result.get("valid") else result.get("failure_reason", ""))
    return result


def _candidate_batch(repo: SourceRepo, explicit: str | None = None) -> str:
    if explicit:
        return explicit
    return "initial_001_060" if int(repo.index or 0) <= 60 else "expansion_061_150"


def _candidate_batch_from_source(source: Any) -> str:
    if not isinstance(source, Mapping):
        return "initial_001_060"
    if source.get("candidate_batch"):
        return str(source["candidate_batch"])
    try:
        index = int(source.get("index") or 0)
    except (TypeError, ValueError):
        index = 0
    return "initial_001_060" if index <= 60 else "expansion_061_150"


def _renumber_candidates(candidates: Sequence[CandidateSource]) -> List[CandidateSource]:
    renumbered: List[CandidateSource] = []
    for idx, candidate in enumerate(candidates, start=1):
        repo = candidate.repo
        renumbered.append(
            CandidateSource(
                SourceRepo(repo.owner, repo.repo, repo.url, repo.domain, repo.kind, idx),
                candidate.candidate_batch,
            )
        )
    return renumbered


def _load_resume_candidate_summaries(cache_dir: Path, metadata_dir: Path) -> List[Mapping[str, Any]]:
    summaries: Dict[str, Mapping[str, Any]] = {}
    collection_path = metadata_dir / "type5_full_project_collection.json"
    if collection_path.exists():
        payload = _safe_read_json(collection_path)
        for summary in payload.get("candidates", []) if isinstance(payload.get("candidates"), list) else []:
            source = summary.get("source", {}) if isinstance(summary, Mapping) else {}
            if isinstance(source, Mapping):
                summaries[canonical_repo(str(source.get("owner", "")), str(source.get("repo", "")))] = summary
    if cache_dir.exists():
        for path in sorted(cache_dir.glob("*/full_project_collection_summary.json")):
            summary = _safe_read_json(path)
            source = summary.get("source", {}) if isinstance(summary, Mapping) else {}
            if isinstance(source, Mapping):
                key = canonical_repo(str(source.get("owner", "")), str(source.get("repo", "")))
                summaries.setdefault(key, summary)
    return [summary for key, summary in summaries.items() if key != "/"]


def _technical_language_bytes(
    languages: Mapping[str, Any] | Sequence[Any] | None,
    config: Mapping[str, Any],
) -> Tuple[int, int]:
    if not isinstance(languages, Mapping):
        return 0, 0
    cheap_config = config.get("cheap_prescreen") or {}
    nontechnical = {
        str(value).lower()
        for value in cheap_config.get("nontechnical_languages", [])
    }
    total = 0
    technical = 0
    for language, raw_bytes in languages.items():
        try:
            byte_count = int(raw_bytes or 0)
        except (TypeError, ValueError):
            byte_count = 0
        total += max(0, byte_count)
        if str(language).lower() not in nontechnical:
            technical += max(0, byte_count)
    return technical, total


def _repository_text(repository: Mapping[str, Any]) -> str:
    values = [
        repository.get("name", ""),
        repository.get("full_name", ""),
        repository.get("description", ""),
        repository.get("homepage", ""),
    ]
    topics = repository.get("topics", [])
    if isinstance(topics, list):
        values.extend(str(topic) for topic in topics)
    return " ".join(str(value).lower().replace("-", " ").replace("_", " ") for value in values if value)


def _bot_pr_share(prs: Sequence[Mapping[str, Any]], bot_patterns: Sequence[str]) -> float:
    if not prs:
        return 0.0
    bot_count = sum(1 for pr in prs if is_bot_identity(_pr_author(pr), bot_patterns))
    return bot_count / max(1, len(prs))


def _summary_cheap_prescreen_pass(summary: Mapping[str, Any]) -> Any:
    if "cheap_prescreen_pass" in summary:
        return summary.get("cheap_prescreen_pass")
    if summary.get("valid"):
        return True
    stage = str(summary.get("rejection_stage", ""))
    if stage in {
        "existing_type5_overlap",
        "repository_metadata",
        "default_branch_metadata",
        "repository_languages",
        "cheap_repository_prescreen",
        "pull_request_discovery",
    }:
        return False
    if stage in {"repository_releases", "repository_milestones", "pr_file_details"}:
        return True
    return ""


def _summary_detailed_collection_attempted(summary: Mapping[str, Any]) -> Any:
    if "detailed_collection_attempted" in summary:
        return summary.get("detailed_collection_attempted")
    if summary.get("valid") or summary.get("detailed_pr_count") or summary.get("technical_detail_mode"):
        return True
    return bool(str(summary.get("rejection_stage", "")) == "pr_file_details")


def _pr_author(pr: Mapping[str, Any]) -> str | None:
    user = pr.get("user")
    if isinstance(user, dict) and user.get("login"):
        return str(user["login"])
    value = pr.get("author_login")
    return str(value) if value else None


def _safe_read_json(path: Path) -> Dict[str, Any]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="benchmark_generation/configs/type5_full_project.yaml")
    parser.add_argument("--cache-dir", default="type5_full_project_source_cache")
    parser.add_argument("--metadata-dir", default="benchmark_metadata")
    parser.add_argument("--bot-config", default="benchmark_generation/configs/type5_bot_patterns.yaml")
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
    bot_config = load_config(args.bot_config, {})
    seed = args.seed if args.seed is not None else int(config.get("root_seed", 2026))
    exclusion_records = existing_type5_repository_exclusion_records(args.metadata_dir)
    write_exclusion_list(exclusion_records, args.metadata_dir)
    token = get_github_token()
    client = GitHubClient(
        token,
        api_version=str(config.get("api_version", "2026-03-10")),
        retry_attempts=int(config.get("retry_attempts", 4)),
        backoff_initial_seconds=float(config.get("backoff_initial_seconds", 2.0)),
    )
    repos = discover_candidate_repositories(client, config, args.cache_dir, exclusion_records, args.metadata_dir)
    if len(repos) < int(config.get("candidate_minimum_before_selection", 120)):
        raise RuntimeError(
            f"Only {len(repos)} candidate repositories discovered or resumed; "
            f"{config.get('candidate_minimum_before_selection', 120)} requested before source selection."
        )
    summaries = collect_full_project_repositories(
        client,
        repos,
        args.cache_dir,
        config,
        bot_config,
        seed=seed,
        exclusion_records=exclusion_records,
    )
    write_collection_outputs(summaries, args.metadata_dir, exclusion_records)


if __name__ == "__main__":
    main()
