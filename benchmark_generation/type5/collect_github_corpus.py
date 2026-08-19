"""Collect and cache public GitHub activity for Type 5 source profiles."""

from __future__ import annotations

import argparse
import csv
import http.client
import json
import logging
import os
import socket
import ssl
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping

from benchmark_generation.core.config import load_config
from benchmark_generation.core.generator import derive_seed
from benchmark_generation.type5.sampling import deterministic_pr_sample
from benchmark_generation.type5.sources import (
    ObservationWindow,
    SourceRepo,
    collection_sources,
    load_type5_sources,
    repo_cache_dir,
)


GITHUB_API_BASE = "https://api.github.com"
DEFAULT_COLLECT_CONFIG: Dict[str, Any] = {
    "api_version": "2026-03-10",
    "per_page": 100,
    "max_pr_pages_per_window": 1000,
    "max_commit_pages_per_window": 50,
    "detailed_pr_sample_size": 60,
    "monthly_pr_sample_target": 10,
    "max_author_sample_share": 0.20,
    "retry_attempts": 4,
    "backoff_initial_seconds": 2.0,
}


class AuthenticationError(RuntimeError):
    """Raised when authenticated public GitHub access is unavailable."""


class GitHubClient:
    def __init__(
        self,
        token: str,
        api_version: str = "2026-03-10",
        retry_attempts: int = 4,
        backoff_initial_seconds: float = 2.0,
    ) -> None:
        if not token:
            raise AuthenticationError("Authenticated GitHub access is required for Type 5 collection.")
        self.token = token
        self.api_version = api_version
        self.retry_attempts = max(1, int(retry_attempts))
        self.backoff_initial_seconds = max(0.1, float(backoff_initial_seconds))

    def get_json(
        self,
        endpoint: str,
        params: Mapping[str, Any] | None,
        cache_path: Path,
        prefer_cache: bool = False,
    ) -> Any:
        url = make_url(endpoint, params)
        cached = read_cache(cache_path)
        if prefer_cache and cached_body_available(cached):
            logging.debug("GitHub resume cache hit: %s", cache_path)
            return cached["body"]
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": self.api_version,
            "Authorization": f"Bearer {self.token}",
            "User-Agent": "mtp-type5-benchmark-generator",
        }
        if cached.get("etag"):
            headers["If-None-Match"] = str(cached["etag"])

        last_error = None
        for attempt in range(1, self.retry_attempts + 1):
            request = urllib.request.Request(url, headers=headers, method="GET")
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    body = json.loads(response.read().decode("utf-8"))
                    payload = {
                        "url": url,
                        "endpoint": endpoint,
                        "params": dict(params or {}),
                        "status": response.status,
                        "etag": response.headers.get("ETag"),
                        "last_modified": response.headers.get("Last-Modified"),
                        "fetched_at": utc_now(),
                        "api_version": self.api_version,
                        "rate_limit": rate_limit_headers(response.headers),
                        "body": body,
                    }
                    write_cache(cache_path, payload)
                    log_rate_limit(endpoint, payload["rate_limit"])
                    return body
            except urllib.error.HTTPError as exc:
                if exc.code == 304 and cached:
                    logging.debug("GitHub cache hit via ETag: %s", endpoint)
                    return cached.get("body")
                if exc.code in {403, 429, 500, 502, 503, 504} and attempt < self.retry_attempts:
                    wait = retry_wait_seconds(exc, self.backoff_initial_seconds, attempt)
                    logging.warning(
                        "GitHub API retry endpoint=%s status=%s wait=%.1fs",
                        endpoint,
                        exc.code,
                        wait,
                    )
                    time.sleep(wait)
                    last_error = exc
                    continue
                last_error = exc
                break
            except TRANSIENT_NETWORK_EXCEPTIONS as exc:
                if is_transient_network_exception(exc) and attempt < self.retry_attempts:
                    wait = self.backoff_initial_seconds * (2 ** (attempt - 1))
                    logging.warning(
                        "GitHub API transient network error endpoint=%s error=%s wait=%.1fs",
                        endpoint,
                        exc.__class__.__name__,
                        wait,
                    )
                    time.sleep(wait)
                    last_error = exc
                    continue
                last_error = exc
                break
        if cached:
            logging.warning("Using cached fallback after GitHub endpoint failure: %s", endpoint)
            return cached.get("body")
        raise RuntimeError(f"GitHub endpoint failed without cached fallback: {endpoint}: {last_error}")

    def paginate(
        self,
        endpoint: str,
        params: Mapping[str, Any] | None,
        cache_dir: Path,
        max_pages: int = 1000,
        prefer_cache: bool = False,
    ) -> List[Any]:
        records: List[Any] = []
        for page in range(1, max_pages + 1):
            page_params = dict(params or {})
            page_params.setdefault("per_page", 100)
            page_params["page"] = page
            body = self.get_json(
                endpoint,
                page_params,
                cache_dir / f"page_{page:04d}.json",
                prefer_cache=prefer_cache,
            )
            if not body:
                break
            if not isinstance(body, list):
                raise RuntimeError(f"Expected paginated list from {endpoint}; got {type(body).__name__}")
            records.extend(body)
            if len(body) < int(page_params["per_page"]):
                break
        return records


TRANSIENT_NETWORK_EXCEPTIONS = (
    urllib.error.URLError,
    TimeoutError,
    http.client.RemoteDisconnected,
    http.client.IncompleteRead,
    http.client.BadStatusLine,
    ConnectionResetError,
    ConnectionAbortedError,
    BrokenPipeError,
    socket.timeout,
    ssl.SSLError,
)


def is_transient_network_exception(exc: BaseException) -> bool:
    """Return whether an exception is a retryable transport disconnect.

    Permanent HTTP responses are handled separately as ``HTTPError``. SSL
    certificate verification errors are intentionally not retried or hidden.
    """

    if isinstance(exc, urllib.error.URLError):
        reason = getattr(exc, "reason", None)
        return True if reason is None else is_transient_network_exception(reason)
    if isinstance(exc, ssl.SSLCertVerificationError):
        return False
    if isinstance(exc, ssl.SSLEOFError):
        return True
    if isinstance(exc, ssl.SSLError):
        text = str(exc).lower()
        return any(token in text for token in ("eof", "timed out", "timeout", "connection reset", "bad record mac"))
    return isinstance(
        exc,
        (
            TimeoutError,
            http.client.RemoteDisconnected,
            http.client.IncompleteRead,
            http.client.BadStatusLine,
            ConnectionResetError,
            ConnectionAbortedError,
            BrokenPipeError,
            socket.timeout,
        ),
    )


def make_url(endpoint: str, params: Mapping[str, Any] | None = None) -> str:
    endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
    url = f"{GITHUB_API_BASE}{endpoint}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    return url


def read_cache(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def safe_read_cache(path: Path) -> Dict[str, Any]:
    try:
        return read_cache(path)
    except (OSError, json.JSONDecodeError):
        return {}


def cached_body_available(cache_record: Mapping[str, Any]) -> bool:
    return isinstance(cache_record, Mapping) and "body" in cache_record


def cached_body(path: Path) -> Any:
    record = safe_read_cache(path)
    if cached_body_available(record):
        return record["body"]
    return None


def write_cache(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def rate_limit_headers(headers: Mapping[str, str]) -> Dict[str, str]:
    return {
        key.lower(): str(value)
        for key, value in headers.items()
        if key.lower().startswith("x-ratelimit") or key.lower() == "retry-after"
    }


def log_rate_limit(endpoint: str, headers: Mapping[str, str]) -> None:
    if headers:
        logging.debug("GitHub rate endpoint=%s headers=%s", endpoint, headers)


def retry_wait_seconds(exc: urllib.error.HTTPError, initial: float, attempt: int) -> float:
    retry_after = exc.headers.get("Retry-After")
    if retry_after:
        try:
            return max(1.0, float(retry_after))
        except ValueError:
            pass
    return initial * (2 ** (attempt - 1))


def get_github_token() -> str:
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    try:
        status = subprocess.run(
            ["gh", "auth", "status"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        status = None
    if status is not None and status.returncode == 0:
        token_result = subprocess.run(
            ["gh", "auth", "token"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        if token_result.returncode == 0 and token_result.stdout.strip():
            return token_result.stdout.strip()
    raise AuthenticationError(
        "Authenticated public GitHub access is required. Set GITHUB_TOKEN or run `gh auth login`."
    )


def collect_repository(
    client: GitHubClient,
    repo: SourceRepo,
    windows: List[ObservationWindow],
    cache_dir: Path,
    config: Mapping[str, Any],
    seed: int,
) -> Dict[str, Any]:
    repo_dir = repo_cache_dir(cache_dir, repo)
    summary_path = repo_dir / "collection_summary.json"
    existing_summary = safe_read_cache(summary_path)
    if is_complete_collection_summary(existing_summary, repo, windows):
        logging.info("Type5 source %02d %s already complete; skipping", repo.index, repo.full_name)
        return existing_summary

    logging.info("Type5 source %02d %s", repo.index, repo.full_name)
    repository = client.get_json(
        f"/repos/{repo.owner}/{repo.repo}",
        {},
        repo_dir / "repository.json",
        prefer_cache=True,
    )
    if repository.get("private") or repository.get("archived") or repository.get("disabled") or repository.get("fork"):
        failure = {
            "source": asdict(repo),
            "failure_reason": "repository_private_archived_disabled_or_fork",
            "collected_at": utc_now(),
        }
        write_cache(repo_dir / "repository_failure.json", failure)
        return failure
    default_branch = str(repository.get("default_branch") or "")
    if default_branch:
        branch = client.get_json(
            f"/repos/{repo.owner}/{repo.repo}/branches/{urllib.parse.quote(default_branch, safe='')}",
            {},
            repo_dir / "default_branch.json",
            prefer_cache=True,
        )
    else:
        branch = {}
    languages = client.get_json(
        f"/repos/{repo.owner}/{repo.repo}/languages",
        {},
        repo_dir / "languages.json",
        prefer_cache=True,
    )
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

    summary = {
        "source": asdict(repo),
        "repository_id": repository.get("id"),
        "default_branch": default_branch,
        "default_branch_head_sha": (((branch.get("commit") or {}).get("sha")) if isinstance(branch, dict) else ""),
        "collection_timestamp": utc_now(),
        "windows": {},
    }
    for window_idx, window in enumerate(windows, start=1):
        summary["windows"][window.name] = collect_window(
            client,
            repo,
            window,
            repo_dir,
            config,
            seed=derive_seed(seed, repo.index * 10 + window_idx, salt=5_000_003),
        )
    write_cache(summary_path, summary)
    return summary


def collect_window(
    client: GitHubClient,
    repo: SourceRepo,
    window: ObservationWindow,
    repo_dir: Path,
    config: Mapping[str, Any],
    seed: int,
) -> Dict[str, Any]:
    window_dir = repo_dir / window.name
    window_summary_path = window_dir / "window_collection.json"
    existing_window = safe_read_cache(window_summary_path)
    if is_complete_window_collection(existing_window, window):
        logging.info("Type5 %s %s already complete; skipping window", repo.full_name, window.name)
        return existing_window

    logging.info("Type5 %s %s PR discovery", repo.full_name, window.name)
    pr_pages_dir = window_dir / "pulls"
    merged_prs = cached_body(window_dir / "merged_pull_requests.json")
    if merged_prs is None:
        merged_prs = discover_merged_pull_requests(
            client,
            repo,
            window,
            pr_pages_dir,
            per_page=int(config.get("per_page", 100)),
            max_pages=int(config.get("max_pr_pages_per_window", 1000)),
            prefer_cache=True,
        )
        write_cache(window_dir / "merged_pull_requests.json", {"body": merged_prs, "fetched_at": utc_now()})
    sample = cached_body(window_dir / "sampled_pull_requests.json")
    if sample is None:
        sample = deterministic_pr_sample(
            merged_prs,
            seed=seed,
            sample_size=int(config.get("detailed_pr_sample_size", 60)),
            monthly_target=int(config.get("monthly_pr_sample_target", 10)),
            max_author_share=float(config.get("max_author_sample_share", 0.20)),
        )
        write_cache(window_dir / "sampled_pull_requests.json", {"body": sample, "fetched_at": utc_now()})
    files_dir = window_dir / "pr_files"
    for idx, pr in enumerate(sample, start=1):
        number = int(pr["number"])
        if cached_body(files_dir / f"{number}.json") is not None:
            logging.debug(
                "Type5 %s %s file details cached: %d/%d PR #%d",
                repo.full_name,
                window.name,
                idx,
                len(sample),
                number,
            )
            continue
        logging.info(
            "Type5 %s %s file details: %d/%d PR #%d",
            repo.full_name,
            window.name,
            idx,
            len(sample),
            number,
        )
        files = client.paginate(
            f"/repos/{repo.owner}/{repo.repo}/pulls/{number}/files",
            {"per_page": int(config.get("per_page", 100))},
            files_dir / str(number),
            max_pages=20,
            prefer_cache=True,
        )
        write_cache(files_dir / f"{number}.json", {"body": files, "fetched_at": utc_now()})

    commits = cached_body(window_dir / "commits.json")
    if commits is None:
        commits = collect_commits(
            client,
            repo,
            window,
            window_dir / "commits",
            per_page=int(config.get("per_page", 100)),
            max_pages=int(config.get("max_commit_pages_per_window", 50)),
            prefer_cache=True,
        )
        write_cache(window_dir / "commits.json", {"body": commits, "fetched_at": utc_now()})
    result = {
        "window": asdict(window),
        "source_pr_total_count": len(merged_prs),
        "source_sampled_pr_count": len(sample),
        "source_commit_count": len(commits),
        "collected_at": utc_now(),
    }
    write_cache(window_summary_path, result)
    return result


def is_complete_collection_summary(
    summary: Mapping[str, Any],
    repo: SourceRepo,
    windows: List[ObservationWindow],
) -> bool:
    if not summary:
        return False
    source = summary.get("source")
    if not isinstance(source, Mapping):
        return False
    if source.get("owner") != repo.owner or source.get("repo") != repo.repo:
        return False
    if not summary.get("repository_id") or not summary.get("collection_timestamp"):
        return False
    collected_windows = summary.get("windows")
    if not isinstance(collected_windows, Mapping):
        return False
    for window in windows:
        window_summary = collected_windows.get(window.name)
        if not isinstance(window_summary, Mapping):
            return False
        if not is_complete_window_collection(window_summary, window):
            return False
    return True


def is_complete_window_collection(
    summary: Mapping[str, Any],
    window: ObservationWindow,
) -> bool:
    if not summary:
        return False
    stored_window = summary.get("window")
    if not isinstance(stored_window, Mapping):
        return False
    required = {
        "source_pr_total_count",
        "source_sampled_pr_count",
        "source_commit_count",
        "collected_at",
    }
    return (
        stored_window.get("name") == window.name
        and stored_window.get("start") == window.start
        and stored_window.get("end") == window.end
        and all(key in summary for key in required)
    )


def discover_merged_pull_requests(
    client: GitHubClient,
    repo: SourceRepo,
    window: ObservationWindow,
    cache_dir: Path,
    per_page: int,
    max_pages: int,
    prefer_cache: bool = False,
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        params = {
            "state": "closed",
            "sort": "updated",
            "direction": "desc",
            "per_page": per_page,
            "page": page,
        }
        body = client.get_json(
            f"/repos/{repo.owner}/{repo.repo}/pulls",
            params,
            cache_dir / f"page_{page:04d}.json",
            prefer_cache=prefer_cache,
        )
        if not body:
            break
        if not isinstance(body, list):
            raise RuntimeError("Pull request list endpoint did not return a list")
        for pr in body:
            if window.contains(pr.get("merged_at")):
                merged.append(pr)
        updated_values = [item.get("updated_at") for item in body if item.get("updated_at")]
        if updated_values and max(updated_values) < window.start:
            break
        if len(body) < per_page:
            break
    return sorted(merged, key=lambda item: (str(item.get("merged_at") or ""), int(item.get("number", 0))))


def collect_commits(
    client: GitHubClient,
    repo: SourceRepo,
    window: ObservationWindow,
    cache_dir: Path,
    per_page: int,
    max_pages: int,
    prefer_cache: bool = False,
) -> List[Dict[str, Any]]:
    return client.paginate(
        f"/repos/{repo.owner}/{repo.repo}/commits",
        {"since": window.start, "until": window.end, "per_page": per_page},
        cache_dir,
        max_pages=max_pages,
        prefer_cache=prefer_cache,
    )


def write_collection_metadata(
    summaries: List[Mapping[str, Any]],
    metadata_dir: str | Path,
    sources: Mapping[str, Any],
) -> None:
    metadata_dir = Path(metadata_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    with (metadata_dir / "type5_sources.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["kind", "index", "owner", "repo", "url", "domain"])
        writer.writeheader()
        for repo in collection_sources(sources):
            writer.writerow(asdict(repo))
    provenance = {
        "api_version": sources["api_version"],
        "collection_timestamp": utc_now(),
        "source_count": len(summaries),
        "sources": list(summaries),
        "privacy_note": "No API tokens are stored. Final public benchmark files must not contain GitHub usernames.",
    }
    (metadata_dir / "type5_data_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default="benchmark_generation/configs/type5_sources.yaml")
    parser.add_argument("--cache-dir", default="type5_source_cache")
    parser.add_argument("--metadata-dir", default="benchmark_metadata")
    parser.add_argument("--config", default="benchmark_generation/configs/type5.yaml")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--primary-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    sources = load_type5_sources(args.sources)
    config = load_config(args.config, DEFAULT_COLLECT_CONFIG)
    config["api_version"] = sources["api_version"]
    token = get_github_token()
    client = GitHubClient(
        token,
        api_version=str(config.get("api_version", "2026-03-10")),
        retry_attempts=int(config.get("retry_attempts", 4)),
        backoff_initial_seconds=float(config.get("backoff_initial_seconds", 2.0)),
    )
    summaries = []
    for repo in collection_sources(sources, primary_only=args.primary_only):
        summaries.append(
            collect_repository(
                client,
                repo,
                list(sources["windows"]),
                Path(args.cache_dir),
                config,
                seed=args.seed,
            )
        )
    write_collection_metadata(summaries, args.metadata_dir, sources)


if __name__ == "__main__":
    main()
