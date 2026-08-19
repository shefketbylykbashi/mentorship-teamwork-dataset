"""Source repository and observation-window utilities for Type 5."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from benchmark_generation.core.config import load_config


@dataclass(frozen=True)
class ObservationWindow:
    name: str
    start: str
    end: str

    @property
    def start_dt(self) -> datetime:
        return parse_github_time(self.start)

    @property
    def end_dt(self) -> datetime:
        return parse_github_time(self.end)

    def contains(self, timestamp: str | None) -> bool:
        if not timestamp:
            return False
        value = parse_github_time(timestamp)
        return self.start_dt <= value < self.end_dt


@dataclass(frozen=True)
class SourceRepo:
    owner: str
    repo: str
    url: str
    domain: str
    kind: str = "primary"
    index: int = 0

    @property
    def full_name(self) -> str:
        return f"{self.owner}/{self.repo}"

    @property
    def cache_key(self) -> str:
        return f"{self.owner}__{self.repo}"


def parse_github_time(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def load_type5_sources(path: str | Path) -> Dict[str, Any]:
    data = load_config(path, {})
    windows = [ObservationWindow(**item) for item in data.get("windows", [])]
    primary = [
        SourceRepo(kind="primary", index=idx + 1, **item)
        for idx, item in enumerate(data.get("primary", []))
    ]
    reserve = [
        SourceRepo(kind="reserve", index=idx + 1, **item)
        for idx, item in enumerate(data.get("reserve", []))
    ]
    if len(primary) != 25:
        raise ValueError(f"Type 5 requires exactly 25 primary repositories; found {len(primary)}")
    if len(windows) != 4:
        raise ValueError(f"Type 5 requires exactly four fixed windows; found {len(windows)}")
    _validate_half_open_windows(windows)
    return {
        "api_version": data.get("api_version", "2026-03-10"),
        "windows": windows,
        "primary": primary,
        "reserve": reserve,
    }


def _validate_half_open_windows(windows: List[ObservationWindow]) -> None:
    previous_end = None
    for window in windows:
        if window.start_dt >= window.end_dt:
            raise ValueError(f"Invalid Type 5 window {window.name}: start must precede end")
        if previous_end is not None and window.start_dt != previous_end:
            raise ValueError("Type 5 observation windows must be contiguous half-open intervals")
        previous_end = window.end_dt


def iter_primary_slots(
    sources: Mapping[str, Any],
) -> Iterable[Tuple[int, SourceRepo, ObservationWindow, int]]:
    """Yield deterministic final-family slots.

    The mapping is repo-major then window-major, so primary repository 1 W1 is
    slot 1 and repository 25 W4 is slot 100.
    """

    windows: List[ObservationWindow] = list(sources["windows"])
    for repo_idx, repo in enumerate(sources["primary"], start=1):
        for window_idx, window in enumerate(windows, start=1):
            slot_id = (repo_idx - 1) * len(windows) + window_idx
            yield slot_id, repo, window, window_idx


def collection_sources(sources: Mapping[str, Any], primary_only: bool = False) -> List[SourceRepo]:
    repos = list(sources["primary"])
    if not primary_only:
        repos.extend(sources.get("reserve", []))
    return repos


def repo_cache_dir(cache_dir: str | Path, repo: SourceRepo) -> Path:
    return Path(cache_dir) / repo.cache_key

