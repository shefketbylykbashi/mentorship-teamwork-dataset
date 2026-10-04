"""Discover immutable final benchmark instances for solver campaigns."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Mapping

from benchmark_generation.core.parser import parse_file

from .common import make_relative, sha256_file


@dataclass(frozen=True)
class BenchmarkInstance:
    family: str
    instance_id: str
    path: str
    sha256: str
    contributors: int
    projects: int
    skills: int
    roles: int

    def to_manifest_row(self) -> dict:
        return asdict(self)


DEFAULT_GENERATED_FAMILIES = ("type2", "type3", "type4", "type5", "type6", "type7")
PILOT_MARKERS = ("pilot", "calibration", "candidate", "rejected", "temporary", "tmp")


def discover_final_benchmark(
    repo_root: str | Path,
    config: Mapping[str, object] | None = None,
) -> List[BenchmarkInstance]:
    """Discover final accepted Type 1-7 instance files.

    Type 1 is represented by the five non-toy Google Hash Code instances.
    Generated families are taken only from direct family directories such as
    ``generated_instances/type4``; pilot/calibration directories are ignored by
    construction and by marker checks.
    """

    repo = Path(repo_root)
    cfg = dict(config or {})
    google_dir = repo / str(cfg.get("google_dir", "google_instances"))
    generated_root = repo / str(cfg.get("generated_root", "generated_instances"))
    generated_families = tuple(cfg.get("generated_families", DEFAULT_GENERATED_FAMILIES))
    include_google_toy = bool(cfg.get("include_google_toy", False))

    rows: List[BenchmarkInstance] = []

    if google_dir.exists():
        for path in sorted(google_dir.glob("*.txt")):
            if not include_google_toy and path.name.startswith("a_"):
                continue
            rows.append(_row_from_path(path, "type1", f"type1_{path.stem}", repo))

    for family in generated_families:
        family = str(family)
        family_dir = generated_root / family
        if not family_dir.exists() or _is_excluded_dir(family_dir):
            continue
        for path in sorted(family_dir.glob("*.txt")):
            if _is_excluded_file(path):
                continue
            rows.append(_row_from_path(path, family, path.stem, repo))

    rows.sort(key=lambda row: (row.family, row.instance_id, row.path))
    return rows


def benchmark_manifest(instances: Iterable[BenchmarkInstance]) -> dict:
    rows = [instance.to_manifest_row() for instance in instances]
    family_counts: dict[str, int] = {}
    for row in rows:
        family_counts[row["family"]] = family_counts.get(row["family"], 0) + 1
    return {
        "count": len(rows),
        "family_counts": dict(sorted(family_counts.items())),
        "instances": rows,
    }


def limit_instances_per_family(
    instances: Iterable[BenchmarkInstance],
    limit: int | None,
) -> List[BenchmarkInstance]:
    """Return the first ``limit`` deterministically ordered instances per family.

    This is intended for local validation campaigns that should exercise every
    benchmark family without launching the full solver-instance grid. The input
    order is not trusted; instances are sorted exactly as the final manifest is
    sorted before selecting each family prefix.
    """

    rows = sorted(instances, key=lambda row: (row.family, row.instance_id, row.path))
    if limit is None:
        return rows
    if limit < 1:
        raise ValueError("instances_per_family must be positive when provided")
    counts: dict[str, int] = {}
    selected: List[BenchmarkInstance] = []
    for row in rows:
        count = counts.get(row.family, 0)
        if count >= limit:
            continue
        selected.append(row)
        counts[row.family] = count + 1
    return selected


def _row_from_path(path: Path, family: str, instance_id: str, repo: Path) -> BenchmarkInstance:
    instance = parse_file(path)
    return BenchmarkInstance(
        family=family,
        instance_id=instance_id,
        path=make_relative(path, repo),
        sha256=sha256_file(path),
        contributors=len(instance.contributors),
        projects=len(instance.projects),
        skills=len(instance.skill_universe),
        roles=instance.total_roles,
    )


def _is_excluded_dir(path: Path) -> bool:
    lower = path.name.lower()
    return any(marker in lower for marker in PILOT_MARKERS)


def _is_excluded_file(path: Path) -> bool:
    lower = path.name.lower()
    return any(marker in lower for marker in PILOT_MARKERS) or not lower.endswith(".txt")

