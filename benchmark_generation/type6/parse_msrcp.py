"""Parser for official MSLIB MSRCPSP ``.msrcp`` files.

The implementation follows the sectioned v2.2 release files and the included
``MSRCPSP_DataFormat`` document. It parses the source MSRCPSP structure only;
it does not reinterpret precedence constraints as MTP constraints.
"""

from __future__ import annotations

import hashlib
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


class MSRCPSPParseError(ValueError):
    """Raised when an MSLIB ``.msrcp`` file violates the documented format."""


@dataclass(frozen=True)
class MSRCPSPActivity:
    activity_id: int
    duration: int
    successors: Tuple[int, ...]


@dataclass(frozen=True)
class ActivityRoleRequirement:
    activity_id: int
    skill_id: int
    required_level: int


@dataclass(frozen=True)
class MSRCPSPInstance:
    dataset: str
    source_filename: str
    source_instance_id: int
    source_zip_entry: str
    source_sha256: str
    n_activities_total: int
    n_resources: int
    n_skills: int
    n_skill_levels: int
    bks_basic: int
    deadline_skill_levels: int
    activities: Tuple[MSRCPSPActivity, ...]
    workforce_binary: Tuple[Tuple[int, ...], ...]
    workforce_levels: Tuple[Tuple[int, ...], ...]
    skill_requirements: Tuple[Tuple[int, ...], ...]
    skill_level_requirement_rows: Tuple[Tuple[int, ...], ...]
    role_requirements: Tuple[ActivityRoleRequirement, ...]
    optional_modules: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)

    @property
    def dummy_start_id(self) -> int:
        return 1

    @property
    def dummy_end_id(self) -> int:
        return self.n_activities_total

    @property
    def real_activity_ids(self) -> Tuple[int, ...]:
        return tuple(range(2, self.n_activities_total))

    @property
    def n_real_activities(self) -> int:
        return max(0, self.n_activities_total - 2)

    @property
    def precedence_arcs(self) -> Tuple[Tuple[int, int], ...]:
        return tuple(
            (activity.activity_id, successor)
            for activity in self.activities
            for successor in activity.successors
        )


def parse_msrcp_text(
    text: str,
    dataset: str = "",
    source_filename: str = "",
    source_zip_entry: str = "",
) -> MSRCPSPInstance:
    source_bytes = text.encode("utf-8")
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    sections = _split_sections(text)
    required_sections = {
        "project module",
        "workforce module",
        "workforce module with skill levels",
        "skill requirements module",
        "skill level requirements module",
    }
    missing = sorted(required_sections - set(sections))
    if missing:
        raise MSRCPSPParseError(f"Missing required MSRCPSP sections: {missing}")

    project = _project_module(sections["project module"])
    n_activities, n_resources, n_skills, n_skill_levels = project["header"]
    activities = project["activities"]
    workforce_binary = _fixed_int_matrix(
        sections["workforce module"], n_resources, n_skills, "Workforce Module"
    )
    workforce_levels = _fixed_int_matrix(
        sections["workforce module with skill levels"],
        n_resources,
        n_skills,
        "Workforce Module with Skill Levels",
    )
    skill_requirements = _fixed_int_matrix(
        sections["skill requirements module"],
        n_activities,
        n_skills,
        "Skill Requirements Module",
    )
    level_rows = _skill_level_rows(
        sections["skill level requirements module"], n_activities
    )
    role_requirements = _role_requirements(skill_requirements, level_rows)
    source_instance_id = source_id_from_filename(source_filename)
    instance = MSRCPSPInstance(
        dataset=dataset,
        source_filename=source_filename,
        source_instance_id=source_instance_id,
        source_zip_entry=source_zip_entry,
        source_sha256=source_sha256,
        n_activities_total=n_activities,
        n_resources=n_resources,
        n_skills=n_skills,
        n_skill_levels=n_skill_levels,
        bks_basic=project["bks_basic"],
        deadline_skill_levels=project["deadline_skill_levels"],
        activities=tuple(activities),
        workforce_binary=tuple(tuple(row) for row in workforce_binary),
        workforce_levels=tuple(tuple(row) for row in workforce_levels),
        skill_requirements=tuple(tuple(row) for row in skill_requirements),
        skill_level_requirement_rows=tuple(tuple(row) for row in level_rows),
        role_requirements=tuple(role_requirements),
        optional_modules={
            name: tuple(lines)
            for name, lines in sections.items()
            if name not in required_sections
        },
    )
    validate_msrcp_instance(instance)
    return instance


def parse_msrcp_file(path: str | Path, dataset: str = "") -> MSRCPSPInstance:
    path = Path(path)
    return parse_msrcp_text(
        path.read_text(encoding="utf-8"),
        dataset=dataset or dataset_from_filename(path.name),
        source_filename=path.name,
        source_zip_entry=str(path),
    )


def parse_msrcp_zip_entry(
    zip_path: str | Path,
    entry_name: str,
    dataset: str,
) -> MSRCPSPInstance:
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        payload = zf.read(entry_name)
    return parse_msrcp_text(
        payload.decode("utf-8", errors="strict"),
        dataset=dataset,
        source_filename=Path(entry_name).name,
        source_zip_entry=entry_name,
    )


def validate_msrcp_instance(instance: MSRCPSPInstance) -> None:
    if instance.n_activities_total != len(instance.activities):
        raise MSRCPSPParseError("Project activity count does not match parsed activities")
    if instance.n_activities_total < 2:
        raise MSRCPSPParseError("MSRCPSP instance must contain dummy start and end activities")
    ids = [activity.activity_id for activity in instance.activities]
    if ids != list(range(1, instance.n_activities_total + 1)):
        raise MSRCPSPParseError("Activities must be contiguous 1-based identifiers")
    for activity in instance.activities:
        for successor in activity.successors:
            if successor < 1 or successor > instance.n_activities_total:
                raise MSRCPSPParseError(f"Successor id out of range: {successor}")
            if successor == activity.activity_id:
                raise MSRCPSPParseError(f"Self-loop at activity {activity.activity_id}")
    start = instance.activities[0]
    end = instance.activities[-1]
    if start.duration != 0 or end.duration != 0:
        raise MSRCPSPParseError("Dummy start/end activities must have zero duration")
    if end.successors:
        raise MSRCPSPParseError("Dummy end activity must have no successors")
    if any(
        any(instance.skill_requirements[idx - 1])
        for idx in (1, instance.n_activities_total)
    ):
        raise MSRCPSPParseError("Dummy start/end activities must not require skills")
    if instance.skill_level_requirement_rows[0] != (-1,):
        raise MSRCPSPParseError("Dummy start activity must have -1 skill-level row")
    if instance.skill_level_requirement_rows[-1] != (-1,):
        raise MSRCPSPParseError("Dummy end activity must have -1 skill-level row")
    for resource_idx, (binary_row, level_row) in enumerate(
        zip(instance.workforce_binary, instance.workforce_levels), start=1
    ):
        for skill_idx, (mastered, level) in enumerate(zip(binary_row, level_row), start=1):
            if mastered not in (0, 1):
                raise MSRCPSPParseError(
                    f"Invalid binary mastery at resource {resource_idx}, skill {skill_idx}"
                )
            if level < 0 or level > instance.n_skill_levels:
                raise MSRCPSPParseError(
                    f"Invalid skill level at resource {resource_idx}, skill {skill_idx}: {level}"
                )
            if bool(mastered) != bool(level):
                raise MSRCPSPParseError(
                    f"Binary and level mastery disagree at resource {resource_idx}, skill {skill_idx}"
                )
    topological_order(instance)


def topological_order(instance: MSRCPSPInstance) -> Tuple[int, ...]:
    indegree = {activity.activity_id: 0 for activity in instance.activities}
    successors = {activity.activity_id: activity.successors for activity in instance.activities}
    for activity in instance.activities:
        for successor in activity.successors:
            indegree[successor] += 1
    ready = sorted(activity_id for activity_id, value in indegree.items() if value == 0)
    order: List[int] = []
    while ready:
        activity_id = ready.pop(0)
        order.append(activity_id)
        for successor in successors[activity_id]:
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
                ready.sort()
    if len(order) != instance.n_activities_total:
        raise MSRCPSPParseError("Precedence graph is not acyclic")
    return tuple(order)


def dataset_from_filename(filename: str) -> str:
    match = re.search(r"MSLIB_Set(\d+)_", filename, re.IGNORECASE)
    return f"MSLIB{match.group(1)}" if match else ""


def source_id_from_filename(filename: str) -> int:
    match = re.search(r"MSLIB_Set\d+_(\d+)\.msrcp$", filename, re.IGNORECASE)
    if not match:
        return 0
    return int(match.group(1))


def _split_sections(text: str) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {}
    current = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("\\*") and line.endswith("*\\"):
            current = _normalize_section_name(line)
            sections[current] = []
            continue
        if not current:
            raise MSRCPSPParseError("Content appears before first section marker")
        sections[current].append(line)
    return sections


def _normalize_section_name(line: str) -> str:
    return (
        line.replace("\\", "")
        .replace("*", "")
        .strip()
        .lower()
    )


def _project_module(lines: Sequence[str]) -> Dict[str, object]:
    if len(lines) < 4:
        raise MSRCPSPParseError("Project Module is incomplete")
    header = _ints(lines[0])
    if len(header) != 4:
        raise MSRCPSPParseError("Project Module header must contain four integers")
    n_activities = header[0]
    bks_basic = _single_int(lines[1], "Project Module BKS line")
    deadline_skill_levels = _single_int(lines[2], "Project Module deadline line")
    activity_lines = lines[3 : 3 + n_activities]
    if len(activity_lines) != n_activities:
        raise MSRCPSPParseError("Project Module has fewer activity rows than declared")
    activities = []
    for index, line in enumerate(activity_lines, start=1):
        values = _ints(line)
        if len(values) < 2:
            raise MSRCPSPParseError(f"Activity row {index} must contain duration and successor count")
        duration, successor_count = values[0], values[1]
        successors = values[2:]
        if len(successors) != successor_count:
            raise MSRCPSPParseError(
                f"Activity {index} declares {successor_count} successors but has {len(successors)}"
            )
        activities.append(MSRCPSPActivity(index, duration, tuple(successors)))
    return {
        "header": tuple(header),
        "bks_basic": bks_basic,
        "deadline_skill_levels": deadline_skill_levels,
        "activities": activities,
    }


def _fixed_int_matrix(
    lines: Sequence[str],
    expected_rows: int,
    expected_cols: int,
    section_name: str,
) -> List[List[int]]:
    rows = [_ints(line) for line in lines[:expected_rows]]
    if len(rows) != expected_rows:
        raise MSRCPSPParseError(
            f"{section_name} has {len(rows)} rows; expected {expected_rows}"
        )
    for row_idx, row in enumerate(rows, start=1):
        if len(row) != expected_cols:
            raise MSRCPSPParseError(
                f"{section_name} row {row_idx} has {len(row)} columns; expected {expected_cols}"
            )
    return rows


def _skill_level_rows(lines: Sequence[str], expected_rows: int) -> List[Tuple[int, ...]]:
    if len(lines) < expected_rows:
        raise MSRCPSPParseError(
            f"Skill Level Requirements Module has {len(lines)} rows; expected {expected_rows}"
        )
    return [tuple(_ints(line)) for line in lines[:expected_rows]]


def _role_requirements(
    skill_requirements: Sequence[Sequence[int]],
    level_rows: Sequence[Sequence[int]],
) -> List[ActivityRoleRequirement]:
    roles: List[ActivityRoleRequirement] = []
    for activity_id, (demand_row, level_row) in enumerate(
        zip(skill_requirements, level_rows), start=1
    ):
        expected = sum(int(value) for value in demand_row)
        if expected == 0:
            if tuple(level_row) not in {tuple(), (-1,)}:
                raise MSRCPSPParseError(
                    f"Activity {activity_id} has no skill demand but nonempty level row"
                )
            continue
        if len(level_row) != expected:
            raise MSRCPSPParseError(
                f"Activity {activity_id} has {expected} skill requirements but "
                f"{len(level_row)} required-level entries"
            )
        level_idx = 0
        for skill_idx, multiplicity in enumerate(demand_row, start=1):
            if int(multiplicity) < 0:
                raise MSRCPSPParseError(
                    f"Activity {activity_id} has negative skill multiplicity"
                )
            for _ in range(int(multiplicity)):
                level = int(level_row[level_idx])
                if level <= 0:
                    raise MSRCPSPParseError(
                        f"Activity {activity_id} has nonpositive required skill level"
                    )
                roles.append(ActivityRoleRequirement(activity_id, skill_idx, level))
                level_idx += 1
    return roles


def _ints(line: str) -> List[int]:
    try:
        return [int(value) for value in line.split()]
    except ValueError as exc:
        raise MSRCPSPParseError(f"Expected integer line, got: {line}") from exc


def _single_int(line: str, label: str) -> int:
    values = _ints(line)
    if len(values) != 1:
        raise MSRCPSPParseError(f"{label} must contain exactly one integer")
    return values[0]


def iter_real_activity_roles(
    instance: MSRCPSPInstance,
) -> Iterable[Tuple[int, Tuple[ActivityRoleRequirement, ...]]]:
    by_activity: Dict[int, List[ActivityRoleRequirement]] = {
        activity_id: [] for activity_id in instance.real_activity_ids
    }
    for role in instance.role_requirements:
        if role.activity_id in by_activity:
            by_activity[role.activity_id].append(role)
    for activity_id in instance.real_activity_ids:
        yield activity_id, tuple(by_activity[activity_id])
