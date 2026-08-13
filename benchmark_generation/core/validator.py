"""Strict independent validation and duplicate hashing for MTP instances."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from .features import build_supply_index, extract_features
from .mentorship import validate_project_assignment
from .models import Contributor, MTPInstance, Project
from .parser import ParseError, instance_to_text, parse_text


CONTRIBUTOR_RE = re.compile(r"^[A-Za-z0-9_]{1,20}$")
PROJECT_RE = re.compile(r"^[A-Za-z0-9_]{1,20}$")
SKILL_RE = re.compile(r"^[A-Za-z0-9_+.\-]{1,40}$")


@dataclass
class ValidationResult:
    ok: bool
    issues: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    def add(self, issue: str) -> None:
        self.ok = False
        self.issues.append(issue)


def validate_file(path: str | Path, fillable_fraction: float = 0.30) -> ValidationResult:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    raw_result = _validate_raw_text(text)
    try:
        instance = parse_text(text, source_name=path.name)
    except ParseError as exc:
        raw_result.add(f"parse_error: {exc}")
        return raw_result
    semantic = validate_instance(instance, fillable_fraction=fillable_fraction)
    raw_result.issues.extend(semantic.issues)
    raw_result.ok = raw_result.ok and semantic.ok
    raw_result.metrics.update(semantic.metrics)
    reparsed = instance_to_text(parse_text(instance_to_text(instance)))
    if reparsed != instance_to_text(instance):
        raw_result.add("canonical_reparse_mismatch")
    return raw_result


def _validate_raw_text(text: str) -> ValidationResult:
    result = ValidationResult(ok=True)
    try:
        text.encode("ascii")
    except UnicodeEncodeError:
        result.add("non_ascii_content")

    lines = text.splitlines()
    if not lines:
        result.add("empty_file")
        return result

    def parts_at(pos: int) -> List[str]:
        if pos >= len(lines):
            result.add("unexpected_eof")
            return []
        parts = lines[pos].split(" ")
        if "" in parts:
            result.add(f"non_single_space_or_empty_token_line_{pos + 1}")
        return [part for part in parts if part != ""]

    header = parts_at(0)
    if len(header) != 2:
        result.add("invalid_header_field_count")
        return result
    try:
        n_contributors = int(header[0])
        n_projects = int(header[1])
    except ValueError:
        result.add("invalid_header_integer")
        return result

    pos = 1
    contributor_names = set()
    for _ in range(n_contributors):
        parts = parts_at(pos)
        pos += 1
        if len(parts) != 2:
            result.add(f"invalid_contributor_header_line_{pos}")
            return result
        name = parts[0]
        if name in contributor_names:
            result.add(f"duplicate_contributor_{name}")
        contributor_names.add(name)
        try:
            n_skills = int(parts[1])
        except ValueError:
            result.add(f"invalid_skill_count_{name}")
            return result
        seen_skills = set()
        for _skill_idx in range(n_skills):
            skill_parts = parts_at(pos)
            pos += 1
            if len(skill_parts) != 2:
                result.add(f"invalid_skill_line_{pos}")
                return result
            if skill_parts[0] in seen_skills:
                result.add(f"duplicate_skill_for_contributor_{name}_{skill_parts[0]}")
            seen_skills.add(skill_parts[0])

    project_names = set()
    for _ in range(n_projects):
        parts = parts_at(pos)
        pos += 1
        if len(parts) != 5:
            result.add(f"invalid_project_header_line_{pos}")
            return result
        name = parts[0]
        if name in project_names:
            result.add(f"duplicate_project_{name}")
        project_names.add(name)
        try:
            n_roles = int(parts[4])
        except ValueError:
            result.add(f"invalid_role_count_{name}")
            return result
        for _role_idx in range(n_roles):
            role_parts = parts_at(pos)
            pos += 1
            if len(role_parts) != 2:
                result.add(f"invalid_role_line_{pos}")
                return result
    if pos != len(lines):
        result.add("trailing_content")
    return result


def validate_instance(
    instance: MTPInstance,
    fillable_fraction: float = 0.30,
    require_positive_project: bool = True,
) -> ValidationResult:
    result = ValidationResult(ok=True)
    if len(instance.contributors) < 1:
        result.add("no_contributors")
    if len(instance.projects) < 1:
        result.add("no_projects")

    contributor_names = set()
    for contributor in instance.contributors:
        if not CONTRIBUTOR_RE.match(contributor.name):
            result.add(f"invalid_contributor_name_{contributor.name}")
        if contributor.name in contributor_names:
            result.add(f"duplicate_contributor_{contributor.name}")
        contributor_names.add(contributor.name)
        if not contributor.skills:
            result.add(f"contributor_without_skills_{contributor.name}")
        for skill, level in contributor.skills.items():
            if not SKILL_RE.match(skill):
                result.add(f"invalid_skill_name_{skill}")
            if not isinstance(level, int) or level < 1 or level > 10:
                result.add(f"invalid_contributor_skill_level_{contributor.name}_{skill}_{level}")

    project_names = set()
    for project in instance.projects:
        if not PROJECT_RE.match(project.name):
            result.add(f"invalid_project_name_{project.name}")
        if project.name in project_names:
            result.add(f"duplicate_project_{project.name}")
        project_names.add(project.name)
        if project.duration < 1:
            result.add(f"invalid_duration_{project.name}")
        if project.score < 1:
            result.add(f"invalid_score_{project.name}")
        if project.best_before < 0:
            result.add(f"invalid_best_before_{project.name}")
        if len(project.roles) < 1:
            result.add(f"project_without_roles_{project.name}")
        for role in project.roles:
            if not SKILL_RE.match(role.skill):
                result.add(f"invalid_role_skill_{role.skill}")
            if not isinstance(role.level, int) or role.level < 1 or role.level > 100:
                result.add(f"invalid_role_level_{project.name}_{role.skill}_{role.level}")

    features = extract_features(instance)
    role_fillability = max(features["direct_fillability"], features["mentorship_fillability"])
    result.metrics["initial_fillable_fraction"] = role_fillability
    if role_fillability + 1e-12 < fillable_fraction:
        result.add(
            f"fillable_fraction_below_threshold_{role_fillability:.6f}_lt_{fillable_fraction:.6f}"
        )

    if require_positive_project and not has_positive_directly_completable_project(instance):
        result.add("no_positive_directly_completable_project")

    return result


def has_positive_directly_completable_project(instance: MTPInstance) -> bool:
    """Return whether at least one project has a direct feasible assignment.

    This is a conservative viability check: direct qualification alone is enough
    to demonstrate a positive attainable score. Mentorship feasibility is still
    validated separately by the dedicated mentorship functions and features.
    """

    contributors = list(instance.contributors)
    for project in instance.projects:
        if project.score <= 0:
            continue
        if project.duration > project.best_before + project.score:
            continue
        if _project_direct_matching_exists(project, contributors):
            return True
    return False


def _project_direct_matching_exists(project: Project, contributors: Sequence[Contributor]) -> bool:
    role_candidates: List[List[int]] = []
    for role in project.roles:
        candidates = [
            idx
            for idx, contributor in enumerate(contributors)
            if contributor.level(role.skill) >= role.level
        ]
        if not candidates:
            return False
        role_candidates.append(candidates)

    order = sorted(range(len(project.roles)), key=lambda idx: len(role_candidates[idx]))
    matched_to_role: Dict[int, int] = {}

    def try_match(role_idx: int, seen: set[int]) -> bool:
        for contributor_idx in role_candidates[role_idx]:
            if contributor_idx in seen:
                continue
            seen.add(contributor_idx)
            previous_role = matched_to_role.get(contributor_idx)
            if previous_role is None or try_match(previous_role, seen):
                matched_to_role[contributor_idx] = role_idx
                return True
        return False

    for role_idx in order:
        if not try_match(role_idx, set()):
            return False
    return True


def initial_role_fillability(instance: MTPInstance) -> float:
    features = extract_features(instance)
    return max(features["direct_fillability"], features["mentorship_fillability"])


def canonical_structure(instance: MTPInstance) -> Dict[str, object]:
    contributors = [
        {"name": c.name, "skills": dict(sorted(c.skills.items()))}
        for c in sorted(instance.contributors, key=lambda item: item.name)
    ]
    projects = []
    for project in sorted(instance.projects, key=lambda item: item.name):
        roles = sorted((role.skill, role.level) for role in project.roles)
        projects.append(
            {
                "name": project.name,
                "duration": project.duration,
                "score": project.score,
                "best_before": project.best_before,
                "roles": roles,
            }
        )
    return {"contributors": contributors, "projects": projects}


def structural_hash(instance: MTPInstance) -> str:
    payload = json.dumps(canonical_structure(instance), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assignment_uses_original_mentorship_rules(
    project: Project,
    assignment: Sequence[str],
    contributors: Mapping[str, Contributor],
) -> bool:
    return validate_project_assignment(project, assignment, contributors)
