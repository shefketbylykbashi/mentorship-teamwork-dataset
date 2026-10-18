"""Canonical solution validation and scoring for Hash Code MTP outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from benchmark_generation.core.mentorship import validate_project_assignment
from benchmark_generation.core.models import Contributor, MTPInstance, Project
from benchmark_generation.core.parser import parse_file


@dataclass
class ParsedSolution:
    projects: List[Tuple[str, List[str]]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass
class EvaluationResult:
    parse_ok: bool
    valid_solution: bool
    verified_score: int | None
    completed_projects: int
    project_scores: List[Dict[str, int | str]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_solution_file(instance_path: str | Path, solution_path: str | Path) -> EvaluationResult:
    instance = parse_file(instance_path)
    text = Path(solution_path).read_text(encoding="utf-8", errors="replace")
    return evaluate_solution(instance, text)


def evaluate_solution(instance: MTPInstance, solution_text: str) -> EvaluationResult:
    parsed = parse_solution_text(solution_text)
    if not parsed.ok:
        return EvaluationResult(
            parse_ok=False,
            valid_solution=False,
            verified_score=None,
            completed_projects=0,
            errors=parsed.errors,
        )
    return score_parsed_solution(instance, parsed)


def parse_solution_text(solution_text: str) -> ParsedSolution:
    lines = solution_text.splitlines()
    if not lines:
        return ParsedSolution(errors=["empty_output"])
    first = lines[0].strip()
    try:
        q = int(first)
    except ValueError:
        return ParsedSolution(errors=["first_line_not_integer"])
    if q < 0:
        return ParsedSolution(errors=["negative_project_count"])

    expected_lines = 1 + 2 * q
    if len(lines) < expected_lines:
        return ParsedSolution(errors=["truncated_output"])
    trailing = lines[expected_lines:]
    if any(line.strip() for line in trailing):
        return ParsedSolution(errors=["trailing_nonempty_output"])

    projects: List[Tuple[str, List[str]]] = []
    pos = 1
    for idx in range(q):
        project_name = lines[pos].strip()
        pos += 1
        contributor_names = lines[pos].strip().split()
        pos += 1
        if not project_name:
            return ParsedSolution(errors=[f"empty_project_name_at_output_index_{idx + 1}"])
        projects.append((project_name, contributor_names))
    return ParsedSolution(projects=projects)


def score_parsed_solution(instance: MTPInstance, solution: ParsedSolution) -> EvaluationResult:
    project_lookup = instance.project_by_name()
    original_contributors = instance.contributor_by_name()
    skills: Dict[str, Dict[str, int]] = {
        name: dict(contributor.skills) for name, contributor in original_contributors.items()
    }
    availability = {name: 0 for name in original_contributors}
    used_projects: set[str] = set()
    total_score = 0
    project_scores: List[Dict[str, int | str]] = []
    errors: List[str] = []

    for output_index, (project_name, contributor_names) in enumerate(solution.projects, start=1):
        project = project_lookup.get(project_name)
        if project is None:
            errors.append(f"unknown_project_{project_name}_at_output_index_{output_index}")
            continue
        if project_name in used_projects:
            errors.append(f"duplicate_project_{project_name}")
            continue
        used_projects.add(project_name)

        role_count = len(project.roles)
        if len(contributor_names) != role_count:
            errors.append(
                f"wrong_contributor_count_for_{project_name}_{len(contributor_names)}_ne_{role_count}"
            )
            continue
        if len(set(contributor_names)) != len(contributor_names):
            errors.append(f"duplicate_contributor_within_project_{project_name}")
            continue
        unknown = [name for name in contributor_names if name not in original_contributors]
        if unknown:
            errors.append(f"unknown_contributor_for_{project_name}_{unknown[0]}")
            continue

        current_lookup = _current_contributors(contributor_names, skills)
        if not validate_project_assignment(project, contributor_names, current_lookup):
            errors.append(f"infeasible_assignment_for_{project_name}")
            continue

        start = max(availability[name] for name in contributor_names) if contributor_names else 0
        finish = start + project.duration
        score = max(0, project.score - max(0, finish - project.best_before))
        total_score += score
        project_scores.append(
            {
                "project": project_name,
                "start": start,
                "finish": finish,
                "score": score,
            }
        )

        _apply_learning(project, contributor_names, skills)
        for name in contributor_names:
            availability[name] = finish

    valid = not errors
    return EvaluationResult(
        parse_ok=True,
        valid_solution=valid,
        verified_score=total_score if valid else None,
        completed_projects=len(project_scores) if valid else 0,
        project_scores=project_scores if valid else [],
        errors=errors,
    )


def _current_contributors(
    names: Sequence[str],
    skills: Dict[str, Dict[str, int]],
) -> Dict[str, Contributor]:
    return {name: Contributor(name=name, skills=dict(skills[name])) for name in names}


def _apply_learning(
    project: Project,
    contributor_names: Sequence[str],
    skills: Dict[str, Dict[str, int]],
) -> None:
    for contributor_name, role in zip(contributor_names, project.roles):
        current_level = skills[contributor_name].get(role.skill, 0)
        if current_level <= role.level:
            skills[contributor_name][role.skill] = current_level + 1

