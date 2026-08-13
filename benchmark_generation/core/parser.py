"""Parser and writer for the official Google Hash Code MTP input format."""

from __future__ import annotations

from pathlib import Path
from typing import List

from .models import Contributor, MTPInstance, Project, Role


class ParseError(ValueError):
    """Raised when an MTP input file is syntactically malformed."""


def _split_line(lines: List[str], index: int) -> List[str]:
    if index >= len(lines):
        raise ParseError("Unexpected end of file")
    parts = lines[index].strip().split()
    if not parts:
        raise ParseError(f"Empty line at {index + 1}")
    return parts


def parse_text(text: str, source_name: str | None = None) -> MTPInstance:
    """Parse a Hash Code MTP instance from text.

    The parser intentionally performs only syntax/count checks. Semantic and
    research viability checks live in ``validator.py`` so generated files are
    revalidated independently of the generator.
    """

    lines = text.splitlines()
    if not lines:
        raise ParseError("Empty file")
    header = _split_line(lines, 0)
    if len(header) != 2:
        raise ParseError("Header must contain contributor and project counts")
    try:
        n_contributors = int(header[0])
        n_projects = int(header[1])
    except ValueError as exc:
        raise ParseError("Header counts must be integers") from exc

    pos = 1
    contributors = []
    for _ in range(n_contributors):
        parts = _split_line(lines, pos)
        pos += 1
        if len(parts) != 2:
            raise ParseError(f"Contributor header at line {pos} must have 2 fields")
        name = parts[0]
        try:
            n_skills = int(parts[1])
        except ValueError as exc:
            raise ParseError(f"Invalid skill count for contributor {name}") from exc
        skills = {}
        for _skill_idx in range(n_skills):
            skill_parts = _split_line(lines, pos)
            pos += 1
            if len(skill_parts) != 2:
                raise ParseError(f"Skill line at line {pos} must have 2 fields")
            try:
                skills[skill_parts[0]] = int(skill_parts[1])
            except ValueError as exc:
                raise ParseError(f"Invalid skill level at line {pos}") from exc
        contributors.append(Contributor(name=name, skills=skills))

    projects = []
    for _ in range(n_projects):
        parts = _split_line(lines, pos)
        pos += 1
        if len(parts) != 5:
            raise ParseError(f"Project header at line {pos} must have 5 fields")
        name = parts[0]
        try:
            duration = int(parts[1])
            score = int(parts[2])
            best_before = int(parts[3])
            n_roles = int(parts[4])
        except ValueError as exc:
            raise ParseError(f"Invalid numeric field for project {name}") from exc
        roles = []
        for _role_idx in range(n_roles):
            role_parts = _split_line(lines, pos)
            pos += 1
            if len(role_parts) != 2:
                raise ParseError(f"Role line at line {pos} must have 2 fields")
            try:
                roles.append(Role(skill=role_parts[0], level=int(role_parts[1])))
            except ValueError as exc:
                raise ParseError(f"Invalid role level at line {pos}") from exc
        projects.append(
            Project(
                name=name,
                duration=duration,
                score=score,
                best_before=best_before,
                roles=tuple(roles),
            )
        )

    if pos != len(lines):
        raise ParseError(f"Trailing content starts at line {pos + 1}")

    return MTPInstance(
        contributors=tuple(contributors),
        projects=tuple(projects),
        source_name=source_name,
    )


def parse_file(path: str | Path) -> MTPInstance:
    path = Path(path)
    return parse_text(path.read_text(encoding="utf-8"), source_name=path.name)


def instance_to_text(instance: MTPInstance) -> str:
    """Serialize an instance to canonical Hash Code text with LF endings."""

    lines = [f"{len(instance.contributors)} {len(instance.projects)}"]
    for contributor in instance.contributors:
        skill_items = list(contributor.skills.items())
        lines.append(f"{contributor.name} {len(skill_items)}")
        for skill, level in skill_items:
            lines.append(f"{skill} {level}")
    for project in instance.projects:
        lines.append(
            f"{project.name} {project.duration} {project.score} "
            f"{project.best_before} {len(project.roles)}"
        )
        for role in project.roles:
            lines.append(f"{role.skill} {role.level}")
    return "\n".join(lines) + "\n"


def write_instance(instance: MTPInstance, path: str | Path, overwrite: bool = False) -> None:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing instance: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(instance_to_text(instance), encoding="utf-8", newline="\n")
