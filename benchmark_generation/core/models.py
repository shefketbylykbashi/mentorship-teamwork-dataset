"""Core dataclasses for Google Hash Code Mentorship and Teamwork instances."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class Contributor:
    """A contributor and their current skill levels."""

    name: str
    skills: Dict[str, int]

    def level(self, skill: str) -> int:
        return self.skills.get(skill, 0)


@dataclass(frozen=True)
class Role:
    """One project role requiring exactly one skill."""

    skill: str
    level: int


@dataclass(frozen=True)
class Project:
    """A project as represented in the official Hash Code input format."""

    name: str
    duration: int
    score: int
    best_before: int
    roles: Tuple[Role, ...]


@dataclass(frozen=True)
class MTPInstance:
    """A complete MTP instance."""

    contributors: Tuple[Contributor, ...]
    projects: Tuple[Project, ...]
    source_name: str | None = None
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def skill_universe(self) -> Tuple[str, ...]:
        skills = set()
        for contributor in self.contributors:
            skills.update(contributor.skills)
        for project in self.projects:
            for role in project.roles:
                skills.add(role.skill)
        return tuple(sorted(skills))

    @property
    def total_roles(self) -> int:
        return sum(len(project.roles) for project in self.projects)

    def contributor_by_name(self) -> Dict[str, Contributor]:
        return {contributor.name: contributor for contributor in self.contributors}

    def project_by_name(self) -> Dict[str, Project]:
        return {project.name: project for project in self.projects}


def iter_roles(instance: MTPInstance) -> Iterable[Role]:
    for project in instance.projects:
        yield from project.roles


def iter_project_roles(instance: MTPInstance) -> Iterable[Tuple[Project, Role]]:
    for project in instance.projects:
        for role in project.roles:
            yield project, role
