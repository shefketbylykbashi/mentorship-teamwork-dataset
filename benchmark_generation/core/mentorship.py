"""Mentorship feasibility under the original Google Hash Code MTP rules."""

from __future__ import annotations

from typing import Mapping, Sequence

from .models import Contributor, Project, Role


def directly_qualified(contributor: Contributor, role: Role) -> bool:
    """Return whether ``contributor`` can fill ``role`` without mentorship."""

    return contributor.level(role.skill) >= role.level


def one_below(contributor: Contributor, role: Role) -> bool:
    """Return whether ``contributor`` is exactly one level short for ``role``."""

    return contributor.level(role.skill) == role.level - 1


def has_same_project_mentor(
    contributor: Contributor,
    role: Role,
    assigned_contributors: Sequence[Contributor],
) -> bool:
    """Return whether another assigned contributor can mentor this role's skill."""

    return any(
        other.name != contributor.name and other.level(role.skill) >= role.level
        for other in assigned_contributors
    )


def can_fill_with_project_context(
    contributor: Contributor,
    role: Role,
    assigned_contributors: Sequence[Contributor],
) -> bool:
    """Check the exact role feasibility rule for one assigned contributor.

    A role is feasible if the assigned contributor has the required level, or
    is exactly one level below and another contributor assigned to the same
    project has the required level in that skill.
    """

    if directly_qualified(contributor, role):
        return True
    return one_below(contributor, role) and has_same_project_mentor(
        contributor, role, assigned_contributors
    )


def validate_project_assignment(
    project: Project,
    assignment: Sequence[str | Contributor],
    contributor_lookup: Mapping[str, Contributor] | None = None,
) -> bool:
    """Validate a complete assignment for one project.

    ``assignment`` must be ordered like ``project.roles``. Items may be
    Contributor objects, or names when ``contributor_lookup`` is provided.
    Contributors must be distinct within a project.
    """

    if len(assignment) != len(project.roles):
        return False

    contributors = []
    for item in assignment:
        if isinstance(item, Contributor):
            contributors.append(item)
        elif contributor_lookup is not None:
            contributor = contributor_lookup.get(item)
            if contributor is None:
                return False
            contributors.append(contributor)
        else:
            return False

    names = [contributor.name for contributor in contributors]
    if len(set(names)) != len(names):
        return False

    return all(
        can_fill_with_project_context(contributor, role, contributors)
        for contributor, role in zip(contributors, project.roles)
    )
