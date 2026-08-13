from benchmark_generation.core.mentorship import validate_project_assignment
from benchmark_generation.core.models import Contributor, Project, Role


def contributors():
    return {
        "direct": Contributor("direct", {"py": 3, "js": 2}),
        "mentor": Contributor("mentor", {"py": 3, "js": 3}),
        "below": Contributor("below", {"py": 2}),
        "two_below": Contributor("two_below", {"py": 1}),
        "other": Contributor("other", {"db": 3}),
        "learner": Contributor("learner", {"js": 2, "py": 3}),
    }


def test_direct_qualification():
    c = contributors()
    project = Project("p", 1, 1, 1, (Role("py", 3),))
    assert validate_project_assignment(project, ["direct"], c)


def test_valid_one_level_below_mentorship():
    c = contributors()
    project = Project("p", 1, 1, 1, (Role("py", 3), Role("db", 3)))
    assert validate_project_assignment(project, ["below", "other"], c | {"other": Contributor("other", {"db": 3, "py": 3})})


def test_invalid_two_levels_below_assignment():
    c = contributors()
    project = Project("p", 1, 1, 1, (Role("py", 3), Role("db", 3)))
    assert not validate_project_assignment(project, ["two_below", "other"], c | {"other": Contributor("other", {"db": 3, "py": 3})})


def test_missing_mentor():
    c = contributors()
    project = Project("p", 1, 1, 1, (Role("py", 3), Role("db", 3)))
    assert not validate_project_assignment(project, ["below", "other"], c)


def test_mentor_must_be_on_same_project():
    c = contributors()
    project = Project("p", 1, 1, 1, (Role("py", 3),))
    assert not validate_project_assignment(project, ["below"], c)


def test_mentor_not_assigned_to_same_project_is_invalid():
    c = contributors()
    project = Project("p", 1, 1, 1, (Role("py", 3), Role("db", 3)))
    assert not validate_project_assignment(project, ["below", "other"], c)


def test_one_contributor_mentoring_multiple_roles():
    c = {
        "mentor": Contributor("mentor", {"py": 3, "db": 3, "js": 3}),
        "below_py": Contributor("below_py", {"py": 2}),
        "below_js": Contributor("below_js", {"js": 2}),
    }
    project = Project("p", 1, 1, 1, (Role("py", 3), Role("js", 3), Role("db", 3)))
    assert validate_project_assignment(project, ["below_py", "below_js", "mentor"], c)


def test_contributor_both_mentoring_and_being_mentored():
    c = {
        "a": Contributor("a", {"py": 2, "js": 3}),
        "b": Contributor("b", {"py": 3, "js": 2}),
    }
    project = Project("p", 1, 1, 1, (Role("py", 3), Role("js", 3)))
    assert validate_project_assignment(project, ["a", "b"], c)
