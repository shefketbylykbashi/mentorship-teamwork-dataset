from benchmark_generation.core.models import Contributor, MTPInstance, Project, Role

from experiments.evaluator_adapter import evaluate_solution


def tiny_instance():
    return MTPInstance(
        contributors=(
            Contributor("Alice", {"py": 1}),
            Contributor("Bob", {"py": 2, "js": 2}),
        ),
        projects=(
            Project("Proj", duration=1, score=10, best_before=1, roles=(Role("py", 2),)),
            Project(
                "Mentor",
                duration=1,
                score=7,
                best_before=2,
                roles=(Role("py", 2), Role("js", 2)),
            ),
        ),
    )


def test_valid_solution_score():
    result = evaluate_solution(tiny_instance(), "1\nProj\nBob\n")
    assert result.parse_ok
    assert result.valid_solution
    assert result.verified_score == 10


def test_zero_score_valid_solution_is_retained():
    instance = MTPInstance(
        contributors=(Contributor("Bob", {"py": 2}),),
        projects=(Project("Late", duration=10, score=1, best_before=0, roles=(Role("py", 2),)),),
    )
    result = evaluate_solution(instance, "1\nLate\nBob\n")
    assert result.valid_solution
    assert result.verified_score == 0


def test_malformed_output_is_invalid_output():
    result = evaluate_solution(tiny_instance(), "not an integer\n")
    assert not result.parse_ok
    assert not result.valid_solution
    assert result.verified_score is None


def test_invalid_schedule_is_invalid_solution():
    result = evaluate_solution(tiny_instance(), "1\nProj\nAlice\n")
    assert result.parse_ok
    assert not result.valid_solution
    assert result.verified_score is None


def test_same_project_mentorship_scores_validly():
    result = evaluate_solution(tiny_instance(), "1\nMentor\nAlice Bob\n")
    assert result.valid_solution
    assert result.verified_score == 7

