from benchmark_generation.core.features import extract_features
from benchmark_generation.core.models import Contributor, MTPInstance, Project, Role


def test_tiny_feature_values():
    instance = MTPInstance(
        contributors=(
            Contributor("c0", {"py": 2, "db": 2}),
            Contributor("c1", {"py": 3, "db": 1}),
        ),
        projects=(
            Project("p0", 2, 10, 4, (Role("py", 3), Role("db", 2))),
        ),
    )
    features = extract_features(instance)
    assert features["n_contributors"] == 2
    assert features["n_projects"] == 1
    assert features["total_roles"] == 2
    assert features["contention"] == 1
    assert features["direct_fillability"] == 1.0
    assert features["mentorship_fillability"] == 1.0
    assert features["deadline_tightness_mean"] == 2.0
    assert features["score_density"] == 2.5
