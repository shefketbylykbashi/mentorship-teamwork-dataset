from benchmark_generation.core.features import FEATURE_NAMES
from benchmark_generation.type2.generate_type2 import make_type2_targets
from benchmark_generation.type3.generate_type3 import make_type3_targets
from benchmark_generation.type4.generate_type4 import allocation_sequence, DEFAULT_CONFIG as TYPE4_CONFIG


def fake_vector(value):
    return {name: float(value) for name in FEATURE_NAMES}


def test_type2_targets_lie_between_anchor_vectors():
    anchors = [fake_vector(0.0), fake_vector(10.0), fake_vector(20.0)]
    targets = make_type2_targets(
        ["a", "b", "c"],
        anchors,
        count=6,
        group_weights={},
        seed=1,
        alpha_range=(0.2, 0.8),
    )
    assert len(targets) == 6
    for target in targets:
        a = anchors[target["anchor_a_index"]][FEATURE_NAMES[0]]
        b = anchors[target["anchor_b_index"]][FEATURE_NAMES[0]]
        value = target["target_vector"][FEATURE_NAMES[0]]
        assert min(a, b) <= value <= max(a, b)


def test_type3_targets_satisfy_configured_radius_bounds():
    config = {
        "distance_group_weights": {},
        "radius_bands": [{"name": "r", "min": 0.1, "max": 0.2}],
    }
    targets = make_type3_targets(["a"], [fake_vector(0.0)], count=5, config=config, seed=2)
    assert len(targets) == 5
    for target in targets:
        assert 0.1 <= target["desired_radius"] <= 0.2


def test_type4_default_sampling_covers_all_scale_bands():
    sequence = allocation_sequence(100, TYPE4_CONFIG)
    assert sequence.count("small") == 34
    assert sequence.count("medium") == 33
    assert sequence.count("large") == 33
