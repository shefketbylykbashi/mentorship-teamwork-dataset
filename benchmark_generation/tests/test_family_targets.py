import math

import pytest

from benchmark_generation.core.generator import GenerationParameters
from benchmark_generation.core.distance import FeatureScaler, feature_weights, weighted_distance
from benchmark_generation.core.features import FEATURE_NAMES
from benchmark_generation.core.search import radius_in_bounds
from benchmark_generation.type2.generate_type2 import make_type2_targets
from benchmark_generation.type3.generate_type3 import (
    _type3_initial_population,
    make_type3_targets,
    random_weighted_direction,
    resolve_anchor_radius_bands,
)
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
        "allow_uncalibrated_radius_bands": True,
    }
    targets = make_type3_targets(["a"], [fake_vector(0.0)], count=5, config=config, seed=2)
    assert len(targets) == 5
    for target in targets:
        assert 0.1 <= target["desired_radius"] <= 0.2
        actual = weighted_distance(
            fake_vector(0.0),
            target["target_vector"],
            feature_weights(config["distance_group_weights"]),
        )
        assert math.isclose(actual, target["desired_radius"], rel_tol=1e-12, abs_tol=1e-12)


def test_type3_weighted_direction_has_unit_norm():
    weights = feature_weights({})
    import random

    direction = random_weighted_direction(random.Random(7), weights)
    norm = math.sqrt(sum(weights[name] * direction[name] ** 2 for name in FEATURE_NAMES))
    assert math.isclose(norm, 1.0, rel_tol=1e-12, abs_tol=1e-12)


def test_type3_target_generation_is_deterministic():
    config = {
        "distance_group_weights": {},
        "radius_bands": [{"name": "r", "min": 0.1, "max": 0.2}],
        "allow_uncalibrated_radius_bands": True,
    }
    a = make_type3_targets(["a"], [fake_vector(0.0)], count=5, config=config, seed=2026)
    b = make_type3_targets(["a"], [fake_vector(0.0)], count=5, config=config, seed=2026)
    assert a == b


def test_type3_regression_out_of_band_radius_is_rejected():
    assert not radius_in_bounds(1.0030, 0.03, 0.08)
    assert not radius_in_bounds(1.6416, 0.03, 0.08)
    assert radius_in_bounds(0.0700000001, 0.03, 0.08)


def test_feature_scaler_handles_constant_and_nonfinite_features():
    constant = {name: 1.0 for name in FEATURE_NAMES}
    constant[FEATURE_NAMES[0]] = float("nan")
    scaler = FeatureScaler.fit([constant, constant])
    normalized = scaler.normalize(constant)
    assert all(math.isfinite(value) for value in normalized.values())


def test_type3_requires_calibrated_radius_file_by_default(tmp_path):
    config = {
        "radius_bands_path": str(tmp_path / "missing.json"),
        "radius_bands": [{"name": "dev", "min": 0.1, "max": 0.2}],
    }
    with pytest.raises(FileNotFoundError):
        resolve_anchor_radius_bands(["a"], config)


def test_type3_development_uncalibrated_radius_opt_in(tmp_path):
    config = {
        "radius_bands_path": str(tmp_path / "missing.json"),
        "radius_bands": [{"name": "dev", "min": 0.1, "max": 0.2}],
        "allow_uncalibrated_radius_bands": True,
    }
    resolved = resolve_anchor_radius_bands(["a"], config)
    assert resolved["a"][0]["name"] == "dev"


def test_type3_rejects_incomplete_calibrated_radius_file(tmp_path):
    path = tmp_path / "calibrated.json"
    path.write_text('{"anchors": {"a": {"bands": [{"name": "inner", "min": 1, "max": 2}]}}}', encoding="utf-8")
    with pytest.raises(ValueError):
        resolve_anchor_radius_bands(["a", "b"], {"radius_bands_path": str(path)})


def test_type3_initial_population_uses_multiple_local_scales():
    params = GenerationParameters(
        n_skills=20,
        skill_skew=0.5,
        n_contributors=100,
        skills_per_contributor_min=2,
        skills_per_contributor_max=5,
        level_mean=4.0,
        level_sd=1.0,
        level_max=10,
        n_profiles=1,
        n_projects=100,
        duration_min=1,
        duration_max=20,
        roles_min=1,
        roles_max=5,
        score_base=1.0,
        score_perturbation=1.5,
        tightness_min=1.0,
        tightness_max=4.0,
        role_skill_alignment=0.7,
        supply_quantile=0.7,
        p_over=0.2,
        p_match=0.3,
        p_mentor=0.2,
        p_future=0.3,
    ).normalized()
    bounds = {key: (float(value), float(value)) for key, value in params.to_dict().items()}
    bounds.update(
        {
            "skill_skew": (0.0, 1.0),
            "level_mean": (2.0, 7.0),
            "score_base": (0.5, 2.0),
            "tightness_min": (0.5, 2.0),
            "tightness_max": (2.0, 6.0),
            "p_over": (0.0, 1.0),
            "p_match": (0.0, 1.0),
            "p_mentor": (0.0, 1.0),
            "p_future": (0.0, 1.0),
        }
    )
    population = _type3_initial_population(
        params,
        bounds,
        {
            "centered_initial_samples": 6,
            "initial_perturbation_scales": [0.0, 0.02, 0.05, 0.10, 0.20],
            "viability_difficulty_prior": [0.35, 0.35, 0.20, 0.10],
            "viability_blends": [1.0, 0.75, 0.50, 0.25, 0.0],
        },
        seed=2026,
    )
    assert len(population) == 6
    assert population[0].to_dict() == params.to_dict()
    unique_parameter_sets = {
        tuple(sorted(member.to_dict().items()))
        for member in population
    }
    assert len(unique_parameter_sets) >= 4


def test_type4_default_sampling_covers_all_scale_bands():
    sequence = allocation_sequence(100, TYPE4_CONFIG)
    assert sequence.count("small") == 34
    assert sequence.count("medium") == 33
    assert sequence.count("large") == 33
