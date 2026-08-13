from benchmark_generation.core.generator import GenerationParameters, generate_instance
from benchmark_generation.core.parser import instance_to_text
from benchmark_generation.core.validator import validate_instance


def small_params():
    return GenerationParameters(
        n_skills=5,
        skill_skew=0.2,
        n_contributors=8,
        skills_per_contributor_min=2,
        skills_per_contributor_max=3,
        level_mean=3.0,
        level_sd=1.0,
        level_max=10,
        n_profiles=1,
        n_projects=6,
        duration_min=1,
        duration_max=5,
        roles_min=1,
        roles_max=3,
        score_base=1.0,
        score_perturbation=1.5,
        tightness_min=1.0,
        tightness_max=4.0,
        role_skill_alignment=0.9,
        supply_quantile=0.6,
        p_over=0.4,
        p_match=0.4,
        p_mentor=0.1,
        p_future=0.1,
        fillable_fraction=0.0,
    )


def test_same_seed_reproducible():
    params = small_params()
    a = generate_instance(params, 123)
    b = generate_instance(params, 123)
    assert instance_to_text(a) == instance_to_text(b)


def test_different_seed_changes_instance():
    params = small_params()
    a = generate_instance(params, 123)
    b = generate_instance(params, 456)
    assert instance_to_text(a) != instance_to_text(b)


def test_generated_instance_valid_syntax_and_semantics():
    params = small_params()
    instance = generate_instance(params, 789)
    result = validate_instance(instance, fillable_fraction=0.0, require_positive_project=False)
    assert result.ok
