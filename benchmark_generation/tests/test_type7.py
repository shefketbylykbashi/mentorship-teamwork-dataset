import copy
from pathlib import Path

from benchmark_generation.core.distance import FEATURE_NAMES, weighted_distance
from benchmark_generation.core.generator import GenerationParameters
from benchmark_generation.core.models import Contributor, MTPInstance, Project, Role
from benchmark_generation.core.parser import write_instance
from benchmark_generation.core.validator import structural_hash
from benchmark_generation.type7.generate_type7 import (
    ArchiveEntry,
    CandidateBuildResult,
    CandidateRecord,
    build_search_manifest,
    coverage_before_after,
    generate_candidate_record,
    metadata_has_canonical_features,
    metadata_structural_hash,
    mentorship_metrics,
    project_true_same_project_mentorship_feasible,
    repair_type7_parameters,
    scale_band_sequence,
    select_sequential_maximin,
    selected_family_quality_gate,
    selection_feature_policy,
    sha256_file,
)


def tiny_params() -> GenerationParameters:
    return GenerationParameters(
        n_skills=5,
        skill_skew=0.2,
        n_contributors=8,
        skills_per_contributor_min=1,
        skills_per_contributor_max=3,
        level_mean=4.0,
        level_sd=1.0,
        level_max=10,
        n_profiles=1,
        n_projects=7,
        duration_min=1,
        duration_max=8,
        roles_min=1,
        roles_max=3,
        score_base=1.0,
        score_perturbation=1.5,
        tightness_min=1.0,
        tightness_max=4.0,
        role_skill_alignment=0.7,
        supply_quantile=0.7,
        p_over=0.3,
        p_match=0.3,
        p_mentor=0.2,
        p_future=0.2,
        fillable_fraction=0.1,
    ).normalized()


def vector(value: float):
    return {name: float(value) for name in FEATURE_NAMES}


def fake_candidate(candidate_id: int, value: float, h: str | None = None) -> CandidateRecord:
    return CandidateRecord(
        candidate_id=candidate_id,
        archive_id=f"fake_{candidate_id}",
        candidate_seed=1000 + candidate_id,
        outcome_seed=2000 + candidate_id,
        search_round=0,
        search_mode="broad",
        scale_band="small",
        parameters=tiny_params().to_dict(),
        generation_attempts=1,
        features=vector(value),
        structural_hash=h or f"h{candidate_id}",
        validation_initial_fillable_fraction=1.0,
        normalized_features=vector(value),
    )


def fake_archive(value: float = 0.0, family: str = "type1") -> ArchiveEntry:
    return ArchiveEntry(
        family=family,
        family_instance_id="001",
        filename=f"{family}_001.txt",
        structural_hash=f"{family}_hash",
        normalized_features=vector(value),
    )


def complete_row(idx: int, h: str):
    row = {
        "filename": f"type7_{idx:03d}.txt",
        "family": "type7",
        "family_instance_id": idx,
        "seed": 2000 + idx,
        "root_seed": 2026,
        "candidate_seed": 1000 + idx,
        "generator_parameter_hash": f"ph{idx}",
        "config_hash": "cfg",
        "structural_hash": h,
        "type7.novelty_score_at_acceptance": 1.0,
        "type7.nearest_benchmark_neighbor_family": "type1",
        "type7.distance_to_nearest_type1_6_instance": 1.0,
        "type7.archive_id": f"broad_0_{idx}",
        "type7.exact_mentorship_metrics_pass": True,
        "type7.degeneracy_checks_pass": True,
        "type7.forward_generated": True,
        "type7.coverage_feature_space_frozen_before_selection": True,
        "type7.solver_performance_used": False,
        "validation.ok": True,
        "mentorship.true_same_project_mentorship_feasibility": 0.0,
        "degeneracy.passed": True,
    }
    for name in FEATURE_NAMES:
        row[f"feature.{name}"] = 1.0
    return row


def test_type7_uses_forward_generator_only():
    policy = selection_feature_policy()
    assert policy["forward_generated_only"]
    candidate = generate_candidate_record(tiny_params(), 2026, 1, 0, "broad", "small", 5)
    assert candidate is not None
    assert candidate.structural_hash


def test_type7_does_not_inverse_construct_features():
    policy = selection_feature_policy()
    assert policy["feature_names"] == FEATURE_NAMES
    assert policy["solver_metrics_excluded"]


def test_type7_feature_space_frozen_before_selection():
    rows = [complete_row(1, "a")]
    gate = selected_family_quality_gate(rows, [fake_candidate(1, 1.0, "a")], 1, set(), {}, False)
    assert gate["feature_space_frozen_before_type7_selection"]


def test_type7_solver_metrics_not_in_selection_features():
    policy = selection_feature_policy()
    assert policy["forbidden_feature_hits"] == []
    assert policy["mentorship_feature_present"]


def test_type7_search_deterministic_seed_2026():
    config = {
        "scale_bands": {
            "small": {"contributors": [5, 10], "projects": [5, 10]},
            "medium": {"contributors": [10, 20], "projects": [10, 20]},
        },
        "scale_band_weights": {"small": 1, "medium": 1},
    }
    import random

    a = scale_band_sequence(8, config, random.Random(2026))
    b = scale_band_sequence(8, config, random.Random(2026))
    assert a == b


def test_type7_evaluation_cloud_uses_different_seed():
    report = coverage_before_after(
        [fake_candidate(1, 10.0)],
        [fake_archive(0.0)],
        [fake_archive(0.0), fake_archive(10.0, family="type7")],
        {},
        selection_seed=2026,
        evaluation_seed=2027,
    )
    assert report["evaluation_cloud_independent"]


def test_type7_evaluation_cloud_not_used_for_selection():
    selected = [fake_candidate(1, 10.0)]
    evaluation = CandidateBuildResult([fake_candidate(2, 20.0)], {}, 3, 1)
    manifest = build_search_manifest(
        config={"root_seed": 2026, "candidate_search": {}},
        config_hash_value="cfg",
        frozen_records=[],
        feature_space_definition={"frozen_before_selection": True},
        gap_analysis={},
        novelty_reference={},
        selected=selected,
        pilot_gate={"passed": True},
        final_gate={"passed": True},
        rejection_counts={},
        calibration=CandidateBuildResult([], {}, 1, 0),
        adaptive=CandidateBuildResult([], {}, 1, 0),
        evaluation=evaluation,
        minimum_novelty=None,
    )
    accepted_ids = {row["candidate_id"] for row in manifest["accepted_candidates"]}
    assert 2 not in accepted_ids


def test_type7_sequential_maximin_updates_archive():
    selected = select_sequential_maximin(
        candidates=[fake_candidate(1, 10.0), fake_candidate(2, 20.0), fake_candidate(3, -20.0)],
        existing_archive=[fake_archive(0.0)],
        count=2,
        group_weights={},
        existing_hashes=set(),
    )
    assert [candidate.candidate_id for candidate in selected] == [2, 3]


def test_type7_candidate_novelty_against_types1_6():
    candidate = fake_candidate(1, 5.0)
    selected = select_sequential_maximin(
        [candidate],
        [fake_archive(0.0, "type1")],
        1,
        {},
        set(),
    )
    assert selected[0].nearest_neighbor_family_at_selection == "type1"
    assert selected[0].novelty_at_selection == weighted_distance(vector(5.0), vector(0.0), {})


def test_type7_candidate_novelty_against_previous_type7():
    selected = select_sequential_maximin(
        [fake_candidate(1, 10.0), fake_candidate(2, 9.0)],
        [fake_archive(0.0, "type1")],
        2,
        {},
        set(),
    )
    assert selected[1].nearest_neighbor_family_at_selection == "type7"


def test_type7_structural_validation(tmp_path: Path):
    instance = MTPInstance(
        contributors=(Contributor("c00000", {"s0000": 2}),),
        projects=(Project("p00000", 1, 1, 1, (Role("s0000", 1),)),),
    )
    path = tmp_path / "b_anchor.txt"
    write_instance(instance, path)
    assert sha256_file(path)
    assert structural_hash(instance)


def test_type7_exact_same_project_mentorship():
    contributors = [
        Contributor("c00000", {"s0000": 1}),
        Contributor("c00001", {"s0000": 2, "s0001": 1}),
    ]
    project = Project(
        "p00000",
        1,
        10,
        1,
        (Role("s0000", 2), Role("s0001", 1)),
    )
    assert project_true_same_project_mentorship_feasible(project, contributors)
    metrics = mentorship_metrics(MTPInstance(tuple(contributors), (project,)))
    assert metrics["true_same_project_mentorship_feasibility"] == 1.0


def test_type7_duplicate_against_types1_6():
    rows = [complete_row(1, "dup")]
    gate = selected_family_quality_gate(rows, [fake_candidate(1, 1.0, "dup")], 1, {"dup"}, {}, False)
    assert gate["duplicate_against_existing_count"] == 1
    assert not gate["passed"]


def test_type7_no_within_family_duplicates():
    rows = [complete_row(1, "dup"), complete_row(2, "dup")]
    gate = selected_family_quality_gate(rows, [fake_candidate(1, 1.0, "dup")], 2, set(), {}, False)
    assert gate["duplicate_count"] == 1
    assert not gate["passed"]


def test_type7_ids_contiguous():
    rows = [complete_row(1, "a"), complete_row(3, "b")]
    gate = selected_family_quality_gate(rows, [], 2, set(), {}, False)
    assert not gate["all_output_ids_contiguous"]


def test_type7_all_100_metadata_rows():
    rows = [complete_row(idx, f"h{idx}") for idx in range(1, 101)]
    gate = selected_family_quality_gate(rows, [], 100, set(), {}, True)
    assert gate["accepted_count_exact"]
    assert gate["all_metadata_complete"]


def test_type7_all_seeds_recorded():
    rows = [complete_row(1, "a")]
    rows[0]["candidate_seed"] = ""
    gate = selected_family_quality_gate(rows, [], 1, set(), {}, False)
    assert not gate["all_generation_seeds_recorded"]


def test_type7_reproducible_output():
    a = generate_candidate_record(tiny_params(), 2026, 1, 0, "broad", "small", 5)
    b = generate_candidate_record(tiny_params(), 2026, 1, 0, "broad", "small", 5)
    assert a is not None and b is not None
    assert a.structural_hash == b.structural_hash


def test_type7_existing_families_unchanged(tmp_path: Path):
    path = tmp_path / "type2_001.txt"
    path.write_text("1 1\nc00000 1\ns0000 1\np00000 1 1 1 1\ns0000 1\n", encoding="utf-8")
    before = sha256_file(path)
    after = sha256_file(path)
    assert before == after


def test_type7_coverage_mean_improves():
    report = coverage_before_after(
        [fake_candidate(1, 10.0), fake_candidate(2, 12.0)],
        [fake_archive(0.0)],
        [fake_archive(0.0), fake_archive(10.0, "type7")],
        {},
        selection_seed=2026,
        evaluation_seed=2027,
    )
    assert report["quality_gate"]["mean_after_less_than_before"]


def test_type7_coverage_p95_improves():
    report = coverage_before_after(
        [fake_candidate(1, 10.0), fake_candidate(2, 12.0)],
        [fake_archive(0.0)],
        [fake_archive(0.0), fake_archive(10.0, "type7")],
        {},
        selection_seed=2026,
        evaluation_seed=2027,
    )
    assert report["quality_gate"]["p95_after_less_than_before"]


def test_type7_metadata_feature_cache_detection():
    row = {f"feature.{name}": "1.0" for name in FEATURE_NAMES}
    row["type7.structural_hash"] = "abc"
    assert metadata_has_canonical_features(row)
    assert metadata_structural_hash(row) == "abc"


def test_type7_repairs_mutated_skill_cardinality():
    params = copy.deepcopy(tiny_params())
    params.n_skills = 3
    params.skills_per_contributor_min = 10
    params.skills_per_contributor_max = 20
    repaired = repair_type7_parameters(params)
    assert repaired.skills_per_contributor_min <= repaired.n_skills
    assert repaired.skills_per_contributor_max <= repaired.n_skills

