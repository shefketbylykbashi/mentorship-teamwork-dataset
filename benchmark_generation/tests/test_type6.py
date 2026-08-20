import copy
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from benchmark_generation.core.generator import GenerationParameters, generate_instance
from benchmark_generation.core.models import Contributor, MTPInstance, Project, Role
from benchmark_generation.core.parser import parse_file
from benchmark_generation.core.validator import structural_hash
from benchmark_generation.type6.download_mslib import inspect_release_archive
from benchmark_generation.type6.build_literature_profiles import (
    parameters_without_bks,
    profile_from_instance,
)
from benchmark_generation.type6.generate_type6 import (
    final_type6_filename,
    parse_type6_final_filename,
    empirical_profile_from_type6_profile,
)
from benchmark_generation.type6.parse_msrcp import (
    MSRCPSPParseError,
    parse_msrcp_text,
    topological_order,
)
from benchmark_generation.type6.quality import (
    final_instance_names_are_anonymized,
    true_same_project_mentorship_metrics,
)
from benchmark_generation.type6.select_sources import (
    DATASETS,
    CATEGORY_SOLUTION,
    SELECTION_DIMENSION_DEFINITIONS,
    SENSITIVE_WORKBOOK_FIELD_AUDIT,
    build_selection_audit,
    freeze_source_selection,
    source_provenance_complete,
    validate_selection_feature_policy,
)
from benchmark_generation.type6.source_features import extract_source_features


def tiny_msrcp_text():
    return """\\* Project Module *\\
4 3 2 5

10

12

0 2 2 3
3 1 4
2 1 4
0 0

\\* Workforce Module *\\
1 1
1 0
0 1

\\* Workforce Module with Skill Levels *\\
3 2
2 0
0 4

\\* Skill Requirements Module *\\
0 0
2 1
0 1
0 0

\\* Skill Level Requirements Module *\\
-1
2 3 2
4
-1

\\* Cost Module *\\
100 1

\\* Common Resource Usage Module *\\
6

\\* Rework Module *\\
0.5 0.5
"""


def test_msrcp_parser():
    instance = parse_msrcp_text(
        tiny_msrcp_text(),
        dataset="MSLIB1",
        source_filename="MSLIB_Set1_42.msrcp",
        source_zip_entry="MSLIB1/MSLIB_Set1_42.msrcp",
    )
    assert instance.dataset == "MSLIB1"
    assert instance.source_instance_id == 42
    assert instance.n_activities_total == 4
    assert instance.n_real_activities == 2
    assert instance.n_resources == 3
    assert instance.n_skills == 2


def test_dummy_start_end_handling():
    instance = parse_msrcp_text(tiny_msrcp_text(), source_filename="MSLIB_Set1_1.msrcp")
    assert instance.dummy_start_id == 1
    assert instance.dummy_end_id == 4
    assert instance.activities[0].duration == 0
    assert instance.activities[-1].duration == 0
    assert instance.skill_level_requirement_rows[0] == (-1,)
    assert instance.skill_level_requirement_rows[-1] == (-1,)


def test_resource_to_contributor_mapping():
    instance = parse_msrcp_text(tiny_msrcp_text(), source_filename="MSLIB_Set1_1.msrcp")
    assert instance.workforce_binary == ((1, 1), (1, 0), (0, 1))
    assert instance.workforce_levels == ((3, 2), (2, 0), (0, 4))


def test_skill_mapping_and_hierarchical_level_mapping():
    instance = parse_msrcp_text(tiny_msrcp_text(), source_filename="MSLIB_Set1_1.msrcp")
    levels_by_role = [(role.skill_id, role.required_level) for role in instance.role_requirements]
    assert levels_by_role == [(1, 2), (1, 3), (2, 2), (2, 4)]


def test_activity_role_multiplicity():
    instance = parse_msrcp_text(tiny_msrcp_text(), source_filename="MSLIB_Set1_1.msrcp")
    roles_for_activity_2 = [role for role in instance.role_requirements if role.activity_id == 2]
    assert len(roles_for_activity_2) == 3
    assert [role.skill_id for role in roles_for_activity_2] == [1, 1, 2]


def test_duration_profile_preservation():
    instance = parse_msrcp_text(tiny_msrcp_text(), source_filename="MSLIB_Set1_1.msrcp")
    features = extract_source_features(instance)
    assert features["duration_mean"] == 2.5
    assert features["duration_median"] == 2.5
    assert features["duration_p10"] == pytest.approx(2.1)
    assert features["duration_p90"] == pytest.approx(2.9)


def test_precedence_dag_parsing():
    instance = parse_msrcp_text(tiny_msrcp_text(), source_filename="MSLIB_Set1_1.msrcp")
    assert topological_order(instance) == (1, 2, 3, 4)
    assert set(instance.precedence_arcs) == {(1, 2), (1, 3), (2, 4), (3, 4)}


def test_critical_path_calculation():
    instance = parse_msrcp_text(tiny_msrcp_text(), source_filename="MSLIB_Set1_1.msrcp")
    features = extract_source_features(instance)
    assert features["critical_path_length"] == 3
    assert features["network_depth"] == 2
    assert features["order_strength"] == 0.0
    assert features["arc_density"] == pytest.approx(4 / 6)


def test_precedence_not_written_to_mtp():
    instance = parse_msrcp_text(tiny_msrcp_text(), source_filename="MSLIB_Set1_1.msrcp")
    features = extract_source_features(instance)
    assert "n_precedence_arcs" in features
    assert all(not str(key).startswith("feature.") for key in features)


def test_score_proxy_positive_and_deterministic():
    instance = parse_msrcp_text(tiny_msrcp_text(), source_filename="MSLIB_Set1_1.msrcp")
    a = extract_source_features(instance)
    b = extract_source_features(instance)
    assert a["bks_basic"] > 0
    assert a["deadline_skill_levels"] > 0
    assert a == b


def test_same_project_mentorship_semantics_source_potential():
    instance = parse_msrcp_text(tiny_msrcp_text(), source_filename="MSLIB_Set1_1.msrcp")
    features = extract_source_features(instance)
    assert features["natural_direct_share"] == 1.0
    assert features["natural_one_below_share"] == 0.25
    assert features["natural_mentor_available_share"] == 0.25


def test_parser_rejects_inconsistent_skill_level_multiplicity():
    text = tiny_msrcp_text().replace("2 3 2", "2 3")
    with pytest.raises(MSRCPSPParseError):
        parse_msrcp_text(text, source_filename="MSLIB_Set1_1.msrcp")


def test_source_selection_deterministic():
    rows = fake_source_rows()
    config = fake_selection_config()
    a = freeze_source_selection(rows, config, seed=2026)
    b = freeze_source_selection(rows, config, seed=2026)
    assert [(row["MSLIB_set"], row["source_instance_id"]) for row in a] == [
        (row["MSLIB_set"], row["source_instance_id"]) for row in b
    ]


def test_exactly_25_sources_per_mslib1_4():
    selected = freeze_source_selection(fake_source_rows(), fake_selection_config(), seed=2026)
    counts = {dataset: 0 for dataset in DATASETS}
    for row in selected:
        counts[row["MSLIB_set"]] += 1
    assert counts == {dataset: 25 for dataset in DATASETS}
    assert len(selected) == 100


def test_no_mslib5_in_final_type6():
    selected = freeze_source_selection(fake_source_rows(), fake_selection_config(), seed=2026)
    assert {row["MSLIB_set"] for row in selected} == set(DATASETS)


def test_source_selection_independent_of_mtp_isa_features():
    config = fake_selection_config()
    assert all(
        not column.startswith("feature.")
        for column in config["selection"]["feature_columns"]
    )


def test_source_selection_policy_excludes_bks_solution_fields():
    config = fake_selection_config()
    validate_selection_feature_policy(config)
    selected = freeze_source_selection(fake_source_rows(), config, seed=2026)
    audit = build_selection_audit(config, [], selected)
    assert audit["policy_passed"]
    assert audit["selection_dimension_count"] == len(config["selection"]["feature_columns"])
    assert SENSITIVE_WORKBOOK_FIELD_AUDIT["LB"]["category"] == CATEGORY_SOLUTION
    assert SENSITIVE_WORKBOOK_FIELD_AUDIT["UB"]["category"] == CATEGORY_SOLUTION
    assert SENSITIVE_WORKBOOK_FIELD_AUDIT["Open?"]["category"] == CATEGORY_SOLUTION
    assert SENSITIVE_WORKBOOK_FIELD_AUDIT["CP"]["category"] == CATEGORY_SOLUTION
    assert all(audit["sensitive_solution_fields_not_used"].values())
    assert SELECTION_DIMENSION_DEFINITIONS["critical_path_length"]["selection_allowed"]


def test_source_provenance_complete_and_no_duplicate_source_selection():
    selected = freeze_source_selection(fake_source_rows(), fake_selection_config(), seed=2026)
    keys = {(row["MSLIB_set"], row["source_instance_id"]) for row in selected}
    assert len(keys) == len(selected)
    assert all(source_provenance_complete(row) for row in selected)


def test_official_archive_counts_if_present():
    archive = Path("type6_source_data/MSLIB_v2_2.zip")
    if not archive.exists():
        pytest.skip("Pinned MSLIB archive is not present in this checkout")
    config = {
        "expected_counts": {
            "MSLIB1": 6600,
            "MSLIB2": 9000,
            "MSLIB3": 11880,
            "MSLIB4": 5000,
        }
    }
    report = inspect_release_archive(archive, config)
    counts = {
        dataset: data["canonical_msrcp_count"]
        for dataset, data in report["canonical_dataset_counts"].items()
    }
    assert counts == config["expected_counts"]
    assert report["contains_msrcpsp_dataformat_document"]
    assert report["contains_parameters_and_bks_workbook"]
    assert report["noncanonical_msrcp_files_excluded"]["MSLIB2"] == [
        "MSLIB2/MSLIB_Set2_3563 copy.msrcp"
    ]


def test_literature_profile_deterministic_for_source_instance():
    instance = parse_msrcp_text(tiny_msrcp_text(), dataset="MSLIB1", source_filename="MSLIB_Set1_1.msrcp")
    row = tiny_selection_row(instance)
    a = profile_from_instance(instance, row, {}, seed=2026, artifact_statement="test artifact")
    b = profile_from_instance(instance, row, {}, seed=2026, artifact_statement="test artifact")
    assert a == b


def test_literature_profile_no_bks_leakage_into_generation_parameters():
    instance = parse_msrcp_text(tiny_msrcp_text(), dataset="MSLIB1", source_filename="MSLIB_Set1_1.msrcp")
    row = tiny_selection_row(instance)
    changed = replace(instance, bks_basic=99999, deadline_skill_levels=88888)
    original = profile_from_instance(instance, row, {}, seed=2026)
    mutated = profile_from_instance(changed, row, {}, seed=2026)
    assert original["parameters"] == mutated["parameters"]
    assert parameters_without_bks(original)


def test_literature_profile_excludes_dummy_activities_from_project_count():
    instance = parse_msrcp_text(tiny_msrcp_text(), dataset="MSLIB1", source_filename="MSLIB_Set1_1.msrcp")
    profile = profile_from_instance(instance, tiny_selection_row(instance), {}, seed=2026)
    assert profile["parameters"]["n_projects"] == 2
    assert profile["source_metrics"]["source_activity_count"] == 2


def test_literature_profile_preserves_source_resource_and_skill_counts():
    instance = parse_msrcp_text(tiny_msrcp_text(), dataset="MSLIB1", source_filename="MSLIB_Set1_1.msrcp")
    profile = profile_from_instance(instance, tiny_selection_row(instance), {}, seed=2026)
    assert profile["parameters"]["n_contributors"] == 3
    assert profile["parameters"]["n_skills"] == 2
    assert profile["source_metrics"]["source_resource_count"] == 3
    assert profile["source_metrics"]["source_skill_count"] == 2


def test_literature_profile_preserves_duration_distribution():
    instance = parse_msrcp_text(tiny_msrcp_text(), dataset="MSLIB1", source_filename="MSLIB_Set1_1.msrcp")
    profile = profile_from_instance(instance, tiny_selection_row(instance), {}, seed=2026)
    assert profile["source_distributions"]["activity_duration_values"] == [3, 2]
    assert profile["source_profile"]["duration_values"] == [3, 2]
    assert profile["parameters"]["duration_min"] == 2
    assert profile["parameters"]["duration_max"] == 3


def test_literature_profile_preserves_hierarchical_levels():
    instance = parse_msrcp_text(tiny_msrcp_text(), dataset="MSLIB1", source_filename="MSLIB_Set1_1.msrcp")
    profile = profile_from_instance(instance, tiny_selection_row(instance), {}, seed=2026)
    assert sorted(profile["source_distributions"]["source_skill_level_values"]) == [2, 2, 3, 4]
    assert sorted(profile["source_distributions"]["required_skill_level_values"]) == [2, 2, 3, 4]
    assert profile["parameters"]["level_max"] == 5
    assert profile["parameters"]["role_level_max"] == 5


def test_literature_profile_does_not_emit_precedence_as_mtp_constraints():
    instance = parse_msrcp_text(tiny_msrcp_text(), dataset="MSLIB1", source_filename="MSLIB_Set1_1.msrcp")
    profile = profile_from_instance(instance, tiny_selection_row(instance), {}, seed=2026)
    assert "precedence" not in json.dumps(profile["parameters"], sort_keys=True).lower()
    assert "precedence_arcs" not in json.dumps(profile["source_profile"], sort_keys=True).lower()
    assert "network" in profile["source_distributions"]


def test_literature_profile_one_below_mentorship_potential():
    instance = parse_msrcp_text(tiny_msrcp_text(), dataset="MSLIB1", source_filename="MSLIB_Set1_1.msrcp")
    profile = profile_from_instance(instance, tiny_selection_row(instance), {}, seed=2026)
    assert profile["source_metrics"]["source_natural_one_below_potential"] == 0.25
    assert profile["source_metrics"]["source_potential_mentor_availability"] == 0.25
    by_skill_level = profile["source_distributions"]["potential_mentor_availability_by_skill_level"]
    assert by_skill_level["skill_001_level_03"]["one_below_roles"] == 1
    assert by_skill_level["skill_001_level_03"]["mentor_available_roles"] == 1


def test_type6_true_same_project_mentorship_distinguishes_global_mentor():
    instance = MTPInstance(
        contributors=(
            Contributor("c00000", {"s0000": 3, "s0001": 2}),
            Contributor("c00001", {"s0000": 2}),
            Contributor("c00002", {"s0001": 2}),
        ),
        projects=(
            Project("p00000", 1, 10, 2, (Role("s0000", 3), Role("s0001", 2))),
            Project("p00001", 1, 10, 2, (Role("s0000", 3),)),
        ),
    )
    metrics = true_same_project_mentorship_metrics(instance)
    assert metrics["direct_fillability"] == 1.0
    assert metrics["global_one_below_potential"] == pytest.approx(2 / 3)
    assert metrics["global_mentor_availability"] == 1.0
    assert metrics["true_same_project_mentorship_feasibility"] == pytest.approx(1 / 3)


def test_type6_generation_deterministic_seed_2026_from_empirical_profile():
    source = parse_msrcp_text(tiny_msrcp_text(), dataset="MSLIB1", source_filename="MSLIB_Set1_1.msrcp")
    profile = profile_from_instance(source, tiny_selection_row(source), {}, seed=2026)
    params = GenerationParameters(**profile["parameters"]).normalized()
    empirical = empirical_profile_from_type6_profile(profile)
    a = generate_instance(params, seed=2026, empirical_profile=empirical, source_name="type6_test.txt")
    b = generate_instance(params, seed=2026, empirical_profile=empirical, source_name="type6_test.txt")
    assert structural_hash(a) == structural_hash(b)
    assert len(a.contributors) == source.n_resources
    assert len(a.projects) == source.n_real_activities
    assert len(a.skill_universe) == source.n_skills


def test_frozen_literature_profile_output_if_present():
    path = Path("benchmark_metadata/type6_literature_profiles.json")
    if not path.exists():
        pytest.skip("Type 6 Phase 4 profile output is not present in this checkout")
    data = json.loads(path.read_text(encoding="utf-8"))
    profiles = data["profiles"]
    assert len(profiles) == 100
    assert Counter(profile["source_dataset"] for profile in profiles) == {
        dataset: 25 for dataset in DATASETS
    }
    first = copy.deepcopy(profiles[0])
    reread = json.loads(path.read_text(encoding="utf-8"))["profiles"][0]
    assert reread == first


def test_type6_skill_count_sanity_output_if_present():
    path = Path("benchmark_metadata/type6_skill_count_sanity_check.json")
    if not path.exists():
        pytest.skip("Type 6 skill-count sanity output is not present in this checkout")
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["all_mslib1_4_sources_have_four_skills"]
    assert report["workbook_and_parser_agree_exactly"]
    assert report["selected_100_four_skills_is_genuine_source_property"]
    assert report["workbook_unique_skill_counts_per_set"] == {dataset: [4] for dataset in DATASETS}
    assert report["parsed_unique_skill_counts_per_set"] == {dataset: [4] for dataset in DATASETS}


def test_type6_pilot_output_if_present():
    report_path = Path("benchmark_metadata/type6_pilot/type6_pilot_report.json")
    output_dir = Path("generated_instances/type6_pilot")
    if not report_path.exists() or not output_dir.exists():
        pytest.skip("Type 6 pilot output is not present in this checkout")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    gate = report["pilot_quality_gate"]
    assert gate["passed"]
    assert gate["accepted_count"] == 10
    assert gate["pilot_count_per_MSLIB_set"] == {
        "MSLIB1": 2,
        "MSLIB2": 2,
        "MSLIB3": 3,
        "MSLIB4": 3,
    }
    assert sorted(path.name for path in output_dir.glob("*.txt")) == [
        f"type6_pilot_{idx:03d}.txt" for idx in range(1, 11)
    ]
    assert all(row["type6.overall_type6_quality_pass"] for row in report["rows"])


def test_type6_final_filename_contains_family_id():
    name = final_type6_filename(1, {"source_dataset": "MSLIB1", "source_instance_id": 3317})
    assert name.startswith("type6_001__")
    assert parse_type6_final_filename(name)["family_instance_id"] == 1


def test_type6_final_filename_contains_mslib_set():
    name = final_type6_filename(26, {"source_dataset": "MSLIB2", "source_instance_id": 1276})
    assert "__MSLIB2__" in name
    assert parse_type6_final_filename(name)["source_dataset"] == "MSLIB2"


def test_type6_final_filename_contains_source_id():
    name = final_type6_filename(51, {"source_dataset": "MSLIB3", "source_instance_id": 5371})
    assert name.endswith("__src5371.txt")
    assert parse_type6_final_filename(name)["source_instance_id"] == 5371


def test_type6_filename_source_matches_metadata():
    report, mapping, metadata_rows, _output_dir = load_type6_final_artifacts()
    rows_by_filename = {row["output_filename"]: row for row in metadata_rows}
    for row in mapping:
        parsed = parse_type6_final_filename(row["output_filename"])
        metadata = rows_by_filename[row["output_filename"]]
        assert parsed["source_dataset"] == row["source_dataset"] == metadata["source_dataset"]
        assert parsed["source_instance_id"] == int(row["source_instance_id"]) == int(metadata["source_instance_id"])
    assert report["final_quality_gate"]["all_filename_source_ids_match_metadata"]
    assert report["final_quality_gate"]["all_filename_source_sets_match_metadata"]


def test_type6_family_ids_contiguous():
    report, mapping, metadata_rows, _output_dir = load_type6_final_artifacts()
    assert [int(row["family_instance_id"]) for row in mapping] == list(range(1, 101))
    assert sorted(int(row["family_instance_id"]) for row in metadata_rows) == list(range(1, 101))
    assert report["final_quality_gate"]["all_output_family_ids_contiguous"]


def test_type6_output_filenames_unique():
    report, mapping, _metadata_rows, output_dir = load_type6_final_artifacts()
    names = [row["output_filename"] for row in mapping]
    assert len(names) == len(set(names)) == 100
    assert sorted(path.name for path in output_dir.glob("*.txt")) == sorted(names)
    assert report["final_quality_gate"]["all_output_filenames_unique"]


def test_type6_all_frozen_sources_used_once():
    report, mapping, _metadata_rows, _output_dir = load_type6_final_artifacts()
    profile_path = Path("benchmark_metadata/type6_literature_profiles.json")
    profiles = json.loads(profile_path.read_text(encoding="utf-8"))["profiles"]
    frozen = Counter(
        (profile["source_dataset"], int(profile["source_instance_id"]), profile["source_sha256"])
        for profile in profiles
    )
    used = Counter(
        (row["source_dataset"], int(row["source_instance_id"]), row["source_sha256"])
        for row in mapping
    )
    assert used == frozen
    assert len(used) == 100
    assert report["final_quality_gate"]["all_frozen_sources_used_exactly_once"]


def test_type6_source_to_output_mapping_one_to_one():
    report, mapping, metadata_rows, _output_dir = load_type6_final_artifacts()
    metadata_names = {row["output_filename"] for row in metadata_rows}
    assert len(mapping) == 100
    assert len({int(row["family_instance_id"]) for row in mapping}) == 100
    assert len({row["output_filename"] for row in mapping}) == 100
    assert len({(row["source_dataset"], int(row["source_instance_id"]), row["source_sha256"]) for row in mapping}) == 100
    assert {row["output_filename"] for row in mapping} == metadata_names
    assert report["final_quality_gate"]["all_source_to_output_mappings_one_to_one"]


def test_type6_source_identifiers_allowed_in_filename():
    report, mapping, _metadata_rows, _output_dir = load_type6_final_artifacts()
    assert all("__MSLIB" in row["output_filename"] and "__src" in row["output_filename"] for row in mapping)
    assert report["final_quality_gate"]["all_output_filenames_include_source_dataset"]
    assert report["final_quality_gate"]["all_output_filenames_include_source_instance_id"]


def test_type6_source_identifiers_not_inside_instance_content():
    report, mapping, _metadata_rows, output_dir = load_type6_final_artifacts()
    for row in mapping:
        path = output_dir / row["output_filename"]
        text = path.read_text(encoding="utf-8")
        assert row["source_dataset"] not in text
        assert row["source_original_filename"] not in text
        assert f"src{row['source_instance_id']}" not in text
        assert "MSLIB_Set" not in text
        assert final_instance_names_are_anonymized(path)
        parse_file(path)
    assert report["final_quality_gate"]["no_source_identifiers_inside_instance_contents"]


def test_type6_original_source_filename_preserved_in_metadata():
    report, mapping, metadata_rows, _output_dir = load_type6_final_artifacts()
    metadata = {row["output_filename"]: row for row in metadata_rows}
    for row in mapping:
        assert row["source_original_filename"].endswith(".msrcp")
        assert metadata[row["output_filename"]]["source_original_filename"] == row["source_original_filename"]
    assert report["final_quality_gate"]["all_metadata_complete"]


def test_type6_final_quality_gate_if_present():
    report, _mapping, _metadata_rows, _output_dir = load_type6_final_artifacts()
    gate = report["final_quality_gate"]
    assert gate["accepted_count"] == 100
    assert gate["final_type6_instances"] == 100
    assert gate["unique_structural_hashes"] == 100
    assert gate["duplicate_count"] == 0
    assert gate["no_BKS_leakage"]
    assert gate["no_precedence_constraints_in_final_MTP"]
    assert gate["shared_generator_regression_check_pass"]
    assert gate["passed"]


def load_type6_final_artifacts():
    report_path = Path("benchmark_metadata/type6_statistical_report.json")
    mapping_path = Path("benchmark_metadata/type6_source_to_output_mapping.json")
    metadata_path = Path("benchmark_metadata/type6_metadata.jsonl")
    output_dir = Path("generated_instances/type6")
    if not report_path.exists() or not mapping_path.exists() or not metadata_path.exists() or not output_dir.exists():
        pytest.skip("Type 6 final output is not present in this checkout")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    metadata_rows = [
        json.loads(line)
        for line in metadata_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return report, mapping, metadata_rows, output_dir


def fake_selection_config():
    return {
        "expected_counts": {dataset: 30 for dataset in DATASETS},
        "selection": {
            "count_per_dataset": 25,
            "group_seed_limit": 5,
            "feature_columns": [
                "n_real_activities",
                "n_resources",
                "n_skills",
                "total_roles",
                "critical_path_length",
                "natural_direct_share",
            ],
            "categorical_columns": ["n_real_activities", "n_resources", "n_skills"],
        },
    }


def fake_source_rows():
    rows = []
    for dataset_idx, dataset in enumerate(DATASETS, start=1):
        for source_id in range(1, 31):
            rows.append(
                {
                    "MSLIB_set": dataset,
                    "source_instance_id": source_id,
                    "source_filename": f"MSLIB_Set{dataset_idx}_{source_id}.msrcp",
                    "source_zip_entry": f"{dataset}/MSLIB_Set{dataset_idx}_{source_id}.msrcp",
                    "source_sha256": f"{dataset_idx:02d}{source_id:064d}"[-64:],
                    "n_real_activities": 30 + (source_id % 3) * 30,
                    "n_resources": 4 + source_id,
                    "n_skills": 4,
                    "n_skill_levels": 5,
                    "total_roles": 20 + source_id * 2,
                    "critical_path_length": 10 + source_id,
                    "natural_direct_share": 0.5 + (source_id % 10) / 20,
                    "selection_reason": "",
                    "selection_seed": "",
                }
            )
    return rows


def tiny_selection_row(instance):
    row = extract_source_features(instance)
    row.update(
        {
            "type6_source_index": 1,
            "selection_rank_within_set": 1,
            "selection_seed": 2026,
            "selection_reason": "unit_test_source",
        }
    )
    return row
