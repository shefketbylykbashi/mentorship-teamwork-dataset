import http.client
import json

import pytest

from benchmark_generation.core.generator import (
    EmpiricalSourceProfile,
    GenerationParameters,
    generate_instance,
)
from benchmark_generation.core.parser import instance_to_text, write_instance
from benchmark_generation.type5 import collect_github_corpus
from benchmark_generation.type5.build_empirical_profiles import assign_contiguous_family_ids
from benchmark_generation.type5.collect_github_corpus import GitHubClient, collect_repository
from benchmark_generation.type5.empirical import (
    activity_score,
    choose_profile_clusters,
    finalize_source_profile,
    percentile_level,
    sample_type5_difficulty_prior,
)
from benchmark_generation.type5.quality import final_instance_names_are_anonymized
from benchmark_generation.type5.generate_type5 import generate_type5, select_pilot_records
from benchmark_generation.type5.sampling import deterministic_pr_sample
from benchmark_generation.type5.sources import ObservationWindow, SourceRepo, load_type5_sources
from benchmark_generation.type5.taxonomy import (
    is_bot_identity,
    should_exclude_path,
    skill_token_for_path,
    subsystem_for_path,
)


def test_type5_source_configuration_has_fixed_design():
    sources = load_type5_sources("benchmark_generation/configs/type5_sources.yaml")
    assert len(sources["primary"]) == 25
    assert len(sources["windows"]) == 4
    assert sources["windows"][0].start == "2024-08-17T00:00:00Z"
    assert sources["windows"][-1].end == "2026-08-17T00:00:00Z"
    assert sources["windows"][0].end_dt == sources["windows"][1].start_dt


def test_type5_path_taxonomy_language_subsystem_and_exclusions():
    assert skill_token_for_path("src/compiler/parser/file.rs") == "Rust::compiler"
    assert skill_token_for_path("packages/react-dom/src/index.js") == "JavaScript::react-dom"
    assert skill_token_for_path(".github/workflows/ci.yml") == "DevOps::workflows"
    assert should_exclude_path("node_modules/pkg/index.js")
    assert should_exclude_path("web/dist/app.min.js")
    assert subsystem_for_path("django/db/models/query.py") == "db"


def test_type5_bot_filtering_patterns():
    patterns = ["\\[bot\\]", "dependabot", "github-actions"]
    assert is_bot_identity("dependabot[bot]", patterns)
    assert is_bot_identity("github-actions", patterns)
    assert not is_bot_identity("human-maintainer", patterns)


def test_type5_pr_sampling_is_deterministic_and_bounded():
    prs = []
    for idx in range(90):
        month = 1 + (idx % 6)
        prs.append(
            {
                "number": idx + 1,
                "merged_at": f"2025-{month:02d}-10T00:00:00Z",
                "user": {"login": f"user{idx % 12}"},
            }
        )
    a = deterministic_pr_sample(prs, seed=2026)
    b = deterministic_pr_sample(prs, seed=2026)
    assert [pr["number"] for pr in a] == [pr["number"] for pr in b]
    assert len(a) == 60


def test_type5_activity_score_and_percentile_levels():
    assert activity_score(100, 10, 5) > activity_score(10, 2, 1)
    assert percentile_level(0, 5) == 1
    assert percentile_level(4, 5) == 10


def test_type5_profile_clustering_selects_valid_k():
    matrix = [
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.0, 1.0, 0.0],
        [0.1, 0.9, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.1, 0.9],
    ]
    result = choose_profile_clusters(matrix, min_k=3, max_k=4, seed=2026)
    assert result["valid"]
    assert 3 <= result["k"] <= 4
    assert len(result["profile_sizes"]) == result["k"]


def test_type5_difficulty_prior_is_deterministic_and_bounded():
    config = {
        "difficulty_prior": {
            "dirichlet_alpha": [25, 45, 20, 10],
            "mentor_bounds": [0.15, 0.30],
            "future_bounds": [0.05, 0.15],
        }
    }
    a = sample_type5_difficulty_prior(2026, config)
    b = sample_type5_difficulty_prior(2026, config)
    assert a == b
    assert abs(sum(a) - 1.0) < 1e-12
    assert 0.10 <= a[2] <= 0.35
    assert 0.03 <= a[3] <= 0.18


def test_type5_parameter_conversion_from_measurement():
    measurement = fake_measurement()
    config = fake_type5_config()
    record = finalize_source_profile(
        measurement,
        config,
        seed=2026,
        family_instance_id=1,
        corpus_deadline_bounds=(1.0, 3.0),
        global_score_k=1.5,
    )
    params = record["parameters"]
    assert params["n_contributors"] == 6
    assert params["n_projects"] == 12
    assert params["n_skills"] == 5
    assert 3 <= params["n_profiles"] <= 8
    assert params["role_skill_alignment"] >= 0.60
    assert record["source_slot_id"] == 1
    assert record["source_profile"]["duration_values"] == [1, 2, 3, 4, 5, 6]


def test_type5_filtered_records_get_contiguous_family_ids():
    records = [
        {"family_instance_id": 1, "source_slot_id": 1},
        {"family_instance_id": 7, "source_slot_id": 7},
        {"family_instance_id": 10, "source_slot_id": 10},
    ]
    replacements = [{"source_slot_id": 7, "replacement_source": "reserve/repo"}]

    result = assign_contiguous_family_ids(records, replacements)

    assert [record["source_slot_id"] for record in result] == [1, 7, 10]
    assert [record["family_instance_id"] for record in result] == [1, 2, 3]
    assert replacements[0]["family_instance_id"] == 2


def test_type5_final_generation_rejects_count_mismatch_before_work(tmp_path):
    corpus_path = tmp_path / "type5_empirical_corpus.json"
    corpus_path.write_text(
        json.dumps(
            {
                "family": "type5",
                "records": [{}],
                "retained_valid_source_configurations": 1,
                "final_expected_type5_count": 1,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="must use all 1 retained"):
        generate_type5(
            output_dir=tmp_path / "instances",
            metadata_dir=tmp_path / "metadata",
            count=2,
            seed=2026,
            config={"root_seed": 2026, "empirical_corpus_path": str(corpus_path)},
            dry_run=True,
            pilot=False,
        )


def test_type5_pilot_selection_does_not_repeat_exact_source_records():
    records = []
    for idx, window in enumerate(["W1", "W2", "W3", "W4"], start=1):
        records.append(
            {
                "family_instance_id": idx,
                "source_slot_id": idx,
                "source_owner": f"repo{idx}",
                "source_repo": "main",
                "source_domain": f"domain{idx}",
                "source_window_name": window,
            }
        )
    for idx in range(5, 13):
        records.append(
            {
                "family_instance_id": idx,
                "source_slot_id": idx,
                "source_owner": f"repo{idx}",
                "source_repo": "main",
                "source_domain": f"domain{idx}",
                "source_window_name": "W1",
            }
        )

    selected = select_pilot_records(records, 10)

    assert len(selected) == 10
    assert len({record["source_slot_id"] for record in selected}) == 10
    assert {record["source_window_name"] for record in selected} == {"W1", "W2", "W3", "W4"}


def test_type5_empirical_profile_never_leaks_semantic_tokens_to_txt(tmp_path):
    params = GenerationParameters(
        n_skills=5,
        skill_skew=1.0,
        n_contributors=8,
        skills_per_contributor_min=2,
        skills_per_contributor_max=3,
        level_mean=4.0,
        level_sd=1.0,
        level_max=10,
        n_profiles=3,
        n_projects=5,
        duration_min=1,
        duration_max=3,
        roles_min=1,
        roles_max=3,
        score_base=1.0,
        score_perturbation=1.2,
        tightness_min=1.0,
        tightness_max=3.0,
        role_skill_alignment=0.8,
        supply_quantile=0.7,
        p_over=0.25,
        p_match=0.45,
        p_mentor=0.20,
        p_future=0.10,
        fillable_fraction=0.0,
    )
    profile = EmpiricalSourceProfile(
        skill_tokens=("alice_private_name", "Python::db", "Go::Core", "Rust::Compiler", "TypeScript::UI"),
        skill_weights=(0.4, 0.2, 0.2, 0.1, 0.1),
        demand_weights=(0.3, 0.3, 0.2, 0.1, 0.1),
        profile_centroids=((1, 0, 0, 0, 0), (0, 1, 1, 0, 0), (0, 0, 0, 1, 1)),
        profile_sizes=(3, 3, 2),
    )
    instance = generate_instance(params, 123, empirical_profile=profile)
    text = instance_to_text(instance)
    assert "alice_private_name" not in text
    assert "Python::db" not in text
    path = tmp_path / "type5_001.txt"
    write_instance(instance, path)
    assert final_instance_names_are_anonymized(path)


def test_type5_empirical_duration_profile_drives_project_durations():
    params = GenerationParameters(
        n_skills=5,
        skill_skew=1.0,
        n_contributors=8,
        skills_per_contributor_min=2,
        skills_per_contributor_max=3,
        level_mean=4.0,
        level_sd=1.0,
        level_max=10,
        n_profiles=3,
        n_projects=8,
        duration_min=1,
        duration_max=20,
        roles_min=1,
        roles_max=2,
        score_base=1.0,
        score_perturbation=1.2,
        tightness_min=1.0,
        tightness_max=3.0,
        role_skill_alignment=0.8,
        supply_quantile=0.7,
        p_over=0.25,
        p_match=0.45,
        p_mentor=0.20,
        p_future=0.10,
        fillable_fraction=0.0,
    )
    profile = EmpiricalSourceProfile(
        skill_weights=(0.2, 0.2, 0.2, 0.2, 0.2),
        demand_weights=(0.2, 0.2, 0.2, 0.2, 0.2),
        profile_centroids=((1, 0, 0, 0, 0), (0, 1, 1, 0, 0), (0, 0, 0, 1, 1)),
        profile_sizes=(3, 3, 2),
        duration_values=(2, 2, 2),
    )
    instance = generate_instance(params, 456, empirical_profile=profile)

    assert {project.duration for project in instance.projects} == {2}


def test_type5_github_client_retries_remote_disconnected(monkeypatch, tmp_path):
    calls = []

    class FakeResponse:
        status = 200
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, _exc_type, _exc, _traceback):
            return False

        def read(self):
            return b'{"ok": true}'

    def fake_urlopen(_request, timeout):
        calls.append(timeout)
        if len(calls) == 1:
            raise http.client.RemoteDisconnected("Remote end closed connection without response")
        return FakeResponse()

    monkeypatch.setattr(collect_github_corpus.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(collect_github_corpus.time, "sleep", lambda _seconds: None)

    client = GitHubClient("token", retry_attempts=2, backoff_initial_seconds=0.01)
    result = client.get_json("/repos/example/repo", {}, tmp_path / "repository.json")

    assert result == {"ok": True}
    assert len(calls) == 2
    cached = json.loads((tmp_path / "repository.json").read_text(encoding="utf-8"))
    assert cached["body"] == {"ok": True}


def test_type5_completed_repository_is_skipped_on_resume(tmp_path):
    windows = [
        ObservationWindow("W1", "2024-08-17T00:00:00Z", "2025-02-17T00:00:00Z"),
        ObservationWindow("W2", "2025-02-17T00:00:00Z", "2025-08-17T00:00:00Z"),
        ObservationWindow("W3", "2025-08-17T00:00:00Z", "2026-02-17T00:00:00Z"),
        ObservationWindow("W4", "2026-02-17T00:00:00Z", "2026-08-17T00:00:00Z"),
    ]
    repo = SourceRepo(
        owner="owner",
        repo="repo",
        url="https://github.com/owner/repo",
        domain="test",
        kind="primary",
        index=1,
    )
    summary = {
        "source": {
            "owner": "owner",
            "repo": "repo",
            "url": "https://github.com/owner/repo",
            "domain": "test",
            "kind": "primary",
            "index": 1,
        },
        "repository_id": 123,
        "default_branch": "main",
        "default_branch_head_sha": "abc",
        "collection_timestamp": "2026-08-19T00:00:00Z",
        "windows": {
            window.name: {
                "window": {
                    "name": window.name,
                    "start": window.start,
                    "end": window.end,
                },
                "source_pr_total_count": 5,
                "source_sampled_pr_count": 5,
                "source_commit_count": 7,
                "collected_at": "2026-08-19T00:00:00Z",
            }
            for window in windows
        },
    }
    repo_dir = tmp_path / "owner__repo"
    repo_dir.mkdir()
    (repo_dir / "collection_summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )

    class FailingClient:
        def get_json(self, *_args, **_kwargs):
            raise AssertionError("completed repository should not perform API requests")

        def paginate(self, *_args, **_kwargs):
            raise AssertionError("completed repository should not paginate")

    result = collect_repository(FailingClient(), repo, windows, tmp_path, {}, seed=2026)

    assert result == summary


def fake_measurement():
    return {
        "valid": True,
        "source": {
            "owner": "owner",
            "repo": "repo",
            "url": "https://github.com/owner/repo",
            "domain": "test",
            "kind": "primary",
            "index": 1,
        },
        "window": {
            "name": "W1",
            "start": "2024-08-17T00:00:00Z",
            "end": "2025-02-17T00:00:00Z",
        },
        "repository_id": 1,
        "default_branch": "main",
        "snapshot_sha": "abc",
        "collection_timestamp": "2026-08-17T00:00:00Z",
        "license": "MIT",
        "merged_pr_count": 12,
        "sampled_pr_count": 12,
        "human_contributor_count": 6,
        "bot_count": 1,
        "skill_count": 5,
        "skill_tokens": ["Python::a", "Python::b", "Go::a", "Rust::a", "TypeScript::a"],
        "skill_supply_weights": [0.3, 0.2, 0.2, 0.2, 0.1],
        "skill_demand_weights": [0.25, 0.25, 0.2, 0.2, 0.1],
        "profile_count": 3,
        "profile_silhouette": 0.5,
        "profile_sizes": [2, 2, 2],
        "profile_centroids": [[1, 0, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 1, 1]],
        "largest_profile_share": 1 / 3,
        "source_skill_skew": 0.6,
        "source_alignment": 0.8,
        "source_supply_concentration": 0.2,
        "source_demand_concentration": 0.2,
        "skills_per_contributor_values": [1, 2, 2, 3, 3, 4],
        "skills_per_contributor_mean": 2.5,
        "roles_per_work_unit_values": [1, 2, 2, 3, 3, 4],
        "roles_per_work_unit_mean": 2.5,
        "roles_per_work_unit_median": 2.5,
        "duration_values": [1, 2, 3, 4, 5, 6],
        "duration_p10": 1.5,
        "duration_median": 3.5,
        "duration_p90": 5.5,
        "deadline_observations": [1.0, 1.5, 2.0, 2.5, 3.0],
        "deadline_proxy_coverage": 0.5,
        "deadline_proxy_mode_raw": "mixed",
        "effort_densities": [0.5, 1.0, 1.5, 2.0],
        "level_mean": 4.0,
        "level_sd": 1.0,
        "supply_quantile_estimate": 0.75,
        "source_quality": {"passed": True, "issues": []},
    }


def fake_type5_config():
    return {
        "initial_candidate_source_slots": 100,
        "final_count_policy": "derived_from_retained_valid_empirical_source_configurations",
        "fillable_fraction": 0.30,
        "skill_count_range": [5, 200],
        "profile_count_range": [3, 8],
        "score_base_range": [0.5, 5.0],
        "score_perturbation_range": [1.0, 3.0],
        "duration_range": [1, 500],
        "roles_per_project_range": [1, 20],
        "tightness_range": [0.5, 6.0],
        "deadline_min_observations": 10,
        "difficulty_prior": {
            "dirichlet_alpha": [25, 45, 20, 10],
            "mentor_bounds": [0.15, 0.30],
            "future_bounds": [0.05, 0.15],
        },
    }
