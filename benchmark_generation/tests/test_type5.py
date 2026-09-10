import http.client
import csv
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
from benchmark_generation.type5.collect_github_corpus import GitHubClient, GitHubNotFoundError, collect_repository
from benchmark_generation.type5.collect_github_full_projects import (
    CandidateSource,
    collect_full_project_repositories,
    collect_full_project_repository,
    discover_candidate_repositories,
    write_collection_outputs,
)
from benchmark_generation.type5.empirical import (
    activity_score,
    choose_profile_clusters,
    finalize_source_profile,
    percentile_level,
    sample_type5_difficulty_prior,
)
from benchmark_generation.type5.quality import final_instance_names_are_anonymized
from benchmark_generation.type5.generate_type5 import generate_type5, select_pilot_records
from benchmark_generation.type5.generate_type5_full_project import (
    final_full_project_quality_gate,
    validate_extension_corpus,
)
from benchmark_generation.type5.full_project import (
    FINAL_EXTENSION_START_ID,
    augment_full_project_measurement,
    existing_type5_repository_exclusion_records,
    finalize_full_project_source_profile,
    full_project_filename,
    select_diverse_full_project_sources,
)
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


def test_type5_github_client_does_not_retry_404(monkeypatch, tmp_path):
    calls = []

    def fake_urlopen(_request, timeout):
        calls.append(timeout)
        raise collect_github_corpus.urllib.error.HTTPError(
            url="https://api.github.com/repos/missing/repo",
            code=404,
            msg="Not Found",
            hdrs={},
            fp=None,
        )

    monkeypatch.setattr(collect_github_corpus.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        collect_github_corpus.time,
        "sleep",
        lambda _seconds: pytest.fail("404 must not be treated as retryable"),
    )

    client = GitHubClient("token", retry_attempts=4, backoff_initial_seconds=0.01)
    with pytest.raises(GitHubNotFoundError) as exc_info:
        client.get_json("/repos/missing/repo", {}, tmp_path / "repository.json")

    assert len(calls) == 1
    assert exc_info.value.status == 404
    assert exc_info.value.endpoint == "/repos/missing/repo"


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


def test_type5_full_project_metadata_404_is_candidate_rejection(tmp_path):
    repo = SourceRepo("missing", "repo", "https://github.com/missing/repo", "test", "full_project", 1)

    class Metadata404Client:
        def __init__(self):
            self.calls = []

        def get_json(self, endpoint, *_args, **_kwargs):
            self.calls.append(endpoint)
            raise GitHubNotFoundError(endpoint, 404, "HTTP Error 404: Not Found")

    client = Metadata404Client()
    summary = collect_full_project_repository(
        client,
        repo,
        tmp_path,
        {"per_page": 100},
        {},
        seed=2026,
        exclusion_records=[],
    )

    assert summary["valid"] is False
    assert summary["failure_reason"] == "github_repository_not_found_or_inaccessible"
    assert summary["rejection_endpoint"] == "/repos/missing/repo"
    assert summary["rejection_http_status"] == 404
    assert summary["rejection_stage"] == "repository_metadata"
    assert summary["github_failure"]["repository"] == "missing/repo"
    assert len(client.calls) == 1


def test_type5_full_project_pulls_404_is_candidate_rejection(tmp_path):
    repo = SourceRepo("gone", "repo", "https://github.com/gone/repo", "test", "full_project", 1)

    class Pulls404Client:
        def __init__(self):
            self.calls = []

        def get_json(self, endpoint, *_args, **_kwargs):
            self.calls.append(endpoint)
            if endpoint == "/repos/gone/repo":
                return {
                    "id": 10,
                    "default_branch": "main",
                    "private": False,
                    "archived": False,
                    "disabled": False,
                    "fork": False,
                }
            if endpoint == "/repos/gone/repo/branches/main":
                return {"commit": {"sha": "abc"}}
            if endpoint == "/repos/gone/repo/languages":
                return {"Python": 1}
            if endpoint == "/repos/gone/repo/pulls":
                raise GitHubNotFoundError(endpoint, 404, "HTTP Error 404: Not Found")
            raise AssertionError(f"unexpected endpoint {endpoint}")

    client = Pulls404Client()
    summary = collect_full_project_repository(
        client,
        repo,
        tmp_path,
        {"per_page": 100, "max_full_pr_pages": 2},
        {},
        seed=2026,
        exclusion_records=[],
    )

    assert summary["valid"] is False
    assert summary["failure_reason"] == "github_repository_not_found_or_inaccessible"
    assert summary["rejection_endpoint"] == "/repos/gone/repo/pulls"
    assert summary["rejection_http_status"] == 404
    assert summary["rejection_stage"] == "pull_request_discovery"
    assert client.calls.count("/repos/gone/repo/pulls") == 1
    assert "/repos/gone/repo/releases" not in client.calls


def test_type5_full_project_404_does_not_terminate_collection_and_next_candidate_runs(tmp_path):
    repos = [
        SourceRepo("gone", "repo", "https://github.com/gone/repo", "test", "full_project", 1),
        SourceRepo("next", "repo", "https://github.com/next/repo", "test", "full_project", 2),
    ]

    class MixedClient:
        def __init__(self):
            self.calls = []

        def get_json(self, endpoint, *_args, **_kwargs):
            self.calls.append(endpoint)
            if endpoint.endswith("/branches/main"):
                return {"commit": {"sha": "abc"}}
            if endpoint.endswith("/languages"):
                return {"Python": 1}
            if endpoint == "/repos/gone/repo":
                return {
                    "id": 1,
                    "default_branch": "main",
                    "private": False,
                    "archived": False,
                    "disabled": False,
                    "fork": False,
                }
            if endpoint == "/repos/gone/repo/pulls":
                raise GitHubNotFoundError(endpoint, 404, "HTTP Error 404: Not Found")
            if endpoint == "/repos/next/repo":
                return {
                    "id": 2,
                    "default_branch": "main",
                    "private": False,
                    "archived": False,
                    "disabled": False,
                    "fork": False,
                }
            if endpoint == "/repos/next/repo/pulls":
                return []
            raise AssertionError(f"unexpected endpoint {endpoint}")

    client = MixedClient()
    summaries = collect_full_project_repositories(
        client,
        repos,
        tmp_path,
        {"per_page": 100, "max_full_pr_pages": 2, "min_merged_prs": 5},
        {},
        seed=2026,
        exclusion_records=[],
    )

    assert len(summaries) == 2
    assert summaries[0]["failure_reason"] == "github_repository_not_found_or_inaccessible"
    assert summaries[1]["failure_reason"] == "fewer_than_five_non_bot_merged_prs"
    assert "/repos/next/repo/pulls" in client.calls


def test_type5_full_project_404_rejection_reason_is_written(tmp_path):
    repo = SourceRepo("gone", "repo", "https://github.com/gone/repo", "test", "full_project", 1)
    summary = {
        "valid": False,
        "failure_reason": "github_repository_not_found_or_inaccessible",
        "source": {
            "owner": repo.owner,
            "repo": repo.repo,
            "url": repo.url,
            "domain": repo.domain,
        },
        "collection_timestamp": "2026-09-09T00:00:00Z",
        "rejection_endpoint": "/repos/gone/repo/pulls",
        "rejection_http_status": 404,
        "rejection_timestamp": "2026-09-09T00:00:00Z",
        "rejection_stage": "pull_request_discovery",
    }

    write_collection_outputs([summary], tmp_path, [])

    with (tmp_path / "type5_full_project_candidates.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows[0]["rejection_reason"] == "github_repository_not_found_or_inaccessible"
    assert rows[0]["rejection_endpoint"] == "/repos/gone/repo/pulls"
    assert rows[0]["rejection_http_status"] == "404"
    assert rows[0]["rejection_stage"] == "pull_request_discovery"
    assert rows[0]["candidate_batch"] == "initial_001_060"
    assert rows[0]["cheap_prescreen_pass"] in {"False", "false", "0"}
    assert rows[0]["detailed_collection_attempted"] in {"False", "false", "0"}
    assert rows[0]["profile_quality_pass"] in {"False", "false", "0"}


def test_type5_full_project_completed_candidate_is_reused_on_resume(tmp_path):
    repo = SourceRepo("done", "repo", "https://github.com/done/repo", "test", "full_project", 1)
    repo_dir = tmp_path / "done__repo"
    repo_dir.mkdir()
    summary = {
        "valid": True,
        "failure_reason": "",
        "source": {
            "owner": "done",
            "repo": "repo",
            "url": "https://github.com/done/repo",
            "domain": "test",
            "kind": "full_project",
            "index": 1,
        },
        "repository_id": 123,
        "collection_timestamp": "2026-09-09T00:00:00Z",
        "project_history_start": "2026-01-01T00:00:00Z",
        "project_history_end": "2026-05-01T00:00:00Z",
        "windows": {},
    }
    (repo_dir / "full_project_collection_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    class FailingClient:
        def get_json(self, *_args, **_kwargs):
            raise AssertionError("completed full-project candidate should use cached summary")

        def paginate(self, *_args, **_kwargs):
            raise AssertionError("completed full-project candidate should not refetch PR file details")

    result = collect_full_project_repository(
        FailingClient(),
        repo,
        tmp_path,
        {},
        {},
        seed=2026,
        exclusion_records=[],
    )

    assert result == summary


def test_type5_full_project_prescreen_rejects_too_few_human_contributors_before_details(tmp_path):
    repo = SourceRepo("small", "team", "https://github.com/small/team", "test", "full_project", 1)

    def pr(number, author):
        return {
            "number": number,
            "user": {"login": author},
            "created_at": f"2026-01-{number:02d}T00:00:00Z",
            "merged_at": f"2026-05-{number:02d}T00:00:00Z",
        }

    class FewAuthorsClient:
        def __init__(self):
            self.calls = []

        def get_json(self, endpoint, *_args, **_kwargs):
            self.calls.append(endpoint)
            if endpoint == "/repos/small/team":
                return {
                    "id": 11,
                    "name": "team",
                    "full_name": "small/team",
                    "description": "technical service",
                    "default_branch": "main",
                    "private": False,
                    "archived": False,
                    "disabled": False,
                    "fork": False,
                }
            if endpoint == "/repos/small/team/branches/main":
                return {"commit": {"sha": "abc"}}
            if endpoint == "/repos/small/team/languages":
                return {"Python": 100}
            if endpoint == "/repos/small/team/pulls":
                return [
                    pr(1, "alice"),
                    pr(2, "bob"),
                    pr(3, "carol"),
                    pr(4, "dave"),
                    pr(5, "alice"),
                ]
            raise AssertionError(f"unexpected endpoint {endpoint}")

        def paginate(self, *_args, **_kwargs):
            raise AssertionError("cheap prescreen should not collect releases, milestones, or PR files")

    client = FewAuthorsClient()
    summary = collect_full_project_repository(
        client,
        repo,
        tmp_path,
        {"per_page": 100, "max_full_pr_pages": 2, "min_merged_prs": 5, "min_human_contributors": 5},
        {},
        seed=2026,
        exclusion_records=[],
    )

    assert summary["valid"] is False
    assert summary["failure_reason"] == "fewer_than_five_plausible_human_contributors"
    assert summary["cheap_prescreen_pass"] is False
    assert summary["detailed_collection_attempted"] is False
    assert summary["plausible_human_contributor_count"] == 4
    assert "/repos/small/team/releases" not in client.calls


def test_type5_full_project_discovery_resumes_cached_candidates_before_expansion(tmp_path):
    cache_dir = tmp_path / "cache"
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    collection = {
        "candidates": [
            {
                "valid": True,
                "source": {
                    "owner": "cached",
                    "repo": "repo",
                    "url": "https://github.com/cached/repo",
                    "domain": "test",
                    "kind": "full_project_search",
                    "index": 1,
                },
                "collection_timestamp": "2026-09-09T00:00:00Z",
            }
        ]
    }
    (metadata_dir / "type5_full_project_collection.json").write_text(json.dumps(collection), encoding="utf-8")

    class SearchClient:
        def get_json(self, endpoint, *_args, **_kwargs):
            assert endpoint == "/search/repositories"
            return {
                "items": [
                    {
                        "owner": {"login": "cached"},
                        "name": "repo",
                        "html_url": "https://github.com/cached/repo",
                    },
                    {
                        "owner": {"login": "new"},
                        "name": "repo",
                        "html_url": "https://github.com/new/repo",
                    },
                ]
            }

    candidates = discover_candidate_repositories(
        SearchClient(),
        {
            "candidate_target": 2,
            "candidate_existing_batch": "initial_001_060",
            "candidate_expansion_batch": "expansion_061_150",
            "candidate_queries": [{"q": "language:Python", "domain": "python", "pages": 1, "per_page": 2}],
        },
        cache_dir,
        [],
        metadata_dir,
    )

    assert [candidate.repo.full_name for candidate in candidates] == ["cached/repo", "new/repo"]
    assert [candidate.candidate_batch for candidate in candidates] == ["initial_001_060", "expansion_061_150"]
    assert [candidate.repo.index for candidate in candidates] == [1, 2]


def test_type5_full_project_candidate_source_batch_is_used_during_collection(tmp_path):
    repo = SourceRepo("gone", "repo", "https://github.com/gone/repo", "test", "full_project", 1)

    class Metadata404Client:
        def get_json(self, endpoint, *_args, **_kwargs):
            raise GitHubNotFoundError(endpoint, 404, "HTTP Error 404: Not Found")

    summaries = collect_full_project_repositories(
        Metadata404Client(),
        [CandidateSource(repo, "expansion_061_150")],
        tmp_path,
        {"per_page": 100},
        {},
        seed=2026,
        exclusion_records=[],
    )

    assert summaries[0]["candidate_batch"] == "expansion_061_150"
    assert summaries[0]["detailed_collection_attempted"] is False


def test_type5_full_project_exclusion_list_uses_frozen_sources(tmp_path):
    metadata_dir = tmp_path
    (metadata_dir / "type5_sources.csv").write_text(
        "\n".join(
            [
                "kind,index,owner,repo,url,domain",
                "primary,1,old,primary,https://github.com/old/primary,test",
                "reserve,1,unused,reserve,https://github.com/unused/reserve,test",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (metadata_dir / "type5_metadata.csv").write_text(
        "source_owner,source_repo\ncontrib,repo\n",
        encoding="utf-8",
    )
    (metadata_dir / "type5_empirical_corpus.json").write_text(
        json.dumps(
            {
                "records": [{"source_owner": "record", "source_repo": "repo"}],
                "replacements": [{"replacement_source": "reserve/used"}],
            }
        ),
        encoding="utf-8",
    )

    records = existing_type5_repository_exclusion_records(
        sources_path=metadata_dir / "type5_sources.csv",
        metadata_csv_path=metadata_dir / "type5_metadata.csv",
        corpus_path=metadata_dir / "type5_empirical_corpus.json",
    )
    repos = {record["repository"] for record in records}

    assert "old/primary" in repos
    assert "contrib/repo" in repos
    assert "record/repo" in repos
    assert "reserve/used" in repos
    assert "unused/reserve" not in repos


def test_type5_full_project_lifetime_gate_rejects_old_project():
    measurement = fake_measurement()
    summary = {
        "project_history_start": "2024-01-01T00:00:00Z",
        "project_history_end": "2026-01-01T00:00:00Z",
        "project_history_days": 731,
        "full_history_merged_pr_count": 10,
        "full_history_human_pr_count": 10,
        "repository_overlap_with_existing_type5": 0,
    }

    result = augment_full_project_measurement(
        measurement,
        summary,
        {"project_history_days_range": [90, 548]},
    )

    assert not result["valid"]
    assert "project_history_above_548_days" in result["failure_reason"]


def test_type5_full_project_source_selection_is_unique_and_deterministic():
    candidates = []
    for idx in range(8):
        measurement = fake_measurement()
        measurement["source"] = {
            "owner": f"owner{idx}",
            "repo": "repo",
            "url": f"https://github.com/owner{idx}/repo",
            "domain": f"domain{idx % 3}",
            "kind": "full_project_candidate",
            "index": idx + 1,
        }
        measurement["project_history_days"] = 100 + idx * 45
        measurement["human_contributor_count"] = 6 + idx
        measurement["merged_pr_count"] = 10 + idx * 2
        measurement["skill_count"] = 5 + idx
        candidates.append(measurement)

    a = select_diverse_full_project_sources(candidates, count=5, seed=2026)
    b = select_diverse_full_project_sources(candidates, count=5, seed=2026)

    assert [item["source"]["owner"] for item in a] == [item["source"]["owner"] for item in b]
    assert len({item["source"]["owner"] for item in a}) == 5


def test_type5_full_project_profile_finalization_sets_extension_fields():
    measurement = fake_measurement()
    measurement.update(
        {
            "project_history_start": "2026-01-01T00:00:00Z",
            "project_history_end": "2026-05-01T00:00:00Z",
            "project_history_days": 120,
            "technical_detail_mode": "complete_pr_file_details",
            "full_history_merged_pr_count": 12,
            "full_history_human_pr_count": 6,
        }
    )
    record = finalize_full_project_source_profile(
        measurement,
        fake_type5_config(),
        seed=2026,
        family_instance_id=92,
        selection_rank=1,
        corpus_deadline_bounds=(1.0, 3.0),
        global_score_k=1.0,
    )

    assert record["family_instance_id"] == 92
    assert record["observation_mode"] == "full_project"
    assert record["source_window_type"] == "complete_eligible_project_history"
    assert record["project_history_days"] == 120
    assert record["source_metrics"]["project_lifetime_band"] == "90_180"


def test_type5_full_project_filename_and_corpus_ids():
    filename = full_project_filename(92, "Owner.Name", "repo_name")
    assert filename == "type5_092__GHFULL__Owner.Name__repo_name.txt"
    records = [
        {
            "family_instance_id": FINAL_EXTENSION_START_ID + idx,
            "source_owner": f"owner{idx}",
            "source_repo": "repo",
            "observation_mode": "full_project",
            "source_window_type": "complete_eligible_project_history",
        }
        for idx in range(20)
    ]

    validate_extension_corpus(records, expected_count=20, start_id=92)


def test_type5_full_project_final_gate_uses_configured_extension_count(tmp_path):
    rows = [
        {
            "family_instance_id": FINAL_EXTENSION_START_ID + idx,
            "filename": f"type5_{FINAL_EXTENSION_START_ID + idx:03d}__GHFULL__owner{idx}__repo.txt",
            "source_repository": f"owner{idx}/repo",
            "source_repository_id": idx + 1,
            "source_url": f"https://github.com/owner{idx}/repo",
            "source_snapshot_sha": f"sha{idx}",
            "source_collection_timestamp": "2026-09-10T00:00:00Z",
            "project_history_start": "2026-01-01T00:00:00Z",
            "project_history_end": "2026-05-01T00:00:00Z",
            "project_history_days": 120,
            "project_history_months": 3.94,
            "source_window_type": "complete_eligible_project_history",
            "observation_mode": "full_project",
            "type5.structural_hash": f"hash{idx}",
            "type5.duplicate_pass": True,
            "type5.structural_validation_pass": True,
            "type5.source_distribution_match_pass": True,
            "type5.initial_realistic_feasibility_pass": True,
            "seed": 2026 + idx,
            "root_seed": 2026,
            "structural_hash": f"hash{idx}",
        }
        for idx in range(17)
    ]
    fixed_hashes = {tmp_path / f"type5_{idx:03d}.txt": f"fixed{idx}" for idx in range(1, 92)}

    gate = final_full_project_quality_gate(
        rows,
        tmp_path,
        fixed_hashes,
        fixed_hashes,
        {
            "selection_count": 17,
            "output_family_id_start": FINAL_EXTENSION_START_ID,
            "existing_type5_repository_exclusion_list": [],
        },
        report_exists=True,
    )

    assert gate["new_full_project_instances_exact_configured_count"] is True
    assert gate["combined_type5_instances_exact_configured_total"] is True
    assert gate["new_source_repositories_unique_exact_configured_count"] is True
    assert gate["family_ids_contiguous_from_configured_start"] is True
    assert gate["configured_output_family_id_end"] == 108
    assert gate["passed"] is True


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
