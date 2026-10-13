import inspect
from pathlib import Path

import pytest

from benchmark_generation.core.config import load_config

from experiments import sharding
from experiments.benchmark_discovery import BenchmarkInstance, discover_final_benchmark
from experiments.run_mtp_campaign import make_run_plan
from experiments.sharding import (
    build_shard_manifest,
    canonical_run_plan,
    select_manifest_runs,
    select_shard,
    shard_size_distribution,
    sha256_json,
    validate_cli_shard_args,
)
from experiments.solver_adapters import SolverDefinition, load_solver_definitions, solver_manifest
from experiments.state import RunSpec


@pytest.fixture(scope="module")
def current_campaign_plan():
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "experiments" / "configs" / "mtp_solver_campaign.yaml", defaults={})
    instances = discover_final_benchmark(root, config.get("benchmark", {}))
    solvers = load_solver_definitions(config)
    plan = make_run_plan(instances, solvers, int(config.get("runs", 10)), int(config.get("root_seed", 2026)))
    return root, config, instances, solvers, plan


def tiny_plan():
    return [
        RunSpec("solver_b", "inst_2", "type2", "i2.txt", 2, 202, 1),
        RunSpec("solver_a", "inst_1", "type1", "i1.txt", 1, 101, 1),
        RunSpec("solver_a", "inst_2", "type2", "i2.txt", 1, 201, 1),
        RunSpec("solver_b", "inst_1", "type1", "i1.txt", 2, 102, 1),
    ]


def test_sharding_77_total_run_count(current_campaign_plan):
    _root, config, instances, solvers, plan = current_campaign_plan
    assert len(instances) == 611
    assert len(solvers) == 10
    assert int(config.get("runs", 10)) == 10
    assert len(plan) == 61100


def test_sharding_77_min_size_793(current_campaign_plan):
    sizes = shard_size_distribution(len(current_campaign_plan[4]), 77)
    assert min(sizes) == 793


def test_sharding_77_max_size_794(current_campaign_plan):
    sizes = shard_size_distribution(len(current_campaign_plan[4]), 77)
    assert max(sizes) == 794


def test_sharding_77_exact_distribution_39x794_38x793(current_campaign_plan):
    sizes = shard_size_distribution(len(current_campaign_plan[4]), 77)
    assert sizes.count(794) == 39
    assert sizes.count(793) == 38


def test_sharding_union_equals_full_campaign(current_campaign_plan):
    plan = current_campaign_plan[4]
    full = {spec.run_id for spec in plan}
    union = set()
    for shard_index in range(77):
        selected, _selection = select_shard(plan, 77, shard_index)
        union.update(spec.run_id for spec in selected)
    assert union == full


def test_sharding_no_overlap(current_campaign_plan):
    plan = current_campaign_plan[4]
    counts = {}
    for shard_index in range(77):
        selected, _selection = select_shard(plan, 77, shard_index)
        for spec in selected:
            counts[spec.run_id] = counts.get(spec.run_id, 0) + 1
    assert all(count == 1 for count in counts.values())


def test_sharding_no_missing_runs(current_campaign_plan):
    plan = current_campaign_plan[4]
    counts = {}
    for shard_index in range(77):
        selected, _selection = select_shard(plan, 77, shard_index)
        for spec in selected:
            counts[spec.run_id] = counts.get(spec.run_id, 0) + 1
    assert set(counts) == {spec.run_id for spec in plan}


def test_sharding_deterministic():
    first, _ = select_shard(tiny_plan(), 3, 1)
    second, _ = select_shard(tiny_plan(), 3, 1)
    assert [spec.run_id for spec in first] == [spec.run_id for spec in second]


def test_sharding_platform_independent():
    plan = tiny_plan()
    forward, _ = select_shard(plan, 2, 0)
    reversed_plan, _ = select_shard(list(reversed(plan)), 2, 0)
    assert [spec.run_id for spec in forward] == [spec.run_id for spec in reversed_plan]
    assert [spec.run_id for spec in canonical_run_plan(plan)] == [
        "solver_a__inst_1__exec_01",
        "solver_b__inst_1__exec_02",
        "solver_a__inst_2__exec_01",
        "solver_b__inst_2__exec_02",
    ]


def test_sharding_does_not_use_builtin_hash():
    assert "hash(" not in inspect.getsource(sharding)


def test_sharding_seed_invariant(current_campaign_plan):
    plan = current_campaign_plan[4]
    full_seed_by_run_id = {spec.run_id: spec.seed for spec in plan}
    selected, _selection = select_shard(plan, 77, 17)
    assert selected
    assert all(full_seed_by_run_id[spec.run_id] == spec.seed for spec in selected)


def test_unsharded_behavior_unchanged():
    instances = [
        BenchmarkInstance("type1", "inst_a", "a.txt", "sha", 1, 1, 1, 1),
        BenchmarkInstance("type2", "inst_b", "b.txt", "sha", 1, 1, 1, 1),
    ]
    solvers = [
        SolverDefinition("solver_b", "s.cpp", "s", True),
        SolverDefinition("solver_a", "s.cpp", "s", True),
    ]
    plan = make_run_plan(instances, solvers, 2, 2026)
    assert [(spec.solver_id, spec.instance_id, spec.execution_index) for spec in plan] == [
        ("solver_a", "inst_a", 1),
        ("solver_a", "inst_a", 2),
        ("solver_a", "inst_b", 1),
        ("solver_a", "inst_b", 2),
        ("solver_b", "inst_a", 1),
        ("solver_b", "inst_a", 2),
        ("solver_b", "inst_b", 1),
        ("solver_b", "inst_b", 2),
    ]


def test_shard_manifest_equals_cli_selection(tmp_path):
    plan = tiny_plan()
    selected, _selection = select_shard(plan, 2, 1)
    manifest = build_shard_manifest(
        run_plan=plan,
        selected=selected,
        shard_index=1,
        num_shards=2,
        root_seed=2026,
        runs_per_solver_instance=2,
        benchmark_manifest_hash="bench",
        solver_manifest_hash="solver",
        campaign_config_hash="config",
    )
    path = tmp_path / "shard_001.json"
    path.write_text(__import__("json").dumps(manifest), encoding="utf-8")

    from_manifest, selection, _payload = select_manifest_runs(
        plan,
        path,
        root_seed=2026,
        runs_per_solver_instance=2,
        benchmark_manifest_hash="bench",
    )
    assert [spec.run_id for spec in from_manifest] == [spec.run_id for spec in selected]
    assert selection.shard_index == 1


def test_invalid_shard_index_rejected():
    with pytest.raises(ValueError):
        validate_cli_shard_args(77, 77, None)


def test_missing_num_shards_argument_rejected():
    with pytest.raises(ValueError):
        validate_cli_shard_args(None, 0, None)
