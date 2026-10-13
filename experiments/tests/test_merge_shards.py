import csv
from collections import Counter
from pathlib import Path

import pytest

from experiments.common import atomic_write_json, read_json
from experiments.merge_shards import main as merge_main
from experiments.sharding import SEED_DERIVATION_VERSION, select_shard, sha256_json
from experiments.state import ERROR, SUCCESS, TIMEOUT, RunSpec, result_path, run_directory


def synthetic_plan(runs=10, solvers=("solver",), instances=(("type1", "inst", "instances/inst.txt"),)):
    return [
        RunSpec(
            solver_id=solver_id,
            instance_id=instance_id,
            family=family,
            instance_path=path,
            execution_index=execution_index,
            seed=100000 + solver_index * 10000 + instance_index * 1000 + execution_index,
            cpu_slots=1,
        )
        for solver_index, solver_id in enumerate(solvers)
        for instance_index, (family, instance_id, path) in enumerate(instances)
        for execution_index in range(1, runs + 1)
    ]


def campaign_manifests_root(
    tmp_path,
    *,
    num_shards=2,
    runs=10,
    solvers=("solver",),
    statuses=None,
    scores=None,
    omit_run_id=None,
):
    plan = synthetic_plan(runs=runs, solvers=solvers)
    benchmark_manifest = {
        "count": 1,
        "family_counts": {"type1": 1},
        "instances": [
            {
                "family": "type1",
                "instance_id": "inst",
                "path": "instances/inst.txt",
                "sha256": "instance-sha",
                "contributors": 1,
                "projects": 1,
                "skills": 1,
                "roles": 1,
            }
        ],
    }
    solver_manifest = {
        "count": len(solvers),
        "solvers": [
            {
                "solver_id": solver_id,
                "source": f"{solver_id}.cpp",
                "binary": f"{solver_id}",
                "source_sha256": "source-sha",
                "binary_sha256": "binary-sha",
                "binary_exists": True,
                "objective": "maximize",
            }
            for solver_id in solvers
        ],
    }
    input_root = tmp_path / "shards"
    input_root.mkdir()
    statuses = statuses or {}
    scores = scores or {}
    for shard_index in range(num_shards):
        selected, selection = select_shard(plan, num_shards, shard_index)
        campaign_dir = input_root / f"kaggle_shard_{shard_index:03d}"
        campaign_dir.mkdir()
        manifest = {
            "experiment_schema_version": "mtp_experiment_campaign_v1",
            "evaluator_version": "mtp_evaluator_adapter_v1",
            "seed_derivation_version": SEED_DERIVATION_VERSION,
            "campaign_id": f"kaggle_shard_{shard_index:03d}",
            "config_hash": "config-sha",
            "config": {"runs": runs, "time_limit_seconds": 180},
            "root_seed": 2026,
            "runs": runs,
            "time_limit_seconds": 180.0,
            "timeout_grace_seconds": 15.0,
            "threads": 4,
            "global_planned_run_count": len(plan),
            "planned_run_count": len(selected),
            "benchmark_manifest_hash": sha256_json(benchmark_manifest),
            "solver_manifest_hash": sha256_json(solver_manifest),
            "benchmark_manifest": benchmark_manifest,
            "solver_manifest": solver_manifest,
            "solver_audit": [],
            "shard": selection.to_campaign_payload(),
        }
        atomic_write_json(campaign_dir / "campaign_manifest.json", manifest)
        atomic_write_json(campaign_dir / "benchmark_manifest.json", benchmark_manifest)
        atomic_write_json(campaign_dir / "solver_manifest.json", solver_manifest)
        for spec in selected:
            if spec.run_id == omit_run_id:
                continue
            status = statuses.get(spec.run_id, SUCCESS)
            score = scores.get(spec.run_id, spec.execution_index)
            write_run_record(campaign_dir, spec, status, score)
    return input_root, plan


def write_run_record(campaign_dir: Path, spec: RunSpec, status: str, score: int):
    run_dir = run_directory(campaign_dir, spec)
    run_dir.mkdir(parents=True, exist_ok=True)
    solution_path = run_dir / "solution.txt"
    solution_path.write_text("0\n", encoding="utf-8")
    result = {
        "run_id": spec.run_id,
        "solver_id": spec.solver_id,
        "instance_id": spec.instance_id,
        "family": spec.family,
        "instance_path": spec.instance_path,
        "execution_index": spec.execution_index,
        "seed": spec.seed,
        "status": status,
        "verified_score": score if status == SUCCESS else None,
        "wall_time_seconds": float(100 - spec.execution_index),
        "solution_path": str(solution_path.relative_to(campaign_dir)),
        "time_limit_seconds": 180.0,
        "timeout_grace_seconds": 15.0,
    }
    atomic_write_json(result_path(run_dir), result)
    atomic_write_json(run_dir / "status.json", result)


def test_merge_77_shards(tmp_path):
    input_root, _plan = campaign_manifests_root(tmp_path, num_shards=77, runs=77)
    output_dir = tmp_path / "merged"

    assert merge_main(["--input-root", str(input_root), "--num-shards", "77", "--output-dir", str(output_dir)]) == 0

    summary = read_json(output_dir / "campaign_summary.json")
    assert summary["planned_runs"] == 77
    assert summary["recorded_runs"] == 77


def test_merge_detects_missing_run(tmp_path):
    plan = synthetic_plan(runs=4)
    omit_run_id = plan[0].run_id
    input_root, _plan = campaign_manifests_root(tmp_path, num_shards=2, runs=4, omit_run_id=omit_run_id)

    with pytest.raises(SystemExit):
        merge_main(["--input-root", str(input_root), "--num-shards", "2", "--output-dir", str(tmp_path / "merged")])


def test_merge_detects_duplicate_run(tmp_path):
    input_root, _plan = campaign_manifests_root(tmp_path, num_shards=2, runs=4)
    shard0 = read_json(input_root / "kaggle_shard_000" / "campaign_manifest.json")
    shard1_path = input_root / "kaggle_shard_001" / "campaign_manifest.json"
    shard1 = read_json(shard1_path)
    shard1["shard"]["run_ids"][0] = shard0["shard"]["run_ids"][0]
    atomic_write_json(shard1_path, shard1)

    with pytest.raises(SystemExit):
        merge_main(["--input-root", str(input_root), "--num-shards", "2", "--output-dir", str(tmp_path / "merged")])


def test_merge_detects_incompatible_benchmark_manifest(tmp_path):
    input_root, _plan = campaign_manifests_root(tmp_path, num_shards=2, runs=4)
    shard1_path = input_root / "kaggle_shard_001" / "campaign_manifest.json"
    shard1 = read_json(shard1_path)
    shard1["benchmark_manifest"]["count"] = 999
    atomic_write_json(shard1_path, shard1)

    with pytest.raises(SystemExit):
        merge_main(["--input-root", str(input_root), "--num-shards", "2", "--output-dir", str(tmp_path / "merged")])


def test_merge_detects_incompatible_solver_manifest(tmp_path):
    input_root, _plan = campaign_manifests_root(tmp_path, num_shards=2, runs=4)
    shard1_path = input_root / "kaggle_shard_001" / "campaign_manifest.json"
    shard1 = read_json(shard1_path)
    shard1["solver_manifest"]["solvers"][0]["source_sha256"] = "different"
    atomic_write_json(shard1_path, shard1)

    with pytest.raises(SystemExit):
        merge_main(["--input-root", str(input_root), "--num-shards", "2", "--output-dir", str(tmp_path / "merged")])


def test_merge_preserves_failed_statuses(tmp_path):
    plan = synthetic_plan(runs=4)
    statuses = {plan[0].run_id: ERROR, plan[1].run_id: TIMEOUT}
    input_root, _plan = campaign_manifests_root(tmp_path, num_shards=2, runs=4, statuses=statuses)
    output_dir = tmp_path / "merged"

    merge_main(["--input-root", str(input_root), "--num-shards", "2", "--output-dir", str(output_dir)])

    with (output_dir / "per_run_results.csv").open(newline="", encoding="utf-8") as fh:
        counts = Counter(row["status"] for row in csv.DictReader(fh))
    assert counts[ERROR] == 1
    assert counts[TIMEOUT] == 1
    assert counts[SUCCESS] == 2


def test_merge_best_of_10_across_multiple_shards(tmp_path):
    plan = synthetic_plan(runs=10)
    scores = {spec.run_id: spec.execution_index * 10 for spec in plan}
    input_root, _plan = campaign_manifests_root(tmp_path, num_shards=2, runs=10, scores=scores)
    output_dir = tmp_path / "merged"

    merge_main(["--input-root", str(input_root), "--num-shards", "2", "--output-dir", str(output_dir)])

    with (output_dir / "best_result_per_solver_instance.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["best_execution_index"] == "10"
    assert rows[0]["best_verified_score"] == "100"
