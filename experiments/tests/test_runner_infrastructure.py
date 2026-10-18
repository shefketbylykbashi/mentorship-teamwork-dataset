import json
import sys
import time
from pathlib import Path

import pytest

from experiments.benchmark_discovery import BenchmarkInstance, limit_instances_per_family
from experiments.common import atomic_write_json, stable_seed
from experiments.reporting import update_best_for_pair
from experiments.run_mtp_campaign import CampaignRunner, prepare_campaign_dir
from experiments.solver_adapters import SolverDefinition
from experiments.state import (
    ERROR,
    INTERRUPTED,
    INVALID_OUTPUT,
    INVALID_SOLUTION,
    SUCCESS,
    TIMEOUT,
    RunLock,
    RunSpec,
    mark_interrupted,
    read_status,
    result_path,
    run_directory,
    write_status,
)


def write_tiny_instance(root: Path) -> Path:
    path = root / "instances" / "tiny.txt"
    path.parent.mkdir(parents=True)
    path.write_text(
        "\n".join(
            [
                "2 1",
                "Alice 1",
                "py 1",
                "Bob 1",
                "py 2",
                "Proj 1 10 1 1",
                "py 2",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def write_solver(root: Path, name: str, body: str) -> Path:
    path = root / f"{name}.py"
    path.write_text(body, encoding="utf-8")
    return path


def solver_def(solver_id: str, script: Path) -> SolverDefinition:
    return SolverDefinition(
        solver_id=solver_id,
        source=str(script.name),
        binary=str(script.name),
        stochastic=True,
        command=(sys.executable, script.name),
        cpu_slots_per_run=1,
    )


def run_spec(solver_id: str, instance_path: Path, root: Path, idx: int = 1) -> RunSpec:
    return RunSpec(
        solver_id=solver_id,
        instance_id="tiny",
        family="test",
        instance_path=str(instance_path.relative_to(root)),
        execution_index=idx,
        seed=stable_seed(2026, solver_id, "tiny", idx),
        cpu_slots=1,
    )


def runner_config(**updates):
    config = {
        "threads": 1,
        "max_parallel_runs": 1,
        "time_limit_seconds": 1.0,
        "timeout_grace_seconds": 0.5,
        "retry_errors": False,
        "retry_timeouts": False,
        "retry_invalid": False,
    }
    config.update(updates)
    return config


def test_nonzero_exit_recorded_and_queue_continues(tmp_path):
    instance = write_tiny_instance(tmp_path)
    ok = write_solver(tmp_path, "ok", "import argparse\nargparse.ArgumentParser().parse_known_args()\nprint('1\\nProj\\nBob')\n")
    fail = write_solver(tmp_path, "fail", "import sys\nsys.exit(2)\n")
    solvers = [solver_def("fail_solver", fail), solver_def("ok_solver", ok)]
    specs = [run_spec("fail_solver", instance, tmp_path), run_spec("ok_solver", instance, tmp_path)]
    runner = CampaignRunner(tmp_path, tmp_path / "campaign", runner_config(), solvers, specs, resume=False)

    runner.run()

    fail_result = json.loads(result_path(run_directory(tmp_path / "campaign", specs[0])).read_text())
    ok_result = json.loads(result_path(run_directory(tmp_path / "campaign", specs[1])).read_text())
    assert fail_result["status"] == ERROR
    assert ok_result["status"] == SUCCESS
    assert ok_result["verified_score"] == 10


def test_timeout_recorded_and_queue_continues(tmp_path):
    instance = write_tiny_instance(tmp_path)
    slow = write_solver(tmp_path, "slow", "import time\ntime.sleep(3)\n")
    ok = write_solver(tmp_path, "ok", "import argparse\nargparse.ArgumentParser().parse_known_args()\nprint('1\\nProj\\nBob')\n")
    solvers = [solver_def("slow_solver", slow), solver_def("ok_solver", ok)]
    specs = [run_spec("slow_solver", instance, tmp_path), run_spec("ok_solver", instance, tmp_path)]
    runner = CampaignRunner(
        tmp_path,
        tmp_path / "campaign",
        runner_config(time_limit_seconds=0.1, timeout_grace_seconds=0.1),
        solvers,
        specs,
        resume=False,
    )

    runner.run()

    slow_result = json.loads(result_path(run_directory(tmp_path / "campaign", specs[0])).read_text())
    ok_result = json.loads(result_path(run_directory(tmp_path / "campaign", specs[1])).read_text())
    assert slow_result["status"] == TIMEOUT
    assert ok_result["status"] == SUCCESS


def test_malformed_and_invalid_outputs_are_separate_states(tmp_path):
    instance = write_tiny_instance(tmp_path)
    malformed = write_solver(tmp_path, "malformed", "print('oops')\n")
    invalid = write_solver(tmp_path, "invalid", "print('1\\nProj\\nAlice')\n")
    solvers = [solver_def("malformed_solver", malformed), solver_def("invalid_solver", invalid)]
    specs = [run_spec("malformed_solver", instance, tmp_path), run_spec("invalid_solver", instance, tmp_path)]
    runner = CampaignRunner(tmp_path, tmp_path / "campaign", runner_config(), solvers, specs, resume=False)

    runner.run()

    malformed_result = json.loads(result_path(run_directory(tmp_path / "campaign", specs[0])).read_text())
    invalid_result = json.loads(result_path(run_directory(tmp_path / "campaign", specs[1])).read_text())
    assert malformed_result["status"] == INVALID_OUTPUT
    assert invalid_result["status"] == INVALID_SOLUTION


def test_resume_skips_valid_success(tmp_path):
    instance = write_tiny_instance(tmp_path)
    ok = write_solver(tmp_path, "ok", "import argparse\nargparse.ArgumentParser().parse_known_args()\nprint('1\\nProj\\nBob')\n")
    solver = solver_def("ok_solver", ok)
    spec = run_spec("ok_solver", instance, tmp_path)
    campaign = tmp_path / "campaign"
    CampaignRunner(tmp_path, campaign, runner_config(), [solver], [spec], resume=False).run()
    first_result = json.loads(result_path(run_directory(campaign, spec)).read_text())

    ok.write_text("import sys\nsys.exit(9)\n", encoding="utf-8")
    CampaignRunner(tmp_path, campaign, runner_config(), [solver], [spec], resume=True).run()
    second_result = json.loads(result_path(run_directory(campaign, spec)).read_text())

    assert first_result["status"] == SUCCESS
    assert second_result["status"] == SUCCESS
    assert second_result["verified_score"] == 10


def test_stale_running_becomes_interrupted_and_recoverable(tmp_path):
    instance = write_tiny_instance(tmp_path)
    spec = run_spec("ok_solver", instance, tmp_path)
    run_dir = run_directory(tmp_path / "campaign", spec)
    write_status(run_dir, "RUNNING", {"pid": -1, "run_id": spec.run_id})
    mark_interrupted(run_dir, "stale")
    assert read_status(run_dir)["status"] == INTERRUPTED


def test_seed_derivation_is_deterministic_and_exec_specific():
    a = stable_seed(2026, "solver", "instance", 1)
    b = stable_seed(2026, "solver", "instance", 1)
    c = stable_seed(2026, "solver", "instance", 2)
    assert a == b
    assert a != c


def test_duplicate_run_claim_prevention(tmp_path):
    run_dir = tmp_path / "run"
    first = RunLock(run_dir)
    second = RunLock(run_dir)
    assert first.acquire({"pid": 123})
    assert not second.acquire({"pid": 456})
    first.release()


def test_best_tie_break_score_runtime_execution_index(tmp_path):
    campaign = tmp_path / "campaign"
    spec1 = RunSpec("solver", "inst", "test", "inst.txt", 1, 1, 1)
    spec2 = RunSpec("solver", "inst", "test", "inst.txt", 2, 2, 1)
    for spec, score, runtime in [(spec1, 5, 2.0), (spec2, 5, 1.0)]:
        run_dir = run_directory(campaign, spec)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "solution.txt").write_text("0\n", encoding="utf-8")
        atomic_write_json(
            result_path(run_dir),
            {
                "run_id": spec.run_id,
                "solver_id": spec.solver_id,
                "instance_id": spec.instance_id,
                "execution_index": spec.execution_index,
                "seed": spec.seed,
                "status": SUCCESS,
                "verified_score": score,
                "wall_time_seconds": runtime,
                "solution_path": str(run_dir / "solution.txt"),
            },
        )
    best = update_best_for_pair(campaign, "solver", "inst")
    assert best["best_execution_index"] == 2


def test_manifest_mismatch_detection(tmp_path):
    campaign = tmp_path / "campaign"
    existing = {"benchmark_manifest": {"count": 1}, "solver_manifest": {}, "runs": 10, "root_seed": 1}
    current = {"benchmark_manifest": {"count": 2}, "solver_manifest": {}, "runs": 10, "root_seed": 1}
    campaign.mkdir()
    atomic_write_json(campaign / "campaign_manifest.json", existing)
    with pytest.raises(SystemExit):
        prepare_campaign_dir(campaign, current, resume=True, allow_mismatch=False)


def test_instances_per_family_limit_is_deterministic():
    rows = [
        BenchmarkInstance("type2", "type2_b", "b.txt", "sha", 1, 1, 1, 1),
        BenchmarkInstance("type1", "type1_c", "c.txt", "sha", 1, 1, 1, 1),
        BenchmarkInstance("type2", "type2_a", "a.txt", "sha", 1, 1, 1, 1),
        BenchmarkInstance("type1", "type1_a", "a.txt", "sha", 1, 1, 1, 1),
        BenchmarkInstance("type1", "type1_b", "b.txt", "sha", 1, 1, 1, 1),
    ]
    selected = limit_instances_per_family(rows, 2)
    assert [(row.family, row.instance_id) for row in selected] == [
        ("type1", "type1_a"),
        ("type1", "type1_b"),
        ("type2", "type2_a"),
        ("type2", "type2_b"),
    ]

