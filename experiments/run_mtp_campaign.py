"""Fault-tolerant runner for MTP solver campaigns."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

from benchmark_generation.core.config import load_config

from .benchmark_discovery import benchmark_manifest, discover_final_benchmark
from .common import (
    atomic_write_json,
    config_hash,
    hostname,
    make_relative,
    now_utc,
    read_json,
    repo_root,
    stable_seed,
)
from .evaluator_adapter import evaluate_solution_file
from .reporting import rebuild_reports, update_best_for_pair
from .solver_adapters import audit_solver_sources, load_solver_definitions, solver_manifest
from .state import (
    ERROR,
    INTERRUPTED,
    INVALID_OUTPUT,
    INVALID_SOLUTION,
    RUNNING,
    SUCCESS,
    TIMEOUT,
    TERMINAL_STATES,
    RunLock,
    RunSpec,
    mark_interrupted,
    read_status,
    recover_stale_running,
    result_path,
    run_directory,
    solution_path,
    stderr_log_path,
    stdout_log_path,
    write_status,
)


ACTIVE_PROCESSES: dict[str, subprocess.Popen] = {}
ACTIVE_LOCK = threading.Lock()
STOP_EVENT = threading.Event()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    config = load_config(args.config, defaults={})
    apply_overrides(config, args)

    campaign_id = args.campaign_id
    if not campaign_id:
        raise SystemExit("--campaign-id is required")
    campaign_dir = root / str(config.get("results_root", "experiment_results")) / campaign_id

    if args.rebuild_reports:
        summary = rebuild_reports(campaign_dir)
        print(f"Rebuilt reports for {campaign_dir}")
        print(summary)
        return 0

    instances = discover_final_benchmark(root, config.get("benchmark", {}))
    solvers = load_solver_definitions(config)
    bench_manifest = benchmark_manifest(instances)
    solv_manifest = solver_manifest(solvers, root)
    audit = audit_solver_sources(solvers, root)
    runs = int(config.get("runs", 10))
    root_seed = int(config.get("root_seed", 2026))
    run_plan = make_run_plan(instances, solvers, runs, root_seed)
    planned_run_count = len(run_plan)
    current_manifest = {
        "campaign_id": campaign_id,
        "created_at": now_utc(),
        "config_hash": config_hash(config),
        "config": config,
        "root_seed": root_seed,
        "runs": runs,
        "time_limit_seconds": float(config.get("time_limit_seconds", 180)),
        "timeout_grace_seconds": float(config.get("timeout_grace_seconds", 15)),
        "threads": int(config.get("threads", 1)),
        "planned_run_count": planned_run_count,
        "benchmark_manifest": bench_manifest,
        "solver_manifest": solv_manifest,
        "solver_audit": audit,
    }

    if args.dry_run:
        print_dry_run(current_manifest, run_plan, solvers, root)
        return 0

    validate_binaries(solv_manifest)
    prepare_campaign_dir(campaign_dir, current_manifest, resume=args.resume, allow_mismatch=args.allow_version_mismatch)
    if args.resume:
        recovered = recover_stale_running(campaign_dir)
        if recovered:
            print(f"Recovered {recovered} stale RUNNING run(s) as INTERRUPTED.")

    install_signal_handlers()
    runner = CampaignRunner(root, campaign_dir, config, solvers, run_plan, resume=args.resume)
    try:
        runner.run()
    finally:
        rebuild_reports(campaign_dir)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/configs/mtp_solver_campaign.yaml")
    parser.add_argument("--campaign-id")
    parser.add_argument("--runs", type=int)
    parser.add_argument("--time-limit", type=float)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--timeout-grace", type=float)
    parser.add_argument("--results-dir", dest="results_root")
    parser.add_argument("--max-parallel-runs", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rebuild-reports", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--retry-timeouts", action="store_true")
    parser.add_argument("--retry-invalid", action="store_true")
    parser.add_argument("--allow-version-mismatch", action="store_true")
    return parser.parse_args(argv)


def apply_overrides(config: dict, args: argparse.Namespace) -> None:
    if args.runs is not None:
        config["runs"] = args.runs
    if args.time_limit is not None:
        config["time_limit_seconds"] = args.time_limit
    if args.threads is not None:
        config["threads"] = args.threads
    if args.timeout_grace is not None:
        config["timeout_grace_seconds"] = args.timeout_grace
    if args.results_root is not None:
        config["results_root"] = args.results_root
    if args.max_parallel_runs is not None:
        config["max_parallel_runs"] = args.max_parallel_runs
    if args.retry_errors:
        config["retry_errors"] = True
    if args.retry_timeouts:
        config["retry_timeouts"] = True
    if args.retry_invalid:
        config["retry_invalid"] = True


def make_run_plan(instances, solvers, runs: int, root_seed: int) -> List[RunSpec]:
    specs: List[RunSpec] = []
    for solver in sorted(solvers, key=lambda item: item.solver_id):
        for instance in instances:
            for execution_index in range(1, runs + 1):
                specs.append(
                    RunSpec(
                        solver_id=solver.solver_id,
                        instance_id=instance.instance_id,
                        family=instance.family,
                        instance_path=instance.path,
                        execution_index=execution_index,
                        seed=stable_seed(root_seed, solver.solver_id, instance.instance_id, execution_index),
                        cpu_slots=solver.cpu_slots_per_run,
                    )
                )
    return specs


def print_dry_run(manifest: dict, run_plan: List[RunSpec], solvers, root: Path) -> None:
    missing = [
        row for row in manifest["solver_manifest"]["solvers"] if not row.get("binary_exists")
    ]
    print("DRY RUN: no solver processes will be executed.")
    print(f"Campaign ID: {manifest['campaign_id']}")
    print(f"Benchmark instances: {manifest['benchmark_manifest']['count']}")
    print(f"Family counts: {manifest['benchmark_manifest']['family_counts']}")
    print(f"Solvers: {manifest['solver_manifest']['count']}")
    print(f"Runs per solver-instance: {manifest['runs']}")
    print(f"Total planned executions: {manifest['planned_run_count']}")
    print(f"Threads / total CPU slots: {manifest['threads']}")
    max_parallel = _max_parallel(manifest["config"], solvers)
    print(f"Max parallel solver processes: {max_parallel}")
    print(f"Missing binaries: {len(missing)}")
    for row in missing[:10]:
        print(f"  missing: {row['solver_id']} -> {row['binary']}")
    if run_plan:
        spec = run_plan[0]
        solver = {s.solver_id: s for s in solvers}[spec.solver_id]
        command = solver.command_for(root, spec.seed, manifest["time_limit_seconds"])
        print(f"Example run ID: {spec.run_id}")
        print(f"Example command: {' '.join(command)} < {spec.instance_path} > solution.txt")


def validate_binaries(solv_manifest: dict) -> None:
    missing = [row for row in solv_manifest["solvers"] if not row.get("binary_exists")]
    if missing:
        ids = ", ".join(row["solver_id"] for row in missing)
        raise SystemExit(f"Missing solver binaries. Run python -m experiments.build_solvers first. Missing: {ids}")


def prepare_campaign_dir(
    campaign_dir: Path,
    manifest: dict,
    resume: bool,
    allow_mismatch: bool,
) -> None:
    manifest_path = campaign_dir / "campaign_manifest.json"
    if manifest_path.exists():
        existing = read_json(manifest_path)
        if not resume:
            raise SystemExit(f"Campaign already exists; use --resume or a new --campaign-id: {campaign_dir}")
        if not allow_mismatch:
            _assert_manifest_compatible(existing, manifest)
    else:
        campaign_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(manifest_path, manifest)
        atomic_write_json(campaign_dir / "benchmark_manifest.json", manifest["benchmark_manifest"])
        atomic_write_json(campaign_dir / "solver_manifest.json", manifest["solver_manifest"])


def _assert_manifest_compatible(existing: dict, current: dict) -> None:
    checks = (
        ("benchmark_manifest", "benchmark manifest"),
        ("solver_manifest", "solver manifest"),
        ("runs", "runs"),
        ("root_seed", "root seed"),
    )
    for key, label in checks:
        if existing.get(key) != current.get(key):
            raise SystemExit(
                f"Campaign {label} differs from current configuration. "
                "Use a new --campaign-id or --allow-version-mismatch."
            )


class CampaignRunner:
    def __init__(
        self,
        root: Path,
        campaign_dir: Path,
        config: dict,
        solvers,
        run_plan: List[RunSpec],
        resume: bool,
    ):
        self.root = root
        self.campaign_dir = campaign_dir
        self.config = config
        self.solvers = {solver.solver_id: solver for solver in solvers}
        self.run_plan = run_plan
        self.resume = resume
        self.total_slots = int(config.get("threads", 1))
        self.max_parallel_runs = _max_parallel(config, solvers)
        self.time_limit = float(config.get("time_limit_seconds", 180))
        self.timeout_grace = float(config.get("timeout_grace_seconds", 15))

    def run(self) -> None:
        pending = [spec for spec in self.run_plan if self._should_run(spec)]
        print(f"Planned executions: {len(self.run_plan)}")
        print(f"Pending executions for this invocation: {len(pending)}")
        if not pending:
            return

        available_slots = self.total_slots
        running: dict[concurrent.futures.Future, RunSpec] = {}
        completed_since_report = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallel_runs) as executor:
            while (pending or running) and not STOP_EVENT.is_set():
                launched = False
                for spec in list(pending):
                    if len(running) >= self.max_parallel_runs:
                        break
                    if spec.cpu_slots > available_slots:
                        continue
                    pending.remove(spec)
                    available_slots -= spec.cpu_slots
                    future = executor.submit(self._run_one_safely, spec)
                    running[future] = spec
                    launched = True
                if not running:
                    break
                done, _ = concurrent.futures.wait(
                    running,
                    timeout=1.0 if launched else 5.0,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    spec = running.pop(future)
                    available_slots += spec.cpu_slots
                    try:
                        future.result()
                    except Exception as exc:
                        self._write_unhandled_error(spec, exc)
                    completed_since_report += 1
                if completed_since_report >= 25:
                    self._print_progress()
                    completed_since_report = 0

            if STOP_EVENT.is_set():
                for spec in running.values():
                    mark_interrupted(run_directory(self.campaign_dir, spec), "runner_interrupted")
                terminate_active_processes()
                print("Interruption requested; stopped scheduling new runs.")

    def _should_run(self, spec: RunSpec) -> bool:
        run_dir = run_directory(self.campaign_dir, spec)
        status = read_status(run_dir)
        if not status:
            return True
        state = status.get("status")
        if state == SUCCESS:
            return not self._success_is_valid(spec)
        if state == RUNNING:
            mark_interrupted(run_dir, "running_state_recovered_before_run")
            return True
        if state == INTERRUPTED:
            return True
        if state == ERROR:
            return bool(self.config.get("retry_errors", False))
        if state == TIMEOUT:
            return bool(self.config.get("retry_timeouts", False))
        if state in {INVALID_OUTPUT, INVALID_SOLUTION}:
            return bool(self.config.get("retry_invalid", False))
        if state in TERMINAL_STATES:
            return False
        return True

    def _success_is_valid(self, spec: RunSpec) -> bool:
        run_dir = run_directory(self.campaign_dir, spec)
        result = read_json(result_path(run_dir), default={}) or {}
        sol_path = solution_path(run_dir)
        if result.get("status") != SUCCESS or not sol_path.exists():
            write_status(run_dir, INVALID_OUTPUT, {**(read_status(run_dir) or {}), "reason": "corrupt_success"})
            return False
        evaluation = evaluate_solution_file(self.root / spec.instance_path, sol_path)
        if not evaluation.valid_solution or result.get("verified_score") != evaluation.verified_score:
            write_status(run_dir, INVALID_OUTPUT, {**(read_status(run_dir) or {}), "reason": "success_revalidation_failed"})
            return False
        return True

    def _run_one_safely(self, spec: RunSpec) -> None:
        try:
            self._run_one(spec)
        except Exception as exc:
            self._write_unhandled_error(spec, exc)

    def _run_one(self, spec: RunSpec) -> None:
        solver = self.solvers[spec.solver_id]
        run_dir = run_directory(self.campaign_dir, spec)
        run_dir.mkdir(parents=True, exist_ok=True)
        lock = RunLock(run_dir, stale_ok=True)
        if not lock.acquire({"run_id": spec.run_id, "solver_id": spec.solver_id, "instance_id": spec.instance_id}):
            return
        try:
            command = solver.command_for(self.root, spec.seed, self.time_limit)
            status_payload = self._base_payload(spec, command)
            status_payload.update(
                {
                    "status": RUNNING,
                    "start_time": now_utc(),
                    "hostname": hostname(),
                    "pid": None,
                    "allocated_cpu_slots": spec.cpu_slots,
                }
            )
            write_status(run_dir, RUNNING, status_payload)

            result = self._execute_process(spec, command)
            atomic_write_json(result_path(run_dir), result)
            write_status(run_dir, result["status"], result)
            if result["status"] == SUCCESS:
                update_best_for_pair(self.campaign_dir, spec.solver_id, spec.instance_id)
        finally:
            lock.release()

    def _execute_process(self, spec: RunSpec, command: List[str]) -> dict:
        run_dir = run_directory(self.campaign_dir, spec)
        instance_path = self.root / spec.instance_path
        sol_path = solution_path(run_dir)
        out_log = stdout_log_path(run_dir)
        err_log = stderr_log_path(run_dir)
        env = os.environ.copy()
        env.setdefault("OMP_NUM_THREADS", str(max(1, spec.cpu_slots)))
        env.setdefault("MKL_NUM_THREADS", str(max(1, spec.cpu_slots)))
        env.setdefault("OPENBLAS_NUM_THREADS", str(max(1, spec.cpu_slots)))
        env.setdefault("NUMEXPR_NUM_THREADS", str(max(1, spec.cpu_slots)))

        start = time.monotonic()
        start_time = now_utc()
        exit_code = None
        termination_reason = "completed"
        status = ERROR
        evaluation = None
        exception_info = None
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        with instance_path.open("rb") as stdin_fh, sol_path.open("wb") as stdout_fh, err_log.open("wb") as stderr_fh:
            proc = subprocess.Popen(
                command,
                cwd=self.root,
                stdin=stdin_fh,
                stdout=stdout_fh,
                stderr=stderr_fh,
                env=env,
                start_new_session=(os.name != "nt"),
                creationflags=creationflags,
            )
            with ACTIVE_LOCK:
                ACTIVE_PROCESSES[spec.run_id] = proc
            self._update_running_pid(spec, proc.pid)
            try:
                exit_code = proc.wait(timeout=self.time_limit + self.timeout_grace)
            except subprocess.TimeoutExpired:
                termination_reason = "external_timeout"
                status = TIMEOUT
                terminate_process_tree(proc)
                try:
                    exit_code = proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    exit_code = proc.wait(timeout=10)
            finally:
                with ACTIVE_LOCK:
                    ACTIVE_PROCESSES.pop(spec.run_id, None)

        wall_time = time.monotonic() - start
        shutil.copyfile(sol_path, out_log)
        if STOP_EVENT.is_set() and status != TIMEOUT:
            status = INTERRUPTED
            termination_reason = "runner_interrupted"
        elif status != TIMEOUT:
            if exit_code != 0:
                status = ERROR
                termination_reason = "nonzero_exit"
            else:
                try:
                    evaluation = evaluate_solution_file(instance_path, sol_path)
                    if not evaluation.parse_ok:
                        status = INVALID_OUTPUT
                        termination_reason = "malformed_output"
                    elif not evaluation.valid_solution:
                        status = INVALID_SOLUTION
                        termination_reason = "invalid_solution"
                    else:
                        status = SUCCESS
                        termination_reason = "success"
                except Exception as exc:
                    status = ERROR
                    termination_reason = "runner_evaluation_exception"
                    exception_info = {
                        "exception_class": exc.__class__.__name__,
                        "exception_message": str(exc),
                        "traceback": traceback.format_exc(),
                    }

        result = self._base_payload(spec, command)
        result.update(
            {
                "status": status,
                "start_time": start_time,
                "finish_time": now_utc(),
                "wall_time_seconds": wall_time,
                "exit_code": exit_code,
                "termination_reason": termination_reason,
                "solution_path": make_relative(sol_path, self.campaign_dir),
                "stdout_log_path": make_relative(out_log, self.campaign_dir),
                "stderr_log_path": make_relative(err_log, self.campaign_dir),
                "verified_score": evaluation.verified_score if evaluation else None,
                "output_validation": evaluation.to_dict() if evaluation else None,
                "exception": exception_info,
            }
        )
        return result

    def _base_payload(self, spec: RunSpec, command: List[str]) -> dict:
        return {
            "run_id": spec.run_id,
            "solver_id": spec.solver_id,
            "instance_id": spec.instance_id,
            "family": spec.family,
            "instance_path": spec.instance_path,
            "execution_index": spec.execution_index,
            "seed": spec.seed,
            "command": command,
            "working_directory": str(self.root),
            "time_limit_seconds": self.time_limit,
            "timeout_grace_seconds": self.timeout_grace,
            "allocated_cpu_slots": spec.cpu_slots,
        }

    def _update_running_pid(self, spec: RunSpec, pid: int) -> None:
        run_dir = run_directory(self.campaign_dir, spec)
        status = read_status(run_dir) or {}
        status["pid"] = pid
        write_status(run_dir, RUNNING, status)

    def _write_unhandled_error(self, spec: RunSpec, exc: Exception | None = None) -> None:
        run_dir = run_directory(self.campaign_dir, spec)
        run_dir.mkdir(parents=True, exist_ok=True)
        result = self._base_payload(spec, [])
        result.update(
            {
                "status": ERROR,
                "finish_time": now_utc(),
                "termination_reason": "runner_unhandled_exception",
                "exit_code": None,
                "verified_score": None,
                "exception": {
                    "exception_class": exc.__class__.__name__ if exc else "FutureException",
                    "exception_message": str(exc) if exc else "future.result raised",
                    "traceback": traceback.format_exc(),
                },
            }
        )
        atomic_write_json(result_path(run_dir), result)
        write_status(run_dir, ERROR, result)

    def _print_progress(self) -> None:
        summary = rebuild_reports(self.campaign_dir)
        print(
            "Completed: "
            f"{summary.get('recorded_runs')} / {len(self.run_plan)} | "
            f"Success: {summary.get('successful_runs')} | "
            f"Errors: {summary.get('error_runs')} | "
            f"Timeouts: {summary.get('timeout_runs')} | "
            f"Invalid: {summary.get('invalid_output_runs', 0) + summary.get('invalid_solution_runs', 0)}"
        )


def _max_parallel(config: dict, solvers) -> int:
    manual = config.get("max_parallel_runs")
    threads = max(1, int(config.get("threads", 1)))
    if manual is not None:
        return max(1, min(int(manual), threads))
    min_slots = min(max(1, solver.cpu_slots_per_run) for solver in solvers) if solvers else 1
    return max(1, threads // min_slots)


def terminate_process_tree(proc: subprocess.Popen) -> None:
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        proc.kill()


def terminate_active_processes() -> None:
    with ACTIVE_LOCK:
        processes = list(ACTIVE_PROCESSES.values())
    for proc in processes:
        terminate_process_tree(proc)


def install_signal_handlers() -> None:
    def handler(_signum, _frame):
        STOP_EVENT.set()
        terminate_active_processes()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, handler)
        except Exception:
            pass


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

