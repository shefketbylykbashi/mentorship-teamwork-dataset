"""Run state files, locks, and resume helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import atomic_write_json, hostname, now_utc, read_json, safe_id


PENDING = "PENDING"
RUNNING = "RUNNING"
SUCCESS = "SUCCESS"
ERROR = "ERROR"
TIMEOUT = "TIMEOUT"
INVALID_OUTPUT = "INVALID_OUTPUT"
INVALID_SOLUTION = "INVALID_SOLUTION"
INTERRUPTED = "INTERRUPTED"

TERMINAL_STATES = {SUCCESS, ERROR, TIMEOUT, INVALID_OUTPUT, INVALID_SOLUTION}
ALL_STATES = TERMINAL_STATES | {PENDING, RUNNING, INTERRUPTED}


@dataclass(frozen=True)
class RunSpec:
    solver_id: str
    instance_id: str
    family: str
    instance_path: str
    execution_index: int
    seed: int
    cpu_slots: int

    @property
    def run_id(self) -> str:
        return run_id(self.solver_id, self.instance_id, self.execution_index)


def run_id(solver_id: str, instance_id: str, execution_index: int) -> str:
    return f"{safe_id(solver_id)}__{safe_id(instance_id)}__exec_{execution_index:02d}"


def run_directory(campaign_dir: str | Path, spec: RunSpec) -> Path:
    return (
        Path(campaign_dir)
        / "runs"
        / safe_id(spec.solver_id)
        / safe_id(spec.instance_id)
        / f"exec_{spec.execution_index:02d}"
    )


def status_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "status.json"


def result_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "result.json"


def solution_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "solution.txt"


def stdout_log_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "stdout.log"


def stderr_log_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "stderr.log"


def lock_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "claim.lock"


def read_status(run_dir: str | Path) -> dict[str, Any] | None:
    return read_json(status_path(run_dir))


def write_status(run_dir: str | Path, status: str, payload: dict[str, Any]) -> None:
    data = dict(payload)
    data["status"] = status
    data.setdefault("updated_at", now_utc())
    atomic_write_json(status_path(run_dir), data)


def mark_interrupted(run_dir: str | Path, reason: str) -> None:
    existing = read_status(run_dir) or {}
    existing.update(
        {
            "status": INTERRUPTED,
            "finish_time": now_utc(),
            "termination_reason": reason,
        }
    )
    atomic_write_json(status_path(run_dir), existing)


class RunLock:
    """Atomic per-run claim lock."""

    def __init__(self, run_dir: str | Path, stale_ok: bool = False):
        self.run_dir = Path(run_dir)
        self.path = lock_path(run_dir)
        self.stale_ok = stale_ok
        self.acquired = False

    def acquire(self, metadata: dict[str, Any]) -> bool:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and self.stale_ok and _lock_is_stale(self.path):
            self.path.unlink(missing_ok=True)
        payload = dict(metadata)
        payload.setdefault("hostname", hostname())
        payload.setdefault("pid", os.getpid())
        payload.setdefault("created_at", now_utc())
        try:
            fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            import json

            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        self.acquired = True
        return True

    def release(self) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)
            self.acquired = False

    def __enter__(self) -> "RunLock":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.release()


def recover_stale_running(campaign_dir: str | Path) -> int:
    recovered = 0
    for status_file in Path(campaign_dir).glob("runs/*/*/exec_*/status.json"):
        status = read_json(status_file)
        if not status or status.get("status") != RUNNING:
            continue
        pid = status.get("pid")
        if pid is None or not process_is_alive(int(pid)):
            mark_interrupted(status_file.parent, "stale_running_recovered_on_resume")
            lock_path(status_file.parent).unlink(missing_ok=True)
            recovered += 1
    return recovered


def process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _lock_is_stale(path: Path) -> bool:
    data = read_json(path, default={}) or {}
    pid = data.get("pid")
    return pid is None or not process_is_alive(int(pid))

