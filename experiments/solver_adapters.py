"""Solver definitions, command construction, and static source audit."""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Mapping

from .common import make_relative, sha256_file


@dataclass(frozen=True)
class SolverDefinition:
    solver_id: str
    source: str
    binary: str
    stochastic: bool
    seed_arg: str | None = "--seed"
    time_limit_arg: str | None = "--time-limit"
    threads_arg: str | None = None
    cpu_slots_per_run: int = 1
    objective: str = "maximize"
    command: tuple[str, ...] | None = None

    def binary_path(self, repo_root: str | Path) -> Path:
        binary = _format_binary(self.binary)
        return (Path(repo_root) / binary).resolve()

    def source_path(self, repo_root: str | Path) -> Path:
        return (Path(repo_root) / self.source).resolve()

    def command_for(
        self,
        repo_root: str | Path,
        seed: int,
        time_limit_seconds: float,
        internal_threads: int | None = None,
    ) -> List[str]:
        if self.command:
            cmd = [_command_part(part, repo_root, idx == 0) for idx, part in enumerate(self.command)]
        else:
            cmd = [str(self.binary_path(repo_root))]
        if self.seed_arg:
            cmd.extend([self.seed_arg, str(int(seed))])
        if self.time_limit_arg:
            cmd.extend([self.time_limit_arg, str(float(time_limit_seconds))])
        if self.threads_arg:
            cmd.extend([self.threads_arg, str(int(internal_threads or self.cpu_slots_per_run))])
        return cmd

    def to_manifest_row(self, repo_root: str | Path) -> dict:
        src = self.source_path(repo_root)
        bin_path = self.binary_path(repo_root)
        row = asdict(self)
        row["source_sha256"] = sha256_file(src) if src.exists() else None
        row["binary_sha256"] = sha256_file(bin_path) if bin_path.exists() else None
        row["binary_exists"] = bin_path.exists()
        row["source"] = make_relative(src, repo_root)
        row["binary"] = make_relative(bin_path, repo_root)
        row["single_or_multithreaded"] = (
            "single-threaded" if self.cpu_slots_per_run == 1 else "internally-multithreaded"
        )
        return row


DEFAULT_SOLVERS = (
    {
        "solver_id": "nanored_original_ab",
        "source": "solvers/Nanored_hashCode2022/ab.cpp",
        "binary": "experiments/bin/nanored_original_ab{exe}",
        "stochastic": True,
    },
    {
        "solver_id": "nanored_original_c",
        "source": "solvers/Nanored_hashCode2022/c.cpp",
        "binary": "experiments/bin/nanored_original_c{exe}",
        "stochastic": True,
    },
    {
        "solver_id": "nanored_original_d",
        "source": "solvers/Nanored_hashCode2022/d.cpp",
        "binary": "experiments/bin/nanored_original_d{exe}",
        "stochastic": True,
    },
    {
        "solver_id": "nanored_original_e",
        "source": "solvers/Nanored_hashCode2022/e.cpp",
        "binary": "experiments/bin/nanored_original_e{exe}",
        "stochastic": False,
    },
    {
        "solver_id": "nanored_original_f",
        "source": "solvers/Nanored_hashCode2022/f.cpp",
        "binary": "experiments/bin/nanored_original_f{exe}",
        "stochastic": False,
    },
    {
        "solver_id": "nanored_refactored_b",
        "source": "solvers/refactored_Nanored/b.cpp",
        "binary": "experiments/bin/nanored_refactored_b{exe}",
        "stochastic": True,
    },
    {
        "solver_id": "nanored_refactored_c",
        "source": "solvers/refactored_Nanored/c.cpp",
        "binary": "experiments/bin/nanored_refactored_c{exe}",
        "stochastic": False,
    },
    {
        "solver_id": "nanored_refactored_d",
        "source": "solvers/refactored_Nanored/d.cpp",
        "binary": "experiments/bin/nanored_refactored_d{exe}",
        "stochastic": True,
    },
    {
        "solver_id": "nanored_refactored_e",
        "source": "solvers/refactored_Nanored/e.cpp",
        "binary": "experiments/bin/nanored_refactored_e{exe}",
        "stochastic": False,
    },
    {
        "solver_id": "nanored_refactored_f",
        "source": "solvers/refactored_Nanored/f.cpp",
        "binary": "experiments/bin/nanored_refactored_f{exe}",
        "stochastic": False,
    },
)


def load_solver_definitions(config: Mapping[str, object] | None = None) -> List[SolverDefinition]:
    cfg = dict(config or {})
    solver_rows = cfg.get("solvers", DEFAULT_SOLVERS)
    return [SolverDefinition(**_solver_defaults(dict(row))) for row in solver_rows]  # type: ignore[arg-type]


def solver_manifest(solvers: Iterable[SolverDefinition], repo_root: str | Path) -> dict:
    rows = [solver.to_manifest_row(repo_root) for solver in solvers]
    return {"count": len(rows), "solvers": rows}


def audit_solver_sources(solvers: Iterable[SolverDefinition], repo_root: str | Path) -> List[dict]:
    rows = []
    for solver in solvers:
        source_path = solver.source_path(repo_root)
        text = source_path.read_text(encoding="utf-8")
        rows.append(
            {
                "solver_id": solver.solver_id,
                "source_file": make_relative(source_path, repo_root),
                "stochastic_or_deterministic": "stochastic" if solver.stochastic else "deterministic",
                "current_random_seed_behavior": _seed_behavior(text),
                "current_stop_condition": _stop_condition(text),
                "current_hardcoded_runtime_if_any": _hardcoded_runtime(text),
                "fixed_iteration_count_if_any": _fixed_iteration(text),
                "skill_storage_representation": _skill_storage(text),
                "hardcoded_skill_capacity_if_any": _hardcoded_skill_capacity(text),
                "single_or_multithreaded": "single-threaded",
                "input_format": "Hash Code MTP instance on stdin",
                "output_format": "Hash Code MTP solution on stdout",
                "stderr_behavior": "diagnostic/progress output only",
                "memory_considerations": "dynamic skill matrix sized to parsed skill universe",
                "compiler_build_requirements": "C++20, optimized release build",
                "cpu_slots_per_run": solver.cpu_slots_per_run,
            }
        )
    return rows


def _solver_defaults(row: dict) -> dict:
    row.setdefault("seed_arg", "--seed")
    row.setdefault("time_limit_arg", "--time-limit")
    row.setdefault("threads_arg", None)
    row.setdefault("cpu_slots_per_run", 1)
    row.setdefault("objective", "maximize")
    if row.get("command") is not None:
        row["command"] = tuple(row["command"])
    return row


def _format_binary(binary: str) -> str:
    return binary.format(exe=".exe" if os.name == "nt" else "")


def _command_part(part: str, repo_root: str | Path, is_executable: bool) -> str:
    path = Path(part)
    if path.is_absolute():
        return str(path)
    if is_executable and ("/" in part or "\\" in part):
        return str((Path(repo_root) / path).resolve())
    return part


def _seed_behavior(text: str) -> str:
    if "parse_solver_options" in text and "--seed" in text:
        return "external --seed initializes solver RNG where stochastic"
    if "mt19937 mt(42)" in text:
        return "hardcoded mt19937 seed 42"
    if re.search(r"mt19937\s+\w+\s*;", text):
        return "default-constructed deterministic mt19937"
    if "mt19937" in text:
        return "uses mt19937"
    return "no RNG detected"


def _stop_condition(text: str) -> str:
    if "time_limit_reached" in text:
        return "checks --time-limit with steady_clock"
    if "600.0" in text:
        return "hardcoded 600-second wall clock"
    if "int T = 10000" in text:
        return "fixed 10000 randomized restarts"
    return "algorithmic exhaustion/no explicit time limit"


def _hardcoded_runtime(text: str) -> str | None:
    return "600.0 seconds" if "600.0" in text else None


def _fixed_iteration(text: str) -> str | None:
    return "10000" if "int T = 10000" in text else None


def _skill_storage(text: str) -> str:
    if "vector<vector<int>> skill" in text:
        return "dynamic vector<vector<int>>"
    if "array<int" in text:
        return "fixed std::array skill matrix"
    return "not detected"


def _hardcoded_skill_capacity(text: str) -> str | None:
    match = re.search(r"const\s+int\s+NS\s*=\s*(\d+)", text)
    if match:
        return match.group(1)
    if "array<int, 800>" in text:
        return "800"
    return None

