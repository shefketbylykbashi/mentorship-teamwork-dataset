"""Compile all configured C++ MTP solvers once for a campaign."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from benchmark_generation.core.config import load_config

from .common import atomic_write_json, now_utc, repo_root, sha256_file
from .solver_adapters import audit_solver_sources, load_solver_definitions


DEFAULT_BUILD = {
    "compiler": "g++",
    "cxx_standard": "c++20",
    "optimization_flags": ["-O3"],
    "extra_flags": [],
    "output_manifest": "experiments/solver_manifest.json",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/configs/mtp_solver_campaign.yaml")
    parser.add_argument("--compiler")
    parser.add_argument("--cxx-standard")
    parser.add_argument("--output-manifest")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    root = repo_root()
    config = load_config(args.config, defaults={})
    build_cfg = dict(DEFAULT_BUILD)
    build_cfg.update(config.get("build", {}))
    if args.compiler:
        build_cfg["compiler"] = args.compiler
    if args.cxx_standard:
        build_cfg["cxx_standard"] = args.cxx_standard
    if args.output_manifest:
        build_cfg["output_manifest"] = args.output_manifest

    solvers = load_solver_definitions(config)
    compiler_version = _compiler_version(str(build_cfg["compiler"]))
    rows = []
    failures = []
    for solver in solvers:
        source = solver.source_path(root)
        binary = solver.binary_path(root)
        binary.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(build_cfg["compiler"]),
            f"-std={build_cfg['cxx_standard']}",
            *list(build_cfg.get("optimization_flags", [])),
            *list(build_cfg.get("extra_flags", [])),
            str(source),
            "-o",
            str(binary),
        ]
        if args.verbose:
            print(" ".join(command))
        started_at = now_utc()
        try:
            completed = subprocess.run(command, cwd=root, text=True, capture_output=True)
            exit_code = completed.returncode
            stdout = completed.stdout
            stderr = completed.stderr
        except FileNotFoundError as exc:
            exit_code = 127
            stdout = ""
            stderr = str(exc)
        row = {
            "solver_id": solver.solver_id,
            "source": str(source.relative_to(root)),
            "binary": str(binary.relative_to(root)),
            "command": command,
            "compiler": build_cfg["compiler"],
            "compiler_version": compiler_version,
            "cxx_standard": build_cfg["cxx_standard"],
            "optimization_flags": build_cfg.get("optimization_flags", []),
            "extra_flags": build_cfg.get("extra_flags", []),
            "started_at": started_at,
            "finished_at": now_utc(),
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
            "source_sha256": sha256_file(source) if source.exists() else None,
            "binary_sha256": sha256_file(binary) if binary.exists() and exit_code == 0 else None,
            "status": "SUCCESS" if exit_code == 0 else "ERROR",
        }
        rows.append(row)
        if exit_code != 0:
            failures.append(solver.solver_id)

    manifest = {
        "created_at": now_utc(),
        "build": build_cfg,
        "compiler_version": compiler_version,
        "solvers": rows,
        "audit": audit_solver_sources(solvers, root),
    }
    manifest_path = root / str(build_cfg["output_manifest"])
    atomic_write_json(manifest_path, manifest)
    print(f"Wrote solver build manifest: {manifest_path}")
    if failures:
        print(f"Build failed for {len(failures)} solver(s): {', '.join(failures)}", file=sys.stderr)
        return 1
    print(f"Built {len(rows)} solver(s).")
    return 0


def _compiler_version(compiler: str) -> str:
    try:
        completed = subprocess.run(
            [compiler, "--version"],
            text=True,
            capture_output=True,
            timeout=10,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        return f"unavailable: {exc}"
    output = (completed.stdout or completed.stderr).strip().splitlines()
    return output[0] if output else "unknown"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

