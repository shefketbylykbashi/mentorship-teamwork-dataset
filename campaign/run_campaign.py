#!/usr/bin/env python3
"""Fault-tolerant runner for the 10-solver x 611-instance MTP campaign.

Every run:  campaign/bin/<solver> SEED < instance > solution
  * killed at --time-limit (default 600 s) wall clock -> status "timeout" (failed);
  * wall time, CPU time and peak RSS are recorded for every run;
  * every output is scored by the vendored Hash Code referee plus a strict
    start-of-project re-check (campaign/validator/batch_validate.js), using a
    small pool of long-lived Node workers (no per-run node start-up/copying).

Work never stops on an error: a failing run is recorded and logged to
errors.log, and the next run starts. Results are appended (one JSON line per
finished run, fsync'ed) to results.jsonl; restarting the same command resumes
and skips every run already recorded.

Only the Python 3.9+ standard library is used.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import queue
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from instances import REPO, discover  # noqa: E402

CAMPAIGN = REPO / "campaign"
SEEDED = ("refactored_b", "refactored_d", "evolved_b", "evolved_d", "evolved_e")
DETERMINISTIC = ("refactored_c", "refactored_e", "refactored_f", "evolved_c", "evolved_f")
ALL_SOLVERS = tuple(sorted(SEEDED + DETERMINISTIC))
STDERR_TAIL_BYTES = 2000
EXE = ".exe" if os.name == "nt" else ""

STOP = threading.Event()
LIVE: dict[int, subprocess.Popen] = {}
LIVE_LOCK = threading.Lock()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))  # respects SLURM/cgroup pinning
    except AttributeError:
        return os.cpu_count() or 1


def parse_seeds(text: str) -> list[int]:
    seeds: list[int] = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        seeds.extend(range(int(a), int(b or a) + 1))
    return seeds


# --------------------------------------------------------------------------- logging
class Log:
    """errors.log + stderr (-> nohup log) for errors; results.jsonl for data."""

    def __init__(self, out: Path):
        self.lock = threading.Lock()
        self.results = open(out / "results.jsonl", "a", encoding="utf-8")
        self.errors = open(out / "errors.log", "a", encoding="utf-8")

    def result(self, rec: dict) -> None:
        line = json.dumps(rec, sort_keys=True) + "\n"
        with self.lock:
            self.results.write(line)
            self.results.flush()
            os.fsync(self.results.fileno())

    def error(self, msg: str) -> None:
        line = f"{now()} {msg}\n"
        with self.lock:
            self.errors.write(line)
            self.errors.flush()
            sys.stderr.write(line)
            sys.stderr.flush()

    def info(self, msg: str) -> None:
        with self.lock:
            sys.stdout.write(f"{now()} {msg}\n")
            sys.stdout.flush()


# --------------------------------------------------------------------------- validator pool
class Validator:
    def __init__(self, node: str):
        self.node = node
        self.proc: subprocess.Popen | None = None
        self.n = 0

    def _start(self) -> None:
        self.proc = subprocess.Popen(
            [self.node, "--max-old-space-size=4096", str(CAMPAIGN / "validator" / "batch_validate.js"), "--server"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, bufsize=1, start_new_session=True)

    def check(self, instance: Path, solution: Path) -> dict:
        for attempt in (1, 2):  # restart a dead worker once
            try:
                if self.proc is None or self.proc.poll() is not None:
                    self._start()
                self.n += 1
                assert self.proc and self.proc.stdin and self.proc.stdout
                self.proc.stdin.write(json.dumps({"id": self.n, "instance": str(instance), "solution": str(solution)}) + "\n")
                self.proc.stdin.flush()
                line = self.proc.stdout.readline()
                if not line:
                    raise RuntimeError("validator worker exited")
                reply = json.loads(line)
                if reply.get("id") != self.n:
                    raise RuntimeError(f"validator reply id mismatch {reply.get('id')} != {self.n}")
                return reply
            except Exception:
                self.close()
                if attempt == 2:
                    raise
        raise AssertionError("unreachable")

    def close(self) -> None:
        if self.proc is not None:
            try:
                self.proc.kill()
                self.proc.wait(timeout=5)
            except Exception:
                pass
        self.proc = None


class ValidatorPool:
    def __init__(self, node: str, size: int):
        self.q: queue.Queue[Validator] = queue.Queue()
        self.all = [Validator(node) for _ in range(size)]
        for v in self.all:
            self.q.put(v)

    def check(self, instance: Path, solution: Path) -> dict:
        v = self.q.get()
        try:
            return v.check(instance, solution)
        finally:
            self.q.put(v)

    def close(self) -> None:
        for v in self.all:
            v.close()


# --------------------------------------------------------------------------- one run
def run_solver(binary: Path, seed: int, instance: Path, sol: Path, err: Path, limit: float) -> dict:
    """Runs one solver. Returns exit code, wall/cpu time, peak RSS, timeout flag."""
    with open(instance, "rb") as fin, open(sol, "wb") as fout, open(err, "wb") as ferr:
        t0 = time.monotonic()
        proc = subprocess.Popen([str(binary), str(seed)], stdin=fin, stdout=fout, stderr=ferr,
                                start_new_session=True)
        with LIVE_LOCK:
            LIVE[proc.pid] = proc
        timed_out = threading.Event()

        def kill() -> None:
            timed_out.set()
            _kill(proc)

        timer = threading.Timer(limit, kill)
        timer.daemon = True
        timer.start()
        cpu = rss = None
        try:
            if hasattr(os, "wait4"):
                _, status, ru = os.wait4(proc.pid, 0)
                proc.returncode = os.waitstatus_to_exitcode(status)
                cpu = ru.ru_utime + ru.ru_stime
                rss = ru.ru_maxrss  # KiB on Linux
            else:  # Windows (local testing only)
                proc.wait()
        finally:
            wall = time.monotonic() - t0
            timer.cancel()
            with LIVE_LOCK:
                LIVE.pop(proc.pid, None)
    return {"exit_code": 124 if timed_out.is_set() else proc.returncode,
            "signal_exit_code": proc.returncode, "timed_out": timed_out.is_set(),
            "wall_s": round(wall, 4), "cpu_s": None if cpu is None else round(cpu, 4),
            "max_rss_kib": rss}


def _kill(proc: subprocess.Popen) -> None:
    try:
        if os.name == "nt":
            proc.kill()
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def tail(path: Path) -> str:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - STDERR_TAIL_BYTES))
            return fh.read().decode("utf-8", "replace").strip()
    except OSError:
        return ""


def do_run(task: dict, cfg: argparse.Namespace, pool: ValidatorPool, log: Log, host: str) -> None:
    if STOP.is_set():
        return
    solver, fam, iid, seed, inst = task["solver"], task["family"], task["instance"], task["seed"], task["path"]
    key = task["key"]
    sol_dir = cfg.out / "solutions" / solver / fam
    sol_dir.mkdir(parents=True, exist_ok=True)
    sol = sol_dir / f"{iid}__seed{seed}.txt"
    err = cfg.out / "tmp" / f"{solver}__{fam}__{iid}__seed{seed}.stderr"
    rec: dict = {"key": key, "solver": solver, "family": fam, "instance": iid, "seed": seed,
                 "host": host, "started_at": now()}
    try:
        rec.update(run_solver(cfg.bin / (solver + EXE), seed, inst, sol, err, cfg.time_limit))
        if STOP.is_set():  # killed by shutdown, not by the solver: do not record, rerun on resume
            return
        rec["solution_bytes"] = sol.stat().st_size
        if rec["timed_out"]:
            rec["status"] = "timeout"
        elif rec["exit_code"] != 0:
            rec["status"] = "crash"
        else:
            rec["solution_sha256"] = sha256(sol)
            v = pool.check(inst, sol)
            rec.update({"score": v.get("score"), "valid": v.get("valid"), "validator_error": v.get("error"),
                        "strict_valid": v.get("strict_valid"), "strict_score": v.get("strict_score"),
                        "strict_error": v.get("strict_error"), "projects": v.get("projects"),
                        "validate_ms": v.get("ms")})
            if v.get("valid") and v.get("strict_valid") and v.get("score") == v.get("strict_score"):
                rec["status"] = "ok"
            elif v.get("valid") or v.get("strict_valid"):
                rec["status"] = "validator_disagree"
            else:
                rec["status"] = "invalid"
        if rec["status"] != "ok":
            rec["stderr_tail"] = tail(err)
        # keep the solution (gzip) unless asked otherwise; failed outputs are always kept
        if cfg.keep_solutions == "none" and rec["status"] == "ok":
            sol.unlink(missing_ok=True)
        elif sol.exists():
            with open(sol, "rb") as src, gzip.open(str(sol) + ".gz", "wb", compresslevel=6) as dst:
                shutil.copyfileobj(src, dst)
            sol.unlink()
            rec["solution_file"] = str((sol_dir / (sol.name + ".gz")).relative_to(cfg.out))
    except Exception as exc:  # never let one run kill the campaign
        rec["status"] = "runner_error"
        rec["runner_error"] = f"{type(exc).__name__}: {exc}"
        rec["traceback"] = traceback.format_exc(limit=5)
    finally:
        try:
            err.unlink(missing_ok=True)
        except OSError:
            pass
    rec["finished_at"] = now()
    log.result(rec)
    if rec["status"] != "ok":
        detail = rec.get("runner_error") or rec.get("strict_error") or rec.get("validator_error") or rec.get("stderr_tail", "")
        detail = " | ".join(str(detail).splitlines()[-3:])[:600]
        log.error(f"{rec['status'].upper()} {key} exit={rec.get('exit_code')} wall={rec.get('wall_s')}s :: {detail}")


# --------------------------------------------------------------------------- planning
def build_plan(cfg: argparse.Namespace) -> list[dict]:
    insts = discover()
    if cfg.families:
        insts = [x for x in insts if x[0] in cfg.families]
    if cfg.instances_per_family:
        seen: dict[str, int] = {}
        kept = []
        for x in insts:
            seen[x[0]] = seen.get(x[0], 0) + 1
            if seen[x[0]] <= cfg.instances_per_family:
                kept.append(x)
        insts = kept
    # Largest instances first: long runs start early, so the tail of the campaign is short.
    insts.sort(key=lambda x: (-x[2].stat().st_size, x[0], x[1]))
    plan = []
    for fam, iid, path in insts:
        for solver in cfg.solvers:
            for seed in (cfg.seeds if solver in SEEDED else cfg.seeds[:1]):
                plan.append({"key": f"{solver}|{fam}/{iid}|{seed}", "solver": solver, "family": fam,
                             "instance": iid, "seed": seed, "path": path})
    return plan


def finished_keys(out: Path, retry_failed: bool) -> set[str]:
    done: dict[str, str] = {}
    p = out / "results.jsonl"
    if p.exists():
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn last line after a hard kill
                if r.get("status") == "ok" or r.get("key") not in done:
                    done[r["key"]] = r.get("status")
    return {k for k, s in done.items() if s == "ok" or not retry_failed}


def tool_version(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.splitlines()[0]
    except Exception as exc:
        return f"unavailable ({exc})"


def git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO / "campaign_results" / "main")
    ap.add_argument("--bin", type=Path, default=CAMPAIGN / "bin")
    ap.add_argument("--jobs", type=int, default=None, help="parallel solver processes (default: CPUs - validators)")
    ap.add_argument("--validators", type=int, default=None, help="Node validator workers (default: max(1, jobs/16))")
    ap.add_argument("--node", default="node")
    ap.add_argument("--time-limit", type=float, default=600.0, help="hard wall-clock kill per run (s)")
    ap.add_argument("--seeds", default="1-10", help="seeds for seeded solvers; deterministic ones use the first")
    ap.add_argument("--solvers", default=",".join(ALL_SOLVERS))
    ap.add_argument("--families", default="", help="comma list, e.g. google,type4 (default all)")
    ap.add_argument("--instances-per-family", type=int, default=0, help="smoke tests: first N per family")
    ap.add_argument("--keep-solutions", choices=("all", "none"), default="all",
                    help="all: gzip every solution; none: delete valid ones after scoring")
    ap.add_argument("--retry-failed", action="store_true", help="rerun runs whose last status was not ok")
    ap.add_argument("--dry-run", action="store_true")
    cfg = ap.parse_args()

    cfg.out = cfg.out.resolve()
    cfg.bin = cfg.bin.resolve()
    cfg.seeds = parse_seeds(cfg.seeds)
    cfg.solvers = [s for s in cfg.solvers.split(",") if s]
    cfg.families = [f for f in cfg.families.split(",") if f]
    unknown = set(cfg.solvers) - set(ALL_SOLVERS)
    if unknown:
        ap.error(f"unknown solvers: {sorted(unknown)}")
    cpus = available_cpus()
    if cfg.validators is None:
        cfg.validators = max(1, (cfg.jobs or cpus) // 16)
    if cfg.jobs is None:
        cfg.jobs = max(1, cpus - cfg.validators)

    plan = build_plan(cfg)
    done = finished_keys(cfg.out, cfg.retry_failed)
    todo = [t for t in plan if t["key"] not in done]
    n_inst = len({(t["family"], t["instance"]) for t in plan})
    summary = (f"instances={n_inst} solvers={len(cfg.solvers)} planned_runs={len(plan)} "
               f"already_done={len(plan) - len(todo)} to_run={len(todo)} jobs={cfg.jobs} "
               f"validators={cfg.validators} cpus={cpus} time_limit={cfg.time_limit}s")
    if cfg.dry_run:
        print(summary)
        hours = len(todo) * cfg.time_limit / 3600 / cfg.jobs
        print(f"upper bound wall clock at {cfg.jobs} parallel: {hours:.1f} h")
        missing = [s for s in cfg.solvers if not (cfg.bin / (s + EXE)).exists()]
        print("missing binaries:", missing or "none")
        if todo:
            t = todo[0]
            print(f"first run: {cfg.bin / t['solver']} {t['seed']} < {t['path']}")
        return 0

    missing = [s for s in cfg.solvers if not os.access(cfg.bin / (s + EXE), os.X_OK)]
    if missing:
        print(f"ERROR: missing solver binaries in {cfg.bin}: {missing}. Run campaign/build_solvers.sh first.", file=sys.stderr)
        return 2
    if shutil.which(cfg.node) is None:
        print(f"ERROR: node not found ({cfg.node}); install Node >= 16 or pass --node /path/to/node", file=sys.stderr)
        return 2

    cfg.out.mkdir(parents=True, exist_ok=True)
    (cfg.out / "tmp").mkdir(exist_ok=True)
    host = socket.gethostname()
    manifest = {
        "started_at": now(), "host": host, "cpus": cpus, "argv": sys.argv, "git_commit": git_commit(),
        "python": sys.version.split()[0], "node": tool_version([cfg.node, "--version"]),
        "gxx": tool_version(["g++", "--version"]),
        "build_info": json.loads((cfg.bin / "build_info.json").read_text()) if (cfg.bin / "build_info.json").exists() else None,
        "binaries_sha256": {s: sha256(cfg.bin / (s + EXE)) for s in cfg.solvers},
        "validator_sha256": {p.name: sha256(p) for p in sorted((CAMPAIGN / "validator").rglob("*.js"))},
        "seeded": [s for s in cfg.solvers if s in SEEDED], "seeds": cfg.seeds,
        "planned_runs": len(plan), "to_run_this_invocation": len(todo), "jobs": cfg.jobs,
        "validators": cfg.validators, "time_limit_s": cfg.time_limit,
    }
    with open(cfg.out / "manifests.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(manifest, sort_keys=True) + "\n")

    log = Log(cfg.out)
    log.info(f"START {summary} out={cfg.out}")
    pool = ValidatorPool(cfg.node, cfg.validators)

    def on_signal(signum, _frame):
        if not STOP.is_set():
            log.error(f"received signal {signum}: stopping; unfinished runs will be redone on resume")
        STOP.set()
        with LIVE_LOCK:
            for p in list(LIVE.values()):
                _kill(p)

    for sig in (signal.SIGINT, signal.SIGTERM) + ((signal.SIGHUP,) if hasattr(signal, "SIGHUP") else ()):
        try:
            signal.signal(sig, signal.SIG_IGN if sig == getattr(signal, "SIGHUP", None) else on_signal)
        except (ValueError, OSError):
            pass

    t0 = time.monotonic()
    counter = {"n": 0}
    clock = threading.Lock()

    def progress() -> None:
        with clock:
            counter["n"] += 1
            n = counter["n"]
        if n % 50 == 0 or n == len(todo):
            el = time.monotonic() - t0
            eta = el / n * (len(todo) - n)
            prog = {"updated_at": now(), "finished_this_invocation": n, "to_run": len(todo),
                    "elapsed_h": round(el / 3600, 2), "eta_h": round(eta / 3600, 2)}
            tmp = cfg.out / "progress.json.tmp"
            tmp.write_text(json.dumps(prog, indent=1))
            os.replace(tmp, cfg.out / "progress.json")

    def job(task: dict) -> None:
        try:
            do_run(task, cfg, pool, log, host)
        except Exception as exc:  # last-resort guard
            log.error(f"RUNNER_ERROR {task['key']} {type(exc).__name__}: {exc}")
        finally:
            progress()

    # Bounded submission keeps memory flat and lets STOP take effect quickly.
    sem = threading.Semaphore(cfg.jobs * 2)
    with ThreadPoolExecutor(max_workers=cfg.jobs) as ex:
        for task in todo:
            if STOP.is_set():
                break
            sem.acquire()
            fut = ex.submit(job, task)
            fut.add_done_callback(lambda _f: sem.release())
    pool.close()
    log.info(f"{'STOPPED' if STOP.is_set() else 'FINISHED'} after {(time.monotonic() - t0) / 3600:.2f} h; "
             f"see {cfg.out / 'results.jsonl'} and errors.log; summarize with campaign/summarize.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
