"""Shared utilities for campaign execution, manifests, and reporting."""

from __future__ import annotations

import hashlib
import json
import os
import re
import socket
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def hostname() -> str:
    return socket.gethostname()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def config_hash(config: Mapping[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return sha256_text(payload)


def stable_seed(root_seed: int, solver_id: str, instance_id: str, execution_index: int) -> int:
    payload = f"{int(root_seed)}\0{solver_id}\0{instance_id}\0{int(execution_index)}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:4], "big", signed=False)
    return seed if seed != 0 else 1


def safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._") or "unnamed"


def make_relative(path: str | Path, base: str | Path | None = None) -> str:
    path = Path(path)
    base_path = Path(base) if base is not None else repo_root()
    try:
        return str(path.resolve().relative_to(base_path.resolve()))
    except ValueError:
        return str(path)


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    finally:
        tmp = Path(tmp_name)
        if tmp.exists():
            tmp.unlink()


def atomic_write_json(path: str | Path, data: Any) -> None:
    atomic_write_text(path, json.dumps(data, indent=2, sort_keys=True) + "\n")


def read_json(path: str | Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def file_exists_and_nonempty(path: str | Path) -> bool:
    path = Path(path)
    return path.exists() and path.is_file() and path.stat().st_size > 0

