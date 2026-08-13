"""Duplicate detection helpers across anchors and generated families."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Set

from .parser import parse_file
from .validator import structural_hash


def structural_hashes_for_paths(paths: Iterable[str | Path]) -> Set[str]:
    hashes: Set[str] = set()
    for path in paths:
        path = Path(path)
        if not path.exists() or path.suffix.lower() != ".txt":
            continue
        hashes.add(structural_hash(parse_file(path)))
    return hashes


def existing_generated_hashes(generated_root: str | Path) -> Set[str]:
    root = Path(generated_root)
    if not root.exists():
        return set()
    return structural_hashes_for_paths(root.glob("**/*.txt"))
