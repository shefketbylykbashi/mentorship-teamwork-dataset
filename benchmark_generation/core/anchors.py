"""Utilities for immutable Google Hash Code anchor instances."""

from __future__ import annotations

from pathlib import Path
from typing import List

from .models import MTPInstance
from .parser import parse_file


TOY_PREFIX = "a_an_example"


def discover_google_instance_paths(google_dir: str | Path) -> List[Path]:
    paths = sorted(Path(google_dir).glob("*.txt"))
    return [path for path in paths if not path.stem.startswith(TOY_PREFIX)]


def load_google_anchors(google_dir: str | Path) -> List[MTPInstance]:
    anchors = []
    for path in discover_google_instance_paths(google_dir):
        anchors.append(parse_file(path))
    if len(anchors) != 5:
        raise ValueError(
            f"Expected five non-trivial Google anchors after excluding toy instance; "
            f"found {len(anchors)} in {google_dir}"
        )
    return anchors
