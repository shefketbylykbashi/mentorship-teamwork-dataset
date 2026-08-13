from pathlib import Path

import pytest

from benchmark_generation.core.anchors import discover_google_instance_paths
from benchmark_generation.core.parser import parse_file, parse_text
from benchmark_generation.core.validator import validate_file


def test_google_instances_parse_and_toy_excluded():
    paths = discover_google_instance_paths(Path("google_instances"))
    assert len(paths) == 5
    assert all(not path.name.startswith("a_an_example") for path in paths)
    for path in paths:
        instance = parse_file(path)
        assert len(instance.contributors) > 0
        assert len(instance.projects) > 0


def test_parser_round_trip_counts():
    text = "1 1\nc0 1\npy 2\np0 1 10 2 1\npy 2\n"
    instance = parse_text(text)
    assert len(instance.contributors) == 1
    assert len(instance.projects) == 1
    assert instance.total_roles == 1


def test_validator_rejects_duplicate_contributor_skill(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("1 1\nc0 2\npy 2\npy 3\np0 1 10 2 1\npy 2\n", encoding="utf-8")
    result = validate_file(path, fillable_fraction=0.0)
    assert not result.ok
    assert any("duplicate_skill_for_contributor" in issue for issue in result.issues)
