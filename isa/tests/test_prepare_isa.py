"""Small deterministic checks for ISA input construction."""

from __future__ import annotations

import hashlib
import csv
import json
import tempfile
import unittest
from pathlib import Path

from benchmark_generation.core.features import FEATURE_NAMES, _cosine_similarity
from isa.prepare_isa import (
    _canonical_campaign_hash,
    _solver_selection,
    expected_score,
    select_features,
)
from isa.summarize_export import summarize


def _summary(*, planned: int = 10, success: int = 8, timeout: int = 2, mean_score: str = "100") -> dict[str, str]:
    return {
        "instance_id": "example", "solver_id": "solver", "planned_runs": str(planned),
        "successful_runs": str(success), "timeout_runs": str(timeout),
        "error_runs": "0", "interrupted_runs": "0", "invalid_output_runs": "0",
        "invalid_solution_runs": "0", "mean_score": mean_score,
    }


class ExpectedScoreTests(unittest.TestCase):
    def test_timeout_is_zero_utility_not_missing(self) -> None:
        self.assertEqual(expected_score(_summary()), 80.0)
        self.assertEqual(expected_score(_summary(success=0, timeout=10, mean_score="")), 0.0)

    def test_inconsistent_counts_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "do not sum"):
            expected_score(_summary(success=9, timeout=2))

    def test_unverified_error_rejected(self) -> None:
        row = _summary(success=7)
        row["error_runs"] = "1"
        with self.assertRaisesRegex(ValueError, "non-timeout"):
            expected_score(row)


class SelectionTests(unittest.TestCase):
    def test_alignment_sum_uses_stable_skill_order(self) -> None:
        left = {"Zulu": 10**12, "Alpha": 1, "Middle": 3}
        right = {"Middle": 3, "Alpha": 1, "Zulu": 10**12}
        self.assertEqual(
            _cosine_similarity(left, right),
            _cosine_similarity(dict(reversed(list(left.items()))), dict(reversed(list(right.items())))),
        )

    def test_one_source_hash_one_representative(self) -> None:
        data = {"count": 3, "solvers": [
            {"solver_id": "a", "source_sha256": "hash_a", "objective": "maximize"},
            {"solver_id": "b", "source_sha256": "hash_b", "objective": "maximize"},
            {"solver_id": "c", "source_sha256": "hash_a", "objective": "maximize"},
        ]}
        all_ids, primary, groups = _solver_selection(data)
        self.assertEqual(all_ids, ["a", "b", "c"])
        self.assertEqual(primary, ["a", "b"])
        self.assertEqual(groups[0]["solver_ids"], ["a", "c"])

    def test_feature_filter_uses_structural_features_only(self) -> None:
        rows = []
        for index in range(5):
            row = {name: 0.0 for name in FEATURE_NAMES}
            row.update({
                "n_contributors": float(index + 1),
                "n_projects": float(index + 1),
                "demand_supply_alignment": float(index) / 4,
                "level_gap_mean": float(index),
                "direct_fillability": float(index) / 4,
                "deadline_tightness_mean": float(index + 1),
                "score_density": float(index + 1),
            })
            rows.append(row)
        selected, decisions, values = select_features(rows)
        self.assertIn("log1p_n_contributors", selected)
        self.assertNotIn("log1p_n_projects", selected)
        self.assertEqual(next(d for d in decisions if d["column"] == "feature_log1p_n_projects")["status"], "excluded_redundant")
        self.assertEqual(len(values["log1p_n_contributors"]), 5)


class HashTests(unittest.TestCase):
    def test_checkout_line_endings_are_only_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.txt"
            path.write_bytes(b"1 1\r\nName 0\r\nJob 1 1 1 0\r\n")
            got, normalized = _canonical_campaign_hash(path)
            self.assertTrue(normalized)
            self.assertEqual(got, hashlib.sha256(b"1 1\nName 0\nJob 1 1 1 0\n").hexdigest())


class ReportTests(unittest.TestCase):
    def test_tie_aware_oracle_headroom(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = [
                {"instances": "x1", "source": "type1", "algo_a": 1.0, "algo_b": 0.8},
                {"instances": "x2", "source": "type1", "algo_a": 0.9, "algo_b": 1.0},
                {"instances": "x3", "source": "type2", "algo_a": 1.0, "algo_b": 1.0},
            ]
            with (root / "metadata.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(metadata[0]))
                writer.writeheader()
                writer.writerows(metadata)
            audits = [
                {"instances": row["instances"], "source": row["source"],
                 "co_best_count": 2 if row["instances"] == "x3" else 1}
                for row in metadata
            ]
            with (root / "instance_audit.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(audits[0]))
                writer.writeheader()
                writer.writerows(audits)
            (root / "audit.json").write_text(json.dumps({
                "n_instances": 3, "primary_solver_ids": ["a", "b"], "good_threshold": 0.95,
                "output_sha256": {
                    name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                    for name in ("metadata.csv", "instance_audit.csv")
                },
            }), encoding="utf-8")
            report = summarize(root)
            self.assertEqual(report["best_fixed_solver_pooled"], "a")
            self.assertAlmostEqual(report["pooled_oracle_headroom_vs_best_fixed"], 1 / 30)
            self.assertEqual(report["by_family"]["type2"]["all_solvers_cobest_count"], 1)
            with self.assertRaises(FileExistsError):
                summarize(root)


if __name__ == "__main__":
    unittest.main()
