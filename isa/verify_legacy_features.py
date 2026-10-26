"""Cross-check re-extracted ISA features against available generation metadata."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from benchmark_generation.core.features import FEATURE_NAMES


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def verify(repo_root: Path, isa_dir: Path) -> dict:
    repo_root = repo_root.resolve()
    isa_dir = isa_dir.resolve()
    audit = json.loads((isa_dir / "audit.json").read_text(encoding="utf-8"))
    expected = audit.get("output_sha256", {}).get("all_features.csv")
    actual = hashlib.sha256((isa_dir / "all_features.csv").read_bytes()).hexdigest()
    if expected != actual:
        raise ValueError("ISA export checksum mismatch: all_features.csv")
    latest = {row["instances"]: row for row in _rows(isa_dir / "all_features.csv")}
    sources = [
        ("type1", repo_root / "benchmark_metadata/google_anchor_features.csv", "instance_name"),
        *((family, repo_root / f"benchmark_metadata/{family}_metadata.csv", "filename")
          for family in ("type2", "type4", "type5", "type6", "type7")),
    ]
    comparisons = 0
    instance_count = 0
    mismatches = []
    family_counts = {}
    for family, path, filename_field in sources:
        rows = _rows(path)
        family_counts[family] = len(rows)
        for legacy in rows:
            stem = Path(legacy[filename_field]).stem
            iid = f"type1_{stem}" if family == "type1" else stem
            if iid not in latest:
                raise ValueError(f"Metadata instance absent from ISA export: {iid}")
            current = latest[iid]
            instance_count += 1
            for name in FEATURE_NAMES:
                old = float(legacy[f"feature.{name}"])
                new = float(current[name])
                comparisons += 1
                if not math.isclose(old, new, rel_tol=1e-10, abs_tol=1e-9):
                    mismatches.append({"instance": iid, "feature": name, "old": old, "new": new})
    report = {
        "meaning": "agreement of newly parsed/recomputed structural features with available frozen generation metadata",
        "metadata_family_counts": family_counts,
        "instances_compared": instance_count,
        "feature_values_compared": comparisons,
        "mismatch_count": len(mismatches),
        "mismatches_first_20": mismatches[:20],
        "excluded_from_comparison": "Type 3 has no final family metadata; Type 5 full-project extension files beyond the original type5_metadata.csv are not covered",
    }
    (isa_dir / "legacy_feature_agreement.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    if mismatches:
        raise ValueError(f"{len(mismatches)} legacy feature values disagree; inspect report")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--input", type=Path, default=Path("isa/outputs/rizana_trial_01"))
    args = parser.parse_args()
    print(json.dumps(verify(args.repo_root, args.input), indent=2))


if __name__ == "__main__":
    main()
