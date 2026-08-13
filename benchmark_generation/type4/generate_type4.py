"""CLI for Type 4 broad random MTP instances."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List

from benchmark_generation.core.config import load_config
from benchmark_generation.core.distance import FeatureScaler
from benchmark_generation.core.duplicates import existing_generated_hashes, structural_hashes_for_paths
from benchmark_generation.core.features import extract_features
from benchmark_generation.core.generator import derive_seed, generate_valid_instance
from benchmark_generation.core.metadata import (
    build_metadata_row,
    config_hash,
    family_statistical_report,
    update_master_manifest,
    write_manifest,
    write_metadata,
)
from benchmark_generation.core.parameter_sampling import band_allocation, sample_type4_parameters
from benchmark_generation.core.parser import write_instance
from benchmark_generation.core.validator import structural_hash, validate_file


DEFAULT_CONFIG: Dict[str, Any] = {
    "family": "type4",
    "count": 100,
    "root_seed": 2026,
    "fillable_fraction": 0.30,
    "max_parameter_attempts": 100,
    "max_generation_attempts": 50,
    "role_level_max": 100,
    "scale_band_allocation": {"small": 34, "medium": 33, "large": 33},
    "scale_bands": {
        "small": {"contributors": [5, 50], "projects": [5, 50]},
        "medium": {"contributors": [50, 500], "projects": [50, 500]},
        "large": {"contributors": [500, 10000], "projects": [500, 10000]},
    },
    "project_contributor_ratio": [0.25, 4.0],
    "difficulty_simplex": {
        "distribution": "symmetric_dirichlet",
        "alpha": [1.0, 1.0, 1.0, 1.0],
    },
    "parameter_ranges": {
        "n_skills": [5, 200],
        "skill_skew": [0.0, 1.5],
        "skills_per_contributor": [1, 20],
        "level_mean": [2.0, 7.0],
        "level_sd": [0.5, 3.0],
        "level_max": 10,
        "n_profiles": 1,
        "duration": [1, 500],
        "roles_per_project": [1, 20],
        "score_base": [0.5, 5.0],
        "score_perturbation": [1.0, 3.0],
        "tightness": [0.5, 6.0],
        "role_skill_alignment": [0.0, 1.0],
        "supply_quantile": [0.5, 0.9],
    },
    "distance_group_weights": {
        "size": 1,
        "supply_demand": 1,
        "levels": 1,
        "mentorship": 1,
        "time": 1,
        "scores": 1,
    },
}


def allocation_sequence(count: int, config: Dict[str, Any]) -> List[str]:
    configured = config.get("scale_band_allocation", {})
    bands = list(config["scale_bands"].keys())
    if sum(int(configured.get(band, 0)) for band in bands) == count:
        sequence: List[str] = []
        for band in bands:
            sequence.extend([band] * int(configured[band]))
        return sequence
    return band_allocation(count, bands)


def generate_type4(
    output_dir: str | Path,
    metadata_dir: str | Path,
    count: int,
    seed: int,
    config: Dict[str, Any],
    overwrite: bool = False,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    output_dir = Path(output_dir)
    metadata_dir = Path(metadata_dir)
    rng = random.Random(seed)
    bands = allocation_sequence(count, config)
    rng.shuffle(bands)
    rows: List[Dict[str, Any]] = []
    feature_rows: List[Dict[str, float]] = []
    normalized_vectors: List[Dict[str, float]] = []

    existing_hashes = existing_generated_hashes(output_dir.parent)
    google_hashes = structural_hashes_for_paths(Path("google_instances").glob("*.txt"))
    existing_hashes.update(google_hashes)

    accepted_hashes = set(existing_hashes)
    accepted_instances = 0

    for idx in range(1, count + 1):
        band = bands[idx - 1]
        instance_seed = derive_seed(seed, idx)
        filename = f"type4_{idx:03d}.txt"
        logging.info("Type4 %03d/%03d band=%s", idx, count, band)
        outcome = None
        last_error = None
        for parameter_attempt in range(1, int(config["max_parameter_attempts"]) + 1):
            param_rng = random.Random(derive_seed(instance_seed, parameter_attempt))
            params = sample_type4_parameters(config, param_rng, band)
            try:
                candidate = generate_valid_instance(
                    params,
                    derive_seed(instance_seed, parameter_attempt, salt=31_337),
                    max_attempts=int(config["max_generation_attempts"]),
                    source_name=filename,
                )
            except RuntimeError as exc:
                last_error = str(exc)
                continue
            h = structural_hash(candidate.instance)
            if h in accepted_hashes:
                last_error = "duplicate_structural_hash"
                continue
            outcome = candidate
            accepted_hashes.add(h)
            break
        if outcome is None:
            raise RuntimeError(f"Could not generate Type4 instance {idx}: {last_error}")

        features = extract_features(outcome.instance)
        feature_rows.append(features)
        if not dry_run:
            path = output_dir / filename
            write_instance(outcome.instance, path, overwrite=overwrite)
            validation = validate_file(path, fillable_fraction=outcome.parameters.fillable_fraction)
            if not validation.ok:
                raise RuntimeError(f"Written instance failed validation: {path}: {validation.issues}")
        else:
            validation = outcome.validation

        row = build_metadata_row(
            filename=filename,
            family="type4",
            family_instance_id=idx,
            seed=outcome.seed,
            generator_version=outcome.generator_version,
            parameters=outcome.parameters.to_dict(),
            features=features,
            validation={
                "ok": validation.ok,
                "issues": validation.issues,
                "initial_fillable_fraction": validation.metrics.get("initial_fillable_fraction", ""),
                "generation_attempts": outcome.attempts,
            },
            extra={
                "type4": {
                    "scale_band": band,
                    "structural_hash": structural_hash(outcome.instance),
                    "rejection_reasons": outcome.rejection_reasons,
                }
            },
        )
        rows.append(row)
        accepted_instances += 1
        logging.info(
            "accepted type4_%03d contributors=%d projects=%d roles=%d fillability=%.3f",
            idx,
            len(outcome.instance.contributors),
            len(outcome.instance.projects),
            outcome.instance.total_roles,
            validation.metrics.get("initial_fillable_fraction", 0.0),
        )

    scaler = FeatureScaler.fit(feature_rows, group_weights=config.get("distance_group_weights"))
    normalized_vectors = [scaler.normalize(features) for features in feature_rows]
    report = family_statistical_report(
        rows,
        normalized_vectors,
        group_weights=scaler.group_weights,
        extra={
            "scale_band_counts": {
                band: sum(1 for row in rows if row.get("type4.scale_band") == band)
                for band in config["scale_bands"]
            },
            "parameter_coverage_note": "Parameters are sampled independently within configured Type 4 ranges except scale-band counts and project/contributor ratio stratification.",
        },
    )

    if not dry_run:
        write_metadata(
            rows,
            metadata_dir / "type4_metadata.csv",
            metadata_dir / "type4_metadata.jsonl",
        )
        write_manifest(metadata_dir / "type4_statistical_report.json", report)
        update_master_manifest(
            metadata_dir / "benchmark_manifest.json",
            "type4",
            {
                "count": accepted_instances,
                "seed": seed,
                "config_hash": config_hash(config),
                "output_dir": str(output_dir),
                "metadata_csv": str(metadata_dir / "type4_metadata.csv"),
                "statistical_report": str(metadata_dir / "type4_statistical_report.json"),
            },
        )
        (metadata_dir / "type4_config_used.json").write_text(
            json.dumps(config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="generated_instances/type4")
    parser.add_argument("--metadata-dir", default="benchmark_metadata")
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--config", default="benchmark_generation/configs/type4.yaml")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="Reserved for future parallel generation.")
    parser.add_argument("--max-attempts", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config = load_config(args.config, DEFAULT_CONFIG)
    count = args.count if args.count is not None else int(config["count"])
    seed = args.seed if args.seed is not None else int(config["root_seed"])
    if args.max_attempts is not None:
        config["max_generation_attempts"] = args.max_attempts
    generate_type4(
        output_dir=args.output_dir,
        metadata_dir=args.metadata_dir,
        count=count,
        seed=seed,
        config=config,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
