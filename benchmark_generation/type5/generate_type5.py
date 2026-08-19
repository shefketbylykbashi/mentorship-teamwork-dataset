"""Generate Type 5 empirically grounded synthetic MTP instances."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from benchmark_generation.core.config import load_config
from benchmark_generation.core.distance import FeatureScaler
from benchmark_generation.core.duplicates import structural_hashes_for_paths
from benchmark_generation.core.features import extract_features
from benchmark_generation.core.generator import GenerationParameters, derive_seed, generate_valid_instance
from benchmark_generation.core.metadata import (
    build_metadata_row,
    config_hash,
    family_statistical_report,
    summarize_numeric,
    update_master_manifest,
    write_manifest,
    write_metadata,
)
from benchmark_generation.core.parser import parse_file, write_instance
from benchmark_generation.core.validator import structural_hash, validate_file
from benchmark_generation.type5.empirical import empirical_profile_from_record
from benchmark_generation.type5.quality import (
    evaluate_type5_quality,
    final_type5_quality_gate,
)


DEFAULT_CONFIG: Dict[str, Any] = {
    "family": "type5",
    "count": 100,
    "root_seed": 2026,
    "empirical_corpus_path": "benchmark_metadata/type5_empirical_corpus.json",
    "fillable_fraction": 0.30,
    "max_generation_attempts_per_source": 30,
    "distance_group_weights": {
        "size": 1,
        "supply_demand": 1,
        "levels": 1,
        "mentorship": 1,
        "time": 1,
        "scores": 1,
    },
}


def generate_type5(
    output_dir: str | Path,
    metadata_dir: str | Path,
    count: int,
    seed: int,
    config: Dict[str, Any],
    overwrite: bool = False,
    dry_run: bool = False,
    pilot: bool = False,
) -> List[Dict[str, Any]]:
    output_dir = Path(output_dir)
    metadata_dir = Path(metadata_dir)
    corpus_path = Path(str(config.get("empirical_corpus_path", "benchmark_metadata/type5_empirical_corpus.json")))
    corpus = load_empirical_corpus(corpus_path)
    records = list(corpus.get("records", []))
    if pilot:
        selected_records = select_pilot_records(records, count)
    else:
        if len(records) < count:
            raise RuntimeError(
                f"Type 5 requires {count} empirical source configurations; found {len(records)}. "
                "Run build_empirical_profiles after authenticated collection."
            )
        selected_records = sorted(records, key=lambda item: int(item["family_instance_id"]))[:count]

    existing_hashes = existing_hashes_for_type5(output_dir, overwrite=overwrite)
    accepted_hashes = set(existing_hashes)
    rows: List[Dict[str, Any]] = []
    feature_rows: List[Dict[str, float]] = []
    normalized_vectors: List[Dict[str, float]] = []
    failed_attempts: List[Dict[str, Any]] = []

    for output_idx, record in enumerate(selected_records, start=1):
        family_id = int(record["family_instance_id"]) if not pilot else output_idx
        filename = f"type5_{family_id:03d}.txt" if not pilot else f"type5_pilot_{output_idx:03d}.txt"
        params = GenerationParameters(**record["parameters"]).normalized()
        empirical_profile = empirical_profile_from_record(record)
        logging.info(
            "Type5 %03d/%03d source=%s/%s window=%s",
            output_idx,
            count,
            record["source_owner"],
            record["source_repo"],
            record["source_window_name"],
        )
        accepted = None
        last_quality = None
        for attempt in range(1, int(config.get("max_generation_attempts_per_source", 30)) + 1):
            attempt_seed = derive_seed(seed, family_id * 1000 + attempt, salt=15_485_863)
            try:
                outcome = generate_valid_instance(
                    params,
                    attempt_seed,
                    max_attempts=1,
                    source_name=filename,
                    empirical_profile=empirical_profile,
                )
            except RuntimeError as exc:
                failed_attempts.append(
                    {
                        "filename": filename,
                        "family_instance_id": family_id,
                        "attempt": attempt,
                        "reason": str(exc),
                    }
                )
                continue
            h = structural_hash(outcome.instance)
            duplicate_pass = h not in accepted_hashes
            quality = evaluate_type5_quality(
                outcome.instance,
                outcome.validation,
                record,
                duplicate_pass=duplicate_pass,
                config=config,
            )
            last_quality = quality
            if quality["overall_type5_quality_pass"]:
                accepted = (outcome, h, quality, attempt)
                accepted_hashes.add(h)
                break
            failed_attempts.append(
                {
                    "filename": filename,
                    "family_instance_id": family_id,
                    "attempt": attempt,
                    "reason": "type5_quality_gate_failed",
                    "quality": quality_summary(quality),
                }
            )
        if accepted is None:
            raise RuntimeError(
                f"Could not generate accepted Type 5 instance for {filename}. "
                f"Last quality={json.dumps(quality_summary(last_quality or {}), sort_keys=True)}"
            )

        outcome, h, quality, outer_attempt = accepted
        features = extract_features(outcome.instance)
        feature_rows.append(features)
        if not dry_run:
            path = output_dir / filename
            write_instance(outcome.instance, path, overwrite=overwrite)
            validation = validate_file(path, fillable_fraction=outcome.parameters.fillable_fraction)
            if not validation.ok:
                raise RuntimeError(f"Written Type 5 instance failed validation: {path}: {validation.issues}")
        else:
            validation = outcome.validation

        row = build_type5_metadata_row(
            filename=filename,
            family_id=family_id,
            root_seed=seed,
            outcome_seed=outcome.seed,
            generator_version=outcome.generator_version,
            parameters=outcome.parameters.to_dict(),
            features=features,
            validation=validation,
            record=record,
            structural_hash=h,
            quality=quality,
            generation_attempts=outer_attempt,
        )
        rows.append(row)
        logging.info(
            "accepted %s contributors=%d projects=%d skills=%d feasibility=%.3f",
            filename,
            len(outcome.instance.contributors),
            len(outcome.instance.projects),
            int(features["n_skills"]),
            quality["metrics"]["realistic_initial_feasibility"],
        )

    scaler = FeatureScaler.fit(feature_rows, group_weights=config.get("distance_group_weights"))
    normalized_vectors = [scaler.normalize(features) for features in feature_rows]
    report = type5_statistical_report(
        rows,
        normalized_vectors,
        scaler.group_weights,
        corpus,
        failed_attempts,
        pilot=pilot,
    )

    if not dry_run:
        metadata_dir.mkdir(parents=True, exist_ok=True)
        report_name = "type5_pilot_statistical_report.json" if pilot else "type5_statistical_report.json"
        write_manifest(metadata_dir / report_name, report)
        if pilot:
            write_metadata(
                rows,
                metadata_dir / "type5_pilot_metadata.csv",
                metadata_dir / "type5_pilot_metadata.jsonl",
            )
            write_pilot_table(rows, metadata_dir / "type5_pilot_table.csv")
            gate = pilot_quality_gate(rows)
            write_manifest(metadata_dir / "type5_pilot_quality_gate.json", gate)
            if not gate["passed"]:
                raise RuntimeError("Type 5 pilot quality gate failed: " + json.dumps(gate, sort_keys=True))
        else:
            provenance_path = metadata_dir / "TYPE5_SOURCES.md"
            methodology_path = Path("METHODOLOGY.md")
            gate = final_type5_quality_gate(
                rows,
                output_dir,
                expected_count=count,
                corpus=corpus,
                report_path=metadata_dir / report_name,
                provenance_path=provenance_path,
                methodology_path=methodology_path,
            )
            report["final_quality_gate"] = gate
            write_manifest(metadata_dir / report_name, report)
            if not gate["passed"]:
                raise RuntimeError("Type 5 final quality gate failed: " + json.dumps(gate, sort_keys=True))
            write_metadata(
                rows,
                metadata_dir / "type5_metadata.csv",
                metadata_dir / "type5_metadata.jsonl",
            )
            update_master_manifest(
                metadata_dir / "benchmark_manifest.json",
                "type5",
                {
                    "count": len(rows),
                    "seed": seed,
                    "config_hash": config_hash(config),
                    "output_dir": str(output_dir),
                    "metadata_csv": str(metadata_dir / "type5_metadata.csv"),
                    "statistical_report": str(metadata_dir / report_name),
                    "empirical_corpus": str(corpus_path),
                },
            )
        (metadata_dir / "type5_config_used.json").write_text(
            json.dumps(config, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    return rows


def load_empirical_corpus(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Type 5 empirical corpus is missing: {path}. Run "
            "`python -m benchmark_generation.type5.build_empirical_profiles` first."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data.get("records"), list):
        raise ValueError(f"Malformed Type 5 empirical corpus: {path}")
    return data


def select_pilot_records(records: Sequence[Mapping[str, Any]], count: int) -> List[Mapping[str, Any]]:
    if len(records) < count:
        raise RuntimeError(f"Pilot requested {count} records, but only {len(records)} source configurations exist")
    ordered = sorted(records, key=lambda item: int(item["family_instance_id"]))
    selected: List[Mapping[str, Any]] = []
    seen_repos = set()
    seen_domains = set()
    seen_windows = set()
    for window in ("W1", "W2", "W3", "W4"):
        for record in ordered:
            repo_key = f"{record['source_owner']}/{record['source_repo']}"
            if record["source_window_name"] == window and repo_key not in seen_repos:
                selected.append(record)
                seen_repos.add(repo_key)
                seen_domains.add(record["source_domain"])
                seen_windows.add(window)
                break
    for record in ordered:
        if len(selected) >= count:
            break
        repo_key = f"{record['source_owner']}/{record['source_repo']}"
        if repo_key in seen_repos and len(seen_repos) >= 5:
            continue
        selected.append(record)
        seen_repos.add(repo_key)
        seen_domains.add(record["source_domain"])
        seen_windows.add(record["source_window_name"])
    for record in ordered:
        if len(selected) >= count:
            break
        if record not in selected:
            selected.append(record)
    return selected[:count]


def existing_hashes_for_type5(output_dir: Path, overwrite: bool) -> set[str]:
    generated_root = output_dir.parent
    paths = []
    if generated_root.exists():
        for path in generated_root.glob("**/*.txt"):
            if overwrite and output_dir in path.parents:
                continue
            paths.append(path)
    paths.extend(Path("google_instances").glob("*.txt"))
    return structural_hashes_for_paths(paths)


def build_type5_metadata_row(
    filename: str,
    family_id: int,
    root_seed: int,
    outcome_seed: int,
    generator_version: str,
    parameters: Mapping[str, Any],
    features: Mapping[str, float],
    validation,
    record: Mapping[str, Any],
    structural_hash: str,
    quality: Mapping[str, Any],
    generation_attempts: int,
) -> Dict[str, Any]:
    validation_payload = {
        "ok": validation.ok,
        "issues": validation.issues,
        "initial_fillable_fraction": validation.metrics.get("initial_fillable_fraction", ""),
        "generation_attempts": generation_attempts,
    }
    extra: Dict[str, Any] = {
        "root_seed": root_seed,
        "instance_seed": outcome_seed,
        "structural_hash": structural_hash,
        "source_owner": record["source_owner"],
        "source_repo": record["source_repo"],
        "source_url": record["source_url"],
        "source_domain": record["source_domain"],
        "source_repository_id": record["source_repository_id"],
        "source_default_branch": record["source_default_branch"],
        "source_snapshot_sha": record["source_snapshot_sha"],
        "source_window_name": record["source_window_name"],
        "source_window_start": record["source_window_start"],
        "source_window_end": record["source_window_end"],
        "source_collection_timestamp": record["source_collection_timestamp"],
        "source_replacement": record["source_replacement"],
        "original_source_if_replaced": record["original_source_if_replaced"],
        "replacement_reason": record["replacement_reason"],
    }
    extra.update(record["source_metrics"])
    extra.update(record["proxy_flags"])
    for key in (
        "structural_validation_pass",
        "duplicate_pass",
        "profiles_pass",
        "skill_skew_pass",
        "alignment_pass",
        "difficulty_mix_pass",
        "initial_realistic_feasibility_pass",
        "source_distribution_match_pass",
        "overall_type5_quality_pass",
    ):
        extra[key] = quality[key]
    extra["type5"] = {
        "structural_hash": structural_hash,
        "structural_validation_pass": quality["structural_validation_pass"],
        "duplicate_pass": quality["duplicate_pass"],
        "profiles_pass": quality["profiles_pass"],
        "skill_skew_pass": quality["skill_skew_pass"],
        "alignment_pass": quality["alignment_pass"],
        "difficulty_mix_pass": quality["difficulty_mix_pass"],
        "initial_realistic_feasibility_pass": quality["initial_realistic_feasibility_pass"],
        "source_distribution_match_pass": quality["source_distribution_match_pass"],
        "overall_type5_quality_pass": quality["overall_type5_quality_pass"],
        "skill_skew_delta": quality["skill_skew_delta"],
        "alignment_delta": quality["alignment_delta"],
        "quality_metrics": quality["metrics"],
        "distribution_match": quality["distribution_match"],
        "semantic_skill_tokens": record["source_profile"]["skill_tokens"],
        "nominal_difficulty_mix": record["nominal_difficulty_mix"],
    }
    return build_metadata_row(
        filename=filename,
        family="type5",
        family_instance_id=family_id,
        seed=outcome_seed,
        generator_version=generator_version,
        parameters=parameters,
        features=features,
        validation=validation_payload,
        extra=extra,
    )


def type5_statistical_report(
    rows: Sequence[Mapping[str, Any]],
    normalized_vectors: Sequence[Mapping[str, float]],
    group_weights: Mapping[str, float],
    corpus: Mapping[str, Any],
    failed_attempts: Sequence[Mapping[str, Any]],
    pilot: bool = False,
) -> Dict[str, Any]:
    report = family_statistical_report(rows, normalized_vectors, group_weights)
    repositories = [f"{row.get('source_owner')}/{row.get('source_repo')}" for row in rows]
    windows = [str(row.get("source_window_name")) for row in rows]
    domains = [str(row.get("source_domain")) for row in rows]
    report.update(
        {
            "pilot": pilot,
            "failed_attempts": len(failed_attempts),
            "failed_attempt_samples": list(failed_attempts[:25]),
            "replacements": corpus.get("replacements", []),
            "duplicate_count": sum(1 for row in rows if not _truthy(row.get("type5.duplicate_pass"))),
            "validation_failures": sum(1 for row in rows if not _truthy(row.get("type5.structural_validation_pass"))),
            "source_coverage": {
                "repository_count": len(set(repositories)),
                "domain_count": len(set(domains)),
                "count_per_repository": counts(repositories),
                "count_per_window": counts(windows),
                "replacements_by_reason": counts(
                    [str(item.get("failure_reason", "")) for item in corpus.get("replacements", [])]
                ),
            },
            "contributor_realism": {
                "profile_count": summarize_numeric([float(row.get("source_profile_count", 0) or 0) for row in rows]),
                "source_skill_skew": summarize_numeric([float(row.get("source_skill_skew", 0) or 0) for row in rows]),
                "generated_skill_skew": summarize_numeric([float(row.get("type5.quality_metrics.generated_skill_skew", 0) or 0) for row in rows]),
            },
            "demand_supply": {
                "source_alignment": summarize_numeric([float(row.get("source_alignment", 0) or 0) for row in rows]),
                "generated_alignment": summarize_numeric([float(row.get("type5.quality_metrics.generated_alignment", 0) or 0) for row in rows]),
            },
            "mentorship": {
                "directly_fillable_share": summarize_numeric([float(row.get("type5.quality_metrics.directly_fillable_share", 0) or 0) for row in rows]),
                "truly_mentorship_feasible_share": summarize_numeric([float(row.get("type5.quality_metrics.truly_valid_initial_mentorship_feasible_share", 0) or 0) for row in rows]),
                "future_share": summarize_numeric([float(row.get("type5.quality_metrics.measured_p_future", 0) or 0) for row in rows]),
                "mentor_share": summarize_numeric([float(row.get("type5.quality_metrics.measured_p_mentor", 0) or 0) for row in rows]),
            },
            "source_match": {
                "skill_skew_delta": summarize_numeric([float(row.get("type5.skill_skew_delta", 0) or 0) for row in rows]),
                "alignment_delta": summarize_numeric([float(row.get("type5.alignment_delta", 0) or 0) for row in rows]),
            },
            "score_proxy_methodology": "Score is an effort-conditioned synthetic proxy; GitHub popularity or comments are not used as value.",
            "difficulty_prior": "Dirichlet concentration proportional to [25,45,20,10], with mentor/future bounds and renormalization.",
        }
    )
    return report


def write_pilot_table(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    fields = [
        "filename",
        "source_repo",
        "source_window_name",
        "feature.n_contributors",
        "feature.n_projects",
        "feature.n_skills",
        "source_profile_count",
        "source_skill_skew",
        "source_alignment",
        "type5.quality_metrics.directly_fillable_share",
        "type5.quality_metrics.truly_valid_initial_mentorship_feasible_share",
        "type5.quality_metrics.measured_p_future",
        "type5.structural_validation_pass",
        "type5.duplicate_pass",
        "type5.overall_type5_quality_pass",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def pilot_quality_gate(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    repos = {f"{row.get('source_owner')}/{row.get('source_repo')}" for row in rows}
    domains = {str(row.get("source_domain")) for row in rows}
    windows = {str(row.get("source_window_name")) for row in rows}
    gate = {
        "accepted_count": len(rows),
        "accepted_count_exact": len(rows) == 10,
        "repository_coverage_at_least_5": len(repos) >= 5,
        "domain_coverage_at_least_3": len(domains) >= 3,
        "all_four_windows_where_possible": len(windows) == 4,
        "all_realism_pass": all(_truthy(row.get("type5.overall_type5_quality_pass")) for row in rows),
        "all_structural_valid": all(_truthy(row.get("type5.structural_validation_pass")) for row in rows),
        "all_duplicate_pass": all(_truthy(row.get("type5.duplicate_pass")) for row in rows),
    }
    gate["passed"] = all(gate.values())
    return gate


def counts(values: Sequence[str]) -> Dict[str, int]:
    result: Dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return dict(sorted(result.items()))


def quality_summary(quality: Mapping[str, Any]) -> Dict[str, Any]:
    if not quality:
        return {}
    return {
        key: quality.get(key)
        for key in (
            "structural_validation_pass",
            "duplicate_pass",
            "profiles_pass",
            "skill_skew_pass",
            "alignment_pass",
            "difficulty_mix_pass",
            "initial_realistic_feasibility_pass",
            "scale_pass",
            "source_distribution_match_pass",
            "overall_type5_quality_pass",
        )
    }


def _truthy(value: Any) -> bool:
    return value is True or str(value).lower() == "true"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="generated_instances/type5")
    parser.add_argument("--metadata-dir", default="benchmark_metadata")
    parser.add_argument("--config", default="benchmark_generation/configs/type5.yaml")
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
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
        config["max_generation_attempts_per_source"] = args.max_attempts
    generate_type5(
        output_dir=args.output_dir,
        metadata_dir=args.metadata_dir,
        count=count,
        seed=seed,
        config=config,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        pilot=args.pilot,
    )


if __name__ == "__main__":
    main()
