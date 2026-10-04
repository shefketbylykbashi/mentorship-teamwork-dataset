"""Generate the Type 5 complete-project extension instances."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from benchmark_generation import __version__
from benchmark_generation.core.config import load_config
from benchmark_generation.core.distance import FeatureScaler
from benchmark_generation.core.duplicates import structural_hashes_for_paths
from benchmark_generation.core.features import extract_features
from benchmark_generation.core.generator import GenerationParameters, derive_seed, generate_valid_instance
from benchmark_generation.core.metadata import (
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
from benchmark_generation.type5.full_project import (
    FINAL_EXTENSION_COUNT,
    FINAL_EXTENSION_START_ID,
    FULL_PROJECT_OBSERVATION_MODE,
    FULL_PROJECT_WINDOW_TYPE,
    PILOT_EXTENSION_COUNT,
    full_project_filename,
    full_project_source_statistics,
    pilot_filename,
    repository_overlap,
    select_pilot_full_project_records,
    write_full_project_methodology_draft,
)
from benchmark_generation.type5.generate_type5 import build_type5_metadata_row, quality_summary
from benchmark_generation.type5.quality import evaluate_type5_quality, final_instance_names_are_anonymized


DEFAULT_CONFIG: Dict[str, Any] = {
    "root_seed": 2026,
    "full_project_corpus_path": "benchmark_metadata/type5_full_project_empirical_corpus.json",
    "fillable_fraction": 0.30,
    "max_generation_attempts_per_source": 30,
    "final_extension_count": FINAL_EXTENSION_COUNT,
    "final_extension_start_id": FINAL_EXTENSION_START_ID,
    "pilot_count": PILOT_EXTENSION_COUNT,
    "distance_group_weights": {
        "size": 1,
        "supply_demand": 1,
        "levels": 1,
        "mentorship": 1,
        "time": 1,
        "scores": 1,
    },
}

FIXED_TYPE5_RE = re.compile(r"^type5_(\d{3})\.txt$")


def generate_type5_full_project_extension(
    output_dir: str | Path,
    metadata_dir: str | Path,
    pilot_output_dir: str | Path,
    pilot_metadata_dir: str | Path,
    config: Dict[str, Any],
    seed: int,
    dry_run: bool = False,
    overwrite_pilot: bool = False,
    overwrite_final: bool = False,
    skip_pilot: bool = False,
) -> Dict[str, Any]:
    """Run the required 5-instance pilot and, if it passes, final generation."""

    metadata_dir = Path(metadata_dir)
    output_dir = Path(output_dir)
    pilot_output_dir = Path(pilot_output_dir)
    pilot_metadata_dir = Path(pilot_metadata_dir)
    corpus = load_full_project_corpus(config.get("full_project_corpus_path"))
    records = list(corpus.get("records", []))
    expected_count = int(config.get("final_extension_count", FINAL_EXTENSION_COUNT))
    start_id = int(config.get("final_extension_start_id", FINAL_EXTENSION_START_ID))
    validate_extension_corpus(records, expected_count, start_id)

    fixed_hashes_before = fixed_type5_file_hashes(output_dir)
    if len(fixed_hashes_before) != start_id - 1:
        raise RuntimeError(
            f"Expected {start_id - 1} frozen fixed-window Type 5 files before extension; "
            f"found {len(fixed_hashes_before)}."
        )

    pilot_result: Dict[str, Any] | None = None
    if not skip_pilot:
        pilot_records = select_pilot_full_project_records(
            records,
            count=int(config.get("pilot_count", PILOT_EXTENSION_COUNT)),
            seed=seed,
        )
        pilot_rows, pilot_report = generate_full_project_rows(
            records=pilot_records,
            output_dir=pilot_output_dir,
            metadata_dir=pilot_metadata_dir,
            seed=derive_seed(seed, 91_005, salt=3_301_001),
            config=config,
            fixed_hashes_before=fixed_hashes_before,
            pilot=True,
            dry_run=dry_run,
            overwrite=overwrite_pilot,
        )
        pilot_gate = pilot_quality_gate(pilot_rows, pilot_report)
        pilot_report["pilot_quality_gate"] = pilot_gate
        pilot_result = {"rows": pilot_rows, "report": pilot_report, "gate": pilot_gate}
        if not dry_run:
            pilot_metadata_dir.mkdir(parents=True, exist_ok=True)
            write_manifest(pilot_metadata_dir / "type5_full_project_pilot_report.json", pilot_report)
            write_metadata(
                pilot_rows,
                pilot_metadata_dir / "type5_full_project_pilot_metadata.csv",
                pilot_metadata_dir / "type5_full_project_pilot_metadata.jsonl",
            )
            write_pilot_table(pilot_rows, pilot_metadata_dir / "type5_full_project_pilot_table.csv")
            write_manifest(pilot_metadata_dir / "type5_full_project_pilot_quality_gate.json", pilot_gate)
        if not pilot_gate["passed"]:
            raise RuntimeError("Type 5 full-project pilot failed: " + json.dumps(pilot_gate, sort_keys=True))

    final_rows, final_report = generate_full_project_rows(
        records=records,
        output_dir=output_dir,
        metadata_dir=metadata_dir,
        seed=seed,
        config=config,
        fixed_hashes_before=fixed_hashes_before,
        pilot=False,
        dry_run=dry_run,
        overwrite=overwrite_final,
    )
    fixed_hashes_after = fixed_type5_file_hashes(output_dir)
    final_gate = final_full_project_quality_gate(
        rows=final_rows,
        output_dir=output_dir,
        fixed_hashes_before=fixed_hashes_before,
        fixed_hashes_after=fixed_hashes_after,
        corpus=corpus,
        report_exists=not dry_run,
    )
    final_report["final_quality_gate"] = final_gate
    final_report["pilot_gate"] = pilot_result["gate"] if pilot_result else {"skipped": True}
    if not dry_run:
        metadata_dir.mkdir(parents=True, exist_ok=True)
        write_metadata(
            final_rows,
            metadata_dir / "type5_full_project_metadata.csv",
            metadata_dir / "type5_full_project_metadata.jsonl",
        )
        write_manifest(metadata_dir / "type5_full_project_statistical_report.json", final_report)
        write_source_to_output_mapping(final_rows, metadata_dir)
        write_data_provenance(final_rows, corpus, metadata_dir)
        write_combined_type5_summary(final_rows, metadata_dir)
        write_full_project_methodology_draft(
            metadata_dir / "type5_full_project_methodology_draft.md",
            selected_count=len(final_rows),
        )
        update_master_manifest(
            metadata_dir / "benchmark_manifest.json",
            "type5_full_project_extension",
            {
                "count": len(final_rows),
                "seed": seed,
                "config_hash": config_hash(config),
                "output_dir": str(output_dir),
                "metadata_csv": str(metadata_dir / "type5_full_project_metadata.csv"),
                "statistical_report": str(metadata_dir / "type5_full_project_statistical_report.json"),
                "empirical_corpus": str(config.get("full_project_corpus_path")),
                "fixed_type5_instances_preserved": final_gate["existing_type5_instances_unchanged"],
            },
        )
        (metadata_dir / "type5_full_project_config_used.json").write_text(
            json.dumps(config, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
    if not final_gate["passed"]:
        raise RuntimeError("Type 5 full-project final quality gate failed: " + json.dumps(final_gate, sort_keys=True))
    return {"pilot": pilot_result, "final_rows": final_rows, "final_report": final_report}


def generate_full_project_rows(
    records: Sequence[Mapping[str, Any]],
    output_dir: Path,
    metadata_dir: Path,
    seed: int,
    config: Mapping[str, Any],
    fixed_hashes_before: Mapping[Path, str],
    pilot: bool,
    dry_run: bool,
    overwrite: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    expected_paths = {
        (
            output_dir
            / (
                pilot_filename(idx, record["source_owner"], record["source_repo"])
                if pilot
                else full_project_filename(int(record["family_instance_id"]), record["source_owner"], record["source_repo"])
            )
        ).resolve()
        for idx, record in enumerate(records, start=1)
    }
    existing_hashes = hashes_for_duplicate_guard(output_dir, exclude_output_dir=overwrite)
    if not dry_run and not overwrite:
        existing_hashes.difference_update(_hashes_for_existing_expected_outputs(expected_paths))
    rows: List[Dict[str, Any]] = []
    feature_rows: List[Dict[str, float]] = []
    failed_attempts: List[Dict[str, Any]] = []
    accepted_hashes = set(existing_hashes) | set(fixed_hashes_before.values())
    for output_idx, record in enumerate(records, start=1):
        family_id = int(record["family_instance_id"])
        filename = (
            pilot_filename(output_idx, record["source_owner"], record["source_repo"])
            if pilot
            else full_project_filename(family_id, record["source_owner"], record["source_repo"])
        )
        params = GenerationParameters(**record["parameters"]).normalized()
        empirical_profile = empirical_profile_from_record(record)
        logging.info(
            "Type5 full-project %s/%s source=%s/%s",
            output_idx,
            len(records),
            record["source_owner"],
            record["source_repo"],
        )
        path = output_dir / filename
        if not dry_run and path.exists() and not overwrite:
            row, features, resumed_hash = resume_existing_full_project_row(
                path=path,
                filename=filename,
                family_id=family_id,
                root_seed=seed,
                record=record,
                params=params,
                empirical_profile=empirical_profile,
                config=config,
                accepted_hashes=accepted_hashes,
                pilot=pilot,
            )
            accepted_hashes.add(resumed_hash)
            feature_rows.append(features)
            rows.append(row)
            logging.info("resumed existing %s", filename)
            continue

        accepted = None
        last_quality = None
        for attempt in range(1, int(config.get("max_generation_attempts_per_source", 30)) + 1):
            attempt_seed = derive_seed(seed, family_id * 1000 + attempt, salt=17_845_091)
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
                    "reason": "type5_full_project_quality_gate_failed",
                    "quality": quality_summary(quality),
                }
            )
        if accepted is None:
            raise RuntimeError(
                f"Could not generate accepted full-project Type 5 instance for {filename}. "
                f"Last quality={json.dumps(quality_summary(last_quality or {}), sort_keys=True)}"
            )

        outcome, h, quality, outer_attempt = accepted
        features = extract_features(outcome.instance)
        feature_rows.append(features)
        if not dry_run:
            if path.exists() and not overwrite:
                raise FileExistsError(f"Refusing to overwrite existing full-project output: {path}")
            write_instance(outcome.instance, path, overwrite=overwrite)
            validation = validate_file(path, fillable_fraction=outcome.parameters.fillable_fraction)
            if not validation.ok:
                raise RuntimeError(f"Written full-project Type 5 instance failed validation: {path}: {validation.issues}")
        else:
            validation = outcome.validation
        row = build_full_project_metadata_row(
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
            pilot=pilot,
        )
        if pilot:
            row["pilot_index"] = output_idx
            row["pilot_source_family_instance_id"] = family_id
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
    report = full_project_statistical_report(
        rows,
        normalized_vectors,
        scaler.group_weights,
        failed_attempts,
        fixed_hashes_before,
        pilot=pilot,
    )
    return rows, report


def resume_existing_full_project_row(
    path: Path,
    filename: str,
    family_id: int,
    root_seed: int,
    record: Mapping[str, Any],
    params: GenerationParameters,
    empirical_profile,
    config: Mapping[str, Any],
    accepted_hashes: set[str],
    pilot: bool,
) -> Tuple[Dict[str, Any], Dict[str, float], str]:
    instance = parse_file(path)
    validation = validate_file(path, fillable_fraction=params.fillable_fraction)
    features = extract_features(instance)
    h = structural_hash(instance)
    duplicate_pass = h not in accepted_hashes
    quality = evaluate_type5_quality(
        instance,
        validation,
        record,
        duplicate_pass=duplicate_pass,
        config=config,
    )
    if not quality["overall_type5_quality_pass"]:
        raise RuntimeError(
            f"Existing full-project output no longer passes quality gates: {path}. "
            f"Quality={json.dumps(quality_summary(quality), sort_keys=True)}"
        )
    matched_seed, generator_version, generation_attempts = match_existing_generation_attempt(
        target_hash=h,
        filename=filename,
        family_id=family_id,
        root_seed=root_seed,
        params=params,
        empirical_profile=empirical_profile,
        config=config,
    )
    row = build_full_project_metadata_row(
        filename=filename,
        family_id=family_id,
        root_seed=root_seed,
        outcome_seed=matched_seed,
        generator_version=generator_version,
        parameters=params.to_dict(),
        features=features,
        validation=validation,
        record=record,
        structural_hash=h,
        quality=quality,
        generation_attempts=generation_attempts,
        pilot=pilot,
    )
    row["resumed_from_existing_file"] = True
    return row, features, h


def match_existing_generation_attempt(
    target_hash: str,
    filename: str,
    family_id: int,
    root_seed: int,
    params: GenerationParameters,
    empirical_profile,
    config: Mapping[str, Any],
) -> Tuple[int, str, int]:
    for attempt in range(1, int(config.get("max_generation_attempts_per_source", 30)) + 1):
        attempt_seed = derive_seed(root_seed, family_id * 1000 + attempt, salt=17_845_091)
        try:
            outcome = generate_valid_instance(
                params,
                attempt_seed,
                max_attempts=1,
                source_name=filename,
                empirical_profile=empirical_profile,
            )
        except RuntimeError:
            continue
        if structural_hash(outcome.instance) == target_hash:
            return outcome.seed, outcome.generator_version, attempt
    return 0, __version__, 0


def _hashes_for_existing_expected_outputs(paths: Iterable[Path]) -> set[str]:
    hashes: set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        try:
            hashes.add(structural_hash(parse_file(path)))
        except Exception:
            continue
    return hashes


def load_full_project_corpus(path: Any) -> Dict[str, Any]:
    if path is None:
        path = DEFAULT_CONFIG["full_project_corpus_path"]
    corpus_path = Path(str(path))
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Full-project Type 5 empirical corpus is missing: {corpus_path}. Run "
            "`python -m benchmark_generation.type5.build_full_project_profiles` first."
        )
    data = json.loads(corpus_path.read_text(encoding="utf-8"))
    if data.get("extension") != "type5_full_project" or not isinstance(data.get("records"), list):
        raise ValueError(f"Malformed Type 5 full-project corpus: {corpus_path}")
    return data


def validate_extension_corpus(records: Sequence[Mapping[str, Any]], expected_count: int, start_id: int) -> None:
    expected_ids = list(range(start_id, start_id + expected_count))
    actual_ids = sorted(int(record.get("family_instance_id", -1)) for record in records)
    if actual_ids != expected_ids:
        raise RuntimeError(f"Full-project corpus must contain contiguous IDs {expected_ids}; found {actual_ids}")
    repos = [f"{record.get('source_owner')}/{record.get('source_repo')}".lower() for record in records]
    if len(set(repos)) != expected_count:
        raise RuntimeError("Full-project corpus must contain exactly one record per selected repository")
    for record in records:
        if record.get("observation_mode") != FULL_PROJECT_OBSERVATION_MODE:
            raise RuntimeError(f"Record {record.get('family_instance_id')} is not full_project observation mode")
        if record.get("source_window_type") != FULL_PROJECT_WINDOW_TYPE:
            raise RuntimeError(f"Record {record.get('family_instance_id')} does not use complete eligible history")


def build_full_project_metadata_row(
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
    pilot: bool,
) -> Dict[str, Any]:
    row = build_type5_metadata_row(
        filename=filename,
        family_id=family_id,
        root_seed=root_seed,
        outcome_seed=outcome_seed,
        generator_version=generator_version,
        parameters=parameters,
        features=features,
        validation=validation,
        record=record,
        structural_hash=structural_hash,
        quality=quality,
        generation_attempts=generation_attempts,
    )
    row.update(
        {
            "extension": "type5_full_project",
            "pilot": pilot,
            "family_instance_label": f"type5_{family_id:03d}",
            "source_repository": record.get("source_repository", f"{record.get('source_owner')}/{record.get('source_repo')}"),
            "source_origin": record.get("source_origin", "GitHub_empirical"),
            "observation_mode": record.get("observation_mode", FULL_PROJECT_OBSERVATION_MODE),
            "source_window_type": record.get("source_window_type", FULL_PROJECT_WINDOW_TYPE),
            "project_history_start": record.get("project_history_start", ""),
            "project_history_end": record.get("project_history_end", ""),
            "project_history_days": record.get("project_history_days", ""),
            "project_history_months": record.get("project_history_months", ""),
            "project_lifetime_band": record.get("project_lifetime_band", ""),
            "technical_detail_mode": record.get("technical_detail_mode", ""),
        }
    )
    return row


def fixed_type5_file_hashes(output_dir: str | Path) -> Dict[Path, str]:
    root = Path(output_dir)
    hashes: Dict[Path, str] = {}
    if not root.exists():
        return hashes
    for path in sorted(root.glob("type5_*.txt")):
        match = FIXED_TYPE5_RE.match(path.name)
        if not match:
            continue
        family_id = int(match.group(1))
        if family_id < FINAL_EXTENSION_START_ID:
            hashes[path.resolve()] = sha256_file(path)
    return hashes


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hashes_for_duplicate_guard(output_dir: Path, exclude_output_dir: bool = False) -> set[str]:
    generated_root = output_dir.parent
    paths: List[Path] = []
    if generated_root.exists():
        for path in generated_root.glob("**/*.txt"):
            if exclude_output_dir and output_dir in path.parents:
                continue
            paths.append(path)
    paths.extend(Path("google_instances").glob("*.txt"))
    return structural_hashes_for_paths(paths)


def full_project_statistical_report(
    rows: Sequence[Mapping[str, Any]],
    normalized_vectors: Sequence[Mapping[str, float]],
    group_weights: Mapping[str, float],
    failed_attempts: Sequence[Mapping[str, Any]],
    fixed_hashes_before: Mapping[Path, str],
    pilot: bool,
) -> Dict[str, Any]:
    report = family_statistical_report(rows, normalized_vectors, group_weights)
    report.update(
        {
            "extension": "type5_full_project",
            "pilot": pilot,
            "failed_attempts": len(failed_attempts),
            "failed_attempt_samples": list(failed_attempts[:25]),
            "existing_fixed_type5_instances_before": len(fixed_hashes_before),
            "source_statistics": full_project_source_statistics(rows),
            "project_history_days": summarize_numeric(_row_values(rows, "project_history_days")),
            "project_history_months": summarize_numeric(_row_values(rows, "project_history_months")),
            "duration_statistics": summarize_numeric(_row_values(rows, "feature.duration_mean")),
            "mentorship_statistics": {
                "direct_fillability": summarize_numeric(_row_values(rows, "type5.quality_metrics.directly_fillable_share")),
                "one_level_below_potential": summarize_numeric(_row_values(rows, "feature.mentorship_fillability")),
                "global_mentor_availability": summarize_numeric(_row_values(rows, "feature.mentor_availability")),
                "true_same_project_mentorship_feasibility": summarize_numeric(
                    _row_values(rows, "type5.quality_metrics.truly_valid_initial_mentorship_feasible_share")
                ),
            },
            "score_effort_correlation": summarize_numeric(_row_values(rows, "feature.score_effort_correlation")),
            "score_proxy_methodology": "Effort-conditioned synthetic MTP project value; GitHub popularity/social metrics are not used.",
            "difficulty_prior": "Reuses the existing Type 5 Dirichlet prior centered on [0.25,0.45,0.20,0.10].",
        }
    )
    return report


def pilot_quality_gate(rows: Sequence[Mapping[str, Any]], report: Mapping[str, Any]) -> Dict[str, Any]:
    repos = {str(row.get("source_repository", "")) for row in rows}
    domains = {str(row.get("source_domain", "")) for row in rows}
    bands = {str(row.get("project_lifetime_band", "")) for row in rows}
    gate = {
        "pilot_accepted_count": len(rows),
        "pilot_accepted_count_exact": len(rows) == PILOT_EXTENSION_COUNT,
        "source_repositories_unique": len(repos) == len(rows),
        "domain_diversity_count": len(domains),
        "lifetime_band_diversity_count": len(bands),
        "all_structural_valid": all(_truthy(row.get("type5.structural_validation_pass")) for row in rows),
        "all_duplicate_checks_pass": all(_truthy(row.get("type5.duplicate_pass")) for row in rows),
        "all_source_correspondence_pass": all(_truthy(row.get("type5.source_distribution_match_pass")) for row in rows),
        "all_exact_mentorship_checks_pass": all(_truthy(row.get("type5.initial_realistic_feasibility_pass")) for row in rows),
        "all_overall_type5_quality_pass": all(_truthy(row.get("type5.overall_type5_quality_pass")) for row in rows),
    }
    gate["passed"] = all(
        [
            gate["pilot_accepted_count_exact"],
            gate["source_repositories_unique"],
            gate["all_structural_valid"],
            gate["all_duplicate_checks_pass"],
            gate["all_source_correspondence_pass"],
            gate["all_exact_mentorship_checks_pass"],
            gate["all_overall_type5_quality_pass"],
        ]
    )
    return gate


def final_full_project_quality_gate(
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    fixed_hashes_before: Mapping[Path, str],
    fixed_hashes_after: Mapping[Path, str],
    corpus: Mapping[str, Any],
    report_exists: bool,
) -> Dict[str, Any]:
    expected_count = int(
        corpus.get("selection_count")
        or corpus.get("final_expected_type5_full_project_count")
        or len(rows)
        or FINAL_EXTENSION_COUNT
    )
    start_id = int(corpus.get("output_family_id_start") or FINAL_EXTENSION_START_ID)
    expected_ids = list(range(start_id, start_id + expected_count))
    expected_combined_count = 91 + expected_count
    expected_end_id = start_id + expected_count - 1
    family_ids = sorted(int(row.get("family_instance_id", -1)) for row in rows)
    repositories = [str(row.get("source_repository", "")).lower() for row in rows]
    exclusion_records = corpus.get("existing_type5_repository_exclusion_list", [])
    overlaps = repository_overlap(repositories, exclusion_records)
    hashes = [str(row.get("type5.structural_hash", "")) for row in rows]
    fixed_unchanged = fixed_hashes_before == fixed_hashes_after
    new_files = [output_dir / str(row.get("filename")) for row in rows]
    gate = {
        "existing_type5_instances": len(fixed_hashes_before),
        "existing_type5_instances_exact_91": len(fixed_hashes_before) == 91,
        "existing_type5_instances_unchanged": fixed_unchanged,
        "new_full_project_instances": len(rows),
        "new_full_project_instances_exact_configured_count": len(rows) == expected_count,
        "combined_type5_instances": len(fixed_hashes_after) + len(rows),
        "combined_type5_instances_exact_configured_total": len(fixed_hashes_after) + len(rows) == expected_combined_count,
        "new_source_repositories_unique": len(set(repositories)),
        "new_source_repositories_unique_exact_configured_count": len(set(repositories)) == expected_count,
        "repository_overlap_with_existing_type5": len(overlaps),
        "overlapping_repositories": overlaps,
        "all_project_histories_at_least_90_days": all(float(row.get("project_history_days", 0) or 0) >= 90 for row in rows),
        "all_project_histories_at_most_548_days": all(float(row.get("project_history_days", 0) or 0) <= 548 for row in rows),
        "all_projects_use_complete_eligible_history": all(row.get("source_window_type") == FULL_PROJECT_WINDOW_TYPE for row in rows),
        "no_fixed_six_month_windows_in_extension": all(row.get("observation_mode") == FULL_PROJECT_OBSERVATION_MODE for row in rows),
        "family_ids_contiguous_from_configured_start": family_ids == expected_ids,
        "configured_output_family_id_start": start_id,
        "configured_output_family_id_end": expected_end_id,
        "unique_new_structural_hashes": len(set(hashes)),
        "unique_new_structural_hashes_exact_configured_count": len(set(hashes)) == expected_count,
        "new_duplicate_count": sum(1 for row in rows if not _truthy(row.get("type5.duplicate_pass"))),
        "all_new_structurally_valid": all(_truthy(row.get("type5.structural_validation_pass")) for row in rows),
        "all_new_duplicate_checks_pass": all(_truthy(row.get("type5.duplicate_pass")) for row in rows),
        "all_new_source_correspondence_pass": all(_truthy(row.get("type5.source_distribution_match_pass")) for row in rows),
        "all_new_exact_mentorship_checks_pass": all(_truthy(row.get("type5.initial_realistic_feasibility_pass")) for row in rows),
        "all_new_source_provenance_complete": all(_source_provenance_complete(row) for row in rows),
        "all_new_metadata_complete": all(row.get("seed") not in ("", None) and row.get("structural_hash") not in ("", None) for row in rows),
        "all_new_generation_seeds_recorded": all(row.get("seed") not in ("", None) and row.get("root_seed") not in ("", None) for row in rows),
        "all_new_output_ids_contiguous_from_configured_start": family_ids == expected_ids,
        "all_new_source_to_output_mappings_one_to_one": len(set(repositories)) == len(rows),
        "no_source_identifiers_inside_MTP_contents": all(final_instance_names_are_anonymized(path) for path in new_files if path.exists()),
        "shared_generator_regression_check_pass": fixed_unchanged,
        "statistical_report_exists": report_exists,
    }
    gate["passed"] = all(
        [
            gate["existing_type5_instances_exact_91"],
            gate["existing_type5_instances_unchanged"],
            gate["new_full_project_instances_exact_configured_count"],
            gate["combined_type5_instances_exact_configured_total"],
            gate["new_source_repositories_unique_exact_configured_count"],
            gate["repository_overlap_with_existing_type5"] == 0,
            gate["all_project_histories_at_least_90_days"],
            gate["all_project_histories_at_most_548_days"],
            gate["all_projects_use_complete_eligible_history"],
            gate["no_fixed_six_month_windows_in_extension"],
            gate["unique_new_structural_hashes_exact_configured_count"],
            gate["new_duplicate_count"] == 0,
            gate["all_new_structurally_valid"],
            gate["all_new_duplicate_checks_pass"],
            gate["all_new_source_correspondence_pass"],
            gate["all_new_exact_mentorship_checks_pass"],
            gate["all_new_source_provenance_complete"],
            gate["all_new_metadata_complete"],
            gate["all_new_generation_seeds_recorded"],
            gate["all_new_output_ids_contiguous_from_configured_start"],
            gate["all_new_source_to_output_mappings_one_to_one"],
            gate["no_source_identifiers_inside_MTP_contents"],
            gate["shared_generator_regression_check_pass"],
            gate["statistical_report_exists"],
        ]
    )
    return gate


def write_pilot_table(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    fields = [
        "pilot_index",
        "pilot_source_family_instance_id",
        "filename",
        "source_repository",
        "source_domain",
        "project_history_days",
        "project_history_months",
        "feature.n_contributors",
        "feature.n_projects",
        "feature.n_skills",
        "feature.total_roles",
        "type5.quality_metrics.directly_fillable_share",
        "type5.quality_metrics.truly_valid_initial_mentorship_feasible_share",
        "type5.quality_metrics.measured_p_future",
        "type5.structural_validation_pass",
        "type5.duplicate_pass",
        "type5.source_distribution_match_pass",
        "type5.overall_type5_quality_pass",
    ]
    _write_rows(rows, path, fields)


def write_source_to_output_mapping(rows: Sequence[Mapping[str, Any]], metadata_dir: Path) -> None:
    fields = [
        "family_instance_id",
        "family_instance_label",
        "output_filename",
        "source_repository",
        "source_repository_id",
        "source_url",
        "source_snapshot_sha",
        "source_snapshot_timestamp",
        "project_history_start",
        "project_history_end",
        "project_history_days",
        "project_history_months",
        "generation_seed",
        "generation_attempt",
        "structural_hash",
    ]
    mapping_rows = []
    for row in rows:
        mapping_rows.append(
            {
                "family_instance_id": row.get("family_instance_id", ""),
                "family_instance_label": row.get("family_instance_label", ""),
                "output_filename": row.get("filename", ""),
                "source_repository": row.get("source_repository", ""),
                "source_repository_id": row.get("source_repository_id", ""),
                "source_url": row.get("source_url", ""),
                "source_snapshot_sha": row.get("source_snapshot_sha", ""),
                "source_snapshot_timestamp": row.get("source_collection_timestamp", ""),
                "project_history_start": row.get("project_history_start", ""),
                "project_history_end": row.get("project_history_end", ""),
                "project_history_days": row.get("project_history_days", ""),
                "project_history_months": row.get("project_history_months", ""),
                "generation_seed": row.get("seed", ""),
                "generation_attempt": row.get("validation.generation_attempts", ""),
                "structural_hash": row.get("structural_hash", ""),
            }
        )
    _write_rows(mapping_rows, metadata_dir / "type5_full_project_source_to_output_mapping.csv", fields)
    (metadata_dir / "type5_full_project_source_to_output_mapping.json").write_text(
        json.dumps(mapping_rows, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def write_data_provenance(rows: Sequence[Mapping[str, Any]], corpus: Mapping[str, Any], metadata_dir: Path) -> None:
    payload = {
        "extension": "type5_full_project",
        "source_origin": "GitHub_empirical",
        "observation_mode": FULL_PROJECT_OBSERVATION_MODE,
        "source_window_type": FULL_PROJECT_WINDOW_TYPE,
        "candidate_count": corpus.get("candidate_count", ""),
        "valid_candidate_count": corpus.get("valid_candidate_count", ""),
        "selected_count": len(rows),
        "repository_overlap_with_existing_type5": corpus.get("repository_overlap_with_existing_type5", ""),
        "api_and_cache_note": "GitHub API data are collected by collect_github_full_projects.py and cached locally; API tokens are not stored.",
        "privacy_note": "Final MTP text files contain only anonymous contributors, skills, and projects.",
        "source_repositories": [row.get("source_repository", "") for row in rows],
    }
    (metadata_dir / "type5_full_project_data_provenance.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def write_combined_type5_summary(rows: Sequence[Mapping[str, Any]], metadata_dir: Path) -> None:
    existing_path = metadata_dir / "type5_metadata.csv"
    existing_rows = _read_csv(existing_path) if existing_path.exists() else []
    summary = {
        "fixed_6_month_window_count": len(existing_rows),
        "full_project_count": len(rows),
        "combined_type5_count": len(existing_rows) + len(rows),
        "subgroups": {
            "fixed_6_month_window": _metadata_summary(existing_rows),
            "full_project": _metadata_summary(rows),
        },
        "note": "Combined statistics preserve the distinction between fixed-window Type 5 and full-project extension instances.",
    }
    (metadata_dir / "type5_full_project_family_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _metadata_summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "contributors": summarize_numeric(_row_values(rows, "feature.n_contributors")),
        "projects": summarize_numeric(_row_values(rows, "feature.n_projects")),
        "skills": summarize_numeric(_row_values(rows, "feature.n_skills")),
        "roles_per_project": summarize_numeric(_row_values(rows, "feature.roles_per_project_mean")),
        "duration": summarize_numeric(_row_values(rows, "feature.duration_mean")),
        "skill_skew": summarize_numeric(_row_values(rows, "source_skill_skew")),
        "alignment": summarize_numeric(_row_values(rows, "feature.demand_supply_alignment")),
        "contention": summarize_numeric(_row_values(rows, "feature.contention")),
        "fillability": summarize_numeric(_row_values(rows, "feature.direct_fillability")),
        "mentorship": summarize_numeric(_row_values(rows, "feature.mentorship_fillability")),
    }


def _write_rows(rows: Sequence[Mapping[str, Any]], path: str | Path, fields: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _row_values(rows: Sequence[Mapping[str, Any]], key: str) -> List[float]:
    values = []
    for row in rows:
        value = row.get(key, "")
        if value in ("", None):
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def _source_provenance_complete(row: Mapping[str, Any]) -> bool:
    required = [
        "source_repository",
        "source_repository_id",
        "source_url",
        "source_snapshot_sha",
        "source_collection_timestamp",
        "project_history_start",
        "project_history_end",
        "project_history_days",
        "project_history_months",
    ]
    return all(row.get(field) not in ("", None) for field in required)


def _truthy(value: Any) -> bool:
    return value is True or str(value).lower() == "true"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="generated_instances/type5")
    parser.add_argument("--metadata-dir", default="benchmark_metadata")
    parser.add_argument("--pilot-output-dir", default="generated_instances/type5_full_project_pilot")
    parser.add_argument("--pilot-metadata-dir", default="benchmark_metadata/type5_full_project_pilot")
    parser.add_argument("--config", default="benchmark_generation/configs/type5_full_project.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite-pilot", action="store_true")
    parser.add_argument("--overwrite-final", action="store_true")
    parser.add_argument("--skip-pilot", action="store_true")
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
    if args.max_attempts is not None:
        config["max_generation_attempts_per_source"] = args.max_attempts
    seed = args.seed if args.seed is not None else int(config.get("root_seed", 2026))
    generate_type5_full_project_extension(
        output_dir=args.output_dir,
        metadata_dir=args.metadata_dir,
        pilot_output_dir=args.pilot_output_dir,
        pilot_metadata_dir=args.pilot_metadata_dir,
        config=config,
        seed=seed,
        dry_run=args.dry_run,
        overwrite_pilot=args.overwrite_pilot,
        overwrite_final=args.overwrite_final,
        skip_pilot=args.skip_pilot,
    )


if __name__ == "__main__":
    main()
