"""Build empirical profiles for the Type 5 full-project extension."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from benchmark_generation.core.config import load_config
from benchmark_generation.core.generator import derive_seed
from benchmark_generation.core.metadata import config_hash
from benchmark_generation.type5.build_empirical_profiles import _corpus_deadline_bounds, _global_score_k
from benchmark_generation.type5.collect_github_corpus import utc_now
from benchmark_generation.type5.empirical import derive_window_measurement
from benchmark_generation.type5.full_project import (
    FINAL_EXTENSION_COUNT,
    FINAL_EXTENSION_START_ID,
    FULL_PROJECT_WINDOW_NAME,
    augment_full_project_measurement,
    assignment_family_ids,
    canonical_repo,
    existing_type5_repository_exclusion_records,
    finalize_full_project_source_profile,
    full_project_candidate_batch_from_source,
    full_project_source_statistics,
    full_project_window,
    repository_overlap,
    select_diverse_full_project_sources,
    source_repo_from_mapping,
    write_candidates_csv,
    write_exclusion_list,
    write_full_project_methodology_draft,
    write_selection_files,
)
from benchmark_generation.type5.sources import SourceRepo, repo_cache_dir


DEFAULT_CONFIG: Dict[str, Any] = {
    "root_seed": 2026,
    "selection_count": FINAL_EXTENSION_COUNT,
    "project_history_days_range": [90, 548],
    "default_tightness_bounds": [0.8, 4.0],
    "score_density_target": 2.0,
}


def build_full_project_profiles(
    cache_dir: str | Path,
    metadata_dir: str | Path,
    config: Dict[str, Any],
    bot_config: Mapping[str, Any],
    seed: int,
) -> Dict[str, Any]:
    metadata_dir = Path(metadata_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    exclusion_records = existing_type5_repository_exclusion_records(metadata_dir)
    write_exclusion_list(exclusion_records, metadata_dir)
    summaries = load_full_project_collection_summaries(cache_dir, metadata_dir)
    if not summaries:
        raise RuntimeError(
            "No collected full-project candidates found. Run "
            "`python -m benchmark_generation.type5.collect_github_full_projects` first."
        )

    measurements: List[Dict[str, Any]] = []
    for idx, summary in enumerate(summaries, start=1):
        source = source_repo_from_mapping(summary.get("source", {}), index=idx)
        if not summary.get("valid"):
            measurements.append(collection_failure_measurement(summary))
            continue
        window = full_project_window(
            str(summary.get("project_history_start")),
            str(summary.get("project_history_end")),
        )
        measurement_seed = derive_seed(seed, idx, salt=7_017_509)
        measurement = derive_window_measurement(
            source,
            window,
            cache_dir,
            config,
            bot_config,
            seed=measurement_seed,
        )
        measurement = augment_full_project_measurement(measurement, summary, config)
        measurements.append(measurement)
        if measurement.get("valid"):
            logging.info(
                "Type5 full-project profile READY %s contributors=%s skills=%s days=%.1f",
                source.full_name,
                measurement.get("human_contributor_count"),
                measurement.get("skill_count"),
                float(measurement.get("project_history_days", 0.0) or 0.0),
            )
        else:
            logging.warning(
                "Type5 full-project profile failed %s reason=%s",
                source.full_name,
                measurement.get("failure_reason"),
            )

    write_candidates_csv(measurements, metadata_dir / "type5_full_project_candidates.csv")
    valid_measurements = [item for item in measurements if item.get("valid")]
    valid_unique_repositories = {
        canonical_repo(
            str((item.get("source", {}) or {}).get("owner", "")),
            str((item.get("source", {}) or {}).get("repo", "")),
        )
        for item in valid_measurements
    }
    selection_count = int(config.get("selection_count", FINAL_EXTENSION_COUNT))
    if len(valid_unique_repositories) < selection_count:
        raise RuntimeError(
            f"Only {len(valid_unique_repositories)} valid unique full-project source profiles are available "
            f"after {len(measurements)} inspected or resumed candidates "
            f"(configured candidate_target={config.get('candidate_target', 'unknown')}). "
            "Stop here and inspect benchmark_metadata/type5_full_project_candidates.csv before increasing the pool."
        )
    selected_measurements = select_diverse_full_project_sources(
        valid_measurements,
        count=selection_count,
        seed=seed,
    )
    selected_measurements = assignment_family_ids(selected_measurements)
    corpus_deadline_bounds = _corpus_deadline_bounds(valid_measurements, config)
    global_score_k = _global_score_k(valid_measurements, config)

    records = []
    for selection_rank, measurement in enumerate(selected_measurements, start=1):
        family_id = FINAL_EXTENSION_START_ID + selection_rank - 1
        profile_seed = derive_seed(seed, family_id, salt=9_993_901)
        records.append(
            finalize_full_project_source_profile(
                measurement,
                config,
                seed=profile_seed,
                family_instance_id=family_id,
                selection_rank=selection_rank,
                corpus_deadline_bounds=corpus_deadline_bounds,
                global_score_k=global_score_k,
            )
        )

    selected_repositories = [canonical_repo(record["source_owner"], record["source_repo"]) for record in records]
    overlap = repository_overlap(selected_repositories, exclusion_records)
    if overlap:
        raise RuntimeError(f"Selected full-project repositories overlap existing Type 5: {overlap}")

    corpus = {
        "family": "type5",
        "extension": "type5_full_project",
        "source_origin": "GitHub_empirical",
        "observation_mode": "full_project",
        "source_window_type": "complete_eligible_project_history",
        "root_seed": seed,
        "built_at": utc_now(),
        "cache_dir": str(cache_dir),
        "config_hash": config_hash(config),
        "selection_count": len(records),
        "final_expected_type5_full_project_count": int(config.get("selection_count", FINAL_EXTENSION_COUNT)),
        "output_family_id_start": FINAL_EXTENSION_START_ID,
        "output_family_id_end": FINAL_EXTENSION_START_ID + len(records) - 1,
        "candidate_count": len(measurements),
        "valid_candidate_count": len(valid_measurements),
        "valid_unique_candidate_count": len(valid_unique_repositories),
        "rejected_candidate_count": len(measurements) - len(valid_measurements),
        "rejection_reasons": _reason_counts(measurements),
        "existing_type5_repository_exclusion_list": list(exclusion_records),
        "repository_overlap_with_existing_type5": len(overlap),
        "records": records,
        "source_statistics": full_project_source_statistics(records),
        "global_calibration": {
            "deadline_tightness_p10": corpus_deadline_bounds[0],
            "deadline_tightness_p90": corpus_deadline_bounds[1],
            "score_effort_normalization_k": global_score_k,
            "score_density_target": float(config.get("score_density_target", 2.0)),
        },
        "methodology_note": (
            "This full-project extension uses complete eligible project histories "
            "and does not modify the accepted fixed-window Type 5 files."
        ),
    }
    if len(records) != int(config.get("selection_count", FINAL_EXTENSION_COUNT)):
        raise RuntimeError(
            f"Full-project selection produced {len(records)} records, "
            f"expected {config.get('selection_count', FINAL_EXTENSION_COUNT)}."
        )
    write_full_project_profile_outputs(corpus, measurements, metadata_dir)
    return corpus


def load_full_project_collection_summaries(
    cache_dir: str | Path,
    metadata_dir: str | Path,
) -> List[Mapping[str, Any]]:
    collection_path = Path(metadata_dir) / "type5_full_project_collection.json"
    if collection_path.exists():
        payload = json.loads(collection_path.read_text(encoding="utf-8"))
        if isinstance(payload.get("candidates"), list):
            return list(payload["candidates"])

    root = Path(cache_dir)
    summaries = []
    if root.exists():
        for path in sorted(root.glob("*/full_project_collection_summary.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                summaries.append(data)
    return summaries


def collection_failure_measurement(summary: Mapping[str, Any]) -> Dict[str, Any]:
    source = summary.get("source", {})
    failure_reason = summary.get("rejection_reason") or summary.get("failure_reason", "collection_failed")
    return {
        "valid": False,
        "failure_reason": failure_reason,
        "rejection_reason": failure_reason,
        "candidate_batch": summary.get("candidate_batch") or full_project_candidate_batch_from_source(source),
        "cheap_prescreen_pass": summary.get("cheap_prescreen_pass", ""),
        "detailed_collection_attempted": summary.get("detailed_collection_attempted", ""),
        "profile_quality_pass": False,
        "source": dict(source),
        "window": {
            "name": FULL_PROJECT_WINDOW_NAME,
            "start": summary.get("project_history_start", ""),
            "end": summary.get("project_history_end", ""),
        },
        "repository_id": summary.get("repository_id", ""),
        "project_history_start": summary.get("project_history_start", ""),
        "project_history_end": summary.get("project_history_end", ""),
        "project_history_days": summary.get("project_history_days", 0.0),
        "project_history_months": summary.get("project_history_months", 0.0),
        "full_history_merged_pr_count": summary.get("full_history_merged_pr_count", 0),
        "full_history_human_pr_count": summary.get("full_history_human_pr_count", 0),
        "human_contributor_count": summary.get("plausible_human_contributor_count", ""),
        "merged_pr_count": summary.get("full_history_merged_pr_count", ""),
        "sampled_pr_count": summary.get("detailed_pr_count", ""),
        "technical_detail_mode": summary.get("technical_detail_mode", ""),
        "rejection_endpoint": summary.get("rejection_endpoint", ""),
        "rejection_http_status": summary.get("rejection_http_status", ""),
        "rejection_timestamp": summary.get("rejection_timestamp", ""),
        "rejection_stage": summary.get("rejection_stage", ""),
    }


def write_full_project_profile_outputs(
    corpus: Mapping[str, Any],
    measurements: Sequence[Mapping[str, Any]],
    metadata_dir: Path,
) -> None:
    (metadata_dir / "type5_full_project_empirical_corpus.json").write_text(
        json.dumps(corpus, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    write_candidates_csv(measurements, metadata_dir / "type5_full_project_candidates.csv")
    write_selection_files(
        corpus["records"],
        corpus["existing_type5_repository_exclusion_list"],
        metadata_dir,
    )
    write_full_project_methodology_draft(
        metadata_dir / "type5_full_project_methodology_draft.md",
        selected_count=len(corpus["records"]),
    )
    provenance = {
        "extension": "type5_full_project",
        "built_at": corpus["built_at"],
        "candidate_count": corpus["candidate_count"],
        "valid_candidate_count": corpus["valid_candidate_count"],
        "selected_count": corpus["selection_count"],
        "repository_overlap_with_existing_type5": corpus["repository_overlap_with_existing_type5"],
        "source_origin": corpus["source_origin"],
        "observation_mode": corpus["observation_mode"],
        "source_window_type": corpus["source_window_type"],
        "global_calibration": corpus["global_calibration"],
        "privacy_note": "Final MTP files are anonymous; public usernames and semantic skill tokens are not written to .txt instances.",
    }
    (metadata_dir / "type5_full_project_data_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _reason_counts(measurements: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for measurement in measurements:
        if measurement.get("valid"):
            continue
        reason = str(measurement.get("failure_reason") or "unknown")
        for part in reason.split(";"):
            if not part:
                continue
            counts[part] = counts.get(part, 0) + 1
    return dict(sorted(counts.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default="type5_full_project_source_cache")
    parser.add_argument("--metadata-dir", default="benchmark_metadata")
    parser.add_argument("--config", default="benchmark_generation/configs/type5_full_project.yaml")
    parser.add_argument("--bot-config", default="benchmark_generation/configs/type5_bot_patterns.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config = load_config(args.config, DEFAULT_CONFIG)
    bot_config = load_config(args.bot_config, {})
    seed = args.seed if args.seed is not None else int(config.get("root_seed", 2026))
    build_full_project_profiles(
        cache_dir=args.cache_dir,
        metadata_dir=args.metadata_dir,
        config=config,
        bot_config=bot_config,
        seed=seed,
    )


if __name__ == "__main__":
    main()
