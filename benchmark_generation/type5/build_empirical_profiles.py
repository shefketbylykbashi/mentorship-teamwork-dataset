"""Derive Type 5 empirical source configurations from cached GitHub data."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

from benchmark_generation.core.config import load_config
from benchmark_generation.core.generator import derive_seed
from benchmark_generation.core.metadata import config_hash
from benchmark_generation.type5.collect_github_corpus import utc_now
from benchmark_generation.type5.empirical import (
    derive_window_measurement,
    finalize_source_profile,
    percentile,
)
from benchmark_generation.type5.sources import (
    SourceRepo,
    collection_sources,
    iter_primary_slots,
    load_type5_sources,
)


DEFAULT_CONFIG: Dict[str, Any] = {
    "root_seed": 2026,
    "count": 100,
    "default_tightness_bounds": [0.8, 4.0],
    "score_density_target": 2.0,
}


def build_empirical_profiles(
    sources_path: str | Path,
    cache_dir: str | Path,
    metadata_dir: str | Path,
    config: Dict[str, Any],
    bot_config: Mapping[str, Any],
    seed: int,
    allow_partial: bool = False,
) -> Dict[str, Any]:
    sources = load_type5_sources(sources_path)
    metadata_dir = Path(metadata_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    measurements: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    all_measurements: List[Dict[str, Any]] = []
    for repo in collection_sources(sources):
        for window_idx, window in enumerate(sources["windows"], start=1):
            measurement_seed = derive_seed(seed, repo.index * 10 + window_idx, salt=7_001_111)
            measurement = derive_window_measurement(
                repo,
                window,
                cache_dir,
                config,
                bot_config,
                seed=measurement_seed,
            )
            key = (repo.kind, repo.full_name, window.name)
            measurements[key] = measurement
            all_measurements.append(measurement)
            if measurement.get("valid"):
                logging.info(
                    "Type5 source profile READY %s %s contributors=%s skills=%s profiles=%s",
                    repo.full_name,
                    window.name,
                    measurement.get("human_contributor_count"),
                    measurement.get("skill_count"),
                    measurement.get("profile_count"),
                )
            else:
                logging.warning(
                    "Type5 source profile failed %s %s reason=%s",
                    repo.full_name,
                    window.name,
                    measurement.get("failure_reason"),
                )

    valid_measurements = [measurement for measurement in all_measurements if measurement.get("valid")]
    corpus_deadline_bounds = _corpus_deadline_bounds(valid_measurements, config)
    global_score_k = _global_score_k(valid_measurements, config)
    records: List[Dict[str, Any]] = []
    replacements: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    used_reserve_windows: set[Tuple[str, str]] = set()
    reserves: List[SourceRepo] = list(sources.get("reserve", []))

    for slot_id, primary, window, window_idx in iter_primary_slots(sources):
        primary_measurement = measurements.get(("primary", primary.full_name, window.name))
        chosen = primary_measurement
        source_replacement = False
        original_source = None
        replacement_reason = ""
        if not chosen or not chosen.get("valid"):
            original_source = asdict(primary)
            replacement_reason = str((chosen or {}).get("failure_reason") or "primary_source_window_unavailable")
            chosen = None
            for reserve in reserves:
                reserve_key = (reserve.full_name, window.name)
                if reserve_key in used_reserve_windows:
                    continue
                candidate = measurements.get(("reserve", reserve.full_name, window.name))
                if candidate and candidate.get("valid"):
                    chosen = candidate
                    source_replacement = True
                    used_reserve_windows.add(reserve_key)
                    replacements.append(
                        {
                            "family_instance_id": slot_id,
                            "failed_primary_source": primary.full_name,
                            "failure_reason": replacement_reason,
                            "replacement_source": reserve.full_name,
                            "window": window.name,
                            "replacement_date": utc_now(),
                            "deterministic_replacement_order": [repo.full_name for repo in reserves],
                        }
                    )
                    break
        if not chosen or not chosen.get("valid"):
            failures.append(
                {
                    "family_instance_id": slot_id,
                    "source": primary.full_name,
                    "window": window.name,
                    "failure_reason": replacement_reason or "no_valid_replacement_available",
                }
            )
            continue

        profile_seed = derive_seed(seed, slot_id * 10 + window_idx, salt=9_900_991)
        records.append(
            finalize_source_profile(
                chosen,
                config,
                seed=profile_seed,
                family_instance_id=slot_id,
                source_replacement=source_replacement,
                original_source=original_source,
                replacement_reason=replacement_reason,
                corpus_deadline_bounds=corpus_deadline_bounds,
                global_score_k=global_score_k,
            )
        )

    corpus = {
        "family": "type5",
        "root_seed": seed,
        "built_at": utc_now(),
        "sources_path": str(sources_path),
        "cache_dir": str(cache_dir),
        "config_hash": config_hash(config),
        "api_version": sources["api_version"],
        "windows": [asdict(window) for window in sources["windows"]],
        "records": sorted(records, key=lambda item: int(item["family_instance_id"])),
        "replacements": replacements,
        "failed_source_windows": failures,
        "global_calibration": {
            "deadline_tightness_p10": corpus_deadline_bounds[0],
            "deadline_tightness_p90": corpus_deadline_bounds[1],
            "score_effort_normalization_k": global_score_k,
            "score_density_target": float(config.get("score_density_target", 2.0)),
        },
        "privacy_note": "Records contain anonymized local contributor aggregates only; GitHub usernames are not included.",
    }

    write_empirical_outputs(corpus, all_measurements, sources, metadata_dir)
    expected_count = int(config.get("count", 100))
    if len(records) != expected_count and not allow_partial:
        raise RuntimeError(
            f"Type 5 empirical profile build produced {len(records)} valid source configurations; "
            f"expected {expected_count}. See type5_source_windows.csv for failed windows."
        )
    return corpus


def write_empirical_outputs(
    corpus: Mapping[str, Any],
    measurements: List[Mapping[str, Any]],
    sources: Mapping[str, Any],
    metadata_dir: Path,
) -> None:
    (metadata_dir / "type5_empirical_corpus.json").write_text(
        json.dumps(corpus, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    _write_records_csv(corpus["records"], metadata_dir / "type5_empirical_corpus.csv")
    _write_measurements_csv(measurements, metadata_dir / "type5_source_windows.csv")
    _write_sources_csv(sources, metadata_dir / "type5_sources.csv")
    _write_provenance_json(corpus, measurements, metadata_dir / "type5_data_provenance.json")
    _write_sources_markdown(corpus, metadata_dir / "TYPE5_SOURCES.md")
    _write_methodology_draft(corpus, metadata_dir / "type5_methodology_draft.md")


def _write_records_csv(records: List[Mapping[str, Any]], path: Path) -> None:
    fieldnames = [
        "family_instance_id",
        "source_owner",
        "source_repo",
        "source_domain",
        "source_window_name",
        "source_replacement",
        "source_merged_pr_count",
        "source_sampled_pr_count",
        "source_human_contributor_count",
        "source_skill_count",
        "source_profile_count",
        "source_skill_skew",
        "source_alignment",
        "source_duration_median",
        "parameters.n_contributors",
        "parameters.n_projects",
        "parameters.n_skills",
        "parameters.n_profiles",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {
                "family_instance_id": record["family_instance_id"],
                "source_owner": record["source_owner"],
                "source_repo": record["source_repo"],
                "source_domain": record["source_domain"],
                "source_window_name": record["source_window_name"],
                "source_replacement": record["source_replacement"],
            }
            row.update(record["source_metrics"])
            row.update({f"parameters.{key}": value for key, value in record["parameters"].items()})
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_measurements_csv(measurements: List[Mapping[str, Any]], path: Path) -> None:
    fieldnames = [
        "kind",
        "owner",
        "repo",
        "domain",
        "window",
        "valid",
        "failure_reason",
        "merged_pr_count",
        "sampled_pr_count",
        "human_contributor_count",
        "bot_count",
        "skill_count",
        "profile_count",
        "source_skill_skew",
        "source_alignment",
        "deadline_proxy_coverage",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for measurement in measurements:
            source = measurement.get("source", {})
            window = measurement.get("window", {})
            writer.writerow(
                {
                    "kind": source.get("kind", ""),
                    "owner": source.get("owner", ""),
                    "repo": source.get("repo", ""),
                    "domain": source.get("domain", ""),
                    "window": window.get("name", ""),
                    "valid": measurement.get("valid", False),
                    "failure_reason": measurement.get("failure_reason", ""),
                    "merged_pr_count": measurement.get("merged_pr_count", ""),
                    "sampled_pr_count": measurement.get("sampled_pr_count", ""),
                    "human_contributor_count": measurement.get("human_contributor_count", ""),
                    "bot_count": measurement.get("bot_count", ""),
                    "skill_count": measurement.get("skill_count", ""),
                    "profile_count": measurement.get("profile_count", ""),
                    "source_skill_skew": measurement.get("source_skill_skew", ""),
                    "source_alignment": measurement.get("source_alignment", ""),
                    "deadline_proxy_coverage": measurement.get("deadline_proxy_coverage", ""),
                }
            )


def _write_sources_csv(sources: Mapping[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["kind", "index", "owner", "repo", "url", "domain"])
        writer.writeheader()
        for repo in collection_sources(sources):
            writer.writerow(asdict(repo))


def _write_provenance_json(
    corpus: Mapping[str, Any],
    measurements: List[Mapping[str, Any]],
    path: Path,
) -> None:
    provenance = {
        "api_version": corpus["api_version"],
        "built_at": corpus["built_at"],
        "source_configuration_count": len(corpus["records"]),
        "replacement_count": len(corpus["replacements"]),
        "failed_source_window_count": len(corpus["failed_source_windows"]),
        "api_records_collected": {
            "merged_prs": sum(int(item.get("merged_pr_count", 0) or 0) for item in measurements),
            "sampled_prs": sum(int(item.get("sampled_pr_count", 0) or 0) for item in measurements),
            "technical_work_units": sum(int(item.get("technical_work_unit_count", 0) or 0) for item in measurements),
        },
        "privacy": "No tokens or public GitHub usernames are written to Type 5 public metadata.",
        "global_calibration": corpus["global_calibration"],
    }
    path.write_text(json.dumps(provenance, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_sources_markdown(corpus: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Type 5 Source Provenance",
        "",
        "Type 5 uses public GitHub activity as aggregate empirical input. Final MTP files remain synthetic and anonymized.",
        "",
        f"Built at: {corpus['built_at']}",
        f"API version: {corpus['api_version']}",
        "",
    ]
    for record in corpus["records"]:
        replacement = "replacement" if record["source_replacement"] else "primary"
        lines.extend(
            [
                f"## type5_{int(record['family_instance_id']):03d}: {record['source_owner']}/{record['source_repo']}",
                "",
                f"- URL: {record['source_url']}",
                f"- Domain: {record['source_domain']}",
                f"- Observation window: {record['source_window_name']} [{record['source_window_start']}, {record['source_window_end']})",
                f"- Snapshot SHA: {record['source_snapshot_sha']}",
                f"- Collection timestamp: {record['source_collection_timestamp']}",
                f"- License identifier: {record.get('source_license', '')}",
                f"- Source role: {replacement}",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_methodology_draft(corpus: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Type 5 Methodology Draft",
        "",
        "Type 5 grounds the synthetic benchmark in empirically observed software-development configurations.",
        "Public GitHub activity is collected from mature open-source repositories, split into four fixed six-month UTC observation windows.",
        "Merged pull requests are treated as empirical work-package proxies, not literal MTP projects.",
        "",
        "## Empirically Observed",
        "",
        "- merged PR counts and sampled PR file paths",
        "- active public contributor activity after bot filtering",
        "- changed technical areas mapped to language/subsystem skill tokens",
        "- PR cycle times",
        "- repository language composition",
        "- release and milestone dates where available",
        "",
        "## Derived Proxies",
        "",
        "- contributor activity scores use 0.60 log-lines, 0.25 log-files, and 0.15 log-PR counts",
        "- skill levels are relative empirical activity-based skill-level proxies on the MTP 1..10 scale",
        "- merged PRs are work-package proxies",
        "- deadline tightness uses milestone due dates, then stable releases, then a corpus-level fallback",
        "",
        "## Synthetic Modeling Decisions",
        "",
        "- MTP scores are effort-conditioned synthetic proxies, not observed project value",
        "- Type 5 difficulty probabilities are sampled from a Dirichlet prior centered on [0.25, 0.45, 0.20, 0.10]",
        "- source profiles parameterize the same forward generator used by other synthetic families",
        "- final instances are accepted only after structural, duplicate, provenance, and realism gates pass",
        "",
        f"Global score normalization K: {corpus['global_calibration']['score_effort_normalization_k']}",
        f"Corpus deadline fallback P10/P90: {corpus['global_calibration']['deadline_tightness_p10']}, {corpus['global_calibration']['deadline_tightness_p90']}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _corpus_deadline_bounds(
    measurements: List[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> Tuple[float, float]:
    values: List[float] = []
    for measurement in measurements:
        values.extend(float(value) for value in measurement.get("deadline_observations", []))
    if not values:
        fallback = config.get("default_tightness_bounds", [0.8, 4.0])
        return float(fallback[0]), float(fallback[1])
    return percentile(values, 0.10), percentile(values, 0.90)


def _global_score_k(measurements: List[Mapping[str, Any]], config: Mapping[str, Any]) -> float:
    values: List[float] = []
    for measurement in measurements:
        values.extend(float(value) for value in measurement.get("effort_densities", []))
    median_density = percentile(values, 0.50)
    if median_density <= 0:
        return 1.0
    return float(config.get("score_density_target", 2.0)) / median_density


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default="benchmark_generation/configs/type5_sources.yaml")
    parser.add_argument("--cache-dir", default="type5_source_cache")
    parser.add_argument("--metadata-dir", default="benchmark_metadata")
    parser.add_argument("--config", default="benchmark_generation/configs/type5.yaml")
    parser.add_argument("--bot-config", default="benchmark_generation/configs/type5_bot_patterns.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--allow-partial", action="store_true")
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
    seed = args.seed if args.seed is not None else int(config["root_seed"])
    build_empirical_profiles(
        args.sources,
        args.cache_dir,
        args.metadata_dir,
        config,
        bot_config,
        seed=seed,
        allow_partial=args.allow_partial,
    )


if __name__ == "__main__":
    main()
