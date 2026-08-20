"""Build Type 6 source feature tables and freeze the 25+25+25+25 selection."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
import zipfile
import re

from benchmark_generation.core.config import load_config
from benchmark_generation.core.metadata import config_hash, summarize_numeric
from benchmark_generation.type6.download_mslib import (
    DEFAULT_CONFIG as DEFAULT_DOWNLOAD_CONFIG,
    canonical_msrcp_entries,
    utc_now,
    verify_or_download_mslib,
)
from benchmark_generation.type6.parse_msrcp import parse_msrcp_text
from benchmark_generation.type6.source_features import extract_source_features


DATASETS = ("MSLIB1", "MSLIB2", "MSLIB3", "MSLIB4")
DEFAULT_CONFIG: Dict[str, Any] = {
    **DEFAULT_DOWNLOAD_CONFIG,
    "root_seed": 2026,
    "selection": {
        "count_per_dataset": 25,
        "seed": 2026,
        "group_seed_limit": 12,
        "feature_columns": [
            "n_activities_total",
            "n_real_activities",
            "n_resources",
            "n_skills",
            "n_skill_levels",
            "n_precedence_arcs",
            "arc_density",
            "order_strength",
            "critical_path_length",
            "network_depth",
            "duration_mean",
            "duration_sd",
            "duration_p10",
            "duration_p90",
            "total_roles",
            "roles_per_activity_mean",
            "roles_per_activity_sd",
            "skills_per_resource_mean",
            "skills_per_resource_sd",
            "skill_level_mean",
            "requirement_level_mean",
            "demand_concentration",
            "supply_concentration",
            "demand_supply_alignment",
            "source_skill_skew",
            "source_contention",
            "qualified_contention",
            "natural_direct_share",
            "natural_one_below_share",
            "natural_mentor_available_share",
            "excel_CNC",
            "excel_OS",
            "excel_SP",
            "excel_SF",
            "excel_SF_alpha",
            "excel_SS",
            "excel_SS_alpha",
            "excel_RA",
            "excel_RA_alpha",
        ],
        "categorical_columns": [
            "n_real_activities",
            "n_resources",
            "n_skills",
            "n_skill_levels",
            "excel_SP",
        ],
    },
}

LOG_TRANSFORM_COLUMNS = {
    "n_activities_total",
    "n_real_activities",
    "n_resources",
    "n_skills",
    "n_precedence_arcs",
    "critical_path_length",
    "network_depth",
    "duration_mean",
    "duration_sd",
    "duration_p10",
    "duration_p90",
    "total_roles",
    "roles_per_activity_mean",
    "roles_per_activity_sd",
    "skills_per_resource_mean",
    "skills_per_resource_sd",
    "source_contention",
}

CATEGORY_SOURCE_NATIVE = "1_source_native_structural_network_resource_skill_indicator"
CATEGORY_DERIVED_SOURCE = "2_deterministic_feature_derived_only_from_source_instance"
CATEGORY_SOLUTION = "3_solution_bks_or_algorithmic_performance_information"

SELECTION_DIMENSION_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "n_activities_total": {
        "definition": "Total project activities including dummy start and dummy end.",
        "provenance": "Project Module header in .msrcp source file.",
        "category": CATEGORY_SOURCE_NATIVE,
        "selection_allowed": True,
        "reason": "Direct source scale field, not a solution result.",
    },
    "n_real_activities": {
        "definition": "Non-dummy activity count, total activities minus dummy start/end.",
        "provenance": "Deterministically derived from Project Module header and dummy-node convention.",
        "category": CATEGORY_DERIVED_SOURCE,
        "selection_allowed": True,
        "reason": "Defines the MSLIB activity/work-package scale without using objective values.",
    },
    "n_resources": {
        "definition": "Number of renewable multi-skilled resources.",
        "provenance": "Project Module header in .msrcp source file.",
        "category": CATEGORY_SOURCE_NATIVE,
        "selection_allowed": True,
        "reason": "Direct workforce scale field.",
    },
    "n_skills": {
        "definition": "Number of skill types.",
        "provenance": "Project Module header in .msrcp source file.",
        "category": CATEGORY_SOURCE_NATIVE,
        "selection_allowed": True,
        "reason": "Direct skill-space scale field.",
    },
    "n_skill_levels": {
        "definition": "Number of hierarchical skill levels.",
        "provenance": "Project Module header in .msrcp source file.",
        "category": CATEGORY_SOURCE_NATIVE,
        "selection_allowed": True,
        "reason": "Direct hierarchical skill structure field.",
    },
    "n_precedence_arcs": {
        "definition": "Count of immediate precedence arcs.",
        "provenance": "Derived by counting successors in Project Module activity rows.",
        "category": CATEGORY_DERIVED_SOURCE,
        "selection_allowed": True,
        "reason": "Deterministic network structure measure.",
    },
    "arc_density": {
        "definition": "Immediate precedence arcs divided by possible acyclic ordered pairs.",
        "provenance": "Derived from parsed precedence arcs and activity count.",
        "category": CATEGORY_DERIVED_SOURCE,
        "selection_allowed": True,
        "reason": "Deterministic network density measure.",
    },
    "order_strength": {
        "definition": "Transitive real-activity precedence comparability share.",
        "provenance": "Derived from transitive closure of parsed precedence DAG.",
        "category": CATEGORY_DERIVED_SOURCE,
        "selection_allowed": True,
        "reason": "Deterministic network complexity measure.",
    },
    "critical_path_length": {
        "definition": "Longest precedence-only path duration from dummy start to dummy end.",
        "provenance": "Independently computed from parsed durations and precedence DAG.",
        "category": CATEGORY_DERIVED_SOURCE,
        "selection_allowed": True,
        "reason": "Allowed CP-like quantity because it is recomputed from source structure, not copied from BKS columns.",
    },
    "network_depth": {
        "definition": "Maximum number of precedence edges on a path to dummy end.",
        "provenance": "Derived from parsed precedence DAG.",
        "category": CATEGORY_DERIVED_SOURCE,
        "selection_allowed": True,
        "reason": "Deterministic topology measure.",
    },
    "duration_mean": {
        "definition": "Mean duration over non-dummy activities.",
        "provenance": "Derived from Project Module activity durations.",
        "category": CATEGORY_DERIVED_SOURCE,
        "selection_allowed": True,
        "reason": "Duration distribution summary, not a solved objective.",
    },
    "duration_sd": {
        "definition": "Population standard deviation of non-dummy activity durations.",
        "provenance": "Derived from Project Module activity durations.",
        "category": CATEGORY_DERIVED_SOURCE,
        "selection_allowed": True,
        "reason": "Duration distribution summary.",
    },
    "duration_p10": {
        "definition": "10th percentile of non-dummy activity durations.",
        "provenance": "Derived from Project Module activity durations.",
        "category": CATEGORY_DERIVED_SOURCE,
        "selection_allowed": True,
        "reason": "Duration distribution summary.",
    },
    "duration_p90": {
        "definition": "90th percentile of non-dummy activity durations.",
        "provenance": "Derived from Project Module activity durations.",
        "category": CATEGORY_DERIVED_SOURCE,
        "selection_allowed": True,
        "reason": "Duration distribution summary.",
    },
    "total_roles": {
        "definition": "Total expanded skill/resource requirements over non-dummy activities.",
        "provenance": "Derived from Skill Requirements Module multiplicities.",
        "category": CATEGORY_DERIVED_SOURCE,
        "selection_allowed": True,
        "reason": "Demand-size measure from source requirements.",
    },
    "roles_per_activity_mean": {
        "definition": "Mean expanded role multiplicity over non-dummy activities.",
        "provenance": "Derived from Skill Requirements Module multiplicities.",
        "category": CATEGORY_DERIVED_SOURCE,
        "selection_allowed": True,
        "reason": "Demand distribution summary.",
    },
    "roles_per_activity_sd": {
        "definition": "Population standard deviation of expanded role multiplicity over non-dummy activities.",
        "provenance": "Derived from Skill Requirements Module multiplicities.",
        "category": CATEGORY_DERIVED_SOURCE,
        "selection_allowed": True,
        "reason": "Demand distribution summary.",
    },
    "skills_per_resource_mean": {
        "definition": "Mean mastered skills per resource.",
        "provenance": "Derived from Workforce Module with Skill Levels.",
        "category": CATEGORY_DERIVED_SOURCE,
        "selection_allowed": True,
        "reason": "Workforce breadth summary.",
    },
    "skills_per_resource_sd": {
        "definition": "Population standard deviation of mastered skills per resource.",
        "provenance": "Derived from Workforce Module with Skill Levels.",
        "category": CATEGORY_DERIVED_SOURCE,
        "selection_allowed": True,
        "reason": "Workforce breadth summary.",
    },
    "skill_level_mean": {
        "definition": "Mean positive resource skill level.",
        "provenance": "Derived from Workforce Module with Skill Levels.",
        "category": CATEGORY_DERIVED_SOURCE,
        "selection_allowed": True,
        "reason": "Hierarchical workforce-depth summary.",
    },
    "requirement_level_mean": {
        "definition": "Mean required level over expanded activity roles.",
        "provenance": "Derived from Skill Level Requirements Module.",
        "category": CATEGORY_DERIVED_SOURCE,
        "selection_allowed": True,
        "reason": "Hierarchical demand-depth summary.",
    },
    "demand_concentration": {
        "definition": "One minus normalized entropy of skill-demand counts.",
        "provenance": "Derived from Skill Requirements Module.",
        "category": CATEGORY_DERIVED_SOURCE,
        "selection_allowed": True,
        "reason": "Demand concentration summary.",
    },
    "supply_concentration": {
        "definition": "One minus normalized entropy of resource skill-supply counts.",
        "provenance": "Derived from Workforce Module with Skill Levels.",
        "category": CATEGORY_DERIVED_SOURCE,
        "selection_allowed": True,
        "reason": "Supply concentration summary.",
    },
    "demand_supply_alignment": {
        "definition": "Cosine similarity between source skill demand and source skill supply.",
        "provenance": "Derived from source demand and supply counts.",
        "category": CATEGORY_DERIVED_SOURCE,
        "selection_allowed": True,
        "reason": "Source structural alignment measure.",
    },
    "source_skill_skew": {
        "definition": "Zipf-like slope proxy estimated from source skill-supply concentration.",
        "provenance": "Derived from source skill supply counts.",
        "category": CATEGORY_DERIVED_SOURCE,
        "selection_allowed": True,
        "reason": "Source workforce heterogeneity measure.",
    },
    "source_contention": {
        "definition": "Total expanded source roles divided by source resources.",
        "provenance": "Derived from source requirements and resource count.",
        "category": CATEGORY_DERIVED_SOURCE,
        "selection_allowed": True,
        "reason": "Source resource-pressure measure.",
    },
    "qualified_contention": {
        "definition": "Mean inverse directly qualified resource count per role, with unqualified roles treated as maximally scarce.",
        "provenance": "Derived from source role levels and workforce levels.",
        "category": CATEGORY_DERIVED_SOURCE,
        "selection_allowed": True,
        "reason": "Skill-level scarcity measure.",
    },
    "natural_direct_share": {
        "definition": "Share of source roles with at least one directly qualified resource.",
        "provenance": "Derived from source role levels and workforce levels.",
        "category": CATEGORY_DERIVED_SOURCE,
        "selection_allowed": True,
        "reason": "Natural source feasibility measure.",
    },
    "natural_one_below_share": {
        "definition": "Share of source roles with at least one resource exactly one level below requirement.",
        "provenance": "Derived from source role levels and workforce levels.",
        "category": CATEGORY_DERIVED_SOURCE,
        "selection_allowed": True,
        "reason": "Natural mentorship-potential measure.",
    },
    "natural_mentor_available_share": {
        "definition": "Share of source roles with both a one-below resource and a directly qualified potential mentor.",
        "provenance": "Derived from source role levels and workforce levels.",
        "category": CATEGORY_DERIVED_SOURCE,
        "selection_allowed": True,
        "reason": "Source hierarchical mentorship-potential measure, not an algorithm result.",
    },
    "excel_CNC": {
        "definition": "Coefficient of Network Complexity.",
        "provenance": "Official workbook Parameter Overview, Network Topology.",
        "category": CATEGORY_SOURCE_NATIVE,
        "selection_allowed": True,
        "reason": "Workbook network indicator, not BKS/performance.",
    },
    "excel_OS": {
        "definition": "Order Strength.",
        "provenance": "Official workbook Parameter Overview, Network Topology.",
        "category": CATEGORY_SOURCE_NATIVE,
        "selection_allowed": True,
        "reason": "Workbook network indicator, not BKS/performance.",
    },
    "excel_SP": {
        "definition": "Serial-Parallel Network Indicator.",
        "provenance": "Official workbook Parameter Overview, Network Topology.",
        "category": CATEGORY_SOURCE_NATIVE,
        "selection_allowed": True,
        "reason": "Workbook network indicator, not BKS/performance.",
    },
    "excel_SF": {
        "definition": "Skill Factor.",
        "provenance": "Official workbook Parameter Overview, Skill Parameters.",
        "category": CATEGORY_SOURCE_NATIVE,
        "selection_allowed": True,
        "reason": "Workbook skill-structure indicator, not BKS/performance.",
    },
    "excel_SF_alpha": {
        "definition": "Skill Factor Variability.",
        "provenance": "Official workbook Parameter Overview, Skill Parameters.",
        "category": CATEGORY_SOURCE_NATIVE,
        "selection_allowed": True,
        "reason": "Workbook skill-structure indicator, not BKS/performance.",
    },
    "excel_SS": {
        "definition": "Skill Strength.",
        "provenance": "Official workbook Parameter Overview, Skill Parameters.",
        "category": CATEGORY_SOURCE_NATIVE,
        "selection_allowed": True,
        "reason": "Workbook skill-supply indicator, not BKS/performance.",
    },
    "excel_SS_alpha": {
        "definition": "Skill Strength Variability.",
        "provenance": "Official workbook Parameter Overview, Skill Parameters.",
        "category": CATEGORY_SOURCE_NATIVE,
        "selection_allowed": True,
        "reason": "Workbook skill-supply indicator, not BKS/performance.",
    },
    "excel_RA": {
        "definition": "Resource availability.",
        "provenance": "Official workbook Parameter Overview, Resource parameters.",
        "category": CATEGORY_SOURCE_NATIVE,
        "selection_allowed": True,
        "reason": "Workbook resource-availability indicator, not BKS/performance.",
    },
    "excel_RA_alpha": {
        "definition": "Resource availability variability.",
        "provenance": "Official workbook Parameter Overview, Resource parameters.",
        "category": CATEGORY_SOURCE_NATIVE,
        "selection_allowed": True,
        "reason": "Workbook resource-availability indicator, not BKS/performance.",
    },
}

SENSITIVE_WORKBOOK_FIELD_AUDIT: Dict[str, Dict[str, Any]] = {
    "LB": {
        "definition": "Best lower bound found by CPLEX within the time limit (linear relaxation).",
        "provenance": "Official workbook Parameter Overview, Best Known Solutions.",
        "category": CATEGORY_SOLUTION,
        "selection_allowed": False,
        "reason": "Solution/BKS information must remain provenance/diagnostic only.",
    },
    "UB": {
        "definition": "Best known solution.",
        "provenance": "Official workbook Parameter Overview, Best Known Solutions and algorithm result sections.",
        "category": CATEGORY_SOLUTION,
        "selection_allowed": False,
        "reason": "Best-known objective values cannot influence source selection.",
    },
    "Open?": {
        "definition": "Closed if best found lower bound equals best found upper bound across procedures; otherwise open.",
        "provenance": "Official workbook Parameter Overview, Best Known Solutions.",
        "category": CATEGORY_SOLUTION,
        "selection_allowed": False,
        "reason": "Solved/open status depends on lower and upper bounds from solution procedures.",
    },
    "CP": {
        "definition": "Critical path based lower bound.",
        "provenance": "Official workbook Parameter Overview, Fast lower bounds.",
        "category": CATEGORY_SOLUTION,
        "selection_allowed": False,
        "reason": "Workbook CP is not selected; Type 6 uses independently recomputed critical_path_length from the parsed DAG instead.",
    },
}


def build_and_select_type6_sources(
    source_dir: str | Path,
    metadata_dir: str | Path,
    config: Dict[str, Any],
    seed: int,
) -> Dict[str, Any]:
    source_dir = Path(source_dir)
    metadata_dir = Path(metadata_dir)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    provenance = verify_or_download_mslib(
        source_dir,
        metadata_dir,
        config,
        download=False,
        extract=True,
    )
    release_root = find_release_root(source_dir)
    workbook_path = release_root / "MSLIB (Parameters and BKS).xlsx"
    excel_indicators, workbook_info = load_workbook_indicators(workbook_path)
    rows = build_source_feature_rows(release_root, config, excel_indicators)
    validate_candidate_table(rows, config)
    write_csv(rows, metadata_dir / "type6_source_candidates.csv")

    previous_selection_path = metadata_dir / "type6_source_selection.csv"
    previous_selection_ids = read_selection_ids(previous_selection_path)
    validate_selection_feature_policy(config)
    selected = freeze_source_selection(rows, config, seed=seed)
    audit = build_selection_audit(config, previous_selection_ids, selected)
    if not audit["policy_passed"]:
        raise RuntimeError(
            "Type 6 source selection audit failed: "
            + json.dumps(audit["disallowed_configured_dimensions"], sort_keys=True)
        )
    if audit["selection_changed_after_audit"] and previous_selection_ids:
        (metadata_dir / "type6_source_selection_pre_audit.json").write_text(
            json.dumps(
                {
                    "preserved_at": utc_now(),
                    "reason": "Corrected selection changed after removing disallowed BKS/performance dimensions.",
                    "source_ids": previous_selection_ids,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    write_selection_audit(audit, metadata_dir)
    write_csv(selected, metadata_dir / "type6_source_selection.csv")
    report = source_selection_report(rows, selected, config, provenance, workbook_info, seed, audit)
    (metadata_dir / "type6_source_selection.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    write_sources_markdown(selected, report, metadata_dir / "TYPE6_SOURCES.md")
    write_methodology_draft(report, metadata_dir / "type6_methodology_draft.md")
    write_phase_provenance(provenance, report, metadata_dir / "type6_data_provenance.json")
    return report


def find_release_root(source_dir: str | Path) -> Path:
    source_dir = Path(source_dir)
    candidates = [
        source_dir / "MSLIB_v2_2" / "MSLIB_v2_2",
        source_dir / "MSLIB_v2_2",
    ]
    for candidate in candidates:
        if (candidate / "MSLIB1.zip").exists() and (candidate / "MSRCPSP_DataFormat.pdf").exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate extracted MSLIB_v2_2 release root under type6_source_data"
    )


def build_source_feature_rows(
    release_root: Path,
    config: Mapping[str, Any],
    excel_indicators: Mapping[Tuple[str, str], Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    expected_counts = dict(config.get("expected_counts", {}))
    rows: List[Dict[str, Any]] = []
    for dataset in DATASETS:
        zip_path = release_root / f"{dataset}.zip"
        expected = int(expected_counts[dataset])
        logging.info("Parsing %s canonical source instances from %s", dataset, zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            canonical, noncanonical = canonical_msrcp_entries(zf, dataset)
            if len(canonical) != expected:
                raise RuntimeError(
                    f"{dataset} canonical count mismatch before parsing: "
                    f"observed {len(canonical)}, expected {expected}"
                )
            if noncanonical:
                logging.warning("%s noncanonical .msrcp files excluded: %s", dataset, noncanonical)
            for idx, entry in enumerate(canonical, start=1):
                if idx == 1 or idx % 500 == 0 or idx == expected:
                    logging.info("Parsing %s %d/%d", dataset, idx, expected)
                filename = str(entry["filename"])
                indicators = excel_indicators.get((dataset, filename))
                if indicators is None:
                    raise RuntimeError(f"Workbook is missing source indicators for {dataset}/{filename}")
                payload = zf.read(str(entry["zip_entry"]))
                instance = parse_msrcp_text(
                    payload.decode("utf-8", errors="strict"),
                    dataset=dataset,
                    source_filename=filename,
                    source_zip_entry=str(entry["zip_entry"]),
                )
                row = extract_source_features(instance, indicators)
                row["canonical_source"] = True
                rows.append(row)
    return rows


def load_workbook_indicators(path: str | Path) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], Dict[str, Any]]:
    path = Path(path)
    try:
        from openpyxl import load_workbook
    except ImportError as exc:
        raise RuntimeError(
            "openpyxl is required to read MSLIB (Parameters and BKS).xlsx for Type 6"
        ) from exc

    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb["MSRCPSP (basic)"]
    headers = [str(value) if value is not None else "" for value in next(ws.iter_rows(min_row=9, max_row=9, values_only=True))]
    subheaders = [str(value) if value is not None else "" for value in next(ws.iter_rows(min_row=10, max_row=10, values_only=True))]
    fieldnames = effective_workbook_fieldnames(subheaders)
    indicators: Dict[Tuple[str, str], Dict[str, Any]] = {}
    canonical_name = re.compile(r"^MSLIB_Set([1-4])_(\d+)\.msrcp$", re.IGNORECASE)
    for row in ws.iter_rows(min_row=11, values_only=True):
        dataset = str(row[0]) if row and row[0] is not None else ""
        name = str(row[1]) if len(row) > 1 and row[1] is not None else ""
        match = canonical_name.match(name)
        if dataset not in DATASETS or not match:
            continue
        values = {
            field: _coerce_workbook_value(row[idx] if idx < len(row) else None)
            for idx, field in enumerate(fieldnames)
            if field
        }
        indicators[(dataset, name)] = values
    info = {
        "path": str(path),
        "sheets": wb.sheetnames,
        "basic_sheet_category_row_9": headers,
        "basic_sheet_column_header_row_10": subheaders,
        "canonical_indicator_rows_per_set": dict(Counter(dataset for dataset, _name in indicators)),
        "available_source_indicator_names": [
            "# Act",
            "CNC",
            "OS",
            "SP",
            "# Skills",
            "SF",
            "SFalpha",
            "SS",
            "SSalpha",
            "# R",
            "RA",
            "RAalpha",
            "LB",
            "UB",
            "Open?",
            "CP",
        ],
    }
    return indicators, info


def effective_workbook_fieldnames(row_11: Sequence[str]) -> List[str]:
    fields = list(row_11)
    if len(fields) > 8:
        fields[8] = "SFalpha"
    if len(fields) > 10:
        fields[10] = "SSalpha"
    if len(fields) > 13:
        fields[13] = "RAalpha"
    return fields


def freeze_source_selection(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    seed: int,
) -> List[Dict[str, Any]]:
    selection_config = config.get("selection", {})
    count_per_dataset = int(selection_config.get("count_per_dataset", 25))
    selected: List[Dict[str, Any]] = []
    source_index = 1
    for dataset in DATASETS:
        dataset_rows = [dict(row) for row in rows if row.get("MSLIB_set") == dataset]
        dataset_selected = stratified_maximin_selection(
            dataset_rows,
            count=count_per_dataset,
            feature_columns=list(selection_config.get("feature_columns", [])),
            categorical_columns=list(selection_config.get("categorical_columns", [])),
            group_seed_limit=int(selection_config.get("group_seed_limit", 12)),
        )
        for rank, row in enumerate(dataset_selected, start=1):
            row["type6_source_index"] = source_index
            row["selection_rank_within_set"] = rank
            row["selection_seed"] = seed
            selected.append(row)
            source_index += 1
    validate_selection(selected, count_per_dataset)
    return selected


def stratified_maximin_selection(
    rows: Sequence[Mapping[str, Any]],
    count: int,
    feature_columns: Sequence[str],
    categorical_columns: Sequence[str],
    group_seed_limit: int,
) -> List[Dict[str, Any]]:
    if len(rows) < count:
        raise RuntimeError(f"Cannot select {count} rows from only {len(rows)} candidates")
    vectors = standardized_vectors(rows, feature_columns)
    row_by_key = {row_key(row): idx for idx, row in enumerate(rows)}
    selected_indices: List[int] = []
    reasons: Dict[int, str] = {}

    groups: Dict[Tuple[str, ...], List[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        groups[group_key(row, categorical_columns)].append(idx)
    ranked_groups = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    for group_idx, (key, indices) in enumerate(ranked_groups[:group_seed_limit], start=1):
        if len(selected_indices) >= count:
            break
        candidate_idx = medoid_index(indices, vectors)
        if candidate_idx not in selected_indices:
            selected_indices.append(candidate_idx)
            reasons[candidate_idx] = f"categorical_group_medoid_{group_idx}:{'|'.join(key)}"

    if not selected_indices:
        selected_indices.append(global_medoid_index(list(range(len(rows))), vectors))
        reasons[selected_indices[0]] = "global_medoid"

    while len(selected_indices) < count:
        remaining = [idx for idx in range(len(rows)) if idx not in selected_indices]
        candidate_idx = max(
            remaining,
            key=lambda idx: (
                min(euclidean(vectors[idx], vectors[selected]) for selected in selected_indices),
                -int(rows[idx].get("source_instance_id", 0)),
                str(rows[idx].get("source_filename", "")),
            ),
        )
        selected_indices.append(candidate_idx)
        nearest = min(euclidean(vectors[candidate_idx], vectors[selected]) for selected in selected_indices[:-1])
        reasons[candidate_idx] = f"maximin_fill_nearest_distance={nearest:.6f}"

    selected_rows = []
    for idx in selected_indices:
        row = dict(rows[idx])
        row["selection_reason"] = reasons[idx]
        selected_rows.append(row)
    return selected_rows


def standardized_vectors(
    rows: Sequence[Mapping[str, Any]],
    feature_columns: Sequence[str],
) -> List[List[float]]:
    raw_columns: Dict[str, List[float]] = {}
    for column in feature_columns:
        values = [_numeric(row.get(column)) for row in rows]
        nonmissing = [value for value in values if value is not None]
        fill = mean(nonmissing) if nonmissing else 0.0
        transformed = [
            transform_value(column, fill if value is None else value)
            for value in values
        ]
        raw_columns[column] = transformed
    means = {column: mean(values) if values else 0.0 for column, values in raw_columns.items()}
    sds = {
        column: pstdev(values) if len(values) > 1 and pstdev(values) > 1e-12 else 1.0
        for column, values in raw_columns.items()
    }
    vectors: List[List[float]] = []
    for idx in range(len(rows)):
        vectors.append(
            [
                (raw_columns[column][idx] - means[column]) / sds[column]
                for column in feature_columns
            ]
        )
    return vectors


def transform_value(column: str, value: float) -> float:
    if column in LOG_TRANSFORM_COLUMNS:
        return math.log1p(max(0.0, value))
    return float(value)


def medoid_index(indices: Sequence[int], vectors: Sequence[Sequence[float]]) -> int:
    if len(indices) == 1:
        return int(indices[0])
    centroid = [
        mean([vectors[idx][dim] for idx in indices])
        for dim in range(len(vectors[indices[0]]))
    ]
    return min(indices, key=lambda idx: (euclidean(vectors[idx], centroid), idx))


def global_medoid_index(indices: Sequence[int], vectors: Sequence[Sequence[float]]) -> int:
    return medoid_index(indices, vectors)


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def group_key(row: Mapping[str, Any], columns: Sequence[str]) -> Tuple[str, ...]:
    return tuple(str(row.get(column, "")) for column in columns)


def row_key(row: Mapping[str, Any]) -> Tuple[str, int]:
    return str(row.get("MSLIB_set", "")), int(row.get("source_instance_id", 0))


def source_selection_report(
    candidates: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    provenance: Mapping[str, Any],
    workbook_info: Mapping[str, Any],
    seed: int,
    selection_audit: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    count_by_set = Counter(str(row.get("MSLIB_set")) for row in candidates)
    selected_by_set = Counter(str(row.get("MSLIB_set")) for row in selected)
    feature_columns = list(config.get("selection", {}).get("feature_columns", []))
    selected_summary = {
        dataset: {
            column: summarize_numeric([
                float(row[column])
                for row in selected
                if row.get("MSLIB_set") == dataset and _numeric(row.get(column)) is not None
            ])
            for column in [
                "n_real_activities",
                "n_resources",
                "n_skills",
                "total_roles",
                "critical_path_length",
                "roles_per_activity_mean",
                "skills_per_resource_mean",
                "natural_direct_share",
                "natural_mentor_available_share",
            ]
        }
        for dataset in DATASETS
    }
    gate = {
        "candidate_counts_match_expected": all(
            count_by_set[dataset] == int(config["expected_counts"][dataset])
            for dataset in DATASETS
        ),
        "selected_count": len(selected),
        "selected_counts_per_set": dict(selected_by_set),
        "exactly_25_per_mslib1_4": all(selected_by_set[dataset] == 25 for dataset in DATASETS),
        "no_mslib5_selected": "MSLIB5" not in selected_by_set,
        "unique_source_selection": len({row_key(row) for row in selected}) == len(selected),
        "selection_independent_of_mtp_isa_features": all(
            not str(column).startswith("feature.") for column in feature_columns
        ),
        "all_source_provenance_complete": all(source_provenance_complete(row) for row in selected),
    }
    gate["passed"] = all(bool(value) for value in gate.values() if not isinstance(value, dict))
    return {
        "family": "type6",
        "phase": "source_selection_phases_1_3",
        "built_at": utc_now(),
        "root_seed": seed,
        "config_hash": config_hash(config),
        "source_release": provenance.get("release", {}),
        "artifact_statement": provenance.get("type6_source_artifact"),
        "archive_report": provenance.get("archive_report", {}),
        "v2_1_tag_archive_evidence": provenance.get("v2_1_tag_archive_evidence", {}),
        "workbook_info": workbook_info,
        "source_format_interpretation": source_format_interpretation(),
        "hierarchical_skill_representation": hierarchical_skill_representation(),
        "candidate_counts_per_set": dict(count_by_set),
        "selection_algorithm": selection_algorithm_description(config),
        "selection_audit": dict(selection_audit or {}),
        "selection_gate": gate,
        "selected_summary_statistics": selected_summary,
        "selected_records": sorted(
            [dict(row) for row in selected],
            key=lambda row: int(row["type6_source_index"]),
        ),
    }


def source_format_interpretation() -> Dict[str, str]:
    return {
        "format_document": "Official MSRCPSP_DataFormat.pdf included in MSLIB_v2_2.zip.",
        "project_module": (
            "Header is number of activities including dummy start/end, renewable multi-skilled "
            "resources, skill types, and skill levels; the next two singleton lines are BKS/basic "
            "and deadline/provenance values; then one activity row per node contains duration, "
            "successor count, and successor ids."
        ),
        "workforce_module": "Binary resource-skill mastery matrix with one row per resource and one column per skill.",
        "workforce_level_module": "Resource-skill level matrix with 0 for unmastered skills and positive levels 1..L for mastered skills.",
        "skill_requirements_module": "Activity-by-skill integer demand multiplicity matrix.",
        "skill_level_requirements_module": (
            "One row per activity; dummy rows are -1. For non-dummy activities, level entries "
            "are expanded in skill order according to the multiplicities in the Skill Requirements Module."
        ),
        "optional_modules": "Cost, common resource usage, and rework modules are parsed as available source metadata only for later phases.",
    }


def hierarchical_skill_representation() -> Dict[str, str]:
    return {
        "available": "yes",
        "resource_levels": "Workforce Module with Skill Levels gives each resource's level for each skill.",
        "requirement_levels": "Skill Level Requirements Module gives one required level per required resource/skill role.",
        "eligibility_interpretation": "The Type 6 mapping uses lower-bound hierarchical eligibility: resource/contributor level >= required role level.",
    }


def selection_algorithm_description(config: Mapping[str, Any]) -> Dict[str, Any]:
    selection = config.get("selection", {})
    return {
        "method": "deterministic source-space stratified maximin",
        "steps": [
            "parse every canonical MSLIB1-4 .msrcp source using only source-native data",
            "join official workbook network/resource/skill indicators where canonical rows exist",
            "standardize configured source indicators within each MSLIB set, with log1p on scale variables",
            "seed selection with medoids of the most frequent categorical source-parameter groups",
            "fill remaining slots by farthest-first maximin distance in standardized source space",
            "freeze dataset-major contiguous type6_source_index values",
        ],
        "selection_seed": selection.get("seed", config.get("root_seed", 2026)),
        "feature_columns": list(selection.get("feature_columns", [])),
        "categorical_columns": list(selection.get("categorical_columns", [])),
    }


def validate_selection_feature_policy(config: Mapping[str, Any]) -> None:
    feature_columns = list(config.get("selection", {}).get("feature_columns", []))
    disallowed = [
        column
        for column in feature_columns
        if not SELECTION_DIMENSION_DEFINITIONS.get(column, {}).get("selection_allowed", False)
    ]
    if disallowed:
        raise RuntimeError(
            "Type 6 source selection contains disallowed or undocumented dimensions: "
            + ", ".join(disallowed)
        )


def build_selection_audit(
    config: Mapping[str, Any],
    previous_selection_ids: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    feature_columns = list(config.get("selection", {}).get("feature_columns", []))
    configured = set(feature_columns)
    rows = []
    for indicator in feature_columns:
        entry = dict(SELECTION_DIMENSION_DEFINITIONS[indicator])
        entry["indicator"] = indicator
        entry["configured_selection_dimension"] = True
        rows.append(entry)
    for indicator, entry in SENSITIVE_WORKBOOK_FIELD_AUDIT.items():
        audit_row = dict(entry)
        audit_row["indicator"] = indicator
        audit_row["configured_selection_dimension"] = indicator in configured
        rows.append(audit_row)
    disallowed = [
        row["indicator"]
        for row in rows
        if row.get("configured_selection_dimension") and not row.get("selection_allowed")
    ]
    corrected_selection_ids = selection_ids(selected)
    return {
        "built_at": utc_now(),
        "policy_passed": not disallowed,
        "selection_dimension_count": len(feature_columns),
        "configured_selection_dimensions": feature_columns,
        "disallowed_configured_dimensions": disallowed,
        "sensitive_solution_fields_not_used": {
            key: key not in configured
            for key in ("LB", "UB", "Open?", "CP")
        },
        "critical_path_selection_source": (
            "critical_path_length is independently computed from the parsed precedence DAG; "
            "the workbook CP lower-bound column is not used."
        ),
        "selection_changed_after_audit": bool(previous_selection_ids)
        and list(previous_selection_ids) != corrected_selection_ids,
        "previous_selection_ids": list(previous_selection_ids),
        "corrected_selection_ids": corrected_selection_ids,
        "audit_rows": rows,
    }


def write_selection_audit(audit: Mapping[str, Any], metadata_dir: str | Path) -> None:
    metadata_dir = Path(metadata_dir)
    rows = list(audit.get("audit_rows", []))
    write_csv(rows, metadata_dir / "type6_selection_audit.csv")
    (metadata_dir / "type6_selection_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def selection_ids(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "type6_source_index": int(row.get("type6_source_index", idx)),
            "MSLIB_set": str(row.get("MSLIB_set")),
            "source_instance_id": int(row.get("source_instance_id", 0)),
            "source_filename": str(row.get("source_filename")),
        }
        for idx, row in enumerate(rows, start=1)
    ]


def read_selection_ids(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return selection_ids(list(csv.DictReader(fh)))


def write_sources_markdown(
    selected: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
    path: str | Path,
) -> None:
    release = report.get("source_release", {})
    archive = report.get("archive_report", {})
    lines = [
        "# Type 6 Source Provenance",
        "",
        str(report["artifact_statement"]),
        "",
        "## Upstream Versioning Notes",
        "",
        f"- GitHub release/tag observed: {release.get('observed_release_tag', '')}",
        f"- Release asset: {release.get('release_asset', '')}",
        f"- SHA-256: {release.get('sha256', '')}",
        f"- Release body states: {release.get('release_body_version_statement', '')}",
        f"- Repository README states: {release.get('repository_readme_version_statement', '')}",
        "- Repository documentation states that previous releases are removed when a new release is published and recommends using the latest release.",
        f"- Internal archive Readme.rtf observed: {archive.get('internal_readme_version_statement', '')}",
        f"- Dataset information files found: {json.dumps(archive.get('dataset_information_files', {}), sort_keys=True)}",
        "",
        "Type 6 selected sources are frozen before any MTP generation. MSLIB5 is excluded.",
        "",
        "## Selected Sources",
        "",
    ]
    for row in sorted(selected, key=lambda item: int(item["type6_source_index"])):
        lines.extend(
            [
                f"### type6_source_{int(row['type6_source_index']):03d}",
                "",
                f"- MSLIB set: {row['MSLIB_set']}",
                f"- Source file: {row['source_filename']}",
                f"- Source instance id: {row['source_instance_id']}",
                f"- Source SHA-256: {row['source_sha256']}",
                f"- Activities/resources/skills: {row['n_real_activities']} / {row['n_resources']} / {row['n_skills']}",
                f"- Selection reason: {row['selection_reason']}",
                "",
            ]
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_methodology_draft(report: Mapping[str, Any], path: str | Path) -> None:
    release = report.get("source_release", {})
    archive = report.get("archive_report", {})
    lines = [
        "# Type 6 Methodology Draft",
        "",
        "Type 6 is a literature-grounded synthetic MTP family derived from published MSRCPSP benchmark configurations.",
        "",
        "## Source Artifact",
        "",
        str(report["artifact_statement"]),
        "",
        "Recorded upstream inconsistency:",
        "",
        f"- GitHub release/tag observed: {release.get('observed_release_tag', '')}",
        f"- Release asset: {release.get('release_asset', '')}",
        f"- SHA-256: {release.get('sha256', '')}",
        f"- Release body states: {release.get('release_body_version_statement', '')}",
        f"- Repository README states: {release.get('repository_readme_version_statement', '')}",
        "- Repository documentation states that previous releases are removed when a new release is published and recommends using the latest release.",
        f"- Internal archive Readme.rtf observed: {archive.get('internal_readme_version_statement', '')}",
        f"- Dataset information files found: {json.dumps(archive.get('dataset_information_files', {}), sort_keys=True)}",
        "",
        "The tiny v2.1 GitHub tag archive is retained only as provenance evidence and is not used as source data.",
        "",
        "## Observed / Source-Native MSLIB Information",
        "",
        "- workforce size, skill count, and skill-level matrices",
        "- real activity count, durations, skill-demand multiplicities, and required levels",
        "- precedence arcs and dummy start/end activities",
        "- workbook network/resource/skill indicators including CNC, OS, SP, SF, SS, RA and alpha variants",
        "- BKS and deadline fields as provenance/diagnostics only",
        "",
        "## Derived From MSLIB",
        "",
        "- critical path, slack, topological position, network depth, order strength, and arc density",
        "- demand/supply alignment, concentration, contention, and source skill skew",
        "- natural direct eligibility and one-level-below mentorship potential under hierarchical levels",
        "",
        "## Synthetic MTP Modeling Decisions",
        "",
        "No Type 6 MTP instances are generated in Phases 1-3. Later phases will use these frozen source records to parameterize the shared forward MTP generator.",
        "MSRCPSP precedence constraints will not be written as MTP constraints, and MTP scores will be literature-structure-conditioned synthetic proxies.",
        "",
        "## Source Selection",
        "",
        json.dumps(report["selection_algorithm"], indent=2, sort_keys=True, default=str),
        "",
    ]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_phase_provenance(
    provenance: Mapping[str, Any],
    report: Mapping[str, Any],
    path: str | Path,
) -> None:
    merged = dict(provenance)
    merged["phase_1_3_report"] = {
        "candidate_counts_per_set": report["candidate_counts_per_set"],
        "selection_gate": report["selection_gate"],
        "workbook_info": report["workbook_info"],
        "source_format_interpretation": report["source_format_interpretation"],
        "hierarchical_skill_representation": report["hierarchical_skill_representation"],
    }
    Path(path).write_text(
        json.dumps(merged, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def validate_candidate_table(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> None:
    counts = Counter(str(row.get("MSLIB_set")) for row in rows)
    expected = dict(config.get("expected_counts", {}))
    for dataset in DATASETS:
        if counts[dataset] != int(expected[dataset]):
            raise RuntimeError(
                f"Parsed candidate count mismatch for {dataset}: "
                f"{counts[dataset]} observed, {expected[dataset]} expected"
            )
    missing_hierarchy = [
        row.get("source_filename")
        for row in rows
        if not row.get("hierarchical_skill_levels_available")
    ]
    if missing_hierarchy:
        raise RuntimeError(f"Hierarchical source levels missing for rows: {missing_hierarchy[:5]}")


def validate_selection(selected: Sequence[Mapping[str, Any]], count_per_dataset: int) -> None:
    counts = Counter(str(row.get("MSLIB_set")) for row in selected)
    if any(counts[dataset] != count_per_dataset for dataset in DATASETS):
        raise RuntimeError(f"Type 6 source selection count mismatch: {dict(counts)}")
    if "MSLIB5" in counts:
        raise RuntimeError("MSLIB5 must not be included in Type 6 final source selection")
    keys = [row_key(row) for row in selected]
    if len(set(keys)) != len(keys):
        raise RuntimeError("Duplicate source selected for Type 6")
    if not all(source_provenance_complete(row) for row in selected):
        raise RuntimeError("Selected Type 6 source provenance is incomplete")


def source_provenance_complete(row: Mapping[str, Any]) -> bool:
    required = [
        "type6_source_index",
        "MSLIB_set",
        "source_instance_id",
        "source_filename",
        "source_zip_entry",
        "source_sha256",
        "n_real_activities",
        "n_resources",
        "n_skills",
        "selection_reason",
        "selection_seed",
    ]
    return all(row.get(key) not in ("", None) for key in required)


def write_csv(rows: Sequence[Mapping[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _coerce_workbook_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _numeric(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default="type6_source_data")
    parser.add_argument("--metadata-dir", default="benchmark_metadata")
    parser.add_argument("--config", default="benchmark_generation/configs/type6.yaml")
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
    seed = args.seed if args.seed is not None else int(config.get("root_seed", 2026))
    report = build_and_select_type6_sources(
        args.source_dir,
        args.metadata_dir,
        config,
        seed=seed,
    )
    print(json.dumps(report["selection_gate"], indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
