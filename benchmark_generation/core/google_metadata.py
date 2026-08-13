"""Compute and persist immutable Google anchor feature metadata."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from .anchors import load_google_anchors
from .distance import FeatureScaler
from .features import FEATURE_NAMES, extract_features
from .metadata import write_metadata
from .validator import structural_hash


def load_anchor_features(
    google_dir: str | Path,
    group_weights: Dict[str, float] | None = None,
) -> Tuple[list, List[Dict[str, float]], List[Dict[str, float]], FeatureScaler]:
    anchors = load_google_anchors(google_dir)
    raw_features = [extract_features(anchor) for anchor in anchors]
    scaler = FeatureScaler.fit(raw_features, group_weights=group_weights)
    normalized = [scaler.normalize(features) for features in raw_features]
    return anchors, raw_features, normalized, scaler


def write_google_anchor_metadata(
    google_dir: str | Path,
    metadata_dir: str | Path,
    group_weights: Dict[str, float] | None = None,
) -> FeatureScaler:
    anchors, raw_features, normalized, scaler = load_anchor_features(
        google_dir, group_weights=group_weights
    )
    rows = []
    for idx, (anchor, features, norm) in enumerate(zip(anchors, raw_features, normalized), start=1):
        row = {
            "instance_name": anchor.source_name,
            "anchor_id": idx,
            "contributors": len(anchor.contributors),
            "projects": len(anchor.projects),
            "roles": anchor.total_roles,
            "structural_hash": structural_hash(anchor),
        }
        row.update({f"feature.{name}": features[name] for name in FEATURE_NAMES})
        row.update({f"normalized.{name}": norm[name] for name in FEATURE_NAMES})
        rows.append(row)
    metadata_dir = Path(metadata_dir)
    write_metadata(
        rows,
        metadata_dir / "google_anchor_features.csv",
        metadata_dir / "google_anchor_features.jsonl",
    )
    return scaler
