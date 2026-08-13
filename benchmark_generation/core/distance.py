"""Feature transformation, normalization and group-balanced distances."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping

from .features import FEATURE_GROUPS, FEATURE_NAMES


LOG_TRANSFORM_FEATURES = {
    "n_contributors",
    "n_projects",
    "n_skills",
    "total_roles",
    "workload_contention",
    "duration_mean",
    "duration_sd",
    "score_mean",
    "score_sd",
}


def transform_features(features: Mapping[str, float]) -> Dict[str, float]:
    transformed = {}
    for name in FEATURE_NAMES:
        value = float(features.get(name, 0.0))
        if math.isnan(value) or math.isinf(value):
            value = 0.0
        if name in LOG_TRANSFORM_FEATURES:
            value = math.log1p(max(0.0, value))
        transformed[name] = value
    return transformed


@dataclass
class FeatureScaler:
    means: Dict[str, float]
    scales: Dict[str, float]
    group_weights: Dict[str, float]

    @classmethod
    def fit(
        cls,
        feature_rows: Iterable[Mapping[str, float]],
        group_weights: Mapping[str, float] | None = None,
    ) -> "FeatureScaler":
        rows = [transform_features(row) for row in feature_rows]
        if not rows:
            raise ValueError("Cannot fit feature scaler without feature rows")
        means = {}
        scales = {}
        for name in FEATURE_NAMES:
            values = [row[name] for row in rows]
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            scale = math.sqrt(variance)
            means[name] = mean
            scales[name] = scale if scale > 1e-12 else 1.0
        default_group_weight = 1.0 / len(FEATURE_GROUPS)
        weights = {
            group: float((group_weights or {}).get(group, default_group_weight))
            for group in FEATURE_GROUPS
        }
        total = sum(weights.values())
        if total <= 0:
            weights = {group: default_group_weight for group in FEATURE_GROUPS}
        else:
            weights = {group: weight / total for group, weight in weights.items()}
        return cls(means=means, scales=scales, group_weights=weights)

    def normalize(self, features: Mapping[str, float]) -> Dict[str, float]:
        transformed = transform_features(features)
        return {
            name: (transformed[name] - self.means[name]) / self.scales[name]
            for name in FEATURE_NAMES
        }

    def distance(self, a: Mapping[str, float], b: Mapping[str, float]) -> float:
        return weighted_distance(a, b, self.group_weights)


def feature_weights(group_weights: Mapping[str, float] | None = None) -> Dict[str, float]:
    default_group_weight = 1.0 / len(FEATURE_GROUPS)
    group_w = {
        group: float((group_weights or {}).get(group, default_group_weight))
        for group in FEATURE_GROUPS
    }
    total = sum(group_w.values())
    if total <= 0:
        group_w = {group: default_group_weight for group in FEATURE_GROUPS}
    else:
        group_w = {group: weight / total for group, weight in group_w.items()}
    weights = {}
    for group, names in FEATURE_GROUPS.items():
        per_feature = group_w[group] / len(names)
        for name in names:
            weights[name] = per_feature
    return weights


def weighted_distance(
    normalized_a: Mapping[str, float],
    normalized_b: Mapping[str, float],
    group_weights: Mapping[str, float] | None = None,
) -> float:
    weights = feature_weights(group_weights)
    total = 0.0
    for name in FEATURE_NAMES:
        diff = float(normalized_a.get(name, 0.0)) - float(normalized_b.get(name, 0.0))
        if math.isnan(diff) or math.isinf(diff):
            diff = 0.0
        total += weights[name] * diff * diff
    return math.sqrt(total)


def pairwise_distances(
    rows: List[Mapping[str, float]],
    group_weights: Mapping[str, float] | None = None,
) -> List[float]:
    distances = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            distances.append(weighted_distance(rows[i], rows[j], group_weights))
    return distances
