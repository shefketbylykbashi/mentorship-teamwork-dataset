"""Deterministic balanced sharding for MTP solver campaign run plans."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .common import read_json, sha256_text
from .state import RunSpec


SHARD_SCHEMA_VERSION = "mtp_shard_manifest_v1"
SHARDING_ALGORITHM = "canonical_sorted_round_robin_modulo_v1"
SEED_DERIVATION_VERSION = "stable_seed_sha256_root_solver_instance_execution_v1"
CANONICAL_ORDER = ("family", "instance_id", "solver_id", "execution_index", "run_id")


class ShardingError(ValueError):
    """Raised when a shard request or shard manifest is invalid."""


@dataclass(frozen=True)
class ShardSelection:
    mode: str
    shard_index: int
    num_shards: int
    global_run_count: int
    shard_run_count: int
    minimum_shard_size: int
    maximum_shard_size: int
    run_ids: tuple[str, ...]
    manifest_path: str | None = None
    manifest_sha256: str | None = None

    def to_campaign_payload(self) -> dict:
        payload = {
            "mode": self.mode,
            "schema_version": SHARD_SCHEMA_VERSION,
            "sharding_algorithm": SHARDING_ALGORITHM,
            "canonical_order": list(CANONICAL_ORDER),
            "shard_index": self.shard_index,
            "num_shards": self.num_shards,
            "global_run_count": self.global_run_count,
            "shard_run_count": self.shard_run_count,
            "minimum_shard_size": self.minimum_shard_size,
            "maximum_shard_size": self.maximum_shard_size,
            "run_ids": list(self.run_ids),
        }
        if self.manifest_path is not None:
            payload["manifest_path"] = self.manifest_path
        if self.manifest_sha256 is not None:
            payload["manifest_sha256"] = self.manifest_sha256
        return payload


def validate_cli_shard_args(
    num_shards: int | None,
    shard_index: int | None,
    shard_manifest: str | Path | None,
) -> None:
    """Validate mutually exclusive CLI sharding modes."""

    if shard_manifest is not None:
        if num_shards is not None or shard_index is not None:
            raise ShardingError("--shard-manifest cannot be combined with --num-shards/--shard-index")
        return
    if num_shards is None and shard_index is None:
        return
    if num_shards is None:
        raise ShardingError("--num-shards is required when --shard-index is supplied")
    if shard_index is None:
        raise ShardingError("--shard-index is required when --num-shards is supplied")
    validate_shard_bounds(num_shards, shard_index)


def validate_shard_bounds(num_shards: int, shard_index: int) -> None:
    if num_shards < 1:
        raise ShardingError("--num-shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ShardingError("--shard-index must satisfy 0 <= shard_index < num_shards")


def canonical_run_sort_key(spec: RunSpec) -> tuple[str, str, str, int, str]:
    return (spec.family, spec.instance_id, spec.solver_id, int(spec.execution_index), spec.run_id)


def canonical_run_plan(run_plan: Iterable[RunSpec]) -> list[RunSpec]:
    """Return the platform-independent global order used for sharding.

    The key is ``(family, instance_id, solver_id, execution_index, run_id)``.
    It does not depend on filesystem iteration order, Python's randomized hash
    seed, worker scheduling, or process launch order.
    """

    return sorted(run_plan, key=canonical_run_sort_key)


def shard_size_distribution(total_runs: int, num_shards: int) -> list[int]:
    validate_shard_bounds(num_shards, 0)
    base = total_runs // num_shards
    remainder = total_runs % num_shards
    return [base + (1 if index < remainder else 0) for index in range(num_shards)]


def select_shard(run_plan: Sequence[RunSpec], num_shards: int, shard_index: int) -> tuple[list[RunSpec], ShardSelection]:
    validate_shard_bounds(num_shards, shard_index)
    ordered = canonical_run_plan(run_plan)
    selected = [spec for index, spec in enumerate(ordered) if index % num_shards == shard_index]
    distribution = shard_size_distribution(len(ordered), num_shards)
    selection = ShardSelection(
        mode="index",
        shard_index=shard_index,
        num_shards=num_shards,
        global_run_count=len(ordered),
        shard_run_count=len(selected),
        minimum_shard_size=min(distribution) if distribution else 0,
        maximum_shard_size=max(distribution) if distribution else 0,
        run_ids=tuple(spec.run_id for spec in selected),
    )
    return selected, selection


def select_manifest_runs(
    run_plan: Sequence[RunSpec],
    manifest_path: str | Path,
    *,
    root_seed: int,
    runs_per_solver_instance: int,
    benchmark_manifest_hash: str | None = None,
) -> tuple[list[RunSpec], ShardSelection, dict]:
    manifest_path = Path(manifest_path)
    manifest = read_json(manifest_path)
    validate_shard_manifest(manifest, manifest_path)

    run_ids = manifest.get("run_ids") or []
    duplicates = _duplicate_items(run_ids)
    if duplicates:
        raise ShardingError(f"Shard manifest contains duplicate run IDs: {duplicates[:3]}")

    run_by_id = {spec.run_id: spec for spec in run_plan}
    missing = [run_id for run_id in run_ids if run_id not in run_by_id]
    if missing:
        raise ShardingError(f"Shard manifest references unknown run IDs: {missing[:3]}")

    if int(manifest.get("global_run_count", -1)) != len(run_plan):
        raise ShardingError("Shard manifest global_run_count differs from current campaign")
    if int(manifest.get("root_seed", -1)) != int(root_seed):
        raise ShardingError("Shard manifest root_seed differs from current campaign")
    if int(manifest.get("runs_per_solver_instance", -1)) != int(runs_per_solver_instance):
        raise ShardingError("Shard manifest runs_per_solver_instance differs from current campaign")
    if benchmark_manifest_hash and manifest.get("benchmark_manifest_hash") != benchmark_manifest_hash:
        raise ShardingError("Shard manifest benchmark_manifest_hash differs from current campaign")

    selected = [run_by_id[run_id] for run_id in run_ids]
    num_shards = int(manifest["num_shards"])
    shard_index = int(manifest["shard_index"])
    distribution = shard_size_distribution(len(run_plan), num_shards)
    selection = ShardSelection(
        mode="manifest",
        shard_index=shard_index,
        num_shards=num_shards,
        global_run_count=len(run_plan),
        shard_run_count=len(selected),
        minimum_shard_size=min(distribution) if distribution else 0,
        maximum_shard_size=max(distribution) if distribution else 0,
        run_ids=tuple(run_ids),
        manifest_path=str(manifest_path),
        manifest_sha256=manifest.get("manifest_sha256"),
    )
    return selected, selection, manifest


def build_shard_manifest(
    *,
    run_plan: Sequence[RunSpec],
    selected: Sequence[RunSpec],
    shard_index: int,
    num_shards: int,
    root_seed: int,
    runs_per_solver_instance: int,
    benchmark_manifest_hash: str,
    solver_manifest_hash: str | None,
    campaign_config_hash: str,
) -> dict:
    distribution = shard_size_distribution(len(run_plan), num_shards)
    payload = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "sharding_algorithm": SHARDING_ALGORITHM,
        "canonical_order": list(CANONICAL_ORDER),
        "shard_index": shard_index,
        "num_shards": num_shards,
        "global_run_count": len(run_plan),
        "shard_run_count": len(selected),
        "minimum_shard_size": min(distribution) if distribution else 0,
        "maximum_shard_size": max(distribution) if distribution else 0,
        "root_seed": int(root_seed),
        "seed_derivation_version": SEED_DERIVATION_VERSION,
        "runs_per_solver_instance": int(runs_per_solver_instance),
        "benchmark_manifest_hash": benchmark_manifest_hash,
        "solver_manifest_hash": solver_manifest_hash,
        "campaign_config_hash": campaign_config_hash,
        "run_ids": [spec.run_id for spec in selected],
        "runs": [compact_run_reference(spec) for spec in selected],
    }
    payload["manifest_sha256"] = manifest_sha256(payload)
    return payload


def compact_run_reference(spec: RunSpec) -> dict:
    return {
        "run_id": spec.run_id,
        "solver_id": spec.solver_id,
        "instance_id": spec.instance_id,
        "family": spec.family,
        "execution_index": spec.execution_index,
        "seed": spec.seed,
    }


def build_top_level_manifest(shard_manifests: Sequence[dict], full_run_plan: Sequence[RunSpec]) -> dict:
    if not shard_manifests:
        raise ShardingError("No shard manifests were provided")
    num_shards = int(shard_manifests[0]["num_shards"])
    global_run_count = len(full_run_plan)
    expected_ids = {spec.run_id for spec in full_run_plan}
    verification = verify_run_id_partition((m.get("run_ids") or [] for m in shard_manifests), expected_ids)
    sizes = [int(m["shard_run_count"]) for m in shard_manifests]
    payload = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "sharding_algorithm": SHARDING_ALGORITHM,
        "canonical_order": list(CANONICAL_ORDER),
        "num_shards": num_shards,
        "global_run_count": global_run_count,
        "minimum_shard_size": min(sizes),
        "maximum_shard_size": max(sizes),
        "shards": [
            {
                "index": int(m["shard_index"]),
                "run_count": int(m["shard_run_count"]),
                "manifest_filename": f"shard_{int(m['shard_index']):03d}.json",
                "manifest_hash": m.get("manifest_sha256"),
            }
            for m in sorted(shard_manifests, key=lambda item: int(item["shard_index"]))
        ],
        "verification": verification,
    }
    payload["manifest_sha256"] = manifest_sha256(payload)
    return payload


def verify_run_id_partition(run_id_groups: Iterable[Iterable[str]], expected_run_ids: set[str]) -> dict:
    counter: Counter[str] = Counter()
    for group in run_id_groups:
        counter.update(group)
    duplicate_run_count = sum(count - 1 for count in counter.values() if count > 1)
    observed = set(counter)
    missing = sorted(expected_run_ids - observed)
    unexpected = sorted(observed - expected_run_ids)
    return {
        "union_run_count": len(observed),
        "duplicate_run_count": duplicate_run_count,
        "missing_run_count": len(missing),
        "unexpected_run_count": len(unexpected),
        "missing_run_ids": missing[:20],
        "unexpected_run_ids": unexpected[:20],
        "passed": duplicate_run_count == 0 and not missing and not unexpected,
    }


def validate_shard_manifest(manifest: dict, path: str | Path | None = None) -> None:
    if not isinstance(manifest, dict):
        raise ShardingError(f"Shard manifest is not a JSON object: {path}")
    if manifest.get("schema_version") != SHARD_SCHEMA_VERSION:
        raise ShardingError(f"Unsupported shard manifest schema_version: {manifest.get('schema_version')}")
    validate_shard_bounds(int(manifest.get("num_shards", 0)), int(manifest.get("shard_index", -1)))
    run_ids = manifest.get("run_ids")
    if not isinstance(run_ids, list) or not all(isinstance(item, str) for item in run_ids):
        raise ShardingError("Shard manifest run_ids must be a list of strings")
    if int(manifest.get("shard_run_count", -1)) != len(run_ids):
        raise ShardingError("Shard manifest shard_run_count does not match run_ids length")
    expected_sha = manifest.get("manifest_sha256")
    if expected_sha and manifest_sha256(manifest) != expected_sha:
        raise ShardingError("Shard manifest_sha256 verification failed")


def sha256_json(data: object) -> str:
    return sha256_text(json.dumps(data, sort_keys=True, separators=(",", ":"), default=str))


def manifest_sha256(data: dict) -> str:
    payload = dict(data)
    payload.pop("manifest_sha256", None)
    return sha256_json(payload)


def _duplicate_items(items: Iterable[str]) -> list[str]:
    counts = Counter(items)
    return sorted(item for item, count in counts.items() if count > 1)
