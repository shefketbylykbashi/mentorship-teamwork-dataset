"""Create deterministic shard manifests for an MTP solver campaign."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from benchmark_generation.core.config import load_config

from .benchmark_discovery import benchmark_manifest, discover_final_benchmark, limit_instances_per_family
from .common import atomic_write_json, config_hash, repo_root
from .run_mtp_campaign import make_run_plan
from .sharding import build_shard_manifest, build_top_level_manifest, select_shard, sha256_json
from .solver_adapters import load_solver_definitions, solver_manifest


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.num_shards < 1:
        raise SystemExit("--num-shards must be >= 1")

    root = repo_root()
    config = load_config(args.config, defaults={})
    if args.runs is not None:
        config["runs"] = args.runs
    if args.instances_per_family is not None:
        config["instances_per_family"] = args.instances_per_family

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"Output directory is not empty; use --overwrite: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_instances = discover_final_benchmark(root, config.get("benchmark", {}))
    instances_per_family = config.get("instances_per_family")
    instances = limit_instances_per_family(
        all_instances,
        int(instances_per_family) if instances_per_family is not None else None,
    )
    solvers = load_solver_definitions(config)
    runs = int(config.get("runs", 10))
    root_seed = int(config.get("root_seed", 2026))

    bench_manifest = benchmark_manifest(instances)
    solv_manifest = solver_manifest(solvers, root)
    full_plan = make_run_plan(instances, solvers, runs, root_seed)
    bench_hash = sha256_json(bench_manifest)
    solver_hash = sha256_json(solv_manifest)
    cfg_hash = config_hash(config)

    shard_manifests = []
    for shard_index in range(args.num_shards):
        selected, _selection = select_shard(full_plan, args.num_shards, shard_index)
        manifest = build_shard_manifest(
            run_plan=full_plan,
            selected=selected,
            shard_index=shard_index,
            num_shards=args.num_shards,
            root_seed=root_seed,
            runs_per_solver_instance=runs,
            benchmark_manifest_hash=bench_hash,
            solver_manifest_hash=solver_hash,
            campaign_config_hash=cfg_hash,
        )
        shard_manifests.append(manifest)
        atomic_write_json(output_dir / f"shard_{shard_index:03d}.json", manifest)

    top_manifest = build_top_level_manifest(shard_manifests, full_plan)
    atomic_write_json(output_dir / "shards_manifest.json", top_manifest)

    print(f"Wrote {args.num_shards} shard manifests to {output_dir}")
    print(f"Global run count: {len(full_plan)}")
    print(f"Shard size range: {top_manifest['minimum_shard_size']}..{top_manifest['maximum_shard_size']}")
    print(f"Verification passed: {top_manifest['verification']['passed']}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/configs/mtp_solver_campaign.yaml")
    parser.add_argument("--runs", type=int)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--instances-per-family", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
