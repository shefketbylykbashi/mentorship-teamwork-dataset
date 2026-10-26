# MTP Instance Benchmark Generation

This repository contains a research-grade generation pipeline for synthetic
Mentorship and Teamwork Problem benchmark families:

```text
Type 2: gap-filling instances between the five non-trivial Google anchors
Type 3: neighborhood instances around the five non-trivial Google anchors
Type 4: broad random instances across scale bands
Type 5: empirically grounded synthetic instances from public GitHub activity
Type 7: remaining coverage-driven synthetic gap-filling instances
```

The original files in `google_instances/` are immutable inputs. The toy
`a_an_example.txt` is excluded from anchor-based generation.

## Commands

Generate Type 4:

```bash
python -m benchmark_generation.type4.generate_type4 \
  --output-dir generated_instances/type4 \
  --metadata-dir benchmark_metadata \
  --count 100 \
  --seed 2026 \
  --config benchmark_generation/configs/type4.yaml
```

Generate Type 2:

```bash
python -m benchmark_generation.type2.generate_type2 \
  --google-dir google_instances \
  --output-dir generated_instances/type2 \
  --metadata-dir benchmark_metadata \
  --count 100 \
  --seed 2026 \
  --config benchmark_generation/configs/type2.yaml
```

Generate Type 3:

```bash
python -m benchmark_generation.type3.generate_type3 \
  --google-dir google_instances \
  --output-dir generated_instances/type3 \
  --metadata-dir benchmark_metadata \
  --count 100 \
  --seed 2026 \
  --config benchmark_generation/configs/type3.yaml
```

Generate Type 5 in four stages. Bulk collection requires authenticated public
GitHub access through `GITHUB_TOKEN` or `gh auth login`; the token is never
written to config or metadata. The empirical build evaluates the 100 primary
repository-window slots, applies the fixed source-quality gates and deterministic
reserve replacement, then records the retained final family size in
`benchmark_metadata/type5_empirical_corpus.json`.

```bash
python -m benchmark_generation.type5.collect_github_corpus \
  --sources benchmark_generation/configs/type5_sources.yaml \
  --cache-dir type5_source_cache \
  --metadata-dir benchmark_metadata \
  --seed 2026 \
  --verbose
```

```bash
python -m benchmark_generation.type5.build_empirical_profiles \
  --sources benchmark_generation/configs/type5_sources.yaml \
  --cache-dir type5_source_cache \
  --metadata-dir benchmark_metadata \
  --config benchmark_generation/configs/type5.yaml \
  --seed 2026 \
  --verbose
```

```bash
python -m benchmark_generation.type5.generate_type5 \
  --output-dir generated_instances/type5_pilot \
  --metadata-dir benchmark_metadata/type5_pilot \
  --config benchmark_generation/configs/type5.yaml \
  --count 10 \
  --seed 2026 \
  --pilot \
  --verbose
```

```bash
python -m benchmark_generation.type5.generate_type5 \
  --output-dir generated_instances/type5 \
  --metadata-dir benchmark_metadata \
  --config benchmark_generation/configs/type5.yaml \
  --seed 2026 \
  --overwrite \
  --verbose
```

For the current collected corpus, the empirical build retained 91 valid source
configurations, so the final command writes `type5_001.txt` through
`type5_091.txt`. You may pass `--count 91` explicitly, but any other non-pilot
count is rejected.

Extend Type 5 with complete eligible GitHub project histories. This is an
opt-in extension and does not regenerate, rename, or overwrite the accepted
`type5_001.txt` through `type5_091.txt` files. Collection again requires
authenticated public GitHub access.

To resume source collection from the existing cache, rerun the collection
command below with the same seed. Completed candidate summaries and PR file
details under `type5_full_project_source_cache` are reused.

```bash
python -m benchmark_generation.type5.collect_github_full_projects \
  --config benchmark_generation/configs/type5_full_project.yaml \
  --cache-dir type5_full_project_source_cache \
  --metadata-dir benchmark_metadata \
  --seed 2026 \
  --verbose
```

```bash
python -m benchmark_generation.type5.build_full_project_profiles \
  --cache-dir type5_full_project_source_cache \
  --metadata-dir benchmark_metadata \
  --config benchmark_generation/configs/type5_full_project.yaml \
  --seed 2026 \
  --verbose
```

```bash
python -m benchmark_generation.type5.generate_type5_full_project \
  --output-dir generated_instances/type5 \
  --metadata-dir benchmark_metadata \
  --pilot-output-dir generated_instances/type5_full_project_pilot \
  --pilot-metadata-dir benchmark_metadata/type5_full_project_pilot \
  --config benchmark_generation/configs/type5_full_project.yaml \
  --seed 2026 \
  --verbose
```

The full-project generator first writes a five-instance pilot from the frozen
selected sources. With the current 17-source full-project selection, if that
pilot passes all gates, it appends `type5_092__GHFULL__...txt` through
`type5_108__GHFULL__...txt`.

Generate Type 7 after Types 1-6 are frozen. The generator audits the actual
files currently present in Types 1-6, writes a frozen manifest, freezes the
coverage feature space, generates a 10-instance pilot, and if the pilot passes
continues automatically to the final 100 Type 7 instances.

```bash
python -m benchmark_generation.type7.generate_type7 \
  --google-dir google_instances \
  --generated-root generated_instances \
  --pilot-output-dir generated_instances/type7_pilot \
  --output-dir generated_instances/type7 \
  --pilot-metadata-dir benchmark_metadata/type7_pilot \
  --metadata-dir benchmark_metadata \
  --count 100 \
  --seed 2026 \
  --config benchmark_generation/configs/type7.yaml \
  --verbose
```

Use `--overwrite` only when intentionally replacing generated files. The
generators never overwrite existing instance files silently.

Run diagnostics after generation:

```bash
python -m benchmark_generation.analysis.diagnostics \
  --metadata-dir benchmark_metadata \
  --output-dir benchmark_metadata/diagnostics
```

Run tests:

```bash
pytest benchmark_generation/tests
```

## Solver Campaign Execution

The production runner for solver experiments lives in `experiments/`. It builds
the ten configured C++ solvers once, discovers accepted final benchmark files at
run time, derives deterministic seeds for every solver-instance-execution tuple,
and persists every run independently.

Build configured solvers:

```bash
python -m experiments.build_solvers \
  --config experiments/configs/mtp_solver_campaign.yaml
```

Inspect a campaign without launching solvers:

```bash
python -m experiments.run_mtp_campaign \
  --campaign-id pc_trial \
  --runs 10 \
  --time-limit 180 \
  --threads 8 \
  --resume \
  --dry-run
```

Local validation campaign: two accepted instances from every Type 1-7 family,
all ten solvers, ten executions each:

```bash
python -m experiments.run_mtp_campaign \
  --campaign-id local_two_per_type \
  --instances-per-family 2 \
  --runs 10 \
  --time-limit 180 \
  --threads 8 \
  --resume \
  --dry-run
```

Remove `--dry-run` after reviewing the plan and confirming the solver binaries
have been built.

Run locally after reviewing the dry run:

```bash
python -m experiments.run_mtp_campaign \
  --campaign-id pc_trial \
  --runs 10 \
  --time-limit 180 \
  --threads 8 \
  --resume
```

Rebuild reports from per-run JSON without rerunning solvers:

```bash
python -m experiments.run_mtp_campaign \
  --campaign-id pc_trial \
  --rebuild-reports
```

See `experiments/README.md` for PowerShell/Linux examples, campaign layout,
resume behavior, and report files.

## First Instance Space Analysis

The executed `rizana_trial_01` campaign has a separate, audited ISA handoff.
It prepares one row per executed instance, selects source-distinct solver
representatives without looking at outcomes, and exports MATLAB
InstanceSpace-compatible metadata with timeout-aware performance. From the
repository root, run:

```bash
python -B -m isa.prepare_isa --campaign experiment_results/rizana_trial_01 --output isa/outputs/rizana_trial_01 --overwrite
python -B -m isa.verify_legacy_features --input isa/outputs/rizana_trial_01
python -B -m isa.summarize_export --input isa/outputs/rizana_trial_01 --overwrite
```

Then follow [`isa/README.md`](isa/README.md) to run the pinned MATLAB toolkit,
inspect descriptive footprints, and perform 1%/5%/10% sensitivity analyses.
The current checkout does not include MATLAB or the campaign's per-run result
table, so this step prepares the data but does not claim to have completed the
MATLAB projection or an out-of-sample algorithm-selection evaluation.

## Outputs

Generated instances use deterministic sortable names:

```text
generated_instances/type2/type2_001.txt ... type2_100.txt
generated_instances/type3/type3_001.txt ... type3_100.txt
generated_instances/type4/type4_001.txt ... type4_100.txt
generated_instances/type5/type5_001.txt ... type5_091.txt
generated_instances/type5/type5_092__GHFULL__...txt ... type5_108__GHFULL__...txt
generated_instances/type7/type7_001.txt ... type7_100.txt
```

Metadata and reports are written under `benchmark_metadata/`, including:

```text
google_anchor_features.csv/jsonl
type2_metadata.csv/jsonl
type3_metadata.csv/jsonl
type4_metadata.csv/jsonl
type5_metadata.csv/jsonl
type5_empirical_corpus.csv/json
type5_source_windows.csv
type5_data_provenance.json
TYPE5_SOURCES.md
type7_existing_benchmark_manifest.csv/json
type7_feature_space_definition.json
type7_feasibility_calibration.json
type7_gap_analysis.json
type7_metadata.csv/jsonl
type7_coverage_before_after.json
type7_search_manifest.json
TYPE7_METHODOLOGY.md
type*_statistical_report.json
benchmark_manifest.json
```

See `METHODOLOGY.md` for the mathematical definitions, validation rules,
distance metric, search procedure and documented implementation choices.
