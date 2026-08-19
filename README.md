# MTP Instance Benchmark Generation

This repository contains a research-grade generation pipeline for synthetic
Mentorship and Teamwork Problem benchmark families:

```text
Type 2: gap-filling instances between the five non-trivial Google anchors
Type 3: neighborhood instances around the five non-trivial Google anchors
Type 4: broad random instances across scale bands
Type 5: empirically grounded synthetic instances from public GitHub activity
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
written to config or metadata.

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
  --count 100 \
  --seed 2026 \
  --overwrite \
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

## Outputs

Generated instances use deterministic sortable names:

```text
generated_instances/type2/type2_001.txt ... type2_100.txt
generated_instances/type3/type3_001.txt ... type3_100.txt
generated_instances/type4/type4_001.txt ... type4_100.txt
generated_instances/type5/type5_001.txt ... type5_100.txt
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
type*_statistical_report.json
benchmark_manifest.json
```

See `METHODOLOGY.md` for the mathematical definitions, validation rules,
distance metric, search procedure and documented implementation choices.
