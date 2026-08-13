# MTP Instance Benchmark Generation

This repository contains a research-grade generation pipeline for synthetic
Mentorship and Teamwork Problem benchmark families:

```text
Type 2: gap-filling instances between the five non-trivial Google anchors
Type 3: neighborhood instances around the five non-trivial Google anchors
Type 4: broad random instances across scale bands
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
```

Metadata and reports are written under `benchmark_metadata/`, including:

```text
google_anchor_features.csv/jsonl
type2_metadata.csv/jsonl
type3_metadata.csv/jsonl
type4_metadata.csv/jsonl
type*_statistical_report.json
benchmark_manifest.json
```

See `METHODOLOGY.md` for the mathematical definitions, validation rules,
distance metric, search procedure and documented implementation choices.
