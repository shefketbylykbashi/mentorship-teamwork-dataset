# Instance generators

The generators for Types 2 to 7. Types 2, 3, 4, 5 and 7 draw instances from one parameterised generator (`core/generator.py`); Type 6 converts files of the MSLIB library. [METHODOLOGY.md](../METHODOLOGY.md) describes the generator, the features used to steer it, and Types 2, 3, 4, 5 and 7. Type 6 is described in `type6/TYPE6_SPEC.md` and `benchmark_metadata/type6_methodology_draft.md`, and Type 7 in more detail in `benchmark_metadata/TYPE7_METHODOLOGY.md`.

Only the Python standard library is needed. The configuration files in `configs/` are JSON-compatible YAML; `openpyxl` is needed only for the Type 6 workbook checks. The original files in `google_instances/` are read-only inputs.

Each generator writes a family of instances, numbered from 001, together with its metadata. The commands below write to a new folder, `regenerated/`, so that the benchmark files in `generated_instances/` and the tables in `benchmark_metadata/` are left as they are. The generators never overwrite existing instance files unless `--overwrite` is given.

## Commands

Generate Type 4:

```bash
python -m benchmark_generation.type4.generate_type4 \
  --output-dir regenerated/type4 \
  --metadata-dir regenerated/metadata \
  --count 100 \
  --seed 2026 \
  --config benchmark_generation/configs/type4.yaml
```

Generate Type 2:

```bash
python -m benchmark_generation.type2.generate_type2 \
  --google-dir google_instances \
  --output-dir regenerated/type2 \
  --metadata-dir regenerated/metadata \
  --count 100 \
  --seed 2026 \
  --config benchmark_generation/configs/type2.yaml
```

Generate Type 3:

```bash
python -m benchmark_generation.type3.generate_type3 \
  --google-dir google_instances \
  --output-dir regenerated/type3 \
  --metadata-dir regenerated/metadata \
  --count 100 \
  --seed 2026 \
  --config benchmark_generation/configs/type3.yaml
```

Generate Type 5 in four stages. The first two collect activity data from GitHub and build the empirical corpus. The corpus behind the benchmark is already in `benchmark_metadata/type5_empirical_corpus.json`, and collecting again returns the repositories as they are today, so skip these two stages to reproduce the benchmark files. Bulk collection requires authenticated public
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
  --output-dir regenerated/type5_pilot \
  --metadata-dir regenerated/metadata/type5_pilot \
  --config benchmark_generation/configs/type5.yaml \
  --count 10 \
  --seed 2026 \
  --pilot \
  --verbose
```

```bash
python -m benchmark_generation.type5.generate_type5 \
  --output-dir regenerated/type5 \
  --metadata-dir regenerated/metadata \
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
  --output-dir regenerated/type5 \
  --metadata-dir regenerated/metadata \
  --pilot-output-dir regenerated/type5_full_project_pilot \
  --pilot-metadata-dir regenerated/metadata/type5_full_project_pilot \
  --config benchmark_generation/configs/type5_full_project.yaml \
  --seed 2026 \
  --verbose
```

The full-project generator first writes a five-instance pilot from the frozen
selected sources. With the current 17-source full-project selection, if that
pilot passes all gates, it appends `type5_092__GHFULL__...txt` through
`type5_108__GHFULL__...txt`.

Generate Type 7 after Types 1-6 are frozen. Type 7 is placed relative to the complete families of Types 2-6, so `--generated-root` must point to a folder that holds all of them (regenerate Types 2-6 into `regenerated/` first, Type 6 with `benchmark_generation.type6.generate_type6`; see `type6/TYPE6_SPEC.md`). The generator audits the actual
files currently present in Types 1-6, writes a frozen manifest, freezes the
coverage feature space, generates a 10-instance pilot, and if the pilot passes
continues automatically to the final 100 Type 7 instances.

```bash
python -m benchmark_generation.type7.generate_type7 \
  --google-dir google_instances \
  --generated-root regenerated \
  --pilot-output-dir regenerated/type7_pilot \
  --output-dir regenerated/type7 \
  --pilot-metadata-dir regenerated/metadata/type7_pilot \
  --metadata-dir regenerated/metadata \
  --count 100 \
  --seed 2026 \
  --config benchmark_generation/configs/type7.yaml \
  --verbose
```

Run the tests:

```bash
pytest benchmark_generation/tests
```
