# MTP Solver Campaign Execution

This package builds and runs the ten configured C++ solvers against the accepted
final benchmark instances. It does not launch the full campaign unless you run
`experiments.run_mtp_campaign` without `--dry-run`.

## Build Solvers

Windows or Linux, with a C++20 compiler on `PATH`:

```bash
python -m experiments.build_solvers \
  --config experiments/configs/mtp_solver_campaign.yaml
```

Override the compiler when needed:

```bash
python -m experiments.build_solvers --compiler clang++
```

## Dry Run

```bash
python -m experiments.run_mtp_campaign \
  --campaign-id pc_trial \
  --runs 10 \
  --time-limit 180 \
  --threads 8 \
  --resume \
  --dry-run
```

The dry run discovers benchmark files, checks configured solver binaries,
derives run IDs and seeds, calculates total planned runs and prints an example
command. It executes no solver process.

## Run Campaign

Local validation campaign with two instances per benchmark family, all solvers,
and ten executions per solver-instance pair:

```powershell
python -m experiments.run_mtp_campaign `
  --campaign-id local_two_per_type `
  --instances-per-family 2 `
  --runs 10 `
  --time-limit 180 `
  --threads 8 `
  --resume
```

This currently plans 14 benchmark instances when all Type 1-7 families are
present: 2 instances x 7 families x 10 solvers x 10 executions = 1,400 planned
runs.

PowerShell:

```powershell
python -m experiments.run_mtp_campaign `
  --campaign-id pc_trial `
  --runs 10 `
  --time-limit 180 `
  --threads 8 `
  --resume
```

Bash on a larger Linux node:

```bash
python -m experiments.run_mtp_campaign \
  --campaign-id supercomputer_trial \
  --runs 10 \
  --time-limit 180 \
  --threads 50 \
  --resume
```

`--threads` is the total CPU-slot budget for the whole runner. Each configured
solver currently consumes one slot per run, so `--threads 50` permits up to 50
simultaneous solver processes unless `--max-parallel-runs` is smaller.

## Rebuild Reports

```bash
python -m experiments.run_mtp_campaign \
  --campaign-id pc_trial \
  --rebuild-reports
```

Reports are reconstructed from authoritative per-run JSON/status files under
the campaign directory.

## Output Layout

```text
experiment_results/<campaign-id>/
  campaign_manifest.json
  benchmark_manifest.json
  solver_manifest.json
  progress.json
  runs/<solver>/<instance>/exec_01/
    status.json
    result.json
    solution.txt
    stdout.log
    stderr.log
  best_results/<solver>/<instance>/
    best_result.json
    best_solution.txt
  summaries/
    per_run_results.csv
    solver_instance_summary.csv
    best_result_per_solver_instance.csv
    solver_summary.csv
    instance_summary.csv
    failure_report.csv
    timeout_report.csv
    invalid_output_report.csv
    campaign_summary.json
    campaign_summary.csv
```

