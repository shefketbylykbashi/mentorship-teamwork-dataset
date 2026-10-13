# Supercomputer Full Solver Campaign Runbook

This runbook executes the accepted final benchmark dataset against all configured
C++ solvers, with 10 executions per solver-instance pair and a 3 minute solver
time limit.

Do this on a compute node or through the scheduler, not on a login node.

## 1. Clone And Enter The Repository

```bash
git clone <YOUR_REPO_URL> mentorship-teamwork-dataset
cd mentorship-teamwork-dataset
git status --short
git rev-parse HEAD
```

Record the commit hash printed by `git rev-parse HEAD` in your experiment notes.

## 2. Load Compiler And Python

Use the module names for your cluster. Examples:

```bash
module purge
module load gcc
module load python
```

Check that the tools are visible:

```bash
g++ --version
python --version
```

The solver build expects a C++20-capable compiler.

## 3. Optional Python Environment

The runner uses the Python standard library plus the existing repository code.
If your cluster requires a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

If `pytest` is not available and you want to run tests:

```bash
python -m pip install pytest
```

## 4. Build Linux Solver Binaries

Do not copy Windows `.exe` files to the supercomputer. Build native Linux
binaries on the cluster:

```bash
python -m experiments.build_solvers \
  --config experiments/configs/mtp_solver_campaign.yaml
```

If the cluster compiler is named differently:

```bash
python -m experiments.build_solvers \
  --config experiments/configs/mtp_solver_campaign.yaml \
  --compiler g++
```

Confirm the build manifest reports successful builds:

```bash
python - <<'PY'
import json
from pathlib import Path
m = json.loads(Path("experiments/solver_manifest.json").read_text())
print([(row["solver_id"], row["status"]) for row in m["solvers"]])
PY
```

Every solver should show `SUCCESS`.

## 5. Dry Run The Full Campaign

This command plans the full dataset. Notice there is no
`--instances-per-family` filter.

For a 50 CPU-slot node:

```bash
python -m experiments.run_mtp_campaign \
  --campaign-id full_dataset_10runs_180s \
  --runs 10 \
  --time-limit 180 \
  --threads 50 \
  --resume \
  --dry-run
```

Check the dry-run output:

- `Missing binaries: 0`
- benchmark family counts look correct
- total planned executions equals:

```text
number_of_discovered_instances x 10 solvers x 10 runs
```

In the current local checkout this was 611 instances, so the planned total was
61,100 runs. Do not hardcode this number; let the runner discover it.

## 5A. Optional Kaggle CPU Shards

Kaggle CPU notebooks currently provide four logical CPU cores. The configured
C++ solvers are single-threaded, so use `--threads 4` to run up to four solver
processes concurrently. Sharding is optional and does not alter run IDs, solver
seeds, solver options, time-limit semantics, or evaluator behavior.

The canonical shard order is:

```text
(instance_family, instance_id, solver_id, execution_index, run_id)
```

After sorting by that key, each run is assigned by:

```text
shard_index = global_sorted_run_index % num_shards
```

For the current accepted dataset this derives 611 instances x 10 solvers x 10
executions = 61,100 runs. With 77 shards the distribution is 39 shards of 794
runs and 38 shards of 793 runs.

Kaggle shard 000 dry run:

```bash
python -m experiments.run_mtp_campaign \
  --campaign-id kaggle_shard_000 \
  --runs 10 \
  --time-limit 180 \
  --timeout-grace 15 \
  --threads 4 \
  --num-shards 77 \
  --shard-index 0 \
  --results-dir /kaggle/working/experiment_results \
  --resume \
  --dry-run
```

Kaggle shard 000 execution:

```bash
python -m experiments.run_mtp_campaign \
  --campaign-id kaggle_shard_000 \
  --runs 10 \
  --time-limit 180 \
  --timeout-grace 15 \
  --threads 4 \
  --num-shards 77 \
  --shard-index 0 \
  --results-dir /kaggle/working/experiment_results \
  --resume
```

Final shard:

```bash
python -m experiments.run_mtp_campaign \
  --campaign-id kaggle_shard_076 \
  --runs 10 \
  --time-limit 180 \
  --timeout-grace 15 \
  --threads 4 \
  --num-shards 77 \
  --shard-index 76 \
  --results-dir /kaggle/working/experiment_results \
  --resume
```

Optional shard manifests can be generated with:

```bash
python -m experiments.create_shards \
  --config experiments/configs/mtp_solver_campaign.yaml \
  --runs 10 \
  --num-shards 77 \
  --output-dir experiments/shards/kaggle_77
```

The manifest form for shard 000 is equivalent to `--num-shards 77
--shard-index 0`:

```bash
python -m experiments.run_mtp_campaign \
  --campaign-id kaggle_shard_000 \
  --shard-manifest experiments/shards/kaggle_77/shard_000.json \
  --time-limit 180 \
  --timeout-grace 15 \
  --threads 4 \
  --results-dir /kaggle/working/experiment_results \
  --resume
```

After all 77 Kaggle outputs have been saved locally, merge them without running
any solver:

```bash
python -m experiments.merge_shards \
  --input-root kaggle_shard_results \
  --num-shards 77 \
  --output-dir experiment_results/kaggle_merged
```

## 6. Run The Full Campaign Interactively Inside An Allocation

If your cluster gives you an interactive allocation, use the same command without
`--dry-run`:

```bash
python -m experiments.run_mtp_campaign \
  --campaign-id full_dataset_10runs_180s \
  --runs 10 \
  --time-limit 180 \
  --threads 50 \
  --resume
```

`--threads 50` means the total CPU-slot budget for the runner. The current ten
solvers are single-threaded, so this permits up to 50 solver processes at once.

## 7. Optional SLURM Batch Script

If your supercomputer uses SLURM, create a job script like this. Adjust account,
partition, wall time, memory, and module names for your site. The example uses
72 hours because the full 61,100-run campaign has a worst-case lower bound of
about 61 hours at 50 concurrent slots before timeout grace and scheduling
overhead.

```bash
cat > run_full_mtp_campaign.slurm <<'EOF'
#!/bin/bash
#SBATCH --job-name=mtp_full_180s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --time=72:00:00
#SBATCH --output=logs/mtp_full_%j.out
#SBATCH --error=logs/mtp_full_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module purge
module load gcc
module load python

python -m experiments.build_solvers \
  --config experiments/configs/mtp_solver_campaign.yaml

python -m experiments.run_mtp_campaign \
  --campaign-id full_dataset_10runs_180s \
  --runs 10 \
  --time-limit 180 \
  --threads "${SLURM_CPUS_PER_TASK}" \
  --resume
EOF
```

Submit it:

```bash
sbatch run_full_mtp_campaign.slurm
```

Monitor it:

```bash
squeue -u "$USER"
tail -f logs/mtp_full_<JOBID>.out
```

## 8. Resume After Interruption

If the job is killed, the SSH connection drops, or the node stops, rerun the same
campaign command with the same `--campaign-id` and `--resume`:

```bash
python -m experiments.run_mtp_campaign \
  --campaign-id full_dataset_10runs_180s \
  --runs 10 \
  --time-limit 180 \
  --threads 50 \
  --resume
```

Already valid `SUCCESS` runs are skipped. Stale `RUNNING` entries are recovered
as `INTERRUPTED` and become eligible to run again.

## 9. Rebuild Reports Anytime

Reports can be rebuilt without rerunning solvers:

```bash
python -m experiments.run_mtp_campaign \
  --campaign-id full_dataset_10runs_180s \
  --rebuild-reports
```

Important report files:

```text
experiment_results/full_dataset_10runs_180s/progress.json
experiment_results/full_dataset_10runs_180s/summaries/per_run_results.csv
experiment_results/full_dataset_10runs_180s/summaries/best_result_per_solver_instance.csv
experiment_results/full_dataset_10runs_180s/summaries/solver_instance_summary.csv
experiment_results/full_dataset_10runs_180s/summaries/solver_summary.csv
experiment_results/full_dataset_10runs_180s/summaries/instance_summary.csv
experiment_results/full_dataset_10runs_180s/summaries/failure_report.csv
experiment_results/full_dataset_10runs_180s/summaries/timeout_report.csv
experiment_results/full_dataset_10runs_180s/summaries/campaign_summary.json
```

Quick status:

```bash
cat experiment_results/full_dataset_10runs_180s/progress.json
```

Status counts from per-run CSV:

```bash
python - <<'PY'
import csv
from collections import Counter
from pathlib import Path
p = Path("experiment_results/full_dataset_10runs_180s/summaries/per_run_results.csv")
with p.open(newline="", encoding="utf-8") as f:
    print(Counter(row["status"] for row in csv.DictReader(f)))
PY
```

## 10. Copy Results Back

After completion, copy the campaign directory back from the supercomputer:

```bash
rsync -avh \
  user@cluster:/path/to/mentorship-teamwork-dataset/experiment_results/full_dataset_10runs_180s/ \
  experiment_results/full_dataset_10runs_180s/
```

The authoritative data are the per-run directories and JSON files under:

```text
experiment_results/full_dataset_10runs_180s/runs/
```

Do not treat `ERROR`, `TIMEOUT`, `INVALID_OUTPUT`, or `INVALID_SOLUTION` as
score zero. Scores are meaningful only for validated `SUCCESS` runs.

