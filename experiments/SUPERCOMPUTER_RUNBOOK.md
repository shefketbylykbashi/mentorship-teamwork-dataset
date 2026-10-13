# Supercomputer Full Solver Campaign Runbook

This runbook executes the accepted final benchmark dataset against all configured
C++ solvers, with 10 executions per solver-instance pair and a 3 minute solver
time limit.

Do this on a compute node or through the scheduler, not on a login node.

This runbook does not require reinstalling or updating the operating system.
For Linux tool checks, optional missing-package commands, module examples, and
hardware/resource audit commands, see
`experiments/LINUX_SETUP_AND_RESOURCE_CHECKS.md`.

## 1. Clone And Enter The Repository

```bash
git clone <YOUR_REPO_URL> mentorship-teamwork-dataset
cd mentorship-teamwork-dataset
git status --short
git rev-parse HEAD
```

Record the commit hash printed by `git rev-parse HEAD` in your experiment notes.

## 2. Prepare Or Load Compiler And Python

The solver build needs a C++20-capable `g++` and Python 3. If those tools are
already available, do not install or update anything. On a supercomputer, use
the module names for your cluster. Examples:

```bash
module purge
module load gcc
module load python
```

On a personal or cloud Linux machine, install only missing user-space tools.
Example for Debian/Ubuntu:

```bash
sudo apt-get install -y build-essential g++ gcc git python3 python-is-python3 python3-venv python3-pip rsync time procps coreutils util-linux
```

For RHEL/Rocky/AlmaLinux, Fedora, SUSE, and Arch commands, or to skip
installation entirely when tools are already present, use
`experiments/LINUX_SETUP_AND_RESOURCE_CHECKS.md`.

Check that the tools are visible:

```bash
which g++
g++ --version
which python3 || which python
python3 --version || python --version
```

The solver build expects a C++20-capable compiler.

If your environment exposes only `python3`, replace `python -m ...` with
`python3 -m ...` in the commands below, or activate a virtual environment first.

## 3. Resource Audit And Thread Budget

Before launching the full run, record what the node actually provides:

```bash
hostname
date -u +%Y-%m-%dT%H:%M:%SZ
uname -a
lscpu
nproc
free -h
df -h .
ulimit -a
```

On SLURM, also record the scheduler allocation:

```bash
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"
echo "SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-}"
scontrol show job "$SLURM_JOB_ID"
```

The current ten solvers are configured with `cpu_slots_per_run: 1`, so
`--threads` is the maximum number of concurrent solver processes. Set it no
higher than the CPU cores granted by the scheduler; for example, use
`--threads 50` only inside a 50-core allocation.

For a more complete resource checklist, including `sinfo`, `squeue`, memory,
storage, and live monitoring commands, see
`experiments/LINUX_SETUP_AND_RESOURCE_CHECKS.md`.

## 4. Optional Python Environment

The runner uses the Python standard library plus the existing repository code.
If your cluster requires a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

If `pytest` is not available and you want to run tests:

```bash
python -m pip install pytest
```

## 5. Build Linux Solver Binaries

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

## 6. Dry Run The Full Campaign

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

## 7. Run The Full Campaign Interactively Inside An Allocation

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

## 8. Optional SLURM Batch Script

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

echo "Job ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK}"
lscpu
free -h
df -h .

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
mkdir -p logs
sbatch run_full_mtp_campaign.slurm
```

Monitor it:

```bash
squeue -u "$USER"
tail -f logs/mtp_full_<JOBID>.out
```

## 9. Resume After Interruption

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

## 10. Rebuild Reports Anytime

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

## 11. Copy Results Back

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

