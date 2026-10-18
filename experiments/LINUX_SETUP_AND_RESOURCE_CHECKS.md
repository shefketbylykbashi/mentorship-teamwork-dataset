# Linux Setup And Resource Checks

This checklist prepares a Linux machine for the MTP solver experiment and records
the hardware resources available for a defensible campaign run. It does not
require reinstalling, upgrading, or otherwise changing the operating system.

First verify the available tools. On a managed supercomputer, prefer environment
modules or the site-provided compiler/Python stack over `sudo`. Use package
manager commands only on a machine you administer, and only when a required tool
is missing.

## 1. Verify Required Tools

Run this first:

```bash
which g++ || true
g++ --version || true
which python3 || which python || true
python3 --version || python --version || true
which git || true
which rsync || true
```

If `g++`, Python 3, `git`, and `rsync` are already available, skip the package
installation section.

## 2. Install Missing User-Space Tools Only

These commands install missing development tools. They are not an instruction to
upgrade the operating system. On a supercomputer, normally skip this section and
use modules instead.

### Debian / Ubuntu

```bash
sudo apt-get install -y \
  build-essential \
  g++ \
  gcc \
  git \
  python3 \
  python-is-python3 \
  python3-venv \
  python3-pip \
  rsync \
  time \
  procps \
  coreutils \
  util-linux
```

### RHEL / Rocky / AlmaLinux / Fedora

```bash
sudo dnf install -y \
  gcc \
  gcc-c++ \
  make \
  git \
  python3 \
  python3-pip \
  rsync \
  time \
  procps-ng \
  coreutils \
  util-linux
```

If `dnf` is unavailable on an older RHEL/CentOS system, use `yum` with the same
package names:

```bash
sudo yum install -y gcc gcc-c++ make git python3 python3-pip rsync time procps-ng coreutils util-linux
```

### openSUSE / SUSE

```bash
sudo zypper install -y \
  gcc \
  gcc-c++ \
  make \
  git \
  python3 \
  python3-pip \
  rsync \
  time \
  procps \
  coreutils \
  util-linux
```

### Arch Linux

```bash
sudo pacman -S --needed \
  base-devel \
  gcc \
  git \
  python \
  python-pip \
  rsync \
  time \
  procps-ng \
  coreutils \
  util-linux
```

## 3. Supercomputer Module Setup

If the cluster uses modules, inspect and load the site-specific compiler and
Python modules instead of installing packages:

```bash
module avail gcc
module avail python
module avail cmake
module purge
module load gcc
module load python
```

Some sites use `ml` as a shorthand:

```bash
ml avail gcc
ml avail python
ml purge
ml gcc
ml python
```

Verify that the loaded tools are the ones that will be used:

```bash
which g++
g++ --version
which python3 || which python
python3 --version || python --version
```

The configured solver build requires a C++20-capable compiler. GCC 10 or newer is
recommended.

The repository examples use `python -m ...`. If your Linux environment exposes
only `python3`, either run the commands as `python3 -m ...` or activate a virtual
environment, where `python` normally points to the selected Python 3 interpreter.

## 4. Optional Python Virtual Environment

The solver campaign uses the Python standard library and this repository's
modules. `pytest` is only needed for test runs. The repository configuration
files are JSON-compatible YAML, so PyYAML is optional for the production solver
campaign.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pytest
```

If `python3 -m venv` is disabled on a supercomputer, use the site-supported
environment mechanism or run with the loaded Python module directly.

## 5. Repository Smoke Check

Run this from the repository root:

```bash
git status --short
git rev-parse HEAD
python3 --version || python --version
g++ --version
python3 -m experiments.build_solvers \
  --config experiments/configs/mtp_solver_campaign.yaml
python3 -m experiments.run_mtp_campaign \
  --campaign-id linux_smoke_dry_run \
  --instances-per-family 1 \
  --runs 1 \
  --time-limit 10 \
  --threads 2 \
  --resume \
  --dry-run
```

Every solver in `experiments/solver_manifest.json` should report `SUCCESS`
after the build.

## 6. Machine Resource Audit

Save this output with the experiment notes before choosing `--threads` or SLURM
resource requests.

```bash
echo "Host: $(hostname)"
echo "Date UTC: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
uname -a
echo
echo "CPU summary"
lscpu
echo
echo "Usable processing units"
nproc --all
nproc
echo
echo "Memory"
free -h
grep -E 'MemTotal|MemAvailable|SwapTotal|SwapFree' /proc/meminfo
echo
echo "Storage"
df -h .
lsblk
echo
echo "Limits"
ulimit -a
echo
echo "Toolchain"
which g++ || true
g++ --version || true
which python3 || which python || true
python3 --version || python --version || true
```

Useful quick checks while a campaign is running:

```bash
uptime
ps -u "$USER" -o pid,ppid,stat,pcpu,pmem,etime,cmd --sort=-pcpu | head -40
top -u "$USER"
df -h experiment_results .
du -sh experiment_results/* 2>/dev/null | sort -h
```

## 7. SLURM Resource Discovery

Use these commands when the machine is managed by SLURM:

```bash
sinfo -o "%P %D %c %m %G %l %a"
squeue -u "$USER"
sacctmgr show user "$USER" format=User,Account,Partition,QOS%30
scontrol show partition
if [ -n "${SLURMD_NODENAME:-}" ]; then
  scontrol show node "$SLURMD_NODENAME"
fi
```

Inside an allocated job, record the allocation actually granted:

```bash
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-}"
echo "SLURM_NTASKS=${SLURM_NTASKS:-}"
echo "SLURM_MEM_PER_NODE=${SLURM_MEM_PER_NODE:-}"
echo "SLURM_MEM_PER_CPU=${SLURM_MEM_PER_CPU:-}"
scontrol show job "$SLURM_JOB_ID"
```

## 8. Choosing `--threads`

For the current campaign config, each solver run uses one CPU slot:

```text
cpu_slots_per_run = 1
```

Use:

```text
--threads = min(allocated CPU cores, desired concurrent solver processes)
```

Examples:

```bash
# 8-core workstation
python -m experiments.run_mtp_campaign --campaign-id local_8core --runs 10 --time-limit 180 --threads 8 --resume

# 50-core allocation
python -m experiments.run_mtp_campaign --campaign-id full_dataset_10runs_180s --runs 10 --time-limit 180 --threads 50 --resume
```

Leave one or two cores unused if the node is shared or if filesystem/reporting
overhead becomes visible. Do not set `--threads` higher than the CPU cores
granted by the scheduler.

## 9. Output And Transfer Tools

The runner writes authoritative per-run outputs under:

```text
experiment_results/<campaign-id>/
```

Copy results back with:

```bash
rsync -avh \
  user@cluster:/path/to/mentorship-teamwork-dataset/experiment_results/<campaign-id>/ \
  experiment_results/<campaign-id>/
```

