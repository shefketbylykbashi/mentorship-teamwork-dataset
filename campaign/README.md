# Solver campaign (10 solvers × 611 instances)

Self-contained runner for the supercomputer campaign. Needs only **Python ≥ 3.9
(stdlib), g++ with C++20, Node ≥ 16**. No pip packages.

| Path | Purpose |
|---|---|
| `solvers/mtp_runtime.hpp` | shared contract: seed parsing, 590 s deadline, skill-capacity guard (`NS = 919`) |
| `campaign/build_solvers.sh` | builds the 10 binaries with `g++ -O2 -std=c++20` into `campaign/bin/` |
| `campaign/run_campaign.py` | runs everything in parallel, validates every output, resumable |
| `campaign/validator/upstream/` | Hash Code referee, vendored unmodified (see `VENDORED.md`) |
| `campaign/validator/batch_validate.js` | fast front-end: long-lived Node workers + strict re-check |
| `campaign/summarize.py` | CSV summaries from `results.jsonl` |
| `campaign/instances.py` | instance discovery (611) and skill-universe measurement |

## Solver contract

```
campaign/bin/<solver> SEED < instance.txt > solution.txt
```

* **Seeded** (10 runs, seeds 1–10): `refactored_b`, `refactored_d`, `evolved_b`, `evolved_d`, `evolved_e`.
* **Deterministic** (1 run, seed 1, seed ignored): `refactored_c`, `refactored_e`, `refactored_f`, `evolved_c`, `evolved_f`.
* Timer starts in the first statement of `main`; search stops at 590 s; the best
  valid solution found so far is written. The runner kills a run at 600 s
  (recorded as `timeout`, exit code 124 = failed).
* Exit 2 = bad/missing seed; exit 3 = instance has more than `NS` distinct skills.

Total: 5 × 10 × 611 + 5 × 1 × 611 = **33,605 runs**.

## On the supercomputer

```bash
git clone <repo-url> mentorship-teamwork-dataset
cd mentorship-teamwork-dataset

# 1. build (once per machine)
bash campaign/build_solvers.sh

# 2. check the plan (prints 611 instances / 33605 runs; runs nothing)
python3 campaign/run_campaign.py --dry-run

# 3. smoke test: 1 instance per family, all solvers, ~1-2 h at 64 jobs
python3 campaign/run_campaign.py --out campaign_results/smoke --instances-per-family 1

# 4. full campaign in the background
mkdir -p campaign_results
nohup python3 campaign/run_campaign.py --out campaign_results/main \
      > campaign_results/main.nohup.log 2>&1 &
```

Defaults: `--jobs` = available CPUs − validator workers (respects SLURM/cgroup
CPU pinning), `--validators` = jobs/16. On a 64-core node that is 60 solver
processes + 4 validators. Set `--jobs` explicitly to leave cores for others.

* **Errors never stop the run.** Each failed run (timeout, crash, invalid
  output, validator disagreement, runner exception) is recorded and the next run
  starts. Only errors (plus a START/FINISHED line) go to the nohup log and to
  `campaign_results/main/errors.log`.
* **Resume**: run the same command again; every run already in
  `results.jsonl` is skipped. Add `--retry-failed` to redo failed runs.
* **Progress**: `cat campaign_results/main/progress.json` (done, ETA).
* **Stop**: `kill <pid>` (SIGTERM) stops scheduling, kills running solvers, and
  leaves unfinished runs to be redone on resume.

## Outputs (`campaign_results/main/`)

* `results.jsonl` — one JSON record per run: solver, family, instance, seed,
  `status` (`ok`, `timeout`, `crash`, `invalid`, `validator_disagree`,
  `runner_error`), `score` (upstream referee), `strict_score`, `wall_s`
  (wall clock incl. reading and writing), `cpu_s`, `max_rss_kib`, exit code,
  solution SHA-256, host, timestamps; stderr tail for failed runs.
* `solutions/<solver>/<family>/<instance>__seed<k>.txt.gz` — every output.
  (`--keep-solutions none` deletes valid outputs after scoring.)
* `manifests.jsonl` — per invocation: host, git commit, compiler/node versions,
  binary + validator hashes, arguments.
* `errors.log`, `progress.json`.

Summaries: `python3 campaign/summarize.py campaign_results/main` writes
`runs.csv`, `solver_instance.csv`, `status_counts.csv` and prints the total
recorded wall-clock time.

## Validation

Every output with exit code 0 is checked twice in the same Node worker:

1. **Upstream referee** (`submission.js`, unmodified) → `valid`, `score`.
2. **Strict re-check** using skill levels *at project start* → `strict_valid`, `strict_score`.

`status = ok` requires both to accept and the scores to be equal. The upstream
referee is lenient in one case (a teammate who levelled up earlier in the same
project is accepted as a mentor; see `validator/upstream/VENDORED.md`). Such
outputs get `validator_disagree` rather than being silently counted.

One manual check:

```bash
node campaign/validator/batch_validate.js google_instances/d_dense_schedule.txt solution.txt
```

Upstream usage still works: `cd campaign/validator/upstream && node index.js data/<x>.in.txt`.

## Note on the older `experiments/` runner

`experiments/run_mtp_campaign.py` passes `--seed/--time-limit` flags and points
at `solvers/Nanored_hashCode2022/`, which is not in the repository. It does not
match the solver contract above; use `campaign/` for this campaign.
