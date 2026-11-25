# Mentorship and Teamwork Problem: benchmark, generators, solvers and results

This repository accompanies the book chapter *Instance Space Analysis for the Mentorship and Teamwork Problem* by Uran Lajçi, Shefket Bylygbashi and Kadri Sylejmani, written for the volume *Instance Space Analysis: Theory, Applications and Advances* (eds. Kate Smith-Miles and Mario Andrés Muñoz, Springer Nature).

The Mentorship and Teamwork Problem (MTP) is the problem of the Google Hash Code 2022 qualification round. The rules and the file formats are in [problem_definition.md](problem_definition.md).

| Path | Content |
|---|---|
| [google_instances/](google_instances/), [generated_instances/](generated_instances/) | The benchmark: 131 instances in seven families |
| [benchmark_instances.csv](benchmark_instances.csv) | One row per instance: family, file, size, SHA-256 and source |
| [benchmark_generation/](benchmark_generation/) | The instance generators (Types 2 to 7) and their tests |
| [benchmark_metadata/](benchmark_metadata/) | Generator records: parameters and features of each generated instance, data sources, calibration |
| [solvers/](solvers/) | The ten C++ solvers |
| [campaign/](campaign/) | The campaign runner and the validator |
| [results/](results/) | One record per run, summaries, and solution files |
| [isa/](isa/) | The 58 instance features, the MATILDA input, settings and outputs |
| [METHODOLOGY.md](METHODOLOGY.md) | How the generators work |

## Benchmark

| Family | Instances | Files | How the instances were obtained |
|---|---|---|---|
| Type 1 (Google) | 5 | `google_instances/b_…` to `f_…` | The five non-trivial Hash Code 2022 instances, unchanged. The toy example `a` is not part of the benchmark. |
| Type 2 (between anchors) | 12 | `generated_instances/type2/` | Generated towards targets on the segment between two Google instances |
| Type 3 (neighbourhood) | 15 | `generated_instances/type3/` | Generated around a Google instance, keeping its numbers of contributors, projects and skills: 5 around `c`, 10 around `f` |
| Type 4 (broad random) | 22 | `generated_instances/type4/` | All generator parameters drawn at random from wide ranges |
| Type 5 (empirically grounded) | 31 | `generated_instances/type5/` | Parameterised from the measured activity of public GitHub repositories |
| Type 6 (converted from MSLIB) | 40 | `generated_instances/type6/` | Converted from MSLIB files, 10 from each of sets 1 to 4 |
| Type 7 (remaining gaps) | 6 | `generated_instances/type7/` | Generated last, in regions that Types 1 to 6 leave empty |
| **All** | **131** | | |

Every file follows the Hash Code input format. Generated instances use generic names (`c00000`, `s0000`); the Google files keep their original names. Twenty instances in Types 2 and 3 have project scores above the competition's limit of 10⁵, and three of them also have best-before days above it. Type 6 file names end in `__MSLIB<k>__src<id>`, the MSLIB set and source file. `type5_099__GHFULL__…` is built from a repository's complete history, the other Type 5 instances from a six-month window.

## Solvers

The portfolio has ten single-threaded C++20 solvers, in five pairs. Each pair was written for one Google instance.

| Solver | Source | Runs per instance |
|---|---|---|
| `original_b` … `original_f` | `solvers/original/{b,c,d,e,f}.cpp` | `b`, `d`: 10 (seeds 1 to 10); `c`, `e`, `f`: 1 (deterministic) |
| `evolved_b` … `evolved_f` | `solvers/evolved/{b,c,d,e,f}.cpp` | `b`, `d`, `e`: 10 (seeds 1 to 10); `c`, `f`: 1 (deterministic) |

The original solvers are the programs of Yoann Coudert-Osmont's [Hash Code 2022 solution](https://github.com/Nanored4498/HashCode2022) (MIT licence), with three engineering changes that leave the algorithms untouched: shared seed handling, a wall-clock deadline of 590 s at which the best solution found so far is written, and a guard on the capacity of the skill array. The evolved solvers were produced from them with [MicroEvolve](https://github.com/uran-lajci/MicroEvolve), each tuned on its own Google instance. The shared runtime is `solvers/mtp_runtime.hpp`.

```bash
bash campaign/build_solvers.sh                 # g++ -O2 -std=c++20, writes campaign/bin/
campaign/bin/evolved_f 1 < google_instances/f_find_great_mentors.txt > solution.txt
```

## Results

All ten solvers were run on all 131 instances: 131 × (5 × 10 + 5 × 1) = 7,205 runs, each with a limit of 10 minutes (search stops at 590 s). Every output was checked by the validator in `campaign/validator/`, and all 7,205 runs are valid.

| File | Content |
|---|---|
| `results/results.jsonl` | One JSON record per run: solver, instance, seed, status, score, wall and CPU time, memory, server, SHA-256 of the solution |
| `results/runs.csv` | The same records as a table |
| `results/solver_instance.csv` | Per solver and instance: number of runs, best and mean score, standard deviation, wall time |
| `results/status_counts.csv` | Runs per solver and status |
| `results/manifests/<SERVER>.jsonl` | Per server: compiler, flags, git commit, hashes of sources, binaries and validator |
| `results/solutions/<solver>/<family>/<instance>__seed<k>.txt.gz` | The submitted solution of a run |

The runs were made on six servers. Solution files are included for the 2,413 runs made on LYDRA and MIRGETA; for the runs on the other four servers the record holds the score and the SHA-256 of the solution.

See [campaign/README.md](campaign/README.md) for how to rerun the campaign and for the validator.

## Instance Space Analysis

[isa/](isa/) holds everything that went into and came out of MATILDA: the 58 features ([isa/FEATURES.md](isa/FEATURES.md)), the metadata file with features and relative performance, the settings, and the output tables. See [isa/README.md](isa/README.md).

```bash
python3 isa/extract_features.py     # instances -> isa/features.csv (58 features)
python3 isa/build_metadata.py       # + results/results.jsonl -> isa/metadata.csv
```

Both scripts use only the Python standard library and reproduce `isa/metadata.csv` exactly.

## Requirements

- Generators: Python 3.10 or later (standard library); see [benchmark_generation/README.md](benchmark_generation/README.md). `pytest benchmark_generation/tests` runs the tests.
- Campaign: Python 3.9 or later (standard library), g++ with C++20, Node 16 or later for the validator.
- ISA scripts: Python 3 (standard library).

## Licence

The code, the generated instances and the results are released under the MIT licence ([LICENSE](LICENSE)). The solvers derive from Yoann Coudert-Osmont's MIT-licensed solution and keep his copyright notice. The validator in `campaign/validator/upstream/` is by Kristóf Holecz and has its own licence (ISC). The five files in `google_instances/` are the original Hash Code 2022 inputs published by Google and are not covered by this licence.
