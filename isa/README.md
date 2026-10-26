# First Instance Space Analysis of the MTP solver campaign

This runbook prepares and executes the first descriptive Instance Space Analysis
(ISA) of `experiment_results/rizana_trial_01`. It uses the MATLAB
[InstanceSpace toolkit](https://github.com/andremun/InstanceSpace) at commit
`5f8cb25682202a8243bd2a2481ab8feb40c8c4ce`. Record any change to that
commit as a new analysis. The toolkit's current API is the `InstanceSpace`
class; older instructions using `buildIS(rootdir)` or the old `example.m` do
not apply to this pinned version.

## Study population and provenance

The executed campaign has **611 instances** and **61,100 recorded runs**:
60,521 successes, 579 timeouts, and no invalid solutions or other recorded
failures. There were ten executions of each of ten solver labels on every
instance. The configured solver budget was 180 seconds with 15 seconds of
external timeout grace; observed successful wall times can exceed 180 seconds.
The family counts are 5, 100, 98, 100, 108, 100, and 100 for Types
1–7 respectively. These are the *executed* counts, not the original target
counts. Type 1 contains the five nontrivial Google Hash Code instances; the
toy example was not run. The authoritative inputs are
[`campaign_manifest.json`](../experiment_results/rizana_trial_01/campaign_manifest.json),
[`campaign_summary.json`](../experiment_results/rizana_trial_01/campaign_summary.json),
[`solver_instance_summary.csv`](../experiment_results/rizana_trial_01/solver_instance_summary.csv),
and [`solver_manifest.json`](../experiment_results/rizana_trial_01/solver_manifest.json).

The Type 3 campaign contains 98 files, and final Type 3 metadata/quality-gate
artifacts are absent. Treat this as a provenance limitation. Audit those 98
inputs using structural criteria and the generator's intended anchor/radius
rules before describing them as a validated neighborhood family. Do not change
the executed manifest or decide exclusions from solver scores. Report both the
all-campaign result and a declared sensitivity analysis if a structural audit
later justifies exclusions.

The ten labels include three pairs with identical source hashes:
`nanored_original_c`/`nanored_refactored_d`,
`nanored_original_e`/`nanored_refactored_e`, and
`nanored_original_f`/`nanored_refactored_f`. The primary portfolio takes the
first solver in the manifest for each unique source hash, yielding seven
source-distinct algorithms. This selection uses source identity and manifest
order, never outcomes. Preserve all ten labels in a separate audit table and
report the ten-label analysis only as sensitivity. A shared source hash is
evidence of duplicated code, although compiled binaries and execution
settings should also be recorded.

## Primary outcome and good-performance definition

Let $s_{iar}$ be the verified score of execution $r$ on instance $i$
with algorithm $a$, where a timeout or failed execution contributes zero.
The campaign-success-policy mean score per planned attempt is

$$
U_{ia}=\frac{1}{10}\sum_{r=1}^{10}s_{iar}
     =\bar{s}^{\mathrm{success}}_{ia}\frac{n^{\mathrm{success}}_{ia}}{10}.
$$

The second expression uses the committed solver-instance summary and is valid
because `mean_score` averages **successful** runs only. Its direct use would
omit the cost of timeouts. Report timeout probability separately. This zero
utility convention means no usable result was available from that execution
by the campaign limit; it is a decision-utility definition, not an imputed
Hash Code score. Preserve the campaign's actual runtime limit and success
rules in the provenance record. Do not call the mean of the ten best scores an
expected single-run result.
Because successful wall times sometimes exceed 180 seconds, these summaries
cannot reconstruct a strict 180-second deployment outcome. That would need
per-run solution histories or a rerun with a hard cutoff.

For comparability across differently sized instances, define
$U_i^*=\max_a U_{ia}$ over the **seven-algorithm primary portfolio** and
$R_{ia}=U_{ia}/U_i^*$. The exporter must reject or explicitly report any
instance with $U_i^*\leq0$ before this division. `metadata.csv` stores
$R_{ia}$ in the `algo_*` columns; the absolute expected scores and timeout
counts remain in the audit outputs. With the toolkit set to maximize relative
performance, an algorithm is *good* at the prespecified 5% threshold when
$R_{ia}\geq0.95$. The 1% and 10% thresholds are sensitivity analyses; label
their results separately. A tie gives every tied algorithm its rightful good
label. Never infer a unique winner from an arbitrary tie break in a plotting
or prediction routine. For a selector, evaluate achieved $R$, regret
$1-R$, and the frequency of selecting a good algorithm.

Best-of-ten is a secondary upper-envelope analysis for a ten-start deployment
policy. Its extra computational budget must be stated explicitly. A useful
additional sensitivity is the conditional-on-success score together with the
timeout rate; it must not replace the primary utility.

## Structural features and selection

Each `feature_*` column must be calculated from the MTP input file before
solver outcomes are inspected. Solver scores, runtimes, timeout indicators,
family labels, file names, generator parameters, and source IDs are **not**
features. The `source` column in `metadata.csv` records family provenance for
plotting and subgroup reports. Keep the complete feature matrix and an exact
description of every feature in the export artifacts, even when the projection
uses only a subset.

The existing extractor provides structural candidates across size, supply
and demand, skill levels, mentorship, time, and scores. The exporter starts
with a declared shortlist of fifteen candidates and uses structural-only
variance and rank-correlation filtering before SIFTED. Before scientific
interpretation, audit the following known issues in
[`features.py`](../benchmark_generation/core/features.py):
`mentorship_fillability` and `mentor_availability` are identical in the
current implementation; the within-project mentorship measure is a proxy;
and the mentorship proxy misses level-0 mentees for level-1 roles.
`direct_fillability` measures the existence of an individual with enough
skill for a role, not whether an entire project has a simultaneous feasible
team. The legacy `infeasible_deadline_share` is the share of projects with
duration greater than best-before (unavoidable lateness), not zero-score
infeasibility. The primary shortlist excludes that deadline variable and the
three flawed mentorship proxies. Retain the feature definition and version in
the audit. Remove exact duplicate columns from projection candidates and do
not interpret a proxy as an assignment-feasibility
rate. A corrected extractor warrants a new ISA export; it does not alter the
instances or campaign outcomes.
Consequently, this first map should not support claims about exact
within-project mentorship feasibility; add a validated matching-aware
feature in a later, versioned analysis before making such claims.

For the first descriptive map, apply the following selection discipline:

1. Apply the exporter's declared transformations, including `log1p` of
   contributor and project counts, contention, schedule pressure, and score
   density. Remove invalid, non-finite, constant, and nearly redundant
   columns using the recorded rules. Record every inclusion and exclusion.
2. Inspect feature distributions by family and a rank-correlation matrix.
   Document strongly correlated candidates; preserve all six conceptual
   groups in the audit even if some groups contribute no projected variable.
3. Run the pinned toolkit's SIFTED selection on the eligible columns with
   `K=10` and its saved random seed. The upstream documentation recommends no
   more than ten PILOT inputs. Record selected variables and selection
   stability under fixed-seed repeats or bootstrap resampling; do not choose
   the most visually attractive map after trying many settings.
4. Use the toolkit's numerical two-dimensional PILOT projection, then inspect
   projection diagnostics and selected-feature loadings. A two-dimensional
   map is a summary of measured relationships, not a certificate that all
   relevant MTP structure has been preserved.

The descriptive SIFTED/PILOT fit uses the entire 611-instance dataset. Its
feature selection is supervised by the same campaign outcomes, so its
in-sample footprint and selection accuracy are **descriptive**. Interpretation
should mention feature group coverage, family coverage, and whether the five
Google anchors lie near generated families. Apparent footprint area alone is
insufficient evidence of broad algorithmic power: also report the count and
family composition of observed good instances in each footprint, footprint
purity, and uncertainty or instability under resampling.

## Run the first ISA

The prepared snapshot is already under `isa/outputs/rizana_trial_01`.
Inspect its `audit.json` first. To regenerate it intentionally from the
campaign, run from the repository root with the benchmark Python environment
(`--overwrite` is required because the snapshot already exists):

```powershell
python -B -m isa.prepare_isa --campaign experiment_results/rizana_trial_01 --output isa/outputs/rizana_trial_01 --overwrite
python -B -m isa.verify_legacy_features --input isa/outputs/rizana_trial_01
python -B -m isa.summarize_export --input isa/outputs/rizana_trial_01 --overwrite
```

The input export includes `metadata.csv` for the seven-solver MATLAB analysis,
`metadata_all_solvers.csv` for the ten-label sensitivity,
`metadata_best_of_ten.csv` for the ten-start sensitivity, `all_features.csv`,
`all_solvers.csv`, `instance_audit.csv`, `feature_selection.json`, and
`audit.json`. Check the export audit before continuing. At minimum,
verify 611 unique instance IDs; seven source-distinct `algo_*` columns;
finite numeric feature and algorithm values; the executed family counts;
6,110 solver-instance summaries; and exactly the 579 campaign timeouts
represented in the audit. Inspect the eligible feature list and any warnings.
Stop and fix the export if a join, hash check, or count fails. Preserve the
input CSV and JSON files and their SHA-256 hashes with the chapter results.
The legacy cross-check writes `legacy_feature_agreement.json` and fails on
disagreement with the available frozen generation metadata. Type 3 and the
Type 5 full-project extension have no comparable rows in those legacy tables.
The final command writes `preflight_report.json`: tie-aware good-instance
counts by family, per-algorithm normalized regret, and the descriptive gap
between the best fixed solver and the unattainable per-instance oracle.

Install MATLAB R2025a or newer and the toolboxes required by the pinned
[upstream README](https://github.com/andremun/InstanceSpace/blob/5f8cb25682202a8243bd2a2481ab8feb40c8c4ce/README.md):
Global Optimization, Parallel Computing, Optimization, Statistics and Machine
Learning, and Financial. The current toolkit is licensed under PolyForm
Noncommercial 1.0.0; check suitability before distribution or commercial
reuse. Obtain the exact toolkit revision:

```powershell
$isaToolkitDir = 'C:\path\to\InstanceSpace'
git clone https://github.com/andremun/InstanceSpace.git $isaToolkitDir
git -C $isaToolkitDir checkout 5f8cb25682202a8243bd2a2481ab8feb40c8c4ce
git -C $isaToolkitDir rev-parse HEAD
```

Replace `C:\path\to\InstanceSpace` with a path you control. In MATLAB, invoke the
repository wrapper from an **empty new output directory**; replace the
toolkit path with the cloned location:

```matlab
repo = 'C:\Users\shefk\source\repos\Python\mentorship-teamwork-dataset';
toolkit = '<absolute path to InstanceSpace>';
addpath(fullfile(repo, 'isa'));
input_dir = fullfile(repo, 'isa', 'outputs', 'rizana_trial_01');
output_dir = fullfile(repo, 'isa', 'outputs', 'rizana_trial_01_matlab_05');
run_matlab_isa(toolkit, input_dir, output_dir, 2026);
```

The wrapper prepares the current toolkit's `options.json` in the output
directory, records its options and seed, and calls the current class API
(`InstanceSpace(output_dir).build()`). For the primary analysis verify in the
saved options that performance is maximized, good performance is relative to
the best (`epsilon=0.05`), SIFTED uses at most ten features, PILOT has two
dimensions, and seed 2026 was applied. Save MATLAB/toolbox versions, toolkit
commit, `run_info.json`, output logs, projection tables, footprints, figures,
and the complete model. The wrapper deliberately refuses a nonempty output directory so that
an earlier run is not silently overwritten.

The primary wrapper also sets `opts.bound.flag=false`, a declared deviation
from the toolkit's default IQR clipping. In this campaign, 429/611 instances
have zero `unmet_skill_share`; under the default median ± 5×IQR rule, roughly
115 nonzero values would be clipped despite a raw maximum of 0.579. The
no-bounding primary run preserves that structural variation. Inspect PRELIM's
normalized feature distributions for other outlier effects, and run a
predeclared bound-on sensitivity in a new directory:

```matlab
run_matlab_isa(toolkit, input_dir, fullfile(repo, 'isa', 'outputs', 'rizana_trial_01_matlab_bound_on'), 2026, 0.05, true);
```

For 1% and 10% threshold sensitivity, reuse the same exported `metadata.csv`
and invoke the wrapper with its fifth `epsilon` argument and distinct empty
output directories:

```matlab
run_matlab_isa(toolkit, input_dir, fullfile(repo, 'isa', 'outputs', 'rizana_trial_01_matlab_01'), 2026, 0.01);
run_matlab_isa(toolkit, input_dir, fullfile(repo, 'isa', 'outputs', 'rizana_trial_01_matlab_10'), 2026, 0.10);
```

Preserve the exact generated `options.json` for each run. Repeating the
projection after changing the threshold can change supervised feature
selection; compare both the selected variables and the footprints. Threshold
counts in the Python export audit refer to its primary 5% definition; use
the corresponding MATLAB run for other thresholds.

For projection and feature-selection stability, repeat the 5% analysis with
a predeclared seed set rather than choosing a visually appealing run:

```matlab
for seed = 2026:2035
    trial_dir = fullfile(repo, 'isa', 'outputs', sprintf('rizana_trial_01_stability_%d', seed));
    run_matlab_isa(toolkit, input_dir, trial_dir, seed, 0.05);
end
```

Each directory must be new. Compare selected-feature frequencies, point-map
alignment after rotation/reflection, and footprint evidence counts; do not
average coordinate axes from unaligned projections.

To assess the duplicate-label and ten-start choices, make separate input
folders whose `metadata.csv` is a copy of `metadata_all_solvers.csv` or
`metadata_best_of_ten.csv` respectively. Run each through the same wrapper
with a fresh output directory and the same seed and 5% threshold. Keep each
alternative's performance interpretation visible in its figures and tables.

## What to report from this first run

Archive a compact, reproducible set of outputs: (i) data inventory and
campaign provenance, (ii) outcome and timeout distributions by algorithm and
family, (iii) feature definitions, exclusions, correlations, and selected
variables, (iv) PILOT fit diagnostics and an instance map colored by family,
(v) TRACE footprints with measured good labels, their evidence counts/purity,
(vi) tied-good frequency and easy/hard instance prevalence, and (vii) the 1%,
5%, and 10% threshold comparison. Distinguish observations from fitted
footprint predictions in captions. State that the 611 campaign inputs were
frozen in its benchmark manifest before this ISA used solver outcomes, while
retaining the Type 3 finalization caveat above.

The 611 rows are a designed benchmark, not a random sample from all MTP
instances. Family sizes differ, and the original Google family has only five
rows. Give both campaign-weighted summaries and family-stratified summaries;
avoid population-wide probability claims from raw footprint area. Highlight
the limited ability of five anchors to support broad claims about original
Hash Code instances.

## Before predictive or algorithm-selection claims

A full-data SIFTED/PILOT map cannot estimate out-of-sample selector accuracy.
For a book-chapter claim about choosing a solver for a new instance, create a
separate evaluation with entire instance groups held out. In every training
fold, refit missing-value handling, transformations, bounds, scaling, SIFTED,
PILOT, model tuning, and footprint/classifier estimation **without the test
instances**. Project test features using only the training fit. The unit of
resampling is the MTP instance, never one of its ten runs or a solver-instance
row. Group Type 2 instances by pair/region and Type 6 instances by external
source if those mappings are available. Use a Type 3 anchor group only after
its missing provenance has been independently reconstructed and checked. Also
report leave-one-family-out stress tests to assess transfer to a new family.
The tiny Type 1 family requires cautious interpretation.

Assess selected-solver expected utility, normalized regret, and good-selection
rate against the best fixed seven-solver choice selected **within each training
fold**, plus an oracle upper bound clearly identified as unattainable. Report
per-family results and uncertainty intervals from instance-level or
provenance-group bootstrap resampling. If individual run records are restored
from the supercomputer, use them to quantify run-to-run score and timeout
uncertainty. The committed summaries support only point estimates of
per-instance expected utility and cannot reconstruct arbitrary run-level
bootstrap samples.

Do not interpret the toolkit's internal cross-validation or random tie break
as this grouped, nested external evaluation. Until that evaluation exists,
describe the outputs as an exploratory first ISA of the fixed campaign.

## Sources to cite

- Smith-Miles and Muñoz, [*Instance Space Analysis for Algorithm Testing:
  Methodology and Software Tools*](https://doi.org/10.1145/3572895), ACM
  Computing Surveys 55(12), 2023.
- Muñoz and Smith-Miles,
  [*Instance Space Analysis: A toolkit for the assessment of algorithmic power*](https://doi.org/10.5281/zenodo.4484107),
  Zenodo, 2020, and the
  [pinned source revision](https://github.com/andremun/InstanceSpace/tree/5f8cb25682202a8243bd2a2481ab8feb40c8c4ce).
