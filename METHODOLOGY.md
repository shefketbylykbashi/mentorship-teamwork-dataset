# MTP Benchmark Generation Methodology

This document describes the reproducible generation pipeline for Benchmark
Types 2, 3 and 4 for the Mentorship and Teamwork Problem (MTP), originally
introduced in the Google Hash Code 2022 qualification round.

## Immutable Anchors

The original files in `google_instances/` are read-only anchors. The toy file
`a_an_example.txt` is excluded. The five non-trivial anchors are parsed,
validated syntactically, measured with the feature extractor, normalized, and
written to `benchmark_metadata/google_anchor_features.csv` and JSONL.

## Forward Generator

Every synthetic candidate is generated forward:

```text
parameters -> contributors/projects/roles -> complete MTP instance
           -> features -> family objective evaluation
```

No instance is constructed by inverting a feature vector. Target feature vectors
for Types 2 and 3 are search objectives only.

For each generated instance the pipeline:

1. creates `n_skills` skills;
2. samples contributors;
3. samples contributor skill sets and integer levels;
4. samples projects;
5. samples duration, score and best-before values;
6. samples project roles;
7. selects role skills;
8. calibrates required levels against contributor supply;
9. validates initial viability;
10. writes canonical Hash Code-compatible text only after acceptance.

Contributor levels are redrawn from a truncated normal distribution until they
fall in `[1, level_max]`, avoiding endpoint probability spikes caused by
clipping. Skill frequencies use Zipf-like weights
`w_i proportional to 1 / rank_i^skill_skew`; `skill_skew = 0` is uniform.

Project score is coupled to effort:

```text
effort_j = duration_j * roles_j
score_j = round(score_base * effort_j * U(1, score_perturbation))
```

Deadlines are relative to duration:

```text
best_before_j = round(duration_j * U(tightness_min, tightness_max)).
```

Role demand interpolates continuously between contributor supply and an
unaligned distribution:

```text
P(role skill=s) =
  role_skill_alignment * P_supply(s)
  + (1-role_skill_alignment) * (0.5 * P_uniform(s) + 0.5 * P_zipf(s)).
```

## Required Skill Levels

For each role independently, the generator inspects contributor levels for the
required skill. If supply exists, the reference level is the configured
`supply_quantile`; otherwise it falls back to the rounded contributor level
mean. The role difficulty category is drawn per role from:

```text
(p_over, p_match, p_mentor, p_future), sum = 1.
```

The category maps to required levels as follows:

```text
p_over   -> U(1, max(1, reference-1))
p_match  -> reference
p_mentor -> reference + 1
p_future -> reference + U(future_offset_min, future_offset_max)
```

Role requirements are capped by the official role-level limit of 100, while
contributor levels are capped at 10.

## Mentorship Rule

The implementation preserves the original Google rule. A contributor assigned
to a role is feasible if:

```text
level(contributor, skill) >= required_level
```

or:

```text
level(contributor, skill) = required_level - 1
```

and another contributor assigned to the same project has that skill at
`required_level` or higher. A mentor must be assigned to the same project, may
mentor multiple roles, and may simultaneously be mentored in another skill.

## Viability

The default minimum initial viability is:

```text
fillable_fraction = 0.30.
```

Initial role fillability is the share of roles that are directly fillable or
mentor-fillable from the initial contributor pool. Because the static role-level
mentorship proxy necessarily implies the existence of a qualified contributor
for that role's skill, the implementation uses the union of the two conditions
rather than summing overlapping shares. Candidates below the threshold are
rejected and regenerated until `max_generation_attempts` is reached. A
conservative positive-score guard also requires at least one project to have a
direct feasible assignment capable of positive score.

## Features

The plan text refers to a 31-feature candidate set, but the enumerated required
features across the six groups total 37. To avoid silently changing the science,
the implementation computes every enumerated feature.

Size and scale:

```text
n_contributors, n_projects, n_skills, total_roles,
roles_per_project_mean, roles_per_project_sd,
skills_per_contributor_mean, skills_per_contributor_sd.
```

Supply and demand:

```text
contention = total_roles / n_contributors
workload_contention = sum(duration_j * roles_j) / n_contributors
demand_concentration = 1 - H(role skill distribution) / log(k)
supply_concentration = 1 - H(contributor skill distribution) / log(k)
demand_supply_alignment = cosine(demand_counts, supply_counts)
unmet_skill_share = roles requiring skills absent from all contributors / total_roles.
```

Skill levels:

```text
contributor_level_mean, contributor_level_sd,
required_level_mean, required_level_sd,
level_gap_mean, level_gap_sd.
```

Level gaps are computed on compatible skills: for each demanded skill, the mean
required level is compared with the mean supplied level for that same skill.

Fillability and mentorship:

```text
direct_fillability = share of roles with any contributor at required level
mentorship_fillability = share with one-below contributor and a distinct qualified mentor
learning_reachable_demand = share not direct but within two levels of current max supply
mentor_availability = share with a one-below contributor and qualified mentor
within_project_mentorship_feasibility = share of projects with potential within-project mentorship.
```

Time and deadlines:

```text
duration_mean, duration_sd,
deadline_tightness_mean = mean(best_before / duration),
deadline_tightness_sd,
infeasible_deadline_share = share with duration > best_before,
schedule_pressure = total effort / sum(best_before).
```

Scores:

```text
score_mean, score_sd,
score_concentration = 1 - normalized entropy of project scores,
score_density = mean(score / effort),
score_effort_correlation = Pearson(score, effort),
penalty_exposure = mean(max(0, duration-best_before) / duration).
```

Undefined correlations, NaN and infinities are replaced by zero.

## Feature Scaling and Distance

Feature distance is never computed on raw heterogeneous features. The scaler:

1. applies `log1p` to raw scale variables and selected large-magnitude measures;
2. standardizes each feature using the fitted mean and standard deviation;
3. replaces constant-feature standard deviations with 1;
4. uses group-balanced Euclidean distance.

With feature groups `G` and group weights normalized to sum to one:

```text
d(x,y) = sqrt( sum_g sum_{f in g} (W_g / |g|) * (x_f - y_f)^2 ).
```

The default gives each of the six conceptual groups equal aggregate weight.

## Type 4: Broad Random Family

Type 4 is origin-driven and does not target Google anchors. Parameters are drawn
from the configured broad ranges. The scale bands come from the local dataset
plan:

```text
small:  contributors/projects 5-50
medium: contributors/projects 50-500
large:  contributors/projects 500-10000
```

For 100 instances the allocation is `34/33/33` over small/medium/large. Counts
within a band are sampled approximately logarithmically. The ratio
`n_projects / n_contributors` is sampled from a configurable log-uniform range
to create low-, medium- and high-contention cases.

The four difficulty probabilities are sampled jointly from a symmetric
Dirichlet distribution with alpha `(1,1,1,1)`. This is an implementation choice,
not a claim from the original benchmark.

## Type 2: Gap-Filling Between Anchors

Type 2 uses measured anchor features, not inferred original generator
parameters, to define target locations. The pipeline:

```text
load anchors -> extract features -> normalize -> compute pairwise distances
-> allocate targets to anchor pairs -> adaptive parameter search
-> forward-generate candidates -> measure -> accept diverse valid candidates.
```

For anchors `a` and `b`, a target is:

```text
t = (1-alpha) * z_a + alpha * z_b, alpha in [0.2, 0.8].
```

Targets are stratified along each segment and pair allocation is proportional to
anchor-pair distance, ensuring all five anchors participate and wider gaps
receive more targets. The adaptive search objective is:

```text
J = d(candidate, target)
  + lambda_novelty * max(0, minimum_distance - nearest_existing_distance)
  + duplicate_penalty * I[structural duplicate].
```

The optimizer is a transparent derivative-free elite perturbation strategy:
sample an initial population, generate candidates forward, retain elites,
perturb parameters with decaying scale, and restart a configurable fraction.
For computational efficiency, each Type 2 target searches within the bounds
defined by its own anchor pair rather than the union of all five anchors. The
initial population is seeded with an interpolated anchor-estimated parameter
vector and perturbations around it. This interpolation is only a search prior:
the accepted instance is still generated forward and selected by its measured
feature distance to the target plus novelty and duplicate penalties.

## Type 3: Neighborhoods Around Anchors

Type 3 constructs local target clouds around each Google anchor. The default
allocation is balanced over the five anchors. Target directions are random
normalized directions in the same group-balanced feature metric. Final Type 3
generation requires an empirical calibration file,
`benchmark_metadata/type3_calibrated_radii.json`, produced by the reachability
diagnostic. The generator fails rather than silently falling back to nominal
bands if that file is missing, malformed, or incomplete. Nominal uncalibrated
bands are available only through an explicit development option.

Earlier nominal bands were:

```text
small [0.03, 0.08], medium [0.08, 0.18],
outer [0.18, 0.35], wide [0.35, 0.55].
```

These are not treated as final methodology; calibrated anchor-specific bands
replace them when the forward generator's empirical local resolution shows that
the nominal values are not reachable.

The search objective includes target distance, novelty, duplicate penalty, and
a radius penalty if the measured candidate is outside its configured radius
band. This prevents collapse onto the anchor and uncontrolled drift. As with
Type 2, the initial search population is seeded near the anchor-estimated
parameter vector only as a computational prior; all candidates remain forward
generated and feature-measured before acceptance.

The Type 3 manifest is written only after an executable quality gate passes:
the accepted count is exact, structural hashes are unique, all structural and
radius validations pass, all five anchors are represented, every calibrated band
expected for every anchor is represented, and required metadata fields are
present.

## Type 5: Empirically Grounded Synthetic Software-Project Family

Type 5 represents empirically grounded synthetic MTP instances. The instances
are not literal copies of GitHub repositories. Public repository activity is
measured first, converted into an empirical source configuration, and then used
to parameterize the same forward MTP generator used by the other synthetic
families:

```text
public GitHub activity -> aggregate source measurements
-> empirical source configuration -> shared forward generator
-> validation and Type 5 realism gates -> benchmark instance.
```

Type 5 uses 25 purposively selected mature public software repositories and
four fixed half-open six-month UTC observation windows:

```text
W1 [2024-08-17T00:00:00Z, 2025-02-17T00:00:00Z)
W2 [2025-02-17T00:00:00Z, 2025-08-17T00:00:00Z)
W3 [2025-08-17T00:00:00Z, 2026-02-17T00:00:00Z)
W4 [2026-02-17T00:00:00Z, 2026-08-17T00:00:00Z)
```

These windows are an implementation design choice, not part of the original
Hash Code MTP definition. The deterministic slot mapping is repository-major
and window-minor, so repository 1 W1 maps to `type5_001`, repository 1 W2 to
`type5_002`, and so on through `type5_100`.

### Empirically Observed

The collector uses authenticated public GitHub REST API access and caches
successful responses under `type5_source_cache/`. The cache is excluded from
version control and may temporarily contain public GitHub usernames returned by
the API. Public benchmark metadata and final `.txt` instances do not contain
real usernames.

For each repository the collector records repository metadata, default-branch
snapshot SHA, language byte counts, releases, milestones, and window-specific
merged pull requests. Merged pull requests are treated as observable work-unit
proxies. For each six-month window all merged PRs are counted; if there are more
than 60, a deterministic time-stratified sample of 60 PRs is selected for
file-level analysis. The sample targets roughly 10 PRs per month and avoids a
single contributor exceeding 20 percent of the detailed sample unless the
source distribution itself makes that impossible.

Changed file paths, additions, deletions, and status are collected for sampled
PRs. File contents are never stored. Obvious bots are filtered with documented
regular expressions such as `[bot]`, dependabot, renovate, github-actions, and
release automation identities.

### Derived Proxies

Skill tokens are inferred from changed technical file paths using a deterministic
language plus subsystem taxonomy. Examples include `Python::db`,
`TypeScript::compiler`, and `Go::scheduler`. The final MTP instance anonymizes
these semantic skills to `s0000`, `s0001`, and so on; semantic tokens are kept
only in research metadata. Vendor, build, generated, lock, binary, minified,
and documentation-only paths are excluded or prevented from dominating.

Rare subsystem skills are merged into their parent language skill unless they
appear in at least two distinct human contributors or at least three sampled
PRs. A source window with fewer than five retained technical skills fails the
source-quality gate rather than receiving invented skills.

For contributor `c` and skill `s`, activity is measured as:

```text
activity_cs =
    0.60 * log1p(lines_changed_cs)
  + 0.25 * log1p(files_touched_cs)
  + 0.15 * log1p(prs_touching_skill_cs).
```

For each retained skill, contributors are ranked by activity and converted to
MTP levels by:

```text
level = 1 + floor(9 * percentile_rank), clipped to [1, 10].
```

These levels are relative empirical activity-based skill-level proxies, not
claims about true human competence.

Contributor specialization is derived by normalizing contributor-skill activity
vectors and clustering them with deterministic k-means for `k = 3..8`. The
chosen `k` maximizes silhouette score when the source data supports meaningful
profiles. Profile centroids and profile sizes are passed to the shared forward
generator through an optional empirical source-profile hook.

PR cycle time is the duration proxy:

```text
duration = max(1, ceil(merged_at - created_at in days)).
```

Per-window duration bounds use robust P10/P90 values. Deadline tightness uses a
milestone due date when available, otherwise the nearest subsequent stable
release within the configured association window, otherwise the corpus-level
P10/P90 fallback. This is documented as a deadline proxy, never as an observed
true deadline.

### Synthetic Modeling Decisions

GitHub does not provide cross-project business value. Type 5 therefore keeps
MTP scores synthetic and grounds only their scale and variability weakly in
engineering effort:

```text
effort_i = log1p(additions_i + deletions_i) * sqrt(max(1, role_count_i)).
```

A corpus-wide normalization constant `K` is chosen so the median normalized
score density is approximately 2.0. Per-window `score_base` is clipped to
`[0.5, 5.0]`, and `score_perturbation` is derived from robust effort-density
variability clipped to `[1.0, 3.0]`. Score is an effort-conditioned synthetic
proxy, not observed project value.

Required role levels are not observable from GitHub. Each Type 5 instance
samples its difficulty mixture from a deterministic Dirichlet prior with
concentration proportional to `[25, 45, 20, 10]`, centered on:

```text
p_over = 0.25, p_match = 0.45, p_mentor = 0.20, p_future = 0.10.
```

The sampled mixture enforces broad bounds on mentorship and future roles, then
renormalizes. This prior is a Type 5 modeling decision, not an empirical GitHub
measurement.

### Type 5 Acceptance

Type 5 is origin-driven and does not use Type 2/3 target-vector search. For each
source window, the source-derived `GenerationParameters` and empirical profile
are held fixed. If an attempt fails structural validation, duplicate detection,
or a realism gate, only the deterministic seed changes.

The Type 5 realism gate requires structural validity, no duplicate structural
hash against Google and generated families, 3 to 8 non-empty generated profiles,
no catastrophic profile collapse unless supported by the source profile,
skill-skew difference at most 0.20, generated demand-supply alignment at least
0.55 and close to the source target, measured mentorship and future-role shares
within broad Type 5 bounds, and:

```text
directly_fillable_share
  + truly_valid_initial_mentorship_feasible_share >= 0.70.
```

The mentorship feasibility measurement respects the original same-project
mentorship rule: a one-below contributor is counted only when a distinct
qualified mentor can also be assigned to the same project.

Before final manifest update, the executable Type 5 final gate checks exact
accepted count, 100 unique hashes, structural validity, duplicate status,
overall realism pass, complete source provenance and metadata, all four windows,
at least 25 source repositories, anonymized final IDs, recorded seeds,
statistical report, provenance report, and methodology documentation.

This source corpus is purposively selected from mature public open-source
software repositories. It is not statistically representative of all software
companies, proprietary projects, non-software teamwork, all countries, or all
organization sizes. GitHub activity is an observable proxy for work, not a
complete measurement of human expertise or organizational decision-making.

## Reproducibility

Each CLI accepts a root seed. Child seeds are derived deterministically by a
fixed integer recurrence, and all random choices use local `random.Random`
instances. Metadata records family, id, seed, generator version, full parameter
set, generation attempts, validation, features, targets, distances and
structural hash. Config files are copied to `benchmark_metadata/*_config_used.json`
and their SHA-256 hashes are recorded in `benchmark_manifest.json`.

## Validation and Duplicates

Written files are reparsed and independently validated. Checks include header
counts, unique contributor/project names, duplicate contributor skill entries,
valid names, valid integer bounds, project/role counts, role skill names,
canonical reparse, initial fillability and a conservative positive-project
guard. Duplicate detection uses a canonical SHA-256 structural hash and is
checked against Google anchors and already generated family files.

## Diagnostics

`benchmark_generation/analysis/diagnostics.py` produces post-generation
feature and distance reports, and optional matplotlib plots. Any PCA or visual
projection added here is diagnostic only and is not used as the instance
generation mechanism or the final ISA projection.
