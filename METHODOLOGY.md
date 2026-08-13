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

## Type 3: Neighborhoods Around Anchors

Type 3 constructs local target clouds around each Google anchor. The default
allocation is balanced over the five anchors. Target directions are random
normalized directions in the same group-balanced feature metric, and radii are
stratified into configurable bands:

```text
small [0.03, 0.08], medium [0.08, 0.18],
outer [0.18, 0.35], wide [0.35, 0.55].
```

The search objective includes target distance, novelty, duplicate penalty, and
a radius penalty if the measured candidate is outside its configured radius
band. This prevents collapse onto the anchor and uncontrolled drift.

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
