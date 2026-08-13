You are working on a research codebase for a **PhD-level / Springer book chapter** on the **Mentorship and Teamwork Problem (MTP)**, originally introduced in the Google Hash Code 2022 Qualification Round.

Your task is to design and implement a **research-grade, reproducible, statistically defensible instance-generation pipeline** for **Benchmark Type 2, Type 3, and Type 4**.

This is not a simple random data-generation task. The generated benchmark will later be used for **Instance Space Analysis (ISA), algorithm footprint analysis, algorithm comparison, and potentially algorithm-selection experiments**. Therefore, methodological correctness, reproducibility, diversity, validation, and documentation are more important than code brevity.

---

# 1. Repository and Existing Data

There is a folder:

```text
google_instances/
```

containing the original Google Hash Code MTP instances.

First inspect the repository and the contents of `google_instances/`.

Do **not modify, overwrite, rename, or regenerate** the original Google files.

Identify the five non-trivial Hash Code instances and exclude the toy instance `a_an_example` from benchmark construction if it is present.

Before writing new code:

1. inspect the repository structure;
2. inspect any existing parsers, generators, feature extractors, validators, or utilities;
3. reuse/refactor existing reliable functionality instead of duplicating it;
4. understand the exact Hash Code input format;
5. verify all assumptions against the actual files/code.

Do not invent repository paths or APIs without checking them first.

---

# 2. Scientific Goal

The final extended benchmark will contain several instance families. For this task implement only:

```text
Type 2 — Gap-filling instances between the original Hash Code instances
Type 3 — Neighborhood instances around the original Hash Code instances
Type 4 — Broad random instances across multiple scales
```

Generate:

```text
100 Type 2 instances
100 Type 3 instances
100 Type 4 instances
```

The generator must follow the conceptual principle:

```text
generator parameters
        ↓
construct contributors/projects/roles
        ↓
complete valid MTP instance
        ↓
compute features from the completed instance
        ↓
evaluate whether the instance satisfies the family objective
```

Critically:

**Never directly construct an MTP instance by trying to invert a desired feature vector.**

Target feature vectors may guide a search over generator parameters, but every candidate must always be generated **forward** from a valid parameter configuration.

ISA itself must not be used as the generator.

---

# 3. Architecture

Prefer a modular structure such as:

```text
benchmark_generation/
    core/
        generator.py
        parser.py
        validator.py
        features.py
        distance.py
        parameter_sampling.py
        metadata.py

    type2/
        generate_type2.py

    type3/
        generate_type3.py

    type4/
        generate_type4.py

    configs/
        type2.yaml
        type3.yaml
        type4.yaml

    tests/
        ...

generated_instances/
    type2/
    type3/
    type4/

benchmark_metadata/
    ...
```

Adapt this structure if the repository already has an established architecture.

Do not unnecessarily duplicate code between Type 2, Type 3, and Type 4.

There should ideally be **one parameterized core MTP generator** and three family-specific generation/search strategies.

---

# 4. Core MTP Generator

Implement or reuse a single parameterized generator capable of generating all three families.

A generated instance must be produced in this order:

```text
1. Generate skill pool
2. Generate contributors
3. Generate contributor skill sets and levels
4. Generate projects
5. Generate durations, scores and best-before values
6. Generate project roles
7. Select required skills
8. Calibrate required skill levels against contributor supply
9. Validate initial viability
10. Write the final Hash Code-compatible instance
```

The generator should support at least these parameters:

```text
n_skills

skill_skew

n_contributors

skills_per_contributor_min
skills_per_contributor_max

level_mean
level_sd
level_max

n_profiles

n_projects

duration_min
duration_max

roles_min
roles_max

score_base
score_perturbation

tightness_min
tightness_max

role_skill_alignment

supply_quantile

p_over
p_match
p_mentor
p_future

fillable_fraction
```

Keep the entire parameter set for every generated instance.

---

# 5. Contributor Generation

Contributors must have:

* unique valid identifiers;
* a configurable number of skills;
* skills sampled from the common skill pool;
* configurable skill-frequency skew;
* valid integer skill levels.

Skill levels should support a truncated normal distribution controlled by:

```text
level_mean
level_sd
level_max
```

Values outside the valid interval should preferably be **redrawn rather than merely clipped**, to avoid artificial probability mass at the minimum/maximum boundary.

Support contributor specialisation using:

```text
n_profiles
```

where appropriate.

If `n_profiles = 1`, contributor skills can be sampled independently.

---

# 6. Project Generation

Each project must have:

```text
duration
score
best_before
roles
```

Duration:

```text
duration ~ configured valid range
```

Number of roles:

```text
roles ~ configured valid range
```

The score should not be completely independent random noise.

Use the conceptual relationship:

```text
effort = duration × number_of_roles
score ≈ score_base × effort × random_perturbation
```

so project value remains broadly related to project effort while still containing enough variation for optimization algorithms to exploit.

The deadline should be generated relative to project duration:

```text
best_before ≈ duration × tightness
```

rather than from one universal absolute deadline interval.

This makes deadline tightness comparable across small and large instances.

---

# 7. Role Skill Demand

Each project role requires exactly one skill.

Implement:

```text
role_skill_alignment
```

to control the relationship between contributor skill supply and role skill demand.

Conceptually:

```text
alignment ≈ 1
→ role demand closely follows available contributor expertise

alignment ≈ 0
→ demand becomes substantially less aligned with supply
```

The implementation should produce a continuous, documented interpolation rather than an arbitrary binary switch.

---

# 8. Required Skill-Level Generation

This mechanism is central to MTP benchmark quality.

Do not assign required role levels completely independently of contributor levels.

For each role:

1. inspect contributor levels for the required skill;
2. determine a reference level using:

```text
supply_quantile
```

3. generate the required level according to a mixture controlled by:

```text
p_over
p_match
p_mentor
p_future
```

Conceptually:

```text
p_over
→ role requirement below reference supply level

p_match
→ role requirement approximately equal to reference level

p_mentor
→ requirement one level above a plausible assigned contributor,
   creating mentorship-relevant assignments

p_future
→ requirement sufficiently above current supply to encourage
   learning/progression before the role becomes attainable
```

The four probabilities must satisfy:

```text
p_over + p_match + p_mentor + p_future = 1
```

Draw the category **per role**, not once for the whole instance.

This is important because mentorship behaviour requires heterogeneous role difficulty within the same project.

---

# 9. Mentorship Correctness

Mentorship must always respect the **actual Google Hash Code MTP rules**.

A contributor may fill a role if:

```text
skill_level >= required_level
```

or:

```text
skill_level == required_level - 1
```

provided another contributor assigned to the same project can validly mentor that skill according to the original problem rules.

Implement a dedicated mentorship validation function.

Add unit tests for at least:

```text
direct qualification

valid one-level-below mentorship

invalid two-level-below assignment

missing mentor

mentor on same project

mentor not assigned to same project

one contributor mentoring multiple roles

contributor both mentoring and being mentored
```

If terminology in the research draft's feature taxonomy is ambiguous relative to the original Hash Code mentorship mechanics, **do not silently alter the science**.

Instead:

1. preserve the original MTP feasibility rules;
2. document the ambiguity;
3. keep the feature definition configurable or clearly documented.

---

# 10. Minimum Initial Viability

Every generated instance must satisfy a configurable minimum viability criterion.

Default:

```text
fillable_fraction = 0.30
```

An instance in which almost nothing can initially be started is not useful for solver comparison.

Reject and regenerate candidates that violate the viability requirement.

Record the reason for every rejection.

Include a configurable:

```text
max_generation_attempts
```

to prevent infinite loops.

---

# 11. Feature Extraction

Implement or reuse a feature extractor that works **without solving the instance**.

Feature extraction should be deterministic.

The candidate feature set should cover the six conceptual groups below.

## A. Size and Scale

Include:

```text
number of contributors
number of projects
number of distinct skills
total roles

roles per project mean
roles per project standard deviation

skills per contributor mean
skills per contributor standard deviation
```

Use logarithmic transformation for raw scale variables when they are used for distance calculations.

---

## B. Supply and Demand

Include:

```text
contention

workload contention

demand concentration

supply concentration

demand-supply alignment

unmet skill share
```

Definitions must be mathematically documented.

For example:

```text
contention =
total number of roles / number of contributors
```

---

## C. Skill Levels

Include:

```text
contributor skill level mean
contributor skill level sd

required role level mean
required role level sd

level gap
level gap dispersion
```

Comparisons between demand and supply should be performed on compatible skills rather than blindly combining unrelated skill distributions.

---

## D. Fillability and Mentorship

Include meaningful measures of:

```text
direct fillability

mentorship-related fillability/dependence

learning-reachable demand

mentor availability

within-project mentorship feasibility
```

These features are especially important because mentorship is the defining mechanism distinguishing MTP from generic team-formation problems.

---

## E. Time and Deadlines

Include:

```text
duration mean
duration sd

deadline tightness mean
deadline tightness sd

infeasible-deadline share

schedule pressure
```

---

## F. Scores

Include:

```text
score mean
score sd

score concentration

score density

score-effort correlation

penalty exposure
```

All formulas must be documented either through docstrings or a methodology markdown file.

---

# 12. Feature Scaling for Coverage Search

Do not calculate Euclidean distance directly over raw heterogeneous features.

Implement an explicit feature-space transformation.

Requirements:

* log-transform appropriate size variables;
* normalize/scalе numeric features;
* avoid domination by one large-magnitude variable;
* handle constant features safely;
* handle NaN/undefined correlation values safely;
* avoid allowing the largest feature group to dominate solely because it has more columns.

Prefer configurable **group-balanced distance weighting** across:

```text
size
supply-demand
levels
mentorship
time
scores
```

For example, each conceptual feature group may receive equal aggregate weight, with weight divided among the features within that group.

Document the exact distance metric.

Do not hide this inside undocumented utility code.

---

# 13. TYPE 4 — Broad Random Family

Implement:

```text
generate_type4.py
```

Goal:

> Generate 100 independent, broadly distributed MTP instances spanning substantially different scales and structural characteristics.

Type 4 must not target positions relative to the Google instances.

It is an **origin-driven random family**.

Use the following research specification as the basis:

```text
n_skills:                       5–200
skill_skew:                     0.0–1.5

n_contributors:                 5–10,000, by scale band

skills_per_contributor:         1–20

level_mean:                     2–7
level_sd:                       0.5–3.0
level_max:                      10

n_profiles:                     1

n_projects:                     5–10,000, by scale band

duration:                       1–500

roles_per_project:              1–20

score_base:                     0.5–5.0

score_perturbation:             1.0–3.0

tightness:                      0.5–6.0

role_skill_alignment:           0.0–1.0

supply_quantile:                0.5–0.9

fillable_fraction:              0.30
```

For:

```text
p_over
p_match
p_mentor
p_future
```

sample the four weights **jointly per instance** so they sum exactly to 1.

If the methodology does not already prescribe the simplex-sampling distribution, make it configurable.

A symmetric Dirichlet distribution may be used as a defensible default, but explicitly document it as an **implementation choice**, not as something specified by the original benchmark design.

---

# 14. Type 4 Scale Bands

Type 4 must contain:

```text
small
medium
large
```

instances.

Do not let uniform integer sampling between 5 and 10,000 cause almost every instance to become large.

Use a deliberate stratified/logarithmic sampling design.

Approximately balance the 100 instances over the three bands.

For example:

```text
~33 small
~33 medium
~34 large
```

However, do not blindly hard-code arbitrary band boundaries if they are already defined elsewhere in the repository or research materials.

Priority:

1. use existing documented scale bands if present;
2. otherwise define configurable, scientifically defensible bands;
3. document the chosen boundaries in the metadata and methodology.

Consider sampling scale approximately logarithmically.

Also vary the ratio:

```text
n_projects / n_contributors
```

rather than making project and contributor counts strongly correlated in every instance.

The family should contain low-, medium-, and high-contention cases.

---

# 15. TYPE 2 — Gap Filling Between Google Instances

Implement:

```text
generate_type2.py
```

Goal:

> Produce 100 valid MTP instances occupying structurally meaningful gaps between the five non-trivial original Google Hash Code instances.

This should **not** mean simple parameter interpolation.

The original instances do not come with known generator parameters.

Use their **measured feature representations**.

Pipeline:

```text
load original Google instances
        ↓
compute feature vectors
        ↓
transform/normalize feature space
        ↓
identify meaningful gaps between originals
        ↓
define target regions/target points
        ↓
search generator parameter space
        ↓
generate candidate instance FORWARD
        ↓
measure its actual features
        ↓
compare candidate to target
        ↓
accept best valid diverse candidates
```

## Gap construction

Because there are only five non-trivial original instances, avoid fragile assumptions requiring large-sample density estimation.

Use a transparent geometrical strategy.

Possible principled strategy:

1. compute pairwise distances between Google instances;
2. construct candidate line segments/regions between structurally separated originals;
3. allocate more targets to large uncovered separations;
4. generate target locations at multiple interpolation positions rather than only exact midpoints;
5. ensure all five original instances participate where appropriate.

Possible interpolation:

```text
target = (1 - α) * feature_A + α * feature_B
```

with values such as:

```text
α ∈ [0.2, 0.8]
```

but preferably generate them with a space-filling design rather than one fixed list.

Do not claim that interpolation itself creates an instance.

The target is only a **search objective**.

---

# 16. Type 2 Parameter Search

For each target location, perform a search over generator parameter configurations.

A simple one-shot random candidate is insufficient for a PhD-level benchmark.

Implement an adaptive search such as:

```text
large initial parameter sample
        ↓
forward-generate candidates
        ↓
feature-distance evaluation
        ↓
retain elite parameter configurations
        ↓
perturb/refine parameters
        ↓
repeat
```

A well-designed evolutionary strategy, cross-entropy search, Bayesian optimization, CMA-ES, or another documented derivative-free strategy is acceptable.

Avoid unnecessary external dependencies if a robust implementation can be written using the current stack.

The objective should include at least:

```text
target feature distance
+
duplicate penalty
+
lack-of-diversity penalty
+
invalidity penalty
```

Potentially:

```text
J(candidate) =
    d(candidate_features, target)
  + λ1 * duplicate_penalty
  + λ2 * crowding_penalty
```

Do not optimize to unrealistically tiny differences in features if that causes structurally degenerate instances.

The purpose is **coverage**, not feature-vector overfitting.

---

# 17. Type 2 Diversity

The 100 accepted Type 2 instances must not simply become 100 small perturbations of one midpoint.

Add explicit diversity control.

Examples:

* minimum pairwise feature distance;
* novelty against previously accepted Type 2 instances;
* balanced target allocation over original-instance pairs/regions;
* duplicate structural signature detection.

Compute and report:

```text
nearest-neighbor distance
mean pairwise distance
minimum pairwise distance
distance to nearest original Google instance
```

for the final family.

---

# 18. TYPE 3 — Neighborhood Around Google Instances

Implement:

```text
generate_type3.py
```

Goal:

> Generate 100 instances around the structural neighborhoods of the five original Google instances.

This is different from Type 2.

Conceptually:

```text
Type 2 = BETWEEN original Google instances

Type 3 = AROUND individual Google instances
```

Distribute the 100 instances approximately evenly:

```text
20 around each of the 5 Google instances
```

unless a scientifically stronger allocation emerges from the data.

---

# 19. Type 3 Target Construction

For every original Google instance:

```text
g_i
```

construct multiple local target points around its normalized feature representation.

Do not create only infinitesimal noise.

Use a meaningful neighborhood with configurable radii.

For example:

```text
small local radius
medium local radius
outer local radius
```

Generate target directions using a space-filling approach.

Potential methods:

```text
Latin Hypercube Sampling

Sobol sampling

random normalized directions with radius stratification
```

The targets should form a local cloud/annulus around each Google instance rather than collapsing onto the original point.

Then use the same **forward generator + parameter search** principle as Type 2.

---

# 20. Type 3 Acceptance Objective

Candidate quality should account for:

```text
distance to the target location

desired distance/radius from the Google anchor

distance from already accepted Type 3 instances

validity

non-duplication
```

The algorithm should prevent two failure modes:

### Failure A

```text
all generated instances are nearly identical to the Google anchor
```

### Failure B

```text
generated instances drift so far away that they are no longer
meaningfully "around" the Google instance
```

Provide configurable:

```text
radius_min
radius_max
```

or an equivalent local-neighborhood criterion.

---

# 21. Original Google Instances as Immutable Anchors

The five non-trivial Google instances are benchmark anchors.

For each original, compute and save:

```text
instance name
counts
all candidate features
normalized feature vector
```

The original instances must remain read-only.

Create a file such as:

```text
benchmark_metadata/google_anchor_features.csv
```

or an equivalent structured file.

---

# 22. Reproducibility

Reproducibility is mandatory.

Every instance must have:

```text
family
family_instance_id
random_seed
generator_version
timestamp if desired
full generator parameters
generation attempt count
validation result
feature values
target vector, if Type 2/3
distance to target, if Type 2/3
anchor/pair information, if Type 2/3
```

A fixed root seed must deterministically derive child seeds.

Example conceptual design:

```text
root_seed = 2026

Type 2 seeds
Type 3 seeds
Type 4 seeds
```

Do not rely on global uncontrolled random state.

Seed:

```text
Python random
NumPy RNG
any optimizer RNG
```

explicitly.

Running the generation twice with the same configuration and seed must reproduce the same files.

---

# 23. Instance File Naming

Use deterministic, sortable names.

For example:

```text
type2_001.txt
type2_002.txt
...
type2_100.txt

type3_001.txt
...
type3_100.txt

type4_001.txt
...
type4_100.txt
```

If useful, encode only family/id in the filename and put all experimental information in metadata rather than creating excessively long filenames.

---

# 24. Metadata

For each family produce a metadata table:

```text
type2_metadata.csv
type3_metadata.csv
type4_metadata.csv
```

and preferably JSON/JSONL as well.

Each row should contain:

```text
instance filename
seed
family
generator parameters
generation attempts
validation information
all feature values

target information where applicable

distance metrics
```

Also save a machine-readable master manifest:

```text
benchmark_manifest.json
```

---

# 25. Validation

Implement a strict independent validator.

Do not trust the generator merely because it produced the file.

After writing each instance, reparse it from disk and validate it independently.

Check at least:

```text
header counts match actual data

unique contributor names

unique project names

valid skill names

no duplicate skill entry for one contributor

valid integer levels

valid durations

valid scores

valid best-before values

valid role counts

role skills exist in the skill universe where expected

all values respect Hash Code format constraints

instance can be parsed again exactly

initial fillability requirement is satisfied
```

Also validate mentorship calculations independently from the generation routine.

Fail loudly on corruption.

---

# 26. Duplicate Detection

Compute a canonical structural hash for every generated instance.

Ensure:

```text
no exact duplicates within Type 2
no exact duplicates within Type 3
no exact duplicates within Type 4
```

Also check duplicates across:

```text
Types 2–4
```

and against:

```text
google_instances/
```

Do not rely only on filenames.

---

# 27. Statistical Validation

After generating each family, produce a statistical report.

At minimum report:

```text
count

contributors min / median / mean / max
projects min / median / mean / max
skills min / median / mean / max
roles min / median / mean / max

contention distribution

skill level distribution

required level distribution

direct fillability

mentorship-related measures

deadline tightness

schedule pressure

score density
```

For Type 2:

```text
distance to target
distance to nearest Google anchor
pair/region coverage
within-family diversity
```

For Type 3:

```text
anchor identity
distance to anchor
radius distribution
within-anchor diversity
between-anchor diversity
```

For Type 4:

```text
scale-band counts
parameter coverage
feature coverage
```

Flag suspiciously degenerate distributions automatically.

---

# 28. Visual Diagnostics

Create a separate analysis/diagnostic script.

Do **not** use final ISA to generate the instances.

However, diagnostic visualization of the measured candidate feature space is encouraged.

Produce plots such as:

```text
pairwise-distance distributions

feature distributions by family

Google-anchor vs Type 2 distances

Google-anchor vs Type 3 distances

Type 4 parameter/scale coverage
```

Optionally produce a clearly labeled exploratory PCA projection for sanity checking.

If PCA is used here, explicitly state:

```text
This visualization is diagnostic only and is not used as the
instance-generation mechanism or the final ISA projection.
```

---

# 29. Computational Efficiency

The generator should be able to handle large Type 4 instances approaching thousands of contributors/projects.

Avoid algorithms that accidentally become:

```text
O(contributors × roles × skills)
```

when indexing can reduce the complexity.

Build efficient structures such as:

```text
skill -> contributors
skill -> level-sorted contributors

skill -> supply frequency
skill -> demand frequency
```

Reuse precomputed statistics when possible.

For Type 2/3 search, avoid writing every rejected candidate to disk.

Prefer:

```text
generate in memory
→ evaluate
→ write only accepted/final candidates
```

Optionally retain lightweight search logs.

---

# 30. Command-Line Interfaces

Each family script should support a clean CLI.

Example:

```bash
python generate_type2.py \
    --google-dir google_instances \
    --output-dir generated_instances/type2 \
    --count 100 \
    --seed 2026 \
    --config configs/type2.yaml
```

Similarly:

```bash
python generate_type3.py ...
python generate_type4.py ...
```

Useful optional arguments:

```text
--count
--seed
--config
--workers
--overwrite
--dry-run
--verbose
--max-attempts
```

Never overwrite existing research data silently.

---

# 31. Configuration

Move meaningful research choices into configuration files instead of hiding them in source code.

Examples:

```text
family size

root seed

parameter ranges

distance weights

target interpolation ranges

Type 3 neighborhood radii

candidate population size

search iterations

elite fraction

novelty penalty

minimum accepted distance

fillable_fraction

scale bands
```

The scripts should save a copy/hash of the configuration used for every experiment.

---

# 32. Logging

Use structured, readable logging.

For Type 2/3, log information such as:

```text
target 017/100
anchor or pair
best current feature distance
accepted/rejected
attempt count
reason for rejection
```

For Type 4:

```text
instance 041/100
scale band
contributors
projects
roles
viability
accepted/rejected
```

At completion provide a summary.

---

# 33. Tests

Create proper automated tests.

Include:

### Parser tests

Known Google instances parse correctly.

### Generator tests

Generated instances satisfy the syntax.

### Reproducibility tests

Same seed/config → identical canonical output.

### Different-seed tests

Different seeds normally → different instances.

### Mentorship tests

All edge cases described above.

### Feature tests

Use manually constructed tiny instances where expected feature values can be calculated by hand.

### Type 2 tests

Generated targets lie between intended anchors according to the defined methodology.

### Type 3 tests

Neighborhood targets satisfy intended radius rules.

### Type 4 tests

Sampling covers multiple scale bands.

### Validator tests

Intentionally corrupted instances are rejected.

---

# 34. Research Methodology Documentation

Create:

```text
METHODOLOGY.md
```

This is extremely important.

Explain mathematically and precisely:

```text
how each family is generated

how generator parameters are sampled

how required skill levels are calibrated

how p_over/p_match/p_mentor/p_future work

how fillable_fraction is evaluated

how the 31 candidate features are calculated

how features are transformed/scaled

how feature distance is defined

how Type 2 gaps are defined

how Type 2 targets are generated

how Type 2 parameter search operates

how Type 3 neighborhoods are defined

how Type 3 targets are generated

how Type 4 scale stratification works

how seeds guarantee reproducibility

how duplicates are detected

how generated instances are validated
```

Use equations where useful.

The methodology must be written so that another researcher could independently reproduce the benchmark.

---

# 35. Important Scientific Constraint

Do not tune the benchmark to make any solver look good or bad.

Generation must remain:

```text
solver-independent
```

Do not use:

```text
GA score
ILP score
runtime
optimality gap
algorithm winner
```

as objectives when constructing Types 2, 3, or 4.

Algorithm results will be evaluated only **after the benchmark has been finalized**.

This separation is necessary to avoid benchmark-selection bias.

---

# 36. Do Not Silently Guess

This work will be used in an academic book chapter.

Whenever the specification leaves an implementation detail undefined — for example:

```text
exact small/medium/large boundaries

simplex distribution for Type 4 difficulty weights

exact Type 3 radius

exact feature-group weights

exact optimizer used for target search
```

do not silently pretend that the value came from the methodology.

Instead:

1. inspect the repository/documentation for an existing definition;
2. if none exists, select a scientifically defensible default;
3. expose it through configuration;
4. document it clearly as an implementation/design decision.

---

# 37. Expected Deliverables

By the end I expect working implementations for:

```text
Type 2 generator/search
Type 3 generator/search
Type 4 generator

shared core generator

MTP parser

strict validator

feature extractor

feature normalization/distance module

metadata/manifest generation

duplicate detection

diagnostic analysis

automated tests

METHODOLOGY.md

README instructions
```

And generated data directories compatible with:

```text
100 Type 2
100 Type 3
100 Type 4
```

---

# 38. Before Coding

Before making substantial changes, inspect the repository and produce a concise technical assessment covering:

```text
existing relevant files/modules

format of google_instances/

what can be reused

what must be implemented

proposed architecture

any ambiguities found in the specification

specific scientific assumptions that require explicit documentation
```

Then proceed with the implementation.

Do not stop merely because a methodological detail is unspecified. Make the strongest defensible research-engineering decision, expose it as configuration, and document the assumption.

---

# 39. Quality Standard

Treat this as **research software intended to support a PhD-level publication**, not competition throwaway code.

Prioritize:

```text
correctness
reproducibility
methodological transparency
modularity
statistical diversity
validation
testability
computational efficiency
clear documentation
```

Every generated instance should be traceable from:

```text
instance
→ seed
→ parameters
→ generator version
→ validation
→ measured features
→ family-specific target/search information
```

The final implementation should make it possible to defend, in a peer-reviewed publication, **why each of Types 2, 3, and 4 exists, exactly how each was produced, and why the resulting benchmark provides more meaningful structural coverage than the five original non-trivial Google Hash Code instances.**
