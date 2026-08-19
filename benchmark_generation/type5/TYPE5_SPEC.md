
You are implementing Type 5 of a research-grade synthetic benchmark for the
Mentorship and Teamwork Problem (MTP).

This benchmark is intended for publication-quality Instance Space Analysis.

TYPE 5 SCIENTIFIC PURPOSE
=========================

Type 5 must represent:

    Instances grounded in REAL software projects and REAL contributor activity.

However, Type 5 must NOT simply copy GitHub repositories directly into arbitrary
MTP files.

The correct methodology is:

    real public software-development activity
                ↓
    empirical contributor/project measurements
                ↓
    empirical source configuration
                ↓
    shared parameterized MTP forward generator
                ↓
    validation + realism quality gates
                ↓
    Type 5 benchmark instance

The resulting Type 5 instances are therefore:

    EMPIRICALLY GROUNDED SYNTHETIC MTP INSTANCES

They are NOT claimed to be literal copies of GitHub projects.

This distinction must be explicit in code, metadata, documentation, and the
book-chapter methodology.

Do not invent data.
Do not fabricate GitHub statistics.
Do not infer information that cannot defensibly be measured.
Do not claim that a proxy variable is an observed variable.

Treat this as PhD-level / book-chapter research software.

============================================================

1. FIRST: INSPECT THE EXISTING REPOSITORY
   ============================================================

Before changing anything:

1. Read AGENTS.md completely if present.
2. Inspect:

   - benchmark_generation/core/
   - generator.py
   - features.py
   - distance.py
   - validator.py
   - metadata.py
   - parser.py
   - parameter_sampling.py
   - Type 2 implementation
   - Type 3 implementation
   - Type 4 implementation
   - existing configs
   - existing metadata
   - tests
   - METHODOLOGY.md
3. Reuse the existing common forward generator.

DO NOT:

- implement a completely separate Type 5 generator,
- modify Google original instances,
- silently change Type 2/3/4 behavior,
- overwrite Type 2/3/4 metadata,
- use solver results in generation,
- generate instances directly from target feature vectors.

The shared principle remains:

    parameters -> forward generation -> complete MTP instance -> measured features

============================================================
2. REQUIRED FINAL OUTPUT
========================

The final Type 5 benchmark must contain exactly:

    100 accepted unique Type 5 instances

stored as:

    generated_instances/type5/type5_001.txt
    ...
    generated_instances/type5/type5_100.txt

Every final instance must be:

- structurally valid,
- valid under actual MTP semantics,
- unique,
- reproducible,
- traceable to a real empirical source configuration,
- realistic according to Type 5 quality gates,
- accompanied by complete metadata.

Do not report Type 5 as complete unless all 100 pass.

============================================================
3. CORE RESEARCH DESIGN: 25 REPOSITORIES × 4 WINDOWS
=====================================================

Use exactly four fixed six-month observation windows per repository.

Use half-open UTC intervals so there are no overlaps:

W1:
    [2024-08-17T00:00:00Z, 2025-02-17T00:00:00Z)

W2:
    [2025-02-17T00:00:00Z, 2025-08-17T00:00:00Z)

W3:
    [2025-08-17T00:00:00Z, 2026-02-17T00:00:00Z)

W4:
    [2026-02-17T00:00:00Z, 2026-08-17T00:00:00Z)

This gives:

    25 repositories × 4 real temporal configurations = 100 source configurations

The dates are an IMPLEMENTATION DESIGN CHOICE and must be documented as such.

Do not claim that the four six-month windows were prescribed by the original MTP
methodology.

Each repository/window should normally generate exactly one final Type 5 instance.

============================================================
4. PRIMARY REAL-WORLD SOURCE CORPUS
===================================

Create:

    benchmark_generation/configs/type5_sources.yaml

containing the following 25 PRIMARY public repositories.

PYTHON / WEB / DATA
-------------------

owner: pallets
repo: flask
url: https://github.com/pallets/flask
domain: python_web

owner: fastapi
repo: fastapi
url: https://github.com/fastapi/fastapi
domain: python_web_api

owner: psf
repo: requests
url: https://github.com/psf/requests
domain: python_library

owner: pydantic
repo: pydantic
url: https://github.com/pydantic/pydantic
domain: python_validation

owner: django
repo: django
url: https://github.com/django/django
domain: python_web_framework

owner: numpy
repo: numpy
url: https://github.com/numpy/numpy
domain: scientific_computing

owner: pandas-dev
repo: pandas
url: https://github.com/pandas-dev/pandas
domain: data_analysis

owner: scikit-learn
repo: scikit-learn
url: https://github.com/scikit-learn/scikit-learn
domain: machine_learning

JVM / DATA ENGINEERING
----------------------

owner: spring-projects
repo: spring-boot
url: https://github.com/spring-projects/spring-boot
domain: java_framework

owner: apache
repo: kafka
url: https://github.com/apache/kafka
domain: event_streaming

owner: apache
repo: spark
url: https://github.com/apache/spark
domain: big_data

owner: apache
repo: airflow
url: https://github.com/apache/airflow
domain: data_orchestration

owner: elastic
repo: elasticsearch
url: https://github.com/elastic/elasticsearch
domain: distributed_search

AI / ML
-------

owner: pytorch
repo: pytorch
url: https://github.com/pytorch/pytorch
domain: deep_learning

owner: tensorflow
repo: tensorflow
url: https://github.com/tensorflow/tensorflow
domain: deep_learning

FRONTEND / JAVASCRIPT / TYPESCRIPT
----------------------------------

owner: react
repo: react
url: https://github.com/react/react
domain: frontend_framework

owner: vitejs
repo: vite
url: https://github.com/vitejs/vite
domain: frontend_tooling

owner: vuejs
repo: core
url: https://github.com/vuejs/core
domain: frontend_framework

owner: nodejs
repo: node
url: https://github.com/nodejs/node
domain: javascript_runtime

owner: microsoft
repo: vscode
url: https://github.com/microsoft/vscode
domain: developer_tooling

CLOUD / INFRASTRUCTURE
----------------------

owner: kubernetes
repo: kubernetes
url: https://github.com/kubernetes/kubernetes
domain: cloud_orchestration

owner: prometheus
repo: prometheus
url: https://github.com/prometheus/prometheus
domain: monitoring

owner: hashicorp
repo: terraform
url: https://github.com/hashicorp/terraform
domain: infrastructure_as_code

SYSTEMS
-------

owner: dotnet
repo: runtime
url: https://github.com/dotnet/runtime
domain: systems_runtime

owner: rust-lang
repo: rust
url: https://github.com/rust-lang/rust
domain: programming_language

============================================================
5. RESERVE SOURCE REPOSITORIES
==============================

A primary repository/window may be unusable because of insufficient activity,
unavailable metadata, or another objective quality problem.

Do NOT fabricate missing data.

Use the following reserve repositories only when needed:

https://github.com/encode/httpx
https://github.com/ansible/ansible
https://github.com/redis/redis
https://github.com/apache/flink

Replacement must preserve the same W1/W2/W3/W4 time window.

Every replacement must be documented with:

- failed primary source,
- failure reason,
- replacement source,
- window,
- date,
- deterministic replacement order.

Do not cherry-pick replacements based on resulting MTP feature-space position.

============================================================
6. OFFICIAL DATA SOURCES
========================

Use public GitHub data only.

Primary GitHub API documentation:

REST API:
https://docs.github.com/en/rest

Repositories:
https://docs.github.com/en/rest/repos/repos

Commits:
https://docs.github.com/en/rest/commits/commits

Repository statistics:
https://docs.github.com/en/rest/metrics/statistics

Pull requests:
https://docs.github.com/en/rest/pulls/pulls

Issues:
https://docs.github.com/en/rest/issues/issues

Milestones:
https://docs.github.com/en/rest/issues/milestones

Releases:
https://docs.github.com/en/rest/releases/releases

Rate limits:
https://docs.github.com/en/rest/using-the-rest-api/rate-limits-for-the-rest-api

Use GitHub REST API version:

    2026-03-10

with headers equivalent to:

    Accept: application/vnd.github+json
    X-GitHub-Api-Version: 2026-03-10

============================================================
7. AUTHENTICATION AND RATE-LIMIT SAFETY
=======================================

Bulk collection must use authenticated public GitHub access.

Attempt authentication in this order:

1. environment variable:
   GITHUB_TOKEN
2. existing authenticated GitHub CLI session:
   gh auth status

Never print the token.
Never log the token.
Never write the token to config.
Never commit the token.

If no authenticated access is available:

    STOP BEFORE BULK COLLECTION.

Explain to the user that authenticated read access is required for practical
collection.

Do not silently collect a tiny partial corpus using unauthenticated limits.

Use read-only access only.
No GitHub write operations are required.

Implement:

- pagination,
- ETag/cache support where practical,
- x-ratelimit-* header logging,
- retry-after handling,
- exponential backoff,
- 403/429 handling,
- resume-safe caching,
- bounded concurrency.

Do not aggressively parallelize API calls.

Cache all successfully fetched public responses so reruns do not unnecessarily
consume API quota.

============================================================
8. DATA PRIVACY / ETHICS
========================

Use only public repository activity.

Do NOT collect or publish:

- contributor email addresses,
- private profile data,
- private repositories,
- authentication data,
- personal demographic data.

Real GitHub usernames must NOT appear in final MTP .txt instances.

Final contributor IDs remain:

    c00000
    c00001
    ...

Final skills remain:

    s0000
    s0001
    ...

Final projects remain:

    p00000
    p00001
    ...

Source usernames may be temporarily present in the local acquisition cache if
returned by GitHub, but final public benchmark metadata should use anonymized
local contributor identifiers.

Do not distribute raw repository source code.

We are using aggregate public development metadata, not republishing the projects.

============================================================
9. BUILD A REPRODUCIBLE RAW DATA COLLECTOR
==========================================

Implement:

    benchmark_generation/type5/collect_github_corpus.py

It must collect and cache source information.

Suggested local structure:

    type5_source_cache/
        pallets__flask/
            repository.json
            languages.json
            releases.json
            milestones.json
            W1/
            W2/
            W3/
            W4/
        ...

Add:

    type5_source_cache/

to .gitignore if raw source cache should not be committed.

Do NOT place API tokens there.

============================================================
10. REPOSITORY-LEVEL DATA TO COLLECT
====================================

For every source repository collect:

A. Repository metadata

Use:

    GET /repos/{owner}/{repo}

Record:

- owner/repository
- canonical URL
- repository ID
- default branch
- archived
- fork
- created_at
- pushed_at
- license identifier if available
- default branch head SHA at collection time
- collection timestamp
- API version

Reject repositories that are:

- private,
- archived,
- disabled,
- obvious mirrors whose GitHub activity does not represent the development
  process,
- or unavailable.

B. Repository languages

Use:

    GET /repos/{owner}/{repo}/languages

Record language byte counts.

Do not interpret repository language byte percentage directly as contributor
skill level.

It is a repository-level composition measurement only.

C. Releases

Use:

    GET /repos/{owner}/{repo}/releases

Collect:

- tag
- created_at
- published_at
- prerelease
- draft

Use published non-draft releases preferentially for temporal/deadline proxies.

D. Milestones

Use:

    GET /repos/{owner}/{repo}/milestones?state=all

Collect:

- title or anonymized ID
- created_at
- updated_at
- due_on
- closed_at
- open_issues
- closed_issues

Milestones are used only when they provide a defensible deadline signal.

============================================================
11. WINDOW-SPECIFIC DATA
========================

For each repository and each W1–W4 window collect activity separately.

The unit of empirical "project/work package" should primarily be:

    merged pull request

This is a proxy for a real development work package.

Do NOT claim that a GitHub PR is literally equivalent to the original meaning
of "project" in MTP.

Document it as a practical empirical work-unit proxy.

============================================================
12. PULL REQUEST COLLECTION
===========================

For every repository/window identify merged pull requests whose merge timestamp
belongs to that window.

Collect enough information to determine:

- PR number
- anonymized author identity
- created_at
- merged_at
- additions if available
- deletions if available
- changed_files count
- milestone if any
- labels if useful
- merge commit SHA if available

Do not use comments, reactions, stars, or popularity as direct measures of
business value.

If necessary, use the Pull Requests API and filter client-side by merged_at.

The implementation must cope with repositories that have very large numbers
of PRs.

Record:

    source_pr_total_count

for the window.

For detailed file-level analysis, do NOT download every PR from huge projects.

Use a deterministic representative sample.

============================================================
13. DETERMINISTIC PR SAMPLING
=============================

For each repository/window:

- analyze all merged PRs if <= 60,
- otherwise analyze exactly 60 representative PRs.

The sampling must be deterministic under root seed 2026.

Stratify primarily by calendar time:

    approximately 10 PRs per month in each six-month window

where possible.

Additionally prevent one author from dominating the detailed sample.

Target:

    no single contributor > 20% of sampled PRs

unless the source data itself makes this impossible.

If impossible, preserve the real distribution and record the concentration.

Do NOT randomly choose a convenient set of PRs each run.

============================================================
14. PR FILE-LEVEL DATA
======================

For each sampled PR retrieve the changed files using the official Pull Request
files endpoint.

For each changed file record only aggregate research information such as:

- path
- additions
- deletions
- status

Do not store source-file content.

Use the file paths to infer technical work areas / skill tokens.

============================================================
15. SKILL TAXONOMY
==================

MTP skills should be inferred from technical development activity.

Do not use raw repository language percentage alone.

Construct skill tokens primarily from:

    programming language + subsystem

Examples:

    Python::ORM
    Python::HTTP
    Java::Core
    Java::Security
    TypeScript::Compiler
    C++::Runtime
    Go::Scheduler

The final .txt benchmark can anonymize them to sXXXX.

Store the semantic source token only in research metadata.

============================================================
16. LANGUAGE MAPPING
====================

Implement one deterministic extension-to-language mapping.

At minimum support:

.py       -> Python
.pyx      -> Cython
.java     -> Java
.kt/.kts  -> Kotlin
.scala    -> Scala
.js/.jsx  -> JavaScript
.ts/.tsx  -> TypeScript
.c/.h     -> C
.cc/.cpp/.cxx/.hpp -> C++
.cs       -> CSharp
.rs       -> Rust
.go       -> Go
.rb       -> Ruby
.php      -> PHP
.swift    -> Swift
.sql      -> SQL
.sh       -> Shell
.ps1      -> PowerShell
.html     -> HTML
.css/.scss/.sass -> CSS
.vue      -> Vue
.r        -> R

Special files may map to:

Dockerfile -> DevOps
*.yml/*.yaml under CI/deployment directories -> DevOps

Do not let lock files or generated configuration dominate skills.

============================================================
17. SUBSYSTEM MAPPING
=====================

Infer subsystem from path.

Use the first meaningful directory component.

Ignore generic wrappers such as:

src
lib
libs
packages
modules
components

and use the next meaningful component when appropriate.

Examples:

    src/compiler/parser/file.rs
        -> Rust::Compiler

    packages/react-dom/src/...
        -> JavaScript::react-dom

    django/db/...
        -> Python::db

    spring-boot-project/.../security/...
        -> Java::security

Make this deterministic.

Do not manually encode repository-specific outcomes unless the rule is
documented in a small explicit mapping table.

============================================================
18. FILES / PATHS TO EXCLUDE
============================

Exclude or down-weight:

node_modules/
vendor/
third_party/
dist/
build/
target/
generated/
coverage/
cache/
binary assets
lock files
minified files
compiled artifacts

Documentation-only changes should not dominate technical skills.

README/docs-only PRs may be excluded from the technical role sample if they do
not touch implementation/build/test/configuration code.

Record the number excluded.

============================================================
19. SKILL TOKEN FILTERING
=========================

Rare one-off skill tokens create noise.

A skill token should normally be retained only if it appears in:

- at least 2 distinct human contributors, OR
- at least 3 sampled PRs,

unless the repository/window is too small.

Merge very rare subsystem tokens back into their parent language skill.

Target a meaningful skill universe.

Hard safety range:

    5 <= n_skills <= 200

If a real window naturally contains fewer than 5 usable technical skills,
mark the source configuration insufficient rather than inventing arbitrary
skills.

============================================================
20. IDENTIFY HUMAN CONTRIBUTORS
===============================

A source contributor is a human contributor active in the observation window.

Use merged PR authors and, when needed, commit authors from:

    GET /repos/{owner}/{repo}/commits?since=...&until=...

Filter obvious bots.

Examples to exclude include identities matching patterns such as:

[bot]
dependabot
renovate
github-actions
pre-commit-ci
codecov
mergify
release automation
known project-specific bots

Maintain:

    benchmark_generation/configs/type5_bot_patterns.yaml

Do not exclude humans merely because their activity is low.

Record:

- total contributors observed
- human contributors retained
- bots removed
- unidentified/null GitHub authors

============================================================
21. CONTRIBUTOR-SKILL ACTIVITY
==============================

For contributors appearing in the detailed PR sample, compute activity per
skill token.

For contributor c and skill s calculate:

    lines_changed_cs
    files_touched_cs
    prs_touching_skill_cs

Then define a deterministic activity score:

    activity_cs =
        0.60 * log1p(lines_changed_cs)
      + 0.25 * log1p(files_touched_cs)
      + 0.15 * log1p(prs_touching_skill_cs)

Do NOT use raw lines changed directly as a skill level.

The log transform reduces domination by extremely large changes.

============================================================
22. MAPPING CONTRIBUTOR ACTIVITY TO MTP LEVEL 1–10
===================================================

For each retained skill s:

1. collect activity_cs for contributors who have that skill;
2. rank contributors by activity within that skill;
3. convert percentile rank to MTP level:

   level =
   1 + floor(9 * percentile_rank)
4. clip to [1, 10].

Absent skill:

    level = 0

Do not claim these values are true human competence measurements.

Document them as:

    relative empirical activity-based skill-level proxies

This phrase is important.

============================================================
23. CONTRIBUTOR PROFILE / SPECIALIZATION MODEL
==============================================

Type 5 requires realistic contributor specialization.

From contributor × skill activity vectors:

- normalize each contributor vector,
- cluster contributor profiles.

Test:

    k = 3, 4, 5, 6, 7, 8

Choose k using silhouette score where statistically meaningful.

Constraints:

    3 <= n_profiles <= 8

If the source data cannot support at least three meaningful contributor
specializations, the source window should fail its Type 5 source-quality gate
instead of fabricating fake clusters.

Store:

- chosen k
- silhouette score
- profile sizes
- profile centroids
- largest profile share
- smallest profile share

The generated contributor profiles should be based on these empirical centroids,
not generic uniformly random profiles.

============================================================
24. EMPIRICAL SKILL SUPPLY
==========================

For each source window calculate:

    supply_s =
        number of contributors with skill s

Also calculate:

- skill supply proportions
- supply concentration
- Gini coefficient or equivalent concentration measure
- Zipf-like skill skew estimate using the same benchmark convention where
  applicable

These values ground:

    skill_skew

============================================================
25. EMPIRICAL ROLE / SKILL DEMAND
=================================

For each sampled merged PR:

1. identify the retained skill tokens touched;
2. treat these as empirical role-demand signals;
3. count how often each skill is required across PRs.

For an empirical work package:

    role_count =
        number of retained distinct skill tokens touched

If role_count > 20:

    retain the 20 skill tokens with the greatest changed-line contribution.

Do not allow a single giant PR to create hundreds of MTP roles.

Calculate:

- roles per work package mean
- sd
- median
- P10
- P90
- demand count per skill
- demand concentration

============================================================
26. DEMAND-SUPPLY ALIGNMENT
===========================

Calculate alignment between:

    contributor skill supply distribution

and:

    empirical PR skill demand distribution

Use the same normalized alignment definition consistently across the benchmark.

Prefer the existing:

    demand_supply_alignment

feature definition if applicable.

Store:

    empirical_role_skill_alignment

The Type 5 generator should preserve relatively strong supply-demand alignment.

Target:

    target_alignment =
        clamp(max(0.60, empirical_alignment), 0.60, 0.95)

IMPORTANT:

0.60 is a Type 5 realism implementation threshold, not an observed universal law.

Document it.

============================================================
27. PROJECT DURATION PROXY
==========================

For each sampled merged PR calculate:

    cycle_time_days =
        merged_at - created_at

Use:

    duration = max(1, ceil(cycle_time_days))

Exclude pathological records such as:

- negative timestamps,
- missing merge timestamps.

Do not remove legitimately long-running PRs solely because they are long.

Use robust source-window bounds:

    duration_min = max(1, round(P10))
    duration_max = max(duration_min, round(P90))

Do not simply use observed min/max because one extreme PR can distort the entire
generated family.

============================================================
28. DEADLINE / BEST-BEFORE PROXY
================================

GitHub does not provide an objective deadline for every PR.

Do NOT pretend otherwise.

For each sampled work package determine deadline proxy in this priority order:

1. milestone due_on, if a meaningful milestone exists;
2. nearest subsequent stable release publication date, if reasonably associated;
3. corpus-level empirical fallback.

For a PR with a usable deadline:

    deadline_window =
        deadline_proxy_date - created_at

Define approximate tightness:

    tightness =
        deadline_window_days / max(1, duration_days)

Clip individual calibration observations to:

    [0.5, 6.0]

For each source window with >= 10 deadline observations:

    tightness_min = P10
    tightness_max = P90

Otherwise use the corpus-level P10/P90 of valid deadline proxies.

Record:

    deadline_proxy_coverage
    deadline_proxy_mode

Possible values:

    milestone
    release
    mixed
    corpus_fallback

Never silently invent "real deadlines".

============================================================
29. SCORE / BUSINESS-VALUE PROXY
================================

GitHub does NOT contain an objective cross-project business-value score.

Therefore:

DO NOT use:

- stars,
- comments,
- reactions,
- contributor fame,
- follower count

as MTP score.

Instead, treat MTP score as SYNTHETIC.

Ground only its SCALE/VARIABILITY weakly in observed engineering effort.

For each sampled work package compute:

    effort_i =
        log1p(additions_i + deletions_i)
        * sqrt(max(1, role_count_i))

Across the entire Type 5 source corpus compute one global normalization constant
K such that the median resulting score density is approximately 2.0.

Then derive source-window:

    score_base

from median normalized effort density and clip it to:

    [0.5, 5.0]

Derive:

    score_perturbation

from robust effort variability, clipped to:

    [1.0, 3.0]

The final generated project scores remain synthetic.

Document explicitly:

    Score is an effort-conditioned synthetic proxy, not observed project value.

============================================================
30. DIFFICULTY MIX: TYPE 5 PRIOR
================================

Required MTP skill levels cannot be directly observed from GitHub.

Therefore the difficulty mixture is an explicit Type 5 modeling prior.

Use expected mixture:

    p_over   = 0.25
    p_match  = 0.45
    p_mentor = 0.20
    p_future = 0.10

Interpretation:

- approximately 25% roles comfortably below available expertise,
- approximately 45% matched to current expertise,
- approximately 20% mentorship/stretch roles,
- approximately 10% future/far-beyond roles.

Do not keep the exact same probabilities for all 100 instances.

For each instance sample a deterministic simplex perturbation around this
expected mixture using root seed 2026.

A defensible implementation:

    Dirichlet concentration proportional to:
        [25, 45, 20, 10]

Then enforce:

    0.15 <= p_mentor <= 0.30
    0.05 <= p_future <= 0.15

Renormalize the four values afterward.

This prior is an IMPLEMENTATION DECISION motivated by the Type 5 design.

Do not call it empirically observed from GitHub.

============================================================
31. MTP MENTORSHIP SEMANTICS MUST BE CORRECT
============================================

Critical rule:

A contributor one level below a role requirement can only fill the role if
ANOTHER contributor assigned to the SAME PROJECT has that required skill at or
above the role requirement.

Therefore:

    someone exists at requirement-1

is NOT by itself sufficient for valid mentorship.

Keep separate concepts:

- directly fillable role
- one-below candidate exists
- valid same-project mentorship feasible
- future/learning-reachable role
- currently impossible role

Do not repeat the earlier taxonomy mistake where one-below availability was
automatically called mentorship-fillable.

============================================================
32. DERIVE GENERATION PARAMETERS FROM EACH SOURCE WINDOW
========================================================

Create:

    benchmark_generation/type5/build_empirical_profiles.py

For every valid source repo/window derive one GenerationParameters-style source
configuration.

Use approximately:

n_contributors:
    number of active human contributors observed in the window

If extremely large:
    use a documented scale-preserving compression only if necessary for runtime,
    and record both source and generated target counts.

Do not silently cap.

n_projects:
    total merged PR count in the window

If extremely large and generator limits require compression:
    use the SAME scale factor used for contributors where scientifically
    appropriate and record it.

n_skills:
    retained empirical technical skill tokens

skills_per_contributor_min:
    max(1, rounded P10 empirical skills/contributor)

skills_per_contributor_max:
    max(min, rounded P90 empirical skills/contributor)
    cap at 20

level_mean:
    empirical mean of inferred 1–10 contributor skill levels

level_sd:
    empirical standard deviation

level_max:
    10

n_profiles:
    empirically chosen k in [3,8]

skill_skew:
    empirical skew estimate

duration_min:
    P10 PR cycle time

duration_max:
    P90 PR cycle time

roles_min:
    max(1, rounded P10 empirical roles/work-package)

roles_max:
    rounded P90
    cap at 20

score_base:
    effort-conditioned synthetic calibration described above

score_perturbation:
    robust empirical effort variability

tightness_min:
    empirical/corpus deadline proxy P10

tightness_max:
    empirical/corpus deadline proxy P90

role_skill_alignment:
    clamp(max(0.60, empirical alignment), 0.60, 0.95)

supply_quantile:
    estimate from empirical author expertise relative to skill suppliers where
    possible.

If enough data exists:
    compute the percentile position of PR authors within the supplier skill
    distribution for skills they modify;
    use the median;
    clip to [0.60, 0.85].

If insufficient:
    use 0.75 and explicitly flag fallback.

p_over / p_match / p_mentor / p_future:
    Type 5 prior described above

fillable_fraction:
    keep the shared generator technical viability floor consistent unless there
    is a documented reason to change it.

Do not abuse this parameter to manufacture Type 5 realism.
Use separate Type 5 quality gates below.

============================================================
33. ONE SOURCE WINDOW -> ONE TYPE 5 INSTANCE
============================================

Each valid repo/window gives one source parameter configuration.

Generate one final MTP instance from it using:

    shared forward generator

not a direct GitHub-to-text conversion.

Root seed:

    2026

Derive deterministic instance seeds from:

    root seed
    source repository index
    window index
    retry number

Example conceptual seed derivation:

    derive_seed(
        2026,
        repository_index,
        window_index,
        retry_index,
        salt=TYPE5_CONSTANT
    )

============================================================
34. DO NOT USE TYPE 2/3 ADAPTIVE FEATURE TARGET SEARCH
======================================================

Type 5 is origin-driven.

DO NOT:

    create target ISA vectors
    optimize toward feature points
    interpolate Google anchors
    use Type 3 radius logic

Correct:

    empirical source configuration
            ↓
    forward generator
            ↓
    validation
            ↓
    realism acceptance/rejection

If an attempt fails a realism criterion:

    regenerate with another deterministic seed

using the SAME source-derived parameter configuration.

Do not mutate parameters until an arbitrary feature target is hit.

============================================================
35. TYPE 5 INSTANCE REALISM QUALITY GATES
=========================================

Every accepted Type 5 instance must pass all structural validation and all
applicable realism gates.

A. STRUCTURAL

- valid contributor count
- valid project count
- unique contributor IDs
- unique project IDs
- unique skills per contributor
- legal skill levels
- valid project durations/scores/deadlines
- correct role counts
- valid skill references
- no malformed text

B. DUPLICATES

Structural hash must not match:

- Google instances
- Type 2
- Type 3
- Type 4
- earlier Type 5 instances
- any other already-generated family

C. CONTRIBUTOR PROFILES

Require:

    3 <= measured/generated profile count <= 8

No empty profile.

Avoid catastrophic profile collapse.

Flag if:

    largest profile > 70% of contributors

unless the empirical source itself strongly supports this.

D. SKILL SKEW

Generated skill supply skew should resemble the empirical source.

Target absolute difference:

    <= 0.20

between empirical source skill_skew and generated measured skill_skew.

If this is repeatedly impossible, diagnose generator limitations.
Do not silently widen tolerance indefinitely.

E. DEMAND-SUPPLY ALIGNMENT

Generated measured alignment should satisfy:

    generated_alignment >= 0.55

and preferably:

    |generated_alignment - source_target_alignment| <= 0.15

F. DIFFICULTY MIX

Measured role difficulty should remain consistent with the Type 5 prior.

Family-level target:

    p_mentor around 0.20
    p_future around 0.10

Per-instance acceptable broad bounds:

    mentorship-oriented share: 0.10–0.30
    future/far-beyond share: 0.03–0.18

Do not confuse nominal generator probabilities with measured outcome.
Store both.

G. INITIAL REALISTIC FEASIBILITY

Because realistic teams normally work near existing competence, require:

    directly_fillable_share
      + truly_valid_initial_mentorship_feasible_share
        >= 0.70

This is stricter than the general technical viability floor.

If the existing feature definitions do not measure this correctly, implement the
measurement correctly rather than approximating it.

H. PROJECT / CONTRIBUTOR SCALE

Generated counts should equal or closely track the source-derived target unless a
documented scale compression was applied.

I. SOURCE-TO-GENERATED DISTRIBUTION MATCH

Compare key properties:

- skills/contributor mean
- roles/project mean
- contributor level mean
- required level mean
- skill skew
- role demand/supply alignment
- duration median
- deadline tightness
- mentorship share
- future share

Do not require exact equality.

Use reasonable tolerances and record deviations.

============================================================
36. MAXIMUM RETRIES
===================

Use a finite retry budget.

For example:

    max_generation_attempts_per_source = 30

If no accepted instance can be produced:

- mark the source window failed;
- report exactly which realism criterion cannot be achieved;
- do not accept the best invalid candidate.

Then use reserve-source replacement policy.

============================================================
37. SOURCE-WINDOW QUALITY GATE
==============================

Before any MTP generation, a real source window itself must be viable.

Require preferably:

- > = 5 human active contributors
  >
- > = 5 merged PRs
  >
- > = 5 retained skills
  >
- enough detailed PRs to estimate roles
- enough skill-evidenced contributors to form at least 3 profiles
- nonzero supply/demand statistics

If a source window fails:

    DO NOT invent missing information.

Use reserve-source replacement.

For borderline cases, report the reason.

============================================================
38. EXACT FINAL 100 MAPPING
===========================

Attempt to maintain:

    primary repo 01 W1 -> type5_001
    primary repo 01 W2 -> type5_002
    primary repo 01 W3 -> type5_003
    primary repo 01 W4 -> type5_004

    primary repo 02 W1 -> type5_005
    ...

Use deterministic ordering.

If a replacement is required:

keep the same family_instance_id slot and record:

    source_replacement=true
    original_source_repo
    replacement_source_repo
    replacement_reason

============================================================
39. METADATA FOR EVERY INSTANCE
===============================

Create:

    benchmark_metadata/type5_metadata.csv
    benchmark_metadata/type5_metadata.jsonl

For each instance record at least:

IDENTITY

- filename
- family=type5
- family_instance_id
- root_seed
- instance_seed
- generator_version
- structural_hash

SOURCE PROVENANCE

- source_owner
- source_repo
- source_url
- source_domain
- source_repository_id
- source_default_branch
- source_snapshot_sha
- source_window_name
- source_window_start
- source_window_end
- source_collection_timestamp
- source_replacement
- original_source_if_replaced

SOURCE ACTIVITY

- source_merged_pr_count
- source_sampled_pr_count
- source_human_contributor_count
- source_bot_count
- source_skill_count
- source_profile_count
- source_profile_silhouette
- source_skill_skew
- source_alignment
- source_skills_per_contributor_mean
- source_roles_per_work_unit_mean
- source_duration_p10
- source_duration_median
- source_duration_p90
- source_deadline_proxy_coverage
- source_deadline_proxy_mode

GENERATOR PARAMETERS

- every GenerationParameters value
- nominal difficulty mixture
- any source scale compression factor

MEASURED GENERATED FEATURES

- all benchmark features
- complete 31-feature vector if that remains the benchmark definition

REALISM VALIDATION

- structural_validation_pass
- duplicate_pass
- profiles_pass
- skill_skew_pass
- alignment_pass
- difficulty_mix_pass
- initial_realistic_feasibility_pass
- source_distribution_match_pass
- overall_type5_quality_pass
- generation_attempts

PROXY FLAGS

- skill_level_proxy = activity_percentile
- project_unit_proxy = merged_pull_request
- deadline_proxy
- score_proxy = effort_conditioned_synthetic

============================================================
40. EMPIRICAL CORPUS METADATA
=============================

Create:

    benchmark_metadata/type5_sources.csv
    benchmark_metadata/type5_empirical_corpus.csv
    benchmark_metadata/type5_empirical_corpus.json
    benchmark_metadata/type5_source_windows.csv
    benchmark_metadata/type5_data_provenance.json

These files must make the real-data grounding independently auditable.

============================================================
41. FAMILY-LEVEL STATISTICAL REPORT
===================================

Create:

    benchmark_metadata/type5_statistical_report.json

Report:

GENERAL

- accepted count
- failed attempts
- replacements
- duplicate count
- validation failures

SOURCE COVERAGE

- repository count
- domain count
- count per repository
- count per window
- replacements by reason

SCALE

- contributors min/mean/median/max
- projects min/mean/median/max
- skills min/mean/median/max
- total roles min/mean/median/max

CONTRIBUTOR REALISM

- skills/contributor
- inferred levels
- profile count
- profile size distributions
- skill skew

DEMAND/SUPPLY

- alignment
- supply concentration
- demand concentration
- unmet skill share

MENTORSHIP

- directly fillable share
- truly mentorship-feasible share
- learning-reachable share
- future share
- mentor availability
- within-project mentorship feasibility

TIME

- duration
- deadline tightness
- infeasible deadline share
- schedule pressure

SCORE

- score mean/sd
- score density
- score-effort relationship

SOURCE MATCH

- empirical vs generated key metric error distributions

DIVERSITY

- pairwise normalized feature distance
- nearest-neighbor distance
- source-within-repo diversity
- across-domain diversity

============================================================
42. DIAGNOSTIC PLOTS
====================

Produce diagnostic plots, not generation targets.

At minimum:

1. source vs generated contributors
2. source vs generated project counts
3. source vs generated skill skew
4. source vs generated demand-supply alignment
5. source vs generated roles/project
6. Type 5 mentorship share distribution
7. Type 5 future-role share distribution
8. PCA projection containing:
   - Google anchors
   - Type 4
   - Type 5

If Type 2/3 are available, optionally include them.

PCA is for diagnostics only.

Do not generate Type 5 in PCA coordinates.

============================================================
43. FAMILY-LEVEL INTERPRETATION CHECK
=====================================

Type 5 should empirically look different from broad random Type 4.

Expected tendency:

Type 5 should generally show:

- stronger contributor specialization,
- non-uniform skill popularity,
- stronger supply-demand alignment,
- role requirements closer to available expertise,
- substantial but not dominant mentorship,
- fewer far-future roles,
- realistic project-size and contributor-size relationships.

Do not force every individual Type 5 instance to dominate every Type 4 instance
on these metrics.

Assess this statistically at family level.

============================================================
44. PRESERVE SOURCE LIMITATIONS
===============================

Document explicitly:

This source corpus is purposively selected from mature public open-source
software repositories.

It is NOT statistically representative of:

- all software companies,
- proprietary projects,
- non-software teamwork,
- all countries,
- all organization sizes.

GitHub activity is an observable proxy for work, not a complete measurement of
human expertise or organizational decision-making.

This limitation belongs in METHODOLOGY.md and the eventual book chapter.

============================================================
45. TESTS
=========

Add research-grade automated tests.

At minimum test:

- source configuration parsing
- fixed window boundaries
- deterministic source ordering
- GitHub pagination
- cache/resume
- no token logging
- bot filtering
- file-extension language mapping
- subsystem extraction
- rare-skill merging
- PR sampling determinism
- contributor activity scoring
- skill percentile -> 1..10 mapping
- profile count 3..8
- source alignment calculation
- duration proxy
- deadline proxy fallback
- score proxy
- deterministic difficulty prior
- Type 5 parameter conversion
- MTP structural validity
- exact mentorship semantics
- duplicate rejection
- realism gates
- source replacement
- deterministic generation with root seed 2026
- metadata completeness
- exact final count

A test must verify:

    real GitHub username never appears in final generated .txt file.

============================================================
46. REPRODUCIBILITY
===================

Root seed:

    2026

Store:

- API version
- collection timestamp
- source repository
- source snapshot SHA
- source observation dates
- config hash
- generator version
- all seeds
- source aggregate measurements

Raw GitHub data itself may change over time due repository administration, so the
stored source snapshot/provenance is mandatory.

Where legally/practically possible store sufficient aggregate source metadata to
reproduce parameter derivation without repeatedly querying GitHub.

============================================================
47. PERFORMANCE
===============

The collection may take time.

Optimize correctly:

- cache every successful API response,
- never refetch cached immutable PR details unnecessarily,
- paginate at 100/page where supported,
- process repository by repository,
- write checkpoint state after every source/window,
- resume after interruption,
- log progress.

Example logging:

    Type5 source 03/25 psf/requests
    W2 PR discovery ...
    merged PRs: 84
    sampled: 60
    file details: 20/60
    contributors retained: 17
    skills retained: 9
    profiles: 4
    source profile READY

Do not reduce scientific quality simply because collection is slow.

============================================================
48. REQUIRED CLI
================

Implement commands equivalent to:

STEP 1 — collect source data

python -m benchmark_generation.type5.collect_github_corpus 
  --sources benchmark_generation/configs/type5_sources.yaml 
  --cache-dir type5_source_cache 
  --metadata-dir benchmark_metadata 
  --seed 2026 
  --verbose

STEP 2 — derive empirical source configurations

python -m benchmark_generation.type5.build_empirical_profiles 
  --sources benchmark_generation/configs/type5_sources.yaml 
  --cache-dir type5_source_cache 
  --metadata-dir benchmark_metadata 
  --config benchmark_generation/configs/type5.yaml 
  --seed 2026 
  --verbose

STEP 3 — pilot

python -m benchmark_generation.type5.generate_type5 
  --output-dir generated_instances/type5_pilot 
  --metadata-dir benchmark_metadata/type5_pilot 
  --config benchmark_generation/configs/type5.yaml 
  --count 10 
  --seed 2026 
  --pilot 
  --verbose

STEP 4 — full generation

python -m benchmark_generation.type5.generate_type5 
  --output-dir generated_instances/type5 
  --metadata-dir benchmark_metadata 
  --config benchmark_generation/configs/type5.yaml 
  --count 100 
  --seed 2026 
  --overwrite 
  --verbose

============================================================
49. PILOT GATE
==============

Before generating final 100:

generate a 10-instance pilot covering:

- at least 5 different source repositories,
- at least 3 technical domains,
- all four temporal windows where possible.

Output a pilot table with:

filename
source_repo
window
contributors
projects
skills
profiles
skill_skew
alignment
direct_fillability
mentorship_feasibility
future_share
structural_valid
duplicate_pass
realism_pass

Do not proceed to final generation until all 10 pilot instances pass.

============================================================
50. FINAL TYPE 5 QUALITY GATE
=============================

Before declaring completion verify automatically:

accepted_count == 100

unique_structural_hashes == 100

all_structurally_valid == true

all_duplicate_checks_pass == true

all_overall_type5_quality_pass == true

all_100_have_source_provenance == true

all_100_have_complete_metadata == true

all_four_windows_represented == true

source_repository_coverage >= 25
    unless documented objective replacements preserve the 100-source design

no_real_usernames_in_final_instances == true

all_seeds_recorded == true

statistical_report_exists == true

provenance_report_exists == true

methodology_documented == true

If any required condition fails:

    DO NOT REPORT TYPE 5 AS COMPLETE.

============================================================
51. METHODOLOGY DOCUMENTATION
=============================

Update METHODOLOGY.md with a dedicated Type 5 section.

Clearly separate:

A. EMPIRICALLY OBSERVED

Examples:

- source contributor activity
- repository language composition
- changed technical areas
- PR counts
- PR cycle times
- contributor specialization
- skill supply
- skill demand
- release/milestone timing

B. DERIVED PROXIES

Examples:

- contributor activity -> relative skill level
- merged PR -> work package
- milestone/release -> deadline proxy
- changed technical areas -> MTP role demand

C. SYNTHETIC MODELING DECISIONS

Examples:

- MTP project score
- p_over / p_match / p_mentor / p_future
- exact acceptance thresholds
- source-window duration
- quality tolerances

Never mix these three categories.

============================================================
52. BOOK-CHAPTER READY METHODOLOGY SUMMARY
==========================================

Prepare a publication-ready methodology subsection in:

    benchmark_metadata/type5_methodology_draft.md

It should explain approximately:

"Type 5 grounds the synthetic benchmark in empirically observed software
development configurations. Public development activity was collected from a
diverse corpus of mature open-source repositories. Each repository was divided
into four fixed six-month observation windows. Contributor activity, technical
skill supply, work-package demand, contributor specialization, development
cycle times, and release/milestone timing were measured from public repository
activity. These observations were not converted directly into benchmark files.
Instead, they parameterized the same forward MTP generator used for the other
synthetic families. Quantities not directly observable from repository data,
including project utility scores and exact role difficulty, were modeled
explicitly as documented proxies or priors."

Do not copy this paragraph blindly if implementation differs.
Make the final methodology match the actual code and data.

============================================================
53. SOURCE CITATION / PROVENANCE FILE
=====================================

Create a human-readable:

    benchmark_metadata/TYPE5_SOURCES.md

For every repository include:

- project name
- GitHub URL
- organization/owner
- domain
- observation windows
- snapshot SHA
- access/collection date
- license identifier if available
- whether used as primary or replacement

This gives the book chapter a precise provenance trail.

============================================================
54. DO NOT FAKE SUCCESS WHEN INTERNET/API DATA IS MISSING
=========================================================

If the agent cannot access GitHub:

STOP.

If authentication is missing:

STOP before bulk collection.

If a source endpoint fails:

retry/cached fallback.

If data still cannot be retrieved:

record source failure and use documented reserve policy.

Never generate "realistic" numbers manually and pretend they were measured from
GitHub.

============================================================
55. FINAL RESPONSE TO ME
========================

When all work is complete, do NOT answer only:

    "Type 5 completed."

Give me a research report containing:

1. exact files created/modified;
2. source repositories successfully collected;
3. any source replacements and reasons;
4. number of API records collected;
5. number of contributors observed;
6. number of bots removed;
7. number of merged PRs represented;
8. skill taxonomy statistics;
9. contributor profile statistics;
10. empirical alignment statistics;
11. duration/deadline proxy statistics;
12. exact score-proxy methodology;
13. exact difficulty prior;
14. pilot results;
15. total final accepted instances;
16. structural validation results;
17. duplicate results;
18. realism quality-gate results;
19. source-to-generated matching statistics;
20. family-level Type 5 statistical summary;
21. reproducibility details;
22. runtime;
23. API/cache limitations;
24. methodology limitations;
25. exact command needed to reproduce Type 5.

Also explicitly state:

    whether the final 100 Type 5 instances are ready for use in the
    Instance Space Analysis benchmark.

Do not claim they are ready unless every final quality gate passes.

============================================================
56. IMPORTANT FINAL PRINCIPLE
=============================

The objective is NOT:

    "make 100 GitHub-looking files."

The objective is:

    construct 100 reproducible MTP benchmark instances whose structural
    configurations are demonstrably grounded in empirical software-project
    and contributor activity while preserving the exact mathematical semantics
    of the Mentorship and Teamwork Problem.

Prioritize methodological defensibility, provenance, correctness,
reproducibility, and transparent proxy definitions over implementation speed.
