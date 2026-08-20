
Implement Type 6 of the MTP benchmark as a literature-grounded synthetic
family derived from the published MSLIB benchmark for the Multi-Skilled
Resource-Constrained Project Scheduling Problem (MSRCPSP).

Read the existing benchmark-generation architecture, Type 2–5 methodology,
shared MTP generator, feature extractor, validator, duplicate checker,
metadata/report infrastructure, and tests before making changes.

Do not modify Type 2, Type 3, Type 4, or Type 5 behavior.

======================================================================
SCIENTIFIC PURPOSE
==================

Type 6 must represent MTP instances grounded in configurations from an
established project-scheduling benchmark from the literature.

Source:

MSLIB – Multi-Skilled Resource-Constrained Project Library
Mario Vanhoucke / Ghent University OR&S group

Repository:
https://github.com/mariovanhoucke/MSLIB-Multi-Skilled-Resource-Constrained-Project-Library

Use and lock the source release/artifact:

MSLIB official release asset MSLIB_v2_2.zip, retrieved from GitHub release
v2.2, SHA-256 f1672d2e2e85091f598e497b898b20c468b0b9a10030c5ca8d8ae798b6529e8c,
with upstream documentation still describing the dataset release as v2.1
(June 2026) in the repository README/release notes.

Do NOT silently use "latest" on every run. Record the observed release tag,
download URL, archive checksum, acquisition date, and upstream provenance
so that the Type 6 family is reproducible if a later MSLIB version appears.

Upstream versioning/provenance note:

- GitHub release/tag observed: v2.2
- release asset: MSLIB_v2_2.zip
- SHA-256: f1672d2e2e85091f598e497b898b20c468b0b9a10030c5ca8d8ae798b6529e8c
- release body states: "This is release v2.1"
- repository README states: Last update: v2.1 (June 2026)
- repository documentation states that previous releases are removed when a
  new release is published and recommends using the latest release.

The Type 6 implementation must identify the source artifact by the actual
downloaded GitHub release asset and checksum, not by an inferred historical
v2.1 archive. Do not claim that the downloaded archive is the historical v2.1
tag archive. Keep the tiny v2.1 tag archive and checksum in provenance as
evidence of why it was not usable, but do not use it as the data source.

Relevant literature:

Snauwaert, J., & Vanhoucke, M. (2022).
Mathematical formulations for project scheduling problems with categorical
and hierarchical skills.
Computers & Industrial Engineering, 169, 108147.

Snauwaert, J., & Vanhoucke, M. (2023).
A classification and new benchmark instances for the multi-skilled
resource-constrained project scheduling problem.
European Journal of Operational Research, 307(1), 1–19.

Type 6 must be described as:

"literature-grounded synthetic MTP instances derived from published
MSRCPSP benchmark configurations."

Do not describe the resulting instances as literal MSRCPSP instances.

======================================================================
IMPORTANT DESIGN DECISION
=========================

Do NOT directly reinterpret the entire MSRCPSP problem as MTP.

MSRCPSP contains concepts that do not exist in MTP, particularly:

- activity precedence constraints;
- a global makespan-minimization objective;
- dummy start/end activities;
- potentially different interpretations of hierarchical skills.

MTP instead contains:

- independent projects;
- project duration;
- project score;
- best-before date;
- skill/level roles;
- contributor learning;
- same-project mentorship.

Therefore use this pipeline:

published MSLIB instance
    ->
parse source workforce/project/skill/network structure
    ->
derive a literature source profile
    ->
parameterize the shared MTP forward generator
    ->
generate a synthetic MTP instance
    ->
validate
    ->
compute ISA features

This preserves the common forward-generation methodology used by the
synthetic benchmark families.

Do NOT perform inverse ISA target-vector generation.

Type 6 is origin-driven, not coverage-target-driven.

======================================================================
WHICH MSLIB DATASETS TO USE
===========================

Use MSLIB1, MSLIB2, MSLIB3, and MSLIB4 for the final Type 6 family.

Target:

MSLIB1: 25 source instances
MSLIB2: 25 source instances
MSLIB3: 25 source instances
MSLIB4: 25 source instances

TOTAL: 100 Type 6 source configurations.

Do NOT include MSLIB5 in the final Type 6 family at this stage.

Reason:

MSLIB1–4 are literature-generated artificial MSRCPSP benchmark sets.
MSLIB5 is empirical and would conceptually overlap with Type 5, which is
already the empirically grounded real-project family.

MSLIB5 may optionally be parsed and reported as a future/held-out
validation set, but must not count among the final 100 Type 6 instances.

======================================================================
PHASE 1 — ACQUIRE AND VERIFY MSLIB
===================================

Create:

benchmark_generation/type6/

with appropriate modules such as:

download_mslib.py
parse_msrcp.py
source_features.py
select_sources.py
build_literature_profiles.py
generate_type6.py
quality.py
diagnostics.py

and configuration:

benchmark_generation/configs/type6.yaml

Store downloaded source data outside generated MTP outputs, e.g.:

type6_source_data/

Do not commit large upstream archives unless repository policy and
upstream licensing clearly permit redistribution.

Check the MSLIB repository/release for licensing or data-use terms.
Record attribution and license information.

If redistribution terms are unclear, keep only:

- deterministic download instructions;
- release/tag;
- archive checksum;
- source filenames;
- provenance metadata;

and do not redistribute the original source files in our public benchmark
repository.

Use the accompanying official MSRCPSP_DataFormat document as the
authoritative format specification.

Do not guess the source file schema.

Parse the actual release format.

Cross-check parsed instance counts/parameters against:

"MSLIB (Parameters and BKS).xlsx"

where possible.

======================================================================
PHASE 2 — SOURCE PARSER
========================

The parser must recover, where present:

- source dataset name: MSLIB1/2/3/4;
- source instance identifier and filename;
- number of real activities;
- dummy start/end nodes;
- precedence arcs;
- activity durations;
- number of resources;
- number of skills;
- resource-to-skill mastery;
- resource skill levels;
- activity skill requirements;
- activity required skill levels;
- multiplicity of required resources per skill/level;
- network statistics;
- source parameters/indicators available in the supplied Excel file;
- BKS values as provenance/diagnostic information only.

Do not use BKS to target final MTP ISA location.

Do not select source instances because their converted MTP instance
looks attractive in ISA space.

Source selection must happen before Type 6 MTP generation.

======================================================================
MSRCPSP -> MTP SEMANTIC MAPPING
===============================

The central conceptual mapping is:

MSRCPSP resource
    ->
MTP contributor

MSRCPSP skill
    ->
MTP skill

MSRCPSP resource skill level
    ->
MTP contributor skill level

MSRCPSP real activity
    ->
source evidence for an MTP project/work-package distribution

MSRCPSP activity duration
    ->
MTP project-duration distribution

MSRCPSP skill requirements
    ->
MTP role-demand distribution

MSRCPSP resource/skill availability
    ->
MTP skill-supply distribution

MSRCPSP workforce breadth/depth
    ->
MTP contributor profiles and skill-level structure

Do not carry real source identifiers into final MTP file contents.

Final MTP files use:

c00000...
s0000...
p00000...

Source names and source skill identifiers belong only in research
metadata/provenance. Final Type 6 filenames may include the MSLIB set and
canonical source instance ID for traceability; the file contents must not.

======================================================================
HIERARCHICAL SKILL SEMANTICS
============================

Inspect the MSLIB data/documentation and determine exactly how skill
levels are represented.

Where hierarchical resource levels and activity requirement levels exist,
preserve their relative structure.

Prefer the source interpretation corresponding to lower-bound hierarchical
skill eligibility when available, because this is closest to MTP direct
eligibility:

contributor level >= required role level

Do not assume a specific MSLIB variant name without checking the
documentation.

Document the exact MSLIB source interpretation used.

If a source instance contains categorical/binary skill information only,
do not fabricate empirical hierarchical levels.

Either:

1. represent mastered categorical skills consistently at level 1 and
   document the limitation;

or

2. exclude that particular source interpretation if the selected
   hierarchical MSLIB representation provides the required level data.

Never invent arbitrary skill expertise and call it observed MSLIB data.

======================================================================
SOURCE CONFIGURATION -> GENERATION PARAMETERS
=============================================

For every selected source instance derive a Type 6 literature profile.

At minimum derive:

n_contributors
    = number of human/renewable source resources

n_projects
    = number of real non-dummy source activities

n_skills
    = number of source skills

skills_per_contributor distribution
    = directly from source workforce

skill-level distribution
    = directly from source hierarchical skill data

skill supply
    = number/proportion of resources mastering each skill at each level

skill skew
    = computed from source workforce

contributor profiles
    = derived from source resource-skill vectors

roles_per_project distribution
    = source activity skill-demand multiplicities

required-level distribution
    = source activity requirement levels

duration distribution
    = source real-activity durations

demand concentration
    = source skill demand over activities

supply concentration
    = source workforce skill supply

demand-supply alignment
    = calculated using the same definition used by the MTP benchmark

contention
    = source demand relative to available skilled resources

network structure
    = retained as source metadata and transformed only into temporal
      pressure proxies; MTP itself must not contain precedence arcs

Do not silently cap source sizes unless technically necessary.

If any compression or scaling becomes unavoidable, stop and report it
before using the compressed source in the final family.

======================================================================
PRECEDENCE NETWORK -> MTP TEMPORAL PRESSURE
===========================================

MTP has no precedence constraints.

Therefore DO NOT encode precedence as fake skills and DO NOT alter MTP
semantics to add precedence constraints.

Instead use the source precedence network only to derive temporal-pressure
characteristics.

For every MSLIB source instance compute from the DAG:

- topological order;
- earliest start;
- earliest finish;
- critical-path length;
- latest start/latest finish under a precedence-only horizon;
- activity slack;
- normalized topological position;
- network depth;
- source network indicators already supplied by MSLIB.

Use these measurements to calibrate MTP:

- deadline tightness distribution;
- best-before-position distribution;
- schedule-pressure parameters.

The exact mapping must be deterministic and documented.

A preferred approach is to derive a normalized temporal position for
source activities and use the empirical distribution of those positions
when generating MTP best-before values.

The source precedence relation itself must NOT appear as a hard constraint
in the final MTP instance.

======================================================================
MTP SCORE
=========

MSRCPSP minimizes makespan and does not naturally contain the same project
score concept as Google Hash Code MTP.

Therefore MTP score is a DERIVED SYNTHETIC FIELD.

Do not claim that it is observed in MSLIB.

Do not use BKS objective values directly as per-project scores.

Construct score from transparent source-derived work characteristics.

Recommended components:

activity/work-package effort
    = duration × role count

skill scarcity
    = inverse availability of the required skill/level in the source
      workforce

network criticality
    = function of normalized source activity slack/topological position

Then use a deterministic bounded score model such as:

raw_score =
    effort
    * f(skill_scarcity)
    * f(network_criticality)

Calibrate only a global score scaling constant if needed to keep scores
numerically reasonable.

Do not optimize this scaling constant to improve ISA coverage.

Metadata/methodology must explicitly say:

"Type 6 scores are literature-structure-conditioned synthetic proxies;
MSLIB does not provide an MTP project-value objective."

======================================================================
MENTORSHIP AND LEARNING
=======================

MSLIB itself is not an MTP mentorship benchmark.

Do NOT inject an arbitrary Type 5-style 20% mentorship target.

Instead derive the natural mentorship potential from source hierarchical
skill structure.

For each source skill-level demand, measure:

- contributors directly qualified:
  level >= requirement
- contributors exactly one level below:
  level == requirement - 1
- whether sufficiently skilled potential mentors exist for that skill

Use these statistics to derive the Type 6 difficulty mixture.

Type 6 should reflect the literature source, not a manually imposed
difficulty prior.

The shared MTP generator may create:

- directly matched roles;
- naturally mentorship-capable roles;
- harder/future roles;

in proportions driven by the selected MSLIB source profile.

All final validity checks must use the EXACT MTP mentorship semantics:

a contributor one level below a role requirement is valid only if another
contributor assigned to the SAME PROJECT has that skill at or above the
required level.

Do not confuse global mentor availability with same-project mentorship
feasibility.

======================================================================
SOURCE SELECTION — 25 PER MSLIB SET
====================================

Do not randomly pick arbitrary files and do not cherry-pick based on the
resulting MTP ISA features.

First create a source-space feature table for every MSLIB1–4 instance
using only source-native information.

Use the actual available MSLIB/Excel indicator names rather than
inventing nonexistent columns.

Candidate source-space dimensions should include available measures of:

- number of activities;
- number of resources;
- number of skills;
- workforce breadth;
- workforce depth/skill levels;
- skill availability/skill strength;
- resource availability;
- average mastered skills per resource;
- activity skill demand;
- network complexity/order strength;
- critical-path/network depth;
- resource/skill contention.

Then select exactly 25 source instances from EACH of MSLIB1–4 using a
deterministic diversity-oriented source-space procedure.

Preferred method:

1. standardize source-native indicators within a clearly documented
   selection space;
2. ensure important categorical parameter settings/subgroups are covered;
3. use deterministic stratified/maximin selection;
4. root seed = 2026;
5. break ties by source filename/ID.

MSLIB4 should deliberately retain coverage of its hard-instance
subgroups, not only one parameter combination.

Selection must be based solely on MSLIB source-space properties.

After those 100 sources are frozen, DO NOT replace them because their MTP
ISA coordinates are inconvenient.

Type 6 is origin-driven.

Write:

benchmark_metadata/type6_source_candidates.csv
benchmark_metadata/type6_source_selection.csv
benchmark_metadata/type6_source_selection.json
benchmark_metadata/TYPE6_SOURCES.md

For every selected source record:

type6 source index
MSLIB set
original filename/ID
release version
source checksum if practical
source activities
resources
skills
selected source indicators
selection stratum/reason
deterministic selection seed

======================================================================
SHARED FORWARD GENERATOR
========================

Reuse the shared MTP forward generator.

Extend it only where necessary to accept literature-derived empirical
distributions/profiles.

Do not create a second incompatible MTP generator if the shared one can
support the required source profile cleanly.

Type 6 generation:

MSLIB source instance
    ->
literature profile
    ->
GenerationParameters/source profile
    ->
shared forward MTP generator
    ->
quality gates
    ->
final instance

One selected MSLIB source configuration produces exactly one Type 6 MTP
instance.

Final target:

100 selected source configurations
->
100 accepted Type 6 MTP instances

Use deterministic retry seeds if generation fails a Type 6 quality gate.

Maximum generation attempts per source:
30

A retry must continue to use the SAME MSLIB source configuration.

Do not replace a generation failure with a different source merely to
reach 100.

Do not accept a "best invalid" candidate.

======================================================================
TYPE 6 QUALITY GATES
====================

Every accepted Type 6 instance must pass:

1. exact MTP structural validation;
2. exact MTP mentorship semantics;
3. one contributor per role and no contributor filling multiple roles
   in the same project;
4. no duplicate structural hash within Type 6;
5. no duplicate against:
   Google anchors,
   Type 2,
   Type 3,
   Type 4,
   Type 5;
6. source provenance complete;
7. source scale correspondence;
8. contributor-count correspondence;
9. project-count correspondence;
10. skill-count correspondence;
11. source-vs-generated skill-skew correspondence;
12. source-vs-generated skills-per-contributor correspondence;
13. source-vs-generated role-count distribution correspondence;
14. source-vs-generated duration distribution correspondence;
15. demand/supply correspondence;
16. required-level distribution correspondence;
17. initial direct + true mentorship feasibility must remain meaningful;
18. no unsupported source concepts represented as if they were observed
    MTP data;
19. no real resource names or source identifiers in final MTP file contents.

Use reasonable distribution tolerances based on sampling variability.

Do not weaken tolerances just because a source repeatedly fails.

If a specific source configuration cannot be represented by the shared
MTP generator while preserving the declared mapping, report the problem.

======================================================================
EXACT SOURCE-PRESERVATION REPORT
================================

For every generated instance record source vs generated values:

MSLIB_set
source_instance_id
source_activity_count
generated_project_count

source_resource_count
generated_contributor_count

source_skill_count
generated_skill_count

source_skills_per_resource_mean/sd
generated_skills_per_contributor_mean/sd

source_skill_level_mean/sd
generated_contributor_level_mean/sd

source_roles_per_activity_mean/sd
generated_roles_per_project_mean/sd

source_duration_mean/sd/median/P10/P90
generated_duration_mean/sd/median/P10/P90

source_skill_skew
generated_skill_skew

source_demand_supply_alignment
generated_demand_supply_alignment

source_contention
generated_contention

source_critical_path
source_network_depth
source_network/order indicators

generated_deadline_tightness
generated_schedule_pressure

source natural one-below potential
generated mentorship-oriented share
generated true same-project mentorship feasibility

Write this to Type 6 metadata.

======================================================================
PILOT
=====

Before generating all 100, generate a 10-instance pilot.

The pilot must contain source instances from all four sets:

MSLIB1
MSLIB2
MSLIB3
MSLIB4

and cover noticeably different:

- workforce sizes;
- project sizes;
- skill breadth;
- skill availability;
- network structures.

Do not take the first 10 files.

Use deterministic selection from the already frozen 100 source records.

Suggested pilot allocation:

2 from MSLIB1
2 from MSLIB2
3 from MSLIB3
3 from MSLIB4

or another deterministic balanced allocation.

Pilot must pass ALL Type 6 quality gates.

Write:

benchmark_metadata/type6_pilot/type6_pilot_report.json
benchmark_metadata/type6_pilot/type6_metadata.csv

Do not proceed to final generation if pilot fails.

======================================================================
FINAL GENERATION
================

If the pilot passes:

generate:

generated_instances/type6/type6_001__MSLIB1__srcXXXX.txt
...
generated_instances/type6/type6_100__MSLIB4__srcXXXX.txt

IDs must be contiguous.

Final filenames must use exactly:

type6_<family_id>__<MSLIB_set>__src<source_instance_id>.txt

The MSLIB set and source instance ID are intentionally present in the final
filename for scientific traceability. Keep original MSLIB source identity out
of the MTP file contents; preserve the complete original source filename and
source checksum in metadata.

Each Type 6 instance must have:

family_instance_id
source_dataset
source_instance_id
source_release
source_file
source_selection_seed
generation_seed
retry_attempt
structural_hash
generator_config_hash
source profile
generation parameters
all 31 benchmark features
quality-gate results

======================================================================
FAMILY REPORT
=============

Create:

benchmark_metadata/type6_metadata.csv
benchmark_metadata/type6_metadata.jsonl
benchmark_metadata/type6_statistical_report.json
benchmark_metadata/type6_data_provenance.json
benchmark_metadata/TYPE6_SOURCES.md
benchmark_metadata/type6_methodology_draft.md

The statistical report must include:

count = 100

count_per_MSLIB_set:
  MSLIB1 = 25
  MSLIB2 = 25
  MSLIB3 = 25
  MSLIB4 = 25

unique_structural_hashes = 100

validation_failures = 0

duplicates = 0

all_structural_valid
all_duplicate_pass
all_source_provenance_complete
all_literature_quality_pass

source/generator correspondence summaries

within-family pairwise ISA-feature distance

PCA/ISA diagnostic plots if the project already produces them

These diagnostics are POST-GENERATION evaluation only.

Never use final ISA coordinates to retroactively choose the source
instances.

======================================================================
METHODOLOGY DOCUMENTATION
=========================

Clearly separate three categories.

A. OBSERVED / SOURCE-NATIVE MSLIB INFORMATION

Examples:

- workforce size;
- source skills;
- source skill levels where available;
- source activity count;
- source role/skill requirements;
- activity durations;
- precedence network;
- source network/resource/skill indicators.

B. DERIVED FROM MSLIB

Examples:

- source skill skew;
- demand/supply alignment;
- critical path;
- slack;
- topological position;
- contention;
- natural one-level-below mentorship potential;
- contributor profile clusters.

C. SYNTHETIC MTP MODELING DECISIONS

Examples:

- MTP project score;
- translation of precedence structure into best-before/deadline pressure;
- MTP learning semantics;
- MTP mentorship semantics;
- any score scaling constant;
- retry-generation mechanics.

Never call Category C "observed from MSLIB."

======================================================================
IMPORTANT SCIENTIFIC LIMITATION
===============================

Document that Type 6 preserves workforce, skill, task-demand and temporal
structure from a published MSRCPSP benchmark, but MSRCPSP and MTP are
different optimization problems.

In particular:

- MSRCPSP precedence constraints are not preserved as hard MTP
  constraints;
- makespan is not an MTP objective;
- MTP scores are synthetic;
- MTP mentorship and skill learning are MTP-specific;
- therefore Type 6 is literature-grounded, not a mathematical
  equivalence transformation of MSRCPSP into MTP.

This limitation must appear in the methodology.

======================================================================
TESTS
=====

Add comprehensive tests.

At minimum:

test_msrcp_parser

test_dummy_start_end_handling

test_resource_to_contributor_mapping

test_skill_mapping

test_hierarchical_level_mapping

test_activity_role_multiplicity

test_duration_profile_preservation

test_precedence_dag_parsing

test_critical_path_calculation

test_precedence_not_written_to_mtp

test_score_proxy_positive_and_deterministic

test_same_project_mentorship_semantics

test_source_selection_deterministic

test_exactly_25_sources_per_mslib1_4

test_no_mslib5_in_final_type6

test_source_selection_independent_of_mtp_isa_features

test_generation_deterministic_seed_2026

test_source_provenance_complete

test_no_duplicate_source_selection

test_no_duplicate_final_instances

test_final_ids_contiguous

test_all_100_metadata_rows

test_no_source_names_in_final_mtp_files

Run the full benchmark-generation test suite.

======================================================================
CLI / EXECUTION PHASES
======================

Design the workflow so I can run it in gated phases similar to Type 5.

Phase 1:
Acquire/verify MSLIB v2.1.

Phase 2:
Parse all MSLIB1–4 sources and build the source feature table.

Phase 3:
Freeze the deterministic 25+25+25+25 source selection.

Phase 4:
Build the 100 literature-derived Type 6 source profiles.

Phase 5:
Generate and validate the 10-instance pilot.

Phase 6:
Generate final 100 Type 6 instances.

Phase 7:
Run tests and final reports.

Provide exact PowerShell commands for every phase.

Do not execute later phases automatically if an earlier phase fails.

======================================================================
FINAL ACCEPTANCE CONDITIONS
===========================

Type 6 is complete only if:

selected_source_count == 100

MSLIB1_selected == 25
MSLIB2_selected == 25
MSLIB3_selected == 25
MSLIB4_selected == 25

retained_source_profiles == 100

accepted_final_instances == 100

unique_structural_hashes == 100

all_structural_validation_pass == true

all_duplicate_checks_pass == true

all_type6_literature_quality_gates_pass == true

all_source_provenance_complete == true

all_metadata_complete == true

all_generation_seeds_recorded == true

no_source_resource_names_in_final_files == true

statistical_report_exists == true

source_selection_report_exists == true

methodology_matches_implementation == true

full_test_suite_passes == true

======================================================================
WHAT TO REPORT TO ME
====================

Start with Phase 1–3 only.

After implementing the parser and deterministic source selection, report:

- files created/changed;
- exact MSLIB release used;
- checksum;
- source format interpretation;
- whether hierarchical levels are available and how they are interpreted;
- total parsed instances per MSLIB set;
- source indicators available;
- selection algorithm;
- exact selected count per set;
- selected-source summary statistics;
- any licensing/provenance concerns;
- tests passed;
- exact commands for building the Type 6 profiles and pilot.

Do NOT jump directly to generating the final 100 until the source parsing
and 25+25+25+25 selection are validated.
