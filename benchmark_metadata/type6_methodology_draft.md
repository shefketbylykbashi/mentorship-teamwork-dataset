# Type 6 Methodology Draft

Type 6 is a literature-grounded synthetic MTP family derived from published MSRCPSP benchmark configurations.

## Source Artifact

MSLIB official release asset MSLIB_v2_2.zip, retrieved from GitHub release v2.2, SHA-256 f1672d2e2e85091f598e497b898b20c468b0b9a10030c5ca8d8ae798b6529e8c, with upstream documentation still describing the dataset release as v2.1 (June 2026).

Recorded upstream inconsistency:

- GitHub release/tag observed: v2.2
- Release asset: MSLIB_v2_2.zip
- SHA-256: f1672d2e2e85091f598e497b898b20c468b0b9a10030c5ca8d8ae798b6529e8c
- Release body states: This is release v2.1
- Repository README states: Last update: v2.1 (June 2026)
- Repository documentation states that previous releases are removed when a new release is published and recommends using the latest release.
- Internal archive Readme.rtf observed: Last update: v2.2 (June 2026)
- Dataset information files found: {"MSLIB1": [], "MSLIB2": [], "MSLIB3": [], "MSLIB4": [], "MSLIB5": ["MSLIB5/MSLIB_INFO_FILE.rtf"]}

The tiny v2.1 GitHub tag archive is retained only as provenance evidence and is not used as source data.

## Observed / Source-Native MSLIB Information

- workforce size, skill count, and skill-level matrices
- real activity count, durations, skill-demand multiplicities, and required levels
- precedence arcs and dummy start/end activities
- workbook network/resource/skill indicators including CNC, OS, SP, SF, SS, RA and alpha variants
- BKS and deadline fields as provenance/diagnostics only

## Derived From MSLIB

- critical path, slack, topological position, network depth, order strength, and arc density
- demand/supply alignment, concentration, contention, and source skill skew
- natural direct eligibility and one-level-below mentorship potential under hierarchical levels

## Synthetic MTP Modeling Decisions

No Type 6 MTP instances are generated in Phases 1-3. Later phases will use these frozen source records to parameterize the shared forward MTP generator.
MSRCPSP precedence constraints will not be written as MTP constraints, and MTP scores will be literature-structure-conditioned synthetic proxies.

## Source Selection

{
  "categorical_columns": [
    "n_real_activities",
    "n_resources",
    "n_skills",
    "n_skill_levels",
    "excel_SP"
  ],
  "feature_columns": [
    "n_activities_total",
    "n_real_activities",
    "n_resources",
    "n_skills",
    "n_skill_levels",
    "n_precedence_arcs",
    "arc_density",
    "order_strength",
    "critical_path_length",
    "network_depth",
    "duration_mean",
    "duration_sd",
    "duration_p10",
    "duration_p90",
    "total_roles",
    "roles_per_activity_mean",
    "roles_per_activity_sd",
    "skills_per_resource_mean",
    "skills_per_resource_sd",
    "skill_level_mean",
    "requirement_level_mean",
    "demand_concentration",
    "supply_concentration",
    "demand_supply_alignment",
    "source_skill_skew",
    "source_contention",
    "qualified_contention",
    "natural_direct_share",
    "natural_one_below_share",
    "natural_mentor_available_share",
    "excel_CNC",
    "excel_OS",
    "excel_SP",
    "excel_SF",
    "excel_SF_alpha",
    "excel_SS",
    "excel_SS_alpha",
    "excel_RA",
    "excel_RA_alpha"
  ],
  "method": "deterministic source-space stratified maximin",
  "selection_seed": 2026,
  "steps": [
    "parse every canonical MSLIB1-4 .msrcp source using only source-native data",
    "join official workbook network/resource/skill indicators where canonical rows exist",
    "standardize configured source indicators within each MSLIB set, with log1p on scale variables",
    "seed selection with medoids of the most frequent categorical source-parameter groups",
    "fill remaining slots by farthest-first maximin distance in standardized source space",
    "freeze dataset-major contiguous type6_source_index values"
  ]
}
## Phase 4 Literature Profiles

Phase 4 reparses each frozen selected MSLIB source file and derives one source profile per selected instance.
Generation parameters preserve source resource, skill, and non-dummy activity counts exactly.
Empirical distributions are retained for skills per resource, source skill levels, role multiplicities, required levels, activity durations, temporal pressure, and natural mentorship potential.
Precedence arcs are not emitted as MTP constraints; only critical path, depth, slack, and topological-position summaries are retained for later temporal-pressure modeling.
MTP score fields remain synthetic and are parameterized from source effort/scarcity/criticality summaries, never from BKS, LB, UB, Open?, or algorithmic results.
A source sanity check confirms that all canonical MSLIB1-4 instances use exactly four skill types in both the workbook # Skills field and parsed .msrcp headers; Type 6 treats this as a literature-source characteristic and limitation.

Profile quality gate:

{
  "all_generation_parameters_without_bks": true,
  "all_source_provenance_complete": true,
  "count_per_MSLIB_set": {
    "MSLIB1": 25,
    "MSLIB2": 25,
    "MSLIB3": 25,
    "MSLIB4": 25
  },
  "dummy_activities_excluded_from_project_count": true,
  "exactly_25_per_mslib1_4": true,
  "no_mslib5_profiles": true,
  "passed": true,
  "precedence_not_generation_constraint": true,
  "profile_count": 100,
  "profile_count_exact": true,
  "source_resource_count_preserved": true,
  "source_skill_count_preserved": true
}

## Phase 6 Final Generation

Final Type 6 output filenames use `type6_<family_id>__<MSLIB_set>__src<source_instance_id>.txt`.
Source identifiers may appear in final filenames and metadata for scientific traceability.
Source identifiers must not appear inside final MTP file contents, which use only anonymous `c00000`, `s0000`, and `p00000` identifiers.

The final family preserves the frozen 25+25+25+25 MSLIB source selection exactly; no source is replaced based on generation difficulty or final ISA position.

All final Type 6 instances have n_skills = 4 because every canonical MSLIB1-4 source instance in the pinned archive has four source skills. This fixed skill cardinality is a Type 6 source-family characteristic and limitation, not a shared MTP generator restriction.

MTP scores are literature-structure-conditioned synthetic proxies. The final score construction is `score = round(K * effort * scarcity * criticality)`, where `K = 10.0`; effort is `duration * max(1, role_count)`, scarcity is derived from directly qualified source workforce availability, and criticality is derived from independently computed source slack and critical-path diagnostics.
BKS, LB, UB, Open?, workbook CP, solver performance, optimality gaps, and other solution information are not generation, score, selection, or acceptance inputs.

MSLIB precedence arcs are not emitted as MTP constraints. They are retained as source metadata and influence only derived temporal-pressure diagnostics such as critical path, slack, topological position, network depth, and best-before calibration.

The source artifact remains identified as MSLIB official release asset `MSLIB_v2_2.zip`, retrieved from GitHub release `v2.2`, SHA-256 `f1672d2e2e85091f598e497b898b20c468b0b9a10030c5ca8d8ae798b6529e8c`. Upstream documentation contains conflicting labels: the release body states `This is release v2.1`, the repository README states `Last update: v2.1 (June 2026)`, and the internal archive README states `Last update: v2.2 (June 2026)`.
