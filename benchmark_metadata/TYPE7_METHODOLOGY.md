# Type 7 Methodology: Remaining Instance-Space Gap Filling

Type 7 is a coverage-driven synthetic MTP family. It is not a new empirical
source and it is not a solver-performance family. Its purpose is to fill
reachable regions of the canonical MTP feature space that remain sparse after
the frozen Types 1-6 benchmark has been accepted.

## Frozen Benchmark Audit

The frozen archive \(B\) is built from the actual files present in the
repository at generation time:

- Type 1 is the five non-toy Google Hash Code instances in `google_instances/`.
- Types 2-6 are the `.txt` files present in `generated_instances/type2` through
  `generated_instances/type6`.

The toy Google file `a_an_example.txt` is excluded. Historical documentation is
not used to infer family counts. For each frozen file the audit records
family, family instance id, filename, SHA-256, structural hash, validation
status, metadata source where available, and the canonical feature vector.
SHA-256 hashes are checked again after Type 7 generation to prove Types 1-6
were not modified.

The audit uses structural validation for existing files. The official Google
anchor `c_collaboration` is valid but does not satisfy the synthetic default
fillable-fraction threshold of 0.30, so the Type 7 archive audit records
fillability metrics while using an existing-file fillability threshold of 0.0.
Generated Type 7 candidates still use the configured `fillable_fraction` gate.

## Canonical Feature Space

Type 7 uses the existing solver-independent canonical feature extractor in
`benchmark_generation.core.features`. The selection feature set is exactly the
canonical benchmark feature list:

\[
F = F_{\text{size}} \cup F_{\text{supply-demand}} \cup F_{\text{levels}}
    \cup F_{\text{mentorship}} \cup F_{\text{time}} \cup F_{\text{scores}}.
\]

No family labels, filenames, seeds, hashes, repository identifiers, source
ids, solver outputs, runtimes, best-known-solution values, or post-solution
measurements participate in Type 7 selection. At least one mentorship feature
from the canonical mentorship group is included.

Raw count and scale-like features use the existing `log1p` transform from
`benchmark_generation.core.distance.LOG_TRANSFORM_FEATURES`; all remaining
features use identity transforms. A z-score scaler is fit once using:

1. the frozen Type 1-6 feature vectors; and
2. the deterministic feasibility-calibration cloud.

The scaler is frozen before any Type 7 selection and is not refit using final
Type 7 instances.

Distances use the existing group-balanced metric:

\[
d(x,y) =
\sqrt{\sum_j w_j (z_{xj} - z_{yj})^2},
\]

where each conceptual feature group receives its configured aggregate weight
and that weight is divided equally among features in the group. Constant
dimensions receive scale 1.0. Non-finite canonical feature values are mapped to
0.0 before transformation, matching the core distance implementation.

## Feasibility Calibration

Type 7 fills feasible gaps, not arbitrary remote feature-space points. A
deterministic calibration cloud is generated using the shared forward MTP
generator and validated parameter ranges inherited from the Type 4 broad
synthetic family. The default Type 7 configuration uses:

- root seed `2026`;
- symmetric Dirichlet sampling for `p_over`, `p_match`, `p_mentor`, and
  `p_future`;
- Type 4 small/medium/large scale definitions;
- configured scale-band sampling weights;
- a configured valid-candidate calibration size.

For the frozen run in this repository, the search budget is configured as 400
broad valid calibration candidates plus two adaptive rounds of 150 valid
candidates each. The independent evaluation cloud contains 500 valid
candidates. The Type 7 large search band is the Type 4 large interval restricted
to 500-2,000 contributors/projects for computational tractability; this is a
documented subset of the already validated Type 4 scale support, not an
expanded or unsupported extreme range.

Only compact candidate records are stored: seed, parameters, parameter hash,
features, validation status, structural hash, and search lineage. Rejected
candidates are summarized by reason.

Plausibility bounds are derived deterministically from the normalized
calibration cloud. The default uses the observed calibration min and max per
feature plus a configurable margin. Candidates outside this empirically
reachable envelope are rejected during adaptive intensification.

## Gap Analysis

Let \(B\) be all frozen Type 1-6 instances in the frozen normalized feature
space. For every calibration candidate \(x\), Type 7 computes:

\[
\text{gap\_score}(x) = \min_{b \in B} d(x,b).
\]

The largest gap scores identify under-covered reachable regions. The gap
analysis also records existing nearest-neighbor statistics, 5-nearest-neighbor
density summaries, nearest-family composition, and the feature deviations of
major gap regions.

## Forward-Only Search

Type 7 never constructs an instance by inverting a target feature vector. Every
candidate follows:

```text
GenerationParameters
  -> shared forward MTP generator
  -> complete valid MTP instance
  -> canonical feature extraction
  -> coverage evaluation
```

The search has four stages:

1. Broad exploration: generate a deterministic candidate bank with stratified
   scale-band sampling.
2. Gap-directed intensification: repeatedly choose elite high-gap parameter
   configurations and mutate `GenerationParameters` within configured bounds.
3. Random restart: retain a configured fraction of globally sampled candidates
   in each adaptive round.
4. Greedy sequential maximin selection: choose final candidates one at a time.

For accepted Type 7 set \(T\), candidate novelty is:

\[
\text{novelty}(x) = \min_{y \in B \cup T} d(x,y).
\]

After each acceptance, the selected instance is inserted into \(T\), so later
acceptances cannot all fill the same gap.

## Pilot Gate

The first 10 selected candidates are written as the Type 7 pilot. The pilot
uses the same frozen scaler, candidate archive, and maximin search used for the
final family. It passes only if all pilot instances are structurally valid,
unique, forward-generated, metadata-complete, non-degenerate, and backed by
recorded seeds and search lineage. If the pilot passes, the first 10 final
Type 7 instances reproduce the pilot candidates deterministically.

## Mentorship Semantics

The canonical selection features are reused unchanged. Type 7 additionally
records exact same-project mentorship diagnostics for accepted candidates.
A contributor one level below a role may satisfy the role only if another
distinct contributor assigned to another role of the same project has the
required skill at or above the required level. Candidate assignments used for
these diagnostics are validated with
`benchmark_generation.core.mentorship.validate_project_assignment`.

Separate Type 7 metrics are recorded for direct fillability, one-level-below
potential, global mentor availability, and true same-project mentorship
feasibility.

## Degeneracy and Duplicate Controls

Candidates are rejected if they duplicate any frozen Types 1-6 structural hash,
duplicate an already generated Type 7 candidate, fail the configured minimum
initial fillability gate, contain too little meaningful skill structure, or
show configured pathological zero-variance project distributions. These checks
are recorded in metadata and rejection summaries.

The structural hash excludes filenames, seeds, metadata, and family labels.

## Independent Coverage Evaluation

Coverage improvement is evaluated on a second valid generator cloud using
`evaluation_seed = 2027` by default. This cloud is generated after selection,
with a different seed, and is never used to choose Type 7 instances.

For each evaluation point \(e\), the report computes nearest-neighbor distance
before and after Type 7:

\[
d_{\text{before}}(e) = \min_{b \in B} d(e,b),
\]

\[
d_{\text{after}}(e) = \min_{y \in B \cup T} d(e,y).
\]

The main Type 7 quality evidence is that the mean and P95 nearest-neighbor
distances decrease after adding Type 7.
