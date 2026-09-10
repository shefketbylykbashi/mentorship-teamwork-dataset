# Type 5 Full-Project Extension Methodology Draft

This document describes the opt-in extension of Type 5 with complete eligible GitHub project histories.
The accepted 91 fixed six-month Type 5 instances remain immutable and are not regenerated.

## Design

Each new source repository contributes at most one empirical profile. The collector observes the repository's complete eligible public technical work history, from the first retained non-bot work unit through the latest retained work unit at the acquisition snapshot. No fixed six-month windows are used in this extension.

Repositories are eligible only when the measured project history is between 90 and 548 days, inclusive. Mature repositories are rejected rather than truncated to fit this interval.

## Source Selection

Selection is source-driven and occurs before MTP generation. Candidate repositories are filtered by repository status, overlap with the frozen Type 5 source list, lifetime, non-bot activity, retained technical skills, and contributor-profile quality. When more valid sources are available than the configured extension count, deterministic farthest-first selection is applied over normalized source-native metrics such as lifetime, contributor count, work-unit count, skill count, skill skew, demand/supply alignment, contention, and duration. Solver scores, ISA positions, and generated MTP performance are not used.

## Empirical Interpretation

Human GitHub contributors are treated as anonymized contributor proxies. Merged pull requests are treated as work-unit proxies. Technical language/subsystem tokens derived from changed file paths are treated as skill proxies. Activity-percentile levels are relative empirical proxies, not literal human competence measurements. Scores and best-before values remain synthetic or derived proxies, using the existing Type 5 methodology.

## Generation

For each selected source, the empirical profile parameterizes the shared forward MTP generator already used by Type 5. The generated text file contains only anonymous cXXXXX contributors, sXXXX skills, and pXXXXX projects. Repository identifiers, usernames, PR numbers, source paths, and semantic skill tokens are excluded from the MTP text and stored only in metadata.

## Quality Gates

The extension requires structural MTP validity, exact mentorship semantics, no structural duplicates against existing benchmark families, source/generation correspondence, complete source provenance, repository uniqueness, zero overlap with existing Type 5 repositories, contiguous output IDs type5_092 through type5_108, and byte-for-byte preservation of type5_001 through type5_091.

Selected full-project sources in the current selection file: 17
