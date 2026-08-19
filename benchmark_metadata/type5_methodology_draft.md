# Type 5 Methodology Draft

Type 5 grounds the synthetic benchmark in empirically observed software-development configurations.
One hundred primary repository-window configurations were initially evaluated from mature open-source repositories split into four fixed six-month UTC observation windows.
After predefined source-quality filtering and deterministic reserve replacement, 91 empirical source configurations were retained.
No quality thresholds were weakened to force an arbitrary target count.
Merged pull requests are treated as empirical work-package proxies, not literal MTP projects.

## Empirically Observed

- merged PR counts and sampled PR file paths
- active public contributor activity after bot filtering
- changed technical areas mapped to language/subsystem skill tokens
- PR cycle times
- repository language composition
- release and milestone dates where available

## Derived Proxies

- contributor activity scores use 0.60 log-lines, 0.25 log-files, and 0.15 log-PR counts
- skill levels are relative empirical activity-based skill-level proxies on the MTP 1..10 scale
- merged PRs are work-package proxies
- deadline tightness uses milestone due dates, then stable releases, then a corpus-level fallback

## Synthetic Modeling Decisions

- MTP scores are effort-conditioned synthetic proxies, not observed project value
- Type 5 difficulty probabilities are sampled from a Dirichlet prior centered on [0.25, 0.45, 0.20, 0.10]
- source profiles parameterize the same forward generator used by other synthetic families
- final instances are accepted only after structural, duplicate, provenance, and realism gates pass

Global score normalization K: 1.8204784532536746
Corpus deadline fallback P10/P90: 0.5, 6.0
