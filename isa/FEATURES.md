# Candidate features for the Instance Space Analysis

The 58 candidate features of an MTP instance, computed by `extract_features.py` from the instance file alone: no solver is run, and file names, family labels and generator parameters are not used. The column in `metadata.csv` is `feature_<name>`. Values are raw; MATILDA normalises them (Box-Cox and z-transformation) in its PRELIM stage and selects among them in SIFTED.

| Family | Content | Features |
|---|---|---|
| 1 | Direct instance parameters | 25 |
| 2 | Distribution shape | 14 |
| 3 | Tightness and ratio features | 19 |
| | **Total** | **58** |

Notation follows the input format. An instance has `C` contributors and `P` projects. Contributor *c* lists `N_c` skills, each with a level. Project *p* has duration `D_p`, score `S_p`, best-before day `B_p` and `R_p` roles, and each role requires one skill at one level. A skill that a contributor does not list counts as level 0.

Minimum, median and maximum are taken over the 131 benchmark instances.

## Family 1: Direct instance parameters

Read straight from the input file. Each list-valued parameter is summarised by its minimum, mean and maximum. The contributor levels and the required levels are pooled over all entries, not averaged per contributor or per project first.

| # | Feature | Definition | Min | Median | Max |
|---|---|---|---|---|---|
| 1.1 | `n_contributors` | C, the number of contributors | 8 | 61 | 8,037 |
| 1.2 | `n_projects` | P, the number of projects | 5 | 1,273 | 19,413 |
| 1.3 | `n_roles` | Σ_p R_p, the total number of roles | 30 | 7,302 | 573,446 |
| 1.4 | `n_skills_held` | distinct skills held at level ≥ 1 by some contributor | 4 | 14 | 800 |
| 1.5 | `n_skills_required` | distinct skills required by some role | 4 | 14 | 800 |
| 1.6 | `duration_min` | min of D_p over projects | 1 | 1 | 3,024 |
| 1.7 | `duration_mean` | mean of D_p over projects | 1 | 11 | 5,984 |
| 1.8 | `duration_max` | max of D_p over projects | 1 | 27 | 9,000 |
| 1.9 | `score_min` | min of S_p over projects | 1 | 101 | 12,601 |
| 1.10 | `score_mean` | mean of S_p over projects | 10.5 | 1,039 | 65,397 |
| 1.11 | `score_max` | max of S_p over projects | 28 | 2,788 | 219,597 |
| 1.12 | `best_before_min` | min of B_p over projects | 1 | 2 | 3,229 |
| 1.13 | `best_before_mean` | mean of B_p over projects | 3.71 | 35.4 | 629,500 |
| 1.14 | `best_before_max` | max of B_p over projects | 6 | 200 | 2,430,759 |
| 1.15 | `roles_per_project_min` | min of R_p over projects | 1 | 2 | 20 |
| 1.16 | `roles_per_project_mean` | mean of R_p over projects | 1 | 7.43 | 39.3 |
| 1.17 | `roles_per_project_max` | max of R_p over projects | 1 | 12 | 59 |
| 1.18 | `skills_per_contributor_min` | min of N_c over contributors | 1 | 1 | 18 |
| 1.19 | `skills_per_contributor_mean` | mean of N_c over contributors | 1 | 2.39 | 18.5 |
| 1.20 | `skills_per_contributor_max` | max of N_c over contributors | 1 | 4 | 20 |
| 1.21 | `contributor_level_min` | min of the level over all (contributor, skill) entries | 1 | 1 | 10 |
| 1.22 | `contributor_level_mean` | mean of the level over all (contributor, skill) entries | 2.49 | 4.79 | 10 |
| 1.23 | `contributor_level_max` | max of the level over all (contributor, skill) entries | 4 | 10 | 10 |
| 1.24 | `required_level_mean` | mean of the required level over all roles | 2.72 | 5.69 | 10.1 |
| 1.25 | `required_level_max` | max of the required level over all roles | 5 | 11 | 20 |

## Family 2: Distribution shape

One spread measure and one shape measure for seven distributions: the six lists of Family 1 and the required levels. For a list x₁ … xₙ with mean μ and population standard deviation σ, `cv` = σ / μ and `skew` = ((1/n) Σ (xᵢ − μ)³) / σ³. When every value in the list is identical (σ = 0), both are set to 0. This records a real property of the instance, for example that every contributor has the same number of skills.

| # | Feature | Definition | Min | Median | Max |
|---|---|---|---|---|---|
| 2.1 | `duration_cv` | coefficient of variation of D_p over projects | 0 | 0.504 | 1.58 |
| 2.2 | `duration_skew` | skewness of D_p over projects | −0.445 | 0.020 | 3.13 |
| 2.3 | `score_cv` | coefficient of variation of S_p over projects | 0.172 | 0.717 | 2.13 |
| 2.4 | `score_skew` | skewness of S_p over projects | −0.435 | 1.06 | 3.07 |
| 2.5 | `best_before_cv` | coefficient of variation of B_p over projects | 0.045 | 0.573 | 1.82 |
| 2.6 | `best_before_skew` | skewness of B_p over projects | −0.084 | 0.525 | 2.77 |
| 2.7 | `roles_per_project_cv` | coefficient of variation of R_p over projects | 0 | 0.368 | 0.741 |
| 2.8 | `roles_per_project_skew` | skewness of R_p over projects | −0.838 | −0.001 | 1.41 |
| 2.9 | `skills_per_contributor_cv` | coefficient of variation of N_c over contributors | 0 | 0.326 | 0.539 |
| 2.10 | `skills_per_contributor_skew` | skewness of N_c over contributors | −38.7 | 0 | 5.20 |
| 2.11 | `contributor_level_cv` | coefficient of variation of the level over all (contributor, skill) entries | 0 | 0.445 | 0.693 |
| 2.12 | `contributor_level_skew` | skewness of the level over all (contributor, skill) entries | −1.52 | 0.030 | 0.670 |
| 2.13 | `required_level_cv` | coefficient of variation of the required level over all roles | 0.136 | 0.420 | 0.633 |
| 2.14 | `required_level_skew` | skewness of the required level over all roles | −2.55 | −0.228 | 0.870 |

## Family 3: Tightness and ratio features

These combine parameters according to the MTP rules, so that each measures one source of difficulty. They are computed by counting and dividing values in the input file.

- **Wave.** D̄ is the mean of D_p over all projects. A contributor works on one project at a time, so waiting for one other project costs about D̄ days, called one wave.
- **Candidates for a role** requiring skill s at level L: the direct candidates d are the contributors with s at level ≥ L; the mentee candidates m are those with s at exactly L − 1 (a contributor without the skill counts as level 0); the exact candidates e are those with s at exactly L. `holders(s)` are the contributors with s at level ≥ 1.
- **Roles that need learning.** M(s) is the highest level of s held by anyone at the start. A role is available if L ≤ M(s); otherwise it needs learning, because a mentor must already hold the required level.
- The counts are made role by role and ignore competition between roles.

| # | Feature | Group | Definition | Min | Median | Max |
|---|---|---|---|---|---|---|
| 3.1 | `frac_late_from_start` | Time pressure | fraction of projects with D_p > B_p: late even if started on day 0 | 0 | 0 | 0.408 |
| 3.2 | `frac_first_wave_deadline` | Time pressure | fraction of projects with 0 ≤ B_p − D_p < D̄: on time from day 0, but late after waiting one wave | 0 | 0.476 | 0.967 |
| 3.3 | `slack_in_waves_median` | Time pressure | median over projects of (B_p − D_p) / D̄ | 0.048 | 1.05 | 487 |
| 3.4 | `lateness_tolerance_median` | Time pressure | median over projects of S_p / D̄: the waves a project can overrun before it is worth nothing | 0.207 | 28.3 | 731 |
| 3.5 | `roles_per_contributor` | Workload and team size | Σ_p R_p / C | 0.095 | 25.7 | 573 |
| 3.6 | `workload_ratio` | Workload and team size | Σ_p D_p R_p / (C × max_p B_p): person-days needed over person-days available before the last deadline | 0.019 | 3.31 | 81.4 |
| 3.7 | `peak_deadline_load` | Workload and team size | max over deadline values t of Σ_{q: B_q ≤ t} D_q R_q / (C × t): the same ratio for the tightest deadline window | 0.019 | 4.41 | 86.9 |
| 3.8 | `largest_team_share` | Workload and team size | max_p R_p / C | 0.001 | 0.071 | 1 |
| 3.9 | `frac_single_role_projects` | Workload and team size | fraction of projects with R_p = 1, in which mentoring is impossible | 0 | 0 | 1 |
| 3.10 | `score_work_correlation` | Project value | Spearman correlation across projects between S_p and D_p × R_p (0 if either is constant) | 0.236 | 0.940 | 0.997 |
| 3.11 | `value_density_cv` | Project value | coefficient of variation of S_p / (D_p × R_p), the score per person-day | 0.022 | 0.281 | 1.47 |
| 3.12 | `frac_roles_need_learning` | Skills, mentoring and learning | fraction of roles that need learning (L > M(s)) | 0 | 0.078 | 0.782 |
| 3.13 | `frac_projects_need_learning` | Skills, mentoring and learning | fraction of projects with at least one role that needs learning | 0 | 0.174 | 1.000 |
| 3.14 | `direct_candidate_share_mean` | Skills, mentoring and learning | mean over available roles of d / C | 0.001 | 0.151 | 0.539 |
| 3.15 | `frac_roles_single_candidate` | Skills, mentoring and learning | fraction of roles with exactly one direct candidate (d = 1) | 0 | 0.081 | 1 |
| 3.16 | `mentoring_share_mean` | Skills, mentoring and learning | mean over available roles of m / (d + m): candidates who can take the role only as mentees | 0.072 | 0.303 | 0.771 |
| 3.17 | `learning_candidate_share_mean` | Skills, mentoring and learning | mean over available roles of (e + m) / (d + m): candidates who would gain a level by taking the role | 0.363 | 0.659 | 0.961 |
| 3.18 | `skill_load_mean` | Skills, mentoring and learning | mean over roles of [roles requiring the role's skill] / holders(s) | 0.038 | 12.4 | 478 |
| 3.19 | `skill_load_max` | Skills, mentoring and learning | max over required skills s of [roles requiring s] / holders(s) | 0.091 | 15.5 | 1,323 |
