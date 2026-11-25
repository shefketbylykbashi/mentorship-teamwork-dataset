#!/usr/bin/env python3
"""Compute the 58 candidate features of the MTP instances for Instance Space Analysis.

Reads every benchmark instance (google_instances/*.txt and
generated_instances/type*/*.txt), computes the 58 candidate features defined in
isa/FEATURES.md (direct instance parameters, distribution shape, tightness and
ratio features), and writes one CSV row per instance:

    Instances,Source,feature_<name>,...

Feature values are raw: no normalisation or standardisation is applied; MATILDA
does that in its PRELIM stage. Features use only the contents of the instance
file, never solver results, file names, family labels or generator parameters.

Instance IDs are the ones used in the campaign results:
    Google inputs  -> Source "google", Instances "google/<name>"
    Generated      -> Source "typeK",  Instances "typeK/<name>"

Usage:
    python3 isa/extract_features.py [--instances DIR] [--out FILE]

isa/build_metadata.py then adds the solver performance columns. Only the Python
standard library is used.
"""

import argparse
import csv
import math
import os
import statistics as st
import sys
from bisect import bisect_left
from collections import defaultdict

# ---------------------------------------------------------------------------
# Feature names, in output column order (without the "feature_" prefix).
# Numbering follows isa/FEATURES.md.
# ---------------------------------------------------------------------------

# Distributions summarised in Families 1 and 2.
DISTRIBUTIONS = [
    "duration",                # D_p over projects
    "score",                   # S_p over projects
    "best_before",             # B_p over projects
    "roles_per_project",       # R_p over projects
    "skills_per_contributor",  # N_c over contributors
    "contributor_level",       # levels over all (contributor, skill) entries
    "required_level",          # required levels over all roles
]

FAMILY_1 = [
    "n_contributors",
    "n_projects",
    "n_roles",
    "n_skills_held",
    "n_skills_required",
    "duration_min", "duration_mean", "duration_max",
    "score_min", "score_mean", "score_max",
    "best_before_min", "best_before_mean", "best_before_max",
    "roles_per_project_min", "roles_per_project_mean", "roles_per_project_max",
    "skills_per_contributor_min", "skills_per_contributor_mean",
    "skills_per_contributor_max",
    "contributor_level_min", "contributor_level_mean", "contributor_level_max",
    "required_level_mean", "required_level_max",
]

FAMILY_2 = [f"{d}_{s}" for d in DISTRIBUTIONS for s in ("cv", "skew")]

FAMILY_3 = [
    # Group A: time pressure
    "frac_late_from_start",
    "frac_first_wave_deadline",
    "slack_in_waves_median",
    "lateness_tolerance_median",
    # Group B: workload and team size
    "roles_per_contributor",
    "workload_ratio",
    "peak_deadline_load",
    "largest_team_share",
    "frac_single_role_projects",
    # Group C: project value
    "score_work_correlation",
    "value_density_cv",
    # Group D: skill supply, mentoring and learning
    "frac_roles_need_learning",
    "frac_projects_need_learning",
    "direct_candidate_share_mean",
    "frac_roles_single_candidate",
    "mentoring_share_mean",
    "learning_candidate_share_mean",
    "skill_load_mean",
    "skill_load_max",
]

FEATURES = FAMILY_1 + FAMILY_2 + FAMILY_3


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_instance(path):
    """Parse an MTP input file.

    Returns (contributors, projects), where
      contributors: list of [(skill, level), ...], one list per contributor,
                    exactly as listed in the file (N_c = its length)
      projects:     list of (D, S, B, roles), roles = list of (skill, level)
    Names are not needed by any feature and are discarded. Parsing is
    token-based, so it works for both naming styles in the benchmark.
    """
    with open(path) as f:
        tok = f.read().split()
    pos = 0

    def nxt():
        nonlocal pos
        pos += 1
        return tok[pos - 1]

    n_contributors, n_projects = int(nxt()), int(nxt())
    contributors = []
    for _ in range(n_contributors):
        nxt()                                   # contributor name
        contributors.append([(nxt(), int(nxt())) for _ in range(int(nxt()))])
    projects = []
    for _ in range(n_projects):
        nxt()                                   # project name
        d, s, b, r = int(nxt()), int(nxt()), int(nxt()), int(nxt())
        roles = [(nxt(), int(nxt())) for _ in range(r)]
        projects.append((d, s, b, roles))
    if pos != len(tok):
        raise ValueError(f"{path}: {len(tok) - pos} unread tokens at end of file")
    return contributors, projects


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def mean0(values):
    """Mean, or 0 for an empty list."""
    return st.fmean(values) if values else 0.0


def cv_and_skew(values):
    """Population coefficient of variation and skewness.

    Both are 0 when all values are identical (zero-variance convention in
    isa/FEATURES.md, Family 2).
    """
    mu = st.fmean(values)
    sd = st.pstdev(values, mu)
    if sd == 0:
        return 0.0, 0.0
    m3 = sum((x - mu) ** 3 for x in values) / len(values)
    return sd / mu, m3 / sd ** 3


def ranks(values):
    """Average ranks (ties share the mean of their positions)."""
    order = sorted(range(len(values)), key=values.__getitem__)
    r = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(a, b):
    """Spearman rank correlation; 0 if either input is constant."""
    ra, rb = ranks(a), ranks(b)
    if st.pstdev(ra) == 0 or st.pstdev(rb) == 0:
        return 0.0
    return st.correlation(ra, rb)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

def compute_features(contributors, projects):
    C, P = len(contributors), len(projects)
    D = [p[0] for p in projects]
    S = [p[1] for p in projects]
    B = [p[2] for p in projects]
    R = [len(p[3]) for p in projects]
    N = [len(c) for c in contributors]
    contributor_levels = [lvl for c in contributors for _, lvl in c]
    roles = [role for p in projects for role in p[3]]
    required_levels = [lvl for _, lvl in roles]
    n_roles = len(roles)

    # Level each contributor holds in each skill (highest if listed twice),
    # then levels held per skill (level >= 1 only), sorted for counting.
    held = defaultdict(list)
    for c in contributors:
        best = {}
        for skill, lvl in c:
            best[skill] = max(lvl, best.get(skill, 0))
        for skill, lvl in best.items():
            if lvl >= 1:
                held[skill].append(lvl)
    for v in held.values():
        v.sort()
    roles_in_skill = defaultdict(int)
    for skill, _ in roles:
        roles_in_skill[skill] += 1

    f = {}

    # ---- Family 1: direct instance parameters ----------------------------
    f["n_contributors"] = C
    f["n_projects"] = P
    f["n_roles"] = n_roles
    f["n_skills_held"] = len(held)
    f["n_skills_required"] = len(roles_in_skill)
    lists = {
        "duration": D,
        "score": S,
        "best_before": B,
        "roles_per_project": R,
        "skills_per_contributor": N,
        "contributor_level": contributor_levels,
        "required_level": required_levels,
    }
    for name in DISTRIBUTIONS:
        v = lists[name]
        for stat, fn in (("min", min), ("mean", st.fmean), ("max", max)):
            key = f"{name}_{stat}"
            if key in FAMILY_1:
                f[key] = fn(v)

    # ---- Family 2: distribution shape ------------------------------------
    for name in DISTRIBUTIONS:
        f[f"{name}_cv"], f[f"{name}_skew"] = cv_and_skew(lists[name])

    # ---- Family 3, group A: time pressure --------------------------------
    d_bar = st.fmean(D)                         # one "wave"
    f["frac_late_from_start"] = sum(d > b for d, b in zip(D, B)) / P
    f["frac_first_wave_deadline"] = sum(
        0 <= b - d < d_bar for d, b in zip(D, B)) / P
    f["slack_in_waves_median"] = st.median((b - d) / d_bar for d, b in zip(D, B))
    f["lateness_tolerance_median"] = st.median(s / d_bar for s in S)

    # ---- Family 3, group B: workload and team size -----------------------
    work = [d * r for d, r in zip(D, R)]        # person-days per project
    f["roles_per_contributor"] = n_roles / C
    f["workload_ratio"] = sum(work) / (C * max(B))
    peak, cumulative = 0.0, 0
    order = sorted(range(P), key=B.__getitem__)
    for idx, k in enumerate(order):
        cumulative += work[k]
        last_with_this_deadline = idx + 1 == P or B[order[idx + 1]] != B[k]
        if last_with_this_deadline:
            peak = max(peak, cumulative / (C * B[k]))
    f["peak_deadline_load"] = peak
    f["largest_team_share"] = max(R) / C
    f["frac_single_role_projects"] = sum(r == 1 for r in R) / P

    # ---- Family 3, group C: project value --------------------------------
    f["score_work_correlation"] = spearman(S, work)
    f["value_density_cv"] = cv_and_skew([s / w for s, w in zip(S, work)])[0]

    # ---- Family 3, group D: skill supply, mentoring and learning ---------
    top = {skill: (held[skill][-1] if held.get(skill) else 0)
           for skill in roles_in_skill}

    def count_at_least(skill, level):
        v = held.get(skill, [])
        return len(v) - bisect_left(v, level)

    def count_exactly(skill, level):
        if level == 0:                          # not listing a skill = level 0
            return C - len(held.get(skill, []))
        return count_at_least(skill, level) - count_at_least(skill, level + 1)

    need_learning = single_candidate = 0
    projects_need_learning = 0
    candidate_share, mentoring_share, learning_share, load = [], [], [], []
    for _, _, _, project_roles in projects:
        project_needs_learning = False
        for skill, level in project_roles:
            if level > top[skill]:
                need_learning += 1
                project_needs_learning = True
            d = count_at_least(skill, level)    # direct candidates
            if d == 1:
                single_candidate += 1
            if d >= 1:                          # role available on day 0
                m = count_exactly(skill, level - 1)   # mentee candidates
                e = count_exactly(skill, level)       # exact candidates
                candidate_share.append(d / C)
                mentoring_share.append(m / (d + m))
                learning_share.append((e + m) / (d + m))
            holders = len(held.get(skill, []))
            if holders:
                load.append(roles_in_skill[skill] / holders)
        projects_need_learning += project_needs_learning
    skill_loads = [roles_in_skill[s] / len(held[s])
                   for s in roles_in_skill if held.get(s)]

    f["frac_roles_need_learning"] = need_learning / n_roles
    f["frac_projects_need_learning"] = projects_need_learning / P
    f["direct_candidate_share_mean"] = mean0(candidate_share)
    f["frac_roles_single_candidate"] = single_candidate / n_roles
    f["mentoring_share_mean"] = mean0(mentoring_share)
    f["learning_candidate_share_mean"] = mean0(learning_share)
    f["skill_load_mean"] = mean0(load)
    f["skill_load_max"] = max(skill_loads) if skill_loads else 0.0

    missing = set(FEATURES) - set(f)
    if missing:
        raise RuntimeError(f"features not computed: {sorted(missing)}")
    return f


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def instance_id(filename):
    """Map a file name to (Instances, Source), as in the campaign results."""
    stem = filename[:-len(".txt")]
    source = stem.split("_", 1)[0] if stem.startswith("type") else "google"
    return f"{source}/{stem}", source


def fmt(value):
    """Integers as integers; floats with full precision."""
    if isinstance(value, int):
        return str(value)
    if not math.isfinite(value):
        raise ValueError(f"non-finite feature value: {value}")
    return repr(float(value))


def find_instances(folder=None):
    """Instance files, sorted by name: one flat folder, or the repository layout."""
    if folder:
        paths = [os.path.join(folder, f) for f in os.listdir(folder)]
    else:
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        folders = [os.path.join(repo, "google_instances")]
        generated = os.path.join(repo, "generated_instances")
        folders += [os.path.join(generated, d) for d in sorted(os.listdir(generated))]
        paths = [os.path.join(d, f) for d in folders if os.path.isdir(d) for f in os.listdir(d)]
    return sorted((p for p in paths if p.endswith(".txt")), key=os.path.basename)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--instances", default=None,
                        help="flat folder of instance .txt files (default: google_instances/ "
                             "and generated_instances/type*/ of this repository)")
    parser.add_argument("--out", default=os.path.join(here, "features.csv"),
                        help="output CSV (default: isa/features.csv)")
    args = parser.parse_args()

    files = find_instances(args.instances)
    if not files:
        sys.exit("no .txt instances found")

    rows = []
    for i, path in enumerate(files, 1):
        name = os.path.basename(path)
        contributors, projects = parse_instance(path)
        feats = compute_features(contributors, projects)
        inst, source = instance_id(name)
        rows.append([inst, source] + [fmt(feats[k]) for k in FEATURES])
        print(f"[{i}/{len(files)}] {name}", file=sys.stderr)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["Instances", "Source"] + [f"feature_{k}" for k in FEATURES])
        writer.writerows(rows)
    print(f"wrote {len(rows)} instances x {len(FEATURES)} features to {args.out}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
