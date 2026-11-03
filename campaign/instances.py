"""Instance discovery and skill-universe measurement for the solver campaign.

The campaign benchmark is the 5 non-toy Google instances (b-f) plus the six
generated families type2..type7. Pilot folders and a_an_example are excluded.

    python campaign/instances.py            # list counts + max distinct skills
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GENERATED_FAMILIES = ("type2", "type3", "type4", "type5", "type6", "type7")


def discover(repo: Path = REPO) -> list[tuple[str, str, Path]]:
    """Return (family, instance_id, path), sorted deterministically."""
    out: list[tuple[str, str, Path]] = []
    for p in sorted((repo / "google_instances").glob("*.txt")):
        if p.stem.startswith("a_"):
            continue
        out.append(("google", p.stem, p))
    for fam in GENERATED_FAMILIES:
        for p in sorted((repo / "generated_instances" / fam).glob("*.txt")):
            out.append((fam, p.stem, p))
    return out


def instance_stats(path: Path) -> dict:
    """Counts one instance, including distinct skill names over contributors
    AND project roles (the solvers put both into one index space)."""
    tok = path.read_bytes().split()
    C, P = int(tok[0]), int(tok[1])
    i = 2
    skills: set[bytes] = set()
    for _ in range(C):
        n = int(tok[i + 1]); i += 2
        skills.update(tok[i:i + 2 * n:2]); i += 2 * n
    roles = total_score = 0
    for _ in range(P):
        S, r = int(tok[i + 2]), int(tok[i + 4]); i += 5
        skills.update(tok[i:i + 2 * r:2]); i += 2 * r
        roles += r; total_score += S
    if i != len(tok):
        raise ValueError(f"{path}: {len(tok) - i} trailing tokens")
    return {"contributors": C, "projects": P, "roles": roles,
            "skills": len(skills), "total_score": total_score,
            "bytes": path.stat().st_size}


def main() -> int:
    rows = [(fam, iid, instance_stats(p)) for fam, iid, p in discover()]
    fams: dict[str, int] = {}
    for fam, _, _ in rows:
        fams[fam] = fams.get(fam, 0) + 1
    print(f"instances: {len(rows)}  {fams}")
    rows.sort(key=lambda r: -r[2]["skills"])
    print("largest skill universes (skills, family/id, C, P, roles):")
    for fam, iid, s in rows[:5]:
        print(f"  {s['skills']:6d}  {fam}/{iid}  C={s['contributors']} P={s['projects']} R={s['roles']}")
    for key in ("contributors", "projects", "roles", "total_score", "bytes"):
        print(f"max {key}: {max(r[2][key] for r in rows)}")
    print(f"NS required = {rows[0][2]['skills']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
