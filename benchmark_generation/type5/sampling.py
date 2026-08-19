"""Deterministic PR sampling for Type 5 detailed file analysis."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Sequence


def pr_author(pr: Dict[str, Any]) -> str:
    user = pr.get("user") or {}
    if isinstance(user, dict):
        return str(user.get("login") or "unknown")
    return str(pr.get("author_login") or "unknown")


def pr_merged_at(pr: Dict[str, Any]) -> str:
    return str(pr.get("merged_at") or "")


def deterministic_pr_sample(
    prs: Sequence[Dict[str, Any]],
    seed: int,
    sample_size: int = 60,
    monthly_target: int = 10,
    max_author_share: float = 0.20,
) -> List[Dict[str, Any]]:
    """Return a deterministic time-stratified PR sample.

    If the window has at most ``sample_size`` merged PRs, all are retained.
    Larger windows are sampled approximately by month while limiting author
    domination where the data permits.
    """

    ordered = sorted(prs, key=lambda item: (pr_merged_at(item), int(item.get("number", 0))))
    if len(ordered) <= sample_size:
        return list(ordered)

    rng = random.Random(seed)
    by_month: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for pr in ordered:
        month = pr_merged_at(pr)[:7]
        by_month[month].append(pr)
    for month_prs in by_month.values():
        month_prs.sort(key=lambda item: (pr_merged_at(item), int(item.get("number", 0))))
        rng.shuffle(month_prs)

    author_counts: Counter = Counter()
    selected: List[Dict[str, Any]] = []
    author_limit = max(1, int(sample_size * max_author_share))

    for month in sorted(by_month):
        for pr in by_month[month][:monthly_target]:
            if len(selected) >= sample_size:
                break
            author = pr_author(pr)
            if author_counts[author] >= author_limit and _has_alternative(
                by_month[month],
                author_counts,
                author_limit,
                selected,
            ):
                continue
            selected.append(pr)
            author_counts[author] += 1
        if len(selected) >= sample_size:
            break

    if len(selected) < sample_size:
        selected_numbers = {int(pr.get("number", 0)) for pr in selected}
        for pr in ordered:
            if len(selected) >= sample_size:
                break
            number = int(pr.get("number", 0))
            if number in selected_numbers:
                continue
            author = pr_author(pr)
            if author_counts[author] >= author_limit and _remaining_has_alternative(
                ordered,
                selected_numbers,
                author_counts,
                author_limit,
            ):
                continue
            selected.append(pr)
            selected_numbers.add(number)
            author_counts[author] += 1

    if len(selected) < sample_size:
        selected_numbers = {int(pr.get("number", 0)) for pr in selected}
        for pr in ordered:
            if len(selected) >= sample_size:
                break
            number = int(pr.get("number", 0))
            if number not in selected_numbers:
                selected.append(pr)
                selected_numbers.add(number)

    return sorted(selected[:sample_size], key=lambda item: (pr_merged_at(item), int(item.get("number", 0))))


def _has_alternative(
    candidates: Sequence[Dict[str, Any]],
    author_counts: Counter,
    author_limit: int,
    selected: Sequence[Dict[str, Any]],
) -> bool:
    selected_numbers = {int(pr.get("number", 0)) for pr in selected}
    return any(
        int(pr.get("number", 0)) not in selected_numbers
        and author_counts[pr_author(pr)] < author_limit
        for pr in candidates
    )


def _remaining_has_alternative(
    candidates: Sequence[Dict[str, Any]],
    selected_numbers: set[int],
    author_counts: Counter,
    author_limit: int,
) -> bool:
    return any(
        int(pr.get("number", 0)) not in selected_numbers
        and author_counts[pr_author(pr)] < author_limit
        for pr in candidates
    )

