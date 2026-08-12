from __future__ import annotations

import hashlib
from collections import defaultdict, deque
from typing import Any, Iterable


def stable_rank(identifier: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{identifier}".encode("utf-8")).hexdigest()


def deterministic_stratified_sample(
    rows: list[dict[str, Any]],
    *,
    limit: int | None,
    seed: int,
    strata: Iterable[str] = ("decision", "difficulty", "reasoning_family", "domain"),
) -> list[dict[str, Any]]:
    if limit is None or limit >= len(rows):
        return rows
    if limit <= 0:
        raise ValueError("limit must be positive")
    keys = list(strata)
    if not keys:
        return sorted(rows, key=lambda row: stable_rank(str(row.get("id")), seed))[:limit]
    groups: dict[str, dict[tuple[str, ...], list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        primary = str(row.get(keys[0]) or "")
        secondary = tuple(str(row.get(key) or "") for key in keys[1:])
        groups[primary][secondary].append(row)

    # First interleave secondary strata inside each primary stratum. Then
    # interleave the primary strata. This prevents a small sample from being
    # monopolized by whichever decision label sorts first lexically.
    primary_queues: dict[str, deque[dict[str, Any]]] = {}
    for primary, secondary_groups in groups.items():
        for group_rows in secondary_groups.values():
            group_rows.sort(key=lambda row: stable_rank(str(row.get("id")), seed))
        ordered_secondary = sorted(
            secondary_groups,
            key=lambda value: stable_rank(f"secondary:{primary}:{value!r}", seed),
        )
        interleaved: deque[dict[str, Any]] = deque()
        secondary_queues = {key: deque(secondary_groups[key]) for key in ordered_secondary}
        while any(secondary_queues.values()):
            for key in ordered_secondary:
                if secondary_queues[key]:
                    interleaved.append(secondary_queues[key].popleft())
        primary_queues[primary] = interleaved
    ordered_primary = sorted(groups, key=lambda value: stable_rank(f"primary:{value}", seed))
    selected: list[dict[str, Any]] = []
    while len(selected) < limit and any(primary_queues.values()):
        for primary in ordered_primary:
            if primary_queues[primary] and len(selected) < limit:
                selected.append(primary_queues[primary].popleft())
    return sorted(selected, key=lambda row: stable_rank(str(row.get("id")), seed))
