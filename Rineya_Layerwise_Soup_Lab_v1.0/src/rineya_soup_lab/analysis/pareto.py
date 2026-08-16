from __future__ import annotations

from typing import Any, Mapping, Sequence


def pareto_front(
    rows: Sequence[Mapping[str, Any]],
    *,
    maximize: Sequence[str],
    minimize: Sequence[str],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for candidate in rows:
        dominated = False
        for other in rows:
            if other is candidate:
                continue
            no_worse = all(float(other[key]) >= float(candidate[key]) for key in maximize) and all(
                float(other[key]) <= float(candidate[key]) for key in minimize
            )
            strictly_better = any(float(other[key]) > float(candidate[key]) for key in maximize) or any(
                float(other[key]) < float(candidate[key]) for key in minimize
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            result.append(dict(candidate))
    return result
