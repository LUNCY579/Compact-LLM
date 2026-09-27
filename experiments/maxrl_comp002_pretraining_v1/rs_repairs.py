"""Composition-only repair and joint-damage counts on dependent RS rows.

Input is one row per evaluated example with actual arm correctness already
established by the frozen scorer. Bootstrap resamples complete parent clusters.
No oracle answer or model response is opened in this module.
"""
from __future__ import annotations

from collections import defaultdict
import random

ARMS = ("BASE", "R", "S", "RS")


def _quantile(sorted_values: list[int], q: float) -> float:
    at = q * (len(sorted_values) - 1)
    lo = int(at)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (at - lo) * (sorted_values[hi] - sorted_values[lo])


def analyse(rows: list[dict], metric: str, *, draws: int = 10_000, seed: int = 20260926) -> dict:
    if metric not in ("strict", "content") or not rows or type(draws) is not int or draws < 1:
        raise ValueError("invalid metric/rows/draws")
    seen = set()
    parents = defaultdict(list)
    optional = None
    for row in rows:
        if type(row) is not dict or set(row) != {"id", "parent_id", "scores"}:
            raise ValueError("invalid row shape")
        row_id, parent, scores = row["id"], row["parent_id"], row["scores"]
        if type(row_id) is not str or not row_id or row_id in seen or type(parent) is not str or not parent:
            raise ValueError("duplicate/missing row or parent identity")
        seen.add(row_id)
        if type(scores) is not dict or not set(ARMS) <= set(scores):
            raise ValueError("missing four-arm correctness vector")
        extras = set(scores) - set(ARMS)
        if extras not in (set(), {"R_to_S", "S_to_R"}):
            raise ValueError("partial or unexpected sequential panel")
        if optional is None:
            optional = extras
        elif extras != optional:
            raise ValueError("inconsistent arm set across rows")
        for arm, outcome in scores.items():
            if type(outcome) is not dict or set(outcome) != {"strict", "content"} or any(type(v) is not bool for v in outcome.values()):
                raise ValueError(f"invalid outcome for {arm}")
            if outcome["strict"] and not outcome["content"]:
                raise ValueError("strict success cannot fail bounded content")
        parents[parent].append({arm: scores[arm][metric] for arm in scores})

    arm_names = list(ARMS) + sorted(optional or ())
    totals = {arm: 0 for arm in arm_names}
    keys = ("composition_only_repair", "joint_damage", "R_only_success",
            "S_only_success", "both_singles_success", "base_success", "all_four_fail")
    per_parent = []
    for parent_rows in parents.values():
        counts = {key: 0 for key in keys}
        for v in parent_rows:
            for arm in arm_names:
                totals[arm] += int(v[arm])
            b, r, s, c = (v[arm] for arm in ARMS)
            counts["composition_only_repair"] += int(not b and not r and not s and c)
            counts["joint_damage"] += int((r or s) and not c)
            counts["R_only_success"] += int(r and not s)
            counts["S_only_success"] += int(s and not r)
            counts["both_singles_success"] += int(r and s)
            counts["base_success"] += int(b)
            counts["all_four_fail"] += int(not (b or r or s or c))
        per_parent.append(counts)
    points = {key: sum(p[key] for p in per_parent) for key in keys}
    rng = random.Random(seed)
    n = len(per_parent)
    samples = {key: [] for key in ("composition_only_repair", "joint_damage")}
    for _ in range(draws):
        drawn = [per_parent[rng.randrange(n)] for _ in range(n)]
        for key in samples:
            samples[key].append(sum(p[key] for p in drawn))
    intervals = {key: {"lower95_count": _quantile(sorted(v), 0.025),
                       "upper95_count": _quantile(sorted(v), 0.975)}
                 for key, v in samples.items()}
    return {"metric": metric, "rows": len(rows), "parents": n,
            "arm_correct": totals, "counts": points, "parent_cluster_bootstrap": intervals,
            "bootstrap_draws": draws, "seed": seed,
            "note": "A composition-only repair needs BASE=R=S=0 and RS=1 on the same row. Counts on correlated siblings are resampled by parent."}
