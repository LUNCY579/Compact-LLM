"""Frozen-metric helpers for later R0/R1 result collection.

These helpers do not select checkpoints or inspect confirmation data.  They summarize exact
per-example rollout rows supplied by a separately authorized evaluator.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from typing import Any, Iterable


def trajectory_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((n / total) * math.log(n / total) for n in counts.values() if n)


def summarize_rollouts(rows: Iterable[dict[str, Any]], pass_threshold: float = 1.0) -> dict[str, Any]:
    """Summarize exact rewards and within-prompt successful-mode diversity."""

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("valid", True):
            groups[str(row["prompt_id"])].append(row)
    for values in groups.values():
        values.sort(key=lambda item: int(item.get("rollout_index", 0)))

    rewards = [float(row["reward"]) for values in groups.values() for row in values]
    k_values = sorted({1, 4, *[min(len(values), len(values)) for values in groups.values()]})
    pass_at_k = {}
    for k in k_values:
        if not groups:
            pass_at_k[str(k)] = 0.0
        else:
            pass_at_k[str(k)] = sum(
                any(float(row["reward"]) >= pass_threshold for row in values[:k])
                for values in groups.values()
            ) / len(groups)

    successful_trajectory_ids = set()
    strategy_counts: Counter[str] = Counter()
    lengths = []
    for values in groups.values():
        for row in values:
            lengths.append(int(row.get("response_length", 0)))
            if float(row["reward"]) >= pass_threshold:
                text = str(row.get("trajectory", ""))
                successful_trajectory_ids.add(trajectory_id(text))
                strategy_counts[str(row.get("strategy", trajectory_id(text)))] += 1

    reward_counts = Counter(str(value) for value in rewards)
    return {
        "prompt_count": len(groups),
        "rollout_count": len(rewards),
        "pass_at_k": pass_at_k,
        "exact_reward_distribution": dict(sorted(reward_counts.items())),
        "correct_solution_diversity": {
            "unique_successful_trajectories": len(successful_trajectory_ids),
            "unique_successful_strategies": len(strategy_counts),
            "strategy_entropy": _entropy(strategy_counts),
            "max_strategy_concentration": (max(strategy_counts.values()) / sum(strategy_counts.values()))
            if strategy_counts
            else 0.0,
        },
        "reasoning_length": {
            "mean": sum(lengths) / len(lengths) if lengths else 0.0,
            "max": max(lengths) if lengths else 0,
        },
    }
