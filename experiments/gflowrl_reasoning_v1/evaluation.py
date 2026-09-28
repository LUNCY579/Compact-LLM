"""Frozen-metric helpers for later matched R0/R1 evaluation.

These functions summarize exact per-rollout records supplied by an authorized evaluator.
They do not select checkpoints, tune hyperparameters, inspect confirmation data, or infer
strategy labels.  Strategy labels must come from a separately frozen metric and are carried
through verbatim.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from statistics import median
from typing import Any, Iterable, Sequence


def trajectory_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((n / total) * math.log(n / total) for n in counts.values() if n)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile_nearest_rank(values: Sequence[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return int(ordered[rank - 1])


def summarize_rollouts(
    rows: Iterable[dict[str, Any]],
    pass_threshold: float = 1.0,
    k_values: Sequence[int] = (1, 4),
) -> dict[str, Any]:
    """Summarize reward, pass@k and successful-mode diversity with explicit denominators.

    pass@k is the empirical probability that at least one of the first k valid rollouts for
    a prompt succeeds.  Prompts with fewer than k valid rollouts are excluded from that k's
    denominator rather than silently treated as failures or as shorter groups.
    """

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    invalid_rows = 0
    for row in rows:
        if row.get("valid", True):
            groups[str(row["prompt_id"])].append(row)
        else:
            invalid_rows += 1
    for values in groups.values():
        values.sort(key=lambda item: int(item.get("rollout_index", 0)))

    requested_k = sorted({int(k) for k in k_values if int(k) > 0})
    if not requested_k:
        raise ValueError("k_values must contain at least one positive integer")

    pass_at_k: dict[str, float] = {}
    pass_at_k_denominators: dict[str, dict[str, int]] = {}
    for k in requested_k:
        eligible = [values for values in groups.values() if len(values) >= k]
        passed = sum(
            any(float(row["reward"]) >= pass_threshold for row in values[:k])
            for values in eligible
        )
        pass_at_k[str(k)] = passed / len(eligible) if eligible else 0.0
        pass_at_k_denominators[str(k)] = {
            "eligible_prompts": len(eligible),
            "passed_prompts": passed,
            "insufficient_valid_rollouts": len(groups) - len(eligible),
        }

    rewards: list[float] = []
    lengths: list[int] = []
    successful_trajectory_ids: set[str] = set()
    global_strategy_counts: Counter[str] = Counter()
    per_prompt_modes: dict[str, dict[str, Any]] = {}
    per_prompt_unique_trajectories: list[float] = []
    per_prompt_unique_strategies: list[float] = []
    per_prompt_entropy: list[float] = []

    for prompt_id, values in sorted(groups.items()):
        prompt_successes = [row for row in values if float(row["reward"]) >= pass_threshold]
        prompt_trajectory_counts: Counter[str] = Counter()
        prompt_strategy_counts: Counter[str] = Counter()

        for row in values:
            rewards.append(float(row["reward"]))
            lengths.append(int(row.get("response_length", 0)))

        for row in prompt_successes:
            tid = trajectory_id(str(row.get("trajectory", "")))
            strategy = str(row.get("strategy", tid))
            prompt_trajectory_counts[tid] += 1
            prompt_strategy_counts[strategy] += 1
            successful_trajectory_ids.add(tid)
            global_strategy_counts[strategy] += 1

        p_entropy = _entropy(prompt_strategy_counts)
        per_prompt_unique_trajectories.append(float(len(prompt_trajectory_counts)))
        per_prompt_unique_strategies.append(float(len(prompt_strategy_counts)))
        per_prompt_entropy.append(p_entropy)
        per_prompt_modes[prompt_id] = {
            "valid_rollouts": len(values),
            "successful_rollouts": len(prompt_successes),
            "unique_successful_trajectories": len(prompt_trajectory_counts),
            "unique_successful_strategies": len(prompt_strategy_counts),
            "successful_strategy_entropy": p_entropy,
            "max_successful_strategy_concentration": (
                max(prompt_strategy_counts.values()) / sum(prompt_strategy_counts.values())
                if prompt_strategy_counts
                else 0.0
            ),
        }

    reward_counts = Counter(str(value) for value in rewards)
    multiple_strategy_prompts = sum(v >= 2 for v in per_prompt_unique_strategies)
    pass_at_1 = pass_at_k.get("1", 0.0)

    return {
        "prompt_count": len(groups),
        "valid_rollout_count": len(rewards),
        "invalid_rollout_count": invalid_rows,
        "pass_at_1": pass_at_1,
        "pass_at_k": pass_at_k,
        "pass_at_k_denominators": pass_at_k_denominators,
        "exact_reward_distribution": dict(sorted(reward_counts.items())),
        "correct_solution_diversity": {
            "unique_successful_trajectories_global": len(successful_trajectory_ids),
            "unique_successful_strategies_global": len(global_strategy_counts),
            "global_strategy_entropy": _entropy(global_strategy_counts),
            "global_max_strategy_concentration": (
                max(global_strategy_counts.values()) / sum(global_strategy_counts.values())
                if global_strategy_counts
                else 0.0
            ),
            "mean_unique_successful_trajectories_per_prompt": _mean(per_prompt_unique_trajectories),
            "mean_unique_successful_strategies_per_prompt": _mean(per_prompt_unique_strategies),
            "mean_successful_strategy_entropy_per_prompt": _mean(per_prompt_entropy),
            "prompts_with_multiple_successful_strategies": multiple_strategy_prompts,
            "fraction_prompts_with_multiple_successful_strategies": (
                multiple_strategy_prompts / len(groups) if groups else 0.0
            ),
            "per_prompt": per_prompt_modes,
        },
        "reasoning_length": {
            "mean": _mean([float(v) for v in lengths]),
            "median": float(median(lengths)) if lengths else 0.0,
            "p90_nearest_rank": _percentile_nearest_rank(lengths, 0.90),
            "max": max(lengths) if lengths else 0,
        },
    }


def paired_repairs_damages(
    r0_rows: Iterable[dict[str, Any]],
    r1_rows: Iterable[dict[str, Any]],
    pass_threshold: float = 1.0,
) -> dict[str, Any]:
    """Pair pass@1 outcomes by prompt and report exact repairs/damages.

    The pass@1 trajectory is rollout_index=0.  A prompt missing a valid index-0 rollout in
    either arm is excluded and reported rather than substituted with a later rollout.
    """

    def first_by_prompt(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        duplicates: set[str] = set()
        for row in rows:
            if not row.get("valid", True) or int(row.get("rollout_index", -1)) != 0:
                continue
            prompt_id = str(row["prompt_id"])
            if prompt_id in out:
                duplicates.add(prompt_id)
            out[prompt_id] = row
        if duplicates:
            raise ValueError(f"duplicate valid rollout_index=0 rows: {sorted(duplicates)}")
        return out

    r0 = first_by_prompt(r0_rows)
    r1 = first_by_prompt(r1_rows)
    common = sorted(set(r0) & set(r1))
    repairs: list[str] = []
    damages: list[str] = []
    stable_correct: list[str] = []
    stable_incorrect: list[str] = []

    for prompt_id in common:
        base_ok = float(r0[prompt_id]["reward"]) >= pass_threshold
        candidate_ok = float(r1[prompt_id]["reward"]) >= pass_threshold
        if not base_ok and candidate_ok:
            repairs.append(prompt_id)
        elif base_ok and not candidate_ok:
            damages.append(prompt_id)
        elif base_ok:
            stable_correct.append(prompt_id)
        else:
            stable_incorrect.append(prompt_id)

    return {
        "paired_prompts": len(common),
        "repairs": len(repairs),
        "damages": len(damages),
        "net_repairs_minus_damages": len(repairs) - len(damages),
        "stable_correct": len(stable_correct),
        "stable_incorrect": len(stable_incorrect),
        "repair_prompt_ids": repairs,
        "damage_prompt_ids": damages,
        "missing_r0_prompt_ids": sorted(set(r1) - set(r0)),
        "missing_r1_prompt_ids": sorted(set(r0) - set(r1)),
    }
