"""Deterministic nested unlocks, uncertainty-aware bands, mixtures and promotion."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .schema import Gates, Stage, Task, digest, require


@dataclass
class TaskStats:
    trials: int = 0
    successes: float = 0
    version: int = 0

    def __post_init__(self):
        require(type(self.trials) is int and self.trials >= 0, "invalid trials")
        require(math.isfinite(self.successes) and 0 <= self.successes <= self.trials, "invalid successes")
        require(type(self.version) is int and self.version >= 0, "invalid statistics version")

    def band(self, current_version: int, gates: Gates) -> str:
        if self.trials < gates.min_observations or current_version - self.version > gates.stale_after_updates:
            return "unknown"
        # Beta(1,1) posterior mean: a 0/4 sample is not proof of impossibility.
        p = (self.successes + 1) / (self.trials + 2)
        if p > .8:
            return "mastered"
        if p > .6:
            return "easy"
        if p >= .15:
            return "frontier"
        return "hard"  # No irreversible lock; these remain eligible for later probing.

    def utility(self) -> float:
        p = (self.successes + 1) / (self.trials + 2)
        return p * (1 - p)  # A sampling heuristic, not an estimate of learning gain.


def unlocked(tasks: list[Task], stage: Stage, seed: int) -> list[Task]:
    result = []
    for source in sorted(stage.source_weights):
        pool = [t for t in tasks if t.source == source and t.split == "train"]
        groups = sorted({t.group for t in pool}, key=lambda g: digest([seed, source, g]))
        # Fractions count parent clusters; rounding down never exceeds the ceiling.
        selected = set(groups[:math.floor(len(groups) * stage.fraction)])
        result.extend(t for t in pool if t.group in selected)
    return sorted(result, key=lambda t: t.key)


def batch(tasks: list[Task], replay: list[Task], stats: dict[str, TaskStats], stage: Stage,
          gates: Gates, version: int, seed: int, replay_fraction: float | None = None) -> tuple[list[Task], dict]:
    require(bool(tasks) and bool(replay), "empty acquisition or retention-training pool")
    rng = random.Random(seed)
    replay_share = stage.replay_fraction if replay_fraction is None else replay_fraction
    require(0 < replay_share < 1, "invalid replay share")
    # At least one replay and one acquisition task; report exact realized proportions.
    n_replay = min(stage.batch_tasks - 1, max(1, round(stage.batch_tasks * replay_share)))
    selected = rng.choices(replay, k=n_replay)
    sources = sorted({t.source for t in tasks if stage.source_weights.get(t.source, 0) > 0})
    require(bool(sources), "no compatible source has positive weight")
    fallbacks = 0
    for _ in range(stage.batch_tasks - n_replay):
        source = rng.choices(sources, weights=[stage.source_weights[s] for s in sources])[0]
        pool = [t for t in tasks if t.source == source]
        band_names = sorted(stage.band_weights)
        desired = rng.choices(band_names, weights=[stage.band_weights[b] for b in band_names])[0]
        candidates = [t for t in pool if stats.get(t.key, TaskStats()).band(version, gates) == desired]
        if not candidates:
            # Explicit, counted fallback within the same source. Never steal a held-out row.
            candidates = pool
            fallbacks += 1
        selected.append(rng.choices(candidates, weights=[max(.02, stats.get(t.key, TaskStats()).utility())
                                                        for t in candidates])[0])
    rng.shuffle(selected)
    realized: dict[str, int] = {}
    for task in selected:
        band = "replay" if task.split == "retention_train" else stats.get(task.key, TaskStats()).band(version, gates)
        label = f"{task.source}/{band}"
        realized[label] = realized.get(label, 0) + 1
    return selected, {"realized": realized, "fallbacks": fallbacks, "replay_tasks": n_replay}


def paired_interval(before: dict[str, float], after: dict[str, float], groups: dict[str, str],
                    gates: Gates, seed: int) -> dict:
    require(set(before) == set(after) == set(groups) and bool(before), "evaluation panel mismatch")
    by_group: dict[str, list[float]] = {}
    for key in before:
        require(all(math.isfinite(x) and 0 <= x <= 1 for x in (before[key], after[key])), "invalid scores")
        by_group.setdefault(groups[key], []).append(after[key] - before[key])
    values = [sum(v) / len(v) for _, v in sorted(by_group.items())]
    require(len(values) >= gates.min_eval_groups, "insufficient independent evaluation groups")
    rng = random.Random(seed)
    samples = sorted(sum(rng.choices(values, k=len(values))) / len(values)
                     for _ in range(gates.bootstrap_samples))
    return {"mean": sum(values) / len(values), "lcb": samples[int(.025 * (len(samples) - 1))],
            "ucb": samples[int(.975 * (len(samples) - 1))], "groups": len(values)}


def gate(capability: dict, retention: dict[str, dict], signal: dict, limits: Gates) -> tuple[bool, list[str]]:
    reasons = []
    if capability["lcb"] < limits.min_gain_lcb:
        reasons.append("capability_lcb")
    for bank, interval in retention.items():
        if interval["lcb"] < -limits.max_retention_drop:
            reasons.append(f"retention_lcb:{bank}")
    if signal["groups"] < limits.min_train_groups:
        reasons.append("insufficient_training_groups")
    if signal["mixed_fraction"] < limits.min_mixed_fraction:
        reasons.append("insufficient_reward_signal")
    if signal["valid_action_rate"] < limits.min_valid_action_rate:
        reasons.append("invalid_actions")
    if signal["truncation_rate"] > limits.max_truncation:
        reasons.append("truncation")
    return not reasons, reasons
