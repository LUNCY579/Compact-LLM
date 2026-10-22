"""Auditable depth/width analysis and causal stopping, using only the stdlib.

A trace is an externally supplied sequence of *same-prefix* readouts. Replaying
one is not a native run, an actual adaptive latency measurement, or a FLOP audit.
Correctness is deliberately excluded from the object supplied to the policy.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import math
from typing import Iterable, Sequence


def positive_int(value: int, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def nonnegative(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def nested_seed(base_seed: int, parent_id: str, rollout: int) -> int:
    """Cell-independent RNG seed: increasing R or K does not rename a rollout."""
    if type(base_seed) is not int or base_seed < 0:
        raise ValueError("base_seed must be a nonnegative integer")
    if not isinstance(parent_id, str) or not parent_id:
        raise ValueError("parent_id required")
    positive_int(rollout, "rollout")
    key = f"cde001:{base_seed}:{len(parent_id)}:{parent_id}:{rollout}"
    return int(digest(key)[:16], 16)


def grid(depths: Sequence[int], widths: Sequence[int]) -> list[dict]:
    for axis in (depths, widths):
        if not axis or len(set(axis)) != len(axis):
            raise ValueError("axes must be nonempty and unique")
        for value in axis:
            positive_int(value, "axis value")
    return [{"depth": r, "width": k, "nominal_RK": r * k,
             "compute_match": "UNQUALIFIED_PROXY_ONLY"}
            for r in sorted(depths) for k in sorted(widths)]


def distribution(values: Sequence[float]) -> tuple[float, ...]:
    p = tuple(nonnegative(v, "probability") for v in values)
    if not p or not math.isclose(math.fsum(p), 1.0, abs_tol=1e-8, rel_tol=0):
        raise ValueError("probabilities must be nonempty and sum to one")
    return p


def total_variation(a: Sequence[float], b: Sequence[float]) -> float:
    a, b = distribution(a), distribution(b)
    if len(a) != len(b):
        raise ValueError("vocabulary mismatch")
    return math.fsum(abs(x - y) for x, y in zip(a, b)) / 2


@dataclass(frozen=True)
class Observation:
    depth: int
    prefix_id: str
    prediction: str
    probabilities: tuple[float, ...]
    cumulative_work: float

    def __post_init__(self):
        positive_int(self.depth, "depth")
        if not self.prefix_id or not isinstance(self.prefix_id, str):
            raise ValueError("prefix_id required")
        if not isinstance(self.prediction, str) or not self.prediction:
            raise ValueError("nonempty prediction required")
        object.__setattr__(self, "probabilities", distribution(self.probabilities))
        nonnegative(self.cumulative_work, "cumulative_work")


@dataclass(frozen=True)
class Trace:
    parent_id: str
    split: str
    rollout: int
    checkpoint_id: str
    observations: tuple[Observation, ...]
    correct: tuple[bool, ...]

    def __post_init__(self):
        if not isinstance(self.parent_id, str) or not self.parent_id:
            raise ValueError("parent_id required")
        if self.split not in {"train", "dev", "test"}:
            raise ValueError("invalid split")
        positive_int(self.rollout, "rollout")
        if not isinstance(self.checkpoint_id, str) or not self.checkpoint_id:
            raise ValueError("checkpoint_id required")
        object.__setattr__(self, "observations", tuple(self.observations))
        object.__setattr__(self, "correct", tuple(self.correct))
        if not self.observations or len(self.observations) != len(self.correct):
            raise ValueError("complete nonempty labels and observations required")
        if any(type(x) is not bool for x in self.correct):
            raise ValueError("correctness must be boolean")
        validate_observations(self.observations)


def validate_observations(observations: Sequence[Observation]) -> None:
    if not observations:
        raise ValueError("empty observations")
    prefix, vocab = observations[0].prefix_id, len(observations[0].probabilities)
    previous_work = 0.0
    for depth, item in enumerate(observations, 1):
        if item.depth != depth:
            raise ValueError("every consecutive depth starting at one is required")
        if item.prefix_id != prefix or len(item.probabilities) != vocab:
            raise ValueError("agreement requires a common prefix and vocabulary")
        if item.cumulative_work < previous_work:
            raise ValueError("cumulative work cannot decrease")
        previous_work = item.cumulative_work


@dataclass(frozen=True)
class Policy:
    threshold: float = 0.0
    patience: int = 2
    full_depth_only: bool = False
    calibration_parents: tuple[str, ...] = ()
    checkpoint_id: str | None = None
    max_depth: int | None = None

    def __post_init__(self):
        nonnegative(self.threshold, "threshold")
        if self.threshold > 1:
            raise ValueError("TV threshold must be in [0,1]")
        positive_int(self.patience, "patience")
        if type(self.full_depth_only) is not bool:
            raise ValueError("full_depth_only must be boolean")
        object.__setattr__(self, "calibration_parents", tuple(self.calibration_parents))
        if any(not isinstance(p, str) or not p for p in self.calibration_parents):
            raise ValueError("invalid calibration parent")
        if self.checkpoint_id is not None and (not isinstance(self.checkpoint_id, str) or not self.checkpoint_id):
            raise ValueError("invalid calibration checkpoint")
        if self.max_depth is not None:
            positive_int(self.max_depth, "max_depth")


def choose_depth(observations: Sequence[Observation], policy: Policy) -> int:
    """Uses only current/past readouts, never correctness or future predictions.

    Validation inspects trace integrity; the decision loop only compares adjacent
    already-observed readouts. Earliest exit is patience+1, never depth one.
    """
    validate_observations(observations)
    if policy.full_depth_only:
        return len(observations)
    stable = 0
    for previous, current in zip(observations, observations[1:]):
        agree = (previous.prediction == current.prediction and
                 total_variation(previous.probabilities, current.probabilities)
                 <= policy.threshold)
        stable = stable + 1 if agree else 0
        if stable >= policy.patience:
            return current.depth
    return len(observations)


def validate_bank(traces: Sequence[Trace]) -> None:
    if not traces:
        raise ValueError("trace bank is empty")
    if len({t.checkpoint_id for t in traces}) != 1:
        raise ValueError("one fixed checkpoint per bank required")
    seen, splits = set(), {}
    for t in traces:
        key = (t.parent_id, t.rollout)
        if key in seen:
            raise ValueError("duplicate parent/rollout")
        seen.add(key)
        if t.parent_id in splits and splits[t.parent_id] != t.split:
            raise ValueError("parent crosses splits")
        splits[t.parent_id] = t.split


def parent_mean(values: Iterable[tuple[str, float]]) -> float:
    grouped = defaultdict(list)
    for parent, value in values:
        grouped[parent].append(value)
    if not grouped:
        raise ValueError("empty values")
    return math.fsum(math.fsum(v) / len(v) for v in grouped.values()) / len(grouped)


def replay(traces: Sequence[Trace], policy: Policy, *, heldout: bool = False) -> dict:
    """Parent-balanced counterfactual quality/work, NOT end-to-end speedup."""
    validate_bank(traces)
    if policy.checkpoint_id is not None and policy.checkpoint_id != traces[0].checkpoint_id:
        raise ValueError("policy checkpoint mismatch")
    if policy.max_depth is not None and any(len(t.observations) != policy.max_depth for t in traces):
        raise ValueError("policy maximum-depth mismatch")
    if heldout:
        if any(t.split != "test" for t in traces):
            raise ValueError("heldout evaluation requires only test parents")
        if set(policy.calibration_parents) & {t.parent_id for t in traces}:
            raise ValueError("calibration/test parent leakage")
    rows = []
    for t in traces:
        r = choose_depth(t.observations, policy)
        labels = t.correct
        rows.append((t.parent_id, {
            "chosen_depth": r,
            "selected_accuracy": float(labels[r - 1]),
            "final_accuracy": float(labels[-1]),
            "oracle_any_depth": float(any(labels)),
            "late_rescue": float(not labels[0] and any(labels[1:])),
            "missed_rescue": float(not labels[r - 1] and any(labels[r:])),
            "reversions": sum(a and not b for a, b in zip(labels, labels[1:])),
            "selected_work_counterfactual": t.observations[r - 1].cumulative_work,
            "full_work": t.observations[-1].cumulative_work,
        }))
    metrics = {key: parent_mean((parent, row[key]) for parent, row in rows)
               for key in rows[0][1]}
    return {"n_parents": len({t.parent_id for t in traces}), "n_traces": len(traces),
            "evidence": "OFFLINE_TRACE_REPLAY_NOT_NATIVE_QUALIFICATION",
            "rates_denominator": "all_parents_balanced_over_rollouts",
            **metrics}


def calibrate(dev: Sequence[Trace], thresholds: Sequence[float], *, patience: int = 2,
              max_accuracy_loss: float = 0.0, max_missed_rescue: float = 0.0) -> Policy:
    validate_bank(dev)
    if any(t.split != "dev" for t in dev):
        raise ValueError("calibrate on DEV only")
    for name, value in (("max_accuracy_loss", max_accuracy_loss),
                        ("max_missed_rescue", max_missed_rescue)):
        nonnegative(value, name)
        if value > 1:
            raise ValueError(f"{name} cannot exceed one")
    if len({len(t.observations) for t in dev}) != 1:
        raise ValueError("calibration requires one fixed maximum depth")
    parents = tuple(sorted({t.parent_id for t in dev}))
    binding = {"checkpoint_id": dev[0].checkpoint_id, "max_depth": len(dev[0].observations)}
    policies = [Policy(patience=patience, full_depth_only=True,
                       calibration_parents=parents, **binding)]
    policies += [Policy(x, patience, False, parents, **binding) for x in thresholds]
    feasible = []
    for policy in policies:
        m = replay(dev, policy)
        if (m["selected_accuracy"] + max_accuracy_loss + 1e-12 >= m["final_accuracy"]
                and m["missed_rescue"] <= max_missed_rescue + 1e-12):
            feasible.append((m["selected_work_counterfactual"],
                             -int(policy.full_depth_only), policy.threshold, policy))
    # Full-depth fallback is always feasible. Prefer it when no work is saved.
    return min(feasible, key=lambda entry: entry[:3])[-1]


def cell_metrics(traces: Sequence[Trace], depth: int, width: int,
                 *, selector: str = "majority") -> dict:
    """Nested rollouts; selection uses predictions, not gold correctness.

    Work is the SUM of supplied per-rollout cumulative costs. No assumption of
    parallel speedup or shared prefill is made. Audit actual costs at the host.
    """
    validate_bank(traces)
    positive_int(depth, "depth")
    positive_int(width, "width")
    if selector not in {"first", "majority"}:
        raise ValueError("selector must be first or majority, never oracle")
    if len({t.split for t in traces}) != 1:
        raise ValueError("one split per depth/width comparison required")
    grouped = defaultdict(dict)
    for trace in traces:
        grouped[trace.parent_id][trace.rollout] = trace
    results = []
    for parent, bank in grouped.items():
        if not set(range(1, width + 1)) <= bank.keys():
            raise ValueError("missing nested rollout")
        chosen = [bank[i] for i in range(1, width + 1)]
        if any(len(t.observations) < depth for t in chosen):
            raise ValueError("missing depth")
        if len({t.split for t in chosen}) != 1:
            raise ValueError("split mismatch")
        predictions = [t.observations[depth - 1].prediction for t in chosen]
        counts = Counter(predictions)
        index = 0 if selector == "first" else max(
            range(width), key=lambda i: (counts[predictions[i]], -i))
        # Same answer for the same parent must receive the same verdict.
        verdicts = {}
        for t, pred in zip(chosen, predictions):
            correct = t.correct[depth - 1]
            if pred in verdicts and verdicts[pred] != correct:
                raise ValueError("inconsistent verifier verdict for identical prediction")
            verdicts[pred] = correct
        results.append({
            "selected_accuracy": float(chosen[index].correct[depth - 1]),
            "mean_final_accuracy": sum(t.correct[depth - 1] for t in chosen) / width,
            "oracle_final_depth_width": float(any(t.correct[depth - 1] for t in chosen)),
            "oracle_any_depth_width": float(any(any(t.correct[:depth]) for t in chosen)),
            "summed_work": math.fsum(t.observations[depth - 1].cumulative_work for t in chosen),
        })
    return {"depth": depth, "width": width, "nominal_RK": depth * width,
            "compute_match": "UNQUALIFIED_PROXY_ONLY", "selector": selector,
            "n_parents": len(results),
            **{key: math.fsum(r[key] for r in results) / len(results)
               for key in results[0]}}
