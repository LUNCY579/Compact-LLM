#!/usr/bin/env python3
"""Shared logic for the single-pass decode-time abstention gate.

Every confidence signal is read from the ``confidence`` block that
``Rineya_Long_Context_Demo_v1/scripts/run_local_hf.py --emit-logprobs`` writes
next to each prediction. Nothing here re-runs the model: a gate is a pure
function of one frozen greedy decode pass.

Scoring reuses ``evaluate_predictions.normalize`` from the published suite so a
gate can never be scored by a private, more forgiving rule.
"""

from __future__ import annotations

import importlib.util
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[2]
SCORER_PATH = (
    REPOSITORY_ROOT / "Rineya_Long_Context_Demo_v1" / "scripts" / "evaluate_predictions.py"
)

ABSTAIN_TEXT = "UNKNOWN"

# Every surface form the published scorer accepts on an unanswerable row.
# A gate "abstains" when the emitted text lands anywhere in this vocabulary,
# not only when it is the literal token the gate writes.
ABSTENTION_SURFACE_FORMS = (
    "UNKNOWN",
    "unknown",
    "not determinable",
    "cannot be determined",
)


def _load_scorer() -> Any:
    spec = importlib.util.spec_from_file_location(
        "rineya_long_context_demo_evaluate_predictions", SCORER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import the published scorer: {SCORER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCORER = _load_scorer()


ABSTENTION_VOCABULARY = frozenset(
    SCORER.normalize(form) for form in ABSTENTION_SURFACE_FORMS
)


def is_abstention(text: str) -> bool:
    return SCORER.normalize(text) in ABSTENTION_VOCABULARY


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


# --------------------------------------------------------------------------
# Signals
# --------------------------------------------------------------------------

# direction "low"  -> abstain when value <= threshold (low confidence)
# direction "high" -> abstain when value >= threshold (high abstention evidence)
SIGNALS: dict[str, tuple[str, Callable[[dict[str, Any]], float]]] = {
    "first_token_logprob_margin": ("low", lambda c: c["first_token"]["logprob_margin"]),
    "first_token_prob_margin": ("low", lambda c: c["first_token"]["prob_margin"]),
    "first_token_logprob": ("low", lambda c: c["first_token"]["token_logprob"]),
    "first_token_entropy": ("high", lambda c: c["first_token"]["entropy"]),
    "answer_sum_logprob": ("low", lambda c: c["answer"]["sum_logprob"]),
    "answer_mean_logprob": ("low", lambda c: c["answer"]["mean_logprob"]),
    "answer_min_logprob": ("low", lambda c: c["answer"]["min_logprob"]),
    "answer_mean_logprob_margin": ("low", lambda c: c["answer"]["mean_logprob_margin"]),
    "answer_min_logprob_margin": ("low", lambda c: c["answer"]["min_logprob_margin"]),
    "answer_mean_entropy": ("high", lambda c: c["answer"]["mean_entropy"]),
    "answer_max_entropy": ("high", lambda c: c["answer"]["max_entropy"]),
    "unknown_exact_logprob": ("high", lambda c: c["unknown"]["exact_logprob"]),
    "unknown_variant_logprob": ("high", lambda c: c["unknown"]["variant_logprob"]),
    "unknown_exact_minus_top1": (
        "high",
        lambda c: c["unknown"]["exact_logprob"] - c["first_token"]["top1_logprob"],
    ),
    "unknown_variant_minus_top1": (
        "high",
        lambda c: c["unknown"]["variant_logprob"] - c["first_token"]["top1_logprob"],
    ),
}


def signal_direction(name: str) -> str:
    if name not in SIGNALS:
        raise KeyError(f"Unknown signal: {name}")
    return SIGNALS[name][0]


def signal_value(name: str, confidence: dict[str, Any]) -> float:
    if name not in SIGNALS:
        raise KeyError(f"Unknown signal: {name}")
    value = float(SIGNALS[name][1](confidence))
    if not math.isfinite(value):
        raise ValueError(f"Signal {name} produced a non-finite value: {value}")
    return value


def abstains(name: str, value: float, threshold: float) -> bool:
    direction = signal_direction(name)
    return value <= threshold if direction == "low" else value >= threshold


# --------------------------------------------------------------------------
# Rows
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GateRow:
    row_id: str
    prediction: str
    answerable: bool
    accepted: frozenset[str]
    signals: dict[str, float]

    @property
    def baseline_correct(self) -> bool:
        return SCORER.normalize(self.prediction) in self.accepted

    @property
    def abstained_by_model(self) -> bool:
        return is_abstention(self.prediction)

    @property
    def abstain_correct(self) -> bool:
        return SCORER.normalize(ABSTAIN_TEXT) in self.accepted


def build_rows(
    confidence_rows: Sequence[dict[str, Any]], key_rows: Sequence[dict[str, Any]]
) -> list[GateRow]:
    by_id = {str(row["id"]): row for row in confidence_rows}
    if len(by_id) != len(confidence_rows):
        raise ValueError("Duplicate ids in the confidence file")

    rows: list[GateRow] = []
    for key in key_rows:
        row_id = str(key["id"])
        source = by_id.get(row_id)
        if source is None:
            raise ValueError(f"{row_id}: missing from the confidence file")
        if source.get("status") != "ok":
            raise ValueError(f"{row_id}: status is {source.get('status')!r}, not ok")
        confidence = source.get("confidence")
        if not isinstance(confidence, dict):
            raise ValueError(f"{row_id}: no confidence block; rerun --emit-logprobs")
        if confidence.get("answer") is None:
            raise ValueError(f"{row_id}: empty generation, answer signals undefined")
        rows.append(
            GateRow(
                row_id=row_id,
                prediction=str(source.get("prediction", "")),
                answerable=bool(key["answerable"]),
                accepted=frozenset(
                    SCORER.normalize(value)
                    for value in key.get("accepted_answers", [key.get("answer", "")])
                ),
                signals={name: signal_value(name, confidence) for name in SIGNALS},
            )
        )
    unexpected = sorted(set(by_id) - {str(key["id"]) for key in key_rows})
    if unexpected:
        raise ValueError(f"Confidence rows outside the answer key: {unexpected}")
    return rows


# --------------------------------------------------------------------------
# Gates
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GateSpec:
    """A gate is a boolean combination of at most two thresholded signals."""

    signal: str
    threshold: float
    signal_b: str | None = None
    threshold_b: float | None = None
    combine: str = "single"  # "single" | "and" | "or"

    def __post_init__(self) -> None:
        if self.combine not in {"single", "and", "or"}:
            raise ValueError(f"Unsupported combine mode: {self.combine}")
        signal_direction(self.signal)
        if self.combine == "single":
            if self.signal_b is not None or self.threshold_b is not None:
                raise ValueError("A single-signal gate must not define a second signal")
        else:
            if self.signal_b is None or self.threshold_b is None:
                raise ValueError(f"combine={self.combine} needs signal_b and threshold_b")
            signal_direction(self.signal_b)

    @property
    def arity(self) -> int:
        return 1 if self.combine == "single" else 2

    def decide(self, row: GateRow) -> bool:
        first = abstains(self.signal, row.signals[self.signal], self.threshold)
        if self.combine == "single":
            return first
        assert self.signal_b is not None and self.threshold_b is not None
        second = abstains(self.signal_b, row.signals[self.signal_b], self.threshold_b)
        return (first and second) if self.combine == "and" else (first or second)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "combine": self.combine,
            "signal": self.signal,
            "direction": signal_direction(self.signal),
            "threshold": self.threshold,
        }
        if self.combine != "single":
            payload["signal_b"] = self.signal_b
            payload["direction_b"] = signal_direction(str(self.signal_b))
            payload["threshold_b"] = self.threshold_b
        payload["rule"] = describe(self)
        payload["abstain_text"] = ABSTAIN_TEXT
        return payload

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "GateSpec":
        return GateSpec(
            signal=str(payload["signal"]),
            threshold=float(payload["threshold"]),
            signal_b=(
                str(payload["signal_b"]) if payload.get("signal_b") is not None else None
            ),
            threshold_b=(
                float(payload["threshold_b"])
                if payload.get("threshold_b") is not None
                else None
            ),
            combine=str(payload.get("combine", "single")),
        )


def _clause(signal: str, threshold: float) -> str:
    operator = "<=" if signal_direction(signal) == "low" else ">="
    return f"{signal} {operator} {threshold:.6g}"


def describe(spec: GateSpec) -> str:
    head = _clause(spec.signal, spec.threshold)
    if spec.combine == "single":
        return f"abstain if {head}"
    assert spec.signal_b is not None and spec.threshold_b is not None
    joiner = " AND " if spec.combine == "and" else " OR "
    return f"abstain if {head}{joiner}{_clause(spec.signal_b, spec.threshold_b)}"


def candidate_thresholds(values: Sequence[float], direction: str) -> list[float]:
    """Cut points that realise every achievable partition of ``values``.

    Cuts sit at midpoints between consecutive observed values, so a selected
    threshold is never pinned to a single tuning observation.
    """

    unique = sorted(set(float(value) for value in values))
    if not unique:
        return []
    spread = unique[-1] - unique[0]
    pad = max(1.0, abs(spread)) if spread == 0 else spread
    cuts = [(low + high) / 2.0 for low, high in zip(unique, unique[1:])]
    below = unique[0] - pad
    above = unique[-1] + pad
    # "abstain on nothing" and "abstain on everything" endpoints.
    return sorted([below, *cuts, above]) if direction == "low" else sorted(
        [below, *cuts, above]
    )


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GateOutcome:
    rows: int
    baseline_correct: int
    gated_correct: int
    delta: int
    abstentions: int
    model_abstentions: int
    added_abstentions: int
    abstention_true_positives: int
    abstention_false_positives: int
    abstention_precision: float
    abstention_recall: float
    unanswerable_rows: int
    gained_ids: tuple[str, ...]
    lost_ids: tuple[str, ...]
    mcnemar_b: int
    mcnemar_c: int
    mcnemar_p: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "baseline_correct": self.baseline_correct,
            "gated_correct": self.gated_correct,
            "delta": self.delta,
            "abstentions": self.abstentions,
            "model_abstentions": self.model_abstentions,
            "added_abstentions": self.added_abstentions,
            "abstention_true_positives": self.abstention_true_positives,
            "abstention_false_positives": self.abstention_false_positives,
            "abstention_precision": round(self.abstention_precision, 6),
            "abstention_recall": round(self.abstention_recall, 6),
            "unanswerable_rows": self.unanswerable_rows,
            "gained_ids": list(self.gained_ids),
            "lost_ids": list(self.lost_ids),
            "mcnemar_b_gate_only_correct": self.mcnemar_b,
            "mcnemar_c_baseline_only_correct": self.mcnemar_c,
            "mcnemar_exact_p": round(self.mcnemar_p, 6),
        }


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value over the discordant pairs."""

    if b < 0 or c < 0:
        raise ValueError("Discordant counts must be non-negative")
    n = b + c
    if n == 0:
        return 1.0
    lower = min(b, c)
    tail = sum(math.comb(n, k) for k in range(lower + 1)) / (2.0**n)
    return min(1.0, 2.0 * tail)


def gated_prediction(spec: GateSpec, row: GateRow) -> str:
    return ABSTAIN_TEXT if spec.decide(row) else row.prediction


def evaluate_gate(spec: GateSpec, rows: Sequence[GateRow]) -> GateOutcome:
    baseline_correct = 0
    gated_correct = 0
    abstentions = 0
    model_abstentions = 0
    added_abstentions = 0
    true_positive = 0
    false_positive = 0
    unanswerable = 0
    gained: list[str] = []
    lost: list[str] = []

    for row in rows:
        base_ok = row.baseline_correct
        forced = spec.decide(row)
        text = ABSTAIN_TEXT if forced else row.prediction
        gate_ok = SCORER.normalize(text) in row.accepted
        baseline_correct += base_ok
        gated_correct += gate_ok
        if not row.answerable:
            unanswerable += 1
        if row.abstained_by_model:
            model_abstentions += 1
        if forced and not row.abstained_by_model:
            added_abstentions += 1
        if is_abstention(text):
            abstentions += 1
            if row.answerable:
                false_positive += 1
            else:
                true_positive += 1
        if gate_ok and not base_ok:
            gained.append(row.row_id)
        if base_ok and not gate_ok:
            lost.append(row.row_id)

    b = len(gained)
    c = len(lost)
    return GateOutcome(
        rows=len(rows),
        baseline_correct=baseline_correct,
        gated_correct=gated_correct,
        delta=gated_correct - baseline_correct,
        abstentions=abstentions,
        model_abstentions=model_abstentions,
        added_abstentions=added_abstentions,
        abstention_true_positives=true_positive,
        abstention_false_positives=false_positive,
        abstention_precision=(true_positive / abstentions) if abstentions else 0.0,
        abstention_recall=(true_positive / unanswerable) if unanswerable else 0.0,
        unanswerable_rows=unanswerable,
        gained_ids=tuple(sorted(gained)),
        lost_ids=tuple(sorted(lost)),
        mcnemar_b=b,
        mcnemar_c=c,
        mcnemar_p=mcnemar_exact_p(b, c),
    )


IDENTITY_GATE = GateSpec(signal="first_token_logprob_margin", threshold=-math.inf)


def oracle_ceiling(rows: Sequence[GateRow]) -> int:
    """Accuracy if abstention were perfect: keep every correct answerable row."""

    total = 0
    for row in rows:
        if row.answerable:
            total += row.baseline_correct
        else:
            total += row.abstain_correct
    return total

