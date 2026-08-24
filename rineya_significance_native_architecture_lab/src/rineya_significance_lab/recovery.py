"""Modular, inference-only recovery decisions for Architecture Experiment E.

The controller deliberately receives only a bounded pre-answer risk score and
its named Cornerstone-1 components.  It never receives a prediction, answer,
evidence annotation, or correctness label.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, Protocol, TypeVar


class RecoveryAction(StrEnum):
    """Runtime paths currently exposed by the bounded prototype."""

    ACCEPT = "accept"
    RECOMPUTE_LARGER_CACHE = "recompute_larger_cache"
    RECOMPUTE_FULL_CACHE = "recompute_full_cache"


@dataclass(frozen=True, slots=True)
class RecoverySignal:
    """Answer-blind signal presented to a :class:`RecoveryController`."""

    case_id: str
    cluster_id: str
    risk_score: float
    components: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.case_id or not self.cluster_id:
            raise ValueError("case_id and cluster_id must be non-empty")
        if not math.isfinite(self.risk_score) or not 0.0 <= self.risk_score <= 1.0:
            raise ValueError("risk_score must be finite and lie in [0, 1]")
        for name, value in self.components.items():
            if not name or not math.isfinite(float(value)):
                raise ValueError("risk components need non-empty names and finite values")
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"risk component {name!r} must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class RecoveryDecision:
    """Auditable controller output made before answer decoding."""

    action: RecoveryAction
    risk_score: float
    threshold: float
    controller_id: str
    reason: str


class RecoveryController(Protocol):
    """Minimal Experiment-E architecture interface."""

    def decide(self, signal: RecoverySignal) -> RecoveryDecision:
        """Choose the cheap accept path or an explicit recovery action."""


@dataclass(frozen=True, slots=True)
class ThresholdRecoveryController:
    """Fixed, deterministic controller used before any learned controller."""

    threshold: float
    recovery_action: RecoveryAction = RecoveryAction.RECOMPUTE_FULL_CACHE
    controller_id: str = "cornerstone-risk-threshold-v1"

    def __post_init__(self) -> None:
        if not math.isfinite(self.threshold):
            raise ValueError("threshold must be finite")
        if self.recovery_action is RecoveryAction.ACCEPT:
            raise ValueError("recovery_action cannot be ACCEPT")
        if not self.controller_id:
            raise ValueError("controller_id must be non-empty")

    def decide(self, signal: RecoverySignal) -> RecoveryDecision:
        recover = signal.risk_score >= self.threshold
        action = self.recovery_action if recover else RecoveryAction.ACCEPT
        relation = ">=" if recover else "<"
        return RecoveryDecision(
            action=action,
            risk_score=signal.risk_score,
            threshold=self.threshold,
            controller_id=self.controller_id,
            reason=f"risk_score {relation} fixed threshold",
        )


PayloadT = TypeVar("PayloadT")


@dataclass(frozen=True, slots=True)
class RecoveryExecution(Generic[PayloadT]):
    """Path result that makes accidental recomputation on accept observable."""

    decision: RecoveryDecision
    output: PayloadT
    recomputation_calls: int


def execute_recovery_decision(
    decision: RecoveryDecision,
    accepted_output: PayloadT,
    *,
    recompute: Callable[[RecoveryAction], PayloadT],
) -> RecoveryExecution[PayloadT]:
    """Execute exactly one selected path.

    The accept branch returns the original object without copying or invoking
    the fallback.  This literal identity property is stronger than an equality
    check and is used by the mechanism smoke.
    """

    if decision.action is RecoveryAction.ACCEPT:
        return RecoveryExecution(decision, accepted_output, 0)
    output = recompute(decision.action)
    return RecoveryExecution(decision, output, 1)
