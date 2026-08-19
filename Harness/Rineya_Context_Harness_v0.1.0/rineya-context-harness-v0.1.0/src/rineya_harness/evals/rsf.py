from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from pydantic import Field

from rineya_harness.domain.models import StrictModel


class RSFTrial(StrictModel):
    case_id: str
    before_active_ids: set[str] = Field(default_factory=set)
    after_active_ids: set[str] = Field(default_factory=set)
    expected_retained_ids: set[str] = Field(default_factory=set)
    expected_evicted_ids: set[str] = Field(default_factory=set)
    later_needed_ids: set[str] = Field(default_factory=set)
    restored_ids: set[str] = Field(default_factory=set)
    pinned_ids: set[str] = Field(default_factory=set)
    before_estimated_tokens: int = Field(default=0, ge=0)
    after_estimated_tokens: int = Field(default=0, ge=0)
    task_success: bool = True


class RSFMetrics(StrictModel):
    trials: int
    task_success_rate: float
    obsolete_eviction_precision: float
    obsolete_eviction_recall: float
    future_utility_recall: float
    deletion_regret_rate: float
    invariant_retention_rate: float
    mean_active_item_reduction: float
    mean_token_reduction: float


@dataclass(slots=True)
class RSFEvaluator:
    """Metric implementation for recursive-state-forgetting ablations."""

    def evaluate(self, trials: Iterable[RSFTrial]) -> RSFMetrics:
        rows = list(trials)
        if not rows:
            return RSFMetrics(
                trials=0,
                task_success_rate=0.0,
                obsolete_eviction_precision=0.0,
                obsolete_eviction_recall=0.0,
                future_utility_recall=0.0,
                deletion_regret_rate=0.0,
                invariant_retention_rate=0.0,
                mean_active_item_reduction=0.0,
                mean_token_reduction=0.0,
            )
        evicted_total = 0
        correct_evicted = 0
        expected_evict_total = 0
        needed_total = 0
        needed_retained = 0
        regret_numerator = 0
        pinned_total = 0
        pinned_retained = 0
        item_reductions: list[float] = []
        token_reductions: list[float] = []
        for trial in rows:
            actual_evicted = trial.before_active_ids - trial.after_active_ids
            evicted_total += len(actual_evicted)
            correct_evicted += len(actual_evicted & trial.expected_evicted_ids)
            expected_evict_total += len(trial.expected_evicted_ids)
            needed_total += len(trial.later_needed_ids)
            needed_retained += len(trial.later_needed_ids & trial.after_active_ids)
            regret_numerator += len(actual_evicted & trial.later_needed_ids)
            pinned_total += len(trial.pinned_ids)
            pinned_retained += len(trial.pinned_ids & trial.after_active_ids)
            if trial.before_active_ids:
                item_reductions.append(
                    (len(trial.before_active_ids) - len(trial.after_active_ids))
                    / len(trial.before_active_ids)
                )
            else:
                item_reductions.append(0.0)
            if trial.before_estimated_tokens:
                token_reductions.append(
                    (trial.before_estimated_tokens - trial.after_estimated_tokens)
                    / trial.before_estimated_tokens
                )
            else:
                token_reductions.append(0.0)
        return RSFMetrics(
            trials=len(rows),
            task_success_rate=sum(row.task_success for row in rows) / len(rows),
            obsolete_eviction_precision=correct_evicted / evicted_total if evicted_total else 1.0,
            obsolete_eviction_recall=(
                correct_evicted / expected_evict_total if expected_evict_total else 1.0
            ),
            future_utility_recall=needed_retained / needed_total if needed_total else 1.0,
            deletion_regret_rate=regret_numerator / evicted_total if evicted_total else 0.0,
            invariant_retention_rate=pinned_retained / pinned_total if pinned_total else 1.0,
            mean_active_item_reduction=sum(item_reductions) / len(item_reductions),
            mean_token_reduction=sum(token_reductions) / len(token_reductions),
        )
