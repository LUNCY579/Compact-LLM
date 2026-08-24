"""Torch-free Cornerstone-1 state, ranking, and accounting primitives."""

from .allocation import (
    AdaptiveBudget,
    SignificanceDiagnostics,
    allocate_global_budget,
    significance_diagnostics,
)
from .budget import BudgetLedger, BudgetReport, CacheGeometry
from .counts import CountTransformContext, transform_counts
from .ranking import Selection, normalized_rank, stable_topk
from .scoring import realized_exposure, reduce_gqa_attention
from .state import SignificanceState, StateInvariantError, atomic_gather

__all__ = [
    "AdaptiveBudget",
    "BudgetLedger",
    "BudgetReport",
    "CacheGeometry",
    "CountTransformContext",
    "Selection",
    "SignificanceDiagnostics",
    "SignificanceState",
    "StateInvariantError",
    "allocate_global_budget",
    "atomic_gather",
    "normalized_rank",
    "realized_exposure",
    "reduce_gqa_attention",
    "significance_diagnostics",
    "stable_topk",
    "transform_counts",
]
