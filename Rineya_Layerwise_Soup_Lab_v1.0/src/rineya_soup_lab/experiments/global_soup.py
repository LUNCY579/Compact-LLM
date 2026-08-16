from __future__ import annotations

import math
from typing import Any, Iterable

from ..candidates import CandidateManager, CandidateResult
from ..model.layer_map import MergeRecipe
from ..run_context import RunContext
from ..utils import write_csv, write_json
from .common import accept_change, compare_candidates, rank_candidates


def _unique_alphas(values: Iterable[float]) -> list[float]:
    return sorted({round(max(0.0, min(1.0, float(value))), 8) for value in values})


def _recipe(alpha: float, label: str | None = None) -> MergeRecipe:
    return MergeRecipe(
        global_alpha=float(alpha),
        label=label or f"global_alpha_{alpha:.4f}",
        metadata={"experiment": "global_interpolation"},
    )


def evaluate_global_alpha(
    context: RunContext,
    manager: CandidateManager,
    alpha: float,
    *,
    alias: str | None = None,
    retain: bool = False,
) -> CandidateResult:
    if math.isclose(alpha, 0.0, abs_tol=1e-12):
        result = manager.load_result("anchor")
        if alias:
            context.state.set_alias(alias, result.candidate_id)
        return result
    if math.isclose(alpha, 1.0, abs_tol=1e-12):
        result = manager.load_result("greedy")
        if alias:
            context.state.set_alias(alias, result.candidate_id)
        return result
    label = alias or f"global_alpha_{alpha:.4f}"
    return manager.evaluate(_recipe(alpha, label), alias=alias, retain=retain)


def run_uniform(context: RunContext, manager: CandidateManager) -> CandidateResult | None:
    settings = context.config.get("experiments.uniform")
    if not settings.get("enabled", True):
        return None
    alpha = float(settings.get("alpha", 0.5))
    result = evaluate_global_alpha(context, manager, alpha, alias="uniform", retain=True)
    write_json(context.path("analysis", "global", "uniform.json"), result.to_record())
    return result


def run_global_sweep(context: RunContext, manager: CandidateManager) -> CandidateResult | None:
    settings = context.config.get("experiments.global_sweep")
    if not settings.get("enabled", True):
        return None
    candidates: dict[float, CandidateResult] = {}
    for alpha in _unique_alphas(settings.get("alphas", [])):
        candidates[alpha] = evaluate_global_alpha(context, manager, alpha)
    first_best = rank_candidates(list(candidates.values()))[0]
    first_alpha = float(first_best.recipe.global_alpha)
    if settings.get("refine_around_best", True):
        radius = max(0.0, float(settings.get("refine_radius", 0.10)))
        step = float(settings.get("refine_step", 0.025))
        if step <= 0:
            raise ValueError("experiments.global_sweep.refine_step must be positive")
        count = int(math.floor(radius / step))
        refined = [first_alpha + offset * step for offset in range(-count, count + 1)]
        for alpha in _unique_alphas(refined):
            if alpha not in candidates:
                candidates[alpha] = evaluate_global_alpha(context, manager, alpha)
    ranked = rank_candidates(list(candidates.values()))
    raw_best = ranked[0]
    best_endpoint = max(
        (manager.load_result("anchor"), manager.load_result("greedy")),
        key=lambda item: item.objective.score,
    )
    if raw_best.candidate_id == best_endpoint.candidate_id:
        best = best_endpoint
        robustly_accepted = True
        comparison = compare_candidates(
            best,
            best_endpoint,
            objective_config=context.config.get("objective"),
            statistics_config=context.config.get("statistics"),
            seed_offset=111,
        )
    else:
        robustly_accepted, comparison = accept_change(
            raw_best,
            best_endpoint,
            objective_config=context.config.get("objective"),
            statistics_config=context.config.get("statistics"),
            seed_offset=111,
        )
        best = raw_best if robustly_accepted else best_endpoint
    # Give the uncertainty-filtered numerical candidate a stable alias and retain its weights.
    context.state.set_alias("global_best", best.candidate_id)
    if best.recipe.kind != "endpoint":
        manager.ensure_materialized(best.candidate_id)
    best = manager.load_result(best.candidate_id)
    selected_comparison = compare_candidates(
        best,
        best_endpoint,
        objective_config=context.config.get("objective"),
        statistics_config=context.config.get("statistics"),
        seed_offset=112,
    )
    rows: list[dict[str, Any]] = []
    for result in sorted(
        {item.candidate_id: item for item in candidates.values()}.values(),
        key=lambda item: float(item.recipe.global_alpha),
    ):
        rows.append(
            {
                "alpha": float(result.recipe.global_alpha),
                "candidate_id": result.candidate_id,
                "label": result.label,
                "objective": result.objective.score,
                "mean_quality": result.evidence.summary.get("mean_quality_score"),
                "worst_decile_quality": result.evidence.summary.get(
                    "worst_decile_mean_quality_score"
                ),
                "complete_utterance_rate": result.evidence.summary.get(
                    "complete_utterance_rate"
                ),
                "median_reply_tokens": result.evidence.summary.get("median_reply_tokens"),
                "duplicate_reply_rate": result.evidence.summary.get("duplicate_reply_rate"),
            }
        )
    write_csv(context.path("analysis", "global", "interpolation_curve.csv"), rows)
    write_json(
        context.path("analysis", "global", "global_selection.json"),
        {
            "selected": best.to_record(),
            "raw_objective_winner": raw_best.to_record(),
            "best_endpoint": best_endpoint.to_record(),
            "raw_winner_robustly_accepted": robustly_accepted,
            "raw_winner_minus_best_endpoint": comparison.to_dict(),
            "selected_minus_best_endpoint": selected_comparison.to_dict(),
            "normalization": {
                "policy": "predeclared objective transforms and weights",
                "objective_config": context.config.get("objective"),
                "noise_control": "paired bootstrap on stable case IDs",
            },
            "ranking": [result.to_record() for result in ranked],
        },
    )
    return best
