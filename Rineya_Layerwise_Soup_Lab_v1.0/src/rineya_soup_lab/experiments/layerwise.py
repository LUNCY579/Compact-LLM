from __future__ import annotations

import itertools
import math
from collections import defaultdict
from typing import Any, Mapping

from ..analysis.bootstrap import paired_bootstrap_objective_difference
from ..analysis.delta_correlation import (
    build_delta_influence_correlation_report,
    join_layer_delta_and_influence,
)
from ..analysis.layer_influence import (
    LayerInfluenceSummary,
    normalized_layer_importance,
    smooth_layer_targets,
    summarize_bidirectional_layer,
)
from ..candidates import CandidateManager, CandidateResult
from ..model.layer_map import MergeRecipe
from ..run_context import RunContext
from ..utils import read_json, write_csv, write_json, write_jsonl
from .common import accept_change, rank_candidates


def _layer_indices(context: RunContext) -> list[int]:
    compatibility_path = context.path("manifests", "compatibility.json")
    if compatibility_path.exists():
        indices = [int(value) for value in read_json(compatibility_path).get("layer_indices", [])]
    else:
        expected = context.config.get("models.expected_layers")
        indices = list(range(int(expected))) if expected is not None else []
    requested = context.config.get("experiments.layer_influence.layers", "all")
    if requested == "all":
        return indices
    selected = sorted({int(value) for value in requested})
    unknown = sorted(set(selected) - set(indices))
    if unknown:
        raise ValueError(f"Requested layer indices do not exist: {unknown}")
    return selected


def _with_layer_alpha(
    global_alpha: float,
    layer_alphas: Mapping[int, float],
    layer: int,
    alpha: float,
    *,
    label: str,
    experiment: str,
) -> MergeRecipe:
    updated = {int(key): float(value) for key, value in layer_alphas.items()}
    if math.isclose(float(alpha), float(global_alpha), abs_tol=1e-12):
        updated.pop(int(layer), None)
    else:
        updated[int(layer)] = float(alpha)
    return MergeRecipe(
        global_alpha=float(global_alpha),
        layer_alphas=updated,
        label=label,
        metadata={"experiment": experiment, "layer": int(layer), "layer_alpha": float(alpha)},
    )


def run_layer_influence(
    context: RunContext,
    manager: CandidateManager,
) -> CandidateResult | None:
    settings = context.config.get("experiments.layer_influence")
    if not settings.get("enabled", True):
        return None
    layers = _layer_indices(context)
    if not layers:
        raise RuntimeError("No transformer layers were discovered for layer-wise analysis.")
    global_best = manager.load_result("global_best")
    anchor = manager.load_result("anchor")
    greedy = manager.load_result("greedy")
    global_alpha = float(global_best.recipe.global_alpha)
    objective_config = context.config.get("objective")
    statistics_config = context.config.get("statistics")
    bootstrap_samples = int(statistics_config.get("bootstrap_samples", 2000))
    confidence_level = float(statistics_config.get("confidence_level", 0.95))
    seed = int(statistics_config.get("seed", 0))
    centered_grid = sorted(
        {max(0.0, min(1.0, float(value))) for value in settings.get("centered_alpha_grid", [])}
        | {global_alpha}
    )
    bidirectional = bool(settings.get("bidirectional_swaps", True))

    intervention_summaries: list[LayerInfluenceSummary] = []
    intervention_matrix: list[dict[str, Any]] = []
    centered_rows: list[dict[str, Any]] = []
    centered_candidates: dict[int, list[CandidateResult]] = defaultdict(list)
    centered_best: dict[int, CandidateResult] = {}
    centered_reliability: dict[int, float] = {}

    for layer in layers:
        forward_recipe = MergeRecipe(
            global_alpha=0.0,
            layer_alphas={layer: 1.0},
            label=f"swap_A_to_G_L{layer:02d}",
            metadata={"experiment": "bidirectional_layer_swap", "direction": "A_to_G", "layer": layer},
        )
        forward = manager.evaluate(forward_recipe)
        reverse: CandidateResult | None = None
        if bidirectional:
            reverse_recipe = MergeRecipe(
                global_alpha=1.0,
                layer_alphas={layer: 0.0},
                label=f"swap_G_to_A_L{layer:02d}",
                metadata={"experiment": "bidirectional_layer_swap", "direction": "G_to_A", "layer": layer},
            )
            reverse = manager.evaluate(reverse_recipe)
        summary, matrix = summarize_bidirectional_layer(
            layer=layer,
            anchor_to_greedy_rows=forward.evidence.rows,
            anchor_rows=anchor.evidence.rows,
            greedy_to_anchor_rows=reverse.evidence.rows if reverse else None,
            greedy_rows=greedy.evidence.rows if reverse else None,
            objective_config=objective_config,
            bootstrap_samples=bootstrap_samples,
            confidence_level=confidence_level,
            seed=seed,
        )
        intervention_summaries.append(summary)
        intervention_matrix.extend(matrix)

        for alpha in centered_grid:
            if math.isclose(alpha, global_alpha, abs_tol=1e-12):
                result = global_best
            else:
                result = manager.evaluate(
                    _with_layer_alpha(
                        global_alpha,
                        {},
                        layer,
                        alpha,
                        label=f"centered_L{layer:02d}_a{alpha:.4f}",
                        experiment="single_layer_centered_interpolation",
                    )
                )
            centered_candidates[layer].append(result)
            centered_rows.append(
                {
                    "layer": layer,
                    "alpha": alpha,
                    "candidate_id": result.candidate_id,
                    "objective": result.objective.score,
                    "objective_delta_vs_global": result.objective.score - global_best.objective.score,
                    "mean_quality": result.evidence.summary.get("mean_quality_score"),
                    "worst_decile_quality": result.evidence.summary.get(
                        "worst_decile_mean_quality_score"
                    ),
                    "median_reply_tokens": result.evidence.summary.get("median_reply_tokens"),
                }
            )
        best = rank_candidates(centered_candidates[layer])[0]
        centered_best[layer] = best
        if best.candidate_id == global_best.candidate_id:
            centered_reliability[layer] = 0.0
        else:
            comparison = paired_bootstrap_objective_difference(
                best.evidence.rows,
                global_best.evidence.rows,
                objective_config,
                samples=bootstrap_samples,
                confidence_level=confidence_level,
                seed=seed + 10000 + layer,
            )
            magnitude = max(0.0, comparison.observed_delta)
            uncertainty = max(1e-12, (comparison.ci_high - comparison.ci_low) / 2.0)
            centered_reliability[layer] = max(
                0.0,
                min(
                    1.0,
                    comparison.probability_improvement
                    * magnitude
                    / (magnitude + uncertainty),
                ),
            )

    importance = normalized_layer_importance(intervention_summaries)
    influence_by_layer = {item.layer: item for item in intervention_summaries}
    minimum_reliability = float(settings.get("minimum_reliability", 0.0))
    targets: dict[int, float] = {}
    reliabilities: dict[int, float] = {}
    for layer in layers:
        target = float(centered_best[layer].recipe.layer_alphas.get(layer, global_alpha))
        # Centered selection answers which coefficient works around the global solution;
        # bidirectional interventions answer whether this layer has stable behavioral signal.
        intervention_reliability = influence_by_layer[layer].reliability
        reliability = math.sqrt(max(0.0, centered_reliability[layer] * intervention_reliability))
        reliability *= min(1.0, importance.get(layer, 0.0) * len(layers) + 0.25)
        reliability = max(0.0, min(1.0, reliability))
        if reliability < minimum_reliability:
            target = global_alpha
            reliability = 0.0
        targets[layer] = target
        reliabilities[layer] = reliability

    smoothed = smooth_layer_targets(
        targets,
        reliabilities,
        global_alpha=global_alpha,
        smoothing_penalty=float(settings.get("smoothing_penalty", 0.05)),
        global_shrinkage_penalty=float(settings.get("global_shrinkage_penalty", 0.10)),
    )
    # Omit exact-global entries so the recipe remains compact and interpretable.
    layer_alphas = {
        layer: alpha
        for layer, alpha in smoothed.items()
        if not math.isclose(alpha, global_alpha, abs_tol=1e-9)
    }
    seed_recipe = MergeRecipe(
        global_alpha=global_alpha,
        layer_alphas=layer_alphas,
        label="layerwise_seed",
        metadata={
            "experiment": "layer_adaptive_behavioral_soup",
            "selection_basis": "centered single-layer interpolation + bidirectional intervention + shrinkage",
        },
    )
    seed_result = manager.evaluate(seed_recipe, alias="layerwise_seed", retain=True)

    influence_rows = [
        {
            **item.to_dict(),
            "normalized_importance": importance.get(item.layer, 0.0),
            "centered_best_alpha": targets[item.layer],
            "centered_reliability": centered_reliability[item.layer],
            "combined_reliability": reliabilities[item.layer],
            "smoothed_seed_alpha": smoothed[item.layer],
        }
        for item in intervention_summaries
    ]
    write_csv(
        context.path("analysis", "layers", "layer_influence_summary.csv"),
        influence_rows,
    )

    compatibility = read_json(context.path("manifests", "compatibility.json"))
    delta_influence_rows = join_layer_delta_and_influence(
        compatibility.get("parameter_delta_statistics") or [],
        influence_rows,
    )
    write_csv(
        context.path("analysis", "layers", "delta_influence_join.csv"),
        delta_influence_rows,
    )
    correlation_samples = max(1000, min(10000, bootstrap_samples))
    delta_influence_report = build_delta_influence_correlation_report(
        delta_influence_rows,
        permutations=correlation_samples,
        bootstrap_samples=correlation_samples,
        confidence_level=confidence_level,
        seed=seed + 40000,
    )
    write_json(
        context.path("analysis", "layers", "delta_influence_correlation.json"),
        delta_influence_report,
    )
    write_jsonl(
        context.path("analysis", "layers", "per_response_layer_influence.jsonl"),
        intervention_matrix,
    )
    category_values: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in intervention_matrix:
        category_values[(int(row["layer"]), str(row["category"]))].append(
            float(row["combined_greedy_advantage"])
        )
    category_rows = [
        {
            "layer": layer,
            "category": category,
            "case_count": len(values),
            "mean_greedy_advantage": sum(values) / len(values),
            "positive_rate": sum(value > 0 for value in values) / len(values),
            "negative_rate": sum(value < 0 for value in values) / len(values),
        }
        for (layer, category), values in sorted(category_values.items())
    ]
    write_csv(
        context.path("analysis", "layers", "layer_influence_by_category.csv"),
        category_rows,
    )
    write_csv(
        context.path("analysis", "layers", "centered_layer_grid.csv"), centered_rows
    )
    write_json(
        context.path("analysis", "layers", "layerwise_seed_selection.json"),
        {
            "global_baseline": global_best.to_record(),
            "layers": layers,
            "centered_alpha_grid": centered_grid,
            "targets_before_smoothing": {str(key): value for key, value in targets.items()},
            "reliabilities": {str(key): value for key, value in reliabilities.items()},
            "smoothed_alphas": {str(key): value for key, value in smoothed.items()},
            "recipe": seed_recipe.to_dict(),
            "seed_result": seed_result.to_record(),
            "delta_influence_diagnostic": delta_influence_report,
            "methodological_boundary": (
                "Layer influence is an intervention-based effect on observed evaluation utility; "
                "it is not claimed to be a unique causal decomposition of a response. Parameter "
                "delta magnitude is diagnostic only and is never used to select coefficients."
            ),
        },
    )
    return seed_result


def run_layer_search(
    context: RunContext,
    manager: CandidateManager,
) -> CandidateResult | None:
    settings = context.config.get("experiments.layer_search")
    if not settings.get("enabled", True):
        return None
    if not settings.get("coordinate_descent", True):
        context.state.set_alias("layerwise_best", context.state.resolve_alias("layerwise_seed"))
        return manager.load_result("layerwise_seed")

    current = manager.load_result("layerwise_seed")
    global_alpha = float(current.recipe.global_alpha)
    current_alphas = {int(key): float(value) for key, value in current.recipe.layer_alphas.items()}
    influence_path = context.path("analysis", "layers", "layer_influence_summary.csv")
    import csv

    with influence_path.open("r", encoding="utf-8", newline="") as handle:
        influence_rows = list(csv.DictReader(handle))
    order = [
        int(row["layer"])
        for row in sorted(
            influence_rows,
            key=lambda row: float(row.get("normalized_importance") or 0.0),
            reverse=True,
        )
    ]
    top_only = settings.get("top_layers_only")
    if top_only is not None:
        order = order[: max(0, int(top_only))]
    alpha_grid = sorted(
        {max(0.0, min(1.0, float(value))) for value in settings.get("alpha_grid", [])}
    )
    max_evaluations = max(0, int(settings.get("max_candidate_evaluations", 120)))
    max_passes = max(1, int(settings.get("max_passes", 2)))
    evaluations = 0
    trace: list[dict[str, Any]] = []

    for pass_index in range(max_passes):
        accepted_this_pass = 0
        for layer in order:
            if evaluations >= max_evaluations:
                break
            trials: list[CandidateResult] = [current]
            current_alpha = float(current_alphas.get(layer, global_alpha))
            for alpha in alpha_grid:
                if math.isclose(alpha, current_alpha, abs_tol=1e-12):
                    continue
                if evaluations >= max_evaluations:
                    break
                recipe = _with_layer_alpha(
                    global_alpha,
                    current_alphas,
                    layer,
                    alpha,
                    label=f"cd_p{pass_index + 1}_L{layer:02d}_a{alpha:.4f}",
                    experiment="layerwise_coordinate_descent",
                )
                trials.append(manager.evaluate(recipe))
                evaluations += 1
            best_trial = rank_candidates(trials)[0]
            accepted = False
            comparison: dict[str, Any] | None = None
            if best_trial.candidate_id != current.candidate_id:
                accepted, boot = accept_change(
                    best_trial,
                    current,
                    objective_config=context.config.get("objective"),
                    statistics_config=context.config.get("statistics"),
                    seed_offset=20000 + pass_index * 1000 + layer,
                )
                comparison = boot.to_dict()
            trace.append(
                {
                    "pass": pass_index + 1,
                    "layer": layer,
                    "current_alpha_before": current_alpha,
                    "current_candidate_before": current.candidate_id,
                    "trial_candidates": [
                        {
                            "candidate_id": trial.candidate_id,
                            "alpha": float(trial.recipe.layer_alphas.get(layer, global_alpha)),
                            "objective": trial.objective.score,
                        }
                        for trial in trials
                    ],
                    "best_trial": best_trial.candidate_id,
                    "accepted": accepted,
                    "comparison": comparison,
                }
            )
            if accepted:
                current = best_trial
                selected_alpha = float(current.recipe.layer_alphas.get(layer, global_alpha))
                if math.isclose(selected_alpha, global_alpha, abs_tol=1e-12):
                    current_alphas.pop(layer, None)
                else:
                    current_alphas[layer] = selected_alpha
                accepted_this_pass += 1
        if evaluations >= max_evaluations or accepted_this_pass == 0:
            break

    pairwise_top_k = max(0, int(settings.get("pairwise_top_k", 0)))
    if pairwise_top_k >= 2 and evaluations < max_evaluations:
        top_layers = order[:pairwise_top_k]
        for layer_a, layer_b in itertools.combinations(top_layers, 2):
            if evaluations >= max_evaluations:
                break
            trials = [current]
            current_a = float(current_alphas.get(layer_a, global_alpha))
            current_b = float(current_alphas.get(layer_b, global_alpha))
            for alpha_a, alpha_b in itertools.product(alpha_grid, repeat=2):
                if evaluations >= max_evaluations:
                    break
                if math.isclose(alpha_a, current_a) and math.isclose(alpha_b, current_b):
                    continue
                updated = dict(current_alphas)
                for layer, alpha in ((layer_a, alpha_a), (layer_b, alpha_b)):
                    if math.isclose(alpha, global_alpha, abs_tol=1e-12):
                        updated.pop(layer, None)
                    else:
                        updated[layer] = alpha
                recipe = MergeRecipe(
                    global_alpha=global_alpha,
                    layer_alphas=updated,
                    label=f"pair_L{layer_a:02d}_{alpha_a:.2f}_L{layer_b:02d}_{alpha_b:.2f}",
                    metadata={"experiment": "pairwise_layer_refinement", "layers": [layer_a, layer_b]},
                )
                trials.append(manager.evaluate(recipe))
                evaluations += 1
            best_trial = rank_candidates(trials)[0]
            accepted = False
            comparison = None
            if best_trial.candidate_id != current.candidate_id:
                accepted, boot = accept_change(
                    best_trial,
                    current,
                    objective_config=context.config.get("objective"),
                    statistics_config=context.config.get("statistics"),
                    seed_offset=30000 + layer_a * 100 + layer_b,
                )
                comparison = boot.to_dict()
            trace.append(
                {
                    "stage": "pairwise",
                    "layers": [layer_a, layer_b],
                    "current_candidate_before": current.candidate_id,
                    "best_trial": best_trial.candidate_id,
                    "accepted": accepted,
                    "comparison": comparison,
                }
            )
            if accepted:
                current = best_trial
                current_alphas = {
                    int(key): float(value) for key, value in current.recipe.layer_alphas.items()
                }

    context.state.set_alias("layerwise_best", current.candidate_id)
    if current.recipe.kind != "endpoint":
        manager.ensure_materialized(current.candidate_id)
    current = manager.load_result(current.candidate_id)
    write_jsonl(context.path("analysis", "layers", "coordinate_search_trace.jsonl"), trace)
    write_json(
        context.path("analysis", "layers", "layerwise_best_selection.json"),
        {
            "selected": current.to_record(),
            "candidate_evaluations": evaluations,
            "max_candidate_evaluations": max_evaluations,
            "passes_requested": max_passes,
            "layer_order": order,
            "trace_path": str(
                context.path("analysis", "layers", "coordinate_search_trace.jsonl")
            ),
        },
    )
    return current
