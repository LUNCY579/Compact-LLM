from __future__ import annotations

"""Pure helpers shared by the Phase 2C ranking, orchestration, gate, and tests."""

import math
from pathlib import Path
from typing import Any, Iterable


def minmax(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    low = min(values.values())
    high = max(values.values())
    if math.isclose(low, high, rel_tol=0.0, abs_tol=1e-12):
        return {key: 0.5 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


def combine_importance(
    rows: list[dict[str, Any]],
    *,
    gradient_weight: float = 0.35,
    ablation_weight: float = 0.45,
    structural_weight: float = 0.20,
) -> list[dict[str, Any]]:
    """Normalize measured signals and return highest-importance layers first.

    Missing structural evidence is allowed. Weights are renormalized per layer,
    so an alternate Phase 2B output root can still be ranked from gradient and
    ablation measurements without inventing zeros.
    """

    gradient = {int(row["current_layer_id"]): float(row["gradient_taylor_score"]) for row in rows}
    ablation = {int(row["current_layer_id"]): max(0.0, float(row["ablation_nll_increase"])) for row in rows}
    structural = {
        int(row["current_layer_id"]): max(0.0, float(row["structural_sensitivity"]))
        for row in rows
        if row.get("structural_sensitivity") is not None
    }
    gradient_n = minmax(gradient)
    ablation_n = minmax(ablation)
    structural_n = minmax(structural)

    combined: list[dict[str, Any]] = []
    for row in rows:
        layer = int(row["current_layer_id"])
        components: list[tuple[float, float]] = [
            (gradient_weight, gradient_n[layer]),
            (ablation_weight, ablation_n[layer]),
        ]
        if layer in structural_n:
            components.append((structural_weight, structural_n[layer]))
        denominator = sum(weight for weight, _ in components)
        score = sum(weight * value for weight, value in components) / max(denominator, 1e-12)
        combined.append(
            {
                **row,
                "normalized_gradient_taylor": gradient_n[layer],
                "normalized_ablation_nll": ablation_n[layer],
                "normalized_structural_sensitivity": structural_n.get(layer),
                "combined_importance_score": score,
            }
        )
    combined.sort(
        key=lambda row: (
            float(row["combined_importance_score"]),
            float(row["ablation_nll_increase"]),
            float(row["gradient_taylor_score"]),
            -int(row["current_layer_id"]),
        ),
        reverse=True,
    )
    for rank, row in enumerate(combined, 1):
        row["importance_rank"] = rank
    return combined


def complementary_layers(ranked_layer_ids: list[int], layer_count: int, count: int = 4) -> list[int]:
    """Choose high-ranked layers across depth regions instead of one cluster."""

    if count <= 0 or layer_count <= 0:
        return []
    selected: list[int] = []
    for bucket in range(count):
        start = math.floor(bucket * layer_count / count)
        end = math.floor((bucket + 1) * layer_count / count)
        candidates = [layer for layer in ranked_layer_ids if start <= layer < max(start + 1, end)]
        if candidates:
            selected.append(candidates[0])
    for layer in ranked_layer_ids:
        if len(selected) >= min(count, layer_count):
            break
        if layer not in selected:
            selected.append(layer)
    return sorted(selected)


def candidate_specs(ranking: list[dict[str, Any]], seed: int = 11) -> list[dict[str, Any]]:
    """Build the required selected-layer controls for one reduced architecture."""

    import random

    ranked_ids = [int(row["current_layer_id"]) for row in ranking]
    layer_count = len(ranked_ids)
    if layer_count == 0:
        raise ValueError("Cannot build Phase 2C candidates from an empty ranking.")
    top = lambda n: sorted(ranked_ids[: min(n, layer_count)])
    low4 = sorted(ranked_ids[-min(4, layer_count) :])
    rng = random.Random(seed + 2203)
    random4 = sorted(rng.sample(sorted(ranked_ids), min(4, layer_count)))
    comp4 = complementary_layers(ranked_ids, layer_count, 4)
    specs = [
        {"name": "all_remaining", "layers": "all", "module_group": "both", "control": "quality_reference"},
        {"name": "top1", "layers": top(1), "module_group": "both", "control": "minimum_layer_budget"},
        {"name": "top2", "layers": top(2), "module_group": "both", "control": "very_low_resource"},
        {"name": "top4", "layers": top(4), "module_group": "both", "control": "primary_selective_candidate"},
        {"name": "top8", "layers": top(8), "module_group": "both", "control": "intermediate_capacity"},
        {"name": "complementary4", "layers": comp4, "module_group": "both", "control": "depth_distributed"},
        {"name": "random4", "layers": random4, "module_group": "both", "control": "parameter_count_control"},
        {"name": "low4", "layers": low4, "module_group": "both", "control": "importance_ranking_control"},
        {"name": "top4_attention", "layers": top(4), "module_group": "attention", "control": "module_group_ablation"},
        {"name": "top4_mlp", "layers": top(4), "module_group": "mlp", "control": "module_group_ablation"},
    ]
    # Deduplicate configurations that collapse on very shallow models while
    # retaining the first, most interpretable name.
    seen: set[tuple[str, str]] = set()
    unique: list[dict[str, Any]] = []
    for spec in specs:
        layer_text = spec["layers"] if isinstance(spec["layers"], str) else ",".join(map(str, spec["layers"]))
        key = (str(layer_text), str(spec["module_group"]))
        if key not in seen:
            seen.add(key)
            unique.append(spec)
    return unique


def quality_index(summary: dict[str, Any]) -> float:
    """Transparent development ranking score; hard gates remain separate."""

    reference_f1 = summary.get("mean_reference_token_f1")
    reference_component = float(reference_f1) if reference_f1 is not None else float(summary.get("mean_heuristic_score", 0.0))
    return (
        0.35 * float(summary.get("mean_heuristic_score", 0.0))
        + 0.30 * float(summary.get("semantic_pass_rate", 0.0))
        + 0.20 * reference_component
        + 0.15 * float(summary.get("complete_utterance_rate", 0.0))
    )


def relative_loss(reference: float, candidate: float) -> float:
    return max(0.0, reference - candidate) / max(abs(reference), 1e-12)


def category_quality_scores(summary: dict[str, Any]) -> dict[str, float]:
    """Build category-level quality using the same components as quality_index.

    New evaluator summaries expose category semantic, lexical-reference, and
    completion values. Older summaries remain readable by falling back to the
    heuristic category score instead of inventing a perfect value.
    """

    heuristic = summary.get("category_scores") or {}
    semantic = summary.get("category_semantic_pass_rates") or {}
    reference_f1 = summary.get("category_reference_token_f1") or {}
    completion = summary.get("category_complete_utterance_rates") or {}
    categories = sorted(set(heuristic) | set(semantic) | set(reference_f1) | set(completion))
    scores: dict[str, float] = {}
    for category in categories:
        h = float(heuristic.get(category, 0.0))
        sem = float(semantic.get(category, h))
        f1_value = reference_f1.get(category)
        f1 = h if f1_value is None else float(f1_value)
        complete = float(completion.get(category, h))
        scores[category] = 0.35 * h + 0.30 * sem + 0.20 * f1 + 0.15 * complete
    return scores


def max_category_loss(reference: dict[str, Any], candidate: dict[str, Any]) -> tuple[float, dict[str, float]]:
    losses: dict[str, float] = {}
    reference_scores = category_quality_scores(reference)
    candidate_scores = category_quality_scores(candidate)
    for category, ref_score in reference_scores.items():
        cand_score = candidate_scores.get(category)
        losses[category] = 1.0 if cand_score is None else max(0.0, ref_score - cand_score)
    return (max(losses.values()) if losses else 0.0), losses


def assess_training_candidate(
    *,
    candidate: dict[str, Any],
    all_remaining: dict[str, Any],
    reduced_frozen: dict[str, Any],
    max_quality_index_loss: float,
    max_frozen_quality_loss: float,
    max_heuristic_loss: float,
    max_semantic_pass_loss: float,
    max_reference_f1_loss: float,
    max_category_loss_allowed: float,
    max_frozen_category_loss: float,
    max_nll_increase: float,
    min_complete_utterance_rate: float,
    minimum_trainable_reduction: float,
    max_new_major_issues: int,
) -> dict[str, Any]:
    cand_summary = candidate["generation_summary"]
    all_summary = all_remaining["generation_summary"]
    cand_quality = quality_index(cand_summary)
    all_quality = quality_index(all_summary)
    frozen_quality = quality_index(reduced_frozen["generation_summary"])
    quality_loss = max(0.0, all_quality - cand_quality)
    frozen_quality_loss = max(0.0, frozen_quality - cand_quality)
    heuristic_loss = max(0.0, float(all_summary["mean_heuristic_score"]) - float(cand_summary["mean_heuristic_score"]))
    semantic_loss = max(0.0, float(all_summary.get("semantic_pass_rate", 1.0)) - float(cand_summary.get("semantic_pass_rate", 1.0)))
    all_f1 = all_summary.get("mean_reference_token_f1")
    cand_f1 = cand_summary.get("mean_reference_token_f1")
    reference_f1_loss = 0.0 if all_f1 is None or cand_f1 is None else max(0.0, float(all_f1) - float(cand_f1))
    category_loss, category_losses = max_category_loss(all_summary, cand_summary)
    frozen_category_loss, frozen_category_losses = max_category_loss(
        reduced_frozen["generation_summary"], cand_summary
    )
    all_nll = float(all_remaining["nll_summary"]["mean_target_nll"])
    cand_nll = float(candidate["nll_summary"]["mean_target_nll"])
    nll_increase = max(0.0, cand_nll - all_nll) / max(abs(all_nll), 1e-12)
    all_trainable = int(all_remaining["training_manifest"].get("trainable_parameter_count", 0))
    cand_trainable = int(candidate["training_manifest"].get("trainable_parameter_count", 0))
    trainable_reduction = 0.0 if all_trainable <= 0 else max(0.0, 1.0 - cand_trainable / all_trainable)
    all_runtime = float(all_remaining["training_manifest"].get("train_metrics", {}).get("train_runtime", 0.0))
    cand_runtime = float(candidate["training_manifest"].get("train_metrics", {}).get("train_runtime", 0.0))
    runtime_reduction = None if all_runtime <= 0 or cand_runtime <= 0 else 1.0 - cand_runtime / all_runtime
    all_vram = float(all_remaining["training_manifest"].get("peak_allocated_vram_gb", 0.0))
    cand_vram = float(candidate["training_manifest"].get("peak_allocated_vram_gb", 0.0))
    vram_reduction = None if all_vram <= 0 or cand_vram <= 0 else 1.0 - cand_vram / all_vram
    denominator = all_quality - frozen_quality
    recovered_fraction = None if denominator <= 1e-8 else (cand_quality - frozen_quality) / denominator

    deviation = candidate.get("output_deviation") or {}
    reasons: list[str] = []
    if int(cand_summary.get("critical_failure_count", 0)) > 0:
        reasons.append("critical semantic failures are nonzero")
    if quality_loss > max_quality_index_loss:
        reasons.append(f"quality-index loss {quality_loss:.4f} exceeds {max_quality_index_loss:.4f}")
    if frozen_quality_loss > max_frozen_quality_loss:
        reasons.append(
            f"quality-index loss vs reduced frozen {frozen_quality_loss:.4f} exceeds "
            f"{max_frozen_quality_loss:.4f}"
        )
    if heuristic_loss > max_heuristic_loss:
        reasons.append(f"heuristic loss {heuristic_loss:.4f} exceeds {max_heuristic_loss:.4f}")
    if semantic_loss > max_semantic_pass_loss:
        reasons.append(f"semantic-pass loss {semantic_loss:.4f} exceeds {max_semantic_pass_loss:.4f}")
    if reference_f1_loss > max_reference_f1_loss:
        reasons.append(f"reference-F1 loss {reference_f1_loss:.4f} exceeds {max_reference_f1_loss:.4f}")
    if category_loss > max_category_loss_allowed:
        reasons.append(f"worst category loss {category_loss:.4f} exceeds {max_category_loss_allowed:.4f}")
    if frozen_category_loss > max_frozen_category_loss:
        reasons.append(
            f"worst category loss vs reduced frozen {frozen_category_loss:.4f} exceeds "
            f"{max_frozen_category_loss:.4f}"
        )
    if nll_increase > max_nll_increase:
        reasons.append(f"target-NLL increase {nll_increase:.4f} exceeds {max_nll_increase:.4f}")
    if float(cand_summary.get("complete_utterance_rate", 0.0)) < min_complete_utterance_rate:
        reasons.append("complete-utterance rate is below the floor")
    if int(deviation.get("new_critical_failure_count", 0)) > 0:
        reasons.append("new critical failures appeared relative to the reduced frozen model")
    if int(deviation.get("new_major_issue_count", 0)) > max_new_major_issues:
        reasons.append(
            f"new major issues {int(deviation.get('new_major_issue_count', 0))} exceed "
            f"{max_new_major_issues}"
        )
    if candidate["name"] != "all_remaining" and trainable_reduction < minimum_trainable_reduction:
        reasons.append(
            f"trainable-parameter reduction {trainable_reduction:.4f} is below {minimum_trainable_reduction:.4f}"
        )
    return {
        "eligible": not reasons,
        "rejection_reasons": reasons,
        "quality_index": cand_quality,
        "quality_index_loss_vs_all_remaining": quality_loss,
        "quality_index_loss_vs_reduced_frozen": frozen_quality_loss,
        "quality_gain_vs_reduced_frozen": cand_quality - frozen_quality,
        "fraction_of_all_remaining_gain_recovered": recovered_fraction,
        "heuristic_loss_vs_all_remaining": heuristic_loss,
        "semantic_pass_loss_vs_all_remaining": semantic_loss,
        "reference_f1_loss_vs_all_remaining": reference_f1_loss,
        "worst_category_loss_vs_all_remaining": category_loss,
        "category_losses": category_losses,
        "worst_category_loss_vs_reduced_frozen": frozen_category_loss,
        "frozen_category_losses": frozen_category_losses,
        "new_critical_failures_vs_reduced_frozen": int(deviation.get("new_critical_failure_count", 0)),
        "new_major_issues_vs_reduced_frozen": int(deviation.get("new_major_issue_count", 0)),
        "relative_nll_increase_vs_all_remaining": nll_increase,
        "trainable_parameter_reduction_vs_all_remaining": trainable_reduction,
        "training_runtime_reduction_vs_all_remaining": runtime_reduction,
        "peak_training_vram_reduction_vs_all_remaining": vram_reduction,
    }


def select_profiles(rows: list[dict[str, Any]], *, balanced_tolerance: float) -> dict[str, dict[str, Any]]:
    eligible = [row for row in rows if row.get("assessment", {}).get("eligible")]
    if not eligible:
        raise ValueError("No Phase 2C candidate passed the automated eligibility gates.")
    maximum = max(
        eligible,
        key=lambda row: (
            float(row["assessment"]["quality_index"]),
            -float(row["nll_summary"]["mean_target_nll"]),
            -int(row["training_manifest"].get("trainable_parameter_count", 0)),
        ),
    )
    max_quality = float(maximum["assessment"]["quality_index"])
    near_max = [
        row for row in eligible
        if max_quality - float(row["assessment"]["quality_index"]) <= balanced_tolerance
    ]
    balanced = min(
        near_max,
        key=lambda row: (
            int(row["training_manifest"].get("trainable_parameter_count", 0)),
            float(row["nll_summary"]["mean_target_nll"]),
            -float(row["assessment"]["quality_index"]),
        ),
    )
    minimum = min(
        eligible,
        key=lambda row: (
            int(row["training_manifest"].get("trainable_parameter_count", 0)),
            -float(row["assessment"]["quality_index"]),
            float(row["nll_summary"]["mean_target_nll"]),
        ),
    )
    return {
        "maximum_quality": maximum,
        "balanced_training": balanced,
        "minimum_training_memory": minimum,
    }


def parse_approved_profiles(gate_text: str) -> set[str]:
    import re

    match = re.search(r"(?im)^\s*-\s*Approved profiles:\s*`?([^`\r\n]+)`?\s*$", gate_text)
    if not match:
        return set()
    return {value.strip() for value in match.group(1).split(",") if value.strip()}


def relative_to_root(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except (ValueError, OSError):
        return str(path)
