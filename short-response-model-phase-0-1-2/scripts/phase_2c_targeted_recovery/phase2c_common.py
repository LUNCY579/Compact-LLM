from __future__ import annotations

"""Pure helpers shared by the Phase 2C ranking, orchestration, gate, and tests."""

import math
from pathlib import Path
from typing import Any, Iterable

from scripts.common.quality_evaluation import require_quality_v3, summary_quality_score
from scripts.common.quality_gate import quality_index as shared_quality_index
from scripts.common.opportunity_evaluation import evaluate_opportunity, resource_reduction


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
    """Tail-aware development ranking; hard gates remain separate."""

    return shared_quality_index(summary)


def relative_loss(reference: float, candidate: float) -> float:
    return max(0.0, reference - candidate) / max(abs(reference), 1e-12)


def category_quality_scores(summary: dict[str, Any]) -> dict[str, float]:
    require_quality_v3(summary)
    return {
        str(key): float(value)
        for key, value in (summary.get("category_quality_scores") or summary.get("category_scores") or {}).items()
    }


def max_category_loss(reference: dict[str, Any], candidate: dict[str, Any]) -> tuple[float, dict[str, float]]:
    losses: dict[str, float] = {}
    reference_scores = category_quality_scores(reference)
    candidate_scores = category_quality_scores(candidate)
    for category, ref_score in reference_scores.items():
        cand_score = candidate_scores.get(category)
        losses[category] = 1.0 if cand_score is None else max(0.0, ref_score - cand_score)
    return (max(losses.values()) if losses else 0.0), losses




def max_category_metric_loss(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    field: str,
) -> tuple[float, dict[str, float]]:
    """Return per-category absolute loss for an independent metric slice."""

    ref_scores = {str(key): float(value) for key, value in (reference.get(field) or {}).items()}
    cand_scores = {str(key): float(value) for key, value in (candidate.get(field) or {}).items()}
    losses: dict[str, float] = {}
    for category, ref_score in ref_scores.items():
        cand_score = cand_scores.get(category)
        losses[category] = 1.0 if cand_score is None else max(0.0, ref_score - cand_score)
    return (max(losses.values()) if losses else 0.0), losses


def assess_training_candidate(
    *,
    candidate: dict[str, Any],
    all_remaining: dict[str, Any],
    reduced_frozen: dict[str, Any],
    quality_profile: str = "balanced",
    **deprecated_thresholds: Any,
) -> dict[str, Any]:
    """Compare trainable-state saving with continuous quality opportunity loss.

    Historical threshold keyword arguments remain accepted for command
    compatibility and are intentionally ignored.
    """

    cand_summary = candidate["generation_summary"]
    all_summary = all_remaining["generation_summary"]
    frozen_summary = reduced_frozen["generation_summary"]
    require_quality_v3(cand_summary, f"Phase 2C candidate {candidate.get('name')}")
    require_quality_v3(all_summary, "Phase 2C all-remaining reference")
    require_quality_v3(frozen_summary, "Phase 2C reduced-frozen reference")

    all_trainable = int(all_remaining["training_manifest"].get("trainable_parameter_count", 0))
    cand_trainable = int(candidate["training_manifest"].get("trainable_parameter_count", 0))
    trainable_reduction = resource_reduction(all_trainable, cand_trainable)
    all_nll = float(all_remaining["nll_summary"]["mean_target_nll"])
    cand_nll = float(candidate["nll_summary"]["mean_target_nll"])
    opportunity = evaluate_opportunity(
        all_summary,
        cand_summary,
        resource_gain=trainable_reduction,
        resource_name="trainable_parameter_reduction",
        profile=quality_profile,
        reference_target_nll=all_nll,
        candidate_target_nll=cand_nll,
    )
    if candidate.get("name") == "all_remaining":
        # This row is the same-seed quality control, not an efficiency
        # candidate. Mark it usable when technically valid so it can anchor
        # repeated comparisons without pretending a zero-saving model won an
        # opportunity trade-off.
        opportunity["eligible"] = bool(opportunity["technical_valid"])
        opportunity["decision"] = "REFERENCE_CONTROL"
        opportunity["decision_explanation"] = "same-seed all-remaining control is technically valid"
        opportunity["opportunity_margin"] = 0.0

    all_runtime = float(all_remaining["training_manifest"].get("train_metrics", {}).get("train_runtime", 0.0))
    cand_runtime = float(candidate["training_manifest"].get("train_metrics", {}).get("train_runtime", 0.0))
    runtime_reduction = resource_reduction(all_runtime, cand_runtime)
    all_vram = float(all_remaining["training_manifest"].get("peak_allocated_vram_gb", 0.0))
    cand_vram = float(candidate["training_manifest"].get("peak_allocated_vram_gb", 0.0))
    vram_reduction = resource_reduction(all_vram, cand_vram)
    all_quality = quality_index(all_summary)
    frozen_quality = quality_index(frozen_summary)
    candidate_quality = quality_index(cand_summary)
    denominator = all_quality - frozen_quality
    recovered_fraction = None if denominator <= 1e-12 else (candidate_quality - frozen_quality) / denominator
    deviation = candidate.get("output_deviation") or {}

    return {
        **opportunity,
        "rejection_reasons": list(opportunity["technical_invalidity_reasons"]) + ([] if opportunity["eligible"] else [opportunity["decision_explanation"]]),
        "quality_index": candidate_quality,
        "mean_quality_score": summary_quality_score(cand_summary),
        "quality_pass_rate": float(cand_summary.get("quality_pass_rate", 0.0)),
        "intelligibility_pass_rate": float(cand_summary.get("intelligibility_pass_rate", 0.0)),
        "hard_failure_case_count": int(cand_summary.get("hard_failure_case_count", 0)),
        "worst_decile_mean_quality_score": float(cand_summary.get("worst_decile_mean_quality_score", 0.0)),
        "quality_gain_vs_reduced_frozen": candidate_quality - frozen_quality,
        "fraction_of_all_remaining_gain_recovered": recovered_fraction,
        "trainable_parameter_reduction_vs_all_remaining": float(trainable_reduction or 0.0),
        "training_runtime_reduction_vs_all_remaining": runtime_reduction,
        "peak_training_vram_reduction_vs_all_remaining": vram_reduction,
        "new_critical_failures_vs_reduced_frozen": int(deviation.get("new_critical_failure_count", 0)),
        "new_major_issues_vs_reduced_frozen": int(deviation.get("new_major_issue_count", 0)),
        "deprecated_threshold_arguments_ignored": sorted(deprecated_thresholds),
    }


def select_profiles(rows: list[dict[str, Any]], *, balanced_tolerance: float | None = None) -> dict[str, dict[str, Any]]:
    """Select quality, opportunity-margin, and minimum-state profiles.

    ``balanced_tolerance`` is a deprecated compatibility argument and does not
    control selection.
    """

    valid = [row for row in rows if (row.get("assessment") or {}).get("technical_valid", True)]
    if not valid:
        raise ValueError("No technically valid Phase 2C candidates are available.")
    worthwhile = [row for row in valid if (row.get("assessment") or {}).get("eligible")]
    maximum = max(
        valid,
        key=lambda row: (
            float((row.get("assessment") or {}).get("quality_index", 0.0)),
            -float(row["nll_summary"].get("mean_target_nll", float("inf"))),
        ),
    )
    if worthwhile:
        balanced = max(
            worthwhile,
            key=lambda row: (
                float(row["assessment"].get("opportunity_margin", float("-inf"))),
                float(row["assessment"].get("resource_gain_fraction", 0.0)),
            ),
        )
        minimum = max(
            worthwhile,
            key=lambda row: (
                float(row["assessment"].get("resource_gain_fraction", 0.0)),
                float(row["assessment"].get("opportunity_margin", float("-inf"))),
            ),
        )
    else:
        balanced = maximum
        minimum = maximum
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
