from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..evaluation.objective import align_rows, case_utility
from .bootstrap import bootstrap_mean


@dataclass(frozen=True)
class LayerInfluenceSummary:
    layer: int
    forward_mean: float | None
    reverse_mean: float | None
    combined_mean: float
    ci_low: float
    ci_high: float
    probability_positive: float
    sign_consistency: float
    reliability: float
    case_count: int

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def per_case_utility_delta(
    candidate_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    objective_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    candidate, baseline = align_rows(candidate_rows, baseline_rows)
    output: list[dict[str, Any]] = []
    for cand, base in zip(candidate, baseline):
        output.append(
            {
                "id": str(cand["id"]),
                "category": str(cand.get("category", "uncategorized")),
                "candidate_utility": case_utility(cand, objective_config),
                "baseline_utility": case_utility(base, objective_config),
                "delta": case_utility(cand, objective_config) - case_utility(base, objective_config),
                "candidate_quality": float(cand.get("quality_score", 0.0)),
                "baseline_quality": float(base.get("quality_score", 0.0)),
                "candidate_reply": str(cand.get("reply", "")),
                "baseline_reply": str(base.get("reply", "")),
            }
        )
    return output


def summarize_bidirectional_layer(
    *,
    layer: int,
    anchor_to_greedy_rows: Sequence[Mapping[str, Any]],
    anchor_rows: Sequence[Mapping[str, Any]],
    greedy_to_anchor_rows: Sequence[Mapping[str, Any]] | None,
    greedy_rows: Sequence[Mapping[str, Any]] | None,
    objective_config: Mapping[str, Any],
    bootstrap_samples: int,
    confidence_level: float,
    seed: int,
) -> tuple[LayerInfluenceSummary, list[dict[str, Any]]]:
    forward = per_case_utility_delta(anchor_to_greedy_rows, anchor_rows, objective_config)
    reverse: list[dict[str, Any]] | None = None
    if greedy_to_anchor_rows is not None and greedy_rows is not None:
        # Positive reverse evidence means the Greedy version of this layer is useful:
        # utility(greedy) - utility(greedy with the Anchor layer).
        raw = per_case_utility_delta(greedy_to_anchor_rows, greedy_rows, objective_config)
        reverse = [{**row, "delta": -float(row["delta"])} for row in raw]

    combined_values: list[float] = []
    matrix_rows: list[dict[str, Any]] = []
    reverse_by_id = {str(row["id"]): row for row in reverse or []}
    for row in forward:
        forward_delta = float(row["delta"])
        reverse_delta = (
            float(reverse_by_id[str(row["id"])]["delta"])
            if str(row["id"]) in reverse_by_id
            else None
        )
        combined = (
            (forward_delta + reverse_delta) / 2.0
            if reverse_delta is not None
            else forward_delta
        )
        combined_values.append(combined)
        matrix_rows.append(
            {
                "id": row["id"],
                "category": row["category"],
                "layer": layer,
                "forward_greedy_advantage": forward_delta,
                "reverse_greedy_advantage": reverse_delta,
                "combined_greedy_advantage": combined,
            }
        )
    boot = bootstrap_mean(
        combined_values,
        samples=bootstrap_samples,
        confidence_level=confidence_level,
        seed=seed + layer * 997,
    )
    mean = float(np.mean(combined_values))
    signs = np.sign(np.asarray(combined_values, dtype=np.float64))
    if mean > 0:
        sign_consistency = float(np.mean(signs >= 0))
    elif mean < 0:
        sign_consistency = float(np.mean(signs <= 0))
    else:
        sign_consistency = 0.5
    half_width = max(1e-12, (float(boot["ci_high"]) - float(boot["ci_low"])) / 2.0)
    signal_to_uncertainty = abs(mean) / (abs(mean) + half_width)
    probability_direction = (
        float(boot["probability_positive"])
        if mean >= 0
        else 1.0 - float(boot["probability_positive"])
    )
    reliability = max(0.0, min(1.0, signal_to_uncertainty * sign_consistency * probability_direction))
    return (
        LayerInfluenceSummary(
            layer=layer,
            forward_mean=float(np.mean([row["delta"] for row in forward])) if forward else None,
            reverse_mean=float(np.mean([row["delta"] for row in reverse])) if reverse else None,
            combined_mean=mean,
            ci_low=float(boot["ci_low"]),
            ci_high=float(boot["ci_high"]),
            probability_positive=float(boot["probability_positive"]),
            sign_consistency=sign_consistency,
            reliability=reliability,
            case_count=len(combined_values),
        ),
        matrix_rows,
    )


def smooth_layer_targets(
    targets: Mapping[int, float],
    reliabilities: Mapping[int, float],
    *,
    global_alpha: float,
    smoothing_penalty: float,
    global_shrinkage_penalty: float,
) -> dict[int, float]:
    """Solve a bounded quadratic shrinkage problem for stable per-layer coefficients.

    Minimize:
      sum_l r_l (a_l - target_l)^2
      + global_penalty * sum_l (a_l - global_alpha)^2
      + smooth_penalty * sum_l (a_l - a_{l-1})^2

    This does not claim adjacent layers must be similar. It merely prevents noisy, weakly
    supported saw-tooth coefficients from being treated as discoveries.
    """

    layers = sorted(targets)
    if not layers:
        return {}
    n = len(layers)
    matrix = np.zeros((n, n), dtype=np.float64)
    vector = np.zeros(n, dtype=np.float64)
    smooth = max(0.0, float(smoothing_penalty))
    shrink = max(0.0, float(global_shrinkage_penalty))
    for position, layer in enumerate(layers):
        reliability = max(0.0, float(reliabilities.get(layer, 0.0)))
        # A tiny ridge keeps the system invertible when all penalties are zero.
        data_weight = reliability + 1e-8
        matrix[position, position] += data_weight + shrink
        vector[position] += data_weight * float(targets[layer]) + shrink * float(global_alpha)
        if position > 0:
            matrix[position, position] += smooth
            matrix[position - 1, position - 1] += smooth
            matrix[position, position - 1] -= smooth
            matrix[position - 1, position] -= smooth
    solution = np.linalg.solve(matrix, vector)
    return {
        layer: max(0.0, min(1.0, float(solution[position])))
        for position, layer in enumerate(layers)
    }


def normalized_layer_importance(summaries: Sequence[LayerInfluenceSummary]) -> dict[int, float]:
    raw = {
        summary.layer: abs(summary.combined_mean) * max(0.0, summary.reliability)
        for summary in summaries
    }
    total = sum(raw.values())
    if total <= 0:
        return {layer: 1.0 / max(1, len(raw)) for layer in raw}
    return {layer: value / total for layer, value in raw.items()}
