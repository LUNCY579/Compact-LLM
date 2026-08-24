"""Metrics over retained semantic entry IDs."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..core.ranking import Selection
from ..core.state import SignificanceState


def evidence_recall(
    state: SignificanceState,
    selection: Selection,
    evidence: NDArray[Any],
) -> NDArray[np.float64]:
    annotation = np.asarray(evidence, dtype=np.bool_)
    if annotation.shape != state.shape:
        raise ValueError("evidence annotation must match state shape")
    recalled = np.zeros(state.shape[:2], dtype=np.float64)
    for layer in range(state.layers):
        for head in range(state.kv_heads):
            required = set(state.entry_ids[layer, head, annotation[layer, head] & state.active[layer, head]].tolist())
            selected = set(state.entry_ids[layer, head, selection.mask[layer, head]].tolist())
            recalled[layer, head] = 1.0 if not required else len(required & selected) / len(required)
    return recalled


def selection_metrics(
    state: SignificanceState,
    selection: Selection,
    *,
    evidence: NDArray[Any] | None = None,
    distractors: NDArray[Any] | None = None,
    stale: NDArray[Any] | None = None,
) -> dict[str, Any]:
    selected_count = selection.mask.sum(axis=-1)
    metrics: dict[str, Any] = {
        "policy_id": selection.policy_id,
        "selected_mean": float(np.mean(selected_count)),
        "selected_min": int(selected_count.min()),
        "selected_max": int(selected_count.max()),
    }
    if evidence is not None:
        recall = evidence_recall(state, selection, evidence)
        complete_fraction = float(np.mean(recall == 1.0))
        metrics.update(
            {
                "evidence_recall_mean": float(np.mean(recall)),
                "fraction_layer_heads_with_complete_evidence": complete_fraction,
                # Backward-compatible alias for superseded Tier-A artifacts.
                # This is not a case-level all-or-none event.
                "all_required_evidence_survival": complete_fraction,
            }
        )
    for label, annotation in (("distractor", distractors), ("stale", stale)):
        if annotation is None:
            continue
        mask = np.asarray(annotation, dtype=np.bool_)
        if mask.shape != state.shape:
            raise ValueError(f"{label} annotation must match state shape")
        denominator = np.maximum(mask.sum(axis=-1), 1)
        rate = (selection.mask & mask).sum(axis=-1) / denominator
        metrics[f"{label}_retention_mean"] = float(np.mean(rate))
    return metrics
