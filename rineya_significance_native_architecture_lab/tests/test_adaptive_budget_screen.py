from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rineya_significance_lab.evaluation.adaptive_budget_screen import (
    screen_adaptive_budget,
    selected_resident_positions,
    state_from_count_boundary,
)

LAB = Path(__file__).resolve().parents[1]


def _first_true_trace():
    run = LAB / "results" / "runs" / "halo_count_causal_controls_v1"
    row = next(
        json.loads(line)
        for line in (run / "predictions.unscored.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line)["condition"] == "true_realized_count"
    )
    with np.load(run / row["boundary_diagnostics"], allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    return row, arrays


def test_uniform_boundary_replay_matches_runtime_selection_and_budget() -> None:
    row, arrays = _first_true_trace()
    summary = row["boundary_summaries"][2]
    state = state_from_count_boundary(arrays, boundary_index=2, layer_count=28)
    screen = screen_adaptive_budget(
        state,
        strategy="uniform",
        average_budget_per_unit=128,
        minimum_per_unit=36,
        maximum_per_unit=220,
        sink_tokens=4,
        recent_tokens=32,
        exponent=1.0,
        seed=29,
    )
    recorded = np.stack(
        [arrays[f"b002_l{layer:03d}_selected"][0].astype(bool) for layer in range(28)],
        axis=0,
    )
    assert np.array_equal(screen.selection.mask, recorded)
    assert int(screen.budget.allocations.sum()) == 28 * 8 * 128
    assert summary["stored_after"] == 128


def test_entropy_allocation_is_heterogeneous_but_globally_matched() -> None:
    _, arrays = _first_true_trace()
    state = state_from_count_boundary(arrays, boundary_index=2, layer_count=28)
    screen = screen_adaptive_budget(
        state,
        strategy="entropy_proportional",
        average_budget_per_unit=128,
        minimum_per_unit=36,
        maximum_per_unit=220,
        sink_tokens=4,
        recent_tokens=32,
        exponent=1.0,
        seed=29,
    )
    assert int(screen.budget.allocations.sum()) == 28 * 8 * 128
    assert int(screen.budget.allocations.min()) >= 36
    assert int(screen.budget.allocations.max()) <= 220
    assert np.unique(screen.budget.allocations).size > 1
    resident = selected_resident_positions(screen)
    assert len(resident) == 28 and all(len(layer) == 8 for layer in resident)


def test_adaptive_artifact_fails_significance_specificity_gate() -> None:
    summary = json.loads(
        (
            LAB
            / "results"
            / "runs"
            / "halo_adaptive_budget_screen_v1"
            / "summary.json"
        ).read_text(encoding="utf-8")
    )
    assert summary["uniform_runtime_parity_checks"] == 33
    assert summary["specificity_gate"]["passed"] is False
    assert summary["decision"] == "NO_SIGNIFICANCE_SPECIFIC_ADVANTAGE"
