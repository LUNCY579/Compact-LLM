from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rineya_significance_lab.benchmarks import FrozenEvidenceAnnotation
from rineya_significance_lab.evaluation.boundary_evidence import (
    score_observed_boundary_evidence,
)
from rineya_significance_lab.evaluation.count_causal import (
    COUNT_CONDITIONS,
    analyze_count_controls,
    holm_adjust,
)
from rineya_significance_lab.evaluation.failure_taxonomy import (
    compressed_retention_failure_labels,
)

LAB = Path(__file__).resolve().parents[1]


def test_holm_adjust_is_monotone_in_sorted_p_values() -> None:
    adjusted = holm_adjust({"a": 0.01, "b": 0.03, "c": 0.20})
    assert adjusted == {"a": 0.03, "b": 0.06, "c": 0.20}


def test_failure_labels_do_not_turn_failed_strict_conjunction_into_total_eviction() -> None:
    partial = {
        "strict_case_all_required_fact_tokens_available_in_at_least_one_kv_head_of_every_evaluated_layer": False,
        "mean_per_layer_required_token_availability_fraction": 0.75,
    }
    assert compressed_retention_failure_labels(correct=False, evidence=partial) == [
        "EVIDENCE_PARTIALLY_RETAINED"
    ]
    assert compressed_retention_failure_labels(
        correct=False,
        evidence={**partial, "mean_per_layer_required_token_availability_fraction": 0.0},
    ) == ["EVIDENCE_EVICTED"]


def test_boundary_evidence_never_counts_future_facts_as_evicted() -> None:
    first = json.loads(
        (LAB / "results" / "benchmarks" / "halo_calibration_v3" / "evaluation.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    annotation = FrozenEvidenceAnnotation.from_dict(first["evidence_annotation"])
    before = score_observed_boundary_evidence(annotation, [[[]]], true_length=0)
    assert before.status == "NO_REQUIRED_FACT_FULLY_OBSERVED"
    assert before.evidence is None

    first_fact = annotation.facts[0]
    boundary = max(first_fact.token_positions) + 1
    resident = [[list(first_fact.token_positions)], [list(first_fact.token_positions)]]
    observed = score_observed_boundary_evidence(
        annotation,
        resident,
        true_length=boundary,
    )
    assert first_fact.fact_id in observed.fully_observed_fact_ids
    assert observed.evidence is not None
    assert observed.evidence["layer_fact_exact_completion_rate"] == 1.0


def _write_trace_pair(root: Path, stem: str, value: int) -> tuple[str, str]:
    resident = root / f"{stem}_resident.npz"
    boundary = root / f"{stem}_boundary.npz"
    np.savez_compressed(resident, layer_000=np.asarray([[[1, 2]]], dtype=np.int64))
    common = np.asarray([[[value, value + 1]]], dtype=np.float32)
    np.savez_compressed(
        boundary,
        b000_l000_effective_counts=common,
        b000_l000_true_counts=common,
        b000_l000_scores=common,
        b000_l000_selected=np.ones_like(common, dtype=np.uint8),
        b000_l000_retained_positions=np.asarray([[[1, 2]]], dtype=np.int64),
    )
    return resident.name, boundary.name


def test_count_analysis_detects_exact_age_and_head_equivalence(tmp_path: Path) -> None:
    rows = []
    for case_index in range(2):
        reference_traces = _write_trace_pair(tmp_path, f"case{case_index}_reference", case_index + 1)
        for condition in COUNT_CONDITIONS:
            identical = condition in {
                "true_realized_count",
                "age_derived_count",
                "head_averaged_count",
            }
            traces = (
                reference_traces
                if identical
                else _write_trace_pair(tmp_path, f"case{case_index}_{condition}", case_index + 10)
            )
            value = 0.5 if identical else 0.4
            rows.append(
                {
                    "case_id": f"case-{case_index}",
                    "scenario_id": f"cluster-{case_index}",
                    "condition": condition,
                    "prediction": "ok" if identical else "different",
                    "correct": True if identical else False,
                    "evidence": {
                        "layer_fact_exact_completion_rate": value,
                        "minimum_per_layer_required_token_availability_fraction": value,
                    },
                    "trace_ids": list(traces),
                }
            )
    result = analyze_count_controls(
        rows,
        artifact_root=tmp_path,
        minimum_clusters=1,
        bootstrap_samples=100,
        randomization_samples=100,
    )
    diagnostics = result["mechanism_diagnostics"]
    assert diagnostics["age_derived_count_exactly_reconstructs_true_realized_count_trajectory"]
    assert diagnostics["head_averaged_count_exactly_reconstructs_true_realized_count_trajectory"]
    assert diagnostics["explicit_count_state_adds_information_beyond_deterministic_age_in_this_regime"] is False
