from __future__ import annotations

import json
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
RUN = LAB_ROOT / "results" / "runs" / "halo_significance_memory_smoke_v1"


def _json(name: str) -> dict:
    return json.loads((RUN / name).read_text(encoding="utf-8"))


def test_canonical_memory_smoke_is_narrow_and_passed() -> None:
    summary = _json("summary.json")
    assert summary["status"] == "PASS"
    assert summary["evidence_class"] == "RUNTIME_COMPATIBILITY_ONLY"
    assert summary["m0_identity_pass"] is True
    assert summary["m4_finite_runtime_pass"] is True
    assert summary["answer_generation_performed"] is False
    assert summary["quality_measured"] is False
    assert len(summary["claims_not_supported"]) >= 3


def test_m0_is_exact_and_m4_reaches_future_attention() -> None:
    diagnostics = _json("runtime_diagnostics.json")
    identity = diagnostics["m0_identity"]
    assert identity["exact_logits"] is True
    assert identity["max_abs_logit_delta"] == 0.0
    assert identity["exact_cache_state"] is True
    assert identity["cache_mismatches"] == []
    assert identity["m0_memory_bytes"] == 0
    assert identity["m0_memory_write_boundaries"] == 0

    runtime = diagnostics["m4_runtime"]
    assert runtime["finite_logits"] is True
    assert runtime["true_length"] == 128
    assert runtime["ordinary_stored_length"] == 96
    assert runtime["memory_layer_writes"] == 28
    assert runtime["attention_accounting"]["calls_with_valid_slots"] == 28
    assert runtime["attention_accounting"]["max_valid_slots_per_batch_head"] == 4
    assert diagnostics["oracle_access"]["answer_key_loaded"] is False
    assert diagnostics["oracle_access"]["evidence_annotations_entered_memory_writes"] is False


def test_slot_artifact_has_exact_capacity_positions_and_byte_ledger() -> None:
    slots = _json("slot_diagnostics.json")
    ledger = slots["byte_ledger"]
    assert ledger["slot_capacity"] == 28 * 8 * 4
    assert ledger["valid_slots"] == ledger["slot_capacity"]
    component_total = sum(
        value for name, value in ledger.items() if name.endswith("_bytes")
    )
    assert component_total == 491_904
    assert len(slots["layers"]) == 28
    for layer in slots["layers"]:
        assert layer["physical_slots_per_batch_head"] == 4
        assert layer["valid_slots"] == 8 * 4
        assert layer["source_count_sum"] == 8 * 32
        assert layer["source_position_max"] < min(layer["last_updated_at"])


def test_case_rows_never_encode_predictions_or_quality_scores() -> None:
    rows = [
        json.loads(line)
        for line in (RUN / "cases.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert [row["condition"] for row in rows] == [
        "SIGNIFICANCE_MEMORY_M0_IDENTITY",
        "SIGNIFICANCE_MEMORY_M4_NORMALIZED_RUNTIME",
    ]
    assert all(row["status"] == "OK" for row in rows)
    assert all(row["prediction"] is None and row["correct"] is None for row in rows)
