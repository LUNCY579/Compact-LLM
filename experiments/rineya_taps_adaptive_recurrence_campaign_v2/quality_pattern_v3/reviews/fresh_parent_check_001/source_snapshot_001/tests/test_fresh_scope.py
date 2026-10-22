"""Pure prospective authority/budget guards; no worker/model/source assets."""
import pytest
from experiments.rineya_taps_adaptive_recurrence_campaign_v2.quality_pattern_v3.fresh_scope import (
    FIXED_CONTRACT, SOURCE, INTERPRETER, STUDY_ID, validate_contract, budget_and_unused,
)


def packet():
    return {**FIXED_CONTRACT, "status": "READY_FOR_FRESH_K01_FP32_DISPATCH",
            "source": str(SOURCE), "interpreter": INTERPRETER,
            "primary_policy": "EVIDENCE_CONSISTENCY_GUARD", "DEV_K1_rate": 160 / 192}


def test_fixed_contract_and_reservation():
    assert validate_contract(packet())
    assert budget_and_unused([{"event": "native_completed", "native_worker_seconds": 3690.2476851}]) == 3690.2476851


@pytest.mark.parametrize("field,value", [("parent_host_available_floor_bytes", 6 * 1024**3),
    ("internal_reserve_seconds", 120), ("use_cache", True), ("extra_budgets", [0, 1, 2]),
    ("dtype", "bfloat16"), ("training", 0), ("model_API_cap", 34597),
    ("primary_policy", "CONFIDENT_RECURRENCE"), ("DEV_K1_rate", .5)])
def test_guard_or_scientific_changes_rejected(field, value):
    p = packet(); p[field] = value
    with pytest.raises(PermissionError):
        validate_contract(p)


def test_consumed_study_and_insufficient_full_budget():
    for event in ({"event": "native_started", "study_id": STUDY_ID},
                  {"event": "native_completed", "study_id": STUDY_ID, "worker_pid": 17}):
        with pytest.raises(PermissionError):
            budget_and_unused([event])
    with pytest.raises(RuntimeError):
        budget_and_unused([{"event": "native_completed", "native_worker_seconds": 4201.}])


@pytest.mark.parametrize("cost", [-1., float("nan"), None, True])
def test_invalid_costs_are_not_zero(cost):
    with pytest.raises(ValueError):
        budget_and_unused([{"event": "native_completed", "native_worker_seconds": cost}])
