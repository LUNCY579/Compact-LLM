"""Regression tests for the TGSA v5.1 programme-level evidence boundary."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tgsa.v5_1_programme import (
    PROGRAMME_SCHEMA_VERSION,
    V51ContractError,
    validate_contract,
)

LAB = Path(__file__).resolve().parents[1]
CONTRACT = LAB / "configs" / "v5_1_programme_contract.json"


def _contract() -> dict[str, object]:
    value = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_the_canonical_v5_1_contract_is_complete_and_valid() -> None:
    contract = _contract()
    report = validate_contract(contract)
    assert contract["schema_version"] == PROGRAMME_SCHEMA_VERSION
    assert report["passed"] is True
    assert report["errors"] == []
    assert all(report["checks"].values())


def test_cross_stage_pooling_and_semantic_laundering_are_rejected() -> None:
    pooled = copy.deepcopy(_contract())
    pooled["global_evidence_rules"]["cross_stage_pooling_allowed"] = True
    with pytest.raises(V51ContractError, match="may not be pooled"):
        validate_contract(pooled)

    laundered = copy.deepcopy(_contract())
    laundered["stages"]["phase2a_trajectory_mtp"]["mtp_semantics"] = (
        "parallel_independent_offsets_v1"
    )
    with pytest.raises(V51ContractError, match="Phase 2A"):
        validate_contract(laundered)


def test_phase2a_requires_isolated_gradient_order_and_no_cascade_side_channel() -> None:
    reordered = copy.deepcopy(_contract())
    reordered["stages"]["phase2a_trajectory_mtp"]["gradient_mode_order"] = [
        "joint",
        "stop_gradient",
        "frozen_probe",
    ]
    with pytest.raises(V51ContractError, match="existing information"):
        validate_contract(reordered)

    side_channel = copy.deepcopy(_contract())
    side_channel["stages"]["phase2a_trajectory_mtp"]["trajectory_readout"][
        "cascade_trajectory_side_channel_allowed"
    ] = True
    with pytest.raises(V51ContractError, match="side channel"):
        validate_contract(side_channel)


def test_phase2b_cannot_drop_a_cell_weaken_exactness_or_gate_on_wall_clock() -> None:
    missing_cell = copy.deepcopy(_contract())
    missing_cell["stages"]["phase2b_system_factorial"]["primary_cells"].pop()
    with pytest.raises(V51ContractError, match="primary cells"):
        validate_contract(missing_cell)

    weakened = copy.deepcopy(_contract())
    weakened["stages"]["phase2b_system_factorial"]["exact_decode_gates"].remove(
        "first_mismatch_rollback"
    )
    with pytest.raises(V51ContractError, match="exact decoding"):
        validate_contract(weakened)

    timed = copy.deepcopy(_contract())
    timed["stages"]["phase2b_system_factorial"]["wall_clock_gating"] = True
    with pytest.raises(V51ContractError, match="wall-clock"):
        validate_contract(timed)
