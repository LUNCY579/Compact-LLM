from __future__ import annotations

import pytest

from rineya_harness.errors import ModelProtocolError
from rineya_harness.orchestration.contracts import QueryPlan
from rineya_harness.orchestration.structured import parse_structured_output


def test_parses_json_fence() -> None:
    plan = parse_structured_output('```json\n{"query":"x","gap_id":"g","reason":"r"}\n```', QueryPlan)
    assert plan.query == "x"


def test_rejects_prose_around_json() -> None:
    with pytest.raises(ModelProtocolError):
        parse_structured_output('Here: {"query":"x"}', QueryPlan)


def test_rejects_wrong_schema() -> None:
    with pytest.raises(ModelProtocolError):
        parse_structured_output('{"query":""}', QueryPlan)


def test_reconstruction_contract_rejects_model_created_pin() -> None:
    from pydantic import ValidationError

    from rineya_harness.domain.enums import MemoryKind
    from rineya_harness.orchestration.contracts import MemoryProposal

    with pytest.raises(ValidationError, match="pinned/global-invariant"):
        MemoryProposal(
            kind=MemoryKind.CURRENT_STATE,
            summary="state",
            pinned=True,
        )
