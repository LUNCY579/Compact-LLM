from __future__ import annotations

import json

import pytest

from rineya_harness.domain.enums import DeleteScope, MemoryKind, ReasonCode
from rineya_harness.domain.models import MemoryDraft, RunRecord, ToolCall
from rineya_harness.errors import ToolRejectedError
from rineya_harness.tools.base import ToolRegistry
from rineya_harness.tools.memory import MemoryToolSet
from rineya_harness.tools.research import ResearchToolSet


@pytest.mark.asyncio
async def test_memory_tools_round_trip(container) -> None:
    run = container.store.create_run(RunRecord(goal="tools"))
    container.memory.add(
        run_id=run.id,
        draft=MemoryDraft(id="state", kind=MemoryKind.CURRENT_STATE, summary="old"),
        expected_revision=0,
    )
    registry = ToolRegistry()
    for definition in MemoryToolSet(container.store, container.memory).definitions():
        registry.register(definition)
    state = await registry.execute(
        ToolCall(name="get_state", arguments={"run_id": run.id})
    )
    assert state.ok is True
    assert state.output["memories"][0]["id"] == "state"
    updated = await registry.execute(
        ToolCall(
            name="update_memory",
            arguments={
                "run_id": run.id,
                "memory_id": "state",
                "changes": {"summary": "new"},
                "expected_revision": 1,
            },
        )
    )
    assert updated.ok is True
    deleted = await registry.execute(
        ToolCall(
            name="delete_memory",
            arguments={
                "run_id": run.id,
                "memory_ids": ["state"],
                "scope": DeleteScope.ACTIVE_CONTEXT.value,
                "reason_code": ReasonCode.OTHER.value,
                "expected_state_revision": 2,
            },
        )
    )
    assert deleted.ok is True
    assert deleted.output["removed_from_active_context"] == ["state"]


@pytest.mark.asyncio
async def test_research_tools(container) -> None:
    registry = ToolRegistry()
    for definition in ResearchToolSet(container.search, container.store).definitions():
        registry.register(definition)
    result = await registry.execute(ToolCall(name="search_web", arguments={"query": "none"}))
    assert result.ok is True
    assert result.output == {"results": []}


@pytest.mark.asyncio
async def test_unknown_tool_is_rejected() -> None:
    registry = ToolRegistry()
    with pytest.raises(ToolRejectedError):
        await registry.execute(ToolCall(name="missing"))


def test_duplicate_tool_registration_rejected(container) -> None:
    registry = ToolRegistry()
    definition = MemoryToolSet(container.store, container.memory).definitions()[0]
    registry.register(definition)
    with pytest.raises(ValueError):
        registry.register(definition)
    assert registry.definitions()[0]["name"] == "get_state"


@pytest.mark.asyncio
async def test_default_container_exposes_exact_small_model_tool_surface(container) -> None:
    definitions = container.tools.definitions()
    assert [item["name"] for item in definitions] == [
        "delete_memory",
        "finish",
        "get_state",
        "inspect_evidence",
        "search_web",
        "update_memory",
    ]
    result = await container.tools.execute(
        ToolCall(
            name="finish",
            arguments={"action": "answer", "answer": "Done.", "citations": []},
        )
    )
    assert result.ok is True
    assert result.output["action"] == "answer"


@pytest.mark.asyncio
async def test_finish_tool_rejects_empty_answer(container) -> None:
    result = await container.tools.execute(
        ToolCall(name="finish", arguments={"action": "answer", "answer": ""})
    )
    assert result.ok is False
    assert result.error_code == "ValidationError"


def test_model_delete_tool_does_not_expose_persistent_purge(container) -> None:
    definitions = {
        item.name: item for item in MemoryToolSet(container.store, container.memory).definitions()
    }
    schema_text = json.dumps(definitions["delete_memory"].schema())
    assert "persistent_purge" not in schema_text


def test_model_update_tool_rejects_lifecycle_fields() -> None:
    from pydantic import ValidationError

    from rineya_harness.tools.memory import UpdateMemoryArgs

    with pytest.raises(ValidationError, match="forbidden fields"):
        UpdateMemoryArgs(
            run_id="run",
            memory_id="memory",
            changes={"pinned": True},
            expected_revision=0,
        )
