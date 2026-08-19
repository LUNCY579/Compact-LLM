from __future__ import annotations

import pytest

from rineya_harness.domain.enums import DeleteScope, MemoryKind, MemoryStatus, ReasonCode
from rineya_harness.domain.models import DeleteMemoryRequest, MemoryDraft, RunRecord
from rineya_harness.errors import StateInvariantError


def _seed(container):
    run = container.store.create_run(RunRecord(goal="test"))
    container.memory.add(
        run_id=run.id,
        draft=MemoryDraft(id="goal", kind=MemoryKind.GLOBAL_INVARIANT, summary="goal", pinned=True),
        expected_revision=0,
    )
    container.memory.add(
        run_id=run.id,
        draft=MemoryDraft(id="temporary", kind=MemoryKind.CURRENT_STATE, summary="old state"),
        expected_revision=1,
    )
    return run.id


def test_delete_memory_evicts_reversibly(container) -> None:
    run_id = _seed(container)
    result = container.memory.delete_memory(
        DeleteMemoryRequest(
            run_id=run_id,
            memory_ids=["temporary"],
            scope=DeleteScope.ACTIVE_CONTEXT,
            reason_code=ReasonCode.COMPLETED_NO_FUTURE_DEPENDENCY,
            expected_state_revision=2,
        )
    )
    assert result.prompt_rebuild_required is True
    assert container.store.get_memory(run_id, "temporary").status is MemoryStatus.EVICTED
    restored = container.memory.restore(run_id=run_id, memory_ids=["temporary"], expected_revision=3)
    assert restored.new_revision == 4
    assert container.store.get_memory(run_id, "temporary").status is MemoryStatus.ACTIVE


def test_pinned_memory_cannot_be_deleted(container) -> None:
    run_id = _seed(container)
    with pytest.raises(StateInvariantError):
        container.memory.delete_memory(
            DeleteMemoryRequest(
                run_id=run_id,
                memory_ids=["goal"],
                scope=DeleteScope.ACTIVE_CONTEXT,
                reason_code=ReasonCode.OTHER,
                expected_state_revision=2,
            )
        )


def test_purge_disabled_by_default(container) -> None:
    run_id = _seed(container)
    container.memory.delete_memory(
        DeleteMemoryRequest(
            run_id=run_id,
            memory_ids=["temporary"],
            scope=DeleteScope.ARCHIVE,
            reason_code=ReasonCode.USER_REQUEST,
            expected_state_revision=2,
        )
    )
    with pytest.raises(StateInvariantError):
        container.memory.delete_memory(
            DeleteMemoryRequest(
                run_id=run_id,
                memory_ids=["temporary"],
                scope=DeleteScope.PERSISTENT_PURGE,
                reason_code=ReasonCode.USER_REQUEST,
                expected_state_revision=3,
            )
        )


def test_supersede_adds_replacement_atomically(container) -> None:
    run = container.store.create_run(RunRecord(goal="test"))
    container.memory.add(
        run_id=run.id,
        draft=MemoryDraft(id="old", kind=MemoryKind.CURRENT_STATE, summary="old"),
        expected_revision=0,
    )
    container.memory.supersede(
        run_id=run.id,
        old_memory_id="old",
        replacement=MemoryDraft(id="new", kind=MemoryKind.CURRENT_STATE, summary="new"),
        expected_revision=1,
    )
    assert container.store.get_memory(run.id, "old").status is MemoryStatus.SUPERSEDED
    assert container.store.get_memory(run.id, "new").status is MemoryStatus.ACTIVE
