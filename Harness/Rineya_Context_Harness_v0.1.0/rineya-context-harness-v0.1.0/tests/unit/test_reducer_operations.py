from __future__ import annotations

import pytest

from rineya_harness.domain.enums import (
    ActorType,
    MemoryKind,
    MemoryOperationType,
    MemoryStatus,
    MemoryTemperature,
)
from rineya_harness.domain.models import MemoryDraft, MemoryMutation, RunRecord, StatePatch
from rineya_harness.errors import StateInvariantError


def _seed(container):
    run = container.store.create_run(RunRecord(goal="reducer"))
    container.memory.add(
        run_id=run.id,
        draft=MemoryDraft(
            id="item",
            kind=MemoryKind.CURRENT_STATE,
            summary="state",
            payload={"nested": {"a": 1}},
        ),
        expected_revision=0,
    )
    return run.id


def test_merge_deep_merges_payload(container) -> None:
    run_id = _seed(container)
    container.memory.update(
        run_id=run_id,
        memory_id="item",
        changes={"payload": {"nested": {"b": 2}}},
        expected_revision=1,
        merge=True,
    )
    assert container.store.get_memory(run_id, "item").payload == {"nested": {"a": 1, "b": 2}}


def test_pin_and_unpin(container) -> None:
    run_id = _seed(container)
    container.store.apply_patch(
        StatePatch(
            run_id=run_id,
            expected_revision=1,
            actor=ActorType.SYSTEM,
            mutations=[MemoryMutation(operation=MemoryOperationType.PIN, target_id="item")],
            idempotency_key="pin-item-key",
        )
    )
    pinned = container.store.get_memory(run_id, "item")
    assert pinned.pinned is True
    assert pinned.temperature is MemoryTemperature.PINNED
    container.store.apply_patch(
        StatePatch(
            run_id=run_id,
            expected_revision=2,
            actor=ActorType.SYSTEM,
            mutations=[MemoryMutation(operation=MemoryOperationType.UNPIN, target_id="item")],
            idempotency_key="unpin-item-key",
        )
    )
    assert container.store.get_memory(run_id, "item").pinned is False


def test_unknown_update_field_rejected(container) -> None:
    run_id = _seed(container)
    with pytest.raises(StateInvariantError):
        container.memory.update(
            run_id=run_id,
            memory_id="item",
            changes={"not_a_field": 1},
            expected_revision=1,
        )


def test_immutable_field_update_rejected(container) -> None:
    run_id = _seed(container)
    with pytest.raises(StateInvariantError):
        container.memory.update(
            run_id=run_id,
            memory_id="item",
            changes={"id": "changed"},
            expected_revision=1,
        )


def test_archive_and_restore(container) -> None:
    run_id = _seed(container)
    container.store.apply_patch(
        StatePatch(
            run_id=run_id,
            expected_revision=1,
            mutations=[MemoryMutation(operation=MemoryOperationType.ARCHIVE, target_id="item")],
            idempotency_key="archive-item-key",
        )
    )
    assert container.store.get_memory(run_id, "item").status is MemoryStatus.ARCHIVED
    container.memory.restore(run_id=run_id, memory_ids=["item"], expected_revision=2)
    assert container.store.get_memory(run_id, "item").status is MemoryStatus.ACTIVE


def test_restore_rejects_active_or_superseded_memory(container) -> None:
    run_id = _seed(container)
    with pytest.raises(StateInvariantError, match="ARCHIVED or EVICTED"):
        container.memory.restore(run_id=run_id, memory_ids=["item"], expected_revision=1)
