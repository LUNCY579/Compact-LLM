from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any

from rineya_harness.domain.enums import (
    MemoryKind,
    MemoryOperationType,
    MemoryStatus,
    MemoryTemperature,
)
from rineya_harness.domain.models import MemoryItem, MemoryMutation
from rineya_harness.errors import NotFoundError, StateInvariantError


def _deep_merge(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _require(items: dict[str, MemoryItem], target_id: str | None) -> MemoryItem:
    if not target_id or target_id not in items:
        raise NotFoundError("memory", target_id or "<missing>")
    return items[target_id]


def project_mutation(
    items: dict[str, MemoryItem],
    mutation: MemoryMutation,
    *,
    run_id: str,
    revision: int,
    now: datetime,
) -> str:
    operation = mutation.operation
    if operation is MemoryOperationType.ADD:
        assert mutation.draft is not None
        if mutation.draft.id in items:
            raise StateInvariantError([f"Memory already exists: {mutation.draft.id}"])
        item = MemoryItem(
            **mutation.draft.model_dump(mode="python"),
            run_id=run_id,
            revision_created=revision,
            revision_updated=revision,
            created_at=now,
            updated_at=now,
        )
        items[item.id] = item
        return item.id

    item = _require(items, mutation.target_id)
    if item.status is MemoryStatus.PURGED:
        raise StateInvariantError([f"Purged memory is immutable: {item.id}"])

    if operation in {MemoryOperationType.UPDATE, MemoryOperationType.MERGE}:
        forbidden = {"id", "run_id", "created_at", "revision_created"}
        illegal = sorted(forbidden.intersection(mutation.changes))
        if illegal:
            raise StateInvariantError([f"Immutable fields cannot change: {illegal}"])
        changes = deepcopy(mutation.changes)
        if operation is MemoryOperationType.MERGE and "payload" in changes:
            incoming = changes.pop("payload")
            if not isinstance(incoming, dict):
                raise StateInvariantError(["MERGE payload must be an object"])
            item.payload = _deep_merge(item.payload, incoming)
        for key, value in changes.items():
            if not hasattr(item, key):
                raise StateInvariantError([f"Unknown memory field: {key}"])
            setattr(item, key, value)
    elif operation is MemoryOperationType.SUPERSEDE:
        item.status = MemoryStatus.SUPERSEDED
        item.kind = MemoryKind.SUPERSEDED_STATE
        item.temperature = MemoryTemperature.COLD
        item.superseded_by = sorted(mutation.replacement_ids)[0]
    elif operation is MemoryOperationType.ARCHIVE:
        item.status = MemoryStatus.ARCHIVED
        item.temperature = MemoryTemperature.COLD
    elif operation is MemoryOperationType.EVICT:
        item.status = MemoryStatus.EVICTED
        item.temperature = MemoryTemperature.COLD
    elif operation is MemoryOperationType.RESTORE:
        if item.status not in {MemoryStatus.ARCHIVED, MemoryStatus.EVICTED}:
            raise StateInvariantError(
                [
                    "RESTORE requires ARCHIVED or EVICTED memory; "
                    f"use checkpoint rollback for {item.status.value}: {item.id}"
                ]
            )
        item.status = MemoryStatus.ACTIVE
        item.temperature = MemoryTemperature.PINNED if item.pinned else MemoryTemperature.WARM
    elif operation is MemoryOperationType.PURGE:
        item.status = MemoryStatus.PURGED
        item.summary = "[purged]"
        item.payload = {"purged": True}
        item.source_ids = set()
        item.dependencies = set()
        item.metadata = {"purged_at_revision": revision}
        item.superseded_by = None
        item.temperature = MemoryTemperature.COLD
    elif operation is MemoryOperationType.PIN:
        item.pinned = True
        item.status = MemoryStatus.ACTIVE
        item.temperature = MemoryTemperature.PINNED
    elif operation is MemoryOperationType.UNPIN:
        # Set the coupled fields atomically; assignment validation would otherwise re-pin the
        # record after observing a transient pinned=False / temperature=PINNED combination.
        object.__setattr__(item, "pinned", False)
        object.__setattr__(item, "temperature", MemoryTemperature.WARM)
    else:
        raise StateInvariantError([f"Unsupported mutation: {operation.value}"])

    item.revision_updated = revision
    item.updated_at = now
    items[item.id] = MemoryItem.model_validate(item.model_dump(mode="python"))
    return item.id
