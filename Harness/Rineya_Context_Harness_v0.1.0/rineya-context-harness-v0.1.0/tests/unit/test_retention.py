from __future__ import annotations

from rineya_harness.domain.enums import MemoryKind, MemoryStatus, MemoryTemperature
from rineya_harness.domain.models import MemoryItem
from rineya_harness.memory.retention import RetentionPolicy


def _item(kind: MemoryKind, *, pinned: bool = False, status=MemoryStatus.ACTIVE) -> MemoryItem:
    return MemoryItem(
        id=f"m-{kind.value}",
        run_id="run",
        kind=kind,
        summary="x",
        pinned=pinned,
        status=status,
        temperature=MemoryTemperature.PINNED if pinned else MemoryTemperature.WARM,
        revision_created=0,
        revision_updated=0,
    )


def test_retention_protects_invariants_but_allows_replaced_active_evidence() -> None:
    policy = RetentionPolicy()
    assert policy.protected(_item(MemoryKind.GLOBAL_INVARIANT))
    evidence = _item(MemoryKind.ACTIVE_EVIDENCE)
    evidence.source_ids = {"ev"}
    assert not policy.protected(evidence)


def test_policy_can_explicitly_protect_active_evidence() -> None:
    evidence = _item(MemoryKind.ACTIVE_EVIDENCE)
    evidence.source_ids = {"ev"}
    assert RetentionPolicy(protect_active_evidence=True).protected(evidence)


def test_event_history_is_evictable() -> None:
    assert RetentionPolicy().default_evictable(_item(MemoryKind.EVENT_HISTORY))


def test_inactive_nonpinned_is_not_protected() -> None:
    assert not RetentionPolicy().protected(
        _item(MemoryKind.CURRENT_STATE, status=MemoryStatus.ARCHIVED)
    )
