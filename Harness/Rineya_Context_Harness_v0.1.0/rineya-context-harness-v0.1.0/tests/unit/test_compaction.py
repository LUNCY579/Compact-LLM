from __future__ import annotations

from rineya_harness.context.compaction import DeterministicCompactor
from rineya_harness.domain.enums import MemoryKind, MemoryStatus
from rineya_harness.domain.models import MemoryDraft, RunRecord


def test_compactor_evicts_only_default_evictable(container) -> None:
    run = container.store.create_run(RunRecord(goal="compact"))
    drafts = [
        MemoryDraft(id="goal", kind=MemoryKind.GLOBAL_INVARIANT, summary="goal", pinned=True),
        MemoryDraft(id="event1", kind=MemoryKind.EVENT_HISTORY, summary="old event"),
        MemoryDraft(id="event2", kind=MemoryKind.EVENT_HISTORY, summary="old event 2"),
    ]
    revision = 0
    for draft in drafts:
        container.memory.add(run_id=run.id, draft=draft, expected_revision=revision)
        revision += 1
    result = DeterministicCompactor(container.store, container.memory).compact_to_item_limit(
        run_id=run.id,
        max_items=2,
        expected_revision=revision,
    )
    assert len(result.deleted_ids) == 1
    assert container.store.get_memory(run.id, "goal").status is MemoryStatus.ACTIVE


def test_compactor_noop_under_limit(container) -> None:
    run = container.store.create_run(RunRecord(goal="compact"))
    result = DeterministicCompactor(container.store, container.memory).compact_to_item_limit(
        run_id=run.id, max_items=2, expected_revision=0
    )
    assert result.delete_result is None
