from __future__ import annotations

from rineya_harness.domain.enums import MemoryKind
from rineya_harness.domain.models import MemoryDraft, RunRecord


def test_checkpoint_restore(container) -> None:
    run = container.store.create_run(RunRecord(goal="test"))
    container.memory.add(
        run_id=run.id,
        draft=MemoryDraft(id="before", kind=MemoryKind.CURRENT_STATE, summary="before"),
        expected_revision=0,
    )
    checkpoint = container.store.create_checkpoint(run.id, label="before-change")
    container.memory.add(
        run_id=run.id,
        draft=MemoryDraft(id="after", kind=MemoryKind.CURRENT_STATE, summary="after"),
        expected_revision=1,
    )
    result = container.store.restore_checkpoint(
        run.id, checkpoint.id, expected_revision=2
    )
    assert result.new_revision == 3
    ids = {item.id for item in container.store.list_memory(run.id, include_inactive=True)}
    assert ids == {"before"}
