from __future__ import annotations

from dataclasses import dataclass

from rineya_harness.domain.enums import ActorType, DeleteScope, ReasonCode
from rineya_harness.domain.models import DeleteMemoryRequest, DeleteMemoryResult
from rineya_harness.memory.retention import RetentionPolicy
from rineya_harness.memory.service import MemoryService
from rineya_harness.state.store import SQLiteStateStore

from .assembler import ContextAssembler


@dataclass(slots=True)
class CompactionResult:
    before_items: int
    after_items: int
    deleted_ids: list[str]
    delete_result: DeleteMemoryResult | None


class DeterministicCompactor:
    """Baseline compactor used when active state exceeds the configured item budget."""

    def __init__(
        self,
        store: SQLiteStateStore,
        memory_service: MemoryService,
        *,
        retention_policy: RetentionPolicy | None = None,
    ) -> None:
        self.store = store
        self.memory_service = memory_service
        self.retention_policy = retention_policy or RetentionPolicy()

    def compact_to_item_limit(
        self,
        *,
        run_id: str,
        max_items: int,
        expected_revision: int,
        trace_id: str | None = None,
    ) -> CompactionResult:
        active = self.store.list_memory(run_id)
        before = len(active)
        excess = max(0, before - max_items)
        if excess == 0:
            return CompactionResult(before, before, [], None)
        candidates = [item for item in reversed(active) if self.retention_policy.default_evictable(item)]
        selected = [item.id for item in candidates[:excess]]
        if len(selected) < excess:
            # Never delete protected state just to satisfy a soft item budget.
            return CompactionResult(before, before, [], None)
        result = self.memory_service.delete_memory(
            DeleteMemoryRequest(
                run_id=run_id,
                memory_ids=selected,
                scope=DeleteScope.ACTIVE_CONTEXT,
                reason_code=ReasonCode.POLICY_RETENTION,
                reason=f"compact active state to at most {max_items} records",
                expected_state_revision=expected_revision,
                idempotency_key=f"compact-{run_id}-{expected_revision}-{max_items}",
            ),
            actor=ActorType.SYSTEM,
            trace_id=trace_id,
        )
        return CompactionResult(before, before - len(selected), selected, result)
