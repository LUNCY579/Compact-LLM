from __future__ import annotations

import hashlib

from rineya_harness.domain.enums import (
    ActorType,
    DeleteScope,
    MemoryOperationType,
    MemoryStatus,
)
from rineya_harness.domain.models import (
    DeleteMemoryRequest,
    DeleteMemoryResult,
    MemoryDraft,
    MemoryMutation,
    PatchResult,
    StatePatch,
)
from rineya_harness.errors import StateInvariantError
from rineya_harness.state.store import SQLiteStateStore

from .retention import RetentionPolicy


class MemoryService:
    def __init__(
        self,
        store: SQLiteStateStore,
        *,
        retention_policy: RetentionPolicy | None = None,
    ) -> None:
        self.store = store
        self.retention_policy = retention_policy or RetentionPolicy()

    def add(
        self,
        *,
        run_id: str,
        draft: MemoryDraft,
        expected_revision: int,
        actor: ActorType = ActorType.SYSTEM,
        idempotency_key: str | None = None,
    ) -> PatchResult:
        return self.store.apply_patch(
            StatePatch(
                run_id=run_id,
                expected_revision=expected_revision,
                actor=actor,
                mutations=[MemoryMutation(operation=MemoryOperationType.ADD, draft=draft)],
                idempotency_key=idempotency_key or f"add-{run_id}-{draft.id}",
                reason=f"add memory {draft.id}",
            )
        )

    def update(
        self,
        *,
        run_id: str,
        memory_id: str,
        changes: dict[str, object],
        expected_revision: int,
        merge: bool = False,
        actor: ActorType = ActorType.SYSTEM,
        idempotency_key: str | None = None,
    ) -> PatchResult:
        operation = MemoryOperationType.MERGE if merge else MemoryOperationType.UPDATE
        return self.store.apply_patch(
            StatePatch(
                run_id=run_id,
                expected_revision=expected_revision,
                actor=actor,
                mutations=[
                    MemoryMutation(operation=operation, target_id=memory_id, changes=changes)
                ],
                idempotency_key=idempotency_key or f"{operation.value}-{run_id}-{memory_id}-{expected_revision}",
                reason=f"{operation.value} memory {memory_id}",
            )
        )

    def supersede(
        self,
        *,
        run_id: str,
        old_memory_id: str,
        replacement: MemoryDraft,
        expected_revision: int,
        actor: ActorType = ActorType.SYSTEM,
        idempotency_key: str | None = None,
    ) -> PatchResult:
        """Add replacement first and supersede the old item in one atomic patch."""

        key = idempotency_key or f"supersede-{run_id}-{old_memory_id}-{replacement.id}"
        return self.store.apply_patch(
            StatePatch(
                run_id=run_id,
                expected_revision=expected_revision,
                actor=actor,
                mutations=[
                    MemoryMutation(operation=MemoryOperationType.ADD, draft=replacement),
                    MemoryMutation(
                        operation=MemoryOperationType.SUPERSEDE,
                        target_id=old_memory_id,
                        replacement_ids={replacement.id},
                        note="replacement committed in the same transaction",
                    ),
                ],
                idempotency_key=key,
                reason=f"supersede {old_memory_id} with {replacement.id}",
            )
        )

    def delete_memory(
        self,
        request: DeleteMemoryRequest,
        *,
        actor: ActorType = ActorType.MODEL,
        trace_id: str | None = None,
    ) -> DeleteMemoryResult:
        """Execute explicit, auditable forgetting.

        `active_context` and `archive` are reversible. A persistent purge is separately gated by
        repository policy, patch intent, archive-before-purge rules, and an optional confirmation
        token.
        """

        current = {
            item.id: item for item in self.store.list_memory(request.run_id, include_inactive=True)
        }
        missing = [memory_id for memory_id in request.memory_ids if memory_id not in current]
        if missing:
            raise StateInvariantError([f"Unknown memory IDs: {missing}"])
        protected = [
            memory_id
            for memory_id in request.memory_ids
            if self.retention_policy.protected(current[memory_id])
        ]
        if protected:
            raise StateInvariantError([f"Protected memory cannot be deleted: {protected}"])

        invalid_replacements: list[str] = []
        for replacement_id in request.replacement_state_ids:
            replacement = current.get(replacement_id)
            if (
                replacement is None
                or replacement.status is not MemoryStatus.ACTIVE
                or replacement_id in request.memory_ids
            ):
                invalid_replacements.append(replacement_id)
        if invalid_replacements:
            raise StateInvariantError(
                [
                    "Replacement state IDs must exist, remain active, and not be deleted: "
                    f"{sorted(set(invalid_replacements))}"
                ]
            )

        operation = {
            DeleteScope.ACTIVE_CONTEXT: MemoryOperationType.EVICT,
            DeleteScope.ARCHIVE: MemoryOperationType.ARCHIVE,
            DeleteScope.PERSISTENT_PURGE: MemoryOperationType.PURGE,
        }[request.scope]
        mutations = [
            MemoryMutation(
                operation=operation,
                target_id=memory_id,
                replacement_ids=set(request.replacement_state_ids),
                reason_code=request.reason_code,
                note=request.reason,
                purge_confirmation_token=request.confirmation_token,
            )
            for memory_id in request.memory_ids
        ]
        digest = hashlib.sha256(
            "|".join(sorted(request.memory_ids)).encode("utf-8")
        ).hexdigest()[:16]
        result = self.store.apply_patch(
            StatePatch(
                run_id=request.run_id,
                expected_revision=request.expected_state_revision,
                actor=actor,
                mutations=mutations,
                idempotency_key=request.idempotency_key
                or f"delete-{request.scope.value}-{digest}-{request.expected_state_revision}",
                allow_purge=request.scope is DeleteScope.PERSISTENT_PURGE,
                reason=(
                    f"delete memory from {request.scope.value}: {request.reason_code.value}"
                ),
                metadata={"replacement_state_ids": request.replacement_state_ids},
            ),
            trace_id=trace_id,
        )
        return DeleteMemoryResult(
            status="replayed" if result.idempotent_replay else "committed",
            run_id=request.run_id,
            new_state_revision=result.new_revision,
            removed_from_active_context=result.removed_from_active_context,
            archived=result.archived_memory_ids,
            purged=result.purged_memory_ids,
            prompt_rebuild_required=result.prompt_rebuild_required,
            state_hash=result.state_hash,
            event_id=result.event_id,
        )

    def restore(
        self,
        *,
        run_id: str,
        memory_ids: list[str],
        expected_revision: int,
        actor: ActorType = ActorType.SYSTEM,
        idempotency_key: str | None = None,
    ) -> PatchResult:
        return self.store.apply_patch(
            StatePatch(
                run_id=run_id,
                expected_revision=expected_revision,
                actor=actor,
                mutations=[
                    MemoryMutation(operation=MemoryOperationType.RESTORE, target_id=memory_id)
                    for memory_id in memory_ids
                ],
                idempotency_key=idempotency_key
                or f"restore-{run_id}-{'-'.join(sorted(memory_ids))}-{expected_revision}",
                reason=f"restore memory: {', '.join(memory_ids)}",
            )
        )
