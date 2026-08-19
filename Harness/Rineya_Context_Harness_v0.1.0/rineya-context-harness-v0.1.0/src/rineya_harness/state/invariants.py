from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime

from rineya_harness.config import MemoryPolicyConfig
from rineya_harness.domain.enums import (
    ActorType,
    MemoryKind,
    MemoryOperationType,
    MemoryStatus,
    MemoryTemperature,
)
from rineya_harness.domain.models import MemoryItem, StatePatch


class StateInvariantValidator:
    def __init__(self, policy: MemoryPolicyConfig) -> None:
        self.policy = policy

    def validate(
        self,
        *,
        before: dict[str, MemoryItem],
        after: dict[str, MemoryItem],
        patch: StatePatch,
        known_evidence_ids: set[str] | None = None,
    ) -> list[str]:
        reasons: list[str] = []
        now = datetime.now(UTC)
        if len([item for item in after.values() if item.status is MemoryStatus.ACTIVE]) > self.policy.max_active_items:
            reasons.append(f"Active memory exceeds configured limit {self.policy.max_active_items}")

        for memory_id, item in after.items():
            if item.id != memory_id:
                reasons.append(f"Memory key/id mismatch: {memory_id} != {item.id}")
            if item.run_id != patch.run_id:
                reasons.append(f"Memory {item.id} belongs to wrong run {item.run_id}")
            if item.pinned and item.status is not MemoryStatus.ACTIVE:
                reasons.append(f"Pinned memory must remain active: {item.id}")
            if item.pinned and item.temperature is not MemoryTemperature.PINNED:
                reasons.append(f"Pinned memory must use PINNED temperature: {item.id}")
            if item.temperature is MemoryTemperature.PINNED and not item.pinned:
                reasons.append(f"PINNED temperature requires pinned=true: {item.id}")
            if item.status is MemoryStatus.PURGED:
                if item.payload != {"purged": True} or item.source_ids or item.dependencies:
                    reasons.append(f"Purged memory retains protected content: {item.id}")
            if (
                item.kind in {MemoryKind.ACTIVE_EVIDENCE, MemoryKind.RESOLVED_EVIDENCE}
                and item.status is not MemoryStatus.PURGED
                and not item.source_ids
            ):
                reasons.append(f"Evidence memory requires source_ids: {item.id}")
            if known_evidence_ids is not None and item.status is not MemoryStatus.PURGED:
                unknown_sources = item.source_ids - known_evidence_ids
                if unknown_sources:
                    reasons.append(
                        f"Memory cites unknown evidence IDs: {item.id} -> "
                        f"{sorted(unknown_sources)}"
                    )
            if item.expires_at is not None and item.expires_at.tzinfo is None:
                reasons.append(f"expires_at must be timezone-aware: {item.id}")
            if item.valid_from is not None and item.valid_from.tzinfo is None:
                reasons.append(f"valid_from must be timezone-aware: {item.id}")
            if (
                item.expires_at is not None
                and item.expires_at <= now
                and item.status is MemoryStatus.ACTIVE
                and item.kind in {MemoryKind.ACTIVE_EVIDENCE, MemoryKind.RESOLVED_EVIDENCE}
            ):
                reasons.append(f"Expired evidence cannot remain active: {item.id}")

        active_dependents: dict[str, list[str]] = defaultdict(list)
        for item in after.values():
            if item.status is not MemoryStatus.ACTIVE:
                continue
            for dependency_id in item.dependencies:
                dependency = after.get(dependency_id)
                if dependency is None:
                    reasons.append(f"Active memory {item.id} depends on missing {dependency_id}")
                    continue
                active_dependents[dependency_id].append(item.id)
                if dependency.status is not MemoryStatus.ACTIVE:
                    reasons.append(
                        f"Active memory {item.id} depends on inactive {dependency_id} ({dependency.status.value})"
                    )

        for mutation in patch.mutations:
            original = before.get(mutation.target_id) if mutation.target_id else None
            if patch.actor is ActorType.MODEL:
                if mutation.operation in {
                    MemoryOperationType.PIN,
                    MemoryOperationType.UNPIN,
                    MemoryOperationType.PURGE,
                }:
                    reasons.append(
                        f"Model actor cannot perform {mutation.operation.value} lifecycle operation"
                    )
                if (
                    mutation.operation is MemoryOperationType.ADD
                    and mutation.draft is not None
                    and (
                        mutation.draft.pinned
                        or mutation.draft.temperature is MemoryTemperature.PINNED
                        or mutation.draft.kind is MemoryKind.GLOBAL_INVARIANT
                    )
                ):
                    reasons.append(
                        "Model actor cannot create pinned or global-invariant memory"
                    )
                if mutation.operation in {
                    MemoryOperationType.UPDATE,
                    MemoryOperationType.MERGE,
                }:
                    allowed_model_fields = {
                        "summary",
                        "payload",
                        "source_ids",
                        "dependencies",
                        "valid_from",
                        "expires_at",
                        "confidence",
                    }
                    attempted = sorted(set(mutation.changes) - allowed_model_fields)
                    if attempted:
                        reasons.append(
                            "Model actor cannot update protected memory fields: "
                            f"{attempted}"
                        )
                    if original is not None and original.kind is MemoryKind.GLOBAL_INVARIANT:
                        reasons.append(
                            f"Model actor cannot update global invariant: {original.id}"
                        )
            if (
                original is not None
                and original.pinned
                and patch.actor not in {ActorType.SYSTEM, ActorType.ADMIN}
                and mutation.operation
                in {
                    MemoryOperationType.UPDATE,
                    MemoryOperationType.MERGE,
                    MemoryOperationType.UNPIN,
                }
            ):
                reasons.append(
                    f"Pinned memory can only be changed by system/admin: {original.id}"
                )

            if mutation.operation in {
                MemoryOperationType.ARCHIVE,
                MemoryOperationType.EVICT,
                MemoryOperationType.PURGE,
                MemoryOperationType.SUPERSEDE,
            }:
                assert mutation.target_id is not None
                original = before.get(mutation.target_id)
                if original is not None and original.pinned:
                    reasons.append(f"Pinned memory cannot be removed or superseded: {original.id}")
            if mutation.operation in {
                MemoryOperationType.ARCHIVE,
                MemoryOperationType.EVICT,
                MemoryOperationType.SUPERSEDE,
            }:
                assert mutation.target_id is not None
                original = before.get(mutation.target_id)
                if (
                    original is not None
                    and original.status is MemoryStatus.ACTIVE
                    and original.kind is MemoryKind.ACTIVE_EVIDENCE
                ):
                    replacements = [after.get(value) for value in mutation.replacement_ids]
                    active_replacements = [
                        value
                        for value in replacements
                        if value is not None and value.status is MemoryStatus.ACTIVE
                    ]
                    if not active_replacements:
                        reasons.append(
                            f"Active evidence requires an active replacement before removal: {original.id}"
                        )
                    elif not any(
                        value.kind is MemoryKind.RESOLVED_EVIDENCE
                        for value in active_replacements
                    ):
                        reasons.append(
                            f"Active evidence replacement must include resolved evidence: {original.id}"
                        )
            if mutation.operation is MemoryOperationType.PURGE:
                if patch.actor is not ActorType.ADMIN:
                    reasons.append("Persistent purge requires an admin actor")
                if not patch.allow_purge or not self.policy.allow_purge:
                    reasons.append("Persistent purge is disabled")
                expected = (
                    self.policy.purge_confirmation_token.get_secret_value()
                    if self.policy.purge_confirmation_token
                    else None
                )
                if expected is not None and mutation.purge_confirmation_token != expected:
                    reasons.append("Persistent purge confirmation token is invalid")
                if self.policy.archive_before_purge:
                    assert mutation.target_id is not None
                    original = before.get(mutation.target_id)
                    if original is not None and original.status not in {
                        MemoryStatus.ARCHIVED,
                        MemoryStatus.EVICTED,
                        MemoryStatus.SUPERSEDED,
                    }:
                        reasons.append(
                            f"Memory {original.id} must be archived, evicted, or superseded before purge"
                        )
            if mutation.operation is MemoryOperationType.SUPERSEDE:
                for replacement_id in mutation.replacement_ids:
                    replacement = after.get(replacement_id)
                    if replacement is None:
                        reasons.append(f"Supersede replacement missing: {replacement_id}")
                    elif replacement.status is not MemoryStatus.ACTIVE:
                        reasons.append(f"Supersede replacement must be active: {replacement_id}")

        return sorted(set(reasons))
