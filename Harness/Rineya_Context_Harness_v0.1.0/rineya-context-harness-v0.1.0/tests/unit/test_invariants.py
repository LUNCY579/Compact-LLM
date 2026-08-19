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


def test_active_dependency_cannot_point_to_evicted_memory(container) -> None:
    run = container.store.create_run(RunRecord(goal="test"))
    container.store.apply_patch(
        StatePatch(
            run_id=run.id,
            expected_revision=0,
            actor=ActorType.SYSTEM,
            mutations=[
                MemoryMutation(
                    operation=MemoryOperationType.ADD,
                    draft=MemoryDraft(id="base", kind=MemoryKind.CURRENT_STATE, summary="base"),
                ),
                MemoryMutation(
                    operation=MemoryOperationType.ADD,
                    draft=MemoryDraft(
                        id="dependent",
                        kind=MemoryKind.CURRENT_STATE,
                        summary="depends",
                        dependencies={"base"},
                    ),
                ),
            ],
            idempotency_key="dependency-seed-key",
        )
    )
    with pytest.raises(StateInvariantError):
        container.store.apply_patch(
            StatePatch(
                run_id=run.id,
                expected_revision=1,
                mutations=[
                    MemoryMutation(operation=MemoryOperationType.EVICT, target_id="base")
                ],
                idempotency_key="dependency-evict-key",
            )
        )


def test_evidence_memory_requires_source_id(container) -> None:
    run = container.store.create_run(RunRecord(goal="test"))
    with pytest.raises(StateInvariantError):
        container.store.apply_patch(
            StatePatch(
                run_id=run.id,
                expected_revision=0,
                mutations=[
                    MemoryMutation(
                        operation=MemoryOperationType.ADD,
                        draft=MemoryDraft(
                            kind=MemoryKind.ACTIVE_EVIDENCE,
                            summary="unsupported evidence",
                        ),
                    )
                ],
                idempotency_key="evidence-source-required",
            )
        )


def test_active_evidence_requires_resolved_replacement(config) -> None:
    from rineya_harness.domain.enums import ActorType, MemoryOperationType, ReasonCode
    from rineya_harness.domain.models import MemoryItem, MemoryMutation, StatePatch
    from rineya_harness.state.invariants import StateInvariantValidator

    raw = MemoryItem(
        id="raw",
        run_id="run",
        kind=MemoryKind.ACTIVE_EVIDENCE,
        summary="raw",
        source_ids={"ev1"},
        revision_created=0,
        revision_updated=0,
    )
    after = {"raw": raw.model_copy(deep=True)}
    after["raw"].status = MemoryStatus.EVICTED
    after["raw"].temperature = MemoryTemperature.COLD
    patch = StatePatch(
        run_id="run",
        expected_revision=0,
        actor=ActorType.MODEL,
        mutations=[
            MemoryMutation(
                operation=MemoryOperationType.EVICT,
                target_id="raw",
                reason_code=ReasonCode.SOURCE_REDUCED_TO_CLAIM,
            )
        ],
        idempotency_key="evict-raw",
    )
    reasons = StateInvariantValidator(config.memory).validate(
        before={"raw": raw}, after=after, patch=patch
    )
    assert any("requires an active replacement" in reason for reason in reasons)


def test_active_evidence_accepts_resolved_replacement(config) -> None:
    from rineya_harness.domain.enums import ActorType, MemoryOperationType, ReasonCode
    from rineya_harness.domain.models import MemoryItem, MemoryMutation, StatePatch
    from rineya_harness.state.invariants import StateInvariantValidator

    raw = MemoryItem(
        id="raw",
        run_id="run",
        kind=MemoryKind.ACTIVE_EVIDENCE,
        summary="raw",
        source_ids={"ev1"},
        revision_created=0,
        revision_updated=0,
    )
    resolved = MemoryItem(
        id="resolved",
        run_id="run",
        kind=MemoryKind.RESOLVED_EVIDENCE,
        summary="claim",
        source_ids={"ev1"},
        revision_created=1,
        revision_updated=1,
    )
    after = {"raw": raw.model_copy(deep=True), "resolved": resolved}
    after["raw"].status = MemoryStatus.EVICTED
    after["raw"].temperature = MemoryTemperature.COLD
    patch = StatePatch(
        run_id="run",
        expected_revision=0,
        actor=ActorType.MODEL,
        mutations=[
            MemoryMutation(
                operation=MemoryOperationType.EVICT,
                target_id="raw",
                replacement_ids={"resolved"},
                reason_code=ReasonCode.SOURCE_REDUCED_TO_CLAIM,
            )
        ],
        idempotency_key="evict-raw-with-replacement",
    )
    reasons = StateInvariantValidator(config.memory).validate(
        before={"raw": raw}, after=after, patch=patch
    )
    assert not reasons


def test_persistent_purge_requires_admin_actor(config) -> None:
    from rineya_harness.domain.enums import ActorType, MemoryOperationType
    from rineya_harness.domain.models import MemoryItem, MemoryMutation, StatePatch
    from rineya_harness.state.invariants import StateInvariantValidator

    original = MemoryItem(
        id="archived",
        run_id="run",
        kind=MemoryKind.EVENT_HISTORY,
        summary="old",
        status=MemoryStatus.ARCHIVED,
        temperature=MemoryTemperature.COLD,
        revision_created=0,
        revision_updated=0,
    )
    purged = original.model_copy(deep=True)
    purged.status = MemoryStatus.PURGED
    purged.summary = "[purged]"
    purged.payload = {"purged": True}
    purged.source_ids = set()
    purged.dependencies = set()
    purged.metadata = {"purged_at_revision": 1}
    patch = StatePatch(
        run_id="run",
        expected_revision=0,
        actor=ActorType.MODEL,
        mutations=[
            MemoryMutation(operation=MemoryOperationType.PURGE, target_id="archived")
        ],
        idempotency_key="unauthorized-purge",
        allow_purge=True,
    )
    enabled = config.memory.model_copy(update={"allow_purge": True})
    reasons = StateInvariantValidator(enabled).validate(
        before={"archived": original}, after={"archived": purged}, patch=patch
    )
    assert "Persistent purge requires an admin actor" in reasons


def test_evidence_memory_rejects_unknown_source_id(container) -> None:
    run = container.store.create_run(RunRecord(goal="unknown evidence"))
    with pytest.raises(StateInvariantError, match="unknown evidence IDs"):
        container.store.apply_patch(
            StatePatch(
                run_id=run.id,
                expected_revision=0,
                mutations=[
                    MemoryMutation(
                        operation=MemoryOperationType.ADD,
                        draft=MemoryDraft(
                            kind=MemoryKind.RESOLVED_EVIDENCE,
                            summary="claim",
                            source_ids={"missing-evidence"},
                        ),
                    )
                ],
                idempotency_key="unknown-evidence-source",
            )
        )
