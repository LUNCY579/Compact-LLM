from __future__ import annotations

import hashlib
import pytest

from rineya_harness.domain.enums import ActorType, MemoryKind, MemoryOperationType
from rineya_harness.domain.models import MemoryDraft, MemoryMutation, RunRecord, StatePatch
from rineya_harness.errors import RevisionConflictError


def test_create_and_read_run(container) -> None:
    run = container.store.create_run(RunRecord(goal="test"))
    assert container.store.get_run(run.id).goal == "test"
    assert container.store.list_events(run.id)[0].event_type == "run_created"


def test_apply_patch_and_idempotent_replay(container) -> None:
    run = container.store.create_run(RunRecord(goal="test"))
    patch = StatePatch(
        run_id=run.id,
        expected_revision=0,
        actor=ActorType.SYSTEM,
        mutations=[
            MemoryMutation(
                operation=MemoryOperationType.ADD,
                draft=MemoryDraft(id="state", kind=MemoryKind.CURRENT_STATE, summary="current"),
            )
        ],
        idempotency_key="test-idempotency-key",
    )
    first = container.store.apply_patch(patch)
    second = container.store.apply_patch(patch)
    assert first.new_revision == 1
    assert second.idempotent_replay is True
    assert len(container.store.list_memory(run.id)) == 1


def test_revision_conflict(container) -> None:
    run = container.store.create_run(RunRecord(goal="test"))
    with pytest.raises(RevisionConflictError):
        container.store.apply_patch(
            StatePatch(
                run_id=run.id,
                expected_revision=1,
                mutations=[
                    MemoryMutation(
                        operation=MemoryOperationType.ADD,
                        draft=MemoryDraft(kind=MemoryKind.CURRENT_STATE, summary="x"),
                    )
                ],
                idempotency_key="conflict-test-key",
            )
        )


def test_export_is_hashed(container) -> None:
    run = container.store.create_run(RunRecord(goal="test"))
    payload = container.store.export_run(run.id)
    assert len(payload["export_sha256"]) == 64


def test_idempotency_key_rejects_different_semantic_patch(container) -> None:
    from rineya_harness.domain.enums import ActorType, MemoryKind, MemoryOperationType
    from rineya_harness.domain.models import MemoryDraft, MemoryMutation, RunRecord, StatePatch
    from rineya_harness.errors import ConflictError

    run = container.store.create_run(RunRecord(goal="idempotency conflict"))
    first = StatePatch(
        run_id=run.id,
        expected_revision=0,
        actor=ActorType.SYSTEM,
        mutations=[
            MemoryMutation(
                operation=MemoryOperationType.ADD,
                draft=MemoryDraft(id="one", kind=MemoryKind.CURRENT_STATE, summary="one"),
            )
        ],
        idempotency_key="same-semantic-key",
    )
    container.store.apply_patch(first)
    different = StatePatch(
        run_id=run.id,
        expected_revision=0,
        actor=ActorType.SYSTEM,
        mutations=[
            MemoryMutation(
                operation=MemoryOperationType.ADD,
                draft=MemoryDraft(id="two", kind=MemoryKind.CURRENT_STATE, summary="two"),
            )
        ],
        idempotency_key="same-semantic-key",
    )
    with pytest.raises(ConflictError, match="different state patch"):
        container.store.apply_patch(different)


def test_evidence_ledger_is_immutable(container) -> None:
    from rineya_harness.domain.models import EvidenceRecord, RunRecord
    from rineya_harness.errors import ConflictError

    run = container.store.create_run(RunRecord(goal="immutable evidence"))
    original = EvidenceRecord(
        id="ev-fixed",
        run_id=run.id,
        source_uri="https://example.org/source",
        passage="Original passage.",
        content_hash=hashlib.sha256("Original passage.".encode()).hexdigest(),
    )
    container.store.add_evidence([original])
    container.store.add_evidence([original])
    changed = original.model_copy(update={"passage": "Changed passage."})
    with pytest.raises(ConflictError, match="Immutable evidence"):
        container.store.add_evidence([changed])
    assert container.store.list_evidence(run.id)[0].passage == "Original passage."


def test_claim_ledger_is_immutable(container) -> None:
    from rineya_harness.domain.models import ClaimRecord, RunRecord
    from rineya_harness.errors import ConflictError

    run = container.store.create_run(RunRecord(goal="immutable claims"))
    original = ClaimRecord(id="claim-fixed", run_id=run.id, statement="Original claim.")
    container.store.add_claims([original])
    container.store.add_claims([original])
    changed = original.model_copy(update={"statement": "Changed claim."})
    with pytest.raises(ConflictError, match="Immutable claim"):
        container.store.add_claims([changed])


def test_claim_ledger_rejects_unknown_evidence(container) -> None:
    from rineya_harness.domain.enums import SupportStatus
    from rineya_harness.domain.models import ClaimRecord, RunRecord
    from rineya_harness.errors import ConflictError

    run = container.store.create_run(RunRecord(goal="claim provenance"))
    claim = ClaimRecord(
        run_id=run.id,
        statement="Supported-looking claim.",
        evidence_ids={"missing"},
        support_status=SupportStatus.SUPPORTED,
    )
    with pytest.raises(ConflictError, match="unknown evidence IDs"):
        container.store.add_claims([claim])


def test_supported_claim_requires_evidence(container) -> None:
    from rineya_harness.domain.enums import SupportStatus
    from rineya_harness.domain.models import ClaimRecord, RunRecord
    from rineya_harness.errors import ConflictError

    run = container.store.create_run(RunRecord(goal="claim evidence"))
    claim = ClaimRecord(
        run_id=run.id,
        statement="Unsupported support declaration.",
        support_status=SupportStatus.SUPPORTED,
    )
    with pytest.raises(ConflictError, match="requires evidence IDs"):
        container.store.add_claims([claim])
