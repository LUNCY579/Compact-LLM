from __future__ import annotations

import hashlib

from rineya_harness.domain.enums import FinishAction, SupportStatus
from rineya_harness.domain.models import ClaimRecord, EvidenceRecord
from rineya_harness.evidence.verifier import AnswerabilityGate


def _evidence() -> EvidenceRecord:
    return EvidenceRecord(
        id="ev",
        run_id="run",
        source_uri="https://example.org",
        passage="fact",
        content_hash=hashlib.sha256("fact".encode()).hexdigest(),
    )


def test_unsupported_claim_searches_more_when_budget_remains() -> None:
    claim = ClaimRecord(
        run_id="run",
        statement="fact",
        support_status=SupportStatus.UNSUPPORTED,
        required_for_answer=True,
    )
    report = AnswerabilityGate().evaluate(
        evidence=[_evidence()], claims=[claim], search_budget_remaining=True
    )
    assert report.action is FinishAction.SEARCH_MORE


def test_conflict_abstains_even_with_search_budget() -> None:
    claim = ClaimRecord(
        run_id="run",
        statement="fact",
        evidence_ids={"ev"},
        support_status=SupportStatus.CONFLICTED,
        required_for_answer=True,
    )
    report = AnswerabilityGate().evaluate(
        evidence=[_evidence()], claims=[claim], search_budget_remaining=True
    )
    assert report.action is FinishAction.ABSTAIN


def test_stale_claim_is_reported() -> None:
    claim = ClaimRecord(
        run_id="run",
        statement="fact",
        evidence_ids={"ev"},
        support_status=SupportStatus.STALE,
        required_for_answer=True,
    )
    report = AnswerabilityGate().evaluate(evidence=[_evidence()], claims=[claim])
    assert report.stale_claim_ids == [claim.id]


def test_source_diversity_counts_only_evidence_linked_to_required_claims() -> None:
    other = EvidenceRecord(
        id="ev_other",
        run_id="run",
        source_uri="https://other.example.net",
        passage="unrelated",
        content_hash=hashlib.sha256(b"unrelated").hexdigest(),
    )
    claim = ClaimRecord(
        run_id="run",
        statement="fact",
        evidence_ids={"ev"},
        support_status=SupportStatus.SUPPORTED,
        required_for_answer=True,
    )
    report = AnswerabilityGate(minimum_independent_domains=2).evaluate(
        evidence=[_evidence(), other],
        claims=[claim],
    )
    assert report.answerable is False
    assert report.independent_domain_count == 1
    assert report.evidence_source_count == 1
