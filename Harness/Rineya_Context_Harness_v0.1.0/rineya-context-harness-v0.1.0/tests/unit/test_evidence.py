from __future__ import annotations

import hashlib

from rineya_harness.domain.enums import FinishAction, SupportStatus
from rineya_harness.domain.models import ClaimRecord, EvidenceRecord, FetchedDocument
from rineya_harness.evidence.ranker import LexicalPassageRanker
from rineya_harness.evidence.verifier import AnswerabilityGate


def test_ranker_prefers_query_terms() -> None:
    text = "Cats sleep often.\n" + "x " * 500 + "\nRineya uses recursive state forgetting."
    document = FetchedDocument(
        requested_url="https://example.org",
        final_url="https://example.org",
        text=text,
        content_type="text/plain",
        content_sha256=hashlib.sha256(text.encode()).hexdigest(),
        status_code=200,
        byte_count=len(text),
    )
    ranker = LexicalPassageRanker(passage_chars=120, overlap_chars=20)
    ranked = ranker.rank("recursive state forgetting", ranker.split(document), limit=1)
    assert "recursive state forgetting" in ranked[0].text


def test_answerability_requires_supported_claim() -> None:
    gate = AnswerabilityGate(require_required_claim=True)
    report = gate.evaluate(evidence=[], claims=[], search_budget_remaining=False)
    assert report.answerable is False
    assert report.action is FinishAction.ABSTAIN
    assert "no_required_claims" in report.reason_codes


def test_answerability_accepts_attributed_claim() -> None:
    evidence = EvidenceRecord(
        id="ev1",
        run_id="run",
        source_uri="https://example.org/a",
        passage="Supported fact.",
        content_hash=hashlib.sha256("Supported fact.".encode()).hexdigest(),
    )
    claim = ClaimRecord(
        run_id="run",
        statement="Supported fact.",
        evidence_ids={"ev1"},
        support_status=SupportStatus.SUPPORTED,
        required_for_answer=True,
    )
    report = AnswerabilityGate().evaluate(evidence=[evidence], claims=[claim])
    assert report.answerable is True
    assert report.action is FinishAction.ANSWER
