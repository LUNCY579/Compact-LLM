from __future__ import annotations

from urllib.parse import urlparse

from rineya_harness.domain.enums import FinishAction, SupportStatus
from rineya_harness.domain.models import AnswerabilityReport, ClaimRecord, EvidenceRecord


class AnswerabilityGate:
    """Code-driven answerability check; model confidence alone is never sufficient."""

    def __init__(
        self,
        *,
        minimum_independent_domains: int = 1,
        require_required_claim: bool = True,
    ) -> None:
        self.minimum_independent_domains = minimum_independent_domains
        self.require_required_claim = require_required_claim

    def evaluate(
        self,
        *,
        evidence: list[EvidenceRecord],
        claims: list[ClaimRecord],
        unresolved_questions: list[str] | None = None,
        search_budget_remaining: bool = False,
    ) -> AnswerabilityReport:
        unresolved_questions = unresolved_questions or []
        evidence_by_id = {record.id: record for record in evidence}
        required_claims = [claim for claim in claims if claim.required_for_answer]
        unsupported: list[str] = []
        conflicted: list[str] = []
        stale: list[str] = []
        supporting_evidence_ids: set[str] = set()

        for claim in required_claims:
            if claim.support_status is SupportStatus.CONFLICTED:
                conflicted.append(claim.id)
            elif claim.support_status is SupportStatus.STALE:
                stale.append(claim.id)
            elif claim.support_status is not SupportStatus.SUPPORTED:
                unsupported.append(claim.id)
            elif not claim.evidence_ids or not claim.evidence_ids.issubset(evidence_by_id):
                unsupported.append(claim.id)
            else:
                supporting_evidence_ids.update(claim.evidence_ids)

        supporting_evidence = [
            evidence_by_id[evidence_id]
            for evidence_id in sorted(supporting_evidence_ids)
            if evidence_id in evidence_by_id
        ]
        domains = {
            (urlparse(str(record.source_uri)).hostname or "").casefold()
            for record in supporting_evidence
            if urlparse(str(record.source_uri)).hostname
        }
        reasons: list[str] = []
        if self.require_required_claim and not required_claims:
            reasons.append("no_required_claims")
        if unsupported:
            reasons.append("required_claim_unsupported")
        if conflicted:
            reasons.append("critical_conflict_unresolved")
        if stale:
            reasons.append("freshness_requirement_failed")
        if unresolved_questions:
            reasons.append("open_information_gap")
        if required_claims and len(domains) < self.minimum_independent_domains:
            reasons.append("insufficient_source_diversity")

        answerable = not reasons
        retryable = bool(
            "no_required_claims" in reasons
            or unsupported
            or stale
            or unresolved_questions
            or (required_claims and len(domains) < self.minimum_independent_domains)
        )
        if answerable:
            action = FinishAction.ANSWER
        elif search_budget_remaining and retryable and not conflicted:
            action = FinishAction.SEARCH_MORE
        else:
            action = FinishAction.ABSTAIN
        return AnswerabilityReport(
            answerable=answerable,
            action=action,
            reason_codes=reasons,
            unsupported_claim_ids=unsupported,
            conflicted_claim_ids=conflicted,
            stale_claim_ids=stale,
            unresolved_questions=unresolved_questions,
            evidence_source_count=len(supporting_evidence),
            independent_domain_count=len(domains),
            metadata={"supporting_evidence_ids": sorted(supporting_evidence_ids)},
        )
