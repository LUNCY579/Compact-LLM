"""Causally valid evidence scoring at an intermediate cache boundary."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..benchmarks.evidence import (
    FrozenEvidenceAnnotation,
    score_posthoc_evidence_retention,
)
from .evidence_endpoints import score_evidence_retention_endpoint


@dataclass(frozen=True, slots=True)
class BoundaryEvidenceScore:
    """Evidence state restricted to facts fully observed by this boundary."""

    true_length: int
    fully_observed_fact_ids: tuple[str, ...]
    partially_observed_fact_ids: tuple[str, ...]
    not_yet_observed_fact_ids: tuple[str, ...]
    evidence: dict[str, Any] | None

    @property
    def status(self) -> str:
        if not self.fully_observed_fact_ids:
            return "NO_REQUIRED_FACT_FULLY_OBSERVED"
        if self.partially_observed_fact_ids or self.not_yet_observed_fact_ids:
            return "SCORED_OBSERVED_FACT_SUBSET"
        return "SCORED_ALL_REQUIRED_FACTS_OBSERVED"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "true_length": self.true_length,
            "fully_observed_fact_ids": list(self.fully_observed_fact_ids),
            "partially_observed_fact_ids": list(self.partially_observed_fact_ids),
            "not_yet_observed_fact_ids": list(self.not_yet_observed_fact_ids),
            "fully_observed_fact_count": len(self.fully_observed_fact_ids),
            "evidence": self.evidence,
            "causal_scope": (
                "only facts whose complete frozen token span is before true_length are scored; "
                "future and boundary-crossing facts are never counted as evicted"
            ),
        }


def score_observed_boundary_evidence(
    annotation: FrozenEvidenceAnnotation,
    resident_positions: Sequence[Sequence[Sequence[int]]],
    *,
    true_length: int,
) -> BoundaryEvidenceScore:
    """Score only evidence facts completely consumed before ``true_length``."""

    if true_length < 0 or true_length > annotation.token_count:
        raise ValueError("true_length must be within the frozen prompt token range")
    fully = []
    partial = []
    unseen = []
    for fact in annotation.facts:
        seen = sum(position < true_length for position in fact.token_positions)
        if seen == len(fact.token_positions):
            fully.append(fact)
        elif seen:
            partial.append(fact)
        else:
            unseen.append(fact)
    evidence: dict[str, Any] | None = None
    if fully:
        subset = FrozenEvidenceAnnotation(
            tokenizer_fingerprint=annotation.tokenizer_fingerprint,
            prompt_sha256=annotation.prompt_sha256,
            token_ids_sha256=annotation.token_ids_sha256,
            token_count=annotation.token_count,
            facts=tuple(fully),
        )
        retention = score_posthoc_evidence_retention(subset, resident_positions)
        evidence = {
            **retention.to_dict(),
            **score_evidence_retention_endpoint(retention).to_dict(),
        }
    return BoundaryEvidenceScore(
        true_length=true_length,
        fully_observed_fact_ids=tuple(fact.fact_id for fact in fully),
        partially_observed_fact_ids=tuple(fact.fact_id for fact in partial),
        not_yet_observed_fact_ids=tuple(fact.fact_id for fact in unseen),
        evidence=evidence,
    )
