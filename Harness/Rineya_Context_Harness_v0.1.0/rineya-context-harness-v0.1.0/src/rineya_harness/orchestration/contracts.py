from __future__ import annotations

from typing import Any, Protocol

from pydantic import Field, model_validator

from rineya_harness.domain.enums import (
    MemoryKind,
    MemoryTemperature,
    ReasonCode,
    SupportStatus,
)
from rineya_harness.domain.models import (
    AnswerabilityReport,
    ContextBundle,
    EvidenceRecord,
    MemoryItem,
    StrictModel,
)

_MODEL_MUTABLE_MEMORY_FIELDS = frozenset(
    {
        "summary",
        "payload",
        "source_ids",
        "dependencies",
        "valid_from",
        "expires_at",
        "confidence",
    }
)

class QueryPlan(StrictModel):
    """One bounded search query for one explicitly named information gap."""

    gap_id: str = Field(default="primary", min_length=1, max_length=256)
    query: str = Field(min_length=1, max_length=2_000)
    reason: str = Field(default="resolve the highest-priority information gap", max_length=2_000)


class ClaimProposal(StrictModel):
    """A proposed atomic claim linked to exact immutable evidence identifiers."""

    statement: str = Field(min_length=1, max_length=8_000)
    evidence_ids: set[str] = Field(default_factory=set)
    support_status: SupportStatus = SupportStatus.UNSUPPORTED
    required_for_answer: bool = True
    freshness_required: bool = False
    conflict_group: str | None = Field(default=None, max_length=256)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryProposal(StrictModel):
    """Compact future-usable state proposed by a reconstruction policy."""

    kind: MemoryKind
    summary: str = Field(min_length=1, max_length=8_000)
    payload: dict[str, Any] = Field(default_factory=dict)
    source_ids: set[str] = Field(default_factory=set)
    dependencies: set[str] = Field(default_factory=set)
    temperature: MemoryTemperature = MemoryTemperature.WARM
    pinned: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def prevent_model_owned_invariants(self) -> "MemoryProposal":
        if (
            self.pinned
            or self.temperature is MemoryTemperature.PINNED
            or self.kind is MemoryKind.GLOBAL_INVARIANT
        ):
            raise ValueError("reconstruction policy cannot create pinned/global-invariant memory")
        return self


class MemoryUpdateProposal(StrictModel):
    target_id: str = Field(min_length=1, max_length=256)
    changes: dict[str, Any] = Field(min_length=1)
    merge: bool = False

    @model_validator(mode="after")
    def restrict_model_mutations(self) -> "MemoryUpdateProposal":
        unsupported = sorted(set(self.changes) - _MODEL_MUTABLE_MEMORY_FIELDS)
        if unsupported:
            raise ValueError(f"model memory update contains forbidden fields: {unsupported}")
        if self.merge and "payload" not in self.changes:
            raise ValueError("merge=true requires a payload change")
        return self


class DeletionProposal(StrictModel):
    """Model proposal only; deterministic policy still authorizes the transition."""

    memory_ids: list[str] = Field(default_factory=list, max_length=128)
    reason_code: ReasonCode = ReasonCode.IRRELEVANT_TO_OPEN_GOALS
    reason: str = Field(default="no longer needed for future decisions", max_length=2_000)
    replacement_state_ids: list[str] = Field(default_factory=list, max_length=128)


class ReconstructionOutcome(StrictModel):
    claims: list[ClaimProposal] = Field(default_factory=list, max_length=64)
    memory_additions: list[MemoryProposal] = Field(default_factory=list, max_length=64)
    memory_updates: list[MemoryUpdateProposal] = Field(default_factory=list, max_length=64)
    deletion: DeletionProposal = Field(default_factory=DeletionProposal)
    unresolved_questions: list[str] = Field(default_factory=list, max_length=32)
    summary: str = Field(default="", max_length=4_000)

    @model_validator(mode="after")
    def require_meaningful_output(self) -> "ReconstructionOutcome":
        if not (
            self.claims
            or self.memory_additions
            or self.memory_updates
            or self.deletion.memory_ids
            or self.unresolved_questions
        ):
            raise ValueError("reconstruction must produce claims, state changes, deletion, or gaps")
        return self


class AnswerDraft(StrictModel):
    answer: str = Field(min_length=1, max_length=20_000)
    citations: list[str] = Field(default_factory=list, max_length=64)
    confidence_note: str | None = Field(default=None, max_length=2_000)


class ResearchPolicy(Protocol):
    """Small-model-facing policy surface.

    The workflow, permissions, state transitions, budgets, deletion guards, and answerability gate
    remain ordinary software. A policy only proposes ambiguous semantic judgments.
    """

    async def plan_query(
        self,
        *,
        question: str,
        active_memory: list[MemoryItem],
        unresolved_questions: list[str],
        prior_queries: list[str],
    ) -> QueryPlan: ...

    async def reconstruct(
        self,
        *,
        question: str,
        active_memory: list[MemoryItem],
        evidence: list[EvidenceRecord],
    ) -> ReconstructionOutcome: ...

    async def direct_answer(self, *, question: str, context: ContextBundle) -> AnswerDraft: ...

    async def grounded_answer(
        self,
        *,
        question: str,
        context: ContextBundle,
        answerability: AnswerabilityReport,
        evidence: list[EvidenceRecord],
    ) -> AnswerDraft: ...
