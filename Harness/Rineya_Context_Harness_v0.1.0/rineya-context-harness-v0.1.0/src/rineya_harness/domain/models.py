from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator

from rineya_harness.ids import new_id

from .enums import (
    ActorType,
    DeleteScope,
    FinishAction,
    MemoryKind,
    MemoryOperationType,
    MemoryStatus,
    MemoryTemperature,
    ReasonCode,
    RouteAction,
    RunMode,
    RunStatus,
    SupportStatus,
    TraceEventType,
    WorkflowNode,
)


def utc_now() -> datetime:
    return datetime.now(UTC)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class RunRecord(StrictModel):
    id: str = Field(default_factory=lambda: new_id("run"), min_length=1, max_length=256)
    goal: str = Field(min_length=1, max_length=20_000)
    mode: RunMode = RunMode.AUTO
    status: RunStatus = RunStatus.CREATED
    revision: int = Field(default=0, ge=0)
    current_node: WorkflowNode = WorkflowNode.INTAKE
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class MemoryDraft(StrictModel):
    id: str = Field(default_factory=lambda: new_id("mem"), min_length=1, max_length=256)
    kind: MemoryKind
    summary: str = Field(min_length=1, max_length=8_000)
    payload: dict[str, Any] = Field(default_factory=dict)
    status: MemoryStatus = MemoryStatus.ACTIVE
    temperature: MemoryTemperature = MemoryTemperature.WARM
    pinned: bool = False
    source_ids: set[str] = Field(default_factory=set)
    dependencies: set[str] = Field(default_factory=set)
    metadata: dict[str, Any] = Field(default_factory=dict)
    valid_from: datetime | None = None
    expires_at: datetime | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def align_pin_and_temperature(self) -> "MemoryDraft":
        if self.pinned:
            object.__setattr__(self, "temperature", MemoryTemperature.PINNED)
            object.__setattr__(self, "status", MemoryStatus.ACTIVE)
        if self.temperature is MemoryTemperature.PINNED:
            object.__setattr__(self, "pinned", True)
            object.__setattr__(self, "status", MemoryStatus.ACTIVE)
        if self.expires_at and self.valid_from and self.expires_at < self.valid_from:
            raise ValueError("expires_at must be on or after valid_from")
        return self


class MemoryItem(MemoryDraft):
    run_id: str = Field(min_length=1, max_length=256)
    revision_created: int = Field(ge=0)
    revision_updated: int = Field(ge=0)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    superseded_by: str | None = None


class MemoryMutation(StrictModel):
    operation: MemoryOperationType
    target_id: str | None = None
    draft: MemoryDraft | None = None
    changes: dict[str, Any] = Field(default_factory=dict)
    replacement_ids: set[str] = Field(default_factory=set)
    reason_code: ReasonCode = ReasonCode.OTHER
    note: str | None = Field(default=None, max_length=2_000)
    purge_confirmation_token: str | None = Field(default=None, max_length=512)

    @model_validator(mode="after")
    def validate_shape(self) -> "MemoryMutation":
        if self.operation is MemoryOperationType.ADD:
            if self.draft is None or self.target_id is not None:
                raise ValueError("ADD requires draft and forbids target_id")
        else:
            if not self.target_id:
                raise ValueError(f"{self.operation.value} requires target_id")
            if self.draft is not None:
                raise ValueError(f"{self.operation.value} forbids draft")
        if self.operation in {MemoryOperationType.UPDATE, MemoryOperationType.MERGE}:
            if not self.changes:
                raise ValueError(f"{self.operation.value} requires changes")
        if self.operation is MemoryOperationType.SUPERSEDE and not self.replacement_ids:
            raise ValueError("SUPERSEDE requires at least one replacement_id")
        return self


class StatePatch(StrictModel):
    id: str = Field(default_factory=lambda: new_id("patch"))
    run_id: str
    expected_revision: int = Field(ge=0)
    actor: ActorType = ActorType.MODEL
    mutations: list[MemoryMutation] = Field(min_length=1, max_length=128)
    idempotency_key: str = Field(min_length=8, max_length=256)
    allow_purge: bool = False
    reason: str = Field(default="state update", min_length=1, max_length=4_000)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("idempotency_key")
    @classmethod
    def no_whitespace_key(cls, value: str) -> str:
        if value.strip() != value or any(ch.isspace() for ch in value):
            raise ValueError("idempotency_key must not contain whitespace")
        return value


class PatchResult(StrictModel):
    patch_id: str
    run_id: str
    previous_revision: int
    new_revision: int
    committed: bool
    idempotent_replay: bool = False
    changed_memory_ids: list[str] = Field(default_factory=list)
    removed_from_active_context: list[str] = Field(default_factory=list)
    archived_memory_ids: list[str] = Field(default_factory=list)
    purged_memory_ids: list[str] = Field(default_factory=list)
    prompt_rebuild_required: bool = False
    state_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    event_id: str
    warnings: list[str] = Field(default_factory=list)


class DeleteMemoryRequest(StrictModel):
    run_id: str
    memory_ids: list[str] = Field(min_length=1, max_length=128)
    scope: DeleteScope = DeleteScope.ACTIVE_CONTEXT
    reason_code: ReasonCode
    reason: str | None = Field(default=None, max_length=2_000)
    replacement_state_ids: list[str] = Field(default_factory=list)
    expected_state_revision: int = Field(ge=0)
    idempotency_key: str | None = Field(default=None, min_length=8, max_length=256)
    confirmation_token: str | None = Field(default=None, max_length=512)


class RestoreMemoryRequest(StrictModel):
    memory_ids: list[str] = Field(min_length=1, max_length=128)
    expected_state_revision: int = Field(ge=0)
    idempotency_key: str | None = Field(default=None, min_length=8, max_length=256)


class DeleteMemoryResult(StrictModel):
    status: Literal["committed", "replayed"]
    run_id: str
    new_state_revision: int
    removed_from_active_context: list[str] = Field(default_factory=list)
    archived: list[str] = Field(default_factory=list)
    purged: list[str] = Field(default_factory=list)
    prompt_rebuild_required: bool
    state_hash: str
    event_id: str


class EvidenceRecord(StrictModel):
    id: str = Field(default_factory=lambda: new_id("ev"))
    run_id: str
    source_uri: str = Field(min_length=1, max_length=8_000)
    title: str = Field(default="Untitled source", max_length=2_000)
    publisher: str | None = Field(default=None, max_length=512)
    query: str | None = Field(default=None, max_length=4_000)
    passage: str = Field(min_length=1, max_length=100_000)
    content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    rank: int | None = Field(default=None, ge=1)
    suspicious_content: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
    retrieved_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def validate_content_hash(self) -> "EvidenceRecord":
        observed = hashlib.sha256(self.passage.encode("utf-8")).hexdigest()
        if observed != self.content_hash:
            raise ValueError("content_hash must be the SHA-256 of passage UTF-8 bytes")
        return self


class ClaimRecord(StrictModel):
    id: str = Field(default_factory=lambda: new_id("claim"))
    run_id: str
    statement: str = Field(min_length=1, max_length=8_000)
    evidence_ids: set[str] = Field(default_factory=set)
    support_status: SupportStatus = SupportStatus.UNSUPPORTED
    required_for_answer: bool = False
    freshness_required: bool = False
    conflict_group: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class RunSnapshot(StrictModel):
    run: RunRecord
    memories: list[MemoryItem] = Field(default_factory=list)
    evidence: list[EvidenceRecord] = Field(default_factory=list)
    claims: list[ClaimRecord] = Field(default_factory=list)


class StateEvent(StrictModel):
    id: str = Field(default_factory=lambda: new_id("evt"))
    run_id: str
    revision: int = Field(ge=0)
    event_type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    trace_id: str | None = None
    span_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class TraceEvent(StrictModel):
    id: str = Field(default_factory=lambda: new_id("trace"))
    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    run_id: str
    event_type: TraceEventType
    node: WorkflowNode | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)


class CheckpointRecord(StrictModel):
    id: str = Field(default_factory=lambda: new_id("chk"))
    run_id: str
    source_revision: int = Field(ge=0)
    label: str = Field(min_length=1, max_length=512)
    state_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ContextBudget(StrictModel):
    total_tokens: int = Field(default=2048, ge=128)
    static_tokens: int = Field(default=240, ge=0)
    goal_tokens: int = Field(default=256, ge=0)
    state_tokens: int = Field(default=512, ge=0)
    evidence_tokens: int = Field(default=704, ge=0)
    recent_tokens: int = Field(default=208, ge=0)
    reserve_output_tokens: int = Field(default=128, ge=1)

    @model_validator(mode="after")
    def validate_sub_budgets(self) -> "ContextBudget":
        allocated = (
            self.static_tokens
            + self.goal_tokens
            + self.state_tokens
            + self.evidence_tokens
            + self.recent_tokens
            + self.reserve_output_tokens
        )
        if allocated > self.total_tokens:
            raise ValueError(
                f"context sub-budgets ({allocated}) exceed total_tokens ({self.total_tokens})"
            )
        return self


class ContextSection(StrictModel):
    name: str
    memory_ids: list[str] = Field(default_factory=list)
    text: str = ""
    estimated_tokens: int = 0


class ContextBundle(StrictModel):
    run_id: str
    revision: int
    messages: list[dict[str, str]]
    sections: list[ContextSection]
    included_memory_ids: list[str]
    excluded_memory_ids: list[str]
    estimated_input_tokens: int
    budget: ContextBudget
    context_hash: str = Field(pattern=r"^[a-f0-9]{64}$")


class SearchResult(StrictModel):
    id: str = Field(default_factory=lambda: new_id("sr"))
    title: str
    url: HttpUrl
    snippet: str = ""
    rank: int = Field(ge=1)
    provider: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class FetchedDocument(StrictModel):
    id: str = Field(default_factory=lambda: new_id("src"))
    requested_url: HttpUrl
    final_url: HttpUrl
    title: str | None = None
    text: str
    content_type: str
    fetched_at: datetime = Field(default_factory=utc_now)
    content_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    status_code: int = Field(ge=100, le=599)
    byte_count: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidencePassage(StrictModel):
    id: str
    source_id: str
    text: str = Field(min_length=1)
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=0)
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_offsets(self) -> "EvidencePassage":
        if self.end_offset < self.start_offset:
            raise ValueError("end_offset must not precede start_offset")
        return self


class AnswerabilityReport(StrictModel):
    answerable: bool
    action: FinishAction
    reason_codes: list[str] = Field(default_factory=list)
    unsupported_claim_ids: list[str] = Field(default_factory=list)
    conflicted_claim_ids: list[str] = Field(default_factory=list)
    stale_claim_ids: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    evidence_source_count: int = 0
    independent_domain_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class RouteDecision(StrictModel):
    action: RouteAction
    reason: str
    query: str | None = None
    clarification: str | None = None


class WorkflowBudget(StrictModel):
    max_rounds: int = Field(default=3, ge=1, le=20)
    max_model_calls: int = Field(default=8, ge=1, le=100)
    max_search_calls: int = Field(default=3, ge=0, le=20)
    max_sources_fetched: int = Field(default=8, ge=0, le=100)
    max_wall_seconds: float = Field(default=90.0, gt=0, le=3600)
    max_active_memory_items: int = Field(default=64, ge=4, le=10_000)


class WorkflowCounters(StrictModel):
    rounds: int = 0
    model_calls: int = 0
    search_calls: int = 0
    sources_fetched: int = 0


class ConversationTurn(StrictModel):
    """A bounded historical chat turn preserved with its original trust role."""

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=20_000)


class ModelMessage(StrictModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None


class ModelRequest(StrictModel):
    messages: list[ModelMessage]
    max_output_tokens: int = Field(default=128, ge=1, le=4096)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    response_schema: dict[str, Any] | None = None
    stop: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ModelResponse(StrictModel):
    text: str
    raw: dict[str, Any] = Field(default_factory=dict)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    finish_reason: str | None = None
    latency_ms: float | None = None


class HarnessRunRequest(StrictModel):
    question: str = Field(min_length=1, max_length=20_000)
    mode: RunMode = RunMode.AUTO
    run_id: str | None = None
    recent_turns: list[ConversationTurn] = Field(default_factory=list, max_length=64)
    budget: WorkflowBudget = Field(default_factory=WorkflowBudget)
    context_budget: ContextBudget = Field(default_factory=ContextBudget)
    metadata: dict[str, Any] = Field(default_factory=dict)


class HarnessAnswer(StrictModel):
    run_id: str
    status: RunStatus
    action: FinishAction
    answer: str
    citations: list[str] = Field(default_factory=list)
    state_revision: int = Field(ge=0)
    trace_id: str
    answerability: AnswerabilityReport | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCall(StrictModel):
    id: str = Field(default_factory=lambda: new_id("call"))
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str | None = None
    requested_at: datetime = Field(default_factory=utc_now)


class ToolResult(StrictModel):
    call_id: str
    name: str
    ok: bool
    output: dict[str, Any] = Field(default_factory=dict)
    error_code: str | None = None
    error_message: str | None = None
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime = Field(default_factory=utc_now)


class FinishRequest(StrictModel):
    action: FinishAction
    answer: str = ""
    citations: list[str] = Field(default_factory=list)
    reason: str | None = None
