from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from rineya_harness.domain.enums import (
    ActorType,
    DeleteScope,
    ReasonCode,
    ToolRisk,
)
from rineya_harness.domain.models import DeleteMemoryRequest, StrictModel
from rineya_harness.memory.service import MemoryService
from rineya_harness.state.store import SQLiteStateStore

from .base import ToolDefinition

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


class GetStateArgs(StrictModel):
    run_id: str = Field(min_length=1, max_length=256)
    include_inactive: bool = False


class UpdateMemoryArgs(StrictModel):
    run_id: str = Field(min_length=1, max_length=256)
    memory_id: str = Field(min_length=1, max_length=256)
    changes: dict[str, Any] = Field(min_length=1)
    expected_revision: int = Field(ge=0)
    merge: bool = False
    idempotency_key: str | None = Field(default=None, min_length=8, max_length=256)

    @model_validator(mode="after")
    def restrict_model_mutations(self) -> "UpdateMemoryArgs":
        unsupported = sorted(set(self.changes) - _MODEL_MUTABLE_MEMORY_FIELDS)
        if unsupported:
            raise ValueError(f"model memory update contains forbidden fields: {unsupported}")
        if self.merge and "payload" not in self.changes:
            raise ValueError("merge=true requires a payload change")
        return self


class ModelDeleteMemoryArgs(StrictModel):
    """Model-visible reversible deletion contract.

    Persistent purge is deliberately absent from this schema and remains a separate admin-only
    operation outside the six-tool small-model surface.
    """

    run_id: str = Field(min_length=1, max_length=256)
    memory_ids: list[str] = Field(min_length=1, max_length=128)
    scope: Literal[DeleteScope.ACTIVE_CONTEXT, DeleteScope.ARCHIVE] = (
        DeleteScope.ACTIVE_CONTEXT
    )
    reason_code: ReasonCode
    reason: str | None = Field(default=None, max_length=2_000)
    replacement_state_ids: list[str] = Field(default_factory=list, max_length=128)
    expected_state_revision: int = Field(ge=0)
    idempotency_key: str | None = Field(default=None, min_length=8, max_length=256)


class MemoryToolSet:
    def __init__(self, store: SQLiteStateStore, service: MemoryService) -> None:
        self.store = store
        self.service = service

    async def get_state(self, args: GetStateArgs) -> dict[str, Any]:
        run = self.store.get_run(args.run_id)
        memories = self.store.list_memory(args.run_id, include_inactive=args.include_inactive)
        return {
            "run": run.model_dump(mode="json"),
            "memories": [item.model_dump(mode="json") for item in memories],
        }

    async def update_memory(self, args: UpdateMemoryArgs) -> dict[str, Any]:
        result = self.service.update(
            run_id=args.run_id,
            memory_id=args.memory_id,
            changes=args.changes,
            expected_revision=args.expected_revision,
            merge=args.merge,
            actor=ActorType.MODEL,
            idempotency_key=args.idempotency_key,
        )
        return result.model_dump(mode="json")

    async def delete_memory(self, args: ModelDeleteMemoryArgs) -> dict[str, Any]:
        request = DeleteMemoryRequest(
            run_id=args.run_id,
            memory_ids=args.memory_ids,
            scope=args.scope,
            reason_code=args.reason_code,
            reason=args.reason,
            replacement_state_ids=args.replacement_state_ids,
            expected_state_revision=args.expected_state_revision,
            idempotency_key=args.idempotency_key,
        )
        result = self.service.delete_memory(request, actor=ActorType.MODEL)
        return result.model_dump(mode="json")

    def definitions(self) -> list[ToolDefinition[Any]]:
        return [
            ToolDefinition(
                name="get_state",
                description="Read the compact, versioned current state for one run.",
                args_model=GetStateArgs,
                risk=ToolRisk.READ_ONLY,
                handler=self.get_state,
            ),
            ToolDefinition(
                name="update_memory",
                description=(
                    "Update compact semantic content on an existing state record. Lifecycle, pin, "
                    "kind, and provenance metadata fields are system-controlled."
                ),
                args_model=UpdateMemoryArgs,
                risk=ToolRisk.MUTATING_REVERSIBLE,
                handler=self.update_memory,
            ),
            ToolDefinition(
                name="delete_memory",
                description=(
                    "Evict obsolete memory from active context or archive it. Both operations are "
                    "reversible; permanent purge is not available to the model."
                ),
                args_model=ModelDeleteMemoryArgs,
                risk=ToolRisk.MUTATING_REVERSIBLE,
                handler=self.delete_memory,
            ),
        ]
