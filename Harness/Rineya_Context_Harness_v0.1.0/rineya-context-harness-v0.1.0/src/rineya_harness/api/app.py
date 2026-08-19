from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from rineya_harness import __version__
from rineya_harness.config import HarnessConfig, load_config
from rineya_harness.domain.enums import ActorType, DeleteScope
from rineya_harness.domain.models import (
    CheckpointRecord,
    DeleteMemoryRequest,
    DeleteMemoryResult,
    HarnessAnswer,
    HarnessRunRequest,
    PatchResult,
    RestoreMemoryRequest,
    RunSnapshot,
)
from rineya_harness.errors import (
    ConflictError,
    HarnessError,
    NotFoundError,
    RevisionConflictError,
    StateInvariantError,
)
from rineya_harness.factory import HarnessContainer, create_container


def create_app(
    *,
    config: HarnessConfig | None = None,
    container: HarnessContainer | None = None,
) -> FastAPI:
    runtime = container or create_container(config or load_config())
    application = FastAPI(
        title="Rineya Context Harness API",
        version=__version__,
        description="Local-first bounded context, evidence, and recursive-state-forgetting runtime.",
    )
    application.state.container = runtime

    @application.exception_handler(NotFoundError)
    async def not_found_handler(_request: Any, exc: NotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"error": type(exc).__name__, "message": str(exc)})

    @application.exception_handler(RevisionConflictError)
    async def revision_handler(_request: Any, exc: RevisionConflictError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"error": type(exc).__name__, "message": str(exc)})

    @application.exception_handler(ConflictError)
    async def conflict_handler(_request: Any, exc: ConflictError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"error": type(exc).__name__, "message": str(exc)})

    @application.exception_handler(StateInvariantError)
    async def invariant_handler(_request: Any, exc: StateInvariantError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"error": type(exc).__name__, "message": str(exc), "reasons": exc.reasons},
        )

    @application.exception_handler(HarnessError)
    async def harness_handler(_request: Any, exc: HarnessError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"error": type(exc).__name__, "message": str(exc)})

    @application.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "model_backend": runtime.model.name,
            "search_provider": runtime.search_provider.name,
        }

    @application.get("/v1/tools")
    async def list_tools() -> list[dict[str, Any]]:
        return runtime.tools.definitions()

    @application.get("/v1/runs")
    async def list_runs(limit: int = 100) -> list[dict[str, Any]]:
        if not 1 <= limit <= 1_000:
            raise HTTPException(status_code=422, detail="limit must be between 1 and 1000")
        return [item.model_dump(mode="json") for item in runtime.store.list_runs(limit=limit)]

    @application.post("/v1/runs", response_model=HarnessAnswer)
    async def run_harness(request: HarnessRunRequest) -> HarnessAnswer:
        return await runtime.workflow.run(request)

    @application.get("/v1/runs/{run_id}", response_model=RunSnapshot)
    async def get_run(run_id: str) -> RunSnapshot:
        return runtime.store.get_snapshot(run_id)

    @application.get("/v1/runs/{run_id}/memory")
    async def list_memory(run_id: str, include_inactive: bool = False) -> list[dict[str, Any]]:
        return [
            item.model_dump(mode="json")
            for item in runtime.store.list_memory(run_id, include_inactive=include_inactive)
        ]

    @application.post("/v1/runs/{run_id}/memory/delete", response_model=DeleteMemoryResult)
    async def delete_memory(run_id: str, request: DeleteMemoryRequest) -> DeleteMemoryResult:
        if request.run_id != run_id:
            raise HTTPException(status_code=422, detail="path run_id must match body run_id")
        if request.scope is DeleteScope.PERSISTENT_PURGE:
            raise HTTPException(
                status_code=403,
                detail="persistent purge is not exposed by the local user API; use an approved admin path",
            )
        return runtime.memory.delete_memory(request, actor=ActorType.USER)

    @application.post("/v1/runs/{run_id}/memory/restore", response_model=PatchResult)
    async def restore_memory(run_id: str, request: RestoreMemoryRequest) -> PatchResult:
        return runtime.memory.restore(
            run_id=run_id,
            memory_ids=request.memory_ids,
            expected_revision=request.expected_state_revision,
            actor=ActorType.USER,
            idempotency_key=request.idempotency_key,
        )

    @application.get("/v1/runs/{run_id}/checkpoints", response_model=list[CheckpointRecord])
    async def list_checkpoints(run_id: str) -> list[CheckpointRecord]:
        return runtime.store.list_checkpoints(run_id)

    @application.post("/v1/runs/{run_id}/checkpoints", response_model=CheckpointRecord)
    async def checkpoint(run_id: str, label: str = "api") -> CheckpointRecord:
        return runtime.store.create_checkpoint(run_id, label=label)

    @application.post("/v1/runs/{run_id}/checkpoints/{checkpoint_id}/restore")
    async def restore_checkpoint(run_id: str, checkpoint_id: str, expected_revision: int) -> dict[str, Any]:
        return runtime.store.restore_checkpoint(
            run_id, checkpoint_id, expected_revision=expected_revision
        ).model_dump(mode="json")

    @application.get("/v1/runs/{run_id}/trace")
    async def get_trace(run_id: str) -> list[dict[str, Any]]:
        runtime.store.get_run(run_id)
        return [event.model_dump(mode="json") for event in runtime.traces.read(run_id)]

    @application.get("/v1/runs/{run_id}/export")
    async def export_run(run_id: str) -> dict[str, Any]:
        return runtime.store.export_run(run_id)

    return application

