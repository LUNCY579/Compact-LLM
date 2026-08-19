from __future__ import annotations

import hashlib
from pathlib import Path

from rineya_harness.config import DatabaseConfig, HarnessConfig, TraceConfig
from rineya_harness.domain.enums import (
    ActorType,
    DeleteScope,
    MemoryKind,
    MemoryOperationType,
    ReasonCode,
    RunMode,
)
from rineya_harness.domain.models import (
    DeleteMemoryRequest,
    FetchedDocument,
    HarnessRunRequest,
    MemoryDraft,
    MemoryMutation,
    RunRecord,
    SearchResult,
    StatePatch,
)
from rineya_harness.factory import HarnessContainer, create_container
from rineya_harness.search.fixture import FixtureSearchProvider
from rineya_harness.web.fixture import FixtureDocumentFetcher


def fixture_container(workdir: Path) -> HarnessContainer:
    url = "https://example.org/rineya-harness"
    text = (
        "The Rineya Context Harness keeps immutable source evidence outside the active prompt. "
        "It reconstructs compact current state and evicts obsolete model-visible memory after "
        "dependency checks. Persistent purge is disabled by default."
    )
    document = FetchedDocument(
        requested_url=url,
        final_url=url,
        title="Rineya Context Harness fixture",
        text=text,
        content_type="text/plain",
        content_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        status_code=200,
        byte_count=len(text.encode("utf-8")),
    )
    search = FixtureSearchProvider(
        fixtures={
            "How does the Rineya Context Harness forget obsolete memory?": [
                SearchResult(
                    title=document.title or "Fixture",
                    url=url,
                    snippet=text,
                    rank=1,
                    provider="fixture",
                )
            ]
        }
    )
    fetcher = FixtureDocumentFetcher(documents={url: document})
    config = HarnessConfig(
        database=DatabaseConfig(path=workdir / "fixture.sqlite3"),
        trace=TraceConfig(directory=workdir / "traces"),
    )
    return create_container(config, fixture_search=search, fixture_fetcher=fetcher)


def build_restaurant_demo(container: HarnessContainer) -> dict[str, object]:
    run = container.store.create_run(
        RunRecord(goal="Walk to Restaurant R", mode=RunMode.DIRECT)
    )
    initial = [
        MemoryDraft(
            id="route_goal",
            kind=MemoryKind.GLOBAL_INVARIANT,
            summary="Reach Restaurant R on foot.",
            payload={"destination": "Restaurant R", "mode": "walking"},
            pinned=True,
        ),
        MemoryDraft(
            id="route_eta_20",
            kind=MemoryKind.CURRENT_STATE,
            summary="ETA is 20 minutes.",
            payload={"eta_minutes": 20},
        ),
        MemoryDraft(
            id="route_light_a",
            kind=MemoryKind.CURRENT_STATE,
            summary="Traffic light A remains.",
            payload={"checkpoint": "A"},
        ),
        MemoryDraft(
            id="route_light_b",
            kind=MemoryKind.CURRENT_STATE,
            summary="Traffic light B remains.",
            payload={"checkpoint": "B"},
        ),
        MemoryDraft(
            id="route_light_c",
            kind=MemoryKind.CURRENT_STATE,
            summary="Traffic light C remains.",
            payload={"checkpoint": "C"},
        ),
    ]
    container.store.apply_patch(
        StatePatch(
            run_id=run.id,
            expected_revision=0,
            actor=ActorType.SYSTEM,
            mutations=[MemoryMutation(operation=MemoryOperationType.ADD, draft=item) for item in initial],
            idempotency_key=f"restaurant-initialize-{run.id}",
            reason="initialize route state",
        )
    )
    replacement = MemoryDraft(
        id="route_eta_15",
        kind=MemoryKind.CURRENT_STATE,
        summary="ETA is 15 minutes.",
        payload={"eta_minutes": 15, "remaining_lights": ["B", "C"], "next_light": "B"},
    )
    current = container.store.get_run(run.id)
    result = container.store.apply_patch(
        StatePatch(
            run_id=run.id,
            expected_revision=current.revision,
            actor=ActorType.SYSTEM,
            mutations=[
                MemoryMutation(operation=MemoryOperationType.ADD, draft=replacement),
                MemoryMutation(
                    operation=MemoryOperationType.SUPERSEDE,
                    target_id="route_eta_20",
                    replacement_ids={replacement.id},
                    reason_code=ReasonCode.SUPERSEDED_BY_CURRENT_STATE,
                ),
                MemoryMutation(
                    operation=MemoryOperationType.EVICT,
                    target_id="route_light_a",
                    replacement_ids={replacement.id, "route_light_b", "route_light_c"},
                    reason_code=ReasonCode.COMPLETED_NO_FUTURE_DEPENDENCY,
                    note="light A was passed",
                ),
            ],
            idempotency_key=f"restaurant-progress-{run.id}",
            reason="replace route state and forget completed checkpoint",
        )
    )
    active = container.store.list_memory(run.id)
    return {
        "run_id": run.id,
        "revision": result.new_revision,
        "active_memory_ids": [item.id for item in active],
        "active_state": [item.model_dump(mode="json") for item in active],
        "inactive_state": [
            item.model_dump(mode="json")
            for item in container.store.list_memory(run.id, include_inactive=True)
            if item.id not in {active_item.id for active_item in active}
        ],
        "prompt_rebuild_required": result.prompt_rebuild_required,
    }


def build_hanoi_demo(container: HarnessContainer) -> dict[str, object]:
    run = container.store.create_run(
        RunRecord(goal="Move three disks from A to C", mode=RunMode.DIRECT)
    )
    initial = [
        MemoryDraft(
            id="hanoi_rules",
            kind=MemoryKind.GLOBAL_INVARIANT,
            summary="Three ordered disks; never place a larger disk on a smaller disk; target peg C.",
            payload={"disk_count": 3, "peg_names": ["A", "B", "C"], "target": "C"},
            pinned=True,
        ),
        MemoryDraft(
            id="hanoi_state_0",
            kind=MemoryKind.CURRENT_STATE,
            summary="A=[3,2,1], B=[], C=[].",
            payload={"A": [3, 2, 1], "B": [], "C": []},
        ),
    ]
    container.store.apply_patch(
        StatePatch(
            run_id=run.id,
            expected_revision=0,
            actor=ActorType.SYSTEM,
            mutations=[MemoryMutation(operation=MemoryOperationType.ADD, draft=item) for item in initial],
            idempotency_key=f"hanoi-initialize-{run.id}",
            reason="initialize hanoi state",
        )
    )
    next_state = MemoryDraft(
        id="hanoi_state_1",
        kind=MemoryKind.CURRENT_STATE,
        summary="A=[3,2], B=[], C=[1].",
        payload={"A": [3, 2], "B": [], "C": [1], "last_move": "1:A->C"},
    )
    current = container.store.get_run(run.id)
    result = container.store.apply_patch(
        StatePatch(
            run_id=run.id,
            expected_revision=current.revision,
            actor=ActorType.SYSTEM,
            mutations=[
                MemoryMutation(operation=MemoryOperationType.ADD, draft=next_state),
                MemoryMutation(
                    operation=MemoryOperationType.SUPERSEDE,
                    target_id="hanoi_state_0",
                    replacement_ids={next_state.id},
                    reason_code=ReasonCode.SUPERSEDED_BY_CURRENT_STATE,
                ),
            ],
            idempotency_key=f"hanoi-move-1-{run.id}",
            reason="replace current board after one legal move",
        )
    )
    return {
        "run_id": run.id,
        "revision": result.new_revision,
        "active_memory": [
            item.model_dump(mode="json") for item in container.store.list_memory(run.id)
        ],
        "event_count": len(container.store.list_events(run.id)),
    }


async def run_fixture_research(container: HarnessContainer) -> dict[str, object]:
    answer = await container.workflow.run(
        HarnessRunRequest(
            question="How does the Rineya Context Harness forget obsolete memory?",
            mode=RunMode.GROUNDED,
        )
    )
    return answer.model_dump(mode="json")
