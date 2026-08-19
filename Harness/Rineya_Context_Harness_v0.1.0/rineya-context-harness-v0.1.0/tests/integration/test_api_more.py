from __future__ import annotations

from fastapi.testclient import TestClient

from rineya_harness.api.app import create_app
from rineya_harness.domain.models import RunRecord


def test_api_get_unknown_run_returns_404(container) -> None:
    client = TestClient(create_app(container=container))
    assert client.get("/v1/runs/missing").status_code == 404


def test_api_checkpoint_and_export(container) -> None:
    run = container.store.create_run(RunRecord(goal="api"))
    client = TestClient(create_app(container=container))
    checkpoint = client.post(f"/v1/runs/{run.id}/checkpoints", params={"label": "x"})
    assert checkpoint.status_code == 200
    exported = client.get(f"/v1/runs/{run.id}/export")
    assert exported.status_code == 200
    assert "export_sha256" in exported.json()


def test_api_exposes_tools_and_run_indexes(container) -> None:
    run = container.store.create_run(RunRecord(goal="index"))
    client = TestClient(create_app(container=container))
    tools = client.get("/v1/tools")
    assert tools.status_code == 200
    assert {item["name"] for item in tools.json()} == {
        "get_state",
        "search_web",
        "inspect_evidence",
        "update_memory",
        "delete_memory",
        "finish",
    }
    runs = client.get("/v1/runs")
    assert runs.status_code == 200
    assert runs.json()[0]["id"] == run.id


def test_api_restore_memory_and_list_checkpoints(container) -> None:
    from rineya_harness.domain.enums import MemoryKind
    from rineya_harness.domain.models import MemoryDraft

    run = container.store.create_run(RunRecord(goal="restore"))
    container.memory.add(
        run_id=run.id,
        draft=MemoryDraft(id="temporary", kind=MemoryKind.EVENT_HISTORY, summary="old"),
        expected_revision=0,
    )
    client = TestClient(create_app(container=container))
    deleted = client.post(
        f"/v1/runs/{run.id}/memory/delete",
        json={
            "run_id": run.id,
            "memory_ids": ["temporary"],
            "scope": "active_context",
            "reason_code": "completed_no_future_dependency",
            "expected_state_revision": 1,
        },
    )
    assert deleted.status_code == 200
    restored = client.post(
        f"/v1/runs/{run.id}/memory/restore",
        json={"memory_ids": ["temporary"], "expected_state_revision": 2},
    )
    assert restored.status_code == 200
    memory = client.get(f"/v1/runs/{run.id}/memory")
    assert memory.json()[0]["status"] == "active"
    created = client.post(f"/v1/runs/{run.id}/checkpoints", params={"label": "after"})
    assert created.status_code == 200
    listed = client.get(f"/v1/runs/{run.id}/checkpoints")
    assert listed.status_code == 200
    assert listed.json()[0]["label"] == "after"


def test_api_does_not_expose_persistent_purge(container) -> None:
    from rineya_harness.domain.enums import MemoryKind
    from rineya_harness.domain.models import MemoryDraft

    run = container.store.create_run(RunRecord(goal="purge"))
    container.memory.add(
        run_id=run.id,
        draft=MemoryDraft(id="history", kind=MemoryKind.EVENT_HISTORY, summary="old"),
        expected_revision=0,
    )
    client = TestClient(create_app(container=container))
    response = client.post(
        f"/v1/runs/{run.id}/memory/delete",
        json={
            "run_id": run.id,
            "memory_ids": ["history"],
            "scope": "persistent_purge",
            "reason_code": "user_request",
            "expected_state_revision": 1,
        },
    )
    assert response.status_code == 403
