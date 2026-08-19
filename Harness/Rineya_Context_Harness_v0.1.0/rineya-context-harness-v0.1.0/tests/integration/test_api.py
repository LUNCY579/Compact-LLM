from __future__ import annotations

from fastapi.testclient import TestClient

from rineya_harness.api.app import create_app


def test_health_endpoint(container) -> None:
    client = TestClient(create_app(container=container))
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
