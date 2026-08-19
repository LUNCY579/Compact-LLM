from __future__ import annotations

import pytest

from rineya_harness.domain.models import ModelMessage, ModelRequest
from rineya_harness.models.deterministic import DeterministicChatModel


@pytest.mark.asyncio
async def test_deterministic_model_queue() -> None:
    model = DeterministicChatModel(responses=["first"])
    response = await model.complete(ModelRequest(messages=[ModelMessage(role="user", content="x")]))
    assert response.text == "first"


@pytest.mark.asyncio
async def test_deterministic_model_handler() -> None:
    model = DeterministicChatModel(handler=lambda request: request.messages[-1].content.upper())
    response = await model.complete(ModelRequest(messages=[ModelMessage(role="user", content="hello")]))
    assert response.text == "HELLO"


@pytest.mark.asyncio
async def test_deterministic_model_fallback() -> None:
    model = DeterministicChatModel()
    response = await model.complete(ModelRequest(messages=[ModelMessage(role="user", content="x")]))
    assert response.text.startswith("I don't know")
