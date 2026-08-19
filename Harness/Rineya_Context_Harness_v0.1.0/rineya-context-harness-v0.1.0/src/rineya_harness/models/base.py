from __future__ import annotations

from typing import Protocol

from rineya_harness.domain.models import ModelRequest, ModelResponse


class ChatModel(Protocol):
    name: str

    async def complete(self, request: ModelRequest) -> ModelResponse: ...
