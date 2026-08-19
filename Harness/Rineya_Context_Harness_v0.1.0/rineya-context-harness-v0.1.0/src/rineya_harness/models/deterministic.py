from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from rineya_harness.domain.models import ModelRequest, ModelResponse


@dataclass(slots=True)
class DeterministicChatModel:
    """Offline model fixture with a handler or FIFO response queue."""

    responses: list[str] = field(default_factory=list)
    handler: Callable[[ModelRequest], str] | None = None
    name: str = "deterministic"

    async def complete(self, request: ModelRequest) -> ModelResponse:
        if self.handler is not None:
            text = self.handler(request)
        elif self.responses:
            text = self.responses.pop(0)
        else:
            text = "I don't know based on the available information."
        return ModelResponse(text=text, finish_reason="stop", latency_ms=0.0)
