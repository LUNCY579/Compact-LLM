from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


@dataclass
class GenerationRequest:
    request_id: str
    messages: list[dict[str, str]]
    reference_output: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResponse:
    request_id: str
    text: str
    latency_ms: float
    backend: str
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class InferenceBackend(Protocol):
    name: str

    def generate(self, requests: list[GenerationRequest]) -> list[GenerationResponse]: ...

    def close(self) -> None: ...
