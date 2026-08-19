from __future__ import annotations

import time
from dataclasses import dataclass
from urllib.parse import urljoin

import httpx

from rineya_harness.domain.models import ModelRequest, ModelResponse


@dataclass(slots=True)
class OllamaChatModel:
    model: str
    base_url: str = "http://127.0.0.1:11434"
    timeout_seconds: float = 60.0
    name: str = "ollama"

    async def complete(self, request: ModelRequest) -> ModelResponse:
        endpoint = urljoin(self.base_url.rstrip("/") + "/", "api/chat")
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [message.model_dump(exclude_none=True) for message in request.messages],
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_output_tokens,
            },
        }
        if request.response_schema is not None:
            payload["format"] = request.response_schema
        started = time.perf_counter()
        async with httpx.AsyncClient(trust_env=False, timeout=httpx.Timeout(self.timeout_seconds)) as client:
            response = await client.post(endpoint, json=payload)
            response.raise_for_status()
            raw = response.json()
        return ModelResponse(
            text=str(raw.get("message", {}).get("content") or ""),
            raw=raw,
            prompt_tokens=raw.get("prompt_eval_count"),
            completion_tokens=raw.get("eval_count"),
            finish_reason=raw.get("done_reason"),
            latency_ms=(time.perf_counter() - started) * 1000,
        )
