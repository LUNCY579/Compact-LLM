from __future__ import annotations

import time
from dataclasses import dataclass, field
from urllib.parse import urljoin

import httpx

from rineya_harness.domain.models import ModelRequest, ModelResponse


@dataclass(slots=True)
class OpenAICompatibleChatModel:
    model: str
    base_url: str
    api_key: str | None = None
    timeout_seconds: float = 60.0
    enable_thinking: bool = False
    extra_body: dict[str, object] = field(default_factory=dict)
    name: str = "openai_compatible"

    async def complete(self, request: ModelRequest) -> ModelResponse:
        endpoint = urljoin(self.base_url.rstrip("/") + "/", "chat/completions")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [message.model_dump(exclude_none=True) for message in request.messages],
            "max_tokens": request.max_output_tokens,
            "temperature": request.temperature,
            "stream": False,
            "stop": request.stop or None,
            **self.extra_body,
        }
        # llama.cpp and some other local servers forward this to a Qwen chat template.
        payload.setdefault("chat_template_kwargs", {"enable_thinking": self.enable_thinking})
        if request.response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "rineya_response",
                    "strict": True,
                    "schema": request.response_schema,
                },
            }
        started = time.perf_counter()
        async with httpx.AsyncClient(trust_env=False, timeout=httpx.Timeout(self.timeout_seconds)) as client:
            response = await client.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
            raw = response.json()
        latency_ms = (time.perf_counter() - started) * 1000
        choice = raw["choices"][0]
        usage = raw.get("usage", {})
        return ModelResponse(
            text=str(choice["message"].get("content") or ""),
            raw=raw,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            finish_reason=choice.get("finish_reason"),
            latency_ms=latency_ms,
        )
