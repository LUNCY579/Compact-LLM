from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from typing import Any

from .base import GenerationRequest, GenerationResponse


class OpenAICompatibleBackend:
    name = "openai_compatible"

    def __init__(self, config: dict[str, Any]) -> None:
        endpoint = str(config.get("endpoint", "http://127.0.0.1:8000/v1/chat/completions"))
        self.endpoint = endpoint.rstrip("/")
        self.model = str(config["model"])
        self.api_key = os.environ.get(str(config.get("api_key_env", "OPENAI_API_KEY")), "")
        self.timeout = float(config.get("timeout_seconds", 120))
        self.retries = int(config.get("retries", 3))
        self.generation = config.get("generation", {}) or {}

    def _one(self, request: GenerationRequest) -> GenerationResponse:
        payload = {
            "model": self.model,
            "messages": request.messages,
            "max_tokens": int(self.generation.get("max_new_tokens", 256)),
            "temperature": float(self.generation.get("temperature", 0.0)),
            "top_p": float(self.generation.get("top_p", 1.0)),
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        start = time.perf_counter()
        last_error: str | None = None
        for attempt in range(self.retries + 1):
            try:
                http_request = urllib.request.Request(
                    self.endpoint,
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers,
                    method="POST",
                )
                with urllib.request.urlopen(http_request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                text = body["choices"][0]["message"]["content"]
                return GenerationResponse(
                    request_id=request.request_id,
                    text=str(text or ""),
                    latency_ms=(time.perf_counter() - start) * 1000,
                    backend=self.name,
                    metadata={"usage": body.get("usage"), "model": body.get("model")},
                )
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self.retries:
                    time.sleep(min(10.0, (2**attempt) + random.random()))
        return GenerationResponse(
            request_id=request.request_id,
            text="",
            latency_ms=(time.perf_counter() - start) * 1000,
            backend=self.name,
            error=last_error,
        )

    def generate(self, requests: list[GenerationRequest]) -> list[GenerationResponse]:
        return [self._one(request) for request in requests]

    def close(self) -> None:
        return None
