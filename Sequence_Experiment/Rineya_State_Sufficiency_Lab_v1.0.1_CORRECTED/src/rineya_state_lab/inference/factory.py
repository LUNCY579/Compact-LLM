from __future__ import annotations

from typing import Any

from .base import InferenceBackend
from .reference import ReferenceBackend
from .replay import ReplayBackend


def create_backend(config: dict[str, Any]) -> InferenceBackend:
    backend_type = str(config.get("type", "reference")).casefold()
    if backend_type == "reference":
        return ReferenceBackend()
    if backend_type == "replay":
        return ReplayBackend(
            config["path"],
            id_field=str(config.get("id_field", "request_id")),
            text_field=str(config.get("text_field", "text")),
        )
    if backend_type in {"local", "local_transformers", "transformers"}:
        from .local_transformers import LocalTransformersBackend

        return LocalTransformersBackend(config)
    if backend_type in {"openai", "openai_compatible", "vllm"}:
        from .openai_compatible import OpenAICompatibleBackend

        return OpenAICompatibleBackend(config)
    raise ValueError(f"Unsupported inference backend type: {backend_type}")
