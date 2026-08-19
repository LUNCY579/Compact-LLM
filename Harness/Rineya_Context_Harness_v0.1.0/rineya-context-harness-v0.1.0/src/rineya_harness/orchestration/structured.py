from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from rineya_harness.errors import ModelProtocolError

ModelT = TypeVar("ModelT", bound=BaseModel)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_structured_output(text: str, model: type[ModelT]) -> ModelT:
    """Parse strict JSON while tolerating one surrounding Markdown fence.

    Arbitrary prose extraction is intentionally not supported: ambiguous structured output should
    fail closed and enter the workflow's explicit repair or fallback path.
    """

    candidate = _FENCE.sub("", text.strip()).strip()
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ModelProtocolError("Model did not return one valid JSON object", raw=text) from exc
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        raise ModelProtocolError("Model JSON did not satisfy the response contract", raw=payload) from exc
