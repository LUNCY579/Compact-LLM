from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import BaseModel

from rineya_harness.domain.enums import ToolRisk
from rineya_harness.domain.models import ToolCall, ToolResult
from rineya_harness.errors import ToolRejectedError

ArgsT = TypeVar("ArgsT", bound=BaseModel)
ToolHandler = Callable[[BaseModel], Awaitable[dict[str, Any]]]


@dataclass(frozen=True, slots=True)
class ToolDefinition(Generic[ArgsT]):
    name: str
    description: str
    args_model: type[ArgsT]
    risk: ToolRisk
    handler: ToolHandler

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "risk": self.risk.value,
            "input_schema": self.args_model.model_json_schema(),
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition[Any]] = {}

    def register(self, definition: ToolDefinition[Any]) -> None:
        if definition.name in self._tools:
            raise ValueError(f"Duplicate tool name: {definition.name}")
        self._tools[definition.name] = definition

    def definitions(self) -> list[dict[str, Any]]:
        return [self._tools[name].schema() for name in sorted(self._tools)]

    async def execute(self, call: ToolCall) -> ToolResult:
        definition = self._tools.get(call.name)
        if definition is None:
            raise ToolRejectedError(call.name, "UNKNOWN_TOOL", "Tool is not registered")
        started = call.requested_at
        try:
            args = definition.args_model.model_validate(call.arguments)
            output = await definition.handler(args)
            return ToolResult(
                call_id=call.id,
                name=call.name,
                ok=True,
                output=output,
                started_at=started,
            )
        except ToolRejectedError:
            raise
        except Exception as exc:
            return ToolResult(
                call_id=call.id,
                name=call.name,
                ok=False,
                error_code=type(exc).__name__,
                error_message=str(exc),
                started_at=started,
            )
