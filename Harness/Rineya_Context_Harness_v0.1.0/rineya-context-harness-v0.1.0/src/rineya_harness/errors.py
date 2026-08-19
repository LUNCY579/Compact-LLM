from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class HarnessError(Exception):
    """Base class for expected harness failures."""


@dataclass(slots=True)
class NotFoundError(HarnessError):
    resource_type: str
    resource_id: str

    def __str__(self) -> str:
        return f"{self.resource_type} not found: {self.resource_id}"


@dataclass(slots=True)
class ConflictError(HarnessError):
    message: str

    def __str__(self) -> str:
        return self.message


@dataclass(slots=True)
class RevisionConflictError(HarnessError):
    run_id: str
    expected: int
    actual: int

    def __str__(self) -> str:
        return (
            f"Revision conflict for run {self.run_id}: expected {self.expected}, "
            f"actual {self.actual}."
        )


@dataclass(slots=True)
class StateInvariantError(HarnessError):
    reasons: list[str]

    def __str__(self) -> str:
        return "State invariant violation: " + "; ".join(self.reasons)


@dataclass(slots=True)
class ToolRejectedError(HarnessError):
    tool_name: str
    reason_code: str
    message: str

    def __str__(self) -> str:
        return f"Tool {self.tool_name} rejected ({self.reason_code}): {self.message}"


@dataclass(slots=True)
class SecurityViolationError(HarnessError):
    code: str
    message: str

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


@dataclass(slots=True)
class BudgetExceededError(HarnessError):
    budget_name: str
    limit: int | float
    observed: int | float

    def __str__(self) -> str:
        return f"Budget {self.budget_name} exceeded: {self.observed} > {self.limit}"


@dataclass(slots=True)
class ModelProtocolError(HarnessError):
    message: str
    raw: Any | None = None

    def __str__(self) -> str:
        return self.message
