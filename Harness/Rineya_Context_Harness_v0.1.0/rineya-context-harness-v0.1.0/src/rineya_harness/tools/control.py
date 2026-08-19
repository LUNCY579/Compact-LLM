from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from rineya_harness.domain.enums import FinishAction, ToolRisk
from rineya_harness.domain.models import StrictModel

from .base import ToolDefinition


class FinishArgs(StrictModel):
    """A model proposal for leaving the bounded tool loop.

    The workflow still verifies answerability, citations, budgets, and legal transitions before
    accepting the proposal.
    """

    action: FinishAction
    answer: str = Field(default="", max_length=20_000)
    citations: list[str] = Field(default_factory=list, max_length=128)
    reason: str | None = Field(default=None, max_length=2_000)

    @model_validator(mode="after")
    def answer_requires_text(self) -> "FinishArgs":
        if self.action in {FinishAction.ANSWER, FinishAction.CLARIFY, FinishAction.ABSTAIN}:
            if not self.answer.strip():
                raise ValueError(f"{self.action.value} requires a non-empty user-visible answer")
        if self.action is not FinishAction.ANSWER and self.citations:
            raise ValueError("citations are only valid for answer proposals")
        return self


class FinishToolSet:
    async def finish(self, args: FinishArgs) -> dict[str, Any]:
        # This tool intentionally returns a proposal. The orchestrator owns the actual transition.
        return args.model_dump(mode="json")

    def definitions(self) -> list[ToolDefinition[Any]]:
        return [
            ToolDefinition(
                name="finish",
                description=(
                    "Propose ANSWER, SEARCH_MORE, CLARIFY, or ABSTAIN. The deterministic "
                    "workflow verifies the proposal before ending or continuing the run."
                ),
                args_model=FinishArgs,
                risk=ToolRisk.READ_ONLY,
                handler=self.finish,
            )
        ]
