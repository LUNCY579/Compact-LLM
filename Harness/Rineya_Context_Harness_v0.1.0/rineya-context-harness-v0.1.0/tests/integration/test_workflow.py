from __future__ import annotations

import pytest

from rineya_harness.domain.enums import FinishAction, MemoryKind, MemoryStatus, RunMode, RunStatus
from rineya_harness.domain.models import HarnessRunRequest


@pytest.mark.asyncio
async def test_fixture_research_searches_reconstructs_forgets_and_answers(research_container) -> None:
    question = "How does Rineya forget obsolete memory?"
    answer = await research_container.workflow.run(
        HarnessRunRequest(question=question, mode=RunMode.GROUNDED)
    )
    assert answer.status is RunStatus.COMPLETED
    assert answer.action is FinishAction.ANSWER
    assert answer.citations
    all_memory = research_container.store.list_memory(answer.run_id, include_inactive=True)
    assert any(item.kind is MemoryKind.RESOLVED_EVIDENCE for item in all_memory)
    assert any(
        item.kind is MemoryKind.ACTIVE_EVIDENCE and item.status is MemoryStatus.EVICTED
        for item in all_memory
    )
    active = research_container.store.list_memory(answer.run_id)
    assert all(item.kind is not MemoryKind.ACTIVE_EVIDENCE for item in active)
    assert research_container.traces.read(answer.run_id)


@pytest.mark.asyncio
async def test_grounded_mode_abstains_without_evidence(container) -> None:
    answer = await container.workflow.run(
        HarnessRunRequest(question="Unknown fixture question", mode=RunMode.GROUNDED)
    )
    assert answer.status is RunStatus.ABSTAINED
    assert answer.action is FinishAction.ABSTAIN


@pytest.mark.asyncio
async def test_direct_mode_uses_direct_policy(container) -> None:
    answer = await container.workflow.run(
        HarnessRunRequest(question="Why does ice float?", mode=RunMode.DIRECT)
    )
    assert answer.status is RunStatus.COMPLETED
    assert "configured Rineya model" in answer.answer


@pytest.mark.asyncio
async def test_workflow_enforces_wall_clock_deadline(container) -> None:
    import asyncio

    from rineya_harness.domain.enums import RunMode, RunStatus
    from rineya_harness.domain.models import HarnessRunRequest, RunRecord, WorkflowBudget
    from rineya_harness.errors import BudgetExceededError
    from rineya_harness.orchestration.contracts import AnswerDraft

    class SlowPolicy:
        async def direct_answer(self, **_kwargs):
            await asyncio.sleep(0.1)
            return AnswerDraft(answer="late")

    container.workflow.policy = SlowPolicy()  # type: ignore[assignment]
    request = HarnessRunRequest(
        question="hello",
        mode=RunMode.DIRECT,
        run_id="deadline-run",
        budget=WorkflowBudget(max_wall_seconds=0.01),
    )
    container.store.create_run(
        RunRecord(id="deadline-run", goal="hello", mode=RunMode.DIRECT)
    )
    with pytest.raises(BudgetExceededError):
        await container.workflow.run(request)
    assert container.store.get_run("deadline-run").status is RunStatus.FAILED


@pytest.mark.asyncio
async def test_direct_answer_rejects_unknown_citation(container) -> None:
    from rineya_harness.domain.models import HarnessRunRequest
    from rineya_harness.errors import ModelProtocolError
    from rineya_harness.orchestration.contracts import AnswerDraft

    class BadCitationPolicy:
        async def direct_answer(self, **_kwargs):
            return AnswerDraft(answer="Claim.", citations=["missing-evidence"])

    container.workflow.policy = BadCitationPolicy()  # type: ignore[assignment]
    with pytest.raises(ModelProtocolError, match="unknown evidence"):
        await container.workflow.run(
            HarnessRunRequest(question="hello", mode=RunMode.DIRECT)
        )
