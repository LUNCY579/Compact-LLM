from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass
from typing import Any

from rineya_harness.config import HarnessConfig, redact_config
from rineya_harness.context.assembler import ContextAssembler
from rineya_harness.context.compaction import DeterministicCompactor
from rineya_harness.domain.enums import (
    ActorType,
    FinishAction,
    MemoryKind,
    MemoryOperationType,
    MemoryStatus,
    MemoryTemperature,
    ReasonCode,
    RouteAction,
    RunMode,
    RunStatus,
    SupportStatus,
    TraceEventType,
    WorkflowNode,
)
from rineya_harness.domain.models import (
    AnswerabilityReport,
    ClaimRecord,
    HarnessAnswer,
    HarnessRunRequest,
    MemoryDraft,
    MemoryMutation,
    RunRecord,
    StatePatch,
    WorkflowCounters,
)
from rineya_harness.errors import BudgetExceededError, ModelProtocolError
from rineya_harness.evidence.verifier import AnswerabilityGate
from rineya_harness.ids import new_id
from rineya_harness.state.store import SQLiteStateStore
from rineya_harness.tracing.jsonl import JsonlTraceRecorder

from .contracts import ResearchPolicy
from .evidence_pipeline import EvidencePipeline
from .router import route_request


@dataclass(slots=True)
class ContextHarnessWorkflow:
    """Bounded search–reconstruct–forget–verify workflow.

    This class is deliberately a deterministic shell around one probabilistic policy. The model may
    propose semantic state changes; ordinary code owns authorization, revisions, dependency checks,
    budgets, evidence persistence, deletion safety, routing, answerability, and completion.
    """

    config: HarnessConfig
    store: SQLiteStateStore
    context: ContextAssembler
    evidence_pipeline: EvidencePipeline
    policy: ResearchPolicy
    verifier: AnswerabilityGate
    compactor: DeterministicCompactor
    traces: JsonlTraceRecorder

    async def run(self, request: HarnessRunRequest) -> HarnessAnswer:
        started = time.monotonic()
        counters = WorkflowCounters()
        trace_id = new_id("trace")
        run, created = self._load_or_create_run(request)
        self.traces.emit(
            trace_id=trace_id,
            run_id=run.id,
            event_type=TraceEventType.RUN_STARTED,
            node=WorkflowNode.INTAKE,
            payload={"created": created, "mode": request.mode.value},
        )
        self.store.update_run(
            run.id,
            status=RunStatus.RUNNING,
            current_node=WorkflowNode.ROUTE,
            event_type="workflow_started",
            trace_id=trace_id,
        )
        try:
            async with asyncio.timeout(request.budget.max_wall_seconds):
                return await self._execute_request(
                    request=request,
                    run_id=run.id,
                    trace_id=trace_id,
                    counters=counters,
                    started=started,
                )
        except TimeoutError as exc:
            error = BudgetExceededError(
                "wall_seconds",
                request.budget.max_wall_seconds,
                time.monotonic() - started,
            )
            self._record_failure(run.id, trace_id, error)
            raise error from exc
        except asyncio.CancelledError:
            self.store.update_run(
                run.id,
                status=RunStatus.CANCELLED,
                current_node=WorkflowNode.ERROR,
                event_type="workflow_cancelled",
                event_payload={"message": "execution cancelled"},
                trace_id=trace_id,
            )
            self.traces.emit(
                trace_id=trace_id,
                run_id=run.id,
                event_type=TraceEventType.ERROR,
                node=WorkflowNode.ERROR,
                payload={"error_type": "CancelledError", "message": "execution cancelled"},
            )
            raise
        except Exception as exc:
            self._record_failure(run.id, trace_id, exc)
            raise

    async def _execute_request(
        self,
        *,
        request: HarnessRunRequest,
        run_id: str,
        trace_id: str,
        counters: WorkflowCounters,
        started: float,
    ) -> HarnessAnswer:
        route = route_request(request.question, request.mode)
        if route.action is RouteAction.CLARIFY:
            return self._finish(
                run_id=run_id,
                trace_id=trace_id,
                status=RunStatus.WAITING,
                action=FinishAction.CLARIFY,
                answer=route.clarification or "Could you clarify that?",
                metadata={"route_reason": route.reason},
            )
        if route.action is RouteAction.ABSTAIN:
            return self._finish(
                run_id=run_id,
                trace_id=trace_id,
                status=RunStatus.ABSTAINED,
                action=FinishAction.ABSTAIN,
                answer="I don't know based on the available information.",
                metadata={"route_reason": route.reason},
            )
        if route.action is RouteAction.DIRECT:
            return await self._run_direct(
                request=request,
                run_id=run_id,
                trace_id=trace_id,
                counters=counters,
                started=started,
                route_reason=route.reason,
            )
        return await self._run_research(
            request=request,
            run_id=run_id,
            trace_id=trace_id,
            counters=counters,
            started=started,
            initial_query=route.query or request.question,
            route_reason=route.reason,
        )

    def _record_failure(self, run_id: str, trace_id: str, exc: Exception) -> None:
        self.store.update_run(
            run_id,
            status=RunStatus.FAILED,
            current_node=WorkflowNode.ERROR,
            event_type="workflow_failed",
            event_payload={"error_type": type(exc).__name__, "message": str(exc)},
            trace_id=trace_id,
        )
        self.traces.emit(
            trace_id=trace_id,
            run_id=run_id,
            event_type=TraceEventType.ERROR,
            node=WorkflowNode.ERROR,
            payload={"error_type": type(exc).__name__, "message": str(exc)},
        )

    def _load_or_create_run(self, request: HarnessRunRequest) -> tuple[RunRecord, bool]:
        if request.run_id:
            return self.store.get_run(request.run_id), False
        run = RunRecord(
            goal=request.question,
            mode=request.mode,
            config_snapshot=redact_config(self.config),
            metadata={"request_metadata": request.metadata},
        )
        self.store.create_run(run)
        goal = MemoryDraft(
            id=f"goal_{run.id}",
            kind=MemoryKind.GLOBAL_INVARIANT,
            summary=request.question,
            payload={"goal": request.question},
            pinned=True,
            metadata={"created_by": "workflow_intake"},
        )
        self.store.apply_patch(
            StatePatch(
                run_id=run.id,
                expected_revision=0,
                actor=ActorType.SYSTEM,
                mutations=[MemoryMutation(operation=MemoryOperationType.ADD, draft=goal)],
                idempotency_key=f"initialize-{run.id}",
                reason="pin immutable task goal",
            )
        )
        return self.store.get_run(run.id), True

    async def _run_direct(
        self,
        *,
        request: HarnessRunRequest,
        run_id: str,
        trace_id: str,
        counters: WorkflowCounters,
        started: float,
        route_reason: str,
    ) -> HarnessAnswer:
        self._check_budget(request, counters, started)
        context = self.context.assemble(
            run_id=run_id,
            user_message=request.question,
            recent_turns=request.recent_turns,
            budget=request.context_budget,
        )
        self._trace_context(trace_id, run_id, context.model_dump(mode="json"))
        counters.model_calls += 1
        self._check_budget(request, counters, started)
        draft = await self.policy.direct_answer(question=request.question, context=context)
        citations = self._validate_citations(
            draft.citations,
            self.store.list_evidence(run_id),
            required=False,
        )
        return self._finish(
            run_id=run_id,
            trace_id=trace_id,
            status=RunStatus.COMPLETED,
            action=FinishAction.ANSWER,
            answer=draft.answer,
            citations=citations,
            metadata={
                "route_reason": route_reason,
                "counters": counters.model_dump(),
                "context_hash": context.context_hash,
            },
        )

    async def _run_research(
        self,
        *,
        request: HarnessRunRequest,
        run_id: str,
        trace_id: str,
        counters: WorkflowCounters,
        started: float,
        initial_query: str,
        route_reason: str,
    ) -> HarnessAnswer:
        self.store.create_checkpoint(run_id, label="before_research")
        prior_queries: list[str] = []
        unresolved: list[str] = []
        answerability: AnswerabilityReport | None = None
        query = initial_query

        for round_index in range(request.budget.max_rounds):
            counters.rounds = round_index + 1
            self._check_budget(request, counters, started)
            self.store.update_run(
                run_id,
                current_node=WorkflowNode.SEARCH,
                event_type="research_round_started",
                event_payload={"round": counters.rounds},
                trace_id=trace_id,
            )
            if request.mode is RunMode.RESEARCH or round_index > 0:
                counters.model_calls += 1
                self._check_budget(request, counters, started)
                plan = await self.policy.plan_query(
                    question=request.question,
                    active_memory=self.store.list_memory(run_id),
                    unresolved_questions=unresolved,
                    prior_queries=prior_queries,
                )
                query = plan.query
            query = self._deduplicate_query(query, prior_queries, request.question, round_index)
            prior_queries.append(query)
            counters.search_calls += 1
            self._check_budget(request, counters, started)
            remaining_sources = max(0, request.budget.max_sources_fetched - counters.sources_fetched)
            batch = await self.evidence_pipeline.gather(
                run_id=run_id,
                query=query,
                max_results=min(self.config.search.max_results, remaining_sources or 1),
                max_sources=remaining_sources,
                passage_limit=self.config.search.max_passages_per_round,
            )
            counters.sources_fetched += len(batch.documents)
            self.traces.emit(
                trace_id=trace_id,
                run_id=run_id,
                event_type=TraceEventType.TOOL_FINISHED,
                node=WorkflowNode.SEARCH,
                payload={
                    "query": query,
                    "results": len(batch.search_results),
                    "documents": len(batch.documents),
                    "evidence_records": len(batch.evidence),
                    "reused_evidence_ids": batch.reused_evidence_ids,
                    "fetch_failures": batch.fetch_failures,
                },
            )
            if batch.evidence:
                self._activate_evidence_memory(run_id, batch.evidence)

            counters.model_calls += 1
            self._check_budget(request, counters, started)
            self.store.update_run(run_id, current_node=WorkflowNode.RECONSTRUCT, trace_id=trace_id)
            outcome = await self.policy.reconstruct(
                question=request.question,
                active_memory=self.store.list_memory(run_id),
                evidence=batch.evidence,
            )
            self._validate_reconstruction(run_id, outcome)
            self._persist_reconstruction(run_id, outcome, trace_id=trace_id)
            unresolved = list(dict.fromkeys(outcome.unresolved_questions))

            self.store.update_run(run_id, current_node=WorkflowNode.COMPACT, trace_id=trace_id)
            current_run = self.store.get_run(run_id)
            compacted = self.compactor.compact_to_item_limit(
                run_id=run_id,
                max_items=min(
                    request.budget.max_active_memory_items,
                    self.config.memory.max_active_items,
                ),
                expected_revision=current_run.revision,
                trace_id=trace_id,
            )
            if compacted.deleted_ids:
                self.traces.emit(
                    trace_id=trace_id,
                    run_id=run_id,
                    event_type=TraceEventType.STATE_PATCH_COMMITTED,
                    node=WorkflowNode.COMPACT,
                    payload={"deleted_ids": compacted.deleted_ids},
                )

            self.store.update_run(
                run_id,
                status=RunStatus.VERIFYING,
                current_node=WorkflowNode.VERIFY,
                trace_id=trace_id,
            )
            search_remaining = (
                counters.rounds < request.budget.max_rounds
                and counters.search_calls < request.budget.max_search_calls
                and counters.sources_fetched < request.budget.max_sources_fetched
            )
            answerability = self.verifier.evaluate(
                evidence=self.store.list_evidence(run_id),
                claims=self.store.list_claims(run_id),
                unresolved_questions=unresolved,
                search_budget_remaining=search_remaining,
            )
            self.traces.emit(
                trace_id=trace_id,
                run_id=run_id,
                event_type=TraceEventType.ANSWER_VERIFIED,
                node=WorkflowNode.VERIFY,
                payload=answerability.model_dump(mode="json"),
            )
            if answerability.action is FinishAction.ANSWER:
                break
            if answerability.action is not FinishAction.SEARCH_MORE:
                return self._finish(
                    run_id=run_id,
                    trace_id=trace_id,
                    status=RunStatus.ABSTAINED,
                    action=FinishAction.ABSTAIN,
                    answer=self._abstention_for(answerability),
                    answerability=answerability,
                    metadata={
                        "route_reason": route_reason,
                        "counters": counters.model_dump(),
                        "queries": prior_queries,
                    },
                )

        if answerability is None or not answerability.answerable:
            report = answerability or self.verifier.evaluate(
                evidence=self.store.list_evidence(run_id),
                claims=self.store.list_claims(run_id),
                unresolved_questions=unresolved or [request.question],
                search_budget_remaining=False,
            )
            return self._finish(
                run_id=run_id,
                trace_id=trace_id,
                status=RunStatus.ABSTAINED,
                action=FinishAction.ABSTAIN,
                answer=self._abstention_for(report),
                answerability=report,
                metadata={
                    "route_reason": route_reason,
                    "counters": counters.model_dump(),
                    "queries": prior_queries,
                },
            )

        # Physical forgetting occurs here: the new model call is assembled only from current ACTIVE
        # memory after all transactional evictions/supersessions have committed.
        context = self.context.assemble(
            run_id=run_id,
            user_message=request.question,
            recent_turns=request.recent_turns,
            budget=request.context_budget,
        )
        self._trace_context(trace_id, run_id, context.model_dump(mode="json"))
        counters.model_calls += 1
        self._check_budget(request, counters, started)
        evidence = self.store.list_evidence(run_id)
        draft = await self.policy.grounded_answer(
            question=request.question,
            context=context,
            answerability=answerability,
            evidence=evidence,
        )
        allowed_citation_ids = {
            evidence_id
            for claim in self.store.list_claims(run_id)
            if claim.required_for_answer and claim.support_status is SupportStatus.SUPPORTED
            for evidence_id in claim.evidence_ids
        }
        citations = self._validate_citations(
            draft.citations,
            evidence,
            required=True,
            allowed_ids=allowed_citation_ids,
        )
        return self._finish(
            run_id=run_id,
            trace_id=trace_id,
            status=RunStatus.COMPLETED,
            action=FinishAction.ANSWER,
            answer=draft.answer,
            citations=citations,
            answerability=answerability,
            metadata={
                "route_reason": route_reason,
                "counters": counters.model_dump(),
                "queries": prior_queries,
                "context_hash": context.context_hash,
                "estimated_input_tokens": context.estimated_input_tokens,
            },
        )

    def _activate_evidence_memory(self, run_id: str, records: list[Any]) -> None:
        run = self.store.get_run(run_id)
        existing = {item.id for item in self.store.list_memory(run_id, include_inactive=True)}
        drafts = [
            self.evidence_pipeline.ledger.memory_from_evidence(record)
            for record in records
            if f"mem_{record.id}" not in existing
        ]
        if not drafts:
            return
        self.store.apply_patch(
            StatePatch(
                run_id=run_id,
                expected_revision=run.revision,
                actor=ActorType.SYSTEM,
                mutations=[
                    MemoryMutation(operation=MemoryOperationType.ADD, draft=draft)
                    for draft in drafts
                ],
                idempotency_key=f"evidence-{run_id}-{run.revision}-{self._digest([d.id for d in drafts])}",
                reason="make newly retrieved evidence visible for reconstruction",
            )
        )

    def _validate_reconstruction(self, run_id: str, outcome: Any) -> None:
        evidence_ids = {record.id for record in self.store.list_evidence(run_id)}
        memory = {item.id: item for item in self.store.list_memory(run_id, include_inactive=True)}
        reasons: list[str] = []
        for claim in outcome.claims:
            unknown = claim.evidence_ids - evidence_ids
            if unknown:
                reasons.append(f"claim cites unknown evidence IDs: {sorted(unknown)}")
            if claim.support_status is SupportStatus.SUPPORTED and not claim.evidence_ids:
                reasons.append("supported claim has no evidence ID")
        for proposed in outcome.memory_additions:
            unknown = proposed.source_ids - evidence_ids
            if unknown:
                reasons.append(f"memory proposal cites unknown evidence IDs: {sorted(unknown)}")
            missing_dependencies = proposed.dependencies - memory.keys()
            if missing_dependencies:
                reasons.append(f"memory proposal has missing dependencies: {sorted(missing_dependencies)}")
            if (
                proposed.pinned
                or proposed.temperature is MemoryTemperature.PINNED
                or proposed.kind is MemoryKind.GLOBAL_INVARIANT
            ):
                reasons.append("model cannot create pinned or global-invariant memory")
        for update in outcome.memory_updates:
            target = memory.get(update.target_id)
            if target is None:
                reasons.append(f"update target does not exist: {update.target_id}")
                continue
            allowed_model_fields = {
                "summary",
                "payload",
                "source_ids",
                "dependencies",
                "valid_from",
                "expires_at",
                "confidence",
            }
            attempted = sorted(set(update.changes) - allowed_model_fields)
            if attempted:
                reasons.append(
                    f"model cannot update protected fields on {update.target_id}: {attempted}"
                )
            if target.kind is MemoryKind.GLOBAL_INVARIANT:
                reasons.append(f"model cannot update global invariant: {update.target_id}")
        for memory_id in outcome.deletion.memory_ids:
            target = memory.get(memory_id)
            if target is None:
                reasons.append(f"delete target does not exist: {memory_id}")
            elif target.pinned:
                reasons.append(f"delete target is pinned: {memory_id}")
        if reasons:
            from rineya_harness.errors import StateInvariantError

            raise StateInvariantError(sorted(set(reasons)))

    def _persist_reconstruction(self, run_id: str, outcome: Any, *, trace_id: str) -> None:
        evidence_ids = {record.id for record in self.store.list_evidence(run_id)}
        claim_records = [
            ClaimRecord(
                run_id=run_id,
                statement=claim.statement,
                evidence_ids=claim.evidence_ids,
                support_status=claim.support_status,
                required_for_answer=claim.required_for_answer,
                freshness_required=claim.freshness_required,
                conflict_group=claim.conflict_group,
                metadata=claim.metadata,
            )
            for claim in outcome.claims
        ]
        self.store.add_claims(claim_records)
        run = self.store.get_run(run_id)
        mutations: list[MemoryMutation] = []
        addition_ids: list[str] = []
        for proposal in outcome.memory_additions:
            draft = MemoryDraft(
                kind=proposal.kind,
                summary=proposal.summary,
                payload=proposal.payload,
                source_ids=proposal.source_ids,
                dependencies=proposal.dependencies,
                temperature=proposal.temperature,
                pinned=proposal.pinned,
                metadata={**proposal.metadata, "reconstructed": True},
            )
            addition_ids.append(draft.id)
            mutations.append(MemoryMutation(operation=MemoryOperationType.ADD, draft=draft))
        for update in outcome.memory_updates:
            mutations.append(
                MemoryMutation(
                    operation=(
                        MemoryOperationType.MERGE if update.merge else MemoryOperationType.UPDATE
                    ),
                    target_id=update.target_id,
                    changes=update.changes,
                )
            )
        for memory_id in outcome.deletion.memory_ids:
            mutations.append(
                MemoryMutation(
                    operation=MemoryOperationType.EVICT,
                    target_id=memory_id,
                    replacement_ids=set(outcome.deletion.replacement_state_ids + addition_ids),
                    reason_code=outcome.deletion.reason_code,
                    note=outcome.deletion.reason,
                )
            )
        if not mutations:
            return
        patch = StatePatch(
            run_id=run_id,
            expected_revision=run.revision,
            actor=ActorType.MODEL,
            mutations=mutations,
            idempotency_key=f"reconstruct-{run_id}-{run.revision}-{self._digest([m.target_id or (m.draft.id if m.draft else '') for m in mutations])}",
            reason="commit validated reconstruction and recursive-state forgetting atomically",
            metadata={
                "claim_count": len(claim_records),
                "evidence_ids": sorted(evidence_ids),
                "unresolved_questions": outcome.unresolved_questions,
            },
        )
        result = self.store.apply_patch(patch, trace_id=trace_id)
        self.traces.emit(
            trace_id=trace_id,
            run_id=run_id,
            event_type=TraceEventType.STATE_PATCH_COMMITTED,
            node=WorkflowNode.RECONSTRUCT,
            payload=result.model_dump(mode="json"),
        )

    def _finish(
        self,
        *,
        run_id: str,
        trace_id: str,
        status: RunStatus,
        action: FinishAction,
        answer: str,
        citations: list[str] | None = None,
        answerability: AnswerabilityReport | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> HarnessAnswer:
        node = {
            FinishAction.ANSWER: WorkflowNode.DONE,
            FinishAction.CLARIFY: WorkflowNode.CLARIFY,
            FinishAction.ABSTAIN: WorkflowNode.ABSTAIN,
            FinishAction.SEARCH_MORE: WorkflowNode.SEARCH,
        }[action]
        run = self.store.update_run(
            run_id,
            status=status,
            current_node=node,
            event_type="workflow_finished",
            event_payload={"action": action.value, "citations": citations or []},
            trace_id=trace_id,
        )
        response = HarnessAnswer(
            run_id=run_id,
            status=status,
            action=action,
            answer=answer,
            citations=citations or [],
            state_revision=run.revision,
            trace_id=trace_id,
            answerability=answerability,
            metadata=metadata or {},
        )
        self.traces.emit(
            trace_id=trace_id,
            run_id=run_id,
            event_type=TraceEventType.RUN_FINISHED,
            node=node,
            payload=response.model_dump(mode="json"),
        )
        return response

    @staticmethod
    def _validate_citations(
        citations: list[str],
        evidence: list[Any],
        *,
        required: bool,
        allowed_ids: set[str] | None = None,
    ) -> list[str]:
        known = {record.id for record in evidence}
        allowed = known if allowed_ids is None else known.intersection(allowed_ids)
        normalized = list(dict.fromkeys(citations))
        unknown = sorted(set(normalized) - known)
        if unknown:
            raise ModelProtocolError(f"Answer cited unknown evidence IDs: {unknown}", raw=citations)
        unrelated = sorted(set(normalized) - allowed)
        if unrelated:
            raise ModelProtocolError(
                f"Answer cited evidence not linked to required supported claims: {unrelated}",
                raw=citations,
            )
        if required and not normalized:
            raise ModelProtocolError(
                "Grounded answer must cite at least one evidence record linked to a required claim",
                raw=citations,
            )
        return normalized

    @staticmethod
    def _abstention_for(report: AnswerabilityReport) -> str:
        if "critical_conflict_unresolved" in report.reason_codes:
            return "The available sources conflict, so I can’t verify one answer yet."
        if "freshness_requirement_failed" in report.reason_codes:
            return "I couldn’t verify sufficiently current evidence."
        return "I don't know based on the available information."

    @staticmethod
    def _deduplicate_query(
        proposed: str,
        prior_queries: list[str],
        question: str,
        round_index: int,
    ) -> str:
        normalized_prior = {" ".join(value.casefold().split()) for value in prior_queries}
        normalized = " ".join(proposed.split())
        if normalized.casefold() not in normalized_prior:
            return normalized
        return f"{question} supporting evidence round {round_index + 1}"

    @staticmethod
    def _digest(values: list[str]) -> str:
        return hashlib.sha256("|".join(sorted(values)).encode("utf-8")).hexdigest()[:16]

    def _check_budget(
        self,
        request: HarnessRunRequest,
        counters: WorkflowCounters,
        started: float,
    ) -> None:
        elapsed = time.monotonic() - started
        checks = [
            ("model_calls", request.budget.max_model_calls, counters.model_calls),
            ("search_calls", request.budget.max_search_calls, counters.search_calls),
            ("sources_fetched", request.budget.max_sources_fetched, counters.sources_fetched),
            ("wall_seconds", request.budget.max_wall_seconds, elapsed),
        ]
        for name, limit, observed in checks:
            if observed > limit:
                raise BudgetExceededError(name, limit, observed)

    def _trace_context(self, trace_id: str, run_id: str, payload: dict[str, Any]) -> None:
        self.traces.emit(
            trace_id=trace_id,
            run_id=run_id,
            event_type=TraceEventType.CONTEXT_REBUILT,
            node=WorkflowNode.ANSWER,
            payload={
                "revision": payload["revision"],
                "context_hash": payload["context_hash"],
                "estimated_input_tokens": payload["estimated_input_tokens"],
                "included_memory_ids": payload["included_memory_ids"],
                "excluded_memory_ids": payload["excluded_memory_ids"],
            },
        )
