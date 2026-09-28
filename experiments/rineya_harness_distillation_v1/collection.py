"""Bounded student-first collection with a training-only PASS/REPLACE boundary."""
from __future__ import annotations

import time
from dataclasses import asdict
from typing import Protocol
from .contracts import (EMPTY, HARNESS_HASH, MODES, Action, Limits, Reference,
                        Review, ReviewRequest, Task, View, digest, render, require, source_hash, sampling_seed)
from .environment import Environment


class Student(Protocol):
    identity: str
    def propose(self, view: View, seed: int) -> str: ...


class Reviewer(Protocol):
    identity: str
    def review(self, request: ReviewRequest) -> Review: ...


class BudgetExceeded(RuntimeError):
    pass


class Ledger:
    """Calls charged before invocation, including failures; wall checks at boundaries.

    This is not an API-token/currency budget or hard callback timeout. No live API
    adapter is bundled. Such adapters need separate reservations and deadlines.
    """
    def __init__(self, limits: Limits, clock=time.monotonic):
        self.limits, self.clock = limits, clock
        self.started = clock()
        self.student_calls = self.reviewer_calls = 0
        self.input_chars = self.output_chars = 0

    def charge(self, kind: str, content: str) -> None:
        require(kind in {"student", "reviewer"}, "unknown ledger actor")
        if (self.student_calls + self.reviewer_calls >= self.limits.max_calls
                or self.clock() - self.started >= self.limits.max_wall_seconds):
            raise BudgetExceeded("collection budget exhausted")
        if kind == "student":
            self.student_calls += 1
        else:
            self.reviewer_calls += 1
        self.input_chars += len(content)

    def snapshot(self) -> dict:
        return {name: getattr(self, name) for name in
                ("student_calls", "reviewer_calls", "input_chars", "output_chars")}


def collect(task: Task, student: Student, *, mode: str, limits: Limits,
            ledger: Ledger, attempt: int = 0, seed: int = 0,
            reviewer: Reviewer | None = None, reference: Reference = EMPTY,
            evaluation: bool = False) -> dict:
    require(mode in MODES, "unsupported collection arm")
    require(type(attempt) is int and attempt >= 0 and type(seed) is int and seed >= 0,
            "invalid attempt/seed")
    require(ledger.limits == limits, "ledger limits differ")
    require(bool(student.identity), "student identity required")
    if evaluation:
        require(task.split in {"calibration", "retention_eval"}, "evaluation role forbidden")
        require(mode == "student_only" and reviewer is None and not reference.rules,
                "deployment evaluation cannot have a reviewer/reference")
    else:
        require(task.split == "train", "collection requires TRAIN; no eval/replay leakage")
    reviewing = mode.startswith("review_")
    require((reviewer is not None) == reviewing, "reviewer/arm mismatch")
    require((mode == "review_k") == bool(reference.rules), "reference/arm mismatch")
    if reviewer is not None:
        require(bool(reviewer.identity), "reviewer identity required")
    env = Environment(task)
    code_hash = source_hash()
    session = digest({"task": task.fingerprint, "attempt": attempt, "seed": seed,
                      "mode": mode, "student": student.identity, "h": HARNESS_HASH,
                      "reviewer": reviewer.identity if reviewer else None,
                      "reference": reference.fingerprint, "source": code_hash,
                      "limits": asdict(limits)})
    record = {"schema": 1, "task_key": task.key, "task_hash": task.fingerprint,
              "session": session, "attempt": attempt, "seed": seed, "mode": mode,
              "student": student.identity, "reviewer": reviewer.identity if reviewer else None,
              "reference_hash": reference.fingerprint, "harness_hash": HARNESS_HASH,
              "source_hash": code_hash, "limits": asdict(limits), "steps": [], "status": "TRUNCATED",
              "evaluation": evaluation, "success": False, "error_type": None}
    before = ledger.snapshot()
    try:
        for index in range(limits.max_steps):
            view = env.view()
            context = render(view)
            if len(context) > limits.max_context_chars:
                record["status"] = "TRUNCATED"
                break
            ledger.charge("student", context)
            proposal = student.propose(view, sampling_seed(task.key, attempt, seed, index))
            require(type(proposal) is str and len(proposal) <= limits.max_response_chars,
                    "student response exceeds cap/type")
            ledger.output_chars += len(proposal)
            request = ReviewRequest(view, proposal, reference, session, index)
            review = None
            if reviewer:
                reviewer_input = context + proposal + str(asdict(reference))
                ledger.charge("reviewer", reviewer_input)
                review = reviewer.review(request)
                require(isinstance(review, Review), "reviewer DTO required")
                # Charge the complete review, not just its eventual SFT target.
                ledger.output_chars += len(str(asdict(review)))
                accepted = review.accept(request)
                require(len(accepted) <= limits.max_response_chars, "replacement exceeds cap")
            else:
                Action.parse(proposal)
                accepted = proposal
            observation = env.execute(accepted)
            record["steps"].append({"view_hash": view.fingerprint, "proposal": proposal,
                                    "review": asdict(review) if review else None,
                                    "accepted": accepted, "observation": observation})
            if env.final is not None:
                record["success"] = env.success()
                record["status"] = "SUCCESS" if record["success"] else "FAILURE"
                break
    except BudgetExceeded as exc:
        record["status"], record["error_type"] = "BUDGET", type(exc).__name__
    except (ValueError, TypeError, KeyError) as exc:
        record["status"], record["error_type"] = "INVALID", type(exc).__name__
    except Exception as exc:
        # Do not place exception text (possibly API keys/private paths) in exports.
        record["status"], record["error_type"] = "ERROR", type(exc).__name__
    record["cost"] = {key: value - before[key] for key, value in ledger.snapshot().items()}
    record["trace_hash"] = digest(record)
    return record
