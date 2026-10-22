"""Contracts, fail-closed budgets, restricted runbooks, and source binding."""
from __future__ import annotations

import hashlib
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from experiments.rineya_harness_distillation_v1 import contracts as rhd
from experiments.rineya_harness_distillation_v1 import environment as rhd_env
from experiments.rineya_harness_distillation_v1.contracts import (
    Action, Task, View, canonical, digest, loads, render, require, validate_catalog,
)
from experiments.rineya_harness_distillation_v1.environment import Environment
from . import VERSION

PINNED_PARENT = "19551ddebc87c93161c303865ce6d0f130de0093"
UPSTREAM = {"contracts.py": "527b6e035e444cd56193de024d4a94b9281129d7",
            "environment.py": "9fe2a782659500f01aa00abdf23005dfd8eee349"}
STEPS = {
    "inspect_requested": "Retrieve only a record key available in public evidence.",
    "check_observed": "Check the public or retrieved arithmetic expression.",
    "answer_observed": "Answer from observed evidence, not a supplied hidden answer.",
    "abstain_missing": "Abstain after the requested evidence is found to be absent.",
    "repair_public_error": "Repair only using public tool feedback; do not invent evidence.",
}
FEEDBACK = {"schema", "unsupported", "leakage", "incomplete", "accepted"}


def source_hash() -> str:
    files = {"rsr/" + p.name: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(Path(__file__).parent.glob("*.py"))}
    for module in (rhd, rhd_env):
        p = Path(module.__file__)
        raw = p.read_bytes()
        blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        require(blob == UPSTREAM[p.name], "upstream changed: requalification required")
        files["rhd/" + p.name] = blob
    return digest(files)


def enroll(tasks: list[Task], ancestors: dict[str, list[str]]) -> dict[str, Task]:
    catalog = validate_catalog(tasks)
    require(set(catalog) == set(ancestors), "complete ancestry enrollment required")
    ownership = {}
    for task in tasks:
        lineage = ancestors[task.key]
        require(type(lineage) is list and bool(lineage), "ancestry must be nonempty")
        for parent in [task.parent] + lineage:
            require(type(parent) is str and bool(parent), "invalid ancestor")
            require(parent not in ownership or ownership[parent] == task.split,
                    "cross-split ancestry contamination")
            ownership[parent] = task.split
    return catalog


@dataclass(frozen=True)
class Runbook:
    steps: tuple[str, ...]

    def __post_init__(self):
        require(type(self.steps) is tuple and 1 <= len(self.steps) <= 8, "runbook length")
        require(all(type(s) is str and s in STEPS for s in self.steps), "runbook vocabulary")
        require(len(set(self.steps)) == len(self.steps), "duplicate runbook operation")

    def encode(self) -> str:
        return canonical({"goal": "evidence_backed_completion", "steps": self.steps})

    @property
    def fingerprint(self) -> str:
        return digest(loads(self.encode()))

    @classmethod
    def parse(cls, raw: str) -> "Runbook":
        require(type(raw) is str and len(raw) <= 2048, "runbook response bound")
        value = loads(raw)
        require(type(value) is dict and set(value) == {"goal", "steps"}, "runbook fields")
        require(value["goal"] == "evidence_backed_completion", "runbook goal")
        require(type(value["steps"]) is list, "runbook steps type")
        return cls(tuple(value["steps"]))

    def guidance(self) -> str:
        return canonical({"runbook": loads(self.encode()),
                          "semantics": {s: STEPS[s] for s in self.steps}})


@dataclass(frozen=True)
class Limits:
    calls: int = 160
    input_chars: int = 1_000_000
    output_chars: int = 100_000
    tools: int = 100
    steps: int = 8
    context_chars: int = 32_000
    response_chars: int = 2048
    new_tokens: int = 256
    tokens: int | None = None
    wall_seconds: float = 120.0

    def __post_init__(self):
        for name in ("calls", "input_chars", "output_chars", "tools", "steps",
                     "context_chars", "response_chars", "new_tokens"):
            require(type(getattr(self, name)) is int and getattr(self, name) > 0, name)
        require(self.response_chars <= 2048, "response exceeds action/runbook parser")
        require(self.tokens is None or (type(self.tokens) is int and self.tokens > 0), "tokens")
        require(type(self.wall_seconds) in (float, int) and math.isfinite(self.wall_seconds)
                and self.wall_seconds > 0, "wall_seconds")


@dataclass(frozen=True)
class Reply:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True)
class Backend:
    identity: str
    origin: str
    generate: Callable[[str, int, int], Reply]
    count_tokens: Callable[[str], int] | None = None

    def __post_init__(self):
        require(type(self.identity) is str and bool(self.identity), "backend identity")
        require(self.origin in {"fixture", "native", "teacher"}, "backend origin")
        require(callable(self.generate), "backend callback")
        require(self.origin == "fixture" or callable(self.count_tokens), "native token counter")


class BudgetExceeded(RuntimeError):
    pass


class Ledger:
    """Preflight reserves maximum response; callback timeouts need host enforcement.

    Errors consume calls. Character and token accounting are never conflated.
    No price, FLOP, or hard process-isolation claim is made by this ledger.
    """
    def __init__(self, limits: Limits):
        self.limits = limits
        self.started = time.monotonic()
        self.used = dict(calls=0, input_chars=0, output_chars=0, tools=0, tokens=0)
        self.token_accounting = True
        self.events: list[dict] = []

    def time_guard(self):
        if time.monotonic() - self.started >= self.limits.wall_seconds:
            raise BudgetExceeded("wall budget")

    def call(self, backend: Backend, role: str, prompt: str, seed: int) -> str:
        self.time_guard()
        require(type(prompt) is str, "prompt type")
        token_input = backend.count_tokens(prompt) if backend.count_tokens else None
        if token_input is not None:
            require(type(token_input) is int and token_input >= 0, "invalid token count")
        require(backend.origin == "fixture" or self.limits.tokens is not None,
                "native calls require explicit token budget")
        reservations = dict(calls=1, input_chars=len(prompt),
                            output_chars=self.limits.response_chars)
        if token_input is not None:
            reservations["tokens"] = token_input + self.limits.new_tokens
        for key, amount in reservations.items():
            cap = getattr(self.limits, key)
            if cap is not None and self.used[key] + amount > cap:
                raise BudgetExceeded(key)
        if len(prompt) > self.limits.context_chars:
            raise BudgetExceeded("context: no truncation")
        if token_input is None:
            self.token_accounting = False
        self.used["calls"] += 1
        self.used["input_chars"] += len(prompt)
        event = {"role": role, "backend": backend.identity, "origin": backend.origin,
                 "prompt": prompt, "seed": seed, "status": "ERROR"}
        self.events.append(event)
        try:
            reply = backend.generate(prompt, seed, self.limits.new_tokens)
            require(isinstance(reply, Reply) and type(reply.text) is str, "callback response")
            self.used["output_chars"] += len(reply.text)
            if token_input is None:
                self.token_accounting = False
            else:
                require(reply.input_tokens == token_input, "input-token accounting mismatch")
                require(type(reply.output_tokens) is int
                        and 0 <= reply.output_tokens <= self.limits.new_tokens,
                        "output-token accounting mismatch")
                self.used["tokens"] += token_input + reply.output_tokens
            event["response"] = reply.text[:self.limits.response_chars]
            require(len(reply.text) <= self.limits.response_chars, "oversized model response")
            self.time_guard()
            event["status"] = "OK"
            return reply.text
        except Exception as exc:
            # An unknown native call may have consumed its full reserved generation.
            if token_input is not None and "response" not in event:
                self.used["tokens"] += token_input + self.limits.new_tokens
            event["error"] = type(exc).__name__
            raise

    def tool(self):
        self.time_guard()
        if self.used["tools"] >= self.limits.tools:
            raise BudgetExceeded("tools")
        self.used["tools"] += 1

    def report(self) -> dict:
        return {**self.used, "tokens": self.used["tokens"] if self.token_accounting else None,
                "limits": asdict(self.limits), "elapsed_seconds": time.monotonic() - self.started}


def student_prompt(view: View, assistance: str = "") -> str:
    # Guidance is appended only at collection. Rewritten SFT calls this without it.
    return render(view) + ("\nPRIVATE_TRAINING_GUIDANCE\n" + assistance if assistance else "")


def seed_for(*parts) -> int:
    return int(digest(parts)[:8], 16) % (2**31)
