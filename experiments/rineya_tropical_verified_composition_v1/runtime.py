"""Explicit budgets and root/frontier rollout accounting. No network or loader."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import random
from typing import Protocol

from .domain import InvalidAction, State, Task, root, step, success, terminal, validate_state

COUNTERS = ("root_attempts", "restart_attempts", "policy_calls", "generated_tokens",
            "prompt_tokens", "env_steps", "replay_steps", "verification_calls",
            "score_calls", "scored_tokens", "join_candidates", "trained_actions",
            "supervised_tokens", "optimizer_steps")


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Ledger:
    limits: dict[str, int] = field(default_factory=dict)
    used: dict[str, int] = field(default_factory=lambda: dict.fromkeys(COUNTERS, 0))
    violations: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        for key, value in self.limits.items():
            if key not in COUNTERS or type(value) is not int or value < 0:
                raise ValueError("Unknown or invalid budget limit")

    def remaining(self, key: str) -> int:
        return self.limits.get(key, 1_000_000) - self.used[key]

    def require(self, **cost: int) -> None:
        if self.violations:
            raise BudgetExceeded("Ledger already exceeded; further work prohibited")
        for key, value in cost.items():
            if key not in COUNTERS or type(value) is not int or value < 0:
                raise ValueError("Unknown or invalid cost")
            if value > self.remaining(key):
                raise BudgetExceeded(f"Insufficient {key}")

    def spend(self, **cost: int) -> None:
        # Debit actual external receipts even if a faulty provider exceeded its cap.
        for key, value in cost.items():
            if key not in COUNTERS or type(value) is not int or value < 0:
                raise ValueError("Unknown or invalid cost")
        for key, value in cost.items():
            self.used[key] += value
            if self.remaining(key) < 0:
                self.violations.append(key)
        if self.violations:
            raise BudgetExceeded("Actual costs exceeded: " + ",".join(self.violations))

    def report(self) -> dict:
        return {"limits": self.limits, "used": self.used.copy(),
                "violations": list(self.violations)}


@dataclass(frozen=True)
class Sample:
    action: str
    generated_tokens: int = 0
    prompt_tokens: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.action, str):
            raise ValueError("Action must be a string")
        for value in (self.generated_tokens, self.prompt_tokens):
            if type(value) is not int or value < 0:
                raise ValueError("Invalid token receipt")


class Policy(Protocol):
    def sample(self, task: Task, state: State, rng: random.Random,
               max_new_tokens: int) -> Sample: ...


@dataclass(frozen=True)
class Episode:
    attempt_id: str
    kind: str
    states: tuple[State, ...]
    actions: tuple[str, ...]
    accepted: bool
    invalid_action: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def rollout(task: Task, policy: Policy, rng: random.Random, ledger: Ledger,
            attempt_id: str, start: State | None = None,
            max_action_tokens: int = 32) -> Episode:
    if type(max_action_tokens) is not int or max_action_tokens < 1:
        raise ValueError("Invalid per-action generation cap")
    state = root(task) if start is None else start
    validate_state(task, state)
    if terminal(task, state):
        raise ValueError("Cannot start at a terminal")
    kind = "root" if state == root(task) else "frontier"
    counter = "root_attempts" if kind == "root" else "restart_attempts"
    ledger.require(**{counter: 1})
    ledger.spend(**{counter: 1})
    states, actions = [state], []
    invalid = None
    while not terminal(task, state):
        ledger.require(policy_calls=1, env_steps=1)
        allowance = min(max_action_tokens, max(0, ledger.remaining("generated_tokens")))
        ledger.spend(policy_calls=1)
        sample = policy.sample(task, state, rng, allowance)
        ledger.spend(generated_tokens=sample.generated_tokens,
                     prompt_tokens=sample.prompt_tokens)
        if sample.generated_tokens > allowance:
            ledger.violations.append("per_action_generation_cap")
            raise BudgetExceeded("Generator exceeded its per-action allowance; actual tokens recorded")
        ledger.spend(env_steps=1)
        try:
            following = step(task, state, sample.action)
        except InvalidAction:
            invalid = sample.action
            break
        actions.append(sample.action)
        states.append(following)
        state = following
    ledger.require(verification_calls=1)
    ledger.spend(verification_calls=1)
    return Episode(attempt_id, kind, tuple(states), tuple(actions),
                   success(task, state), invalid)


def replay(task: Task, actions: tuple[str, ...], ledger: Ledger,
           require_success: bool = True) -> tuple[State, ...]:
    state, states = root(task), [root(task)]
    ledger.require(replay_steps=len(actions), verification_calls=1)
    ledger.spend(verification_calls=1)
    for action in actions:
        ledger.spend(replay_steps=1)
        state = step(task, state, action)
        states.append(state)
    if require_success and not success(task, state):
        raise ValueError("Unverified or incomplete path")
    return tuple(states)


class ScriptedPolicy:
    """Mechanism fixture ONLY; not a Rineya rollout or learning result."""
    def __init__(self, actions: tuple[str, ...]):
        self.actions = iter(actions)

    def sample(self, task: Task, state: State, rng: random.Random,
               max_new_tokens: int) -> Sample:
        return Sample(next(self.actions, "INVALID"))
