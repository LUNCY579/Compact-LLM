"""Small deterministic test domain. State identity includes task, rules and depth."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Iterable

ACTIONS = ("inc_x", "inc_y", "double_x", "double_y", "swap", "INVALID")
VERSION = "register-pair-v1"
ROLES = ("train", "replay_train", "dev", "test", "guard")


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def integer(value: object, name: str, low: int, high: int) -> None:
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")


@dataclass(frozen=True)
class Task:
    task_id: str
    parent_id: str
    role: str
    initial: tuple[int, int]
    goal: tuple[int, int]
    horizon: int = 5
    bound: int = 32
    rules: str = VERSION

    def __post_init__(self) -> None:
        if not self.task_id or not self.parent_id or self.role not in ROLES:
            raise ValueError("Invalid task identity or role")
        if self.rules != VERSION:
            raise ValueError("Unqualified rule version")
        integer(self.horizon, "horizon", 1, 12)
        integer(self.bound, "bound", 1, 1024)
        for pair in (self.initial, self.goal):
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError("Registers must be immutable pairs")
            for value in pair:
                integer(value, "register", -self.bound, self.bound)
        if self.initial == self.goal:
            raise ValueError("Initially solved problems are not enrolled")

    @property
    def key(self) -> str:
        return digest(asdict(self))

    @property
    def semantic_key(self) -> str:
        # Excludes labels: relabeling an identical world cannot evade split checks.
        return digest({k: v for k, v in asdict(self).items()
                       if k not in ("task_id", "parent_id", "role")})

    @classmethod
    def from_dict(cls, data: dict) -> Task:
        data = dict(data)
        data["initial"] = tuple(data["initial"])
        data["goal"] = tuple(data["goal"])
        return cls(**data)


@dataclass(frozen=True, order=True)
class State:
    task_key: str
    depth: int
    x: int
    y: int


def root(task: Task) -> State:
    return State(task.key, 0, *task.initial)


def validate_state(task: Task, state: State) -> None:
    if state.task_key != task.key:
        raise ValueError("Cross-task/state-interface mismatch")
    integer(state.depth, "depth", 0, task.horizon)
    integer(state.x, "x", -task.bound, task.bound)
    integer(state.y, "y", -task.bound, task.bound)
    if state.depth == 0 and (state.x, state.y) != task.initial:
        raise ValueError("Forged root state")


def success(task: Task, state: State) -> bool:
    validate_state(task, state)
    return (state.x, state.y) == task.goal


def terminal(task: Task, state: State) -> bool:
    return success(task, state) or state.depth == task.horizon


class InvalidAction(ValueError):
    """An attempted action is counted but cannot become a graph edge."""


def step(task: Task, state: State, action: str) -> State:
    validate_state(task, state)
    if terminal(task, state):
        raise InvalidAction("Action after terminal state")
    x, y = state.x, state.y
    if action == "inc_x":
        x += 1
    elif action == "inc_y":
        y += 1
    elif action == "double_x":
        x *= 2
    elif action == "double_y":
        y *= 2
    elif action == "swap":
        x, y = y, x
    else:
        raise InvalidAction("Unknown action; no execution of model-generated code")
    if max(abs(x), abs(y)) > task.bound:
        raise InvalidAction("Register bound exceeded")
    return State(task.key, state.depth + 1, x, y)


def prompt(task: Task, state: State) -> str:
    """The same state always receives the same history-free policy input."""
    validate_state(task, state)
    return json.dumps({"rules": task.rules, "registers": [state.x, state.y],
                       "goal": task.goal, "remaining_steps": task.horizon - state.depth,
                       "bound": task.bound, "actions": ACTIONS[:-1]},
                      sort_keys=True, separators=(",", ":"))


def audit_splits(tasks: Iterable[Task]) -> dict:
    ids: set[str] = set()
    parents: dict[str, str] = {}
    worlds: dict[str, str] = {}
    counts: dict[str, int] = {}
    for task in tasks:
        if task.task_id in ids:
            raise ValueError("Duplicate task ID")
        ids.add(task.task_id)
        for key, registry in ((task.parent_id, parents), (task.semantic_key, worlds)):
            if key in registry and registry[key] != task.role:
                raise ValueError("Cross-role parent or semantic-world leakage")
            registry[key] = task.role
        counts[task.role] = counts.get(task.role, 0) + 1
    if not ids:
        raise ValueError("Empty task manifest")
    return {"tasks": len(ids), "parents": len(parents), "roles": counts}
