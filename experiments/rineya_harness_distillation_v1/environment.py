"""Deterministic local environment. No eval, subprocess, network, or hidden tools."""
from __future__ import annotations

import re
from .contracts import Action, Message, Task, View, canonical, require


def arithmetic(expression: str) -> str:
    match = re.fullmatch(r"\s*(-?\d{1,6})\s*([+*\-])\s*(-?\d{1,6})\s*", expression)
    require(match is not None, "unsupported bounded arithmetic")
    a, op, b = int(match[1]), match[2], int(match[3])
    return str(a + b if op == "+" else a - b if op == "-" else a * b)


class Environment:
    def __init__(self, task: Task):
        self._task = task
        self._messages = [Message("user", task.prompt)]
        self.final: Action | None = None

    def view(self) -> View:
        return View(tuple(self._messages))

    def execute(self, raw: str) -> str | None:
        require(self.final is None, "action after terminal")
        action = Action.parse(raw)
        if action.kind in {"answer", "abstain"}:
            observation = None
            self.final = action
        elif action.kind == "check":
            try:
                result = arithmetic(action.value)
                value = {"kind": "check", "ok": True, "result": result}
            except ValueError:
                value = {"kind": "check", "ok": False, "error": "unsupported_expression"}
            observation = canonical(value)
        else:
            records = dict(self._task.records)
            value = {"kind": "retrieve", "key": action.value,
                     "found": action.value in records, "result": records.get(action.value)}
            observation = canonical(value)
        self._messages.append(Message("assistant", raw))
        if observation is not None:
            self._messages.append(Message("tool", observation))
        return observation

    def success(self) -> bool:
        if self.final is None:
            return False
        if self._task.expected is None:
            return self.final.kind == "abstain"
        return self.final.kind == "answer" and self.final.value.strip() == self._task.expected.strip()
