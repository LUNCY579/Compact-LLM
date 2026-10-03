"""Small deterministic action-level environment for prerequisite_plan worlds.

This is an actual reset/step state transition interface, not just fixed answer
pairs. No network, shell, file-write, or real-world actions are exposed. It uses
sparse terminal task reward; it does not claim to implement a complete RL trainer.
"""
from __future__ import annotations
import copy
from typing import Any
from .common import digest
from .oracle import audit_answer, replay_transition, check_world


class PlanningEnvironment:
    def __init__(self, world: dict[str, Any], max_steps: int = 128):
        check_world(world)
        if world["family"] != "prerequisite_plan":
            raise ValueError("This environment supports prerequisite_plan only")
        if type(max_steps) is not int or not 1 <= max_steps <= 512:
            raise ValueError("Invalid step budget")
        self._world = copy.deepcopy(world)
        self._certificate = audit_answer(world)
        self._actions = {a["name"]: a for a in world["records"]}
        self.max_steps = max_steps
        self.reset()

    def _goal(self) -> bool:
        q = self._world["query"]
        return self.state & q["goal_on"] == q["goal_on"] and not self.state & q["goal_off"]

    def observation(self) -> dict[str, Any]:
        # No reference answer, optimal cost, hidden state, or future tape.
        q = self._world["query"]
        return {"state": self.state, "bits": q["bits"], "goal_on": q["goal_on"], "goal_off": q["goal_off"],
                "actions": copy.deepcopy(list(self._actions.values())), "spent_cost": self.cost,
                "steps_used": self.steps, "steps_remaining": self.max_steps - self.steps}

    def reset(self) -> dict[str, Any]:
        self.state = self._world["query"]["initial"]
        self.cost = self.steps = 0
        self.terminated = self.truncated = False
        self.history: list[dict[str, Any]] = []
        return self.observation()

    def step(self, action: str) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self.terminated or self.truncated:
            raise RuntimeError("Reset is required after an episode ends")
        if not isinstance(action, str):
            raise ValueError("Action must be a name, STOP, or IMPOSSIBLE")
        before = self.state
        self.steps += 1
        reward = 0.0
        reason = "transition"
        if action == "STOP":
            self.terminated = True
            success = self._goal()
            optimal = success and self.cost == self._certificate.get("optimal_cost")
            reward = float(optimal)
            reason = "optimal_goal" if optimal else "suboptimal_goal" if success else "premature_stop"
        elif action == "IMPOSSIBLE":
            self.terminated = True
            reward = float(self._certificate.get("status") == "IMPOSSIBLE")
            reason = "verified_impossibility" if reward else "false_impossibility"
        elif action not in self._actions:
            self.terminated = True
            reason = "unknown_action"
        else:
            try:
                self.state = replay_transition("prerequisite_plan", self.state, self._actions[action])
                self.cost += self._actions[action]["cost"]
            except ValueError:
                self.terminated = True
                reason = "failed_precondition"
        if not self.terminated and self.steps >= self.max_steps:
            self.truncated = True
            reason = "step_budget_exhausted"
        entry = {"action": action, "before": before, "after": self.state, "cost": self.cost, "reward": reward, "reason": reason}
        self.history.append(entry)
        info = {"reason": reason, "history_sha256": digest(self.history)}
        return self.observation(), reward, self.terminated, self.truncated, info
