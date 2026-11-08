"""A resumable state machine, not a step-number-based dataset unlock script."""
from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path

from .backends import Backend
from .curriculum import TaskStats, batch, gate, paired_interval, unlocked
from .data import Catalog
from .evidence import BudgetExhausted, Ledger, Store, check_resources, file_hash
from .schema import Plan, Task, digest, require

TERMINAL = {"COMPLETE", "EXHAUSTED", "ROLLED_BACK", "BLOCKED", "BOOTSTRAP_REQUIRED", "NO_SIGNAL"}


def validate_enrollment(plan: Plan, catalog: Catalog, capabilities: frozenset[str]) -> None:
    """Cheap data eligibility gate, callable before any model allocation."""
    pools = {split: catalog.subset(split, capabilities)
             for split in ("train", "calibration", "retention_train", "retention_eval")}
    require(all(pools.values()), "missing compatible train/calibration/replay/retention split")
    require(all(t.verifier == "exact-v1" for pool in pools.values() for t in pool),
            "bundled runtime only supports exact-v1")
    require(len({t.group for t in pools["calibration"]}) >= plan.gates.min_eval_groups,
            "too few independent calibration groups")
    for source in {t.source for t in pools["retention_eval"]}:
        require(len({t.group for t in pools["retention_eval"] if t.source == source}) >= plan.gates.min_eval_groups,
                f"too few independent retention groups: {source}")
    for stage in plan.stages:
        pool = unlocked(pools["train"], stage, plan.seed)
        require(len({t.group for t in pool}) >= plan.gates.min_train_groups,
                f"{stage.name}: too few unlocked training groups")
        require(all(weight == 0 or any(t.source == source for t in pool)
                    for source, weight in stage.source_weights.items()),
                f"{stage.name}: positive source has no eligible rows")


class Engine:
    def __init__(self, plan: Plan, catalog: Catalog, backend: Backend, directory: Path, *,
                 allow_training: bool = False):
        validate_enrollment(plan, catalog, backend.capabilities)
        self.plan, self.catalog, self.backend = plan, catalog, backend
        self.allow_training = allow_training
        self.identity = {"schema": 1, "plan": plan.fingerprint, "catalog": catalog.fingerprint,
                         "backend": backend.identity,
                         "source": digest({p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")})}
        self.store = Store(directory, self.identity)
        self.state = {"status": "NEW", "stage": 0, "updates": 0, "stage_updates": 0,
                      "version": backend.version, "collection": 0, "stats": {}, "events": [],
                      "baseline": None, "anchor": None, "passes": 0, "last_eval_version": -1,
                      "zero_windows": 0, "bootstrap_updates": 0, "evaluation_due": False, "signal": self._empty_signal(),
                      "last_good": None, "replay_fraction": plan.stages[0].replay_fraction}
        self.ledger = Ledger(plan.limits, self.state)
        self.train = catalog.subset("train", backend.capabilities)
        self.replay = catalog.subset("retention_train", backend.capabilities)
        self.calibration = catalog.subset("calibration", backend.capabilities)
        self.retention = catalog.subset("retention_eval", backend.capabilities)

    @staticmethod
    def _empty_signal() -> dict:
        return {"groups": 0, "mixed": 0, "episodes": 0, "valid": 0, "truncated": 0,
                "reward_sum": 0., "parents": [], "infrastructure_groups": 0}

    @property
    def stage(self):
        return self.plan.stages[self.state["stage"]]

    def event(self, event: str, **payload):
        self.state["events"].append({"event": event, "version": self.backend.version,
                                     "stage": self.stage.name, **payload})

    def _resource_guard(self):
        require(self.plan.fingerprint == self.identity["plan"], "plan was mutated during run")
        if self.backend.identity["kind"] != "synthetic":
            check_resources(self.plan.limits, self.store.root,
                            cuda=str(self.backend.identity.get("device", "cpu")).startswith("cuda"), preflight=False)

    def resume(self):
        self.state = self.store.restore(self.backend)
        self.ledger = Ledger(self.plan.limits, self.state)
        if self.state["pending"] is not None:
            # The checkpoint is PRE-operation: model/optimizer are restored, but work
            # may have happened. Charge the complete reservation and never auto-refund.
            self.ledger.settle()
            self.event("RECOVERED_INTERRUPTED_OPERATION", accounting="worst_case_charged")
            self.store.commit(self.state, self.backend)
        require(self.backend.version == self.state["version"], "policy/controller version mismatch")

    def _collect(self, tasks: list[Task], *, evaluation: bool = False):
        self._resource_guard()
        g, cap = self.stage.group_size, self.stage.max_new_tokens
        # Evaluation decoding remains frozen even after a rollout budget increase.
        if evaluation:
            g, cap = self.plan.stages[0].group_size, self.plan.stages[0].max_new_tokens
        calls = len(tasks) * g
        self.ledger.reserve(generated_tokens=calls * cap,
                            processed_tokens=2 * calls * (self.plan.limits.max_prompt_tokens + cap),
                            rollout_calls=calls)
        self.state["collection"] += 1
        self.store.commit(self.state, self.backend)
        groups, generated, processed = [], 0, 0
        for index, task in enumerate(tasks):
            seed_key = [self.plan.seed, "eval", task.key] if evaluation else [self.plan.seed, self.state["collection"], index]
            seed = int(digest(seed_key)[:15], 16)
            episodes = self.backend.collect(task.public(), g, cap, seed)
            require(len(episodes) == g, "incomplete rollout group")
            for episode in episodes:
                require(episode.key == task.key and episode.policy_version == self.backend.version,
                        "wrong task or stale policy rollout")
                require(len(episode.prompt_ids) <= self.plan.limits.max_prompt_tokens
                        and len(episode.response_ids) <= cap, "rollout exceeded reserved limits")
                generated += len(episode.response_ids)
                processed += 2 * (len(episode.prompt_ids) + len(episode.response_ids))
            if any(e.infrastructure_error for e in episodes):
                require(not evaluation, "evaluation infrastructure failure is not a model failure")
                self.state["signal"]["infrastructure_groups"] += 1
                continue
            rewards = [self.catalog.verify(task.key, e.text, e.terminated) if e.valid_action else 0.0 for e in episodes]
            groups.append((episodes, rewards))
            if not evaluation:
                signal = self.state["signal"]
                signal["groups"] += 1
                signal["mixed"] += int(max(rewards) > min(rewards))
                signal["episodes"] += len(episodes)
                signal["valid"] += sum(e.valid_action for e in episodes)
                signal["truncated"] += sum(not e.terminated for e in episodes)
                signal["reward_sum"] += sum(rewards)
                if task.group not in signal["parents"]:
                    signal["parents"].append(task.group)
                stat = TaskStats(**self.state["stats"].get(task.key, {}))
                # Refresh stale statistics rather than treating old competence as current.
                if self.backend.version - stat.version > self.plan.gates.stale_after_updates:
                    stat = TaskStats()
                stat.trials += len(rewards)
                stat.successes += sum(rewards)
                stat.version = self.backend.version
                self.state["stats"][task.key] = asdict(stat)
        self.ledger.settle({"generated_tokens": generated, "processed_tokens": processed, "rollout_calls": calls})
        self.store.commit(self.state, self.backend)
        return groups

    def _scores(self, tasks: list[Task]) -> dict[str, float]:
        groups = self._collect(tasks, evaluation=True)
        require(len(groups) == len(tasks), "incomplete evaluation panel")
        return {episodes[0].key: sum(rewards) / len(rewards) for episodes, rewards in groups}

    def _panels(self) -> dict:
        return {"calibration": self._scores(self.calibration),
                "retention": {source: self._scores([t for t in self.retention if t.source == source])
                              for source in sorted({t.source for t in self.retention})}}

    def _interval(self, before, after, tasks):
        return paired_interval(before, after, {t.key: t.group for t in tasks}, self.plan.gates, self.plan.seed)

    def _retention_intervals(self, panels):
        return {source: self._interval(self.state["baseline"]["retention"][source], scores,
                                        [t for t in self.retention if t.source == source])
                for source, scores in panels["retention"].items()}

    def _signal(self):
        s = self.state["signal"]
        return {"groups": len(s["parents"]), "mixed_fraction": s["mixed"] / max(1, s["groups"]),
                "valid_action_rate": s["valid"] / max(1, s["episodes"]),
                "truncation_rate": s["truncated"] / max(1, s["episodes"]),
                "mean_reward": s["reward_sum"] / max(1, s["episodes"])}

    def qualify(self) -> dict:
        require(self.state["status"] in {"NEW", "REQUALIFY"}, "qualification is not valid in this state")
        if self.state["baseline"] is None:
            self.state["baseline"] = self._panels()
            self.state["anchor"] = self.state["baseline"]["calibration"]
            self.store.commit(self.state, self.backend, good=True)
        else:
            panels = self._panels()
            retention = self._retention_intervals(panels)
            if any(x["lcb"] < -self.plan.gates.max_retention_drop for x in retention.values()):
                self._rollback("bootstrap_retention_gate")
                return self.report()
        self.state["signal"] = self._empty_signal()
        pool = unlocked(self.train, self.stage, self.plan.seed)
        # One representative per parent, selected deterministically and answer-unaware.
        representatives = {t.group: t for t in reversed(pool)}
        pool = sorted(representatives.values(), key=lambda t: digest([self.plan.seed, "qualification", t.key]))
        self._collect(pool[:self.plan.qualification_tasks])
        signal = self._signal()
        okay = (signal["groups"] >= self.plan.gates.min_train_groups
                and signal["mixed_fraction"] >= self.plan.gates.min_mixed_fraction
                and signal["valid_action_rate"] >= self.plan.gates.min_valid_action_rate
                and signal["truncation_rate"] <= self.plan.gates.max_truncation)
        self.state["status"] = "READY" if okay else ("BOOTSTRAP_REQUIRED" if signal["mean_reward"] < .2 else "NO_SIGNAL")
        self.event("QUALIFICATION", status=self.state["status"], signal=signal,
                   evidence="synthetic" if self.backend.identity["kind"] == "synthetic" else "native_inference_only")
        self.state["signal"] = self._empty_signal()
        self.store.commit(self.state, self.backend, good=okay)
        return self.report()

    def _optimize(self, groups, replay, *, sft=False):
        require(self.allow_training, "optimizer requires explicit training authorization")
        self._resource_guard()
        if sft:
            cost = 2 * len(replay) * self.plan.limits.max_context_tokens
        else:
            cost = 6 * sum(len(e.prompt_ids) + len(e.response_ids) for group, _ in groups for e in group)
            cost += 2 * len(replay) * self.plan.limits.max_context_tokens
        self.ledger.reserve(processed_tokens=cost, optimizer_steps=1)
        self.store.commit(self.state, self.backend)
        previous_version = self.backend.version
        metrics = self.backend.sft(replay) if sft else self.backend.update(groups, replay)
        require(self.backend.version == previous_version + 1, "backend did not advance exactly one policy version")
        self.ledger.settle()  # Conservative forward/backward accounting; no claimed FLOP measurement.
        self.state["version"] = self.backend.version
        self.state["updates"] += 1
        if sft:
            self.state["bootstrap_updates"] += 1
        else:
            self.state["stage_updates"] += 1
            self.state["evaluation_due"] = self.state["stage_updates"] % self.stage.eval_every == 0
        self.event("SFT_UPDATE" if sft else "RL_UPDATE", metrics=metrics)
        self.store.commit(self.state, self.backend)

    def _rollback(self, reason):
        version = self.backend.version
        require(self.state["last_good"] is not None, "no safe rollback checkpoint")
        self.store.restore(self.backend, self.state["last_good"])
        # Keep spent budgets and controller history. Never resurrect stale trajectories.
        self.backend.version = version + 1
        self.state.update(status="ROLLED_BACK", version=self.backend.version, passes=0, stats={}, evaluation_due=False)
        self.event("ROLLBACK", reason=reason, restored=self.state["last_good"])
        self.store.commit(self.state, self.backend)

    def evaluate(self):
        require(self.backend.version > self.state["last_eval_version"], "duplicate evaluation cannot promote")
        panels = self._panels()
        capability = self._interval(self.state["anchor"], panels["calibration"], self.calibration)
        retention = self._retention_intervals(panels)
        signal = self._signal()
        passed, reasons = gate(capability, retention, signal, self.plan.gates)
        if capability["mean"] <= 0:
            passed, reasons = False, [*reasons, "no_observed_capability_gain"]
        self.state["last_eval_version"] = self.backend.version
        self.state["evaluation_due"] = False
        self.state["signal"] = self._empty_signal()
        self.event("EVALUATION", capability=capability, retention=retention, signal=signal,
                   passed=passed, reasons=reasons)
        if any(x["mean"] < -self.plan.gates.max_retention_drop for x in retention.values()):
            self._rollback("retention_regression")
            return
        self.state["passes"] = self.state["passes"] + 1 if passed else 0
        if any(r.startswith("retention") for r in reasons):
            self.state["replay_fraction"] = min(.35, max(self.stage.replay_fraction, self.state["replay_fraction"] * 1.25))
        if self.state["passes"] >= self.plan.gates.consecutive_passes:
            self.event("PROMOTION", from_stage=self.stage.name)
            if self.state["stage"] + 1 == len(self.plan.stages):
                self.state["status"] = "COMPLETE"
            else:
                self.state["stage"] += 1
                self.state["stage_updates"] = 0
                self.state["passes"] = 0
                self.state["anchor"] = panels["calibration"]
                self.state["replay_fraction"] = self.stage.replay_fraction
                self.state["status"] = "READY"
            self.store.commit(self.state, self.backend, good=True)
        else:
            self.store.commit(self.state, self.backend, good=passed)

    def step(self):
        require(self.allow_training, "training is not authorized")
        require(self.state["status"] in {"READY", "TRAINING"}, "run is not trainable")
        if self.state["evaluation_due"]:
            self.evaluate()
            return
        if self.state["stage_updates"] >= self.stage.max_updates:
            self.state["status"] = "EXHAUSTED"
            self.event("STOP", reason="stage_update_limit")
            self.store.commit(self.state, self.backend)
            return
        self.state["status"] = "TRAINING"
        pool = unlocked(self.train, self.stage, self.plan.seed)
        selected, mixture = batch(pool, self.replay, {k: TaskStats(**v) for k, v in self.state["stats"].items()},
                                  self.stage, self.plan.gates, self.backend.version,
                                  self.plan.seed + self.state["collection"], self.state["replay_fraction"])
        acquisition = [t for t in selected if t.split == "train"]
        # Replay CE still supplies a preservation gradient when all replay rewards are 1.
        rehearsal = [(t, t.answer) for t in selected if t.split == "retention_train"]
        self.event("MIXTURE", **mixture)
        groups = self._collect(acquisition)
        usable = [(episodes, rewards) for episodes, rewards in groups
                  if all(e.terminated for e in episodes) and max(rewards) > min(rewards)]
        self.event("ROLLOUT_FILTER", collected_groups=len(groups), usable_groups=len(usable),
                   dropped_truncated=sum(any(not e.terminated for e in episodes) for episodes, _ in groups))
        if not usable:
            self.state["zero_windows"] += 1
            if self.state["zero_windows"] >= self.plan.gates.zero_signal_patience:
                self.state["status"] = "BOOTSTRAP_REQUIRED" if self._signal()["mean_reward"] < .2 else "NO_SIGNAL"
                self.event("STOP", reason="no_on_policy_reward_signal")
            self.store.commit(self.state, self.backend)
            return
        self.state["zero_windows"] = 0
        self._optimize(usable, rehearsal)
        if self.state["evaluation_due"]:
            self.evaluate()
        self.store.commit(self.state, self.backend)

    def run(self, max_actions: int | None = None):
        require(self.allow_training, "explicit training authorization required")
        try:
            if self.state["status"] in {"NEW", "REQUALIFY"}:
                self.qualify()
            actions = 0
            while self.state["status"] not in TERMINAL and (max_actions is None or actions < max_actions):
                self.step()
                actions += 1
        except BudgetExhausted as error:
            self.state["status"] = "EXHAUSTED"
            self.event("STOP", reason=str(error))
            self.store.commit(self.state, self.backend)
        return self.report()

    def bootstrap(self, demonstrations):
        require(self.allow_training and self.state["status"] == "BOOTSTRAP_REQUIRED", "bootstrap not authorized/state mismatch")
        eligible = {t.key for t in unlocked(self.train, self.stage, self.plan.seed)}
        require(all(t.key in eligible and t.split == "train" and self.catalog.verify(t.key, response)
                    for t, response in demonstrations), "bootstrap contains locked/unverified/nontraining task")
        rng = random.Random(self.plan.seed + self.state["bootstrap_updates"])
        while self.state["bootstrap_updates"] < self.plan.bootstrap_max_steps:
            rows = rng.choices(demonstrations, k=min(self.stage.batch_tasks, len(demonstrations)))
            # Include a fixed replay contribution during bootstrap too.
            replay_task = self.replay[self.state["bootstrap_updates"] % len(self.replay)]
            rows.append((replay_task, replay_task.answer))
            self._optimize([], rows, sft=True)
        self.state["status"] = "REQUALIFY"
        self.state["zero_windows"] = 0
        self.store.commit(self.state, self.backend)
        return self.qualify()

    def report(self):
        return {"experiment": "PRLC-001", "status": self.state["status"], "stage": self.stage.name,
                "policy_version": self.backend.version, "optimizer_updates": self.state["updates"],
                "budgets_used": self.state["used"], "identity": self.identity,
                "native_model_run": self.backend.identity["kind"] != "synthetic",
                "last_events": self.state["events"][-5:]}
