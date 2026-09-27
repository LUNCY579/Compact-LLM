"""Prefix-only, deterministic supported-history replay; no model or network calls.

Section-3 parent actions: root opens the next recorded branch; an observed leaf
continues its recorded chain. Missing continuations are explicit support misses,
NOT negative native measurements. Policies receive frozen value objects only.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Protocol

from .schema import Outcome, World, fields, integer, number, require


@dataclass(frozen=True)
class Plan:
    budget: int = 12
    workers: int = 1
    max_rounds: int = 24
    retention_floor: float = 0.5
    max_parameter_delta: int = 0
    max_latency_ratio: float = 1.25
    max_host_gib: float = 8.0
    max_vram_gib: float = 8.0
    paper_beta_work: float = 0.01
    paper_beta_parallel: float = 0.0
    seeds: tuple[int, ...] = (17, 29, 43)
    generations: int = 6
    mutations: int = 6
    search_seed: int = 20260926

    @classmethod
    def parse(cls, raw: dict) -> Plan:
        fields(raw, "budget workers max_rounds retention_floor max_parameter_delta max_latency_ratio "
                    "max_host_gib max_vram_gib paper_beta_work paper_beta_parallel seeds generations "
                    "mutations search_seed")
        for key in ("budget", "workers", "max_rounds", "generations", "mutations"):
            integer(raw[key], key, 1, 10000)
        require(raw["workers"] <= 64, "worker cap is 64")
        require(raw["generations"] * raw["mutations"] <= 4096, "search ceiling is 4096 proposals")
        integer(raw["max_parameter_delta"], "max_parameter_delta", -10**12, 10**12)
        integer(raw["search_seed"], "search_seed", 0, 2**32 - 1)
        number(raw["retention_floor"], "retention_floor", 0, 1)
        for key in ("max_latency_ratio", "max_host_gib", "max_vram_gib", "paper_beta_work", "paper_beta_parallel"):
            number(raw[key], key)
        require(type(raw["seeds"]) in (tuple, list) and 1 <= len(raw["seeds"]) <= 32, "empty seeds")
        for seed in raw["seeds"]:
            integer(seed, "seed", 0, 2**32 - 1)
        require(len(set(raw["seeds"])) == len(raw["seeds"]), "duplicate seeds")
        return cls(**{**raw, "seeds": tuple(raw["seeds"])})


def gate(outcome: Outcome, plan: Plan) -> bool:
    """Separate correctness/qualification from measured but uncompetitive quality."""
    return (outcome.status == "measured" and outcome.qualified and outcome.protocol_valid
            and outcome.retention is not None and outcome.retention >= plan.retention_floor
            and outcome.capability is not None
            and outcome.parameter_delta is not None and outcome.parameter_delta <= plan.max_parameter_delta
            and outcome.latency_ratio is not None and outcome.latency_ratio <= plan.max_latency_ratio
            and outcome.peak_host_gib is not None and outcome.peak_host_gib <= plan.max_host_gib
            and outcome.peak_vram_gib is not None and outcome.peak_vram_gib <= plan.max_vram_gib)


def frontier(points: list[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    """Maximization frontier, sorted by capability. Inputs already normalized [0,1]."""
    for point in points:
        require(len(point) == 2, "expected two metrics")
        for value in point:
            number(value, "Pareto coordinate", 0, 1)
    result = []
    best_y = -1.0
    for x, y in sorted(set(points), reverse=True):
        if y > best_y:
            result.append((x, y))
            best_y = y
    return tuple(reversed(result))


def hypervolume(points: list[tuple[float, float]]) -> float:
    """Exact 2-D union of rectangles from fixed reference (0,0), no fitted maxima."""
    result, last_x = 0.0, 0.0
    for x, y in frontier(points):
        result += (x - last_x) * y
        last_x = x
    return result


@dataclass(frozen=True)
class Observation:
    handle: str
    parent: str | None
    depth: int
    outcome: Outcome
    admissible: bool


@dataclass(frozen=True)
class View:
    observations: tuple[Observation, ...]
    legal: tuple[str, ...]
    remaining: int
    workers: int
    rounds: int
    stagnation: int


@dataclass(frozen=True)
class Decision:
    batch: tuple[str, ...]
    reason: str


class Policy(Protocol):
    def choose(self, view: View) -> Decision: ...


@dataclass(frozen=True)
class Spec:
    method: str = "adaptive"
    explore: float = 0.25
    trend: float = 0.5
    repair: float = 0.25
    depth_penalty: float = 0.02
    width: int = 1
    patience: int = 12
    min_depth: int = 2

    @classmethod
    def parse(cls, raw: dict) -> Spec:
        fields(raw, "method explore trend repair depth_penalty width patience min_depth")
        require(raw["method"] in ("fixed", "greedy", "random", "adaptive"), "unregistered policy")
        for key in ("explore", "trend", "repair", "depth_penalty"):
            number(raw[key], key, 0, 2)
        integer(raw["width"], "width", 1, 64)
        integer(raw["patience"], "patience", 1, 10000)
        integer(raw["min_depth"], "min_depth", 1, 100)
        return cls(**raw)


class Controller:
    """Allowlisted policy DSL, not arbitrary generated Python or an LLM sandbox."""
    def __init__(self, spec: Spec, seed: int):
        self.spec = Spec.parse(asdict(spec))
        self.rng = random.Random(seed)
        self.visits: dict[str, int] = {}

    def choose(self, view: View) -> Decision:
        if not view.legal or view.remaining <= 0:
            return Decision((), "no eligible prefix actions or attempt budget")
        spec = self.spec
        observed = {o.handle: o for o in view.observations}
        width = min(view.workers, view.remaining, spec.width)
        if spec.method == "random":
            actions = list(view.legal)
            self.rng.shuffle(actions)
            return Decision(tuple(actions[:width]), "seeded random prefix-legal actions")
        if spec.method == "fixed":
            # Fixed least-visited/shallowest schedule, independent of outcomes.
            actions = sorted(view.legal, key=lambda h: (self.visits.get(h, 0), observed[h].depth, h))
        else:
            def chain(handle: str) -> list[Observation]:
                history = []
                while handle is not None:
                    node = observed[handle]
                    history.append(node)
                    handle = node.parent
                return list(reversed(history))

            def quality(node: Observation) -> float:
                out = node.outcome
                return out.capability * out.retention if node.admissible else 0.0

            def priority(handle: str) -> float:
                history = chain(handle)
                node = history[-1]
                valid = [quality(o) for o in history if o.admissible]
                anchor = max(valid, default=0.0)
                if spec.method == "greedy":
                    return anchor
                if handle == "root":
                    return anchor + spec.explore
                gain = valid[-1] - valid[-2] if len(valid) >= 2 else 0.0
                failure_bonus = spec.repair if node.outcome.failure == "implementation" else 0.0
                shallow_bonus = spec.explore if node.depth < spec.min_depth else 0.0
                return anchor + spec.trend * gain + failure_bonus + shallow_bonus - spec.depth_penalty * node.depth

            leaves = [observed[h] for h in view.legal if h != "root"]
            if (spec.method == "adaptive" and view.stagnation >= spec.patience and leaves
                    and all(o.depth >= spec.min_depth for o in leaves)
                    and all(o.outcome.failure != "implementation" for o in leaves)):
                return Decision((), "prefix-wide stagnation after minimum exploration depth")
            actions = sorted(view.legal, key=lambda h: (-priority(h), self.visits.get(h, 0), h))
        chosen = tuple(actions[:width])
        for handle in chosen:
            self.visits[handle] = self.visits.get(handle, 0) + 1
        return Decision(chosen, f"{spec.method}: prefix-only ranking; width={width}")


class Replay:
    """Environment owns hidden records. Never pass this object to policy.choose."""
    def __init__(self, world: World, plan: Plan):
        self.world = world
        self.plan = Plan.parse(asdict(plan))
        require(gate(world.baseline, plan), "baseline fails registered resource/retention gates")
        self.children: dict[str, list] = {}
        for node in world.nodes:
            self.children.setdefault(node.parent, []).append(node)
        self.hidden_to_handle = {"root": "root"}
        self.handle_to_hidden = {"root": "root"}
        self.observed = [Observation("root", None, 0, world.baseline, True)]
        self.exhausted: set[str] = set()
        self.spent = self.rounds = self.stagnation = self.misses = 0
        self.points = [(world.baseline.capability, world.baseline.retention)]
        self.hv = hypervolume(self.points)
        self.area = 0.0
        self.gpu = self.wall = 0.0
        self.gpu_complete = self.wall_complete = True
        self.trace: list[dict] = []
        self.curve = [self._point()]

    def _point(self) -> dict:
        return {"attempts": self.spent, "rounds": self.rounds, "hypervolume": self.hv,
                "gpu_seconds": self.gpu if self.gpu_complete else None,
                "wall_seconds": self.wall if self.wall_complete else None}

    def view(self) -> View:
        internal = {o.parent for o in self.observed[1:]}
        legal = tuple(o.handle for o in self.observed
                      if (o.handle == "root" or o.handle not in internal) and o.handle not in self.exhausted)
        return View(tuple(self.observed), legal, self.plan.budget - self.spent,
                    self.plan.workers, self.rounds, self.stagnation)

    def step(self, decision: Decision) -> bool:
        """Atomic batch: validate everything before revealing ANY outcome."""
        require(isinstance(decision, Decision), "policy must return Decision")
        view = self.view()
        batch = decision.batch
        require(type(batch) is tuple and all(type(h) is str for h in batch), "batch must be tuple of handles")
        require(isinstance(decision.reason, str) and bool(decision.reason.strip()), "decision needs a reason")
        require(len(batch) == len(set(batch)), "duplicate batch action")
        require(len(batch) <= min(view.workers, view.remaining), "batch exceeds workers/budget")
        require(all(h in view.legal for h in batch), "illegal or unrevealed action")
        require(self.rounds < self.plan.max_rounds, "round budget exhausted")
        if not batch:
            return False
        old_hv = self.hv
        self.area += old_hv * len(batch)
        revealed, unsupported, gpu_costs, wall_costs = [], [], [], []
        for handle in batch:
            hidden = self.handle_to_hidden[handle]
            candidates = [n for n in self.children.get(hidden, ()) if n.key not in self.hidden_to_handle]
            if not candidates:
                self.exhausted.add(handle)
                self.misses += 1
                unsupported.append(handle)
                # An unsupported counterfactual has no known native runtime.
                self.gpu_complete = self.wall_complete = False
                continue
            node = candidates[0]
            fresh = f"n{len(self.observed):05d}"  # reveal-order aliases, never source IDs
            parent = next(o for o in self.observed if o.handle == handle)
            observation = Observation(fresh, handle, parent.depth + 1, node.outcome, gate(node.outcome, self.plan))
            self.hidden_to_handle[node.key] = fresh
            self.handle_to_hidden[fresh] = node.key
            self.observed.append(observation)
            revealed.append(asdict(observation))
            if observation.admissible:
                self.points.append((node.outcome.capability, node.outcome.retention))
            if node.outcome.gpu_seconds is None:
                self.gpu_complete = False
            else:
                gpu_costs.append(node.outcome.gpu_seconds)
            if node.outcome.wall_seconds is None:
                self.wall_complete = False
            else:
                wall_costs.append(node.outcome.wall_seconds)
        self.gpu += sum(gpu_costs)
        # Serialized Rineya default W=1; W>1 is an idealized synchronized replay.
        self.wall += max(wall_costs, default=0.0)
        self.spent += len(batch)
        self.rounds += 1
        self.hv = hypervolume(self.points)
        self.stagnation = 0 if self.hv > old_hv + 1e-15 else self.stagnation + 1
        self.trace.append({"round": self.rounds, "prefix": asdict(view), "decision": asdict(decision),
                           "revealed": revealed, "unsupported": unsupported,
                           "point": self._point()})
        self.curve.append(self._point())
        return True

    def run(self, policy: Policy) -> dict:
        stop = "attempt_budget"
        while self.spent < self.plan.budget:
            if self.rounds >= self.plan.max_rounds:
                stop = "round_budget"
                break
            view = self.view()
            if not view.legal:
                stop = "observed_support_exhausted"
                break
            decision = policy.choose(view)
            if not self.step(decision):
                stop = "policy_stop"
                self.trace.append({"round": self.rounds, "prefix": asdict(view),
                                   "decision": asdict(decision), "revealed": [], "unsupported": []})
                break
        auc = (self.area + self.hv * (self.plan.budget - self.spent)) / self.plan.budget
        valid = [o for o in self.observed if o.admissible]
        best = max(o.outcome.capability for o in valid)
        represented = len(self.observed) - 1
        paper_score = (best - self.plan.paper_beta_work * represented
                       + self.plan.paper_beta_parallel * represented / max(1, self.rounds))
        return {"world_id": self.world.key, "kind": self.world.kind, "world_hash": self.world.content_sha256,
                "attempts": self.spent, "revealed": represented, "support_misses": self.misses,
                "rounds": self.rounds, "stop": stop, "auc": auc,
                "auc_gain": auc - self.curve[0]["hypervolume"], "hypervolume": self.hv,
                "best_valid_capability": best, "pareto": frontier(self.points),
                "paper_score_diagnostic": paper_score, "gpu_seconds": self.gpu if self.gpu_complete else None,
                "wall_seconds": self.wall if self.wall_complete else None,
                "cost_scope": "recorded-or-unknown; not live GPU savings",
                "curve": self.curve, "trace": self.trace}


def evaluate(world: World, spec: Spec, plan: Plan, seed: int) -> dict:
    return Replay(world, plan).run(Controller(spec, seed))
