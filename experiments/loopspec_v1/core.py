"""Independent implementation of arXiv:2609.17184, Eqs. 2--6 / Algorithm 1.

No model loading, training, network access, or global random-state mutation.
Probabilities are processed BEFORE drafting and verification. Sampling preserves
an output law, not the token sequence of a different sampler using the same seed.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Protocol, Sequence

PMF = tuple[float, ...]


def pmf(values: Sequence[float]) -> PMF:
    p = tuple(float(x) for x in values)
    if not p or any(not math.isfinite(x) or x < 0 for x in p):
        raise ValueError("PMF must be finite, nonnegative, and nonempty")
    if not math.isclose(math.fsum(p), 1.0, rel_tol=0, abs_tol=1e-10):
        raise ValueError("PMF must sum to one")
    return p


def residual(p: PMF, q: PMF) -> PMF | None:
    p, q = pmf(p), pmf(q)
    if len(p) != len(q):
        raise ValueError("vocabulary mismatch")
    mass = tuple(max(a - b, 0.0) for a, b in zip(p, q))
    z = math.fsum(mass)
    # Never insert epsilon mass or silently fall back to an unrelated law.
    return tuple(x / z for x in mass) if z > 0 else None


def draw(p: PMF, rng: random.Random) -> int:
    p = pmf(p)
    u = rng.random()
    if not 0 <= u < 1:
        raise ValueError("random draw must be in [0,1)")
    cumulative = 0.0
    for i, value in enumerate(p):
        cumulative += value
        if u < cumulative:
            return i
    return max(i for i, value in enumerate(p) if value > 0)


@dataclass(frozen=True)
class Sampling:
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0

    def __post_init__(self):
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("invalid temperature")
        if type(self.top_k) is not int or self.top_k < 0:
            raise ValueError("invalid top_k")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError("invalid top_p")

    def distribution(self, logits: Sequence[float]) -> PMF:
        z = [float(x) for x in logits]
        if not z or any(math.isnan(x) or x == math.inf for x in z):
            raise ValueError("invalid logits")
        if max(z) == -math.inf:
            raise ValueError("all tokens masked")
        order = sorted(range(len(z)), key=lambda i: (-z[i], i))
        if self.temperature == 0:
            return tuple(float(i == order[0]) for i in range(len(z)))
        retained = order[:self.top_k] if self.top_k else order
        weights = [0.0] * len(z)
        for i in retained:
            weights[i] = math.exp((z[i] - z[order[0]]) / self.temperature)
        total = math.fsum(weights)
        weights = [w / total for w in weights]
        if self.top_p < 1:
            cumulative, keep = 0.0, set()
            for i in retained:
                keep.add(i)
                cumulative += weights[i]
                if cumulative >= self.top_p:
                    break
            weights = [w if i in keep else 0.0 for i, w in enumerate(weights)]
            total = math.fsum(weights)
            weights = [w / total for w in weights]
        return pmf(weights)


@dataclass(frozen=True)
class Proposal:
    token: int
    distribution: PMF
    child: int | None = None

    def __post_init__(self):
        object.__setattr__(self, "distribution", pmf(self.distribution))
        if type(self.token) is not int or not 0 <= self.token < len(self.distribution):
            raise ValueError("invalid proposal token")
        if self.distribution[self.token] <= 0:
            raise ValueError("proposal token has no support")


def verify(target: PMF, proposals: Sequence[Proposal], rng: random.Random) -> tuple[int, int | None]:
    """Return (token, accepted proposal index); None means residual resampling."""
    rho = pmf(target)
    for j, candidate in enumerate(proposals):
        q, token = candidate.distribution, candidate.token
        if len(rho) != len(q):
            raise ValueError("vocabulary mismatch")
        u = rng.random()
        if not 0 <= u < 1:
            raise ValueError("random draw must be in [0,1)")
        if u < min(1.0, rho[token] / q[token]):
            return token, j
        rho = residual(rho, q)
        if rho is None:
            raise ArithmeticError("reached an impossible zero-mass rejection")
    return draw(rho, rng), None


def fallback_distribution(first: Proposal, deeper: PMF, gated: bool) -> PMF | None:
    deeper = pmf(deeper)
    if len(deeper) != len(first.distribution):
        raise ValueError("vocabulary mismatch")
    if gated and deeper[first.token] >= first.distribution[first.token]:
        return None
    return residual(deeper, first.distribution)


class Backend(Protocol):
    """Every state owns one token's caches; its ancestors are immutable history.

    advance() must batch the SAME shared block across mixed recurrent depths.
    readout() must include the full native coda, normalization, and logit biases,
    with coda caches isolated by readout depth. No approximate target is allowed.
    A native implementation also needs independent full-prefix qualification.
    """
    total_depth: int
    vocab_size: int
    max_context: int

    def prefill(self, prompt: tuple[int, ...], readout_depths: tuple[int, ...]) -> tuple[Any, Sequence[float]]: ...
    def start(self, parent: Any, token: int) -> Any: ...
    def advance(self, states: Sequence[Any]) -> None: ...
    def readout(self, states: Sequence[Any]) -> list[Sequence[float]]: ...
    def cache_bytes(self, states: Sequence[Any]) -> int: ...


@dataclass(frozen=True)
class Plan:
    first: int | None = 1
    second: int | None = 2
    gated: bool = True
    max_branches: int = 8
    max_new_tokens: int = 32
    eos_ids: tuple[int, ...] = ()
    sampling: Sampling = field(default_factory=Sampling)
    seed: int = 0

    def validate(self, model: Backend, prompt: tuple[int, ...]):
        r = model.total_depth
        if type(self.gated) is not bool or not isinstance(self.sampling, Sampling):
            raise ValueError("invalid gating or sampling configuration")
        if not isinstance(self.eos_ids, tuple):
            raise ValueError("eos_ids must be a tuple of token IDs")
        if any(type(n) is not int or n < 1 for n in (model.vocab_size, model.max_context)):
            raise ValueError("invalid backend dimensions")
        if type(r) is not int or r < 1:
            raise ValueError("invalid target recurrent depth")
        for n in (self.max_branches, self.max_new_tokens, self.seed):
            if type(n) is not int:
                raise ValueError("integer configuration required")
        if self.max_branches < 1 or self.max_new_tokens < 0:
            raise ValueError("invalid resource bounds")
        if self.first is None:
            if self.second is not None:
                raise ValueError("second proposal requires first")
        else:
            if type(self.first) is not int or not 1 <= self.first < r or r % self.first:
                raise ValueError("first depth must be below R and divide R")
            if self.second is not None and (
                type(self.second) is not int or not self.first < self.second < r
                or self.second % self.first
            ):
                raise ValueError("invalid second proposal depth")
        if not prompt or any(type(x) is not int or not 0 <= x < model.vocab_size for x in prompt + self.eos_ids):
            raise ValueError("nonempty prompt and valid token IDs required")
        # The last emitted token need not itself be fed back into the model.
        if len(prompt) + max(0, self.max_new_tokens - 1) > model.max_context:
            raise ValueError("context budget exceeded; truncation is forbidden")


@dataclass
class Branch:
    state: Any
    prefix: tuple[int, ...]
    parent: int | None
    proposals: list[Proposal] = field(default_factory=list)


@dataclass
class Result:
    tokens: list[int]
    stop_reason: str
    stats: dict[str, Any]
    commits: list[dict[str, Any]]


def decode(model: Backend, prompt: Sequence[int], plan: Plan) -> Result:
    """Batch-one request, bounded speculative tree; no native speed claim.

    A cap suppresses only child computation, never truncates/renormalizes a
    proposal after sampling. Accepted candidates without a child restart safely.
    EOS is actionable only after authoritative verification.
    """
    prompt = tuple(prompt)
    plan.validate(model, prompt)
    stats = dict(recurrent_batches=0, branch_steps=0, readouts=0,
                 peak_active=0, peak_logical_kv_bytes=0, suppressed_children=0,
                 primary=0, secondary=0, residual=0, restarts=0,
                 primary_proposals=0, secondary_proposals=0, second_readouts=0,
                 prefill_seconds=0.0, decode_seconds=0.0)
    if plan.max_new_tokens == 0:
        return Result([], "length", stats, [])
    depths = tuple(sorted({model.total_depth} | {d for d in (plan.first, plan.second) if d is not None}))
    draft_rng, verify_rng = random.Random(plan.seed ^ 0x4C53504543), random.Random(plan.seed)
    start_time = perf_counter()
    state, logits = model.prefill(prompt, depths)
    if len(logits) != model.vocab_size:
        raise ValueError("prefill vocabulary mismatch")
    output = [draw(plan.sampling.distribution(logits), verify_rng)]
    stats["prefill_seconds"] = perf_counter() - start_time
    stats["peak_logical_kv_bytes"] = model.cache_bytes([state])
    commits = [dict(source="prefill", token=output[0], retained_depth=0)]
    active: dict[int, Branch] = {}
    next_id = 0

    def stopped() -> str | None:
        if output[-1] in plan.eos_ids:
            return "eos"
        return "length" if len(output) >= plan.max_new_tokens else None

    def spawn(parent_state: Any, token: int, prefix: tuple[int, ...], parent_id: int | None) -> int:
        nonlocal next_id
        new_id, next_id = next_id, next_id + 1
        active[new_id] = Branch(model.start(parent_state, token), prefix + (token,), parent_id)
        return new_id

    def propose(branch_id: int, q: PMF):
        b = active[branch_id]
        token = draw(q, draft_rng)
        child = None
        useful = token not in plan.eos_ids and len(b.prefix) - len(prompt) + 1 < plan.max_new_tokens
        if useful and len(active) < plan.max_branches:
            child = spawn(b.state, token, b.prefix, branch_id)
        elif useful:
            stats["suppressed_children"] += 1
        stats["primary_proposals" if not b.proposals else "secondary_proposals"] += 1
        b.proposals.append(Proposal(token, q, child))

    def measure():
        stats["peak_active"] = max(stats["peak_active"], len(active))
        stats["peak_logical_kv_bytes"] = max(stats["peak_logical_kv_bytes"], model.cache_bytes([b.state for b in active.values()]))
        if len(active) > plan.max_branches:
            raise AssertionError("branch cap violated")

    tick_start = perf_counter()
    reason = stopped()
    if reason:
        return Result(output, reason, stats, commits)
    root = spawn(state, output[0], prompt, None)
    try:
        while not reason:
            measure()
            snapshot = list(active)
            old_depths = [active[i].state.depth for i in snapshot]
            model.advance([active[i].state for i in snapshot])
            if any(active[i].state.depth != old + 1 for i, old in zip(snapshot, old_depths)):
                raise ValueError("backend must advance every state by exactly one loop")
            stats["recurrent_batches"] += 1
            stats["branch_steps"] += len(snapshot)
            ready = [i for i in snapshot if active[i].state.depth in depths]
            readouts = model.readout([active[i].state for i in ready]) if ready else []
            if len(readouts) != len(ready) or any(len(z) != model.vocab_size for z in readouts):
                raise ValueError("backend readout batch/vocabulary mismatch")
            stats["readouts"] += len(ready)
            target = None
            for i, z in zip(ready, readouts):
                b = active[i]
                q = plan.sampling.distribution(z)
                if b.state.depth == model.total_depth:
                    if i != root:
                        raise AssertionError("unverified descendant reached commitment")
                    target = q
                elif b.state.depth == plan.first:
                    propose(i, q)
                elif b.state.depth == plan.second:
                    stats["second_readouts"] += 1
                    q2 = fallback_distribution(b.proposals[0], q, plan.gated)
                    if q2 is not None:
                        propose(i, q2)
            measure()
            if target is None:
                continue
            b = active[root]
            if b.prefix != prompt + tuple(output):
                raise AssertionError("target is not conditioned on committed prefix")
            token, accepted = verify(target, b.proposals, verify_rng)
            source = ("primary" if accepted == 0 else "secondary") if accepted is not None else "residual"
            stats[source] += 1
            chosen = b.proposals[accepted].child if accepted is not None else None
            retained = active[chosen].state.depth if chosen is not None else 0
            output.append(token)
            commits.append(dict(source=source, token=token, retained_depth=retained))
            reason = stopped()
            if reason:
                break
            if chosen is None:
                active.clear()
                root = spawn(b.state, token, b.prefix, None)
                stats["restarts"] += 1
            else:
                keep = {chosen}
                for i in list(active):
                    p = active[i].parent
                    while p is not None:
                        if p == chosen:
                            keep.add(i)
                            break
                        p = active[p].parent
                active = {i: branch for i, branch in active.items() if i in keep}
                root = chosen
                active[root].parent = None
            # Progress cannot require more recurrent ticks than serial decoding.
            if stats["recurrent_batches"] > model.total_depth * (plan.max_new_tokens - 1):
                raise AssertionError("scheduler failed to make bounded progress")
    finally:
        active.clear()  # Never clear caches owned by surviving ancestor references.
    stats["decode_seconds"] = perf_counter() - tick_start
    return Result(output, reason, stats, commits)
