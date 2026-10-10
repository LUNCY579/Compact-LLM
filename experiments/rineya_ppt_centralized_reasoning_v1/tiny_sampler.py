"""Exact, enumerable CPU apparatus for PPT-001; not a native model adapter.

The two-slot horizon is deliberately fixed. Integer powers keep every target,
proposal and transition rational. EOS is scored once; subsequent PAD has mass
one. Restart is uniform over *horizon positions*, including positions after EOS.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction as F
from itertools import product
from math import lcm
from random import Random
from types import MappingProxyType

HORIZON = 2
TOKENS = ("A", "B", "EOS")
PAD = "PAD"
Record = tuple[str, ...]


def _power(power: int) -> None:
    if type(power) is not int or power < 1:
        raise ValueError("the exact fixture requires a positive integer power")


def _restart(restart: int) -> None:
    if type(restart) is not int or not 0 <= restart < HORIZON:
        raise ValueError("restart must be a horizon position")


def _draw(distribution: dict, rng: Random):
    """Draw rational masses with integer arithmetic, without float rounding."""
    if not distribution or any(p < 0 for p in distribution.values()):
        raise ValueError("invalid probability distribution")
    if sum(distribution.values(), F(0)) != 1:
        raise ValueError("probability distribution must sum to one")
    denominator = lcm(*(p.denominator for p in distribution.values()))
    ticket = rng.randrange(denominator)
    for value, mass in distribution.items():
        ticket -= mass.numerator * (denominator // mass.denominator)
        if ticket < 0:
            return value
    raise AssertionError("unreachable rational draw")


class TinyModel:
    """Supplied prefix-dependent probability table, with seven canonical states."""

    table = MappingProxyType({
        (): MappingProxyType({"A": F(1, 2), "B": F(1, 3), "EOS": F(1, 6)}),
        ("A",): MappingProxyType({"A": F(1, 5), "B": F(1, 2), "EOS": F(3, 10)}),
        ("B",): MappingProxyType({"A": F(1, 2), "B": F(1, 8), "EOS": F(3, 8)}),
    })

    def __init__(self) -> None:
        self.states = tuple((first, second)
                            for first in TOKENS
                            for second in ((PAD,) if first == "EOS" else TOKENS))

    def validate(self, record: Record) -> None:
        if not isinstance(record, tuple) or record not in self.states:
            raise ValueError("record must have two canonical EOS/PAD horizon slots")

    def realized_length(self, record: Record) -> int:
        self.validate(record)
        return record.index("EOS") + 1 if "EOS" in record else HORIZON

    def probability(self, record: Record) -> F:
        self.validate(record)
        score = F(1)
        for index, token in enumerate(record):
            if token == PAD:
                break
            score *= self.table[record[:index]][token]
        return score

    def restart_distribution(self, record: Record) -> tuple[F, ...]:
        self.validate(record)
        return (F(1, HORIZON),) * HORIZON

    def proposal_probability(self, record: Record, power: int, restart: int) -> F:
        self.validate(record)
        _power(power)
        _restart(restart)
        score = F(1)
        for index in range(restart, HORIZON):
            token = record[index]
            if token == PAD:
                break
            row = self.table[record[:index]]
            score *= row[token] ** power / sum((p ** power for p in row.values()), F(0))
        return score

    def suffix_distribution(self, record: Record, power: int, restart: int) -> dict[Record, F]:
        self.validate(record)
        _power(power)
        _restart(restart)
        return {candidate: self.proposal_probability(candidate, power, restart)
                for candidate in self.states if candidate[:restart] == record[:restart]}

    def target(self, power: int) -> dict[Record, F]:
        _power(power)
        weights = {record: self.probability(record) ** power for record in self.states}
        normalizer = sum(weights.values(), F(0))
        return {record: weight / normalizer for record, weight in weights.items()}


@dataclass(frozen=True)
class LocalEvent:
    before: Record
    proposal: Record
    after: Record
    power: int
    restart: int
    forward: F
    reverse: F
    acceptance: F
    accepted: bool
    token_draws: int


class TinySampler:
    def __init__(self, model: TinyModel, power: int) -> None:
        _power(power)
        self.model, self.power = model, power

    def local_acceptance(self, old: Record, new: Record, restart: int) -> F:
        self.model.validate(old)
        self.model.validate(new)
        _restart(restart)
        if old[:restart] != new[:restart]:
            return F(0)
        forward = self.model.proposal_probability(new, self.power, restart)
        reverse = self.model.proposal_probability(old, self.power, restart)
        return min(F(1), (self.model.probability(new) / self.model.probability(old)) ** self.power
                   * reverse / forward)

    def local_kernel(self, restart: int | None = None) -> dict:
        if restart is not None:
            _restart(restart)
        kernel = {}
        for old in self.model.states:
            row = {new: F(0) for new in self.model.states}
            restarts = ((restart, F(1)),) if restart is not None else enumerate(
                self.model.restart_distribution(old))
            for position, position_mass in restarts:
                for new, proposal_mass in self.model.suffix_distribution(old, self.power, position).items():
                    acceptance = self.local_acceptance(old, new, position)
                    row[new] += position_mass * proposal_mass * acceptance
                    row[old] += position_mass * proposal_mass * (1 - acceptance)
            kernel[old] = row
        return kernel

    def step(self, record: Record, rng: Random, restart: int | None = None) -> LocalEvent:
        self.model.validate(record)
        if restart is None:
            restart = _draw(dict(enumerate(self.model.restart_distribution(record))), rng)
        _restart(restart)
        # Enumerate/draw the whole suffix; this is a tiny CPU oracle apparatus,
        # not a performance model of autoregressive generation or native caches.
        proposal = _draw(self.model.suffix_distribution(record, self.power, restart), rng)
        forward = self.model.proposal_probability(proposal, self.power, restart)
        reverse = self.model.proposal_probability(record, self.power, restart)
        acceptance = self.local_acceptance(record, proposal, restart)
        accepted = _draw({True: acceptance, False: 1 - acceptance}, rng)
        draws = sum(token != PAD for token in proposal[restart:])
        return LocalEvent(record, proposal, proposal if accepted else record, self.power,
                          restart, forward, reverse, acceptance, accepted, draws)


@dataclass(frozen=True)
class Replica:
    power: int
    record: Record
    base_probability: F
    proposal_probability: F


def make_replica(model: TinyModel, power: int, record: Record) -> Replica:
    return Replica(power, record, model.probability(record),
                   model.proposal_probability(record, power, 0))


def swap_acceptance(model: TinyModel, left_record: Record, right_record: Record,
                    left_power: int, right_power: int) -> F:
    _power(left_power)
    _power(right_power)
    return min(F(1), (model.probability(right_record) / model.probability(left_record))
               ** (left_power - right_power))


@dataclass(frozen=True)
class SwapEvent:
    before: tuple[Replica, Replica]
    after: tuple[Replica, Replica]
    acceptance: F
    accepted: bool


def exchange(model: TinyModel, left: Replica, right: Replica, rng: Random) -> SwapEvent:
    # Check cache identity even for rejected exchanges; stale scores fail closed.
    if left != make_replica(model, left.power, left.record) or right != make_replica(
            model, right.power, right.record):
        raise ValueError("replica scores do not match this model, record and receiver power")
    acceptance = swap_acceptance(model, left.record, right.record, left.power, right.power)
    accepted = _draw({True: acceptance, False: 1 - acceptance}, rng)
    after = ((make_replica(model, left.power, right.record),
              make_replica(model, right.power, left.record)) if accepted else (left, right))
    return SwapEvent((left, right), after, acceptance, accepted)


def joint_kernel(model: TinyModel, powers: tuple[int, int] = (1, 2),
                 exchange_enabled: bool = True) -> dict:
    """Local update of both replicas, then optional exchange. May be nonreversible."""
    if not isinstance(powers, tuple) or len(powers) != 2:
        raise ValueError("the exact joint fixture supports exactly two powers")
    if type(exchange_enabled) is not bool:
        raise ValueError("exchange_enabled must be boolean")
    local = [TinySampler(model, power).local_kernel() for power in powers]
    states = tuple(product(model.states, repeat=2))
    kernel = {}
    for old in states:
        row = {new: F(0) for new in states}
        for middle in states:
            mass = local[0][old[0]][middle[0]] * local[1][old[1]][middle[1]]
            if not mass:
                continue
            acceptance = (swap_acceptance(model, *middle, *powers)
                          if exchange_enabled else F(0))
            row[middle[::-1]] += mass * acceptance
            row[middle] += mass * (1 - acceptance)
        kernel[old] = row
    return kernel


def broken_truncation_kernel(model: TinyModel, power: int) -> dict:
    """NEGATIVE CONTROL ONLY: state-length restart selection with no MH correction.

    The superficially plausible truncation ignores post-EOS horizon positions.
    Its restart law depends on state, but its acceptance omits that ratio.
    It must fail the exact stationary-distribution qualification.
    """
    sampler = TinySampler(model, power)
    fixed = [sampler.local_kernel(restart) for restart in range(HORIZON)]
    return {old: {new: sum((fixed[restart][old][new] / model.realized_length(old)
                           for restart in range(model.realized_length(old))), F(0))
                  for new in model.states} for old in model.states}
