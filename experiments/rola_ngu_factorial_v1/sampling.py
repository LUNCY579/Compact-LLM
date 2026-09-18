"""Event-driven NGU mechanics. Both arms use the same asynchronous interface.

No generator, trainer thread, model loading, or code execution is provided here.
Binary rewards only; every completion carries its original behavior logprobs.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import math


@dataclass(frozen=True)
class Completion:
    sample_id: str
    prompt_id: str
    policy_version: int
    reward: int
    behavior_logps: tuple[float, ...]

    def validate(self, learner_version: int) -> None:
        if not self.sample_id or not self.prompt_id:
            raise ValueError('sample and prompt identities required')
        if type(self.policy_version) is not int or not 0 <= self.policy_version <= learner_version:
            raise ValueError('invalid/future behavior version')
        if type(self.reward) is not int or self.reward not in (0, 1):
            raise ValueError('only verified binary rewards supported')
        if not self.behavior_logps or any(not math.isfinite(x) or x > 0 for x in self.behavior_logps):
            raise ValueError('finite original behavior logprobs required')


@dataclass
class Episode:
    prompt_id: str = ''
    count: int = 0
    reward_sum: int = 0
    rounds: int = 0
    samples: list[Completion] = field(default_factory=list)


@dataclass(frozen=True)
class Decision:
    status: str
    samples: tuple[Completion, ...]
    advantages: tuple[float, ...]
    total_count: int
    baseline: float
    stale_count: int


class Sampler:
    def __init__(self, *, ngu: bool, seed: int, group_size: int = 4,
                 continuation: float = .75, max_age: int = 4,
                 max_rounds: int = 8, max_pending: int = 32):
        if type(ngu) is not bool or type(seed) is not int or seed < 0:
            raise ValueError('invalid sampler mode/seed')
        if any(type(x) is not int or x <= 0 for x in (group_size, max_age, max_rounds, max_pending)):
            raise ValueError('invalid sampler bounds')
        if group_size < 2 or not math.isfinite(continuation) or not 0 <= continuation < 1:
            raise ValueError('bounded NGU requires K>=2 and p<1')
        self.ngu, self.seed, self.group_size = ngu, seed, group_size
        self.continuation, self.max_age, self.max_rounds = continuation, max_age, max_rounds
        self.max_pending = max_pending
        self.pending: dict[str, Episode] = {}
        self.seen: set[str] = set()
        self.closed_episodes: set[str] = set()
        self.learner_version = 0
        self.generated_tokens = 0

    def ingest(self, episode_id: str, batch: tuple[Completion, ...], learner_version: int) -> Decision:
        # Validate first: rejected events must not consume IDs or mutate episodes.
        if episode_id in self.closed_episodes or not episode_id or type(learner_version) is not int or learner_version < self.learner_version:
            raise ValueError('invalid episode or regressing learner version')
        if len(batch) != self.group_size:
            raise ValueError('incomplete generation group')
        for sample in batch:
            sample.validate(learner_version)
        ids = [x.sample_id for x in batch]
        if len(set(ids)) != len(ids) or any(x in self.seen for x in ids):
            raise ValueError('duplicate/replayed completion')
        if len({x.prompt_id for x in batch}) != 1 or len({x.policy_version for x in batch}) != 1:
            raise ValueError('a generation batch must have one prompt and policy version')
        if episode_id not in self.pending and len(self.pending) >= self.max_pending:
            raise ValueError('pending episode cap reached')
        old = self.pending.get(episode_id)
        if old and old.prompt_id != batch[0].prompt_id:
            raise ValueError('episode prompt changed')
        self.learner_version = learner_version
        self.seen.update(ids)
        self.generated_tokens += sum(len(x.behavior_logps) for x in batch)
        e = self.pending.pop(episode_id, Episode(prompt_id=batch[0].prompt_id))
        e.samples.extend(batch)
        e.count += len(batch)
        e.reward_sum += sum(x.reward for x in batch)
        e.rounds += 1
        baseline = e.reward_sum / e.count
        fresh = tuple(x for x in e.samples if learner_version - x.policy_version < self.max_age)
        stale = e.count - len(fresh)
        pos = sum(x.reward for x in fresh)
        neg = len(fresh) - pos
        if e.reward_sum:
            self.closed_episodes.add(episode_id)
            if pos and neg:
                positive = 1 - baseline
                advantages = tuple(positive if x.reward else -pos * positive / neg for x in fresh)
                return Decision('UPDATE', fresh, advantages, e.count, baseline, stale)
            return Decision('EASY' if e.reward_sum == e.count else 'STALE_NO_SIGNAL',
                            (), (), e.count, baseline, stale)
        # All-failure retries retain counts even when old training samples expire.
        if self.ngu and e.rounds < self.max_rounds:
            key = f'{self.seed}:{episode_id}:{e.count}'.encode()
            u = int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') / 2**64
            if u < self.continuation:
                e.samples = list(fresh)
                self.pending[episode_id] = e
                return Decision('RETRY', (), (), e.count, baseline, stale)
        self.closed_episodes.add(episode_id)
        return Decision('CAPPED' if self.ngu and e.rounds >= self.max_rounds else 'GIVE_UP',
                        (), (), e.count, baseline, stale)


class TokenBudget:
    """Reserve before dispatch, settle actual work, retain all failed-attempt costs.

    Units are prompt-plus-generated tokens across completions, not GPU FLOPs.
    Prefill sharing must not silently change the accounting between arms.
    """
    def __init__(self, limit: int):
        if type(limit) is not int or limit <= 0:
            raise ValueError('positive integer budget required')
        self.limit, self.spent = limit, 0
        self.pending: dict[str, int] = {}
        self.closed: set[str] = set()

    def reserve(self, request: str, maximum: int) -> None:
        if not request or request in self.pending or request in self.closed:
            raise ValueError('duplicate/invalid request')
        if type(maximum) is not int or maximum <= 0:
            raise ValueError('positive maximum required')
        if self.spent + sum(self.pending.values()) + maximum > self.limit:
            raise ValueError('budget exhausted before dispatch')
        self.pending[request] = maximum

    def settle(self, request: str, actual: int | None = None) -> None:
        maximum = self.pending[request]
        # A failed request with unknown usage is conservatively fully charged.
        actual = maximum if actual is None else actual
        if type(actual) is not int or not 0 <= actual <= maximum:
            raise ValueError('usage exceeds reservation')
        self.spent += actual
        del self.pending[request]
        self.closed.add(request)
