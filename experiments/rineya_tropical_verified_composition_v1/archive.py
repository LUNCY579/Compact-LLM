"""Append-only observed transition DAG, exact replay and bounded top-k composition."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import random
from typing import Callable

from .domain import State, Task, digest, integer, root, step, success, terminal
from .runtime import Episode, Ledger, Policy, replay, rollout


@dataclass
class Edge:
    before: State
    action: str
    after: State
    sources: set[str] = field(default_factory=set)
    training_uses: int = 0

    @property
    def key(self) -> str:
        return digest([asdict(self.before), self.action, asdict(self.after)])


@dataclass(frozen=True)
class Candidate:
    task_key: str
    actions: tuple[str, ...]
    origin: str
    discovered_at: int
    support_attempts: tuple[str, ...]

    @property
    def key(self) -> str:
        return digest([self.task_key, self.actions])


# A scored path is (log-probability, tuple of edge IDs).
ScoredPath = tuple[float, tuple[str, ...]]


def top(paths: list[ScoredPath], k: int) -> list[ScoredPath]:
    unique = {edges: score for score, edges in paths}
    return sorted(((score, edges) for edges, score in unique.items()),
                  key=lambda item: (-item[0], item[1]))[:k]


class Archive:
    def __init__(self, task: Task, max_edges: int = 1024, max_paths: int = 64):
        if task.role not in ("train", "replay_train"):
            raise ValueError("Evaluation tasks cannot enter a training archive")
        integer(max_edges, "max_edges", 1, 100_000)
        integer(max_paths, "max_paths", 1, 10_000)
        self.task = task
        self.max_edges, self.max_paths = max_edges, max_paths
        self.edges: dict[str, Edge] = {}
        self.episodes: dict[str, Episode] = {}
        self.candidates: dict[str, Candidate] = {}
        self.root_observed: set[str] = set()
        self.accepting_states: set[State] = set()
        self.starts: dict[State, int] = {}
        self.scores: dict[str, float] = {}
        self.revision: str | None = None
        self.capacity_events = 0

    def _edge_ids(self, actions: tuple[str, ...]) -> tuple[str, ...]:
        state, ids = root(self.task), []
        for action in actions:
            following = step(self.task, state, action)
            key = Edge(state, action, following).key
            if key not in self.edges:
                raise ValueError("Path uses an unobserved transition")
            ids.append(key)
            state = following
        return tuple(ids)

    def _admit(self, actions: tuple[str, ...], origin: str, ledger: Ledger) -> Candidate:
        if not actions:
            raise ValueError("Empty solution")
        edges = self._edge_ids(actions)
        key = digest([self.task.key, actions])
        if key in self.candidates:
            return self.candidates[key]
        if len(self.candidates) >= self.max_paths:
            self.capacity_events += 1
            raise RuntimeError("Archive capacity reached; no silent eviction")
        states = replay(self.task, actions, ledger)
        if states[-1] not in self.accepting_states:
            raise ValueError("No observed accepting terminal supports this solution")
        support = tuple(sorted({min(self.edges[e].sources) for e in edges}))
        candidate = Candidate(self.task.key, actions, origin, len(self.episodes), support)
        self.candidates[key] = candidate
        return candidate

    def collect(self, policy: Policy, rng: random.Random, ledger: Ledger,
                attempt_id: str, prefix: tuple[str, ...] = (),
                complete_restarts: bool = True) -> Episode:
        if not attempt_id or attempt_id in self.episodes:
            raise ValueError("Missing or duplicate attempt identity")
        start = root(self.task)
        if prefix:
            self._edge_ids(prefix)
            start = replay(self.task, prefix, ledger, require_success=False)[-1]
            if terminal(self.task, start):
                raise ValueError("Restart prefix already terminal")
        episode = rollout(self.task, policy, rng, ledger, attempt_id, start)
        self.episodes[attempt_id] = episode
        self.starts[start] = self.starts.get(start, 0) + 1
        additions = [Edge(before, action, after, {attempt_id}) for before, action, after
                     in zip(episode.states, episode.actions, episode.states[1:])]
        new_keys = {edge.key for edge in additions} - self.edges.keys()
        if len(self.edges) + len(new_keys) > self.max_edges:
            self.capacity_events += 1
            raise RuntimeError("Graph capacity reached; episode/cost retained, no eviction")
        for edge in additions:
            if edge.key in self.edges:
                self.edges[edge.key].sources.add(attempt_id)
            else:
                self.edges[edge.key] = edge
        self.revision = None  # Every refresh must explicitly rescore under one policy.
        if episode.accepted:
            self.accepting_states.add(episode.states[-1])
            actions = prefix + episode.actions
            if episode.kind == "root":
                self.root_observed.add(digest([self.task.key, actions]))
                self._admit(actions, "root_sample", ledger)
            elif complete_restarts:
                self._admit(actions, "restart_completion", ledger)
        return episode

    def rescore(self, revision: str, scorer: Callable[[Task, State, str], float],
                ledger: Ledger) -> None:
        if not revision:
            raise ValueError("A policy revision is required")
        self.revision = None
        scores = {}
        for key, edge in sorted(self.edges.items()):
            ledger.require(score_calls=1)
            ledger.spend(score_calls=1)
            value = float(scorer(self.task, edge.before, edge.action))
            if not math.isfinite(value) or value > 1e-7:
                raise ValueError("Expected finite nonpositive canonical-action log-probability")
            scores[key] = value
        self.scores, self.revision = scores, revision

    def dynamic_program(self, revision: str, k: int = 4
                        ) -> tuple[dict[State, list[ScoredPath]], dict[State, list[ScoredPath]]]:
        integer(k, "k", 1, 16)
        if revision != self.revision or set(self.scores) != set(self.edges):
            raise ValueError("Stale or incomplete edge scoring")
        states = {root(self.task)}
        outgoing: dict[State, list[Edge]] = {}
        for edge in self.edges.values():
            states.update((edge.before, edge.after))
            outgoing.setdefault(edge.before, []).append(edge)
        ordered = sorted(states)
        forward: dict[State, list[ScoredPath]] = {root(self.task): [(0.0, ())]}
        backward: dict[State, list[ScoredPath]] = {
            state: [(0.0, ())] for state in self.accepting_states}
        for state in ordered:
            for edge in sorted(outgoing.get(state, []), key=lambda item: item.key):
                additions = [(score + self.scores[edge.key], path + (edge.key,))
                             for score, path in forward.get(state, [])]
                forward[edge.after] = top(forward.get(edge.after, []) + additions, k)
        for state in reversed(ordered):
            for edge in sorted(outgoing.get(state, []), key=lambda item: item.key):
                additions = [(self.scores[edge.key] + score, (edge.key,) + path)
                             for score, path in backward.get(edge.after, [])]
                backward[state] = top(backward.get(state, []) + additions, k)
        return forward, backward

    def frontier(self, revision: str, kappa: float = 1.0) -> tuple[str, ...]:
        if not math.isfinite(kappa) or kappa < 0:
            raise ValueError("Invalid frontier exploration weight")
        forward, backward = self.dynamic_program(revision)
        eligible = [s for s in forward if forward[s] and not backward.get(s)
                    and s.depth > 0 and not terminal(self.task, s)]
        if not eligible:
            return ()
        depth_counts = {s.depth: sum(n for state, n in self.starts.items()
                                    if state.depth == s.depth) for s in eligible}
        depth = min(depth_counts, key=lambda d: (depth_counts[d], d))
        state = min((s for s in eligible if s.depth == depth),
                    key=lambda s: (-(forward[s][0][0] - kappa * math.log1p(
                        self.starts.get(s, 0))), s))
        return tuple(self.edges[e].action for e in forward[state][0][1])

    def compose(self, revision: str, ledger: Ledger, k: int = 4,
                candidate_cap: int = 64) -> list[Candidate]:
        integer(candidate_cap, "candidate_cap", 0, 10_000)
        forward, backward = self.dynamic_program(revision, k)
        joins: dict[tuple[str, ...], float] = {}
        for state in sorted(set(forward) & set(backward)):
            for left_score, left in forward[state]:
                for right_score, right in backward[state]:
                    path = left + right
                    if path:
                        joins[path] = left_score + right_score
        added = []
        for _, path in top([(score, path) for path, score in joins.items()], len(joins)):
            actions = tuple(self.edges[e].action for e in path)
            if digest([self.task.key, actions]) in self.candidates:
                continue
            if len(added) >= candidate_cap:
                break
            ledger.require(join_candidates=1)
            ledger.spend(join_candidates=1)
            added.append(self._admit(actions, "graph_composition", ledger))
        return added

    def coverage_selection(self, revision: str, k: int = 2) -> list[Candidate]:
        integer(k, "coverage_k", 1, 64)
        self.dynamic_program(revision)
        paths = sorted(self.candidates.values(), key=lambda c: c.key)
        if not paths:
            return []
        ids = {c.key: self._edge_ids(c.actions) for c in paths}
        best = min(paths, key=lambda c: (-sum(self.scores[e] for e in ids[c.key]), c.key))
        remainder = [c for c in paths if c != best]
        remainder.sort(key=lambda c: (-sum(1 / math.sqrt(1 + self.edges[e].training_uses)
                                           for e in ids[c.key]), c.key))
        return [best] + remainder[:k - 1]

    def mark_trained(self, candidates: list[Candidate]) -> None:
        edge_ids = set()
        for candidate in candidates:
            if candidate.key not in self.candidates:
                raise ValueError("Cannot train on an uncertified candidate")
            edge_ids.update(self._edge_ids(candidate.actions))
        # Count iterations, not duplicated appearances within the same update.
        for edge_id in edge_ids:
            self.edges[edge_id].training_uses += 1

    def export(self) -> dict:
        return {"task": asdict(self.task), "task_key": self.task.key,
                "revision": self.revision, "capacity_events": self.capacity_events,
                "episodes": [e.to_dict() for e in self.episodes.values()],
                "edges": [{"id": e.key, "before": asdict(e.before), "action": e.action,
                           "after": asdict(e.after), "sources": sorted(e.sources),
                           "training_uses": e.training_uses} for e in self.edges.values()],
                "candidates": [{**asdict(c), "id": c.key,
                                "sampled_from_root": c.key in self.root_observed}
                               for c in self.candidates.values()]}
