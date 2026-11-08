"""Strict paired offline scorer. No generation, task-bank access, or automatic GO."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import comb
import random
from typing import Sequence
from .contracts import Identity, finite, require, same_token_grid, sha256


@dataclass(frozen=True)
class Outcome:
    example_id: str
    group_id: str
    family: str
    prompt_sha256: str
    input_ids_sha256: str
    correct: bool
    format_ok: bool
    unknown_expected: bool
    literal_unknown_ok: bool
    status: str = "ok"

    def validate(self) -> None:
        for value in (self.example_id, self.group_id, self.family):
            require(isinstance(value, str) and bool(value), "missing outcome ID/family")
        sha256(self.prompt_sha256)
        sha256(self.input_ids_sha256)
        require(self.status == "ok", "incomplete/error predictions cannot be dropped or scored as a negative")
        for value in (self.correct, self.format_ok, self.unknown_expected, self.literal_unknown_ok):
            require(type(value) is bool, "outcome values must be actual booleans")


@dataclass(frozen=True)
class Panel:
    model: Identity
    registration_sha256: str
    scorer_sha256: str
    decoding_sha256: str
    role: str
    expected_ids: tuple[str, ...]
    outcomes: tuple[Outcome, ...]

    def validate(self):
        for value in (self.registration_sha256, self.scorer_sha256, self.decoding_sha256):
            sha256(value)
        require(self.role in {"SELECT", "CHECK", "RETENTION"}, "not a reviewed open evaluation role")
        require(0 < len(self.expected_ids) <= 4096 and len(set(self.expected_ids)) == len(self.expected_ids),
                "empty/duplicate/oversized expected ID set")
        require(all(isinstance(x, str) and x for x in self.expected_ids), "invalid expected ID")
        for row in self.outcomes:
            row.validate()
        ids = [row.example_id for row in self.outcomes]
        require(len(ids) == len(set(ids)) and set(ids) == set(self.expected_ids), "missing/extra/duplicate outcomes")


def paired_report(receiver: Panel, candidate: Panel, *, bootstrap_samples: int = 1000,
                  seed: int = 17) -> dict:
    receiver.validate()
    candidate.validate()
    same_token_grid(receiver.model, candidate.model)
    require(receiver.model.precision == candidate.model.precision, "precision mismatch needs a separately registered contrast")
    for field in ("registration_sha256", "scorer_sha256", "decoding_sha256", "role"):
        require(getattr(receiver, field) == getattr(candidate, field), f"{field} mismatch")
    require(set(receiver.expected_ids) == set(candidate.expected_ids), "panel ID sets differ")
    require(type(bootstrap_samples) is int and 100 <= bootstrap_samples <= 10000, "invalid bootstrap budget")
    require(type(seed) is int, "seed must be integer")
    a, b = {r.example_id: r for r in receiver.outcomes}, {r.example_id: r for r in candidate.outcomes}
    counts = {"both_correct": 0, "candidate_only": 0, "receiver_only": 0, "both_wrong": 0}
    groups = defaultdict(list)
    families = defaultdict(list)
    formatting_regressions = 0
    unknown = []
    changes = []
    for key in sorted(a):
        r, c = a[key], b[key]
        require((r.group_id, r.family, r.prompt_sha256, r.input_ids_sha256, r.unknown_expected) ==
                (c.group_id, c.family, c.prompt_sha256, c.input_ids_sha256, c.unknown_expected), "paired row metadata mismatch")
        label = "both_correct" if r.correct and c.correct else (
            "receiver_only" if r.correct else "candidate_only" if c.correct else "both_wrong")
        counts[label] += 1
        delta = int(c.correct) - int(r.correct)
        changes.append(delta)
        groups[r.group_id].append(delta)
        families[r.family].append(delta)
        formatting_regressions += int(r.format_ok and not c.format_ok)
        if r.unknown_expected:
            unknown.append(int(r.literal_unknown_ok and not c.literal_unknown_ok))
    wins, losses = counts["candidate_only"], counts["receiver_only"]
    discordant = wins + losses
    exact_p = 1.0 if discordant == 0 else min(1.0, 2 * sum(comb(discordant, k)
                 for k in range(min(wins, losses) + 1)) / (2**discordant))
    clustered = list(groups.values())
    ci = None
    if len(clustered) >= 2:
        rng = random.Random(seed)
        samples = []
        for _ in range(bootstrap_samples):
            selected = [clustered[rng.randrange(len(clustered))] for _ in clustered]
            samples.append(sum(sum(g) for g in selected) / sum(len(g) for g in selected))
        samples.sort()
        ci = [samples[int(0.025 * (len(samples) - 1))], samples[int(0.975 * (len(samples) - 1))]]
    return {"n": len(a), "groups": len(groups), "role": receiver.role,
            "receiver_accuracy": sum(r.correct for r in a.values()) / len(a),
            "candidate_accuracy": sum(r.correct for r in b.values()) / len(b),
            "paired_gain": sum(changes) / len(changes), "paired_counts": counts,
            "family_paired_gain": {f: sum(d) / len(d) for f, d in sorted(families.items())},
            "cluster_bootstrap_95pct": ci, "bootstrap_samples": bootstrap_samples, "seed": seed,
            "row_mcnemar_exact_p_descriptive_only": exact_p,
            "row_test_assumption": "independent rows; not confirmatory for clustered examples or selected arms",
            "format_regressions": formatting_regressions,
            "unknown_cases": len(unknown), "literal_unknown_regressions": sum(unknown),
            "functional_transfer_established": False, "automatic_promotion": False}
