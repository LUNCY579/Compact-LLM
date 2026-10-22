"""Append-only relabel PROPOSALS with in-process verifier replay.

No dataset is overwritten; no generated code is executed. A native verifier must
be independently audited/sandboxed. A callback/digest is not a security proof.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Callable, Collection

from .core import digest


@dataclass(frozen=True)
class Verdict:
    valid: bool
    score: float  # higher is better, under one frozen, declared objective

    def __post_init__(self):
        if type(self.valid) is not bool or isinstance(self.score, bool):
            raise ValueError("invalid verifier verdict")
        if not isinstance(self.score, (int, float)) or not math.isfinite(self.score):
            raise ValueError("score must be finite")


def propose(*, parent_id: str, split: str, prompt: str, old: str, candidate: str,
            protected_parents: Collection[str], verifier: Callable[[str, str], Verdict],
            verifier_sha256: str, generation_contract_sha256: str,
            deployment_contract_sha256: str, objective_id: str) -> dict:
    """Replay old/new solutions; retain only validity or STRICT objective gains."""
    if not isinstance(parent_id, str) or not parent_id or split != "train":
        raise ValueError("relabeling requires an identified TRAIN parent")
    if parent_id in protected_parents:
        raise ValueError("protected parent cannot be relabeled")
    if any(not isinstance(x, str) or not x.strip()
           for x in (prompt, candidate, objective_id)) or not isinstance(old, str):
        raise ValueError("invalid prompt, solution, or objective")
    for value in (verifier_sha256, generation_contract_sha256, deployment_contract_sha256):
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError("SHA256 identifiers required")
    if generation_contract_sha256 != deployment_contract_sha256:
        raise ValueError("generation/deployment decoder contract mismatch")
    before, after = verifier(prompt, old), verifier(prompt, candidate)
    if not isinstance(before, Verdict) or not isinstance(after, Verdict):
        raise ValueError("verifier must return typed Verdict objects")
    accepted = (candidate != old and after.valid and
                (not before.valid or after.score > before.score))
    return {
        "schema": "cde001.relabel-proposal.v1", "parent_id": parent_id, "split": split,
        "prompt_sha256": digest(prompt), "old_sha256": digest(old),
        "candidate_sha256": digest(candidate), "verifier_sha256": verifier_sha256,
        "decoder_contract_sha256": deployment_contract_sha256, "objective_id": objective_id,
        "old_valid": before.valid, "candidate_valid": after.valid,
        "old_score": before.score, "candidate_score": after.score,
        "accepted": accepted, "dataset_mutated": False,
        "reason": "strict_verified_improvement" if accepted else "no_strict_verified_improvement",
        "claim_ceiling": "verifier_replayed_proposal_not_proven_learned_capability",
    }
