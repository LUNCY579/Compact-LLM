"""Explicit bridge to the inherited PRLC-001, without importing/loading a model."""
from __future__ import annotations

from .contracts import HARNESS_HASH, Action, View, digest, render, require


class PRLCStudent:
    """Wrap an already constructed/qualified PRLC Backend for next-action inference.

    Constructor does not load weights. Creating the upstream backend and approving
    calls are operator responsibilities. This wrapper is not a native qualification.
    """
    def __init__(self, backend, *, max_new_tokens: int, allow_model_calls: bool = False):
        require(allow_model_calls is True, "explicit model-call authorization required")
        require(type(max_new_tokens) is int and max_new_tokens > 0, "positive token cap required")
        self.backend, self.cap = backend, max_new_tokens
        self.version = backend.version
        self.identity = "prlc:" + digest({"backend": backend.identity, "version": backend.version})

    def propose(self, view: View, seed: int) -> str:
        from experiments.progressive_rl_curriculum_v1.schema import Prompt
        require(self.backend.version == self.version, "student changed during collection")
        prompt = Prompt(view.fingerprint, render(view))
        episodes = self.backend.collect(prompt, 1, self.cap, seed)
        require(len(episodes) == 1, "expected one proposal, not best-of-N")
        episode = episodes[0]
        require(episode.key == prompt.key and episode.policy_version == self.version,
                "mismatched/stale episode")
        require(episode.terminated and episode.valid_action and not episode.infrastructure_error, "incomplete native action")
        return episode.text


def to_prlc_demonstrations(rows: list[dict]):
    """Convert audited SFT action rows into PRLC Task/response pairs.

    These exact-v1 answers are action-imitation targets, NOT downstream task golds.
    Re-run compile_dataset on original traces before calling this transport helper.
    Native PRLC calibration remains separate from RHD multi-turn deployment eval.
    """
    from experiments.progressive_rl_curriculum_v1.schema import Task
    require(bool(rows), "no SFT rows")
    output, seen = [], set()
    for row in rows:
        require(row["split"] == "train" and row["harness_hash"] == HARNESS_HASH, "training/harness mismatch")
        require(row["id"] not in seen, "duplicate SFT row")
        seen.add(row["id"])
        Action.parse(row["completion"])
        output.append((Task(id=row["id"], source="rhd_v1", revision=row["trajectory_hash"],
                            group=row["parent"], split="train", prompt=row["prompt"],
                            answer=row["completion"], domain="procedural_action", verifier="exact-v1",
                            metadata={"origin": row["task_key"], "mode": row["mode"],
                                      "skill": row["skill"], "harness": HARNESS_HASH}), row["completion"]))
    return output
