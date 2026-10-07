"""Strict experiment configuration; all arms share an actor-update budget."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

ARMS = {
    "control": ("distill", "dpo", "rl1", "rl2"),
    "soup_only": ("distill", "dpo", "rl1", "rl2", "tail_soup"),
    "layered": ("distill", "distill_soup", "dpo", "rl1", "sft1", "soup1"),
    "full": ("distill", "distill_soup", "dpo", "rl1", "rl2", "sft1", "soup1", "sft2", "soup2", "critic_fit", "rl3"),
}
ARMS["full_no_critic"] = tuple(x for x in ARMS["full"] if x != "critic_fit")
ARMS["full_no_soup"] = tuple(x for x in ARMS["full"] if "soup" not in x)


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


@dataclass(frozen=True)
class Config:
    arm: str = "layered"
    seed: int = 20261007
    actor_steps: int = 60
    critic_steps: int = 12
    group_size: int = 4
    max_prompt_tokens: int = 1024
    max_new_tokens: int = 256
    max_context_tokens: int = 2048
    learning_rate: float = 1e-5
    critic_lr: float = 5e-4
    dpo_beta: float = .1
    kl_beta: float = .02
    replay_weight: float = .1
    clip_epsilon: float = .2
    logprob_tolerance: float = 1e-3
    soup_alphas: tuple[float, ...] = (0., .1, .25, .5, .75, 1.)
    tail_checkpoints: int = 5
    max_retention_drop: float = .01
    min_acquisition_gain: float = .05
    min_eval_groups: int = 128
    bootstrap_samples: int = 2000
    required_suites: tuple[str, ...] = ("acquisition", "retention", "instruction", "general_qa", "roleplay_contract", "abstention", "long_context")
    critic_min_auc: float = .6
    processed_token_budget: int = 100_000_000
    generated_token_budget: int = 2_000_000
    rollout_budget: int = 100_000
    wall_seconds: int = 14400
    host_pre_gib: float = 7.4
    host_worker_gib: float = 2.
    gpu_pre_gib: float = 4.
    gpu_worker_gib: float = .5
    disk_gib: float = 2.
    train_last_layers: int = 4
    trainable_fraction_cap: float = .05

    def __post_init__(self):
        require(self.arm in ARMS, "unknown experiment arm")
        for key in ("actor_steps", "critic_steps", "group_size", "max_prompt_tokens", "max_new_tokens", "max_context_tokens", "tail_checkpoints", "min_eval_groups", "bootstrap_samples", "processed_token_budget", "generated_token_budget", "rollout_budget", "wall_seconds", "train_last_layers"):
            require(type(getattr(self, key)) is int and getattr(self, key) > 0, f"invalid {key}")
        require(type(self.seed) is int and self.seed >= 0, "invalid seed")
        require(self.group_size >= 2 and self.tail_checkpoints >= 2, "groups/tails require >=2")
        require(self.actor_steps >= 2 * len(self.train_stages), "need at least two actor updates per training stage")
        require(self.max_prompt_tokens + self.max_new_tokens <= self.max_context_tokens, "context overflow")
        for key in ("learning_rate", "critic_lr", "dpo_beta", "logprob_tolerance"):
            require(math.isfinite(getattr(self, key)) and getattr(self, key) > 0, f"invalid {key}")
        for key in ("kl_beta", "replay_weight", "host_pre_gib", "host_worker_gib", "gpu_pre_gib", "gpu_worker_gib", "disk_gib"):
            require(math.isfinite(getattr(self, key)) and getattr(self, key) >= 0, f"invalid {key}")
        for key in ("clip_epsilon", "max_retention_drop", "min_acquisition_gain", "critic_min_auc", "trainable_fraction_cap"):
            require(math.isfinite(getattr(self, key)) and 0 <= getattr(self, key) < 1, f"invalid {key}")
        require(self.trainable_fraction_cap > 0, "no trainable capacity")
        require(len(set(self.required_suites)) == len(self.required_suites) and "acquisition" in self.required_suites and "retention" in self.required_suites, "acquisition and retention suites required")
        require(len(set(self.soup_alphas)) == len(self.soup_alphas) and 0. in self.soup_alphas and 1. in self.soup_alphas, "soup grid must include both parents")
        require(all(math.isfinite(x) and 0 <= x <= 1 for x in self.soup_alphas), "convex soup coefficients required")

    @property
    def train_stages(self):
        return tuple(x for x in ARMS[self.arm] if "soup" not in x and x != "critic_fit")

    def schedule(self):
        q, r = divmod(self.actor_steps, len(self.train_stages))
        allocation = {s: q + (i < r) for i, s in enumerate(self.train_stages)}
        return [(s, self.critic_steps if s == "critic_fit" else allocation.get(s, 0)) for s in ARMS[self.arm]]

    @property
    def fingerprint(self):
        return digest(asdict(self))

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        for key in ("required_suites", "soup_alphas"):
            if key in value:
                value[key] = tuple(value[key])
        return cls(**value)
