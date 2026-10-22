"""Separate EMA rate from KL direction; default planning never starts jobs."""
from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
from .objectives import validate_rate

PAPER_RATES = (0., .02, .05, .10, .25)


@dataclass(frozen=True)
class Arm:
    method: str = "sd"
    rollout_source: str | None = "student"
    kl_direction: str | None = "reverse"
    ema_rate: float = 0.

    def __post_init__(self):
        validate_rate(self.ema_rate)
        if self.method == "sft":
            if self.rollout_source is not None or self.kl_direction is not None or self.ema_rate != 0:
                raise ValueError("SFT has no rollout-source, KL-direction or EMA axis")
        elif self.method != "sd" or self.rollout_source not in {"student", "teacher"} or self.kl_direction not in {"forward", "reverse"}:
            raise ValueError("Invalid self-distillation axes")

    @property
    def name(self):
        return "sft" if self.method == "sft" else f"sd-{self.rollout_source}-{self.kl_direction}-ema{self.ema_rate:g}"


def arms(rates=PAPER_RATES, include_sft=True):
    if not rates or len(set(rates)) != len(rates):
        raise ValueError("Nonempty unique coupling rates required")
    result = [Arm("sd", source, direction, rate)
              for source, rate, direction in itertools.product(("student", "teacher"), rates, ("forward", "reverse"))]
    if include_sft:
        result.append(Arm("sft", None, None, 0.))
    return result


def make_plan(tasks=("spatial_ordinary", "spatial_rotated"), seeds=(17, 29, 43), learning_rates=(1e-5,), rates=PAPER_RATES):
    for values in (tasks, seeds, learning_rates):
        if not values or len(set(values)) != len(values):
            raise ValueError("Unique nonempty factors required")
    if any(type(s) is not int or not 0 <= s < 2**32 for s in seeds):
        raise ValueError("Invalid seeds")
    if any(not math.isfinite(lr) or lr <= 0 for lr in learning_rates):
        raise ValueError("Invalid learning rates")
    cells = [dict(task=task, seed=seed, learning_rate=lr, arm=asdict(arm), arm_id=arm.name)
             for task, seed, lr, arm in itertools.product(tasks, seeds, learning_rates, arms(rates))]
    return {"status": "PLAN_ONLY", "launch_authorized": False, "cells": cells, "run_count": len(cells),
            "method_cells_per_task_seed_lr": len(arms(rates)),
            "matching": "same reference information, train examples and optimizer steps; NOT equal FLOPs",
            "warning": "SFT LR tuning and native budget qualification are independent prerequisites"}


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        f.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def checkpoint_manifest(path):
    path = Path(path)
    if not path.is_dir() or not (path / "config.json").is_file():
        raise ValueError("Exact local model directory required")
    entries = sorted(path.rglob("*"))
    if any(p.is_symlink() for p in entries):
        raise ValueError("Materialize symlinks before checkpoint pinning")
    files = [p for p in entries if p.is_file()]
    if not any(p.suffix == ".safetensors" for p in files):
        raise ValueError("Safetensors model weights required")
    records = [{"path": str(p.relative_to(path)), "bytes": p.stat().st_size, "sha256": hash_file(p)} for p in files]
    return {"files": records, "sha256": digest_json(records)}


def validate_config(c):
    fields = {"checkpoint", "checkpoint_sha256", "dataset", "dataset_sha256", "task", "seed", "steps",
              "learning_rate", "trainable_names", "max_trainable_parameters", "max_new_tokens",
              "max_sequence_tokens", "device", "min_free_cuda_before_gib", "min_free_cuda_during_gib"}
    if not isinstance(c, dict) or set(c) != fields:
        raise ValueError("Config must contain exactly the documented fields")
    for key in ("checkpoint_sha256", "dataset_sha256"):
        if not isinstance(c[key], str) or not re.fullmatch(r"[0-9a-f]{64}", c[key]):
            raise ValueError("Fill real checkpoint and dataset SHA256s")
    for key in ("checkpoint", "dataset", "task"):
        if not isinstance(c[key], str) or not c[key]:
            raise ValueError("Nonempty local paths and task required")
    bounds = {"steps": (3, 64), "max_new_tokens": (1, 128), "max_sequence_tokens": (8, 512),
              "max_trainable_parameters": (1, 20_000_000), "seed": (0, 2**32 - 1)}
    for key, (lo, hi) in bounds.items():
        if type(c[key]) is not int or not lo <= c[key] <= hi:
            raise ValueError(f"Invalid bounded integer: {key}")
    for key in ("learning_rate", "min_free_cuda_before_gib", "min_free_cuda_during_gib"):
        if isinstance(c[key], bool) or not isinstance(c[key], (float, int)) or not math.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f"Invalid positive finite parameter: {key}")
    if c["device"] not in {"cpu", "cuda"} or c["max_new_tokens"] >= c["max_sequence_tokens"]:
        raise ValueError("Invalid device or sequence budgets")
    names = c["trainable_names"]
    if not isinstance(names, list) or not names or len(set(names)) != len(names) or any(not isinstance(n, str) or not n for n in names):
        raise ValueError("Exact unique trainable parameter names required; no wildcard selection")
