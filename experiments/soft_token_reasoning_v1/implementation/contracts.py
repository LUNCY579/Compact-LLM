"""Experiment contracts that keep D0 and S1 scientifically comparable."""
from __future__ import annotations

from copy import deepcopy
from typing import Any


MODEL_ID = "Qwen/Qwen3-0.6B"
TOKENIZER_ID = "Qwen/Qwen3-0.6B"
DATASET_ID = "Rineya-Puzzle-Reasoning-51k"
TRAIN_SPLIT = "train"
HELD_OUT_SPLIT = "validation"
TEST_SPLIT = "test"


def validate_arm_pair(discrete: dict[str, Any], soft: dict[str, Any]) -> None:
    """Check that only the soft-token mechanism differs between arms."""

    if discrete.get("arm") != "D0" or soft.get("arm") != "S1":
        raise ValueError("expected D0 and S1 arm configurations")
    for key in ("model_id", "tokenizer_id", "dataset_id", "train_split", "held_out_split", "test_split", "max_reasoning_tokens", "seeds", "pass_at_k", "sealed_access", "gflowrl"):
        if discrete.get(key) != soft.get(key):
            raise ValueError(f"arm mismatch outside soft mechanism: {key}")
    if discrete["train_inference"].get("train_soft"):
        raise ValueError("D0 must use discrete training")
    if not discrete["train_inference"].get("infer_hard") or not soft["train_inference"].get("infer_hard"):
        raise ValueError("both arms must use hard inference")
    if not soft["train_inference"].get("train_soft"):
        raise ValueError("S1 must use soft training")
    if soft["train_inference"].get("temperature") != 0.5:
        raise ValueError("S1 temperature must match the paper's soft-token setup")
    if soft["train_inference"].get("noise_scale") != 0.33:
        raise ValueError("S1 noise scale must match the paper's default")
    if discrete.get("sealed_access") or soft.get("sealed_access"):
        raise ValueError("sealed evaluation is not allowed in qualification")
    if discrete.get("gflowrl") or soft.get("gflowrl"):
        raise ValueError("GFlowRL is not part of this experiment")


def comparison_projection(config: dict[str, Any]) -> dict[str, Any]:
    """Return the factors that must remain equal across D0 and S1."""

    result = deepcopy(config)
    result.pop("arm", None)
    result.pop("description", None)
    result["train_inference"] = {
        "infer_hard": result["train_inference"].get("infer_hard"),
        "train_soft": result["train_inference"].get("train_soft"),
    }
    return result
