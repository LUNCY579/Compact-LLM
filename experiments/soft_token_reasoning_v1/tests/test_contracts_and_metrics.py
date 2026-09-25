from __future__ import annotations

import json

import pytest

from experiments.soft_token_reasoning_v1.evaluation.metrics import (
    correct_solution_diversity,
    paired_repairs_damages,
    pass_at_k,
    retention,
)
from experiments.soft_token_reasoning_v1.implementation.binding import bind_dataset_manifest
from experiments.soft_token_reasoning_v1.implementation.contracts import validate_arm_pair


def arm_pair():
    base = {
        "model_id": "Qwen/Qwen3-0.6B",
        "tokenizer_id": "Qwen/Qwen3-0.6B",
        "dataset_id": "Rineya-Puzzle-Reasoning-51k",
        "train_split": "train",
        "held_out_split": "validation",
        "test_split": "test",
        "seeds": [1],
        "pass_at_k": [1, 8],
        "max_reasoning_tokens": 8,
        "sealed_access": False,
        "gflowrl": False,
    }
    d0 = {**base, "arm": "D0", "train_inference": {"train_soft": False, "infer_hard": True, "temperature": 1.0, "add_noise": False}}
    s1 = {**base, "arm": "S1", "train_inference": {"train_soft": True, "infer_hard": True, "temperature": 0.5, "noise_scale": 0.33, "add_noise": True}}
    return d0, s1


def test_arm_pair_changes_only_soft_training_mechanism():
    validate_arm_pair(*arm_pair())


def test_gflowrl_and_sealed_paths_are_rejected():
    d0, s1 = arm_pair()
    s1["gflowrl"] = True
    with pytest.raises(ValueError):
        validate_arm_pair(d0, s1)
    s1["gflowrl"] = False
    s1["sealed_access"] = True
    with pytest.raises(ValueError):
        validate_arm_pair(d0, s1)


def test_dataset_manifest_binds_exact_split_ids(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"train": {"rows": 3, "bytes": 10, "sha256": "a" * 64}, "validation": {"rows": 1, "bytes": 4, "sha256": "b" * 64}}), encoding="utf-8")
    binding = bind_dataset_manifest(manifest, "toy", ("train", "validation"))
    assert binding["dataset_id"] == "toy"
    assert binding["splits"]["train"]["sha256"] == "a" * 64
    assert binding["sealed_access"] is False


def test_pass_k_diversity_and_paired_counts_are_frozen():
    groups = [[True, False, False], [False, True, False]]
    assert pass_at_k(groups, 1) == 0.5
    assert pass_at_k(groups, 2) == 1.0
    outputs = [[{"correct": True, "answer_norm": "a"}, {"correct": True, "answer_norm": "b"}], [{"correct": False, "answer_norm": "x"}]]
    assert correct_solution_diversity(outputs) == 0.5
    baseline = {"a": True, "b": False, "c": True}
    candidate = {"a": False, "b": True, "c": True}
    assert paired_repairs_damages(baseline, candidate) == {"repairs": 1, "damages": 1, "unchanged_correct": 1}
    assert retention(baseline, candidate) == 0.5
