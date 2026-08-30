from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tgsa.mtp import (
    MTPConfig,
    build_tgsa_mtp_model,
    combined_tgsa_mtp_loss,
    flatten_tgsa_control,
    multi_token_prediction_loss,
    shift_future_targets,
)
from tgsa.phase1_model import (
    TinyLMConfig,
    build_canonical_supernet,
    state_dict_sha256,
)
from tgsa.phase2_mtp import (
    PHASE2_SCHEMA_VERSION,
    analyze_factorial,
    load_phase2_config,
    run_phase2_study,
)


def make_lm_config(arm: str = "learned_delta") -> TinyLMConfig:
    return TinyLMConfig(
        vocab_size=96,
        max_sequence_length=32,
        embed_dim=32,
        num_heads=4,
        num_layers=1,
        ffn_multiplier=2.0,
        dropout=0.0,
        bias=False,
        trajectory_rank=8,
        arm=arm,  # type: ignore[arg-type]
        tie_embeddings=True,
    )


def canonical_state(seed: int = 17) -> dict[str, torch.Tensor]:
    canonical = build_canonical_supernet(make_lm_config(), initialization_seed=seed)
    return {key: value.clone() for key, value in canonical.state_dict().items()}


def test_future_target_alignment() -> None:
    labels = torch.tensor([[10, 11, 12, 13, -100]])
    assert shift_future_targets(labels, offset=0).tolist() == [[10, 11, 12, 13, -100]]
    assert shift_future_targets(labels, offset=1).tolist() == [[11, 12, 13, -100, -100]]
    assert shift_future_targets(labels, offset=3).tolist() == [[13, -100, -100, -100, -100]]


def test_mtp_modes_are_parameter_and_initialization_matched() -> None:
    state = canonical_state()
    models = {
        mode: build_tgsa_mtp_model(
            make_lm_config(),
            MTPConfig(max_horizon=3, transition_rank=8, mode=mode),  # type: ignore[arg-type]
            initialization_seed=17,
            canonical_state=state,
        )
        for mode in ("dormant", "backbone", "trajectory")
    }
    assert len({state_dict_sha256(model.mtp_head) for model in models.values()}) == 1
    assert len({state_dict_sha256(model.backbone) for model in models.values()}) == 1
    parameter_counts = {
        sum(parameter.numel() for parameter in model.parameters())
        for model in models.values()
    }
    assert len(parameter_counts) == 1

    input_ids = torch.randint(0, 96, (2, 8))
    labels = torch.randint(0, 96, (2, 8))
    outputs = {
        mode: model(input_ids, labels=labels).logits for mode, model in models.items()
    }
    torch.testing.assert_close(outputs["dormant"], outputs["backbone"])
    torch.testing.assert_close(outputs["backbone"], outputs["trajectory"])


def test_trajectory_conditioning_uses_existing_differentiable_control() -> None:
    state = canonical_state(23)
    model = build_tgsa_mtp_model(
        make_lm_config(),
        MTPConfig(max_horizon=3, transition_rank=8, mode="trajectory"),
        initialization_seed=23,
        canonical_state=state,
    )
    input_ids = torch.randint(0, 96, (2, 10))
    labels = torch.randint(0, 96, (2, 10))
    output = model(input_ids, labels=labels)

    assert output.trajectory_control.requires_grad
    assert output.block_telemetry[-1] is not None
    assert output.block_telemetry[-1].control_signals is not None  # type: ignore[union-attr]
    reconstructed = flatten_tgsa_control(
        output.block_telemetry,
        block_index=-1,
        embed_dim=32,
    )
    torch.testing.assert_close(output.trajectory_control, reconstructed)

    output.trajectory_control.retain_grad()
    loss, _, _ = multi_token_prediction_loss(
        output.mtp_logits,
        labels,
        horizon_weights=model.mtp_config.weights,
    )
    loss.backward()
    assert output.trajectory_control.grad is not None
    assert torch.isfinite(output.trajectory_control.grad).all()
    assert output.trajectory_control.grad.abs().sum() > 0


def test_dormant_mode_keeps_auxiliary_loss_out_of_training_objective() -> None:
    state = canonical_state(29)
    model = build_tgsa_mtp_model(
        make_lm_config(),
        MTPConfig(max_horizon=3, transition_rank=8, mode="dormant", loss_weight=0.7),
        initialization_seed=29,
        canonical_state=state,
    )
    input_ids = torch.randint(0, 96, (2, 9))
    labels = torch.randint(0, 96, (2, 9))
    loss_mask = torch.zeros_like(labels, dtype=torch.bool)
    loss_mask[:, -2] = True
    output = model(input_ids, labels=labels)
    objective = combined_tgsa_mtp_loss(
        output,
        labels,
        loss_mask,
        mtp_config=model.mtp_config,
    )
    torch.testing.assert_close(objective.total_loss, objective.autoregressive_loss)
    assert objective.mtp_loss.item() > 0


def _base_phase2_config(tmp_path: Path) -> dict[str, object]:
    return {
        "schema_version": PHASE2_SCHEMA_VERSION,
        "study_id": "unit-smoke",
        "confirmatory": False,
        "expected_base_commit": "9b5fe9e000994d7f84c9f42326aff3d5f4682aa1",
        "cells": [
            {"tgsa_arm": "history_control", "mtp_mode": "dormant"},
            {"tgsa_arm": "history_control", "mtp_mode": "backbone"},
        ],
        "seeds": [7],
        "dataset": {
            "seed": 101,
            "split_generation_order": ["test", "dev", "train"],
            "vocabulary": {"domain_size": 8, "max_hops": 3, "max_abs_step": 3},
            "families": {
                "direct_lookup": {
                    "min_nodes": 4,
                    "max_nodes": 6,
                    "min_hops": 1,
                    "max_hops": 1,
                },
                "pointer_chase": {
                    "min_nodes": 4,
                    "max_nodes": 6,
                    "min_hops": 2,
                    "max_hops": 2,
                    "reject_trivial": True,
                },
                "state_update": {
                    "min_steps": 2,
                    "max_steps": 2,
                    "reject_trivial": True,
                },
            },
            "splits": {
                "train": {"direct_lookup": 4, "pointer_chase": 4, "state_update": 4},
                "dev": {"direct_lookup": 2, "pointer_chase": 2, "state_update": 2},
                "test": {"direct_lookup": 2, "pointer_chase": 2, "state_update": 2},
            },
        },
        "model": {
            "max_sequence_length": 48,
            "embed_dim": 16,
            "num_heads": 4,
            "num_layers": 1,
            "ffn_multiplier": 2.0,
            "dropout": 0.0,
            "bias": False,
            "trajectory_rank": 4,
            "tie_embeddings": True,
        },
        "mtp": {
            "max_horizon": 2,
            "transition_rank": 4,
            "loss_weight": 0.2,
            "horizon_weights": [1.0],
            "control_block_index": -1,
            "pad_token_id": 0,
            "bias": False,
        },
        "training": {
            "steps": 1,
            "batch_size": 2,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
        },
        "evaluation": {"split": "dev", "batch_size": 3},
        "runtime": {
            "device": "cpu",
            "deterministic_algorithms": True,
            "num_threads": 1,
        },
        "registration": {
            "status": "draft",
            "phase1_competence_report": None,
        },
    }


def test_nonconfirmatory_rehearsal_emits_smoke_only_artifacts(tmp_path: Path) -> None:
    config = _base_phase2_config(tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    report = run_phase2_study(
        config_path=config_path,
        dataset_dir=tmp_path / "dataset",
        output_dir=tmp_path / "output",
        repo_root=tmp_path,
    )
    assert report["status"] == "complete"
    assert report["decision"] == "smoke_only"
    assert report["competence_precondition"]["required"] is False  # type: ignore[index]
    assert len(report["runs"]) == 2  # type: ignore[arg-type]
    for run in report["runs"]:  # type: ignore[assignment]
        assert run["gradient_audit"]["mtp_head_gradient_norm"] > 0
        assert run["evaluation"]["teacher_forced_future_prefix"]["claim_scope"].endswith(
            "not throughput"
        )
    assert (tmp_path / "output" / "phase2_report.json").is_file()
    assert not list((tmp_path / "output").rglob("predictions_test*.jsonl"))


def test_confirmatory_missing_competence_fails_before_output_creation(tmp_path: Path) -> None:
    config = _base_phase2_config(tmp_path)
    config["confirmatory"] = True
    config["registration"] = {
        "status": "frozen",
        "phase1_competence_report": "missing_competence.json",
    }
    config_path = tmp_path / "confirmatory.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "must_not_exist"
    with pytest.raises(FileNotFoundError, match="competence report"):
        run_phase2_study(
            config_path=config_path,
            dataset_dir=tmp_path / "dataset",
            output_dir=output,
            repo_root=tmp_path,
        )
    assert not output.exists()
    assert not (tmp_path / "dataset").exists()


def test_test_split_cannot_be_selected_for_calibration(tmp_path: Path) -> None:
    config = _base_phase2_config(tmp_path)
    config["evaluation"]["split"] = "test"  # type: ignore[index]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="development split"):
        load_phase2_config(path)


def test_factorial_analysis_computes_difference_in_differences() -> None:
    runs = []
    values = {
        "history_control__mtp_dormant": 0.20,
        "history_control__mtp_backbone": 0.25,
        "learned_delta__mtp_dormant": 0.21,
        "learned_delta__mtp_backbone": 0.31,
    }
    for seed in (1, 2):
        for cell, accuracy in values.items():
            runs.append(
                {
                    "seed": seed,
                    "cell_key": cell,
                    "evaluation": {"answer_accuracy": accuracy},
                }
            )
    analysis = analyze_factorial(runs)
    interaction = analysis["contrasts"]["mtp_x_trajectory_mechanism"]  # type: ignore[index]
    assert interaction["mean_difference_in_differences"] == pytest.approx(0.05)
    assert analysis["decision"] == "descriptive_only"
