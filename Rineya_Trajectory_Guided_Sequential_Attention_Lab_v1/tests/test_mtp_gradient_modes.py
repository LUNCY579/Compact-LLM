"""Gradient-isolation and telemetry tests for trajectory-aware MTP."""

from __future__ import annotations

import pytest
import torch

from tgsa.mtp import (
    MTPConfig,
    build_tgsa_mtp_model,
    combined_tgsa_mtp_loss,
    flatten_tgsa_control,
    multi_token_prediction_loss,
)
from tgsa.phase1_model import (
    TinyLMConfig,
    build_canonical_supernet,
    state_dict_sha256,
)


def _lm_config() -> TinyLMConfig:
    return TinyLMConfig(
        vocab_size=96,
        max_sequence_length=32,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        ffn_multiplier=2.0,
        dropout=0.0,
        bias=False,
        trajectory_rank=8,
        arm="learned_delta",
        tie_embeddings=True,
    )


def _canonical_state(seed: int = 71) -> dict[str, torch.Tensor]:
    model = build_canonical_supernet(
        _lm_config(),
        initialization_seed=seed,
    )
    return {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }


def _build(
    gradient_mode: str,
    *,
    control_block_index: int = -1,
    seed: int = 71,
):
    return build_tgsa_mtp_model(
        _lm_config(),
        MTPConfig(
            max_horizon=3,
            transition_rank=8,
            loss_weight=0.20,
            mode="trajectory",
            gradient_mode=gradient_mode,  # type: ignore[arg-type]
            control_block_index=control_block_index,
        ),
        initialization_seed=seed,
        canonical_state=_canonical_state(seed),
    )


def _batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = torch.randint(0, 96, (2, 9))
    labels = torch.randint(0, 96, (2, 9))
    loss_mask = torch.zeros_like(labels, dtype=torch.bool)
    loss_mask[:, -2] = True
    return input_ids, labels, loss_mask


def _mtp_only_backward(model):
    input_ids, labels, _loss_mask = _batch()
    model.zero_grad(set_to_none=True)
    output = model(input_ids, labels=labels)
    loss, _, _ = multi_token_prediction_loss(
        output.mtp_logits,
        labels,
        horizon_weights=model.mtp_config.weights,
    )
    loss.backward()
    return output, loss


def _gradient_norm(parameters) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = float(parameter.grad.detach().float().norm().item())
        total += value * value
    return total**0.5


def test_trajectory_mtp_requests_control_only_from_registered_block() -> None:
    model = _build("joint", control_block_index=0)
    input_ids, labels, _loss_mask = _batch()
    output = model(input_ids, labels=labels)

    assert output.block_telemetry[0] is not None
    assert output.block_telemetry[1] is None
    telemetry = output.block_telemetry[0]
    assert telemetry is not None
    assert telemetry.control_signals is not None
    assert telemetry.control_signals.shape == (2, 4, 9, 8)
    assert telemetry.control_signals.requires_grad
    assert telemetry.carrier_states is None
    assert telemetry.velocities is None
    assert telemetry.importance_scores is None
    assert telemetry.attention_maps is None

    reconstructed = flatten_tgsa_control(
        output.block_telemetry,
        block_index=0,
        embed_dim=32,
    )
    assert reconstructed.shape == (2, 9, 32)
    assert reconstructed.requires_grad
    torch.testing.assert_close(
        output.trajectory_control,
        reconstructed,
    )


def test_gradient_modes_are_parameter_and_initialization_matched() -> None:
    models = {
        mode: _build(mode)
        for mode in ("joint", "stop_gradient", "frozen_probe")
    }
    assert len(
        {
            state_dict_sha256(model.mtp_head)
            for model in models.values()
        }
    ) == 1
    assert len(
        {
            state_dict_sha256(model.backbone)
            for model in models.values()
        }
    ) == 1
    assert len(
        {
            sum(parameter.numel() for parameter in model.parameters())
            for model in models.values()
        }
    ) == 1


def test_joint_mtp_loss_reaches_control_and_backbone() -> None:
    model = _build("joint")
    output, _loss = _mtp_only_backward(model)

    assert output.trajectory_control.requires_grad
    assert _gradient_norm(model.mtp_head.parameters()) > 0.0
    assert _gradient_norm(model.backbone.parameters()) > 0.0
    assert model.backbone.token_embedding.weight.grad is not None


def test_stop_gradient_isolates_auxiliary_loss_from_every_backbone_path() -> None:
    model = _build("stop_gradient")
    output, _loss = _mtp_only_backward(model)

    assert not output.trajectory_control.requires_grad
    assert _gradient_norm(model.mtp_head.parameters()) > 0.0
    assert _gradient_norm(model.backbone.parameters()) == 0.0
    assert model.backbone.token_embedding.weight.grad is None
    assert model.backbone.lm_head.weight.grad is None


def test_frozen_probe_keeps_backbone_fixed_and_excludes_ar_from_objective() -> None:
    model = _build("frozen_probe")
    assert model.backbone.training is False
    assert all(
        not parameter.requires_grad
        for parameter in model.backbone.parameters()
    )

    model.train()
    assert model.training is True
    assert model.mtp_head.training is True
    assert model.backbone.training is False

    input_ids, labels, loss_mask = _batch()
    output = model(input_ids, labels=labels)
    objective = combined_tgsa_mtp_loss(
        output,
        labels,
        loss_mask,
        mtp_config=model.mtp_config,
    )
    assert objective.autoregressive_in_training_objective is False
    torch.testing.assert_close(
        objective.total_loss,
        model.mtp_config.loss_weight * objective.mtp_loss,
    )
    objective.total_loss.backward()
    assert _gradient_norm(model.mtp_head.parameters()) > 0.0
    assert _gradient_norm(model.backbone.parameters()) == 0.0


def test_stop_gradient_keeps_autoregressive_backbone_training() -> None:
    model = _build("stop_gradient")
    input_ids, labels, loss_mask = _batch()
    output = model(input_ids, labels=labels)
    objective = combined_tgsa_mtp_loss(
        output,
        labels,
        loss_mask,
        mtp_config=model.mtp_config,
    )
    assert objective.autoregressive_in_training_objective is True
    torch.testing.assert_close(
        objective.total_loss,
        objective.autoregressive_loss
        + model.mtp_config.loss_weight * objective.mtp_loss,
    )
    objective.total_loss.backward()
    assert _gradient_norm(model.backbone.parameters()) > 0.0
    assert _gradient_norm(model.mtp_head.parameters()) > 0.0


def test_dormant_mode_rejects_meaningless_gradient_modes() -> None:
    for mode in ("stop_gradient", "frozen_probe"):
        with pytest.raises(ValueError, match="dormant MTP"):
            MTPConfig(
                mode="dormant",
                gradient_mode=mode,  # type: ignore[arg-type]
            )


def test_frozen_probe_requires_positive_auxiliary_weight() -> None:
    with pytest.raises(ValueError, match="positive MTP loss_weight"):
        MTPConfig(
            mode="trajectory",
            gradient_mode="frozen_probe",
            loss_weight=0.0,
        )
