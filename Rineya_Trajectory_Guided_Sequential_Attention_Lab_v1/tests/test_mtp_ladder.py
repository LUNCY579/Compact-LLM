from __future__ import annotations

import torch

from tgsa.mtp import (
    MTPConfig,
    build_tgsa_mtp_model,
    combined_tgsa_mtp_loss,
    multi_token_prediction_loss,
)
from tgsa.phase1_model import TinyLMConfig, build_canonical_supernet, state_dict_sha256


def _config(arm: str) -> TinyLMConfig:
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


def _canonical_state(seed: int) -> dict[str, torch.Tensor]:
    canonical = build_canonical_supernet(
        _config("learned_delta"),
        initialization_seed=seed,
    )
    return {key: value.detach().clone() for key, value in canonical.state_dict().items()}


def test_full_registered_tgsa_ladder_runs_with_parameter_matched_mtp() -> None:
    seed = 41
    state = _canonical_state(seed)
    input_ids = torch.randint(0, 96, (2, 8))
    labels = torch.randint(0, 96, (2, 8))
    arms = (
        "cascade",
        "history_control",
        "raw_delta",
        "learned_delta",
        "qkv_conditioned",
        "predictor_corrector",
        "full",
    )
    all_mtp_hashes: set[str] = set()

    for arm in arms:
        models = {
            mode: build_tgsa_mtp_model(
                _config(arm),
                MTPConfig(max_horizon=3, transition_rank=8, mode=mode),  # type: ignore[arg-type]
                initialization_seed=seed,
                canonical_state=state,
            )
            for mode in ("dormant", "backbone")
        }
        all_mtp_hashes.update(
            state_dict_sha256(model.mtp_head) for model in models.values()
        )
        assert len({state_dict_sha256(model.backbone) for model in models.values()}) == 1
        assert len(
            {
                sum(parameter.numel() for parameter in model.parameters())
                for model in models.values()
            }
        ) == 1

        outputs = {
            mode: model(input_ids, labels=labels)
            for mode, model in models.items()
        }
        torch.testing.assert_close(
            outputs["dormant"].logits,
            outputs["backbone"].logits,
        )
        for output in outputs.values():
            assert len(output.mtp_logits) == 2
            assert all(torch.isfinite(logits).all() for logits in output.mtp_logits)

        if arm == "learned_delta":
            loss_mask = torch.zeros_like(labels, dtype=torch.bool)
            loss_mask[:, -2] = True
            objective = combined_tgsa_mtp_loss(
                outputs["dormant"],
                labels,
                loss_mask,
                mtp_config=models["dormant"].mtp_config,
            )
            objective.total_loss.backward()
            assert all(
                parameter.grad is None for parameter in models["dormant"].mtp_head.parameters()
            )

    # The MTP module is arm-independent and initialized from one derived seed.
    assert len(all_mtp_hashes) == 1


def test_full_control_ladder_backpropagates_through_existing_tgsa_coordinate() -> None:
    seed = 43
    state = _canonical_state(seed)
    input_ids = torch.randint(0, 96, (2, 9))
    labels = torch.randint(0, 96, (2, 9))
    trajectory_eligible_arms = (
        "history_control",
        "raw_delta",
        "learned_delta",
        "qkv_conditioned",
        "predictor_corrector",
        "full",
    )

    for arm in trajectory_eligible_arms:
        model = build_tgsa_mtp_model(
            _config(arm),
            MTPConfig(max_horizon=3, transition_rank=8, mode="trajectory"),
            initialization_seed=seed,
            canonical_state=state,
        )
        output = model(input_ids, labels=labels)
        output.trajectory_control.retain_grad()
        loss, _, _ = multi_token_prediction_loss(
            output.mtp_logits,
            labels,
            horizon_weights=model.mtp_config.weights,
        )
        loss.backward()

        assert output.trajectory_control.requires_grad
        assert output.trajectory_control.grad is not None
        assert torch.isfinite(output.trajectory_control.grad).all()
        assert output.trajectory_control.grad.abs().sum() > 0
