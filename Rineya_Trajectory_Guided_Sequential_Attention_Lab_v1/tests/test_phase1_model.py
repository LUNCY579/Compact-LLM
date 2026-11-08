from __future__ import annotations

import pytest
import torch

from tgsa.phase1_model import (
    ControlIntervention,
    TinyLMConfig,
    answer_logits,
    build_model,
    masked_answer_loss,
    paired_initialization_audit,
    state_dict_sha256,
)


def _config(arm: str = "learned_delta") -> TinyLMConfig:
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


def _batch(batch_size: int = 4, sequence: int = 12) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(77)
    input_ids = torch.randint(0, 96, (batch_size, sequence), generator=generator)
    key_padding_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    return input_ids, key_padding_mask


def test_history_and_delta_are_exactly_parameter_and_initialization_matched() -> None:
    audit = paired_initialization_audit(
        base_config=_config("learned_delta"),
        initialization_seed=31415,
    )

    assert audit["history_delta_exact_state_structure"] is True
    assert audit["history_delta_exact_initialization"] is True
    assert audit["history_delta_parameter_count_equal"] is True
    assert audit["history_trainable_parameters"] == audit["learned_delta_trainable_parameters"]
    assert audit["cascade_delta_shared_initialization"]["identical"] is True


def test_canonical_initializer_matches_shared_cascade_tensors() -> None:
    seed = 991
    cascade = build_model(_config("cascade"), initialization_seed=seed)
    delta = build_model(_config("learned_delta"), initialization_seed=seed)
    cascade_state = cascade.state_dict()
    delta_state = delta.state_dict()

    common = sorted(
        key
        for key in cascade_state.keys() & delta_state.keys()
        if cascade_state[key].shape == delta_state[key].shape
        and cascade_state[key].dtype == delta_state[key].dtype
    )
    assert common
    assert all(torch.equal(cascade_state[key], delta_state[key]) for key in common)


def test_required_arms_have_finite_forward_and_backward() -> None:
    input_ids, key_padding_mask = _batch()
    answer_indices = torch.tensor([3, 4, 5, 6])
    answer_tokens = torch.tensor([1, 2, 3, 4])
    labels = torch.full(input_ids.shape, -100, dtype=torch.long)
    loss_mask = torch.zeros(input_ids.shape, dtype=torch.bool)
    rows = torch.arange(input_ids.shape[0])
    labels[rows, answer_indices] = answer_tokens
    loss_mask[rows, answer_indices] = True

    for arm in ("cascade", "history_control", "learned_delta"):
        model = build_model(_config(arm), initialization_seed=123)
        output = model(input_ids, key_padding_mask=key_padding_mask)
        loss = masked_answer_loss(output.logits, labels, loss_mask)
        loss.backward()

        assert output.logits.shape == (4, 12, 96)
        assert torch.isfinite(loss)
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )

        if arm == "learned_delta":
            trajectory_gradients = [
                parameter.grad
                for name, parameter in model.named_parameters()
                if "trajectory_encoders" in name and parameter.grad is not None
            ]
            assert trajectory_gradients
            assert sum(float(gradient.abs().sum()) for gradient in trajectory_gradients) > 0.0


def test_tiny_model_preserves_causal_prefix_invariance() -> None:
    model = build_model(_config("learned_delta"), initialization_seed=456).eval()
    input_ids, key_padding_mask = _batch(batch_size=2, sequence=14)
    changed = input_ids.clone()
    changed[:, 8:] = (changed[:, 8:] + 17) % model.config.vocab_size

    with torch.inference_mode():
        original_logits = model(input_ids, key_padding_mask=key_padding_mask).logits
        changed_logits = model(changed, key_padding_mask=key_padding_mask).logits

    torch.testing.assert_close(original_logits[:, :8], changed_logits[:, :8], atol=1.0e-6, rtol=1.0e-6)


def test_interventions_are_finite_distribution_audited_and_hooks_cleanup() -> None:
    model = build_model(_config("learned_delta"), initialization_seed=222).eval()
    input_ids, key_padding_mask = _batch(batch_size=4)
    hashes: dict[str, str] = {}

    with torch.inference_mode():
        baseline = model(input_ids, key_padding_mask=key_padding_mask).logits
        hashes["identity"] = state_dict_sha256(model)
        for mode in ("zero", "negate", "roll_batch"):
            with ControlIntervention(
                model,
                mode,  # type: ignore[arg-type]
                capture_records=True,
                max_records=64,
            ) as intervention:
                logits = model(input_ids, key_padding_mask=key_padding_mask).logits
                assert torch.isfinite(logits).all()
                assert intervention.records
                if mode == "roll_batch":
                    for record in intervention.records:
                        assert record["output_rms"] == pytest.approx(record["input_rms"], abs=1.0e-7)
            assert sum(len(module._forward_pre_hooks) for module in model.modules()) == 0
            hashes[mode] = state_dict_sha256(model)

    assert torch.isfinite(baseline).all()
    assert len(set(hashes.values())) == 1, "inference interventions must not mutate model weights"


def test_batch_roll_rejects_singleton_batches() -> None:
    model = build_model(_config("learned_delta"), initialization_seed=333).eval()
    input_ids, key_padding_mask = _batch(batch_size=1)

    with ControlIntervention(model, "roll_batch"):
        with pytest.raises(ValueError, match="batch size"):
            model(input_ids, key_padding_mask=key_padding_mask)

    assert sum(len(module._forward_pre_hooks) for module in model.modules()) == 0


def test_answer_logit_gather_and_masked_loss_contract() -> None:
    logits = torch.randn(3, 7, 11, generator=torch.Generator().manual_seed(9))
    answer_indices = torch.tensor([1, 3, 5])
    gathered = answer_logits(logits, answer_indices)
    assert gathered.shape == (3, 11)
    for row, index in enumerate(answer_indices.tolist()):
        torch.testing.assert_close(gathered[row], logits[row, index])

    labels = torch.full((3, 7), -100, dtype=torch.long)
    mask = torch.zeros((3, 7), dtype=torch.bool)
    labels[torch.arange(3), answer_indices] = torch.tensor([2, 4, 6])
    mask[torch.arange(3), answer_indices] = True
    loss = masked_answer_loss(logits, labels, mask)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
