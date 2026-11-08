"""Independent mathematical probes on CPU scalars only; no native model."""
import pytest
import torch

from experiments.rineya_taps_adaptive_recurrence_campaign_v2 import cpu_causal_replay_native as replay


def test_opposite_and_zero_mean_FD_are_uncertain_without_relaxing_gate():
    def triplet(derivative, finite_difference):
        return {
            "nominal": {"loss": 1.0, "sensitivity": derivative},
            "plus": {"loss": 1.0 + .01 * finite_difference},
            "minus": {"loss": 1.0 - .01 * finite_difference},
        }
    for finite_difference in (0.0, -.00001):
        result = replay.aggregate_case([triplet(.00101, finite_difference)], {}, {"balance": .5})
        assert not result["direction_neutral"]
        assert result["mean_sign_numerically_uncertain"]
        assert result["mean_level_FD"]["absolute_error"] <= result["mean_level_FD"]["registered_tolerance"]
    with pytest.raises(FloatingPointError, match="discrepancy"):
        replay.aggregate_case([triplet(.1, -.1)], {}, {"balance": .5})


def test_EOS_mass_gradient_matches_closed_form_and_uniform_shift_invariance():
    logits = torch.tensor([.3, -1.0, 1.2, -2.1], dtype=torch.float32, requires_grad=True)
    accepted = (1, 3)
    loss, canonical = replay.objective(logits, kind="EOS", eos_ids=accepted)
    actual, = torch.autograd.grad(loss, logits)
    expected = logits.detach().softmax(0)
    expected[list(accepted)] -= logits.detach()[list(accepted)].softmax(0)
    assert torch.allclose(actual, expected, atol=1e-7)
    assert abs(float(actual.sum())) < 1e-6
    shifted, _ = replay.objective(logits.detach() + 7, kind="EOS", eos_ids=accepted)
    assert float(shifted) == pytest.approx(float(loss), abs=1e-6)
    assert float(loss) < float(canonical)
