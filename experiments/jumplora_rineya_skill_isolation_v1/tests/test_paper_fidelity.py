import pytest
import torch
from torch import nn

from experiments.jumplora_rineya_skill_isolation_v1.jump_lora import (
    JumpLoRALinear,
    JumpSchedule,
    LoRALinear,
    symmetric_jump_relu,
)
from experiments.jumplora_rineya_skill_isolation_v1.integration import inject_arm


def test_scale_is_outside_jump_threshold():
    m = JumpLoRALinear(
        nn.Linear(2, 2, bias=False),
        rank=1,
        alpha=4.0,
        schedule=JumpSchedule(0, 1),
        threshold=0.1,
    )
    with torch.no_grad():
        m.lora_A.copy_(torch.tensor([[1.0, 0.25]]))
        m.lora_B.copy_(torch.tensor([[0.2], [0.2]]))
    m.set_step(1)

    torch.testing.assert_close(
        m.raw_delta(),
        torch.tensor([[0.2, 0.05], [0.2, 0.05]]),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        m.dense_delta(),
        torch.tensor([[0.8, 0.2], [0.8, 0.2]]),
        rtol=0.0,
        atol=0.0,
    )
    assert m.support_mask().tolist() == [[True, False], [True, False]]
    torch.testing.assert_close(
        m.effective_delta(),
        torch.tensor([[0.8, 0.0], [0.8, 0.0]]),
        rtol=0.0,
        atol=0.0,
    )


def test_dense_lora_scale_still_applies():
    m = LoRALinear(nn.Linear(2, 2, bias=False), rank=1, alpha=4.0)
    with torch.no_grad():
        m.lora_A.fill_(1.0)
        m.lora_B.fill_(0.5)
    torch.testing.assert_close(
        m.dense_delta(),
        (m.lora_B @ m.lora_A) * 4.0,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(m.effective_delta(), m.dense_delta(), rtol=0.0, atol=0.0)


def test_nonzero_hand_ste():
    x = torch.tensor([0.55, -0.8, 0.1], requires_grad=True)
    tau = torch.tensor(0.5, requires_grad=True)
    symmetric_jump_relu(x, tau, 0.2).sum().backward()
    assert tau.grad.item() == pytest.approx(-2.5)
    assert x.grad.tolist() == pytest.approx([1.0, 1.0, 0.0])


def test_per_module_thresholds_and_matched_counts():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(2, 2, bias=False)
            self.v_proj = nn.Linear(2, 3, bias=False)

    model = M()
    h = inject_arm(model, "J3", rank=1, alpha=4, schedule=JumpSchedule(0, 1))
    for binding in h.bindings:
        with torch.no_grad():
            binding.adapter.lora_A.copy_(torch.tensor([[1.0, 0.3]]))
            binding.adapter.lora_B.copy_(
                torch.arange(
                    1,
                    binding.adapter.out_features + 1,
                    dtype=torch.float32,
                ).reshape(-1, 1)
                * 0.2
            )

    counts = h.paper_initial_support_counts()
    assert h.initialize_jump_thresholds(counts) == counts
    thresholds = {float(binding.adapter.threshold.detach()) for binding in h.bindings}
    assert len(thresholds) == 2

    model2 = M()
    h2 = inject_arm(model2, "J1", rank=1, alpha=4, schedule=JumpSchedule(0, 1))
    for src, dst in zip(h.bindings, h2.bindings):
        with torch.no_grad():
            dst.adapter.lora_A.copy_(src.adapter.lora_A)
            dst.adapter.lora_B.copy_(src.adapter.lora_B)
    assert h2.freeze_matched_support_counts(counts) == counts
