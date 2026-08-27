from __future__ import annotations

import torch

from rineya_pcsm.integration import to_native_update_plan
from rineya_pcsm.plan import PCSMPlan, TensorPlan


def test_converter_selects_only_supported_stabilizers() -> None:
    mask = torch.tensor([[False, True], [True, False]])
    stabilizer = torch.tensor([[0.25, 0.4], [0.9, 1.0]], dtype=torch.float32)
    rich = PCSMPlan(
        base_fingerprint="a" * 64,
        descriptor_dimension=24,
        entries=(
            TensorPlan(
                name="weight",
                role="role",
                shape=(2, 2),
                support_mask=mask,
                stabilizer=stabilizer,
                peer_names=("peer",),
                attention_alphas=torch.ones(1),
            ),
        ),
    )
    native = to_native_update_plan(rich)
    assert native.entries[0].flat_indices.tolist() == [1, 2]
    assert torch.equal(native.entries[0].stabilizer, torch.tensor([0.4, 0.9]))
