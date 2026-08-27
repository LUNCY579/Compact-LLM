from __future__ import annotations

import torch

from rineya_pcsm.real_plans import (
    calibrate_pcsm_plan,
    classical_tensor_plan,
    peer_context,
    without_stabilizer,
)


def weights() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    return {
        f"model.layers.{index}.mlp.down_proj.weight": torch.randn(
            8, 12, generator=generator
        )
        for index in range(4)
    }


def test_peer_context_excludes_self_and_shuffle_changes_structure() -> None:
    values = weights()
    target = "model.layers.2.mlp.down_proj.weight"
    gradient = torch.ones_like(values[target])
    real = peer_context(
        values,
        target_name=target,
        task_gradient=gradient,
        descriptor_dimension=24,
        mode="real",
        seed=17,
    )
    shuffled = peer_context(
        values,
        target_name=target,
        task_gradient=gradient,
        descriptor_dimension=24,
        mode="shuffled",
        seed=17,
    )
    assert real[-1] is not None
    assert target not in real[-1].peer_names
    assert not torch.equal(real[3].element, shuffled[3].element)


def test_real_and_shuffled_have_equal_capacity_and_exact_k() -> None:
    values = weights()
    target = "model.layers.2.mlp.down_proj.weight"
    gradient = torch.randn_like(values[target])
    meta = gradient + 0.1
    results = [
        calibrate_pcsm_plan(
            values,
            target_name=target,
            calibration_gradient=gradient,
            meta_gradient=meta,
            base_fingerprint="a" * 64,
            k=9,
            seed=17,
            mode=mode,
            steps=2,
        )
        for mode in ("real", "shuffled")
    ]
    assert all(result.plan.entries[0].k == 9 for result in results)
    assert all(len(result.trace) == 2 for result in results)


def test_classical_controls_and_stabilizer_ablation_preserve_support() -> None:
    value = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    gradient = torch.flip(value, dims=(0, 1))
    plan = classical_tensor_plan(
        value,
        gradient,
        target_name="target",
        base_fingerprint="b" * 64,
        k=5,
        seed=11,
        method="gradient",
    )
    ablated = without_stabilizer(plan)
    assert torch.equal(plan.entries[0].support_mask, ablated.entries[0].support_mask)
    assert torch.equal(
        ablated.entries[0].stabilizer, torch.ones_like(ablated.entries[0].stabilizer)
    )
