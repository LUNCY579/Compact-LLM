from __future__ import annotations

import torch

from rineya_cross_scale.direction_transfer import (
    project_weight,
    ridge_linear_map,
    spectral_direction_transfer,
)


def test_ridge_recovers_linear_map() -> None:
    generator = torch.Generator().manual_seed(1)
    true = torch.randn(6, 4, generator=generator)
    source = torch.randn(40, 6, generator=generator)
    target = source @ true
    estimated = ridge_linear_map(source, target, ridge=1e-6)
    assert torch.allclose(estimated, true, atol=1e-4)


def test_project_weight_shapes() -> None:
    donor = torch.randn(8, 6)
    input_map = torch.eye(6, 3)
    output_map = torch.eye(8, 4)
    projected = project_weight(donor, input_map, output_map)
    assert tuple(projected.shape) == (4, 3)


def test_spectral_direction_transfer_preserves_target_shape() -> None:
    target = torch.randn(4, 5)
    donor = torch.randn(8, 10)
    mixed, stats = spectral_direction_transfer(target, donor, rank=3, strength=0.1)
    assert tuple(mixed.shape) == (4, 5)
    assert 0 < stats["donor_kept_energy"] <= 1
    zero, _ = spectral_direction_transfer(target, donor, rank=3, strength=0.0)
    assert torch.allclose(zero, target)
