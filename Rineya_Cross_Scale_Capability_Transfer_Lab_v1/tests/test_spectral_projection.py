from __future__ import annotations

import torch

from rineya_cross_scale.spectral_projection import spectral_profile_transfer


def test_spectral_transfer_is_deterministic_and_norm_preserving() -> None:
    generator = torch.Generator().manual_seed(17)
    target = torch.randn(20, 12, generator=generator)
    donor = torch.randn(30, 18, generator=generator) * 2.0
    first, first_stats = spectral_profile_transfer(
        target, donor, rank=4, strength=1.0, seed_key="x"
    )
    second, second_stats = spectral_profile_transfer(
        target, donor, rank=4, strength=1.0, seed_key="x"
    )
    assert torch.equal(first, second)
    assert first_stats == second_stats
    assert first.shape == target.shape
    assert torch.isfinite(first).all()
    assert torch.linalg.vector_norm(first).item() == pytest.approx(
        torch.linalg.vector_norm(target).item(), rel=1e-5
    )
    assert first_stats["relative_change"] > 0


import pytest
