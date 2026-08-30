from __future__ import annotations

import torch

from rineya_cross_scale.cka import linear_cka, monotonic_alignment, pairwise_cka


def test_linear_cka_is_one_for_identical_spaces() -> None:
    generator = torch.Generator().manual_seed(0)
    base = torch.randn(32, 8, generator=generator)
    rotated = base @ torch.linalg.qr(torch.randn(8, 8, generator=generator)).Q
    assert abs(linear_cka(base, rotated) - 1.0) < 1e-5


def test_monotonic_alignment_is_strictly_increasing() -> None:
    similarity = torch.tensor(
        [
            [0.9, 0.2, 0.1, 0.0],
            [0.3, 0.8, 0.4, 0.1],
            [0.1, 0.3, 0.85, 0.2],
        ]
    )
    result = monotonic_alignment(similarity)
    assert result["mapping"] == [0, 1, 2]
    assert result["strictly_increasing"] is True


def test_pairwise_cka_shape() -> None:
    target = [torch.randn(16, 4) for _ in range(3)]
    donor = [torch.randn(16, 6) for _ in range(5)]
    matrix = pairwise_cka(target, donor)
    assert tuple(matrix.shape) == (3, 5)
