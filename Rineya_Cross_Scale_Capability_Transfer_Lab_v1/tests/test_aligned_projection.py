from __future__ import annotations

import torch

from rineya_cross_scale.aligned_projection import (
    deterministic_row_ids,
    fit_embedding_alignment,
    fit_embedding_procrustes,
)


def test_deterministic_row_ids() -> None:
    assert deterministic_row_ids(16, 4, 0.5) == (2, 6, 10, 14)


def test_fit_embedding_alignment_recovers_linear_relation() -> None:
    generator = torch.Generator().manual_seed(17)
    donor = torch.randn(64, 8, generator=generator)
    true_map = torch.randn(8, 4, generator=generator)
    target = donor @ true_map
    alignment = fit_embedding_alignment(
        target,
        donor,
        sample_count=32,
        heldout_count=16,
        ridge_relative=1e-6,
    )
    projected = donor @ alignment.target_from_donor.T
    relative_error = torch.linalg.vector_norm(projected - target) / torch.linalg.vector_norm(
        target
    )
    assert relative_error.item() < 1e-3
    assert alignment.metrics["heldout_mean_cosine"] > 0.999
    assert alignment.heldout_ids
    assert not (set(alignment.sample_ids) & set(alignment.heldout_ids))


def test_procrustes_alignment_has_exact_target_cycle() -> None:
    generator = torch.Generator().manual_seed(23)
    donor, _ = torch.linalg.qr(torch.randn(128, 8, generator=generator))
    q, _ = torch.linalg.qr(torch.randn(8, 4, generator=generator))
    target = donor @ q * 0.7
    alignment = fit_embedding_procrustes(
        target,
        donor,
        sample_count=64,
        heldout_count=32,
    )
    assert alignment.metrics["heldout_mean_cosine"] > 0.98
    assert alignment.metrics["target_cycle_relative_error"] < 1e-5
    assert abs(alignment.scale - 0.7) < 0.03
