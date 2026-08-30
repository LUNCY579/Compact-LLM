from __future__ import annotations

import pytest

from tgsa.phase2_analysis import paired_factorial_effects


def _records() -> list[dict[str, object]]:
    rows = []
    for seed, jitter in ((1, 0.00), (2, 0.01), (3, -0.01)):
        cells = {
            "parallel_ar": 0.50 + jitter,
            "parallel_mtp": 0.55 + jitter,
            "tgsa_ar": 0.60 + jitter,
            "tgsa_mtp": 0.72 + jitter,
        }
        for arm, value in cells.items():
            rows.append({"seed": seed, "arm": arm, "score": value})
    return rows


def test_factorial_effects_and_interaction_are_paired_and_oriented() -> None:
    analysis = paired_factorial_effects(
        _records(),
        metric="score",
        bootstrap_samples=500,
    )
    effects = analysis["effects"]
    assert effects["mtp_on_parallel"]["mean"] == pytest.approx(0.05)
    assert effects["mtp_on_tgsa"]["mean"] == pytest.approx(0.12)
    assert effects["tgsa_without_mtp"]["mean"] == pytest.approx(0.10)
    assert effects["interaction"]["mean"] == pytest.approx(0.07)
    assert effects["combined_vs_parallel_ar"]["mean"] == pytest.approx(0.22)


def test_incomplete_seed_block_fails_closed() -> None:
    rows = _records()
    rows = [row for row in rows if not (row["seed"] == 2 and row["arm"] == "tgsa_mtp")]
    with pytest.raises(ValueError, match="incomplete paired seed blocks"):
        paired_factorial_effects(rows, metric="score")
