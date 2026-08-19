from __future__ import annotations

from rineya_state_lab.evaluation.stats import exact_mcnemar, paired_bootstrap_difference


def test_mcnemar_no_discordance() -> None:
    result = exact_mcnemar([True, False], [True, False])
    assert result["p_value_two_sided"] == 1.0


def test_paired_bootstrap_direction() -> None:
    result = paired_bootstrap_difference([0] * 20, [1] * 20, samples=500, seed=1)
    assert result["lower"] > 0
