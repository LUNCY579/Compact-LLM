from __future__ import annotations

import pytest

from rineya_pcsm.statistics import (
    paired_bootstrap_interval,
    paired_comparison,
    paired_differences,
    paired_sign_flip_test,
    summarize_sample,
)


def test_paired_differences_validate_shape_and_direction() -> None:
    assert paired_differences([3.0, 5.0], [1.0, 2.0]).tolist() == [2.0, 3.0]
    with pytest.raises(ValueError, match="identical shapes"):
        paired_differences([1.0], [1.0, 2.0])


def test_paired_bootstrap_is_deterministic_and_excludes_zero_for_clear_effect() -> None:
    first = [2.0, 2.2, 2.4, 2.6, 2.8]
    second = [1.0, 1.1, 1.2, 1.3, 1.4]
    first_interval = paired_bootstrap_interval(first, second, resamples=2_000, seed=17)
    second_interval = paired_bootstrap_interval(first, second, resamples=2_000, seed=17)
    assert first_interval == second_interval
    assert first_interval.low > 0.0


def test_exact_sign_flip_p_value_is_enumerated() -> None:
    result = paired_sign_flip_test([1.0, 1.0, 1.0], [0.0, 0.0, 0.0])
    assert result.exact
    assert result.permutations == 8
    assert result.p_value == pytest.approx(0.25)


def test_comparison_bundles_raw_paired_effect_and_statistics() -> None:
    result = paired_comparison(
        [1.5, 1.7, 1.9, 2.1, 2.3],
        [1.0, 1.1, 1.2, 1.3, 1.4],
        bootstrap_resamples=1_000,
        seed=23,
    )
    assert result["difference_definition"] == "first_minus_second"
    assert len(result["differences"]) == 5  # type: ignore[arg-type]
    assert result["sign_flip_test"]["exact"] is True  # type: ignore[index]
    assert summarize_sample([1.0, 2.0, 3.0])["mean"] == pytest.approx(2.0)
