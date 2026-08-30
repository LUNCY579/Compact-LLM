from __future__ import annotations

from rineya_cross_scale.validation import short_period_repetition


def test_short_period_repetition() -> None:
    assert short_period_repetition([1, 2, 1, 2, 1, 2, 1, 2])
    assert short_period_repetition([3, 3, 3, 3, 3, 3])
    assert not short_period_repetition([1, 2, 3, 4, 5, 6, 7, 8])
    assert not short_period_repetition([1, 2, 1, 2])
