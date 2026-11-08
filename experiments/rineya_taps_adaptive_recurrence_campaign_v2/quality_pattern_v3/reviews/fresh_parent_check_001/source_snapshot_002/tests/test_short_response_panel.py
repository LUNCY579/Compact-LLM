"""Copying panel source invariants; no models or emitted evaluation corpus."""
import pytest
from experiments.rineya_taps_adaptive_recurrence_campaign_v2.quality_pattern_v3.short_response_panel import (
    build_panel, solve_prompt, score_response,
)


def test_complete_panel_and_prompt_only_positive_control():
    rows = build_panel()
    assert len(rows) == len({r["case_id"] for r in rows}) == 24
    assert all(solve_prompt(r["prompt"]) == r["gold"] for r in rows)
    assert all(r["max_output_tokens"] == 8 for r in rows)


def test_full_response_scorer_preserves_case_and_punctuation():
    assert score_response(" READY\n", "READY")["correct"]
    for response in ("ready", "READY.", "The answer is READY", "READY ignored"):
        assert not score_response(response, "READY")["correct"]
    assert not score_response("blue  sky", "blue sky")["correct"]


def test_unsupported_prompt_rejected():
    with pytest.raises(ValueError):
        solve_prompt("Reply READY and then continue")
