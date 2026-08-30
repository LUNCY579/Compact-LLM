"""The escalation ladder must permit a diagnosed response and nothing else.

The failure mode this guards is not "the gate was too strict" -- it is a gate
that gets re-run with a slightly different regime until it happens to pass. Every
test below is a way that could happen.
"""

from __future__ import annotations

import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
if str(LAB / "scripts") not in sys.path:
    sys.path.insert(0, str(LAB / "scripts"))

from phase1_v4_escalation import (  # noqa: E402
    BASE,
    BOTH,
    MEASUREMENT_CAP,
    MEMORIZATION_CHECK,
    MIXTURE_CHECK,
    MORE_DATA,
    MORE_UPDATES,
    next_step,
)

ALL_CHECKS = (
    "all_arms_beat_trivial", "all_arms_reach_absolute_floor",
    "answer_nll_beats_uniform", "every_primary_family_competent",
    "predictions_are_input_dependent", "no_primary_family_saturated",
    MEMORIZATION_CHECK, MIXTURE_CHECK,
)


def _report(examples: int, steps: int, *failed: str) -> dict:
    return {
        "verdict": "GO" if not failed else "NO_GO",
        "checks": {name: name not in failed for name in ALL_CHECKS},
        "regime": {"train_examples": examples, "steps": steps},
    }


def test_the_base_regime_is_the_first_move() -> None:
    step, _ = next_step([])
    assert step == BASE
    assert (step.train_examples, step.steps) == (50_000, 20_000)


def test_a_mixture_failure_escalates_the_update_budget() -> None:
    step, reason = next_step([_report(50_000, 20_000, MIXTURE_CHECK)])
    assert step == MORE_UPDATES
    assert (step.train_examples, step.steps) == (50_000, 30_000)
    assert "truncates" in reason


def test_a_memorization_failure_escalates_data_not_budget() -> None:
    """More updates at a fixed batch size means more epochs, which is the cause."""

    step, reason = next_step([_report(50_000, 20_000, MEMORIZATION_CHECK)])
    assert step == MORE_DATA
    assert (step.train_examples, step.steps) == (200_000, 20_000)
    assert "epochs" in reason


def test_both_failures_compose() -> None:
    step, _ = next_step([_report(50_000, 20_000, MIXTURE_CHECK, MEMORIZATION_CHECK)])
    assert step == BOTH
    assert (step.train_examples, step.steps) == (200_000, 30_000)


def test_a_go_ends_the_ladder() -> None:
    step, reason = next_step([_report(50_000, 20_000)])
    assert step is None
    assert "already exists" in reason


def test_a_failure_with_no_registered_response_rejects_the_regime() -> None:
    """A dead family is not a budget problem, and no rung addresses it."""

    step, reason = next_step([_report(50_000, 20_000, "every_primary_family_competent")])
    assert step is None
    assert "no registered response" in reason
    assert "REJECT" in reason


def test_a_saturated_ceiling_is_also_unanswerable() -> None:
    step, reason = next_step([_report(50_000, 20_000, "no_primary_family_saturated")])
    assert step is None
    assert "REJECT" in reason


def test_a_mixed_failure_is_unanswerable_if_any_part_is() -> None:
    """A memorization gap AND a dead family must not be treated as answerable."""

    step, reason = next_step(
        [_report(50_000, 20_000, MEMORIZATION_CHECK, "answer_nll_beats_uniform")]
    )
    assert step is None
    assert "answer_nll_beats_uniform" in reason


def test_the_response_is_relative_to_the_regime_that_failed() -> None:
    """A 200k regime failing on mixture must escalate updates, not cut data.

    An earlier revision hard-coded the rungs as absolute (examples, steps) pairs,
    so this case returned "50k x 30k" -- a REDUCTION in data as the response to a
    problem data had already solved.
    """

    step, _ = next_step([
        _report(50_000, 20_000, MEMORIZATION_CHECK),
        _report(200_000, 20_000, MIXTURE_CHECK),
    ])
    assert step is not None
    assert (step.train_examples, step.steps) == (200_000, 30_000)


def test_a_mid_ladder_memorization_failure_keeps_the_larger_budget() -> None:
    step, _ = next_step([
        _report(50_000, 20_000, MIXTURE_CHECK),
        _report(50_000, 30_000, MEMORIZATION_CHECK),
    ])
    assert step is not None
    assert (step.train_examples, step.steps) == (200_000, 30_000)


def test_a_regime_already_at_maximum_data_is_rejected() -> None:
    step, reason = next_step([
        _report(50_000, 20_000, MEMORIZATION_CHECK),
        _report(200_000, 20_000, MEMORIZATION_CHECK),
    ])
    assert step is None
    assert "already at the registered maximum of 200000 examples" in reason
    assert "REJECT" in reason


def test_a_regime_already_at_maximum_updates_is_rejected() -> None:
    step, reason = next_step([
        _report(50_000, 30_000, MIXTURE_CHECK),
    ])
    assert step is None
    assert "already at the registered maximum of 30000 updates" in reason


def test_the_top_of_the_ladder_rejects_rather_than_repeating() -> None:
    step, reason = next_step([
        _report(50_000, 20_000, MIXTURE_CHECK, MEMORIZATION_CHECK),
        _report(200_000, 30_000, MIXTURE_CHECK, MEMORIZATION_CHECK),
    ])
    assert step is None
    assert "REJECT" in reason


def test_a_rung_cannot_be_taken_twice() -> None:
    """If the registered response was already measured, the diagnosis was wrong."""

    step, reason = next_step([
        _report(200_000, 20_000, MIXTURE_CHECK),
        _report(200_000, 30_000, MEMORIZATION_CHECK),
        _report(200_000, 20_000, MIXTURE_CHECK),
    ])
    assert step is None
    assert "already been measured" in reason


def test_the_cap_counts_distinct_regimes() -> None:
    history = [
        _report(50_000, 20_000, MIXTURE_CHECK),
        _report(50_000, 30_000, MIXTURE_CHECK),
        _report(200_000, 20_000, MIXTURE_CHECK),
        _report(200_000, 30_000, MIXTURE_CHECK),
    ]
    step, reason = next_step(history)
    assert step is None
    assert "REJECTED" in reason
    assert f"{MEASUREMENT_CAP} distinct regimes" in reason


def test_re_measuring_one_regime_does_not_consume_the_cap() -> None:
    """A correction after a documented harness fix is not an escalation.

    It does not change what is being tested, so it cannot be a way of searching
    for a regime that passes. Repeated re-measurement WITHOUT a fix is caught by
    the rung-already-attempted rule instead.
    """

    history = [_report(50_000, 20_000, MEMORIZATION_CHECK) for _ in range(6)]
    step, reason = next_step(history)
    assert step == MORE_DATA, reason


def test_the_ladder_terminates_from_every_starting_point() -> None:
    """Exhaustive: no sequence of permitted moves can run forever."""

    from itertools import product

    answerable = (MIXTURE_CHECK, MEMORIZATION_CHECK)
    for failures in product(answerable, repeat=3):
        history: list[dict] = []
        step, _ = next_step(history)
        moves = 0
        while step is not None:
            moves += 1
            assert moves <= MEASUREMENT_CAP, f"ladder did not terminate for {failures}"
            history.append(_report(step.train_examples, step.steps,
                                   failures[min(moves - 1, len(failures) - 1)]))
            step, _ = next_step(history)
