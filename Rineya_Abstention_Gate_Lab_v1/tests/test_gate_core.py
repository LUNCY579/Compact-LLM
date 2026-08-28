from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import gate_core as core  # noqa: E402
import sweep_thresholds as sweep  # noqa: E402


def make_row(
    row_id: str,
    prediction: str,
    answerable: bool,
    **signals: float,
) -> core.GateRow:
    accepted = (
        frozenset({core.SCORER.normalize("Talia Frost")})
        if answerable
        else frozenset(core.ABSTENTION_VOCABULARY)
    )
    values = {name: 0.0 for name in core.SIGNALS}
    values.update(signals)
    return core.GateRow(
        row_id=row_id,
        prediction=prediction,
        answerable=answerable,
        accepted=accepted,
        signals=values,
    )


# --------------------------------------------------------------------------
# McNemar
# --------------------------------------------------------------------------


def test_mcnemar_reproduces_the_published_cascade_p_values():
    # These two are quoted in the audit that motivated this lab.
    assert core.mcnemar_exact_p(13, 6) == pytest.approx(0.167, abs=5e-4)
    assert core.mcnemar_exact_p(9, 5) == pytest.approx(0.424, abs=5e-4)


def test_mcnemar_is_symmetric_and_bounded():
    for b, c in ((0, 0), (1, 0), (7, 3), (30, 30), (2, 19)):
        p = core.mcnemar_exact_p(b, c)
        assert p == core.mcnemar_exact_p(c, b)
        assert 0.0 <= p <= 1.0
    assert core.mcnemar_exact_p(0, 0) == 1.0
    assert core.mcnemar_exact_p(1, 0) == 1.0  # 2 * 0.5, clipped at 1


def test_mcnemar_rejects_negative_counts():
    with pytest.raises(ValueError):
        core.mcnemar_exact_p(-1, 3)


# --------------------------------------------------------------------------
# Signals and thresholds
# --------------------------------------------------------------------------


def test_every_signal_has_a_direction_and_reads_the_confidence_block():
    confidence = {
        "first_token": {
            "logprob_margin": 2.0,
            "prob_margin": 0.5,
            "token_logprob": -0.1,
            "entropy": 0.3,
            "top1_logprob": -0.1,
        },
        "answer": {
            "sum_logprob": -0.4,
            "mean_logprob": -0.2,
            "min_logprob": -0.3,
            "mean_logprob_margin": 3.0,
            "min_logprob_margin": 2.0,
            "mean_entropy": 0.25,
            "max_entropy": 0.3,
        },
        "unknown": {"exact_logprob": -8.0, "variant_logprob": -7.5},
    }
    for name in core.SIGNALS:
        assert core.signal_direction(name) in {"low", "high"}
        assert math.isfinite(core.signal_value(name, confidence))
    assert core.signal_value("unknown_exact_minus_top1", confidence) == pytest.approx(-7.9)


def test_signal_value_rejects_non_finite():
    confidence = {"unknown": {"exact_logprob": float("-inf")}}
    with pytest.raises(ValueError):
        core.signal_value("unknown_exact_logprob", confidence)


def test_abstains_respects_direction_at_the_boundary():
    assert core.abstains("first_token_logprob_margin", 1.0, 1.0) is True
    assert core.abstains("first_token_logprob_margin", 1.1, 1.0) is False
    assert core.abstains("unknown_exact_logprob", -5.0, -5.0) is True
    assert core.abstains("unknown_exact_logprob", -5.1, -5.0) is False


def test_candidate_thresholds_enumerate_every_achievable_partition():
    values = [-3.0, -1.0, -1.0, 0.5, 4.0]
    for direction in ("low", "high"):
        cuts = core.candidate_thresholds(values, direction)
        array = np.array(values)
        masks = {
            tuple(array <= cut if direction == "low" else array >= cut) for cut in cuts
        }
        # 4 distinct values -> 5 partitions, including empty and full.
        assert len(masks) == 5
        assert tuple([False] * len(values)) in masks
        assert tuple([True] * len(values)) in masks


def test_candidate_thresholds_never_sit_on_an_observed_value():
    values = [-3.0, -1.0, 0.5, 4.0]
    assert not (set(core.candidate_thresholds(values, "low")) & set(values))


def test_candidate_thresholds_handles_a_constant_signal():
    cuts = core.candidate_thresholds([2.0, 2.0, 2.0], "high")
    array = np.array([2.0, 2.0, 2.0])
    masks = {tuple(array >= cut) for cut in cuts}
    assert masks == {(True, True, True), (False, False, False)}


# --------------------------------------------------------------------------
# Gate specs
# --------------------------------------------------------------------------


def test_single_signal_spec_rejects_a_stray_second_signal():
    with pytest.raises(ValueError):
        core.GateSpec(
            signal="unknown_exact_logprob",
            threshold=-5.0,
            signal_b="first_token_entropy",
        )


def test_pair_spec_requires_both_halves():
    with pytest.raises(ValueError):
        core.GateSpec(signal="unknown_exact_logprob", threshold=-5.0, combine="and")


def test_spec_roundtrips_through_json():
    spec = core.GateSpec(
        signal="answer_mean_entropy",
        threshold=0.36,
        signal_b="unknown_variant_logprob",
        threshold_b=-6.6,
        combine="and",
    )
    restored = core.GateSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert restored == spec
    assert "AND" in core.describe(restored)


def test_and_or_combination_logic():
    row = make_row(
        "x", "Talia Frost", True, unknown_exact_logprob=-4.0, first_token_entropy=0.1
    )
    fires = core.GateSpec(signal="unknown_exact_logprob", threshold=-5.0)
    quiet = core.GateSpec(signal="first_token_entropy", threshold=9.0)
    assert fires.decide(row) is True
    assert quiet.decide(row) is False
    and_spec = core.GateSpec(
        signal="unknown_exact_logprob",
        threshold=-5.0,
        signal_b="first_token_entropy",
        threshold_b=9.0,
        combine="and",
    )
    or_spec = core.GateSpec(
        signal="unknown_exact_logprob",
        threshold=-5.0,
        signal_b="first_token_entropy",
        threshold_b=9.0,
        combine="or",
    )
    assert and_spec.decide(row) is False
    assert or_spec.decide(row) is True


def test_identity_gate_never_fires():
    row = make_row("x", "Talia Frost", True, first_token_logprob_margin=-1e9)
    assert core.IDENTITY_GATE.decide(row) is False


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def sample_rows() -> list[core.GateRow]:
    return [
        # unanswerable, wrongly answered, high UNKNOWN mass -> gate rescues it
        make_row("A-U", "Mosaic", False, unknown_exact_logprob=-1.0),
        # unanswerable, wrongly answered, low UNKNOWN mass -> gate misses it
        make_row("B-U", "Mosaic", False, unknown_exact_logprob=-20.0),
        # unanswerable, already abstained
        make_row("C-U", "UNKNOWN", False, unknown_exact_logprob=-0.5),
        # answerable and correct, high UNKNOWN mass -> gate destroys it
        make_row("D-B", "Talia Frost", True, unknown_exact_logprob=-1.0),
        # answerable and correct, low UNKNOWN mass -> untouched
        make_row("E-B", "Talia Frost", True, unknown_exact_logprob=-20.0),
        # answerable and wrong, low UNKNOWN mass -> untouched, still wrong
        make_row("F-B", "Mosaic", True, unknown_exact_logprob=-20.0),
    ]


def test_baseline_outcome_counts_model_abstentions():
    outcome = core.evaluate_gate(core.IDENTITY_GATE, sample_rows())
    assert outcome.baseline_correct == 3  # C-U, D-B, E-B
    assert outcome.gated_correct == 3
    assert outcome.abstentions == 1
    assert outcome.abstention_true_positives == 1
    assert outcome.abstention_precision == 1.0
    assert outcome.abstention_recall == pytest.approx(1 / 3)
    assert outcome.mcnemar_p == 1.0


def test_gate_gains_and_losses_are_attributed_to_the_right_rows():
    spec = core.GateSpec(signal="unknown_exact_logprob", threshold=-5.0)
    outcome = core.evaluate_gate(spec, sample_rows())
    assert outcome.gained_ids == ("A-U",)
    assert outcome.lost_ids == ("D-B",)
    assert outcome.delta == 0
    assert outcome.gated_correct == 3
    assert outcome.abstentions == 3  # A-U, C-U, D-B
    assert outcome.added_abstentions == 2  # C-U already abstained
    assert outcome.abstention_true_positives == 2
    assert outcome.abstention_false_positives == 1
    assert outcome.abstention_precision == pytest.approx(2 / 3)
    assert outcome.abstention_recall == pytest.approx(2 / 3)
    assert outcome.mcnemar_b == 1 and outcome.mcnemar_c == 1


def test_gate_never_overwrites_an_existing_abstention_with_an_answer():
    rows = sample_rows()
    spec = core.GateSpec(signal="unknown_exact_logprob", threshold=1e9)  # never fires
    for row in rows:
        assert core.gated_prediction(spec, row) == row.prediction


def test_oracle_ceiling_keeps_correct_answerable_rows_and_fixes_every_unanswerable():
    assert core.oracle_ceiling(sample_rows()) == 5  # A-U B-U C-U + D-B E-B


def test_is_abstention_accepts_every_scored_surface_form():
    for text in ("UNKNOWN", "unknown", "Unknown.", "not determinable", "cannot be determined"):
        assert core.is_abstention(text)
    assert not core.is_abstention("Talia Frost")


# --------------------------------------------------------------------------
# build_rows guards
# --------------------------------------------------------------------------

KEY = [
    {"id": "A-U", "answerable": False, "accepted_answers": ["UNKNOWN", "unknown"]},
    {"id": "B-B", "answerable": True, "accepted_answers": ["Talia Frost"]},
]

CONFIDENCE = {
    "first_token": {
        "logprob_margin": 1.0,
        "prob_margin": 0.2,
        "token_logprob": -0.2,
        "entropy": 0.4,
        "top1_logprob": -0.2,
    },
    "answer": {
        "sum_logprob": -0.4,
        "mean_logprob": -0.2,
        "min_logprob": -0.2,
        "mean_logprob_margin": 1.0,
        "min_logprob_margin": 1.0,
        "mean_entropy": 0.4,
        "max_entropy": 0.4,
    },
    "unknown": {"exact_logprob": -6.0, "variant_logprob": -5.0},
}


def ok_row(row_id: str) -> dict:
    return {
        "id": row_id,
        "status": "ok",
        "prediction": "Mosaic",
        "confidence": json.loads(json.dumps(CONFIDENCE)),
    }


def test_build_rows_follows_answer_key_order():
    rows = core.build_rows([ok_row("B-B"), ok_row("A-U")], KEY)
    assert [row.row_id for row in rows] == ["A-U", "B-B"]


def test_build_rows_rejects_a_missing_prediction():
    with pytest.raises(ValueError, match="missing from the confidence file"):
        core.build_rows([ok_row("A-U")], KEY)


def test_build_rows_rejects_a_non_ok_status():
    bad = ok_row("B-B") | {"status": "skipped_context_overflow"}
    with pytest.raises(ValueError, match="not ok"):
        core.build_rows([ok_row("A-U"), bad], KEY)


def test_build_rows_rejects_a_capture_without_logprobs():
    bare = {"id": "B-B", "status": "ok", "prediction": "Mosaic"}
    with pytest.raises(ValueError, match="emit-logprobs"):
        core.build_rows([ok_row("A-U"), bare], KEY)


def test_build_rows_rejects_extra_rows_outside_the_key():
    with pytest.raises(ValueError, match="outside the answer key"):
        core.build_rows([ok_row("A-U"), ok_row("B-B"), ok_row("C-C")], KEY)


def test_build_rows_rejects_an_empty_generation():
    empty = ok_row("B-B")
    empty["confidence"]["answer"] = None
    with pytest.raises(ValueError, match="empty generation"):
        core.build_rows([ok_row("A-U"), empty], KEY)


# --------------------------------------------------------------------------
# Vectorised sweep must agree with the reference implementation
# --------------------------------------------------------------------------


def test_vectorised_scoring_matches_gate_core_on_every_threshold():
    rng = np.random.default_rng(17)
    rows = [
        make_row(
            f"R-{index}{'U' if index % 4 == 3 else 'B'}",
            "UNKNOWN" if index % 7 == 0 else ("Talia Frost" if index % 3 else "Mosaic"),
            index % 4 != 3,
            unknown_exact_logprob=float(rng.normal(-8.0, 4.0)),
            first_token_entropy=float(rng.gamma(2.0, 0.3)),
        )
        for index in range(40)
    ]
    arrays = sweep.row_arrays(rows)
    for signal in ("unknown_exact_logprob", "first_token_entropy"):
        built = sweep.build_sweep(signal, rows)
        scored = sweep.score_masks(built.masks, arrays)
        for index, threshold in enumerate(built.thresholds):
            spec = core.GateSpec(signal=signal, threshold=float(threshold))
            reference = core.evaluate_gate(spec, rows)
            assert int(scored["gated_correct"][index]) == reference.gated_correct
            assert int(scored["abstentions"][index]) == reference.abstentions
            assert int(scored["added_abstentions"][index]) == reference.added_abstentions
            assert int(scored["gained"][index]) == reference.mcnemar_b
            assert int(scored["lost"][index]) == reference.mcnemar_c
            assert scored["abstention_precision"][index] == pytest.approx(
                reference.abstention_precision
            )
            assert scored["abstention_recall"][index] == pytest.approx(
                reference.abstention_recall
            )


def test_select_primary_applies_the_pre_registered_tie_breaks():
    table = [
        {
            "combine": "single",
            "signal": "unknown_exact_logprob",
            "threshold": -5.0,
            "gated_correct": 47,
            "added_abstentions": 9,
            "abstention_precision": 0.9,
        },
        {
            "combine": "single",
            "signal": "answer_mean_entropy",
            "threshold": 0.4,
            "gated_correct": 47,
            "added_abstentions": 4,  # fewer added abstentions wins
            "abstention_precision": 0.5,
        },
        {
            "combine": "single",
            "signal": "first_token_entropy",
            "threshold": 0.9,
            "gated_correct": 48,  # ... but only among equal accuracy
            "added_abstentions": 40,
            "abstention_precision": 0.1,
        },
        {
            "combine": "and",
            "signal": "first_token_entropy",
            "threshold": 0.9,
            "gated_correct": 99,  # pairs must never be selected
            "added_abstentions": 1,
            "abstention_precision": 1.0,
        },
    ]
    assert sweep.select_primary(table)["signal"] == "first_token_entropy"
    assert sweep.select_primary(table[:2])["signal"] == "answer_mean_entropy"


def test_select_primary_needs_at_least_one_single_signal_gate():
    with pytest.raises(ValueError):
        sweep.select_primary([{"combine": "and", "gated_correct": 1}])


# --------------------------------------------------------------------------
# The lab must score with the published scorer, not a private rule
# --------------------------------------------------------------------------


def test_scorer_is_the_published_one():
    assert core.SCORER_PATH.name == "evaluate_predictions.py"
    spec = importlib.util.spec_from_file_location("published", core.SCORER_PATH)
    assert spec is not None
    assert core.SCORER.normalize("  Talia Frost. ") == "talia frost"

