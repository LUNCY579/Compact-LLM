from __future__ import annotations

import pytest

from rineya_state_lab.evaluation.scorers import score_answer, state_equivalence


@pytest.mark.parametrize(
    ("expected", "predicted", "answer_type"),
    [
        ("94", "94", "integer"),
        ("$124.68", "$124.68", "decimal"),
        ("x = 40", "x=40", "equation"),
        ("1/18", "2/36", "fraction"),
        ("x ≥ 36", "x >= 36", "inequality"),
        ("False", "false", "boolean"),
        ("09:44", "9:44", "time"),
        ("129 N", "129 newtons", "measurement"),
        ("$2853.52", "$2,853.52", "money"),
        ("{6, 7, 9}", "{9,6,7}", "set"),
        ("A=OFF, B=ON", "B=ON, A=OFF", "state"),
        ("mean 17, range 46", "range=46; mean=17", "statistic"),
        ("35ILIM!", "35ILIM!", "string"),
        ("brushes, doors, skies", "brushes, doors, skies", "text_transformation"),
        ("Faye", "Faye", "ordering"),
        ("67", "67", "program_output"),
    ],
)
def test_supported_answer_types(expected: str, predicted: str, answer_type: str) -> None:
    result = score_answer(expected, predicted, predicted, answer_type)
    assert result.correct_strict, result
    assert result.correct_lenient, result


def test_units_are_strict_but_number_is_lenient() -> None:
    result = score_answer("55 km", "55", "55", "rate")
    assert not result.correct_strict
    assert result.correct_lenient


def test_state_equivalence_keeps_numeric_conflicts_out() -> None:
    result = state_equivalence("Current value: 94.", "The current value is 95.", f1_threshold=0.5)
    assert not result["lenient_equivalent"]
