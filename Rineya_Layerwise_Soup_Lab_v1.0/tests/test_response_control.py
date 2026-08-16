from rineya_soup_lab.vendor.official_v45.response_control import (
    finalize_response,
    first_sentence_boundary,
    sentence_boundaries,
    validate_response_control_mode,
)


def test_decimal_version_and_abbreviation_are_not_boundaries() -> None:
    assert not first_sentence_boundary("The value is 3.14").found
    assert not first_sentence_boundary("Use v2.1").found
    assert not first_sentence_boundary("Ask Dr. Smith").found
    assert first_sentence_boundary("The value is 3.14.").found


def test_strict_finalization_and_eos_only() -> None:
    strict = finalize_response("First sentence. Second sentence.", "one_sentence")
    assert strict.text == "First sentence."
    assert strict.trimmed is True
    ordinary = finalize_response("First sentence. Second sentence.", "eos_only")
    assert ordinary.text == "First sentence. Second sentence."
    assert ordinary.trimmed is False


def test_cjk_and_ellipsis_boundaries() -> None:
    assert first_sentence_boundary("好呀！").found
    assert finalize_response("可以。再见。", "one_sentence").text == "可以。"
    assert len(sentence_boundaries("Well...")) == 1


def test_invalid_mode() -> None:
    try:
        validate_response_control_mode("punctuation_only")
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid response-control mode was accepted")
