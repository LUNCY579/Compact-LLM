from __future__ import annotations

from rineya_state_lab.evaluation.parser import parse_response, parse_stateful_response


def test_conclusion_sentence_is_explicit() -> None:
    result = parse_response("Reasoning.\nTherefore, the answer is 27.")
    assert result.status == "OK"
    assert result.strict_final == "27."


def test_stateful_json_and_fence() -> None:
    text = """```json
    {"next_reasoning_step":"Add 2.","updated_compact_state":"Value: 5","done":false,"final_answer":null}
    ```"""
    result = parse_stateful_response(text)
    assert result.structured == {
        "next_reasoning_step": "Add 2.",
        "updated_compact_state": "Value: 5",
        "done": False,
        "final_answer": None,
    }
    assert result.json_object is not None


def test_conflicting_explicit_answers_fail_strict() -> None:
    result = parse_response("Final answer: 2\nAnswer: 3")
    assert result.conflict
    assert result.strict_final is None
    assert result.status == "CONFLICTING_FINALS"


def test_refusal_detected() -> None:
    result = parse_response("I cannot answer this question.")
    assert result.refusal
    assert result.status == "REFUSAL"
