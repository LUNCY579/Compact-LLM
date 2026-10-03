import pytest

from experiments.reasoning_posttraining_v1.text_contact import audit_and_filter


def row(name, user, system="Common system instruction", answer="ok"):
    return {"id": name, "messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": answer}],
        "metadata": {"reply_class": "insufficient_information"}}


def test_short_arithmetic_stem_detected_under_wrong_class_and_new_operands():
    panel = [row("panel", "Please compute: What is 10% of 100?")]
    candidate = row("new", "At the market: What is 25% of 72?")
    kept, report = audit_and_filter([candidate], panel)
    assert not kept
    assert report["dropped"] == [{"id": "new",
        "reasons": ["digit_blinded_panel_stem"],
        "matched_panel_stem_ids": ["panel"], "reply_class": "insufficient_information"}]


def test_shared_system_and_answer_do_not_exclude_unrelated_user_task():
    shared = "This shared system instruction has many more than eight words in it."
    candidate = row("clean", "Name a color.", shared, "What is 10% of 100?")
    kept, report = audit_and_filter([candidate], [row("panel", "What is 10% of 100?", shared)])
    assert kept == [candidate]
    assert kept[0] is candidate
    assert report["dropped_rows"] == 0


def test_id_and_long_user_text_contacts_report_all_reasons():
    panel = [row("same", "prefix: One two three four five six seven eight nine.")]
    candidate = row("same", "ONE, two three four five six seven eight unrelated ending.")
    kept, report = audit_and_filter([candidate], panel)
    assert not kept
    assert report["reason_counts"] == {"panel_id": 1, "user_8gram": 1}


def test_all_user_turns_checked_and_clean_order_preserved():
    panel = [row("panel", "What is 10% of 100?")]
    hit = row("hit", "What is 25% of 72?")
    hit["messages"] += [{"role": "user", "content": "Now say hello."}]
    a, b = row("a", "Name a color."), row("b", "Name a fruit.")
    kept, report = audit_and_filter([a, hit, b], panel)
    assert kept == [a, b]
    assert report["dropped"][0]["id"] == "hit"


@pytest.mark.parametrize("panel", [[], [row("p", "Heading: ")],
                                     [row("p", "hello"), row("p", "goodbye")]])
def test_empty_or_ambiguous_panel_cannot_claim_clean(panel):
    with pytest.raises(ValueError):
        audit_and_filter([row("x", "hello")], panel)


def test_normalization_keeps_numeric_edit_from_bypassing_guard():
    kept, report = audit_and_filter([row("x", "WHAT  is ２５% of ７２?")],
                                   [row("p", "What is 10% of 100?")])
    assert not kept
    assert report["reason_counts"] == {"digit_blinded_panel_stem": 1}
