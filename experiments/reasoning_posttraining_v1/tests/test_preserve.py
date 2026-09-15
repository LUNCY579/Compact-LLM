"""Fixture gates for the preservation lane and the quarantine guard.

Hermetic: every test builds its own rows, and nothing reads the real corpora or the real
registry. The load-bearing ones are
`test_shortcut_predictions_are_answer_agnostic` (the leak check that caught a dishonest baseline
in the v2 lane), `test_quarantine_refusal_is_not_decorative`, and
`test_tolerance_does_not_hand_credit_to_a_different_answer`.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from experiments.reasoning_posttraining_v1 import preserve as pres
from experiments.reasoning_posttraining_v1 import quarantine as quar

NO_EV = pres.ABSTENTION_SENTENCES["NO_EVIDENCE"]
CONFLICT = pres.ABSTENTION_SENTENCES["SOURCES_CONFLICT"]


def row(source_id, decision, gold, *, sub_bank="direct_reasoning", pair_id=None, aliases=(),
        cause=None, question="Which option is cheaper?"):
    return pres._row(
        sub_bank=sub_bank, source_id=source_id, decision=decision, cause=cause, gold=gold,
        prompt=[{"role": "system", "content": "s"}, {"role": "user", "content": question}],
        pair_id=pair_id, aliases=aliases,
        strata={"difficulty": "moderate", "domain": "work"})


def bank_fixture():
    return [
        row("a1", "answer", "Option B.", question="A costs 9 and B costs 4. Which is cheaper?"),
        row("a2", "abstain", NO_EV, cause="insufficient_information",
            question="A costs 9 and B's price is missing. Which is cheaper?"),
        row("b1", "conflict", CONFLICT, sub_bank="procedural_rag",
            cause="direct_conflict_equal_authority",
            question="Audit 7 says 12 and audit 8 says 30. Which applies?"),
    ]


def paired_fixture():
    return [
        row("p_ab", "abstain", NO_EV, sub_bank="v3_answer_or_abstain", pair_id="v3_pair:000",
            cause="insufficient_information",
            question="A car travels for 4 hours, but its speed is not given. Distance?"),
        row("p_an", "answer", "120 miles.", sub_bank="v3_answer_or_abstain",
            pair_id="v3_pair:000", question="A car travels 30 mph for 4 hours. Distance?"),
    ]


# ----------------------------------------------------------------- the two-sentence contract

def test_both_registered_abstention_sentences_are_recognized_and_distinguished():
    assert pres.abstention_kind(NO_EV) == "NO_EVIDENCE"
    assert pres.abstention_kind(CONFLICT) == "SOURCES_CONFLICT"
    assert pres.abstention_kind("Option B.") is None
    # The two must never collapse into each other.
    assert pres.abstention_kind(NO_EV) != pres.abstention_kind(CONFLICT)


def test_a_row_whose_gold_is_not_a_registered_sentence_cannot_be_labelled_abstaining():
    """PRES-C4: the gold decides, never the family name."""
    with pytest.raises(ValueError, match="no registered abstention sentence"):
        row("x", "conflict", "Tier 2.")
    with pytest.raises(ValueError, match="labelled answer but its gold IS"):
        row("y", "answer", NO_EV)


def test_required_response_kind_is_derived_from_the_gold_not_the_cause():
    """PRES-C5: a direct_reasoning conflict row still requires the NO_EVIDENCE sentence."""
    got = row("dr", "abstain", NO_EV, cause="conflicting_evidence")
    assert got["abstention_cause"] == "conflicting_evidence"
    assert got["required_response_kind"] == "NO_EVIDENCE"


def test_wrong_abstention_kind_is_its_own_channel_and_never_counted_correct():
    bank = bank_fixture()
    # Abstain with the missing-information sentence on the row that requires the conflict one.
    got = pres.summarize(bank, ["Option B.", NO_EV, NO_EV])
    assert got["abstention_channel"]["wrong_abstention_kind"] == 1
    assert got["abstention_channel"]["abstained_at_all"] == 2
    assert got["abstention_channel"]["correct"] == 1
    assert got["reasons"]["wrong_abstention_kind"] == 1
    assert got["correct"] == 2  # a1 and a2, not b1


def test_three_error_directions_stay_apart():
    bank = bank_fixture()
    got = pres.summarize(bank, [NO_EV, "Option B.", CONFLICT])
    assert got["answer_channel"]["false_abstention"] == 1
    assert got["abstention_channel"]["missing_abstention"] == 1
    assert got["abstention_channel"]["correct"] == 1
    assert got["correct"] == 1


# ----------------------------------------------------------------- tolerance must not leak

def test_tolerance_does_not_hand_credit_to_a_different_answer():
    bank = [row("a", "answer", "120 miles.")]
    for wrong in ("120", "miles", "120 kilometres.", "about 120 miles", "1200 miles."):
        assert pres.summarize(bank, [wrong])["correct"] == 0, wrong
    for right in ("120 miles.", "120 miles", "  120   Miles.  ", '"120 miles."'):
        assert pres.summarize(bank, [right])["correct"] == 1, right


def test_strict_mode_forgives_nothing_the_tolerant_mode_forgives():
    bank = [row("a", "answer", "120 miles.")]
    assert pres.summarize(bank, ["120 miles"], tolerant=True)["correct"] == 1
    assert pres.summarize(bank, ["120 miles"], tolerant=False)["correct"] == 0


def test_shipped_aliases_are_honoured_but_only_the_shipped_ones():
    bank = [row("a", "answer", "Atlas.", aliases=["Atlas"])]
    assert pres.summarize(bank, ["Atlas"])["correct"] == 1
    assert pres.summarize(bank, ["Falcon"])["correct"] == 0


# ----------------------------------------------------------------- shortcut honesty

def test_shortcut_predictions_are_answer_agnostic():
    """Delete every reference field; the baselines must be byte-identical. This is the check
    that caught an oracle-assisted baseline in the v2 lane."""
    bank = bank_fixture() + paired_fixture()
    before = pres.shortcut_predictions(bank)
    stripped = [{k: v for k, v in r.items()
                 if k not in ("reference_text", "reference_aliases", "required_response_kind",
                              "answerable", "decision", "abstention_cause")}
                for r in bank]
    after = pres.shortcut_predictions(stripped)
    assert before == after
    assert not any("reference" in name for name in before)


def test_constant_abstain_floor_is_the_no_evidence_base_rate_not_all_abstentions():
    """The second sentence is what keeps the floor below 0.5."""
    bank = bank_fixture()
    got = pres.floor(bank)
    assert got["shortcut_floor_predictor"] == "constant_abstain"
    # 1 of 3 rows requires NO_EVIDENCE; the conflict row is not credited.
    assert got["baselines"]["constant_abstain"]["correct"] == 1
    assert got["baselines"]["constant_conflict_sentence"]["correct"] == 1
    assert got["baselines"]["constant_abstain"]["accuracy"] == round(1 / 3, 4)


def test_positive_control_fails_when_a_gold_does_not_verify():
    bank = bank_fixture()
    assert pres.positive_control(bank)["status"] == "PASS"
    broken = [dict(bank[0], reference_text="Option B"), *bank[1:]]
    broken[0]["reference_aliases"] = []
    got = pres.positive_control(broken)
    assert got["status"] == "PASS"  # strict key still matches itself
    # A gold that is not its own answer is what must fail:
    worse = [dict(bank[0], reference_text="Option B.", reference_aliases=[]), *bank[1:]]
    worse[0] = dict(worse[0], required_response_kind="NO_EVIDENCE", answerable=False)
    assert pres.positive_control(worse)["status"] == "FAIL"
    assert pres.positive_control(worse)["model_calls"] == 0


def test_summarize_refuses_a_mismatched_prediction_count():
    with pytest.raises(ValueError, match="one prediction per row"):
        pres.summarize(bank_fixture(), ["Option B."])


# ----------------------------------------------------------------- minimal pairs

def test_minimal_pair_calibration_names_all_four_outcomes():
    bank = paired_fixture()
    sensitive = pres.summarize(bank, [NO_EV, "120 miles."])["minimal_pair_calibration"]
    assert sensitive == {"complete_pairs": 1, "evidence_sensitive": 1}
    always = pres.summarize(bank, [NO_EV, NO_EV])["minimal_pair_calibration"]
    assert always["abstains_regardless"] == 1
    never = pres.summarize(bank, ["120 miles.", "120 miles."])["minimal_pair_calibration"]
    assert never["answers_regardless"] == 1
    flipped = pres.summarize(bank, ["120 miles.", NO_EV])["minimal_pair_calibration"]
    assert flipped["inverted"] == 1


def test_pair_detection_requires_exactly_one_abstain_and_one_answer():
    def raw(rid, gold, group):
        return {"id": rid, "metadata": {"counterfactual_group_id": group},
                "messages": [{"role": "user", "content": "q"},
                             {"role": "assistant", "content": gold}]}
    pairs, lone_a, lone_b = pres.v3_pairs([
        raw("1", NO_EV, "g1"), raw("2", "12.", "g1"),          # a real pair
        raw("3", NO_EV, "g2"), raw("4", NO_EV, "g2"),          # both abstain: not a pair
        raw("5", "7.", "g3")])                                  # singleton
    assert len(pairs) == 1 and pairs[0][0]["id"] == "1"
    assert {r["id"] for r in lone_a} == {"3", "4"}
    assert {r["id"] for r in lone_b} == {"5"}


# ----------------------------------------------------------------- allocation determinism

def test_allocation_is_deterministic_and_hits_the_registered_total():
    groups = [f"g{i}" for i in range(7)]
    available = {g: 20 for g in groups}
    first = pres._largest_remainder(96, groups, available)
    assert sum(first.values()) == 96 and first == pres._largest_remainder(96, groups, available)
    assert max(first.values()) - min(first.values()) <= 1


def test_allocation_refuses_to_invent_rows_it_does_not_have():
    with pytest.raises(ValueError, match="cannot draw"):
        pres._largest_remainder(100, ["a", "b"], {"a": 10, "b": 10})


def test_registered_sizes_sum_to_the_registered_total():
    assert sum(sum(v.values()) for v in pres.SIZES.values()) == pres.TOTAL == 768


# ----------------------------------------------------------------- admission audit

def test_leakage_against_another_bank_fails_the_gate():
    bank = bank_fixture()
    leaked = [{"id": "elsewhere",
               "messages": [{"role": "user", "content": bank[0]["prompt"][-1]["content"]},
                            {"role": "assistant", "content": "x"}]}]
    got = pres.admission_audit(bank, {"some_train_bank": leaked})
    assert got["no_leakage"] is False and got["clean"] is False
    assert got["against"][0]["text_collisions"]["last_user_turn"] == 1


def test_internal_digit_blinded_redundancy_is_reported_but_does_not_fail_the_gate():
    """PRES-C6. Two rows differing only by a number are a designed minimal pair, not a defect."""
    bank = paired_fixture()
    got = pres.admission_audit(bank, {})
    assert got["clean"] is True and got["no_leakage"] is True
    assert got["internal"]["distinct_questions_exact"] == 2
    assert got["internal"]["duplicate_prompt_counts"]["last_user_turn"] == 0
    assert got["internal"]["per_sub_bank_effective_clusters"][
        "v3_answer_or_abstain"]["rows"] == 2


def test_two_rows_posing_the_identical_question_do_fail_the_gate():
    bank = [row("a", "answer", "Option B.", question="Same?"),
            row("b", "answer", "Option C.", question="Same?")]
    assert pres.admission_audit(bank, {})["clean"] is False


# ----------------------------------------------------------------- the quarantine guard

def registry_fixture(**over):
    entry = {"key": "some/train/bank", "reason": "SELECTION_EXPOSED",
             "bars": ["TRAIN", "CONFIRMATION"], "asserted_rows": 2, "rows": 2,
             "evidence_path": "e", "lifted": False, "present": True,
             "path": "some/train/bank.jsonl", "ids": ["x1", "x2"]}
    entry.update(over)
    return {"record": "RRP2_QUARANTINE", "no_quarantine_is_lifted": not entry["lifted"],
            "entries": [entry], "sha256": "deadbeef"}


def test_quarantine_refusal_is_not_decorative():
    registry = registry_fixture()
    quar.refuse(["clean1", "clean2"], registry, use="TRAIN")          # no hit, no raise
    with pytest.raises(ValueError, match="barred from TRAIN"):
        quar.refuse(["clean1", "x2"], registry, use="TRAIN")


def test_a_bar_is_scoped_so_a_dev_bank_can_be_built_from_a_train_barred_split():
    """The whole reason `bars` is a set and not a boolean."""
    registry = registry_fixture()
    quar.refuse(["x1"], registry, use="DEV")                          # permitted
    with pytest.raises(ValueError):
        quar.refuse(["x1"], registry, use="CONFIRMATION")


def test_any_read_bars_every_use_including_dev():
    registry = registry_fixture(bars=["ANY_READ"])
    for use in ("TRAIN", "DEV", "SELECTION", "CONFIRMATION"):
        with pytest.raises(ValueError):
            quar.refuse(["x1"], registry, use=use)


def test_assert_readable_blocks_a_sealed_path_and_allows_others():
    registry = registry_fixture(bars=["ANY_READ"], path="a/fresh_sealed.jsonl",
                                reason="POLICY_LOCKED_OTHER_PROGRAMME")
    with pytest.raises(ValueError, match="ANY_READ-barred"):
        quar.assert_readable(registry, "a/fresh_sealed.jsonl")
    quar.assert_readable(registry, "a/dev.jsonl")


def test_load_refuses_a_registry_with_a_lifted_entry():
    body = registry_fixture(lifted=True)
    body["no_quarantine_is_lifted"] = False
    path = Path(__file__).with_name("_tmp_quarantine.json")
    path.write_text(json.dumps(body), encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="lifted"):
            quar.load(path)
    finally:
        path.unlink()


def test_every_spec_entry_uses_a_registered_reason_and_use():
    for spec in quar.SPEC:
        assert spec["reason"] in quar.REASONS, spec["key"]
        assert set(spec["bars"]) <= set(quar.USES), spec["key"]
        assert spec["bars"], spec["key"]


def test_the_two_fresh_sealed_splits_are_any_read_barred():
    """They belong to another programme's pre-registration. Spending them here is not ours."""
    sealed = [s for s in quar.SPEC if "fresh_sealed" in s["key"]]
    assert len(sealed) == 2
    for spec in sealed:
        assert spec["bars"] == ("ANY_READ",)
        assert spec["reason"] == "POLICY_LOCKED_OTHER_PROGRAMME"


# ----------------------------------------------------------------- module hygiene

@pytest.mark.parametrize("module", ["preserve", "quarantine"])
def test_module_does_not_import_torch_at_module_level(module):
    """The foreground controller imports these; torch there costs the child its RAM headroom."""
    code = (f"import experiments.reasoning_posttraining_v1.{module}, sys; "
            "print('torch' in sys.modules)")
    root = Path(pres.__file__).parents[2]
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(root))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


def test_every_correction_states_what_it_changes():
    ids = [c["id"] for c in pres.CORRECTIONS]
    assert ids == ["PRES-C1", "PRES-C2", "PRES-C3", "PRES-C4", "PRES-C5", "PRES-C6"]
    for correction in pres.CORRECTIONS:
        for field in ("registered", "measured", "resolution", "changes_a_number"):
            assert correction[field].strip(), (correction["id"], field)


def test_the_bank_never_claims_to_be_confirmation():
    got = pres.summarize(bank_fixture(), ["Option B.", NO_EV, CONFLICT])
    assert got["confirmation_eligible"] is False
    assert got["quality_claim"] is False
    assert got["correct"] == 3
