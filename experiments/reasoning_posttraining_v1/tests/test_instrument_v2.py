"""Fixture gates for the Forge v2 lane: strata, shapes, shortcuts and the verifier channels.

Hermetic. The pure functions need nothing; the scored paths run against a stub verifier.

A correction to what this file used to claim about itself (CL-043). The docstring previously
called `test_v2_shortcuts_are_answer_agnostic_except_the_quarantined_one` "the load-bearing
test ... what caught a baseline of mine that was secretly reading the reference". It could not
have been: it deleted only `reference`, `reference_text` and `certificate`, while the leak it is
credited with finding ran through `answer_shape`. A new leak through `answer_shape`, `stratum` or
`world` would have shipped silently. It is now a PERTURBATION test -- every reference-derived
field is replaced with a different valid value and the honest predictors must not move -- which
does detect that whole class.
"""
import json

import pytest

from experiments.reasoning_posttraining_v1 import instrument as ins

WORLD = {"family": "ledger_replay", "query": {"account": "C", "cutoff": 16}, "records": []}


def v2_bank_fixture():
    return [
        {"id": "a", "group_id": "a", "family": "ledger_replay", "stratum": "ANSWER",
         "answer_shape": "value", "reference": {"value": 7}, "reference_text": '{"value":7}',
         "world": dict(WORLD), "certificate": {"value": 7},
         "prompt": [{"role": "system", "content": "s"},
                    {"role": "user",
                     "content": 'RECORDS\nQUERY {"account":"C","cutoff":16}'}]},
        {"id": "b", "group_id": "b", "family": "temporal_route", "stratum": "IMPOSSIBLE",
         "answer_shape": "status", "reference": {"status": "IMPOSSIBLE"},
         "reference_text": '{"status":"IMPOSSIBLE"}', "world": dict(WORLD), "certificate": None,
         "prompt": [{"role": "system", "content": "s"},
                    {"role": "user", "content": 'RECORDS\nQUERY {"from":"N1","to":"N9"}'}]},
    ]


def golded(bank):
    for row in bank:
        row["world"] = {**row["world"], "_gold": row["reference"]}
    return bank


class StubRewards:
    """Minimal stand-in for forge_rl.rewards with the same channel contract."""

    @staticmethod
    def score_response(world, response, certificate=None):
        out = {"reward": 0.0, "correct": False, "format_valid": False, "schema_valid": False,
               "feasible": False, "reason": "parse_failure"}
        try:
            obj = json.loads(response)
        except Exception:
            return out
        if not isinstance(obj, dict):
            return out
        out.update(format_valid=True, schema_valid=True, reason="incorrect_value")
        if obj == world["_gold"]:
            out.update(correct=True, feasible=True, reward=1.0, reason="exact_match")
        return out


MODS = {"rewards": StubRewards}


# ---------------------------------------------------------------- strata and shapes

def test_v2_stratum_reads_the_structured_status():
    assert ins.v2_stratum({"value": 7}) == "ANSWER"
    assert ins.v2_stratum({"status": "UNKNOWN"}) == "UNKNOWN"
    assert ins.v2_stratum({"status": "IMPOSSIBLE"}) == "IMPOSSIBLE"
    assert set(ins.V2_STATUSES) == {"ANSWER", "UNKNOWN", "IMPOSSIBLE"}
    assert len(ins.V2_FAMILIES) == 8


def test_every_v2_answer_shape_has_an_empty_witness():
    for shape in ("path", "actions", "order", "items", "value", "answer", "status"):
        assert shape in ins.V2_EMPTY_WITNESS, shape
    assert ins._v2_shape({"cost": 1, "path": ["N1"]}) == "path"
    assert ins._v2_shape({"actions": [], "cost": 0}) == "actions"
    assert ins._v2_shape({"status": "UNKNOWN"}) == "status"
    with pytest.raises(ValueError, match="unrecognized v2 answer shape"):
        ins._v2_shape({"nonsense": 1})


# ---------------------------------------------------------------- shortcut honesty

def test_v2_shortcuts_are_answer_agnostic_except_the_quarantined_one():
    """Perturb EVERY reference-derived field, not just three of them.

    The old version deleted `reference`, `reference_text` and `certificate` only, which left
    `answer_shape`, `stratum` and `world` readable -- so it was blind to exactly the leak class
    it was credited with catching (CL-043). Replacing each field with a different valid value
    keeps `shortcut_predictions_v2` callable while making any dependence on them visible.
    """
    bank = v2_bank_fixture()
    preds = ins.shortcut_predictions_v2(bank, train_majority='{"status":"IMPOSSIBLE"}')
    perturbed = []
    for row in bank:
        other_shape = "status" if row["answer_shape"] != "status" else "value"
        other_stratum = "UNKNOWN" if row["stratum"] != "UNKNOWN" else "ANSWER"
        perturbed.append({**row,
                          "reference": {"value": -99999},
                          "reference_text": '{"value":-99999}',
                          "certificate": {"value": -99999},
                          "answer_shape": other_shape,
                          "stratum": other_stratum,
                          "world": {**row["world"], "_gold": {"value": -99999}}})
    again = ins.shortcut_predictions_v2(perturbed, train_majority='{"status":"IMPOSSIBLE"}')
    for name in preds:
        if name in ins.V2_ORACLE_ASSISTED:
            continue
        assert preds[name] == again[name], f"{name} depends on a reference-derived field"
    assert "oracle_shape_empty_witness" in ins.V2_ORACLE_ASSISTED
    assert "oracle_shape_empty_witness" in preds


def test_the_strengthened_perturbation_test_would_catch_an_answer_shape_leak():
    """Proves the strengthening is real: a predictor keyed on answer_shape now moves."""
    bank = v2_bank_fixture()
    shapes = [row["answer_shape"] for row in bank]
    flipped = ["status" if s != "status" else "value" for s in shapes]
    assert shapes != flipped
    # This is what `oracle_shape_empty_witness` does, and under perturbation it changes --
    # which is precisely what the old delete-three-fields version could not see.
    before = ins.shortcut_predictions_v2(bank, train_majority="{}")
    after = ins.shortcut_predictions_v2(
        [{**r, "answer_shape": f} for r, f in zip(bank, flipped)], train_majority="{}")
    assert before["oracle_shape_empty_witness"] != after["oracle_shape_empty_witness"]


# ---------------------------------------------------------------- the prompt-copy shortcut

def test_copy_prompt_sentinel_reads_the_literal_the_prompt_prints():
    """CL-040: the v2 rule text prints the complete gold object for unanswerable worlds."""
    bank = [dict(v2_bank_fixture()[1])]
    bank[0]["prompt"] = [
        {"role": "system", "content": "s"},
        {"role": "user",
         "content": 'TASK If no route exists, return {"status":"IMPOSSIBLE"}.\nQUERY {"a":1}'}]
    preds = ins.shortcut_predictions_v2(bank, train_majority="{}")
    assert preds["copy_prompt_sentinel"] == ['{"status":"IMPOSSIBLE"}']
    assert ins._prompt_sentinel(bank[0]["prompt"]) == "IMPOSSIBLE"


def test_copy_prompt_sentinel_falls_back_when_no_literal_is_printed():
    bank = [dict(v2_bank_fixture()[0])]
    assert ins._prompt_sentinel(bank[0]["prompt"]) is None
    preds = ins.shortcut_predictions_v2(bank, train_majority="{}")
    assert preds["copy_prompt_sentinel"] == [ins.V2_SENTINEL_FALLBACK]


def test_copy_prompt_sentinel_counts_toward_the_floor():
    """It reads only the prompt, so quarantining it would understate the floor."""
    assert "copy_prompt_sentinel" not in ins.V2_ORACLE_ASSISTED
    preds = ins.shortcut_predictions_v2(v2_bank_fixture(), train_majority="{}")
    assert "copy_prompt_sentinel" in preds


def test_the_leak_check_compares_raw_text_not_a_json_dump():
    """The defect itself, pinned. json.dumps escapes the quotes, so the needle never matched."""
    prompt = [{"role": "system", "content": "s"},
              {"role": "user", "content": 'return {"status":"UNKNOWN"} if unclear'}]
    needle = '{"status":"UNKNOWN"}'
    raw = "\n".join(x["content"] for x in prompt)
    assert needle in raw, "the raw-text haystack must find a printed gold"
    assert needle not in json.dumps(prompt), (
        "this is the old defect: the dumped haystack cannot contain the unescaped needle")


def test_the_quarantined_baseline_really_does_depend_on_the_reference():
    """Proves the quarantine is necessary, not decorative."""
    bank = v2_bank_fixture()
    preds = ins.shortcut_predictions_v2(bank, train_majority="{}")
    flipped = [dict(row, answer_shape="value") for row in bank]
    other = ins.shortcut_predictions_v2(flipped, train_majority="{}")
    assert preds["oracle_shape_empty_witness"] != other["oracle_shape_empty_witness"]


def test_prose_baseline_floors_the_format_channel():
    preds = ins.shortcut_predictions_v2(v2_bank_fixture(), train_majority="{}")
    assert preds["prose_non_json"] == ["The answer is zero."] * 2
    got = ins.summarize_v2(golded(v2_bank_fixture()), preds["prose_non_json"], MODS)
    assert got["format_valid"] == 0 and got["schema_valid"] == 0 and got["correct"] == 0
    assert got["verifier_reasons"] == {"parse_failure": 2}


def test_constant_baselines_emit_valid_json_so_they_test_semantics_not_format():
    preds = ins.shortcut_predictions_v2(v2_bank_fixture(),
                                        train_majority='{"status":"IMPOSSIBLE"}')
    for name in ("constant_unknown", "constant_impossible", "constant_train_majority",
                 "zero_value", "copy_query_integer"):
        for text in preds[name]:
            assert isinstance(json.loads(text), dict), name


# ---------------------------------------------------------------- scoring channels

def test_v2_summary_keeps_format_schema_and_semantics_apart():
    bank = golded(v2_bank_fixture())
    got = ins.summarize_v2(bank, [r["reference_text"] for r in bank], MODS)
    assert got["rows"] == 2 and got["correct"] == 2
    assert got["format_valid"] == 2 and got["schema_valid"] == 2
    assert got["per_stratum"]["ANSWER"] == {"n": 1, "correct": 1}
    assert got["per_stratum"]["IMPOSSIBLE"] == {"n": 1, "correct": 1}
    assert got["per_answer_shape"]["value"] == {"n": 1, "correct": 1}
    assert got["quality_claim"] is False
    assert "alternative optima accepted" in got["scored_by"]


def test_unparseable_output_is_not_booked_as_an_unsupported_answer():
    """CL-042: `predicted_status is None` covers both, and merging them overstates confidence."""
    bank = golded(v2_bank_fixture())
    # Row b is IMPOSSIBLE. Answer it with valid JSON carrying no status -> unsupported answer.
    got = ins.summarize_v2(bank, ['{"value":7}', '{"value":3}'], MODS)
    assert got["unsupported_answer_on_unanswerable"] == 1
    assert got["unparseable_on_unanswerable"] == 0
    # Now fail to parse on the same row -> that is a format failure, not a confident answer.
    got = ins.summarize_v2(bank, ['{"value":7}', "not json at all"], MODS)
    assert got["unsupported_answer_on_unanswerable"] == 0
    assert got["unparseable_on_unanswerable"] == 1
    assert got["format_valid"] == 1


def test_v2_summary_separates_three_abstention_error_directions():
    bank = golded(v2_bank_fixture())
    # abstain on the answerable row, answer on the unanswerable row
    got = ins.summarize_v2(bank, ['{"status":"UNKNOWN"}', '{"value":0}'], MODS)
    assert got["false_abstention_on_answerable"] == 1
    assert got["unsupported_answer_on_unanswerable"] == 1
    assert got["wrong_sentinel_on_unanswerable"] == 0
    # the WRONG sentinel is its own category, never lumped into the others
    got = ins.summarize_v2(bank, ['{"value":7}', '{"status":"UNKNOWN"}'], MODS)
    assert got["wrong_sentinel_on_unanswerable"] == 1
    assert got["unsupported_answer_on_unanswerable"] == 0
    assert got["false_abstention_on_answerable"] == 0


def test_v2_summary_refuses_mismatched_prediction_count():
    with pytest.raises(ValueError):
        ins.summarize_v2(golded(v2_bank_fixture()), ['{"value":7}'], MODS)


def test_v2_positive_control_fails_when_a_gold_does_not_verify():
    bank = golded(v2_bank_fixture())
    assert ins.positive_control_v2(bank, MODS)["status"] == "PASS"
    broken = [dict(bank[0], reference_text='{"value":999}'), bank[1]]
    got = ins.positive_control_v2(broken, MODS)
    assert got["status"] == "FAIL" and got["failed"] == 1
    assert got["failures"][0]["id"] == "a"
    assert got["model_calls"] == 0


def test_score_v2_reports_the_predicted_status_for_abstention_accounting():
    bank = golded(v2_bank_fixture())
    got = ins.score_v2('{"status":"UNKNOWN"}', bank[0], MODS)
    assert got["predicted_status"] == "UNKNOWN" and got["correct"] is False
    got = ins.score_v2('{"value":7}', bank[0], MODS)
    assert got["predicted_status"] is None and got["correct"] is True
    got = ins.score_v2("not json", bank[0], MODS)
    assert got["predicted_status"] is None and got["format_valid"] is False
