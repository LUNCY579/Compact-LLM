"""Synthetic scorer/paired-analysis fixtures only; no native capability evidence."""
import copy
import json

import pytest

from experiments.rineya_quality_sft_v1 import EXPERIMENT, SOURCE_CORE, SOURCE_WEIGHT_SHA256
from experiments.rineya_quality_sft_v1.analysis import compare_quality, summarize_source
from experiments.rineya_quality_sft_v1.contract import DEV_COUNTS
from experiments.rineya_quality_sft_v1.evaluation import prediction_hash, score_prediction, shortcut_controls, evaluation_messages, score_shortcut_controls, ADJUDICATION_RUBRIC, RUBRIC_SHA256
from experiments.rineya_quality_sft_v1.integrity import digest


def row(identifier="fixture", scorer="number", answer="1", dimension="solving", answerable=True, contract=None):
    spec = {"dimension": dimension, "scorer": scorer, "answer": answer, "answerable": answerable, "critical": False}
    if contract is not None:
        spec["contract"] = contract
    return {"id": identifier, "parent_id": identifier, "split": "dev", "category": "acquisition", "family": "fixture-family",
        "evaluation": spec, "messages": [{"role": "system", "content": "Fixture system"},
            {"role": "user", "content": "Original fixture query; label: public-cue"},
            {"role": "assistant", "content": "OLD HISTORY"},
            {"role": "user", "content": "Original final query; label: visible-cue"},
            {"role": "assistant", "content": "DO NOT READ TRAINING TARGET"}]}


def adjudication(item, prediction, *, correct=True, abstention=False, rating=None):
    return {"row_id": item["id"], "prediction_sha256": prediction_hash(prediction), "rubric_sha256": RUBRIC_SHA256,
        "status": "RESOLVED", "reviews": [{"reviewer": identifier, "independent": True, "blinded": True,
            "model_role_visible": False, "semantic_correct": correct, "semantic_abstention": abstention,
            "communication_rating": rating, "error_evidence": {}} for identifier in ("reviewer-one", "reviewer-two")]}


def test_gold_is_explicit_and_target_is_never_scored_or_forwarded():
    item = row()
    assert score_prediction(item, "1")["correct"] is True
    item["messages"][-1]["content"] = "CORRUPTED TRAINING TARGET"
    assert score_prediction(item, "1")["correct"] is True
    assert "CORRUPTED TRAINING TARGET" not in str(evaluation_messages(item))
    assert "OLD HISTORY" in str(evaluation_messages(item))
    del item["evaluation"]["answer"]
    with pytest.raises(ValueError): score_prediction(item, "1")


def test_number_semantics_and_declared_format_are_separate():
    item = row(contract={"kind": "boxed_int"})
    result = score_prediction(item, "1")
    assert result["correct"] is True and result["strict_format"] is False
    assert result["error_categories"] == []  # No guessed mechanism.
    assert score_prediction(item, "\\boxed{1}")["strict_format"] is True
    assert score_prediction(item, "2")["correct"] is False
    result = score_prediction(item, "I cannot answer")
    assert result["correct"] is None and result["status"] == "INCOMPLETE"


@pytest.mark.parametrize("prediction", ['{"x":true}', '{"x":1,"x":1}', '{"x":NaN}'])
def test_json_rejects_coercion_duplicate_keys_and_nonfinite(prediction):
    item = row(scorer="json_exact", answer='{"x":1}')
    assert score_prediction(item, prediction)["correct"] is not True


def test_json_legal_exponents_preserve_exact_finite_values_instead_of_float_overflow():
    item = row(scorer="json_exact", answer='{"x":1e400}')
    assert score_prediction(item, '{"x":1e400}')["correct"] is True
    assert score_prediction(item, '{"x":1e500}')["correct"] is False
    assert score_prediction(item, '{"x":NaN}')["correct"] is not True
    nested = row(scorer="json_exact", answer='{"x":[1e400]}')
    assert score_prediction(nested, '{"x":[Infinity]}')["correct"] is not True


def test_json_fence_can_be_semantically_right_but_wrong_declared_format():
    item = row(scorer="json_exact", answer='{"x":1}', contract={"kind": "json_object", "validshape": {"keys": ["x"]}})
    result = score_prediction(item, '```json\n{"x":1}\n```')
    assert result["correct"] is True and result["strict_format"] is False
    fixed = row(scorer="fixed_list", answer='["a","b"]')
    assert score_prediction(fixed, '["a","b"]')["correct"] is True
    assert score_prediction(fixed, '["b","a"]')["correct"] is False
    fixed["evaluation"]["contract"] = {"kind": "lines", "length": 2}
    assert score_prediction(fixed, "a\nb")["correct"] is True
    assert score_prediction(row(scorer="json_exact", answer="null"), "null")["correct"] is True


def test_literal_unknown_does_not_manufacture_semantic_abstention():
    item = row(scorer="semantic_review", answer="UNKNOWN", dimension="grounding_answerability", answerable=False)
    result = score_prediction(item, "UNKNOWN")
    assert result["literal_unknown"] is True and result["semantic_abstention"] is None
    assert result["correct"] is None and result["status"] == "INCOMPLETE"
    natural = "I cannot determine that from the supplied evidence."
    result = score_prediction(item, natural, adjudication(item, natural, abstention=True))
    assert not result["literal_unknown"] and result["semantic_abstention"] is True and result["correct"] is True
    assert not score_prediction(item, "UNKNOWN.")["literal_unknown"]


def test_reviews_need_two_independent_blinded_hash_bound_reviewers():
    item = row(scorer="semantic_review", answer="friendly", dimension="communication")
    review = adjudication(item, "friendly", rating=3)
    assert score_prediction(item, "friendly", review)["communication_rating"] == 3
    changed = copy.deepcopy(review)
    changed["reviews"][1]["semantic_correct"] = False
    assert score_prediction(item, "friendly", changed)["correct"] is None
    assert score_prediction(item, "friendly", changed)["status"] == "INCOMPLETE"
    for mutation in ("hash", "same_reviewer", "unblinded", "role_visible", "rubric"):
        changed = copy.deepcopy(review)
        if mutation == "hash": changed["prediction_sha256"] = "0" * 64
        if mutation == "same_reviewer": changed["reviews"][1]["reviewer"] = changed["reviews"][0]["reviewer"]
        if mutation == "unblinded": changed["reviews"][1]["blinded"] = False
        if mutation == "role_visible": changed["reviews"][1]["model_role_visible"] = True
        if mutation == "rubric": changed["rubric_sha256"] = "0" * 64
        with pytest.raises(ValueError): score_prediction(item, "friendly", changed)


def test_public_rubric_is_frozen_for_adjudication_and_explicit_row_pins():
    assert RUBRIC_SHA256 == digest(ADJUDICATION_RUBRIC)
    item = row(scorer="semantic_review", answer="friendly", dimension="communication")
    item["evaluation"]["rubric_sha256"] = RUBRIC_SHA256
    assert score_prediction(item, "friendly", adjudication(item, "friendly", rating=3))["status"] == "COMPLETE"
    item["evaluation"]["rubric_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="rubric"):
        score_prediction(item, "friendly", adjudication(item, "friendly", rating=3))


def test_wrong_answer_never_implies_knowledge_or_execution_mechanism():
    item = row()
    result = score_prediction(item, "2")
    assert result["correct"] is False and result["error_categories"] == []
    assert result["diagnostic_status"] == "UNASSIGNED_NO_MECHANISM_CLAIM"
    review = adjudication(item, "2", correct=False)
    for r in review["reviews"]:
        r["error_evidence"] = {"missingknowledge": {"kind": "guess", "rationale": "Wrong answer", "artifact_sha256": "a" * 64}}
    with pytest.raises(ValueError): score_prediction(item, "2", review)


def test_shortcut_controls_are_fixed_and_original_input_only():
    item = row()
    expected = shortcut_controls(item)
    assert expected["constant_refusal"] == "UNKNOWN" and expected["constant_answer"] == "0"
    assert expected["label_cue"] == "visible-cue"
    assert "TARGET" not in str(expected)
    item["evaluation"]["answer"] = "LEAKED GOLD"
    item["messages"][-1]["content"] = "LEAKED TARGET"
    assert shortcut_controls(item) == expected
    report = score_shortcut_controls([row()])
    assert report["model_calls"] == 0 and not report["control_choices_use_gold"]
    assert report["controls"]["constant_refusal"]["correctness"] is None


def identity():
    return {"experiment": EXPERIMENT, "source_core": SOURCE_CORE, "source_weight_sha256": SOURCE_WEIGHT_SHA256,
        "compiled_plan_sha256": "1" * 64, "code_sha256": "2" * 64, "runtime": {"fixture": "only"},
        "initial_frozen_state_sha256": "3" * 64, "initial_parameters_sha256": "4" * 64,
        "generation_sha256": "5" * 64, "scorer_sha256": "6" * 64, "checkout": "fixture-only",
        "review_sha256": "7" * 64}


def full_fixture():
    rows, baseline, final, before_reviews, after_reviews = [], [], [], {}, {}
    for category, count in DEV_COUNTS.items():
        for number in range(count):
            item = row(category + str(number))
            item.update(category=category, family=category + str(number % 2))
            a = b = "1"
            if category in {"acquisition", "harder_transfer"}:
                item["evaluation"]["dimension"] = "solving" if category == "acquisition" else "harder_transfer"
                a = "1" if number < 64 else "0"
                b = "1" if number < (80 if category == "acquisition" else 72) else "0"
            if category == "grounding_contracts":
                item["evaluation"]["dimension"] = "grounding_answerability" if number < 96 else "output_contract"
                item["evaluation"]["critical"] = number < 96
                if number < 64:
                    item["evaluation"].update(scorer="semantic_review", answer="UNKNOWN", answerable=False)
                    a = b = "UNKNOWN"
            if category == "preservation_dialogue" and number < 8:
                item["evaluation"].update(scorer="semantic_review", dimension="communication", answer="friendly")
                a = b = "friendly"
            rows.append(item)
            base = {"id": item["id"], "input_sha256": prediction_hash(item["id"]), "generated_ids": [2, 1],
                "stop_reason": "eos", "context_tokens": 4, "generated_tokens": 2}
            baseline.append(dict(base, prediction=a))
            final.append(dict(base, prediction=b))
            if item["evaluation"]["scorer"] == "semantic_review":
                abstention = item["evaluation"]["answerable"] is False
                rating = 3 if item["evaluation"]["dimension"] == "communication" else None
                before_reviews[item["id"]] = adjudication(item, a, abstention=abstention, rating=rating)
                after_reviews[item["id"]] = adjudication(item, b, abstention=abstention, rating=rating)
    costs = {"forward_calls": 512, "processed_input_tokens": 2048, "generated_tokens": 1024, "wall_seconds": 1.}
    before = {"identity": identity(), "records": baseline, "costs": dict(costs), "updates": 0,
              "role": "source", "fixture_only": True}
    after = {"identity": identity(), "records": final, "costs": dict(costs),
             "updates": [{"step": 1, "loss": 1., "preclip_gradient_norm": 1., "update_l2": .001}],
             "role": "response_1e-5", "fixture_only": True}
    kwargs = {"eos_token_id": 1, "input_hashes": {r["id"]: r["input_sha256"] for r in baseline},
              "baseline_adjudications": before_reviews, "candidate_adjudications": after_reviews}
    return rows, before, after, kwargs


def test_full512_paired_quality_gates_and_fixture_attribution():
    rows, before, after, kwargs = full_fixture()
    result = compare_quality(rows, before, after, **kwargs)
    assert result["status"] == "DEV_CONDITIONS_MET"
    assert result["macro_family_gain_pp"]["acquisition"] == 12.5
    assert result["macro_family_gain_pp"]["harder_transfer"] == 6.25
    assert result["transitions"]["preservation_dialogue"]["source_retained_fraction"] == 1
    assert result["overall_repairs"] == 24 and result["overall_damages"] == 0
    assert result["advancement_conditions_met"] and not result["native_advancement_conditions_met"]
    assert not result["promotion_allowed"]


def test_missing_review_capture_cost_or_nonfinite_update_cannot_advance():
    rows, before, after, kwargs = full_fixture()
    variants = []
    missing = copy.deepcopy(kwargs)
    missing["candidate_adjudications"].pop(next(iter(missing["candidate_adjudications"])))
    variants.append((copy.deepcopy(after), missing))
    missing_capture = copy.deepcopy(after)
    missing_capture["records"].pop()
    variants.append((missing_capture, kwargs))
    missing_costs = copy.deepcopy(after)
    missing_costs["costs"] = None
    variants.append((missing_costs, kwargs))
    bad_update = copy.deepcopy(after)
    bad_update["updates"][0]["update_l2"] = float("nan")
    variants.append((bad_update, kwargs))
    for changed, options in variants:
        result = compare_quality(rows, before, changed, **options)
        assert result["status"] == "INCOMPLETE" and not result["advancement_conditions_met"]


def test_identity_original_input_and_duplicate_coverage_fail_closed():
    rows, before, after, kwargs = full_fixture()
    changed = copy.deepcopy(after)
    changed["identity"]["scorer_sha256"] = "8" * 64
    with pytest.raises(ValueError): compare_quality(rows, before, changed, **kwargs)
    changed = copy.deepcopy(after)
    changed["records"][0]["input_sha256"] = "9" * 64
    with pytest.raises(ValueError): compare_quality(rows, before, changed, **kwargs)
    changed = copy.deepcopy(after)
    changed["records"].append(changed["records"][0])
    with pytest.raises(ValueError): compare_quality(rows, before, changed, **kwargs)


def test_weaker_update_without_acquisition_is_not_success():
    rows, before, after, kwargs = full_fixture()
    for base, final in zip(before["records"], after["records"]):
        final["prediction"] = base["prediction"]
    after["updates"][0]["update_l2"] = 0.
    result = compare_quality(rows, before, after, **kwargs)
    assert result["status"] == "DEV_CONDITIONS_NOT_MET"
    assert not result["gates"]["acquisition_at_least_5pp"]


def test_complete_official_eos_list_accepts_secondary_terminal_and_rejects_earlier_eos():
    rows, before, after, kwargs = full_fixture()
    kwargs.pop("eos_token_id")
    kwargs["eos_token_ids"] = [151645, 151643]
    for envelope in (before, after):
        for record in envelope["records"]:
            record["generated_ids"] = [2, 151643]
    result = compare_quality(rows, before, after, **kwargs)
    assert result["status"] == "DEV_CONDITIONS_MET"
    assert result["source_summary"]["eos_token_ids"] == [151645, 151643]
    after["records"][0]["generated_ids"] = [151645, 151643]
    assert compare_quality(rows, before, after, **kwargs)["status"] == "INCOMPLETE"


def test_critical_damage_and_preservation_damage_are_reported_separately():
    rows, before, after, kwargs = full_fixture()
    critical = next(item["id"] for item in rows if item["category"] == "grounding_contracts"
                    and item["evaluation"]["scorer"] == "number" and item["evaluation"]["critical"])
    next(record for record in after["records"] if record["id"] == critical)["prediction"] = "0"
    result = compare_quality(rows, before, after, **kwargs)
    assert result["new_critical_grounding_loss_ids"] == [critical]
    assert result["status"] == "DEV_CONDITIONS_NOT_MET" and result["overall_damages"] == 1
    assert result["transitions"]["preservation_dialogue"]["damages"] == 0


def test_safe_noncanonical_refusal_to_unsafe_assertion_is_new_critical_loss_even_when_both_wrong():
    rows, before, after, kwargs = full_fixture()
    item = next(row for row in rows if row["evaluation"]["critical"] and row["evaluation"]["answerable"] is False)
    item["evaluation"]["contract"] = {"kind": "literal_unknown"}
    baseline_prediction = "I cannot determine that from the supplied evidence."
    candidate_prediction = "It is 7."
    next(record for record in before["records"] if record["id"] == item["id"])["prediction"] = baseline_prediction
    next(record for record in after["records"] if record["id"] == item["id"])["prediction"] = candidate_prediction
    kwargs["baseline_adjudications"][item["id"]] = adjudication(item, baseline_prediction, correct=False, abstention=True)
    kwargs["candidate_adjudications"][item["id"]] = adjudication(item, candidate_prediction, correct=False, abstention=False)
    result = compare_quality(rows, before, after, **kwargs)
    assert result["source_summary"]["scored"][item["id"]]["correct"] is False
    assert result["source_summary"]["scored"][item["id"]]["strict_format"] is False
    assert result["candidate_summary"]["scored"][item["id"]]["correct"] is False
    assert result["new_critical_answer_correctness_loss_ids"] == []
    assert result["new_critical_safe_abstention_loss_ids"] == [item["id"]]
    assert result["new_critical_grounding_loss_ids"] == [item["id"]]
    assert not result["gates"]["zero_new_critical_grounding_losses"]
    assert result["status"] == "DEV_CONDITIONS_NOT_MET"


def test_zero_preservation_source_success_denominator_is_undefined_not_perfect():
    rows, before, after, kwargs = full_fixture()
    preservation = {item["id"] for item in rows if item["category"] == "preservation_dialogue"}
    for record in before["records"]:
        if record["id"] not in preservation:
            continue
        if record["id"] in kwargs["baseline_adjudications"]:
            for review in kwargs["baseline_adjudications"][record["id"]]["reviews"]:
                review["semantic_correct"] = False
        else:
            record["prediction"] = "0"
    result = compare_quality(rows, before, after, **kwargs)
    assert result["transitions"]["preservation_dialogue"]["source_correct"] == 0
    assert result["transitions"]["preservation_dialogue"]["source_retained_fraction"] is None
    assert result["status"] == "INCONCLUSIVE" and not result["advancement_conditions_met"]
