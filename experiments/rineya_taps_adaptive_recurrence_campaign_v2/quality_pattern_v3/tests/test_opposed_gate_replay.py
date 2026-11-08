"""Meaningful malformed-evidence, weighting and nomination fixtures only."""
from copy import deepcopy
import pytest

from experiments.rineya_taps_adaptive_recurrence_campaign_v2.quality_pattern_v3 import opposed_gate_replay as replay
from experiments.rineya_taps_adaptive_recurrence_campaign_v2.quality_pattern_v3.opposed_gate_policy import POLICIES, CONFIDENT, UNCERTAIN, GUARD
from experiments.rineya_taps_adaptive_recurrence_campaign_v2.quality_pattern_v3.probe_analysis import READ_BODY, TASK_CONTEXT, digest, text_sha


def input_fixture():
    population = []
    readouts = []
    for index, probability in enumerate((.5, .25)):
        text = "DOCUMENTS: use only A-approved records. Equal approved values give that value; absent/conflicting values give UNKNOWN.\nABC D 7;DEF A 9.\nQuestion: ABC? Reply one integer or UNKNOWN."
        ref = {"case_id": f"fixture{index}", "split": "DEVELOPMENT", "prompt": text,
               "prompt_sha256": text_sha(text), "rendered_prompt": text,
               "rendered_prompt_sha256": text_sha(text), "input_token_ids": [7, 8],
               "prefix_ids_sha256": digest([7, 8]), "prefix_tokens": 2}
        population.append(ref)
        for mode in ("OBSERVED_K0", "OBSERVED_UNIT"):
            row = {key: None for key in READ_BODY | TASK_CONTEXT}
            row.update({k: ref[k] for k in ("case_id", "split", "prompt_sha256", "rendered_prompt_sha256", "prefix_ids_sha256")})
            row.update(schema="PASSIVE_INPUT_FEATURES_V1", phase="TASK", mode=mode,
                       vocabulary_rows=151936, UNKNOWN_token_id=51378, finite=True,
                       target_independent=True, teacher_conditioned=False, preceding_gold_tokens=0,
                       prefix_kind="ORIGINAL_INPUT_ONLY", logits_changed=False,
                       top1_probability=probability, top1_token_id=10)
            readouts.append(row)
    return population, readouts


def test_decisions_have_no_label_or_response_phase():
    population, readouts = input_fixture()
    result, _ = replay.decisions_before_labels(population, readouts)
    assert not result["label_phase_started"]
    assert result["rates"][CONFIDENT] == .5
    assert result["rates"][UNCERTAIN] == .5
    assert result["rates"][GUARD] == 0.
    assert all(row["certificate"] == "ABSENT" for row in result["decisions"][GUARD])


@pytest.mark.parametrize("mutate", [
    lambda p, r: p[0].update(gold="UNKNOWN"),
    lambda p, r: r[0].update(gold="UNKNOWN"),
    lambda p, r: r.append(deepcopy(r[0])),
    lambda p, r: r.pop(),
    lambda p, r: r[0].update(teacher_conditioned=True),
    lambda p, r: r[0].update(prompt_sha256="changed"),
    lambda p, r: r[0].update(split="OPEN_VALIDATION"),
    lambda p, r: r[0].update(top1_probability=float("nan")),
])
def test_decision_inputs_fail_closed(mutate):
    population, readouts = input_fixture()
    mutate(population, readouts)
    with pytest.raises(ValueError):
        replay.decisions_before_labels(population, readouts)


def history_fixture():
    population, readouts = input_fixture()
    _, indexed = replay.decisions_before_labels(population, readouts)
    bank, history = [], []
    for ref in population:
        bank.append({"case_id": ref["case_id"], "split": "DEVELOPMENT", "prompt": ref["prompt"],
                     "category": "document", "parent_id": ref["case_id"], "gold": "UNKNOWN",
                     "answer_kind": "integer_or_unknown"})
        for arm, alpha in (("A0", None), ("A1", None), ("A2", 1.)):
            history.append({"case_id": ref["case_id"], "split": "DEVELOPMENT", "arm": arm,
                            "fixed_alpha": alpha, "physical_run": True, "configured_seed": 17,
                            "seed": 17, "extra_loops": int(arm != "A0"), "safety_stopped": False,
                            "token_ids": [10, 151643], "output_tokens": 2,
                            "full_model_calls_including_gold_nll": 2, "termination": "eos",
                            "eos_emitted": True, "truncated": False, "response": "UNKNOWN", "score": 1.})
    return population, indexed, bank, history


def test_eos_token_counts_and_aliases_are_not_scientific_rows():
    population, indexed, bank, history = history_fixture()
    history.append({**history[0], "arm": "A4_example", "physical_run": False})
    cases, physical = replay.grade_history(population, bank, history, indexed)
    assert len(cases) == 2
    assert all(len(rows) == 2 for rows in physical.values())


@pytest.mark.parametrize("mutate", [
    lambda p, b, h: h[0].update(output_tokens=1),
    lambda p, b, h: h[0].update(termination="cap"),
    lambda p, b, h: h[0].update(token_ids=[10, 11]),
    lambda p, b, h: h[0].update(safety_stopped=True),
    lambda p, b, h: h[0].update(physical_run=False),
    lambda p, b, h: h[2].update(response="9", score=0.),
    lambda p, b, h: h.append(deepcopy(h[0])),
    lambda p, b, h: b[0].update(prompt="different user text"),
])
def test_invalid_baseline_evidence_rejected(mutate):
    population, indexed, bank, history = history_fixture()
    mutate(population, bank, history)
    with pytest.raises(ValueError):
        replay.grade_history(population, bank, history, indexed)


def test_parent_then_family_weights_and_fractional_random_endpoints():
    cases = {
        "a0": {"category": "A", "parent_id": "a", "gold": "1"},
        "a1": {"category": "A", "parent_id": "a", "gold": "1"},
        "b0": {"category": "B", "parent_id": "b", "gold": "UNKNOWN"},
    }
    def row(score):
        return {key: score for key in ("correct", "format_valid", "literal_UNKNOWN", "false_UNKNOWN",
                                      "unsupported_canonical_answer", "eos_emitted", "truncated")}
    summary = replay.summarize(cases, {"a0": row(.5), "a1": row(.5), "b0": row(1.)})
    assert summary["equal_family_parent_accuracy"] == .75
    assert summary["correct"] == 2.
    assert summary["parent_units"] == 2


def nomination_fixture():
    entries = {}
    for policy in POLICIES:
        entries[policy] = {"summary": {"equal_family_parent_accuracy": .4},
                           "random_mean": {"equal_family_parent_accuracy": .3},
                           "vs_unit": {"losses": [], "family_delta": {"A": .1},
                                       "format_delta": 0., "unsupported_delta": 0., "false_UNKNOWN_delta": 0.}}
    return entries


def test_nomination_tie_priority_and_any_retention_loss():
    entries = nomination_fixture()
    assert replay.nominate(entries, .3, .2) == CONFIDENT
    entries = nomination_fixture()
    entries[CONFIDENT]["vs_unit"]["losses"] = ["single lost correct case"]
    assert replay.nominate(entries, .3, .2) == UNCERTAIN


def test_nomination_rejects_guard_header_equality_and_all_harms():
    entries = nomination_fixture()
    entries[CONFIDENT]["vs_unit"]["format_delta"] = -1.
    entries[UNCERTAIN]["vs_unit"]["false_UNKNOWN_delta"] = 1.
    assert replay.nominate(entries, .3, .4) is None
    assert not entries[GUARD]["mechanical_nomination_checks"]["guard_beats_header"]


def test_admission_not_interchangeable_with_previous_probe(tmp_path):
    with pytest.raises(PermissionError):
        replay.verify_admission(tmp_path / "probe_admission_v3.json")
