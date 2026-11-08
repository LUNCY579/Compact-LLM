import copy
import json
from types import SimpleNamespace

import pytest

from experiments.reasoning_posttraining_v1 import grounded_replay as gr
from experiments.reasoning_posttraining_v1 import controller


def source():
    return {"id": "test", "procedure_family": "approval", "decision": "answer",
            "question": "For audit APP77, record 2, apply the trusted procedure. What is the result?",
            "messages": [{"role": "system", "content": "Use supplied evidence."},
                         {"role": "user", "content": "For audit APP77, record 2, apply the trusted procedure. What is the result?"},
                         {"role": "assistant", "content": "Tier 2."}],
            "procedures": [{"procedure_id": "p", "trust_level": "trusted",
                            "text": "For audit APP77, bind only the explicitly requested record. " + gr.RULES[5]}],
            "evidence": [{"document_id": "d", "trust_level": "standard",
                          "text": "Audit APP77, record 2. Request amount is $1200. Risk is low."}]}


@pytest.mark.parametrize("rule,text,expected", [
    (0, "Relay Atlas latency is 17 ms. Relay Beacon latency is 12 ms.", "Relay Beacon."),
    (1, "Demand is 97. Pool Atlas capacity 64; Pool Beacon 96; Pool Cedar 128; Pool Delta 192.", "Pool Cedar."),
    (2, "Required score is 70. Observed score is 70.", "Pass."),
    (3, "Candidate Atlas version 2.10.1. Candidate Beacon version 2.9.20.", "Version 2.10.1."),
    (4, "Task Atlas priority 3. Task Beacon priority 1.", "Task Beacon."),
    (5, "Request amount is $999. Risk is high.", "Tier 2."),
    (5, "Request amount is $999. Risk is low.", "Tier 1."),
    (6, "Retry attempt number is 20.", "60 seconds."),
    (6, "Retry attempt number is 3.", "8 seconds."),
    (7, "Remaining quota is 20. Requested amount is 20.", "Allow."),
    (7, "Remaining quota is 20. Requested amount is 21.", "Deny."),
])
def test_independent_solver_on_hand_calculated_cases(rule, text, expected):
    assert gr.solve_record(gr.RULES[rule], text) == expected


def test_render_keeps_evidence_and_rule_but_never_target_metadata():
    row = source()
    row.update(approved_answer="SECRET-GOLD", reference="SECRET-GOLD", decision="SECRET-LABEL")
    prompt = gr.render_procedure(row)
    assert row["procedures"][0]["text"] in prompt[-1]["content"]
    assert row["evidence"][0]["text"] in prompt[-1]["content"]
    assert "SECRET" not in prompt[-1]["content"]
    assert gr.solve_prompt(prompt)["answer"] == "Tier 2."
    row["messages"][-1]["content"] = "WRONG-GOLD"
    assert gr.audit_source(row)["gold_matches"] is False
    assert gr.render_procedure(row) == prompt


def test_missing_evidence_and_nonbinding_sibling_are_not_an_answer():
    row = source()
    row["evidence"][0]["text"] = row["evidence"][0]["text"].replace("record 2.", "record 20.")
    assert gr.solve_prompt(gr.render_procedure(row))["answer"] == gr.MISSING
    row["evidence"][0]["text"] = row["evidence"][0]["text"].replace("APP77", "APP770")
    assert gr.solve_prompt(gr.render_procedure(row))["answer"] == gr.MISSING


def test_conflicting_inputs_with_agreeing_results_are_not_forced_abstention():
    row = source()
    alternate = copy.deepcopy(row["evidence"][0])
    alternate["document_id"] = "alternate"
    alternate["text"] = alternate["text"].replace("Risk is low.", "Risk is high.")
    row["evidence"].append(alternate)
    assert gr.solve_prompt(gr.render_procedure(row))["answer"] == "Tier 2."
    row["evidence"][0]["text"] = row["evidence"][0]["text"].replace("$1200", "$900")
    assert gr.solve_prompt(gr.render_procedure(row))["answer"] == gr.CONFLICT


def test_unknown_rule_is_refused_even_when_evidence_is_missing():
    row = source()
    row["procedures"][0]["text"] += " Unreviewed modification."
    row["evidence"] = []
    with pytest.raises(ValueError, match="unrecognized visible procedure"):
        gr.solve_prompt(gr.render_procedure(row))


def test_untrusted_evidence_cannot_supply_the_answer():
    row = source()
    row["evidence"][0]["trust_level"] = "untrusted"
    assert gr.solve_prompt(gr.render_procedure(row))["answer"] == gr.MISSING


def test_tied_minimum_is_refused_instead_of_inventing_a_winner():
    with pytest.raises(ValueError, match="tie"):
        gr.solve_record(gr.RULES[0], "Relay Atlas latency is 7 ms. Relay Beacon latency is 7 ms.")


def test_prior_assistant_facts_stay_in_the_prompt():
    row = source()
    row["messages"] = [{"role": "system", "content": "s"},
                       {"role": "user", "content": "Choose a room."},
                       {"role": "assistant", "content": "Room 12."},
                       {"role": "user", "content": "Which room did you choose?"},
                       {"role": "assistant", "content": "Room 12."}]
    assert gr.conversation_prompt(row) == row["messages"][:-1]


def test_invalidated_bank_refused_before_prepared_data_or_grant_is_opened(monkeypatch):
    sha = next(iter(controller.INVALIDATED_BANKS))
    monkeypatch.setattr(controller, "stable_hash", lambda _: sha)
    with pytest.raises(ValueError, match="invalidated evaluation bank"):
        controller.inputs(SimpleNamespace(bank="old", heldout_bank=[]))
