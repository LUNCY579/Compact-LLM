"""Pre-execution validation for MRS-2. Synthetic fixtures only.

These are software checks. Passing them is NOT a model result and NOT evidence of
efficacy. They exist to make the instrument safe to point at a checkpoint.
"""
import json

import pytest

from experiments.multi_reasoning_scout_v2 import core, prompts, runner
from experiments.multi_reasoning_scout_v2.analyze import constant_baselines, degeneracy
from experiments.multi_reasoning_scout_v2.parsing import answer, majority, parse
from experiments.multi_reasoning_scout_v1.analysis import factorial

ROLES = list(prompts.ROLES)


# --------------------------------------------------------------- label blindness

def test_no_gold_answer_enters_any_role_prompt():
    """The answer tokens 0-3 occur naturally inside problem text, so absence of the
    gold string is not the right assertion. The right one is structural: a prompt is
    EXACTLY the fixed system text, the fixed role text and the public problem, with
    no channel through which a label could travel."""
    gold = core.gold_labels()
    for item in core.public_bank():
        assert gold[item.id]["id"] == item.id
        for role in prompts.INDEPENDENT:
            messages = prompts.messages(item, role)
            assert [m["role"] for m in messages] == ["system", "user"]
            assert messages[0]["content"] == prompts.SYSTEM
            assert messages[1]["content"] == prompts.ROLES[role] + "\n\nPROBLEM\n" + item.problem
            # The gold record carries fields that must never appear anywhere.
            blob = messages[0]["content"] + messages[1]["content"]
            for field in ("answerable", "difficulty", "verification", "scenario_id"):
                assert field not in blob


def test_prompt_is_a_pure_function_of_item_and_role():
    item = core.public_bank()[0]
    for role in prompts.INDEPENDENT:
        assert prompts.messages(item, role) == prompts.messages(item, role)


def test_selector_sees_exactly_the_registered_fields():
    item = core.public_bank()[0]
    records = [{"role": "R0", "case_id": item.id, "parsed": {"answer": "1", "secret": "x"},
                "format_failure": False, "terminated": True, "raw_text": "LEAK", "gold": "3"}]
    projected = prompts.projection(records)
    assert projected == [{"role": "R0", "answer": "1", "format_failure": False, "terminated": True}]
    blob = json.dumps(projected)
    for leak in ("LEAK", "secret", "gold", '"3"'):
        assert leak not in blob
    msg = prompts.messages(item, "S2", records)[1]["content"]
    assert "LEAK" not in msg and "secret" not in msg


def test_independent_roles_cannot_see_earlier_outputs():
    item = core.public_bank()[0]
    record = [{"role": "R0", "case_id": item.id, "parsed": {"answer": "1"},
               "format_failure": False, "terminated": True}]
    for role in prompts.INDEPENDENT:
        with pytest.raises(ValueError):
            prompts.messages(item, role, record)


def test_candidate_from_another_item_is_rejected():
    a, b = core.public_bank()[0], core.public_bank()[1]
    record = [{"role": "R0", "case_id": b.id, "parsed": {"answer": "1"},
               "format_failure": False, "terminated": True}]
    with pytest.raises(ValueError):
        prompts.messages(a, "S2", record)


def test_unregistered_role_or_non_public_item_is_rejected():
    item = core.public_bank()[0]
    with pytest.raises(TypeError):
        prompts.messages(item, "ORACLE")
    with pytest.raises(TypeError):
        prompts.messages({"id": "x", "problem": "p"}, "R0")


# --------------------------------------------------------------- parser fidelity

@pytest.mark.parametrize("token", ["0", "1", "2", "3", "UNKNOWN"])
def test_parser_accepts_the_vocabulary(token):
    r = parse(token, "R0", True)
    assert not r["format_failure"] and r["parsed"]["answer"] == token
    assert r["repair_count"] == 0


@pytest.mark.parametrize("bad", ["4", "unknown", "The answer is 1", '{"answer": "1"}',
                                 "1.", "", "  ", "ANSWER: 1", "1\n2"])
def test_parser_rejects_everything_else(bad):
    r = parse(bad, "R0", True)
    assert r["format_failure"] and r["parsed"] == {} and r["repair_count"] == 0


def test_nontermination_fails_closed_even_when_the_body_is_valid():
    r = parse("1", "R0", False)
    assert r["format_failure"] and r["nontermination"] and answer(r) is None


def test_malformed_output_stays_visible_and_is_never_repaired():
    r = parse("the answer is probably 2", "R0", True)
    assert r["format_failure"] and "parse_error" in r and r["repair_count"] == 0
    assert answer(r) is None


def test_majority_and_its_registered_tie_policy():
    def rec(a):
        return parse(a, "R0", True)
    assert majority([rec("1"), rec("1"), rec("2")]) == "1"
    # Tie resolves to the earliest valid role in the frozen order.
    assert majority([rec("2"), rec("1")]) == "2"
    assert majority([rec("x"), rec("1")]) == "1"
    assert majority([rec("x"), rec("y")]) is None


# --------------------------------------------------------------- bank integrity

def test_bank_is_the_mrs1_bank_unmodified():
    binding = core.mrs1_binding()
    assert binding["mrs1_freeze_sha256"] == (
        "278ce0962bd10864fba7d8ce2e7248579f18b22495d176c4f0f40db903a68837")
    assert len(binding["bank"]) == 3


def test_bank_population_and_splits():
    from collections import Counter
    bank = core.public_bank()
    assert len(bank) == 120
    assert dict(Counter(i.split for i in bank)) == {"SMOKE": 8, "CAL": 16, "CHECK": 96}
    assert len({i.id for i in bank}) == 120
    assert len({i.scenario_id for i in bank}) == 120


def test_no_quarantined_id_is_in_the_bank():
    access = core.read_json(core.MRS1_RECORDS / "ACCESS_MANIFEST.json")
    assert not {i.id for i in core.public_bank()} & set(access["quarantined_ids"])


def test_gold_is_stored_separately_from_model_visible_text():
    bank = core.public_bank()
    fields = set(vars(bank[0]))
    assert "answer" not in fields and "gold" not in fields
    gold = core.gold_labels()
    assert set(gold[bank[0].id]) >= {"answer", "answerable", "difficulty"}


# ------------------------------------------------------- baselines and degeneracy

def _row(cid, world, family, answers, split="CHECK"):
    outputs = {k: parse(v, k, True) for k, v in answers.items()}
    for k, rec in outputs.items():
        rec.update(case_id=cid, role=k, output_tokens=2, prompt_tokens=10, total_tokens=12,
                   generation_calls=1, forward_calls=2, wall_seconds=.01)
    return {"item": {"id": cid, "scenario_id": world, "family": family, "split": split,
                     "problem": "p"},
            "outputs": outputs,
            "arms": {"A0": "R0", "A1": "A1", "A2": "A2_selector", "A3": "A3_selector",
                     "S1": "S1_diagnostic"},
            "S0": majority([outputs["R0"], outputs["R1"], outputs["R2"], outputs["R3"]]),
            "A2_S0": majority([outputs[k] for k in ("R0", "D1", "D2", "D3")]),
            "logical_calls": {"A0": ["R0"], "A1": ["A1"],
                              "A2": ["R0", "D1", "D2", "D3", "A2_selector"],
                              "A3": ["R0", "R1", "R2", "R3", "A3_selector"],
                              "S1": ["R0", "R1", "R2", "R3", "S1_diagnostic"]}}


ALL_KEYS = ("R0", "A1", "D1", "D2", "D3", "A2_selector", "R1", "R2", "R3",
            "A3_selector", "S1_diagnostic")


def test_constant_baseline_reports_the_answer_prior():
    rows = [_row(f"c{i}", f"w{i}", "evidence_chain", {k: "0" for k in ALL_KEYS})
            for i in range(10)]
    labels = {f"c{i}": {"answer": "0" if i < 6 else "1", "answerable": True, "difficulty": "hard"}
              for i in range(10)}
    base = constant_baselines(rows, labels)
    assert base["best_constant_answer"] == "0"
    assert base["best_constant_accuracy"] == pytest.approx(.6)
    assert base["per_constant_answer"]["1"] == pytest.approx(.4)


def test_degeneracy_flags_a_constant_role():
    # This is the exact pathology MRS-FP-1 observed on the frozen source model.
    rows = [_row(f"c{i}", f"w{i}", "evidence_chain", {k: "0" for k in ALL_KEYS})
            for i in range(8)]
    deg = degeneracy(rows)
    assert deg["R0"]["constant"] is True
    assert deg["R0"]["distinct_answers"] == 1
    assert deg["R0"]["modal_share"] == pytest.approx(1.)


def test_degeneracy_does_not_flag_a_varied_role():
    rows = [_row(f"c{i}", f"w{i}", "evidence_chain",
                 {k: ("0" if i % 2 else "1") for k in ALL_KEYS}) for i in range(8)]
    assert degeneracy(rows)["R0"]["constant"] is False


# ------------------------------------------------- future C/D 2x2 on dummy fixtures

def _cell(correct_by_id):
    """Minimal summarize-shaped payload the factorial analysis consumes.

    The four hash bindings are required by `factorial`: it refuses to pair cells that
    were not produced under the same frozen artifacts. Dummy values are shared across
    the pair so the fixture exercises the arithmetic, not the binding check.
    """
    stub = "0" * 64
    return {"freeze_sha256": stub, "bank_sha256": stub, "prompt_sha256": stub,
            "protocol_sha256": stub,
            "per_item": [{"id": k, "scenario_id": "w" + k, "family": "evidence_chain",
                          "split": "CHECK", "difficulty": "hard", "answerable": True,
                          "correct": {"A0": v[0], "A1": v[0], "A2": v[0], "A3": v[1],
                                      "S1": v[1]}}
                         for k, v in sorted(correct_by_id.items())]}


def test_future_2x2_computes_the_intended_deltas():
    # A wrong->C correct on c1 (training repair); A wrong->B correct on c2 (multi repair);
    # C correct->D wrong on c3 (posttraining multi damage).
    old = _cell({"c1": (False, False), "c2": (False, True), "c3": (True, True),
                 "c4": (True, True)})
    new = _cell({"c1": (True, True), "c2": (False, True), "c3": (True, False),
                 "c4": (True, True)})
    result = factorial(old, new)
    overall = result["overall"]
    # A = old A0, B = old A3, C = new A0, D = new A3.
    assert overall["cells"] == {"A": .5, "B": .75, "C": .75, "D": .75}
    assert overall["deltas"] == {"delta_training_direct": .25, "delta_multi_old": .25,
                                 "delta_multi_new": 0., "interaction": -.25}
    counts = overall["counts"]
    assert counts["training_only_repair"] == 1        # c1: A wrong, C correct
    assert counts["multi_only_repair"] == 1           # c2: A wrong, B correct
    assert counts["both_available"] == 0
    assert counts["posttrain_multi_extra"] == 1       # c2: C wrong, D correct
    assert counts["multi_lost_after_training"] == 1   # c3: B correct, D wrong
    assert counts["harm_from_training"] == 0
    assert counts["harm_from_multi_pretrain"] == 0
    assert counts["harm_from_multi_posttrain"] == 1   # c3
    assert counts["all_four_correct"] == 1            # c4
    assert counts["training_makes_multi_harmful"] == 1
    # All 16 boolean correctness patterns are enumerated, and these four items land
    # in exactly the four expected ones.
    assert len(overall["patterns_ABCD"]) == 16
    assert {k: v for k, v in overall["patterns_ABCD"].items() if v} == {
        "0011": 1, "0101": 1, "1110": 1, "1111": 1}
    for delta in overall["clustered_CI"].values():
        assert delta["world_clusters"] == 4 and delta["draws"] == 2000
    assert result["evidence_class"] == "SOFTWARE_FIXTURE"
    assert result["confirmatory_claim"] is False and result["release_claim"] is False
    assert "evidence_chain" in result["per_family"]


def test_future_2x2_refuses_mismatched_populations():
    old = _cell({"c1": (True, True)})
    new = _cell({"c2": (True, True)})
    with pytest.raises((ValueError, KeyError)):
        factorial(old, new)


def test_future_2x2_refuses_a_native_claim_without_model_identity():
    old = _cell({"c1": (True, True)})
    new = _cell({"c1": (True, True)})
    old["native_execution_pass"] = True
    with pytest.raises(ValueError):
        factorial(old, new)


# ------------------------------------------------------------ guards and hygiene

def test_evidence_writes_are_create_only(tmp_path):
    target = tmp_path / "receipt.json"
    core.write_new(target, {"a": 1})
    with pytest.raises(FileExistsError):
        core.write_new(target, {"a": 2})


def test_resource_ceilings_fail_closed():
    from experiments.multi_reasoning_scout_v1 import budget
    good = {"host_available_bytes": 8 * 2**30, "gpu_free_bytes": 7 * 2**30,
            "disk_free_bytes": 20 * 2**30, "host_rss_bytes": 2**30,
            "gpu_allocated_bytes": 0, "gpu_peak_bytes": 0}
    budget.check(dict(good), preload=True)
    for key, value in (("host_available_bytes", 2**30), ("gpu_free_bytes", 2**30),
                       ("disk_free_bytes", 2**30)):
        with pytest.raises(RuntimeError):
            budget.check({**good, key: value}, preload=True)
    with pytest.raises(RuntimeError):
        budget.check({**good, "host_available_bytes": float("nan")}, preload=True)
    with pytest.raises(RuntimeError):
        budget.check({**good, "host_available_bytes": None}, preload=True)


def test_smoke_gate_fails_closed_and_covers_every_consumed_role():
    # Every role an arm actually consumes must be gated, or a silent hole exists.
    consumed = set()
    for keys in ({"A0": ["R0"], "A1": ["A1"],
                  "A2": ["R0", "D1", "D2", "D3", "A2_selector"],
                  "A3": ["R0", "R1", "R2", "R3", "A3_selector"],
                  "S1": ["R0", "R1", "R2", "R3", "S1_diagnostic"]}).values():
        consumed |= set(keys)
    # A1 and S1_diagnostic are descriptive arms, not part of the primary contrast.
    assert consumed - {"A1", "S1_diagnostic"} == set(runner.SMOKE_GATED)

    def row(valid):
        rec = parse("1" if valid else "junk", "R0", True)
        return {"outputs": {k: rec for k in runner.SMOKE_GATED}}

    assert runner.smoke_pass([row(True)] * 8)[0] is True
    assert runner.smoke_pass([row(False)] * 8)[0] is False
    # Exactly the MRS-1 threshold: 6 of 8 passes, 5 of 8 does not.
    assert runner.smoke_pass([row(True)] * 6 + [row(False)] * 2)[0] is True
    assert runner.smoke_pass([row(True)] * 5 + [row(False)] * 3)[0] is False


def test_the_observed_native_failure_mode_would_now_pass():
    # Both candidates emitted a bare "3"/"0"/"UNKNOWN" under MRS-1 and scored 0/8.
    for token in ("3", "0", "UNKNOWN"):
        assert not parse(token, "R0", True)["format_failure"]


def test_estimate_stays_inside_the_unchanged_budget():
    from experiments.multi_reasoning_scout_v1 import budget
    est = runner.estimate()
    per_item = est["unique_generation_token_allowance_per_item"]
    assert per_item * est["cases_per_model"] <= budget.LIMITS["max_generated_tokens_per_model"]
    assert est["unique_calls_per_item"] * est["cases_per_model"] <= budget.LIMITS["max_calls_per_model"]
    assert est["limits"]["training_steps"] == 0 and est["limits"]["retries"] == 0


def test_prompt_manifest_is_hashed_and_stable():
    m = prompts.manifest()
    assert len(m["sha256"]) == 64
    assert m["sha256"] == prompts.manifest()["sha256"]
    assert m["output_contract"] == "C1_BARE"


def test_system_prompt_is_the_probe_contract_verbatim():
    import importlib.util
    from pathlib import Path
    probe = Path(core.ROOT) / "experiments/mrs_format_probe_v1/contracts.py"
    spec = importlib.util.spec_from_file_location("_probe_contracts", probe)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert prompts.SYSTEM == module.CONTRACTS["C1_BARE"]["system"]
