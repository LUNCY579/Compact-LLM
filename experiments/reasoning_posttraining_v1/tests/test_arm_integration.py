"""Exercise builder-to-native seams without pretrained weights or CUDA."""
from types import SimpleNamespace

import pytest
import torch

from experiments.reasoning_posttraining_v1 import arm, controller, native_runtime
from experiments.reasoning_posttraining_v1.tests.test_arm import meta_for, policy_for
from experiments.reasoning_posttraining_v1.tests.test_data_contract import Tokenizer


def compiled():
    rows, rejected = arm.compile_training(
        ["w1"], {"w1": policy_for("w1")},
        {"w1": meta_for("w1", "ledger_replay", "ANSWER")}, Tokenizer())
    assert not rejected
    return rows


class EOSModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.lm_head = SimpleNamespace(weight=self.weight)

    def forward(self, **kwargs):
        return SimpleNamespace(logits=torch.tensor([[[0., 1.]]]), past_key_values=None)


def test_real_builder_rows_reach_native_probe(monkeypatch):
    """Previously fails KeyError('reference') on the first training baseline probe."""
    rows = compiled()
    monkeypatch.setattr(native_runtime, "hidden_states", lambda *args: torch.zeros(1))
    monkeypatch.setattr(native_runtime, "chunked_sum",
                        lambda *args, **kwargs: (torch.tensor(0.), 1))
    got = native_runtime.evaluate(EOSModel(), rows, Tokenizer(), [1], "fixture",
                                  lambda _: None, lambda: None, max_new=1)
    assert got["rows"] == 1
    assert got["records"][0]["reference"] == '{"value":7}'


def test_malformed_dev_prompt_cannot_earn_a_clean_leakage_audit():
    with pytest.raises(ValueError, match="prompt"):
        arm.disjointness(compiled(), {"broken": [{"id": "dev", "prompt": []}]})


def test_parent_alias_collision_is_detected():
    rows = compiled()
    dev = {"id": "different", "parent_id": rows[0]["parent_id"],
           "prompt": policy_for("unrelated")["prompt"]}
    assert not arm.disjointness(rows, {"dev": [dev]})["clean"]


def test_all_dev_banks_must_be_nonempty():
    with pytest.raises(ValueError, match="nonempty"):
        arm.disjointness(compiled(), {"missing": []})


def test_training_heldout_path_runs_and_retains_shared_window_receipt(tmp_path, monkeypatch):
    import json
    from experiments.reasoning_posttraining_v1 import instrument
    paths = [tmp_path / f"bank{i}.jsonl" for i in range(2)]
    for p in paths:
        p.write_text(json.dumps({"id": p.stem, "reference": "7"}), encoding="utf-8")
    calls = []
    def qualify(*args):
        calls.append(args[2])
        return {"status": "CPU_FIXTURE", "window": args[2]}
    monkeypatch.setattr(instrument, "qualify_evaluation_window", qualify)
    windows, caps = [], []
    def compile_bank(bank, *a, **kw):
        windows.append(kw["max_sequence"])
        return bank, []
    def evaluate(*a, **kw):
        caps.append(kw["max_new"])
        return {"fixture": True}
    monkeypatch.setattr(instrument, "compile_bank", compile_bank)
    protocols = {controller.stable_hash(p): {"max_sequence": w, "max_new_tokens": cap}
                 for p, w, cap in zip(paths, [4096, 2048], [128, 64])}
    got = controller.evaluate_heldout(
        SimpleNamespace(heldout_bank=paths, eval_max_sequence=4096),
        SimpleNamespace(max_sequence=2048), {"max_new_tokens": 128, "evaluation_protocols": protocols}, EOSModel(),
        Tokenizer(), [1], SimpleNamespace(reserve=lambda _: None), tmp_path,
        evaluate, lambda: None)
    assert len(got) == 2
    assert calls == [4096]
    assert windows == [4096, 2048]
    assert caps == [128, 64]
    receipt = json.loads((tmp_path / "heldout-bank0.json").read_text())
    assert receipt["candidate_state_before"] == receipt["candidate_state_after"]
    assert receipt["evaluation_window_qualification"]["window"] == 4096


def test_heldout_mutation_is_refused(tmp_path, monkeypatch):
    import json
    from experiments.reasoning_posttraining_v1 import instrument
    path = tmp_path / "bank.jsonl"
    path.write_text(json.dumps({"id": "dev", "reference": "7"}), encoding="utf-8")
    monkeypatch.setattr(instrument, "compile_bank", lambda bank, *a, **k: (bank, []))
    model = EOSModel()
    def corrupt(*a, **k):
        with torch.no_grad():
            model.weight.add_(1)
        return {}
    with pytest.raises(ValueError, match="changed candidate"):
        controller.evaluate_heldout(
            SimpleNamespace(heldout_bank=[path], eval_max_sequence=2048),
            SimpleNamespace(max_sequence=2048), {"max_new_tokens": 1}, model,
            Tokenizer(), [1], SimpleNamespace(reserve=lambda _: None), tmp_path,
            corrupt, lambda: None)


def test_original_a1_budget_cannot_start_but_corrected_registration_can(tmp_path):
    from experiments.reasoning_posttraining_v1 import budget, campaign, instrument
    old = {"experiment": instrument.ARM_A1_EXPERIMENT, "limits": dict(instrument.ARM_A1_LIMITS)}
    journal = budget.Journal(tmp_path / "old", old, create=True)
    with pytest.raises(RuntimeError, match="insufficient approved budget"):
        journal.start("train", "train", 10)
    new = {"experiment": campaign.EXPERIMENT, "limits": dict(campaign.LIMITS)}
    journal = budget.Journal(tmp_path / "new", new, create=True)
    journal.start("train", "train", 10)
    assert journal.read()["used_or_reserved"]["adaptive_model_selection_rounds"] == 1
    assert instrument.ARM_A1_LIMITS["adaptive_model_selection_rounds"] == 0


def test_corrected_scope_refuses_bank_substitution_or_protocol_overrides(tmp_path):
    from experiments.reasoning_posttraining_v1 import campaign
    path = tmp_path / "bank"
    path.write_text("[]", encoding="utf-8")
    sha = controller.stable_hash(path)
    plan = {"heldout_banks": {str(path.resolve()): sha},
            "evaluation_protocols": {sha: {"max_sequence": 2048, "max_new_tokens": 64}}}
    args = SimpleNamespace(heldout_bank=[path], max_new_tokens=None, eval_max_sequence=None)
    campaign.validate_execution_scope(args, plan, compiled())
    args.max_new_tokens = 128
    with pytest.raises(ValueError, match="frozen baseline protocols"):
        campaign.validate_execution_scope(args, plan, compiled())
    args.max_new_tokens = None
    path.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="registered held-out banks"):
        campaign.validate_execution_scope(args, plan, compiled())


def test_shuffle_is_deterministic_and_preserves_the_selected_worlds():
    from experiments.reasoning_posttraining_v1 import campaign
    rows = [{"id": str(i), "family": str(i // 10), "stratum": "ANSWER"} for i in range(100)]
    shuffled = campaign.interleave(rows)
    assert shuffled == campaign.interleave(list(reversed(rows)))
    assert {r["id"] for r in shuffled} == {r["id"] for r in rows}
    assert shuffled != rows
    assert max(r["rows"] for r in campaign.runs(shuffled)) < 10
