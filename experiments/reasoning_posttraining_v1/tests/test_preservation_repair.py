import copy
from types import SimpleNamespace

import pytest

from experiments.reasoning_posttraining_v1 import grounded_replay as gr, repair_preservation as repair
from experiments.reasoning_posttraining_v1.serialization import compile_dialogue, compile_row
from experiments.reasoning_posttraining_v1.tests.test_data_contract import Tokenizer
from experiments.reasoning_posttraining_v1.tests.test_grounded_replay import source


class DialogueTokenizer(Tokenizer):
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kw):
        history = messages if add_generation_prompt else messages[:-1]
        prefix = "".join("<|im_start|>" + m["role"] + "\n" + m["content"] + self.eos_token + "\n"
                         for m in history)
        prefix += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        text = prefix if add_generation_prompt else prefix + messages[-1]["content"] + self.eos_token + "\n"
        return self.encode(text) if tokenize else text


def test_dialogue_preserves_prior_assistant_fact_but_supervises_only_final_turn():
    prompt = [{"role": "system", "content": "Remember the thread."},
              {"role": "user", "content": "Pick a room."},
              {"role": "assistant", "content": "Room 12."},
              {"role": "user", "content": "Which room?"}]
    tok = DialogueTokenizer()
    row = compile_dialogue(prompt, "Room 12.", tok)
    assert "Room 12." in tok.decode(row["prompt_ids"])
    assert row["labels"][:row["answer_start"]] == [-100] * row["answer_start"]
    assert tok.decode(row["ids"][row["answer_start"]:row["answer_end"]]) == "Room 12."
    assert row["labels"][row["answer_end"]] == tok.eos_token_id
    assert row["labels"][-1] == -100
    with pytest.raises(ValueError, match="only explicit"):
        compile_row(prompt, "Room 12.", tok)


@pytest.mark.parametrize("roles", [["system", "assistant"], ["system", "user", "user", "user"],
                                  ["system", "user", "assistant"]])
def test_dialogue_refuses_missing_or_invalid_turns(roles):
    with pytest.raises(ValueError):
        compile_dialogue([{"role": r, "content": "x"} for r in roles], "7", DialogueTokenizer())


def bank_row(row):
    return {"id": "procedural_rag:" + row["id"], "sub_bank": "procedural_rag",
            "source_id": row["id"], "reference_text": row["messages"][-1]["content"],
            "answerable": True, "decision": "answer", "pair_id": None,
            "required_response_kind": None, "prompt": copy.deepcopy(row["messages"][:-1]),
            "cluster_id": "old-conservative-template", "reference_aliases": [], "strata": {}}


def test_repaired_bank_requires_evidence_and_builds_a_matched_removal_pair():
    s = source()
    s["topic_id"] = "app77"
    old = bank_row(s)
    rows, bad, pairs, omitted = repair.corrected_rows([old], {"procedural_rag": {s["id"]: s}})
    assert not bad
    assert rows[0]["cluster_id"] == old["cluster_id"]
    assert omitted["procedural_rule_omitted"] == 1
    assert gr.solve_prompt(rows[0]["prompt"])["answer"] == s["messages"][-1]["content"]
    assert len(pairs) == 2 and pairs[0]["pair_id"] == pairs[1]["pair_id"]
    assert pairs[0]["prompt"][0] == pairs[1]["prompt"][0]
    import json
    a, b = [json.loads(r["prompt"][1]["content"][len(gr.MARKER):]) for r in pairs]
    assert a["question"] == b["question"] and a["procedures"] == b["procedures"]
    assert len(a["evidence"]) == 1 and b["evidence"] == []
    assert pairs[1]["reference_text"] == gr.MISSING
    assert old["prompt"] == s["messages"][:-1], "historical bank must remain untouched"


def test_bad_source_gold_is_excluded_not_silently_relabelled():
    s = source()
    s["messages"][-1]["content"] = gr.CONFLICT
    s["decision"] = "conflict"
    old = bank_row(s)
    rows, bad, pairs, _ = repair.corrected_rows([old], {"procedural_rag": {s["id"]: s}})
    assert not rows and not pairs
    assert bad[0]["action"] == "EXCLUDE_NOT_RELABEL"
    assert s["messages"][-1]["content"] == gr.CONFLICT


def test_baseline_scope_cannot_turn_into_training_or_change_generation(tmp_path):
    from experiments.reasoning_posttraining_v1.sources import stable_hash
    path = tmp_path / "bank.jsonl"
    path.write_text("[]", encoding="utf-8")
    plan = {"experiment": repair.AMENDED_EXPERIMENT,
            "baseline_protocol": {"bank_sha256": stable_hash(path), "max_sequence": 2048,
                                   "max_new_tokens": 64,
                                   "prediction_storage": "create_only_hash_chained_rows_v1",
                                   "cache_policy": "release_unused_between_cases"}}
    args = SimpleNamespace(command="baseline", bank=path, heldout_bank=None,
                           max_sequence=2048, max_new_tokens=64)
    repair.validate_scope(args, plan)
    args.max_new_tokens = 128
    with pytest.raises(ValueError, match="inference protocol"):
        repair.validate_scope(args, plan)
    args.command = "train"
    with pytest.raises(ValueError, match="not training"):
        repair.validate_scope(args, plan)


def test_new_inference_qualification_executes_2048_and_counts_all_forwards():
    import torch
    from experiments.reasoning_posttraining_v1.instrument import qualify_inference_window

    class CausalFixture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.calls = 0

        def forward(self, input_ids, past_key_values=None, logits_to_keep=8, **kw):
            self.calls += 1
            x = input_ids.float().cumsum(dim=1)
            if past_key_values is not None:
                x += past_key_values
            return SimpleNamespace(logits=x[:, -logits_to_keep:, None] * self.weight,
                                   past_key_values=x[:, -1:])

    class ShapeTokenizer:
        eos_token_id = 9

        def encode(self, text, **kw):
            return [3] if text == "8" else [1, 2]

    model, charges = CausalFixture(), []
    result = qualify_inference_window(model, ShapeTokenizer(), 2048, charges.append, lambda: None)
    assert model.calls == result["qualification_forward_calls"] == 6
    assert charges == [{"evaluation_and_qualification_model_calls": 6}]
    assert result["backward_passes"] == result["optimizer_steps"] == 0
    assert result["source_state_before"] == result["source_state_after"]
