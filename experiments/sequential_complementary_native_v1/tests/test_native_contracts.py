"""CPU software fixtures only. No checkpoint download, CUDA call, or native PASS.

Explicitly fabricated live-worker records below test adapter rejection paths; they
are not approvals or claims that a pretrained model was executed.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from experiments.model_quality_gpu_v1.integrity import digest, file_hash, write_new
from experiments.sequential_complementary_v1 import reasoning
from experiments.sequential_complementary_v1.data import PublicCase
from experiments.sequential_complementary_v1.integrity import ContractError
from experiments.sequential_complementary_native_v1 import contracts as c, native, run


class Tokenizer:
    eos_token = "<eos>"
    eos_token_id = 250

    def __init__(self):
        self.vocab = {"\n": 1, self.eos_token: self.eos_token_id}

    def encode(self, text, add_special_tokens=False):
        result = []
        for token in re.findall(r"<eos>|\n|[^\s<>]+|<[^>]+>", text):
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab) + 1
            result.append(self.vocab[token])
        return result

    def decode(self, ids, skip_special_tokens=False):
        reverse = {v: k for k, v in self.vocab.items()}
        return "".join(reverse.get(t, "token") for t in ids)

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert enable_thinking is False
        prompt = messages[:2]
        text = "\n".join(m["role"] + "\n" + m["content"] for m in prompt) + "\n<think>\n\n</think>\n\n"
        if not add_generation_prompt:
            text += messages[-1]["content"] + self.eos_token + "\n"
        return self.encode(text) if tokenize else text


class Cache:
    def __init__(self, length):
        self.length = length

    def get_seq_length(self):
        return self.length


class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(256, 4)

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(last_hidden_state=self.embed_tokens(input_ids))


class Model(torch.nn.Module):
    def __init__(self, *, generate=False, fail=False, bad_cache=False):
        super().__init__()
        self.model = Backbone()
        self.lm_head = torch.nn.Linear(4, 256, bias=False)
        self.lm_head.weight = self.model.embed_tokens.weight
        self.generate_eos, self.fail, self.bad_cache = generate, fail, bad_cache
        self.calls = []
        self.eval().requires_grad_(False)

    def forward(self, input_ids, past_key_values=None, use_cache=False, return_dict=True, logits_to_keep=1):
        assert not torch.is_grad_enabled()
        assert torch.is_inference_mode_enabled()
        previous = past_key_values.get_seq_length() if past_key_values is not None else 0
        self.calls.append((input_ids.shape[1], previous, use_cache, logits_to_keep))
        if self.fail:
            raise RuntimeError("fixture forward failure")
        logits = self.lm_head(self.model(input_ids).last_hidden_state)
        if self.generate_eos:
            logits.zero_(); logits[..., 250] = 1
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:]
        cache = Cache(previous + input_ids.shape[1] + int(self.bad_cache)) if use_cache else None
        return SimpleNamespace(logits=logits, past_key_values=cache)


def backend(tmp_path, **kwargs):
    events, charges = [], []
    write_new(tmp_path / "worker_claim.json", {"pid": os.getpid(), "fixture_only": True})
    return native.GenerationBackend(Model(**kwargs), Tokenizer(), charges.append, lambda **_: {},
                                    events.append, tmp_path, _key=native._LIVE), events, charges


def approved(binding=None):
    binding = binding or {"fixture": True}
    value = c.grant_request(binding)
    value.update(decision="APPROVED", approved_by="SOFTWARE_FIXTURE_NOT_AN_APPROVAL",
                 authorization_reference="pytest fixture", inference_qualification_permission=True)
    return binding, value


def tensor_file(path, header=None, payload=b"\0" * 8):
    header = header or {"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    return path


def test_exact_inventory_is_310_and_tail220():
    full, tail = c.shapes(), c.trained_shapes()
    assert len(full) == 310 and sum(math.prod(v) for v in full.values()) == 596049920
    assert len(tail) == 220 and sum(math.prod(v) for v in tail.values()) == 314618880
    assert all(8 <= int(n.split(".")[2]) <= 27 for n in tail)
    assert len({c.SOURCE_SHA, c.TRAINED_SHA, c.SUBJECT_SHA}) == 3
    assert c.SUBJECT != c.source_lock()["source_id"]


def test_header_reader_accepts_complete_tiny_safe_tensor(tmp_path):
    value = c.inventory(tensor_file(tmp_path / "tiny"), {"w": [2]}, "F32")
    assert value["tensor_count"] == 1 and value["parameters"] == 2


@pytest.mark.parametrize("mutation", ["dtype", "name", "shape", "boolshape", "offset", "gap", "booloffset", "metadata", "trailing", "short"])
def test_header_reader_rejects_malformed_tensors(tmp_path, mutation):
    header = {"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
    payload = b"\0" * 8
    if mutation == "dtype": header["w"]["dtype"] = "BF16"
    if mutation == "name": header["other"] = header.pop("w")
    if mutation == "shape": header["w"]["shape"] = [1, 2]
    if mutation == "boolshape": header["w"]["shape"] = [True, 2]
    if mutation == "offset": header["w"]["data_offsets"] = [0, 7]
    if mutation == "gap": header["w"]["data_offsets"] = [1, 9]; payload += b"\0"
    if mutation == "booloffset": header["w"]["data_offsets"] = [False, 8]
    if mutation == "metadata": header["__metadata__"] = {"bad": 1}
    if mutation == "trailing": payload += b"x"
    if mutation == "short": payload = b"x"
    with pytest.raises(ValueError):
        c.inventory(tensor_file(tmp_path / "tiny", header, payload), {"w": [2]}, "F32")


def test_header_rejects_duplicate_json_and_huge_header(tmp_path):
    path = tmp_path / "tiny"
    text = b'{"w":{},"w":{}}'
    path.write_bytes(struct.pack("<Q", len(text)) + text)
    with pytest.raises(ValueError, match="duplicate"):
        c.inventory(path, {"w": [2]}, "F32")
    path.write_bytes(struct.pack("<Q", 2**50))
    with pytest.raises(ValueError):
        c.inventory(path, {"w": [2]}, "F32")


def test_sparse_reconstruction_is_not_scalar_shrink():
    source = torch.zeros(8)
    trained = torch.tensor([4., -4., 4., 3., 2., 1., 0., 0.])
    selected = torch.tensor([4., -4., 0., 0., 0., 0., 0., 0.])
    native.verify_sparse_tensor(source, trained, selected)
    with pytest.raises(ValueError):
        native.verify_sparse_tensor(source, trained, trained * .25)
    with pytest.raises(ValueError):
        native.verify_sparse_tensor(source, trained, selected.bfloat16())


def test_signed_zero_bits_are_not_silently_equal():
    assert not native.bit_equal(torch.tensor([0.]), torch.tensor([-0.]))


@pytest.mark.parametrize("flag", c.CLOSED)
def test_no_approval_can_open_closed_lane(flag):
    binding, grant = approved()
    grant[flag] = True
    with pytest.raises(ValueError): c.validate_approval(grant, binding)


@pytest.mark.parametrize("value", [-1, 1.0, True, None, 1025])
def test_numeric_budget_rejects_invalid_values(value):
    binding, grant = approved()
    grant["limits"]["evaluation_and_qualification_model_calls"] = value
    with pytest.raises(ValueError): c.validate_approval(grant, binding)


@pytest.mark.parametrize("mutation", ["pending", "old_experiment", "old_schema", "wrong_binding", "missing_budget", "extra_budget", "excluded_lane", "no_approver", "no_reference", "training_tokens"])
def test_approval_requires_exact_new_scope(mutation):
    binding, grant = approved()
    if mutation == "pending": grant["decision"] = "PENDING"
    if mutation == "old_experiment": grant["experiment"] = "RRP-001-MICROFIT"
    if mutation == "old_schema": grant["schema"] = "RRP_GRANT_V1"
    if mutation == "wrong_binding": grant["binding_sha256"] = "a" * 64
    if mutation == "missing_budget": del grant["limits"]["retries"]
    if mutation == "extra_budget": grant["limits"]["future_training"] = 1
    if mutation == "excluded_lane": grant["excludes"].remove("training")
    if mutation == "no_approver": grant["approved_by"] = " "
    if mutation == "no_reference": grant["authorization_reference"] = None
    if mutation == "training_tokens": grant["limits"]["processed_training_tokens_all_attempts"] = 1
    with pytest.raises(ValueError): c.validate_approval(grant, binding)


def test_inference_only_approval_does_not_require_training_true():
    binding, grant = approved()
    c.validate_approval(grant, binding)
    assert all(grant[k] is False for k in c.CLOSED)
    assert grant["limits"]["retries"] == grant["limits"]["adaptive_model_selection_rounds"] == 0


def test_proposal_is_never_an_approval():
    binding = {"fixture": True}
    with pytest.raises(ValueError): c.validate_approval(c.grant_request(binding), binding)


def test_paths_cannot_dirty_checkout_or_follow_symlink(tmp_path, monkeypatch):
    root = tmp_path / "checkout"; root.mkdir()
    monkeypatch.setattr(c, "ROOT", root)
    with pytest.raises(ValueError): c.external(root / "outputs")
    alias = tmp_path / "alias"; alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError): c.external(alias / "outputs")
    assert c.external(tmp_path / "records") == tmp_path / "records"


@pytest.fixture
def git_repo(tmp_path, monkeypatch):
    path = tmp_path / "repo"; path.mkdir()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(path), *args], text=True, stderr=subprocess.STDOUT).strip()
    git("init", "-b", c.BRANCH)
    git("config", "user.name", "CPU fixture")
    git("config", "user.email", "fixture@example.invalid")
    (path / "code.py").write_text("original\n")
    git("add", "."); git("commit", "-m", "fixture")
    head = git("rev-parse", "HEAD")
    monkeypatch.setattr(c, "ROOT", path); monkeypatch.setattr(c, "PARENT", head)
    return path, head, git


def test_checkout_accepts_only_clean_exact_descendant(git_repo):
    path, head, git = git_repo
    assert c.checkout(head) == head
    with pytest.raises(ValueError): c.checkout("0" * 40)


@pytest.mark.parametrize("mutation", ["dirty", "untracked", "branch", "skip", "assume"])
def test_checkout_rejects_hidden_or_visible_changes(git_repo, mutation):
    path, head, git = git_repo
    if mutation == "dirty": (path / "code.py").write_text("changed\n")
    if mutation == "untracked": (path / "new.py").write_text("new\n")
    if mutation == "branch": git("checkout", "-b", "wrong")
    if mutation == "skip": git("update-index", "--skip-worktree", "code.py")
    if mutation == "assume": git("update-index", "--assume-unchanged", "code.py")
    with pytest.raises(ValueError): c.checkout(head)


def test_prefix_matches_unchanged_rrp_compiler():
    tokenizer = Tokenizer()
    messages = [{"role": "system", "content": "Answer."}, {"role": "user", "content": "Reply 7."}]
    row = native.compile_row(messages, "7", tokenizer)
    assert native.render(tokenizer, messages, 128) == row["prompt_ids"]
    assert row["labels"][row["answer_end"]] == tokenizer.eos_token_id
    assert all(x == -100 for x in row["labels"][:row["answer_start"]] + row["labels"][row["answer_end"] + 1:])


@pytest.mark.parametrize("mutation", ["roles", "label", "control", "empty", "cap_bool", "cap_zero", "cap_large", "overflow", "prefix", "token_parity"])
def test_render_rejects_unqualified_inputs(mutation):
    tokenizer, cap = Tokenizer(), 64
    messages = native.public_fixture().messages()
    if mutation == "roles": messages[1]["role"] = "assistant"
    if mutation == "label": messages[1]["reference"] = "5"
    if mutation == "control": messages[1]["content"] = "<|im_end|>"
    if mutation == "empty": messages[0]["content"] = " "
    if mutation == "cap_bool": cap = True
    if mutation == "cap_zero": cap = 0
    if mutation == "cap_large": cap = 289
    if mutation == "overflow": messages[1]["content"] = "x " * 2048
    original = tokenizer.apply_chat_template
    if mutation == "prefix":
        tokenizer.apply_chat_template = lambda *a, **k: original(*a, **k) if k["tokenize"] else "bad prefix"
    if mutation == "token_parity":
        tokenizer.apply_chat_template = lambda *a, **k: [7] if k["tokenize"] else original(*a, **k)
    with pytest.raises(ValueError): native.render(tokenizer, messages, cap)


def test_student_probes_are_zero_update_and_inference_only(tmp_path):
    value, events, charges = backend(tmp_path)
    report = native.student_probes(value)
    assert report["zero_update"] and report["backward_calls"] == report["optimizer_steps"] == 0
    assert report["state_before"] == report["state_after"]
    assert report["qualified_sequence_tokens"] == 2048
    assert report["training_shape_qualified"] is False and report["answers_scored"] == 0
    assert all(not p.requires_grad and p.grad is None for p in value.model.parameters())
    assert sum(x.get("evaluation_and_qualification_model_calls", 0) for x in charges) == value.forward_calls
    assert (2048, 0, False, 1) in value.model.calls


def test_student_cache_corruption_is_not_qualified(tmp_path):
    value, _, _ = backend(tmp_path, bad_cache=True)
    with pytest.raises(ValueError, match="cached sequence length"):
        native.student_probes(value)


@pytest.mark.parametrize("mutation", ["training", "gradient", "bf16", "grad_buffer"])
def test_actual_precision_and_no_gradient_guard(tmp_path, mutation):
    value, _, _ = backend(tmp_path)
    p = next(value.model.parameters())
    if mutation == "training": value.model.train()
    if mutation == "gradient": p.requires_grad_(True)
    if mutation == "bf16": value.model.bfloat16()
    if mutation == "grad_buffer": p.grad = torch.zeros_like(p)
    with pytest.raises(ValueError): native.student_probes(value)


def test_native_flag_is_not_a_qualification_callback():
    with pytest.raises(ContractError):
        reasoning.run_case(native.public_fixture(), "MR", object(), reasoning.Budget(5, 288, 10, lambda x: None), lambda x: None, native=True)


@pytest.mark.parametrize("mutation", ["key", "closed", "case", "arm", "reserve", "sink", "finished"])
def test_typed_adapter_still_requires_live_scope(tmp_path, mutation):
    value, events, _ = backend(tmp_path)
    case, arm, sink = native.public_fixture(), "MR", value.emit
    budget = reasoning.Budget(5, 288, 10, value.reserve_request)
    if mutation == "key": value._key = "a" * 64
    if mutation == "closed": value.close()
    if mutation == "case": case = dataclasses.replace(case, user="unregistered problem")
    if mutation == "arm": arm = "O"
    if mutation == "reserve": budget.reserve = lambda x: None
    if mutation == "sink": sink = lambda x: None
    if mutation == "finished": write_new(tmp_path / "supervisor.json", {})
    with pytest.raises(ContractError): reasoning.run_case(case, arm, value, budget, sink, native=True)


def test_native_rep_mr_use_frozen_policy_and_fresh_caches(tmp_path):
    value, events, charges = backend(tmp_path, generate=True)
    report = native.mr_probes(value, 10)
    assert report["zero_update"] and report["answers_scored"] == 0
    assert [r["arm"] for r in report["results"]] == ["REP", "MR"]
    assert all(r["classification"] == "NATIVE_QUALIFICATION_ONLY" for r in report["results"])
    assert all(r["malformed"] and not r["quality_claim"] for r in report["results"])
    assert len(value.model.calls) == 10 and all(call[1] == 0 for call in value.model.calls)
    assert sum(x.get("generation_requests", 0) for x in charges) == 10
    assert sum(x.get("generated_tokens_all_candidates", 0) for x in charges) == 576
    assert value.generated_tokens == 10  # EOS is counted, never silently removed.
    assert len([x for x in events if x["event"] == "NATIVE_TOKEN_RECORD"]) == 10
    assert [x["generated_ids_including_terminal"] for x in events if x["event"] == "NATIVE_TOKEN_RECORD"] == [[250]] * 10


def test_failed_forward_remains_reserved_and_cannot_retry(tmp_path):
    value, events, charges = backend(tmp_path, fail=True)
    budget = reasoning.Budget(5, 288, 10, value.reserve_request)
    with pytest.raises(RuntimeError, match="fixture forward"):
        reasoning.run_case(native.public_fixture(), "MR", value, budget, value.emit, native=True)
    assert budget.calls == 1 and value.forward_calls == 1
    assert sum(x.get("generated_tokens_all_candidates", 0) for x in charges) == 64
    assert any(x["event"] == "FAILED" for x in events)
    with pytest.raises(ContractError):
        reasoning.run_case(native.public_fixture(), "MR", value, budget, value.emit, native=True)


def test_generation_cannot_switch_prompt_after_authorization(tmp_path):
    value, _, _ = backend(tmp_path, generate=True)
    budget = reasoning.Budget(5, 288, 10, value.reserve_request)
    value.authorize(native.public_fixture(), "MR", budget, value.emit)
    budget.charge(64)
    with pytest.raises(ValueError, match="prompt differs"):
        value.generate(native.public_fixture().messages(), 64)
    assert value.forward_calls == 0


class Callback:
    def __init__(self, selector='{"selected":0}'):
        self.texts = iter(["FINAL: 5"] * 4 + [selector])
        self.prompts = []

    def check_context(self, messages, cap): pass

    def generate(self, messages, cap):
        self.prompts.append(messages)
        return {"text": next(self.texts), "terminated": True, "input_tokens": 1, "output_tokens": 1}


@pytest.mark.parametrize("arm,calls,tokens", [("O", 1, 64), ("T", 1, 288), ("REP", 5, 288), ("MR", 5, 288)])
def test_inherited_cpu_reasoning_behavior_is_preserved(arm, calls, tokens):
    value, charges, events = Callback(), [], []
    budget = reasoning.Budget(calls, tokens, 10, charges.append)
    result = reasoning.run_case(native.public_fixture(), arm, value, budget, events.append)
    assert result["classification"] == "SOFTWARE_FIXTURE_ONLY" and result["calls"] == calls
    assert budget.reserved_tokens == tokens and len(value.prompts) == calls
    if calls == 5:
        assert all("CANDIDATE_DATA_JSON" not in p[-1]["content"] for p in value.prompts[:4])
        assert "CANDIDATE_DATA_JSON" in value.prompts[-1][-1]["content"]


@pytest.mark.parametrize("selector", ['{"selected":true}', '{"selected":8}', '{"selected":0,"answer":"5"}', 'FINAL: 5'])
def test_inherited_selector_malformed_behavior_preserved(selector):
    value = Callback(selector)
    result = reasoning.run_case(native.public_fixture(), "MR", value, reasoning.Budget(5, 288, 10, lambda x: None), lambda x: None)
    assert result["malformed"] and not result["terminated"]


@pytest.mark.parametrize("command", ["train", "baseline", "teacher", "confirm", "resume"])
def test_cli_exposes_no_closed_command(command):
    with pytest.raises(SystemExit): run.parser().parse_args([command])


def test_pending_approval_rejected_before_assets_or_native(tmp_path, monkeypatch):
    binding = {"fixture": True}
    write_new(tmp_path / "binding.json", binding)
    write_new(tmp_path / "grant.json", c.grant_request(binding))
    called = []
    monkeypatch.setattr(c, "revalidate", lambda x: called.append(x))
    with pytest.raises(ValueError): run.inputs(tmp_path / "binding.json", tmp_path / "grant.json")
    assert not called


def test_frontend_import_does_not_import_torch_or_native():
    code = "import sys; from experiments.sequential_complementary_native_v1 import run; assert 'torch' not in sys.modules; assert 'experiments.sequential_complementary_native_v1.native' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], cwd=c.ROOT, check=True, timeout=10)
