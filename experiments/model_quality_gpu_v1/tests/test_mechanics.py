from __future__ import annotations
import gc
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest
import torch
from torch.nn import functional as F

from experiments.model_quality_gpu_v1 import integrity as I
from experiments.model_quality_gpu_v1.checkpoints import (restore_checkpoint, save_checkpoint,
                                                         validate_checkpoint)
from experiments.model_quality_gpu_v1.config import ARMS, Design, rate_at, validate_native_design
from experiments.model_quality_gpu_v1.data import (compile_plan, encode_row, group_updates, overlap_audit,
                                                  read_rows, review_template, targets, validate_plan, validate_review)
from experiments.model_quality_gpu_v1.engine import expected_progress, train_cell
from experiments.model_quality_gpu_v1.evaluation import classify
from experiments.model_quality_gpu_v1.losses import chunked_sum
from experiments.model_quality_gpu_v1.native import parameter_digest, select_tail
from experiments.model_quality_gpu_v1.optimizer import CpuAdamW
from experiments.model_quality_gpu_v1.resources import bounded_child, lease
from experiments.model_quality_gpu_v1.tests.fixtures import ByteTokenizer, TinyLM, row, synthetic_plan


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(1)
    torch.manual_seed(17)


@pytest.mark.parametrize("field,value", [
    ("processed_budget", 0), ("processed_budget", True), ("max_sequence", 0), ("ce_chunk", -1),
    ("warmup_fraction", 0), ("decay_fraction", 0), ("beta2", 1), ("epsilon", 0),
    ("clip_norm", float("nan")), ("weight_decay", -1), ("order_seed", -1),
    ("selection_seed", True), ("min_budget_fraction", 1.1), ("minimum_lr_fraction", 0),
    ("update_tokens", 128), ("timeout_seconds", 0)])
def test_design_rejects_invalid(field, value):
    with pytest.raises(ValueError): replace(Design(), **{field: value})


def test_wsd_boundaries_and_distinct_arms():
    d = Design()
    assert len(ARMS) == 6
    assert rate_at(1.0, 10, 100, d) == pytest.approx(0.5)
    assert rate_at(1.0, 20, 100, d) == 1.0
    assert rate_at(1.0, 80, 100, d) == 1.0
    assert rate_at(1.0, 100, 100, d) == pytest.approx(0.1)
    assert rate_at(1.0, 90, 100, d) < rate_at(1.0, 85, 100, d)
    for invalid in (0, 101, -2):
        with pytest.raises(ValueError): rate_at(1, invalid, 100, d)
    with pytest.raises(ValueError): validate_native_design(replace(d, beta2=0.999))
    validate_native_design(Design(order_seed=29))


@pytest.mark.parametrize("bad", ['{"a":1,"a":2}', '{"x":NaN}', '{"x":Infinity}'])
def test_json_rejects_bad(bad):
    with pytest.raises(ValueError): I.decode(bad)


def test_create_only_and_digest(tmp_path):
    p = tmp_path / "a.json"
    I.write_new(p, I.seal({"a": [1, "é"]}))
    assert I.unseal(I.read_json(p)) == {"a": [1, "é"]}
    with pytest.raises(FileExistsError): I.write_new(p, {})
    v = I.read_json(p); v["a"] = []
    with pytest.raises(ValueError): I.unseal(v)
    # Traversal rejection needs no symlink, so it runs unconditionally and stays covered
    # even where the symlink half below cannot.
    with pytest.raises(ValueError): I.new_dir(tmp_path / ".." / "unsafe")
    # Creating a symlink on Windows needs admin or Developer Mode, and this package's own
    # runbook targets Windows. Unguarded, this line fails with WinError 1314 and takes the
    # create-only, seal/unseal and traversal assertions down with it -- reporting a
    # privilege the operator lacks as a broken integrity layer.
    alias = tmp_path / "link"
    try:
        alias.symlink_to(p)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink creation unavailable, cannot exercise symlink rejection: {exc}")
    with pytest.raises(ValueError): I.file_hash(alias)


def test_chat_response_mask_includes_eos_excludes_prefix_suffix():
    r = encode_row(row(1), ByteTokenizer(), Design())
    response, whole = targets(r, "response"), targets(r, "whole")
    assert all(x == -100 for x in response[:r["answer_start"]])
    assert response[r["answer_end"]] == 2
    assert all(x == -100 for x in response[r["answer_end"] + 1:])
    assert whole[0] == -100 and whole[1:] == r["ids"][1:]
    assert sum(x != -100 for x in response) == 6


@pytest.mark.parametrize("kind", ["template", "prefix", "eos", "long", "native_ids"])
def test_tokenization_mismatch_fails(kind):
    tok, d = ByteTokenizer(), Design()
    if kind == "template":
        old = tok.apply_chat_template
        tok.apply_chat_template = lambda *a, **kw: ("bad" if not kw["tokenize"] else old(*a, **kw))
    elif kind == "prefix":
        old = tok.encode
        tok.encode = lambda text, **kw: ([4] + old(text, **kw) if text.endswith("assistant:") else old(text, **kw))
    elif kind == "eos": tok.eos_token_id = 3
    elif kind == "long": d = replace(d, max_sequence=4)
    else:
        old = tok.apply_chat_template
        tok.apply_chat_template = lambda *a, **kw: ([5] if kw["tokenize"] else old(*a, **kw))
    with pytest.raises(ValueError): encode_row(row(1), tok, d)


@pytest.mark.parametrize("kind", ["id", "family", "prompt", "digit_blind"])
def test_cross_split_overlap(kind):
    a, b = row(0), row(1, "dev")
    if kind == "id": b["id"] = a["id"]
    if kind == "family": b["template_family"] = a["template_family"]
    if kind == "prompt": b["messages"][0]["content"] = a["messages"][0]["content"].upper()
    if kind == "digit_blind":
        a["messages"][0]["content"] = "How many 17?"
        b["messages"][0]["content"] = "How many 24?"
    with pytest.raises(ValueError): overlap_audit({"train": [a], "dev": [b]})


@pytest.mark.parametrize("kind", ["roles", "delimiter", "blank", "family", "duplicate"])
def test_raw_schema(tmp_path, kind):
    r = row(0)
    if kind == "roles": r["messages"].append({"role": "user", "content": "another"})
    if kind == "delimiter": r["messages"][0]["content"] = "text <|im_end|>"
    if kind == "blank": r["messages"][-1]["content"] = " "
    if kind == "family": r.pop("template_family")
    text = json.dumps(r) + "\n"
    if kind == "duplicate": text *= 2
    p = tmp_path / "train.jsonl"; p.write_text(text)
    with pytest.raises(ValueError): read_rows(p)


def test_plan_determinism_order_vs_selection():
    a, b = synthetic_plan(), synthetic_plan(order_seed=29)
    assert a == synthetic_plan()
    assert {r["id"] for r in a["train"]} == {r["id"] for r in b["train"]}
    assert [r["id"] for r in a["train"]] != [r["id"] for r in b["train"]]
    assert sum(len(r["ids"]) for r in a["train"]) == a["processed_tokens"]
    for arm, (objective, _) in ARMS.items():
        p = expected_progress(validate_plan(a), objective, len(a["groups"]))
        assert p["processed_tokens"] == a["processed_tokens"]
    assert expected_progress(validate_plan(a), "whole", len(a["groups"]))["supervised_tokens"] > \
        expected_progress(validate_plan(a), "response", len(a["groups"]))["supervised_tokens"]


@pytest.mark.parametrize("kind", ["hash", "budget", "group", "mask", "ids"])
def test_invalid_plan(kind):
    v = synthetic_plan()
    if kind == "hash": v["sha256"] = "0"
    else:
        v = I.unseal(v)
        if kind == "budget": v["processed_tokens"] += 1
        if kind == "group": v["groups"] = [[0]]
        if kind == "mask": v["train"][0]["answer_start"] = 0
        if kind == "ids": v["train"][0]["ids"][1] = -1
        v = I.seal(v)
    with pytest.raises(ValueError): validate_plan(v)


def test_pending_review_never_authorizes():
    p = synthetic_plan(); review = review_template(p)
    with pytest.raises(ValueError): validate_review(review, p["sha256"])
    for k in review:
        if type(review[k]) is bool: review[k] = True
    review.update(decision="APPROVED", reviewer="fixture-only", rationale="not native evidence")
    validate_review(review, p["sha256"])
    with pytest.raises(ValueError): validate_review(review, "other-plan")


@pytest.mark.parametrize("objective", ["whole", "response"])
@pytest.mark.parametrize("chunk", [1, 3, 64])
def test_chunk_value_and_gradient_equivalence(objective, chunk):
    r = encode_row(row(0), ByteTokenizer(), Design())
    n = len(r["ids"])
    labels = torch.tensor([targets(r, objective)])
    h = torch.randn(1, n, 8, requires_grad=True)
    w = torch.randn(266, 8, requires_grad=True)
    a, count = chunked_sum(h, w, labels, chunk)
    a.backward(); gh, gw = h.grad.clone(), w.grad.clone()
    h.grad = w.grad = None
    b = F.cross_entropy(F.linear(h[:, :-1], w).float().reshape(-1, 266),
                        labels[:, 1:].reshape(-1), ignore_index=-100, reduction="sum")
    b.backward()
    torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-6)
    torch.testing.assert_close(gh, h.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(gw, w.grad, atol=1e-5, rtol=1e-5)
    if objective == "response":
        assert not gh[:, :r["answer_start"] - 1].any()
        assert gh[:, r["answer_end"] - 1].abs().sum() > 0
        assert not gh[:, r["answer_end"]:].any()


def test_empty_labels_fail():
    with pytest.raises(ValueError): chunked_sum(torch.randn(1, 3, 8), torch.randn(266, 8),
                                              torch.full((1, 3), -100), 2)


@pytest.mark.parametrize("decay", [0.0, 0.02])
@pytest.mark.parametrize("beta2", [0.95, 0.999])
def test_optimizer_matches_torch_adamw(decay, beta2):
    d = replace(Design(), weight_decay=decay, beta2=beta2)
    a = torch.nn.Parameter(torch.randn(5, 7)); b = torch.nn.Parameter(a.detach().clone())
    ours = CpuAdamW([("p", a)], d)
    theirs = torch.optim.AdamW([b], lr=1e-3, betas=(d.beta1, d.beta2),
                               eps=d.epsilon, weight_decay=decay, foreach=False)
    for step in range(5):
        g = torch.randn_like(a)
        a.grad = g.clone(); b.grad = g.clone()
        lr = 1e-3 / (step + 1)
        theirs.param_groups[0]["lr"] = lr
        ours.step(lr); theirs.step()
        torch.testing.assert_close(a, b, atol=2e-7, rtol=2e-7)
        torch.testing.assert_close(ours.state["p"][0], theirs.state[b]["exp_avg"], atol=0, rtol=0)
        torch.testing.assert_close(ours.state["p"][1], theirs.state[b]["exp_avg_sq"], atol=0, rtol=0)


def test_invalid_gradient_prevalidation():
    a, b = [torch.nn.Parameter(torch.ones(2)) for _ in range(2)]
    opt = CpuAdamW([("a", a), ("b", b)], Design())
    a.grad = torch.ones_like(a); b.grad = torch.full_like(b, float("nan"))
    with pytest.raises(ValueError): opt.step(1e-3)
    assert torch.equal(a, torch.ones(2)) and opt.step_count == 0


@pytest.mark.parametrize("arm", sorted(ARMS))
def test_six_arm_synthetic_execution(tmp_path, arm):
    p = synthetic_plan()
    m = TinyLM()
    result = train_cell(m, p, arm, tmp_path, {"scope": "synthetic", "arm": arm})
    assert result["status"] == "ENGINEERING_COMPLETE_QUALITY_UNDECIDED"
    assert result["synthetic"] and not result["release_permission"]
    assert result["progress"]["processed_tokens"] == p["processed_tokens"]
    assert I.read_json(tmp_path / "final_dev.json")["response_targets"] == 6


@pytest.mark.parametrize("arm", ["whole_1e-5", "response_1e-5"])
def test_interrupted_replay_exact(tmp_path, arm):
    plan = synthetic_plan(); source = TinyLM()
    a, b, c = copy.deepcopy(source), copy.deepcopy(source), copy.deepcopy(source)
    full, part, resumed = [I.new_dir(tmp_path / n) for n in ("full", "part", "resumed")]
    binding = {"scope": "synthetic", "plan": plan["sha256"], "arm": arm}
    train_cell(a, plan, arm, full, binding)
    r = train_cell(b, plan, arm, part, binding, stop_after_updates=1)
    final = train_cell(c, plan, arm, resumed, binding, resume=part / r["checkpoint"])
    assert parameter_digest(a) == parameter_digest(c)
    assert I.read_json(full / "final_dev.json") == I.read_json(resumed / "final_dev.json")
    assert final["progress"] == I.read_json(full / "result.json")["progress"]


@pytest.mark.parametrize("kind", ["binding", "hash", "shape", "moment", "extra", "path"])
def test_checkpoint_tamper_is_atomic(tmp_path, kind):
    from safetensors.torch import load_file, save_file
    p = torch.nn.Parameter(torch.randn(4)); opt = CpuAdamW([("p", p)], Design())
    binding = {"scope": "synthetic"}; directory = tmp_path / "c"
    save_checkpoint(directory, opt, binding, {"updates": 0})
    manifest = I.unseal(I.read_json(directory / "manifest.json"))
    file = directory / "tensor-0000.safetensors"
    if kind == "binding": binding = {"scope": "other"}
    if kind == "hash": file.write_bytes(b"corrupt")
    if kind in ("shape", "moment"):
        # Copy out of the mapping before writing back to the same path. `load_file`
        # memory-maps the file, and Windows refuses to write a file that still has a
        # user-mapped section open (os error 1224). Cloning detaches the tensors so the
        # mapping can be released. This surfaces only under the package's own pinned
        # safetensors 0.7.0; a newer one happened to hide it.
        loaded = load_file(str(file))
        state = {name: tensor.clone() for name, tensor in loaded.items()}
        del loaded
        gc.collect()
        if kind == "shape": state["parameter"] = torch.zeros(1)
        else: state["exp_avg_sq"].fill_(-1)
        save_file(state, str(file)); manifest["entries"][0]["sha256"] = I.file_hash(file)
    if kind == "extra": (directory / "extra").write_text("x")
    if kind == "path": manifest["entries"][0]["file"] = "../escape"
    (directory / "manifest.json").write_bytes(I.canonical(I.seal(manifest)))
    before = p.detach().clone()
    with pytest.raises(ValueError): restore_checkpoint(directory, opt, binding)
    assert torch.equal(p, before) and opt.step_count == 0


def test_checkpoint_refuses_pending_gradients(tmp_path):
    p = torch.nn.Parameter(torch.ones(2)); opt = CpuAdamW([("p", p)], Design()); p.grad = torch.ones(2)
    with pytest.raises(ValueError): save_checkpoint(tmp_path / "bad", opt, {}, {"updates": 0})


def test_resource_lock_never_steals(tmp_path):
    p = tmp_path / "gpu.lock"
    with lease(p):
        with pytest.raises(FileExistsError):
            with lease(p): pass
    assert not p.exists()


def test_watchdog_preserves_failed_attempt(tmp_path):
    directory = tmp_path / "timeout"
    code = bounded_child([sys.executable, "-c", "import time;time.sleep(10)"], directory, 1)
    assert code == 2
    assert I.read_json(directory / "supervisor.json")["status"] == "TIMEOUT_OR_INTERRUPTED"
    with pytest.raises(FileExistsError): bounded_child([sys.executable, "-c", "pass"], directory, 1)


@pytest.mark.parametrize("text,ok", [("7", True), ("7.0", True), ("The answer is 7", False),
                                    ("17", False), ("7 or 8", False), ("NaN", False)])
def test_numeric_contract_no_substring(text, ok):
    assert classify(text, {"scorer": "number", "answers": ["7"], "answerable": True})["correct"] is ok


def test_unknown_and_semantics_not_conflated():
    spec = {"scorer": "exact", "answers": ["UNKNOWN"], "answerable": False}
    assert classify("UNKNOWN", spec)["correct"]
    assert not classify("I don't know", spec)["literal_unknown"]
    assert not classify("UNKNOWN or 7", spec)["correct"]
    assert classify("arbitrary prose", None)["correct"] is None


def test_dev_report_and_tamper(tmp_path):
    from experiments.model_quality_gpu_v1.analyze import summarize
    p = synthetic_plan(); m = TinyLM()
    train_cell(m, p, "response_1e-5", tmp_path, {"arm": "response_1e-5"})
    assert summarize(tmp_path)["metrics"]["scored"] == 0  # no synthetic native generation claim
    (tmp_path / "final_dev.json").write_text("{}")
    with pytest.raises(ValueError): summarize(tmp_path)


@pytest.mark.parametrize("mutation", ["status", "binding", "steps", "checks", "gradient", "hash", "nan", "tokens"])
def test_qualification_receipt_fails_closed(mutation):
    from experiments.model_quality_gpu_v1.run import CHECKS, verify_qualification
    body = {"schema": 1, "phase": "NATIVE_GPU_QUALIFICATION", "status": "PASS", "binding": {"fixture": True},
            "checks": {k: True for k in CHECKS}, "optimizer_steps": 0, "quality_permission": False,
            "release_permission": False, "metrics": {"trainable_parameters": 314618880,
                "trainable_tensors": 220, "finite_gradient_tensors": 220, "probe_tokens": 32,
                "cache_max_abs": 0.0, "source_state_before": "a"*64, "source_state_after": "a"*64}}
    verify_qualification(I.seal(body), {"fixture": True})
    if mutation == "status": body["status"] = "SYNTHETIC_PASS"
    if mutation == "binding": body["binding"] = {}
    if mutation == "steps": body["optimizer_steps"] = 1
    if mutation == "checks": body["checks"].pop(next(iter(CHECKS)))
    if mutation == "gradient": body["metrics"]["finite_gradient_tensors"] = 219
    if mutation == "hash": body["metrics"]["source_state_before"] = body["metrics"]["source_state_after"] = None
    if mutation == "nan": body["metrics"]["cache_max_abs"] = -1.0
    if mutation == "tokens": body["metrics"]["probe_tokens"] = 2000
    with pytest.raises(ValueError): verify_qualification(I.seal(body), {"fixture": True})


def test_weighted_update_not_mean_of_row_means():
    # Loss contribution of each row is weighted by the group's supervised token count.
    w = torch.randn(11, 4, requires_grad=True)
    ha = torch.randn(1, 3, 4, requires_grad=True); hb = torch.randn(1, 7, 4, requires_grad=True)
    ya = torch.tensor([[-100, 1, 2]]); yb = torch.tensor([[-100, 2, 3, 4, 5, 6, 7]])
    la, na = chunked_sum(ha, w, ya, 2); lb, nb = chunked_sum(hb, w, yb, 2)
    weighted = (la + lb) / (na + nb)
    native = F.cross_entropy(torch.cat((F.linear(ha[:, :-1], w).reshape(-1, 11),
                                        F.linear(hb[:, :-1], w).reshape(-1, 11))),
                             torch.cat((ya[:, 1:].flatten(), yb[:, 1:].flatten())))
    torch.testing.assert_close(weighted, native)
    assert not torch.isclose(weighted, (la / na + lb / nb) / 2)


def test_checkpoint_rng_replays(tmp_path):
    import random
    from experiments.model_quality_gpu_v1.checkpoints import restore_checkpoint
    p = torch.nn.Parameter(torch.randn(3)); opt = CpuAdamW([("p", p)], Design())
    random.seed(19); torch.manual_seed(19)
    save_checkpoint(tmp_path / "rng", opt, {"test": True}, {"updates": 0})
    python_expected, torch_expected = random.random(), torch.rand(4)
    random.seed(30); torch.manual_seed(30)
    restore_checkpoint(tmp_path / "rng", opt, {"test": True})
    assert random.random() == python_expected
    assert torch.equal(torch.rand(4), torch_expected)


def test_unicode_template_boundary():
    r = row(0); r["messages"][0]["content"] = "東京: reply with café."
    r["messages"][-1]["content"] = "café"
    tok = ByteTokenizer(); encoded = encode_row(r, tok, Design())
    assert tok.decode(encoded["ids"][encoded["answer_start"]:encoded["answer_end"]]) == "café"
    assert targets(encoded, "response")[encoded["answer_end"]] == tok.eos_token_id


def test_final_partial_update_is_present(tmp_path):
    plan = synthetic_plan(); p = validate_plan(plan)
    assert sum(len(p["train"][i]["ids"]) for i in p["groups"][-1]) < p["design"]["update_tokens"]
    result = train_cell(TinyLM(), plan, "response_1e-5", tmp_path, {"scope": "fixture"})
    steps = [I.decode(line) for line in (tmp_path / "steps.jsonl").read_text().splitlines()]
    assert len(steps) == len(p["groups"]) == result["progress"]["updates"]
    assert steps[-1]["processed_tokens"] == p["processed_tokens"]
    assert steps[-1]["lr"] == pytest.approx(1e-6)
