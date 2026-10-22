import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from experiments.rineya_bf16_attention_gproj_v1.audit import metric_passes
from experiments.rineya_bf16_attention_gproj_v1.backends import SDPA
from experiments.rineya_bf16_attention_gproj_v1.capture import load_capture, save_capture, sdpa_session
from experiments.rineya_bf16_attention_gproj_v1.core import witness
from experiments.rineya_bf16_attention_gproj_v1.protocol import (
    checkpoint_manifest, digest_json, load_rows, parameter_hash, validate_config,
    verify_qualification, write_json,
)
from experiments.rineya_bf16_attention_gproj_v1.runner import matched_train, prepare_output, train_arm
from experiments.rineya_bf16_attention_gproj_v1.native import qualification_batches
from experiments.rineya_bf16_attention_gproj_v1.__main__ import main


def config():
    return {"checkpoint": "/not_loaded", "checkpoint_sha256": "a" * 64,
            "data": "/not_read", "data_sha256": "b" * 64, "device": "cpu", "backend": "math",
            "trainable_names": ["q_proj.weight"], "max_trainable_parameters": 1000,
            "max_tokens": 32, "max_steps": 2, "seeds": [17, 29], "learning_rate": .001,
            "clip_norm": 1.0, "min_free_cuda_gib": 4.0,
            "audit_relative_tolerance": .05, "audit_absolute_tolerance": 1e-6, "capture_call_indices": [1]}


def rows():
    return [{"id": str(i), "group_id": str(i), "split": split, "admitted_for": split,
             "purpose": purpose, "source": "synthetic_fixture", "input_ids": [1, i + 3, 2, 7],
             "labels": [-100, -100, 2, 7]}
            for i, (split, purpose) in enumerate([
                ("train", "acquisition"), ("train", "acquisition"),
                ("dev", "acquisition"), ("dev", "retention")])]


class TinyAttentionModel(nn.Module):
    """Mechanics fixture only; cannot pass the native 0.6B loader's size guard."""
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=16)
        self.emb = nn.Embedding(16, 16).bfloat16()
        self.q_proj = nn.Linear(16, 16, bias=False)  # FP32 master
        self.k_proj = nn.Linear(16, 8, bias=False).bfloat16()
        self.v_proj = nn.Linear(16, 8, bias=False).bfloat16()
        self.lm_head = nn.Linear(16, 16, bias=False).bfloat16()
        self.requires_grad_(False)
        self.q_proj.weight.requires_grad_(True)
        self.eval()

    def get_input_embeddings(self):
        return self.emb

    def forward(self, input_ids=None, inputs_embeds=None, labels=None, **kwargs):
        x = self.emb(input_ids) if inputs_embeds is None else inputs_embeds
        b, n, _ = x.shape
        q = self.q_proj(x).view(b, n, 2, 8).transpose(1, 2)
        k = self.k_proj(x).view(b, n, 1, 8).transpose(1, 2)
        v = self.v_proj(x).view(b, n, 1, 8).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
        logits = self.lm_head(y.transpose(1, 2).reshape(b, n, 16))
        loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, 16), labels[:, 1:].reshape(-1))
        return SimpleNamespace(loss=loss, logits=logits)


def test_capture_roundtrip_tamper_detection(tmp_path):
    path = tmp_path / "case.pt"
    save_capture(witness(), path)
    restored = load_capture(path)
    assert torch.equal(restored.q, witness().q)
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="hash"): load_capture(path)


def test_native_capture_preserves_forward_and_upstream_derivative():
    q = torch.randn(1, 2, 4, 8).bfloat16().requires_grad_()
    k, v = torch.randn_like(q), torch.randn_like(q)
    captures = []
    with sdpa_session("native", captures=captures) as state:
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        upstream = torch.randn_like(out)
        out.backward(upstream)
    assert state["calls"] == 1 and len(captures) == 1
    assert torch.equal(captures[0].saved_output, out)
    assert torch.equal(captures[0].u, upstream)
    assert torch.equal(captures[0].q, q)
    assert F.scaled_dot_product_attention is SDPA


@pytest.mark.parametrize("failure", ["dropout", "rectangular", "length", "bytes", "nested"])
def test_sessions_fail_closed_and_restore(failure):
    q = torch.ones(1, 1, 4, 8).bfloat16().requires_grad_()
    with pytest.raises((ValueError, RuntimeError)):
        with sdpa_session("native", captures=[], max_tokens=2 if failure == "length" else 8,
                          max_capture_bytes=1 if failure == "bytes" else 4096):
            if failure == "nested":
                with sdpa_session("native"): pass
            else:
                k = q[..., :2, :] if failure == "rectangular" else q
                F.scaled_dot_product_attention(q, k, k, dropout_p=.1 if failure == "dropout" else 0)
    assert F.scaled_dot_product_attention is SDPA


@pytest.mark.parametrize("mutation", ["sealed", "admission", "id", "group_overlap", "token_overlap",
                                      "length", "supervision", "negative", "cohort"])
def test_row_contract_rejects_bad_data(tmp_path, mutation):
    data = rows()
    if mutation == "sealed": data[0]["split"] = "sealed"
    if mutation == "admission": data[0]["admitted_for"] = "dev"
    if mutation == "id": data[1]["id"] = data[0]["id"]
    if mutation == "group_overlap": data[2]["group_id"] = data[0]["group_id"]
    if mutation == "token_overlap": data[2]["input_ids"] = data[0]["input_ids"]
    if mutation == "length": data[0]["input_ids"] *= 10
    if mutation == "supervision": data[0]["labels"] = [-100] * 4
    if mutation == "negative": data[0]["input_ids"][0] = -1
    if mutation == "cohort": data.pop()
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in data))
    with pytest.raises(ValueError): load_rows(path, 32)


def test_valid_rows_and_hashes(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows()))
    assert len(load_rows(path, 32)) == 4
    assert digest_json({"b": 1, "a": 2}) == digest_json({"a": 2, "b": 1})


@pytest.mark.parametrize("key,value", [("backend", "auto"), ("max_steps", 1000), ("learning_rate", float("nan")),
                                      ("checkpoint_sha256", "TODO"), ("seeds", [17, 17]),
                                      ("trainable_names", []), ("device", "automatic")])
def test_config_rejects_unpinned_or_unbounded(key, value):
    cfg = config()
    cfg[key] = value
    with pytest.raises(ValueError): validate_config(cfg)


def test_valid_configuration():
    validate_config(config())


def test_checkpoint_manifest_requires_real_local_files(tmp_path):
    with pytest.raises(ValueError): checkpoint_manifest(tmp_path)
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"fixture_not_loadable_weights")
    first = checkpoint_manifest(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"changed")
    assert first["sha256"] != checkpoint_manifest(tmp_path)["sha256"]


def test_nonoverwriting_outputs_and_source_protection(tmp_path):
    path = tmp_path / "record.json"
    write_json(path, {"a": 1})
    with pytest.raises(FileExistsError): write_json(path, {"a": 2})
    with pytest.raises(ValueError): prepare_output(tmp_path / "inside", str(tmp_path))


def test_training_cannot_start_without_explicit_opt_in(tmp_path):
    with pytest.raises(PermissionError): matched_train({}, {}, tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("mutation", ["status", "identity", "captures", "dispatch", "cohorts"])
def test_qualification_cannot_be_synthetic_stale_or_incomplete(mutation):
    report = {"status": "NATIVE_NUMERICAL_QUALIFIED", "identity": {"hash": "same"},
              "capture_count": 1, "backend_events": ["aten::_scaled_dot_product_attention_math"],
              "cohorts": ["acquisition", "retention"]}
    if mutation == "status": report["status"] = "SYNTHETIC_ONLY"
    if mutation == "identity": report["identity"] = {"hash": "other"}
    if mutation == "captures": report["capture_count"] = 0
    if mutation == "dispatch": report["backend_events"] = []
    if mutation == "cohorts": report["cohorts"] = ["acquisition"]
    with pytest.raises(ValueError): verify_qualification(report, {"hash": "same"})


@pytest.mark.parametrize("arm", ["native", "fp32", "cast_only", "gproj"])
def test_bounded_pilot_updates_only_selected_parameters(arm):
    torch.manual_seed(17)
    model = TinyAttentionModel()
    initial = parameter_hash(model.named_parameters())
    report, delta = train_arm(model, rows(), config(), arm, 17)
    assert report["frozen_parameters_unchanged"] and not report["capability_claim"]
    assert len(report["steps"]) == 2
    assert parameter_hash(model.named_parameters()) != initial
    assert list(delta) == ["q_proj.weight"]
    assert all(step["actual_update_l2"] > 0 for step in report["steps"])
    assert F.scaled_dot_product_attention is SDPA


def test_paired_seed_repeats_recipe_and_starting_metrics():
    torch.manual_seed(17)
    original = TinyAttentionModel()
    a, _ = train_arm(copy.deepcopy(original), rows(), config(), "gproj", 17)
    b, _ = train_arm(copy.deepcopy(original), rows(), config(), "gproj", 17)
    assert a["before"] == b["before"] and a["after"] == b["after"]
    assert [s["row_id"] for s in a["steps"]] == [s["row_id"] for s in b["steps"]]


def test_full_model_no_update_capture_harness_on_tiny_fixture():
    torch.manual_seed(17)
    model = TinyAttentionModel()
    before = parameter_hash(model.named_parameters())
    captures, details, events = qualification_batches(model, rows(), config())
    assert parameter_hash(model.named_parameters()) == before
    assert len(captures) == 2 and events
    assert len(details) == 2
    for record in details:
        assert record["losses"]["cast_only"] == record["losses"]["gproj"]


def test_gate_uses_absolute_error_when_reference_is_near_zero():
    assert metric_passes({"finite": True, "absolute_l2": 1e-8, "relative_l2": None}, .05, 1e-6)
    assert not metric_passes({"finite": True, "absolute_l2": 1, "relative_l2": None}, .05, 1e-6)


def test_cpu_cli_smoke(tmp_path):
    path = tmp_path / "synthetic.json"
    assert main(["synthetic", "--output", str(path)]) == 0
    assert json.loads(path.read_text())["status"] == "SYNTHETIC_ONLY"


def test_requested_late_call_capture_and_missing_site_failure():
    q = torch.ones(1, 1, 4, 8).bfloat16().requires_grad_()
    cases = []
    with sdpa_session("native", captures=cases, capture_call_indices=(2,)):
        first = F.scaled_dot_product_attention(q, q, q)
        second = F.scaled_dot_product_attention(first, q, q)
        second.sum().backward()
    assert len(cases) == 1 and cases[0].metadata["call_index"] == 2
    cfg = config()
    cfg["capture_call_indices"] = [2]
    with pytest.raises(RuntimeError, match="sites"):
        qualification_batches(TinyAttentionModel(), rows(), cfg)


def test_capture_sidecar_cannot_be_overwritten(tmp_path):
    path = tmp_path / "case.pt"
    path.with_suffix(".pt.json").write_text("earlier run")
    with pytest.raises(FileExistsError): save_capture(witness(), path)
    assert not path.exists()
