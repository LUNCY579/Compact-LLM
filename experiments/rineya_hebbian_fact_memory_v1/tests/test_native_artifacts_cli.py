import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn
from safetensors.torch import save_file
from experiments.rineya_hebbian_fact_memory_v1.artifacts import save_memory, load_memory, json_write, sha256
from experiments.rineya_hebbian_fact_memory_v1.memory import BuildConfig, FrozenBilinear, build
from experiments.rineya_hebbian_fact_memory_v1.native import (
    MemoryBlend, intervene, fingerprint, validate_rows, capture, screen, verify_checkpoint,
)
from experiments.rineya_hebbian_fact_memory_v1.cli import main, read_json

torch.set_num_threads(1)


class MockLM(nn.Module):
    """Toy interface test ONLY. Recomputes its mock cache, not a native Qwen run."""
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=11)
        self.embed = nn.Embedding(11, 4)
        self.model = nn.Module()
        block = nn.Module()
        block.mlp = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 4))
        self.model.layers = nn.ModuleList([block])
        self.head = nn.Linear(4, 11)
        self.register_buffer("nonpersistent", torch.ones(2), persistent=False)

    def forward(self, input_ids, use_cache=False, past_key_values=None):
        ids = input_ids if past_key_values is None else torch.cat([past_key_values, input_ids], 1)
        hidden = self.embed(ids)
        hidden = hidden + self.model.layers[0].mlp(hidden)
        logits = self.head(hidden)
        return SimpleNamespace(logits=logits[:, -input_ids.shape[1]:],
                               past_key_values=ids if use_cache else None)


PATH = "model.layers.0.mlp"
ROWS = [{"id": f"r{i}", "split": "train" if i < 4 else "check", "input_ids": [0, i+1]} for i in range(6)]


def toy_memory():
    return build(torch.eye(4), torch.eye(4), BuildConfig(width=8))[0]


@pytest.mark.parametrize("alpha", [-.1, 1.1, float("nan"), float("inf"), True])
def test_invalid_blend(alpha):
    with pytest.raises(ValueError):
        MemoryBlend(nn.Identity(), toy_memory(), alpha)


def test_zero_bypass_skips_memory_and_restores_identity():
    model = MockLM().eval()
    original = model.model.layers[0].mlp
    memory = toy_memory()
    tokens = torch.tensor([[0, 1]])
    baseline = model(tokens).logits
    before = fingerprint(model)
    with intervene(model, PATH, memory, 0) as wrapper:
        with patch.object(wrapper.memory, "forward", side_effect=AssertionError("must not execute")):
            assert torch.equal(model(tokens).logits, baseline)
    assert model.model.layers[0].mlp is original and fingerprint(model) == before


def test_exception_and_nested_patch_restore():
    model = MockLM().eval()
    original = model.model.layers[0].mlp
    with pytest.raises(RuntimeError):
        with intervene(model, PATH, toy_memory(), .1):
            with pytest.raises(ValueError, match="nested"):
                with intervene(model, PATH, toy_memory(), .1):
                    pass
            raise RuntimeError("injected failure")
    assert model.model.layers[0].mlp is original
    with pytest.raises(ValueError):
        with intervene(model, "mlp", toy_memory(), .1):
            pass


def test_capture_last_position_and_hook_cleanup():
    model = MockLM().eval()
    keys, targets = capture(model, PATH, ROWS)
    assert keys.shape == targets.shape == (6, 4)
    torch.testing.assert_close(keys[0], model.embed(torch.tensor([1]))[0].double())
    assert not model.model.layers[0].mlp._forward_hooks
    with patch.object(model, "forward", side_effect=RuntimeError("fault")):
        with pytest.raises(RuntimeError):
            capture(model, PATH, ROWS)
    assert not model.model.layers[0].mlp._forward_hooks


@pytest.mark.parametrize("fault", ["duplicate_id", "duplicate_prompt", "token_bool", "token_negative", "bad_split", "unknown_field", "missing_check", "empty_tokens", "length"])
def test_rows_fail_closed(fault):
    rows = copy.deepcopy(ROWS)
    if fault == "duplicate_id": rows[1]["id"] = rows[0]["id"]
    if fault == "duplicate_prompt": rows[4]["input_ids"] = rows[0]["input_ids"]
    if fault == "token_bool": rows[0]["input_ids"] = [True]
    if fault == "token_negative": rows[0]["input_ids"] = [-1]
    if fault == "bad_split": rows[0]["split"] = "test"
    if fault == "unknown_field": rows[0]["answer"] = "leak"
    if fault == "missing_check": rows[4]["split"] = "train"
    if fault == "empty_tokens": rows[0]["input_ids"] = []
    if fault == "length": rows[0]["input_ids"] = [0]*257
    with pytest.raises(ValueError):
        validate_rows(rows, 11)


def test_mock_screen_is_reversible_and_does_not_claim_fact_efficacy():
    with torch.random.fork_rng():
        torch.manual_seed(77)
        model = MockLM().eval()
    before = fingerprint(model)
    _, result = screen(model, ROWS, PATH, BuildConfig(width=8), alpha=.0001)
    assert result["zero_alpha_exact_parity"] and result["restoration_verified"]
    assert result["baseline_cache"]["pass"] and result["candidate_cache"]["pass"]
    assert result["screen_pass"] and not result["native_qualified"]
    assert result["kind"].endswith("NOT_FACT_EFFICACY")
    assert fingerprint(model) == before
    model.train()
    with pytest.raises(ValueError, match="eval"):
        screen(model, ROWS, PATH, BuildConfig(width=8), alpha=.1)


def test_nonpersistent_buffers_are_fingerprinted():
    model = MockLM().eval()
    before = fingerprint(model)
    model.nonpersistent += 1
    assert fingerprint(model) != before


def test_artifact_roundtrip_and_create_only(tmp_path):
    model = toy_memory()
    root = tmp_path/"memory"
    manifest = save_memory(root, model, {"fixture": True})
    restored, loaded = load_memory(root)
    assert manifest == loaded
    for x, y in zip(model.buffers(), restored.buffers()):
        assert torch.equal(x, y)
    with pytest.raises(FileExistsError):
        save_memory(root, model, {})
    with pytest.raises(FileExistsError):
        json_write(root/"manifest.json", {})


@pytest.mark.parametrize("tamper", ["bytes", "shape", "extra", "schema", "budget"])
def test_artifact_tamper_detection(tmp_path, tamper):
    root = tmp_path/"memory"
    save_memory(root, toy_memory(), {})
    p = root/"manifest.json"
    if tamper == "bytes":
        with (root/"memory.safetensors").open("ab") as f: f.write(b"bad")
    elif tamper == "extra": (root/"extra").write_text("x")
    elif tamper == "shape":
        manifest = json.loads(p.read_text()); manifest["shapes"]["a"] = [1, 2]
        p.write_text(json.dumps(manifest))
    elif tamper == "schema":
        manifest = json.loads(p.read_text()); manifest["schema"] = "other"
        p.write_text(json.dumps(manifest))
    with pytest.raises((ValueError, MemoryError)):
        load_memory(root, max_bytes=1 if tamper == "budget" else 1024**2)


def test_checkpoint_inventory_rejects_extra_and_pickle(tmp_path):
    p = tmp_path/"model"; p.mkdir()
    (p/"config.json").write_text('{}')
    save_file({"weight": torch.ones(2)}, str(p/"model.safetensors"))
    manifest = {f.name: sha256(f) for f in p.iterdir()}
    assert verify_checkpoint(p, manifest) == manifest
    (p/"unknown.json").write_text('{}')
    with pytest.raises(ValueError): verify_checkpoint(p, manifest)
    (p/"unknown.json").unlink()
    (p/"unsafe.bin").write_bytes(b"not executed")
    with pytest.raises(ValueError, match="pickle"): verify_checkpoint(p, manifest)


def test_cli_fit_edit_roundtrip(tmp_path):
    config = tmp_path/"config.json"; config.write_text('{"width":8}')
    data = tmp_path/"data.safetensors"
    save_file({"keys": torch.eye(4), "targets": torch.eye(4)}, str(data))
    original = tmp_path/"original"; edited = tmp_path/"edited"
    assert main(["fit", "--config", str(config), "--data", str(data), "--out", str(original)]) == 0
    assert main(["edit", "--config", str(config), "--data", str(data), "--memory", str(original), "--out", str(edited)]) == 0
    assert main(["fit", "--config", str(config), "--data", str(data), "--out", str(original)]) == 2
    assert load_memory(edited)[1]["receipt"]["full_retained_bank_rebuild"]


def test_cli_missing_native_authorization_cannot_load(tmp_path):
    with patch("experiments.rineya_hebbian_fact_memory_v1.native.load_qwen_local", side_effect=AssertionError("forbidden")):
        assert main(["native-screen", "--out", str(tmp_path/"run"), "--config", "missing",
                     "--model-dir", "missing", "--manifest", "missing", "--rows", "missing",
                     "--module", PATH, "--alpha", ".01"]) == 2
    assert not (tmp_path/"run").exists()


def test_nonfinite_json_rejected(tmp_path):
    p = tmp_path/"bad.json"; p.write_text('{"ridge":NaN}')
    with pytest.raises(ValueError): read_json(str(p))
    with pytest.raises(ValueError): json_write(tmp_path/"out.json", {"bad": float("nan")})
    assert not (tmp_path/"out.json").exists()
