"""Tiny CPU implementation fixtures only; no native Rineya binding or bank grant."""
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from experiments.rineya_quality_sft_v1 import EXPERIMENT
from experiments.rineya_quality_sft_v1 import compiler, engine, native, integrity
from experiments.rineya_quality_sft_v1.contract import Design


class Decoder(nn.Module):
    def __init__(self, embed):
        super().__init__()
        self.embed_tokens = embed
        self.layers = nn.ModuleList([nn.Linear(4, 4, bias=False) for _ in range(28)])
        self.norm = nn.LayerNorm(4)
        self.register_buffer("constant", torch.tensor([1.]))
    def forward(self, input_ids, **kwargs):
        x = self.embed_tokens(input_ids)
        # One observed math-SDPA operation; small dimensions, not native geometry.
        q = x.mean(-1, keepdim=True).unsqueeze(1)
        attention = F.scaled_dot_product_attention(q, q, q, is_causal=True).squeeze(1)
        x = x + .01 * attention
        for layer in self.layers:
            x = x + .01 * layer(x)
        return SimpleNamespace(last_hidden_state=self.norm(x))


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        embed = nn.Embedding(16, 4)
        self.model = Decoder(embed)
        self.lm_head = nn.Linear(4, 16, bias=False)
        self.lm_head.weight = embed.weight
        self.config = SimpleNamespace(vocab_size=16, use_cache=False)
    def gradient_checkpointing_enable(self, **kwargs):
        self.checkpointing = kwargs
    def forward(self, input_ids, use_cache=False, past_key_values=None, **kwargs):
        combined = torch.cat((past_key_values, input_ids), -1) if past_key_values is not None else input_ids
        hidden = self.model(input_ids=combined).last_hidden_state
        logits = self.lm_head(hidden)
        if past_key_values is not None:
            logits = logits[:, -input_ids.shape[-1]:]
        return SimpleNamespace(logits=logits, past_key_values=combined.detach().clone() if use_cache else None)


class Tokenizer:
    eos_token_id = 15
    chat_template = "CPU fixture template only"
    def decode(self, ids, **kwargs):
        return ":".join(str(value) for value in ids if value != self.eos_token_id)


def fixture(monkeypatch):
    torch.manual_seed(17)
    model = TinyModel().eval()
    rows = [{"id": f"fixture-{index}", "ids": [1, 2, 3 + index, 4, 15, 5],
             "prompt_ids": [1, 2, 3 + index], "answer_start": 3, "answer_end": 4,
             "reference": "CPU fixture", "parent_id": str(index)} for index in range(3)]
    body = {"experiment": EXPERIMENT, "kind": "fixture", "design": Design().as_dict(),
            "train": rows, "dev": rows, "groups": [[0], [1], [2]], "processed_tokens": 18,
            "tokenizer": {"eos_token_id": 15, "chat_template_sha256": integrity.digest(Tokenizer.chat_template)}}
    sealed = integrity.seal(body)
    monkeypatch.setattr(compiler, "validate_plan", lambda value: integrity.unseal(value))
    binding = {"experiment": EXPERIMENT, "compiled_plan_sha256": sealed["sha256"], "fixture_only": True,
               "initial_parameters_sha256": native.parameter_digest(model),
               "initial_frozen_state_sha256": native.freeze_state_digest(model)}
    return model, sealed, binding


def test_zero_update_both_losses_maximum_shapes_cache_and_official_recovery(tmp_path, monkeypatch):
    model, plan, binding = fixture(monkeypatch)
    monkeypatch.setattr(native.CpuAdamW, "step", lambda *a, **k: pytest.fail("Qualification cannot step"))
    before, frozen = native.parameter_digest(model), native.freeze_state_digest(model)
    directory = tmp_path / "qualification"
    directory.mkdir()
    report = native.qualify(model, Tokenizer(), plan, binding, directory=directory)
    body = native.verify_qualification(report, binding)
    assert body["status"] == "FIXTURE_QUALIFIED" and body["training_updates"] == 0
    assert body["optimizer_step_calls"] == 0 and body["optimizer_objects_created"] == 1
    assert body["checks"]["checkpoint_recovery_zero_update"] is True
    assert body["checkpoint_recovery"]["sentinel_restore_verified"] is True
    assert body["checks"]["maximum512_both_backward"] is True
    assert body["checks"]["maximum2048_eval_forward"] is True
    assert {entry["objective"] for entry in body["diagnostics"]} == {"whole", "response"}
    assert native.parameter_digest(model) == before and native.freeze_state_digest(model) == frozen
    assert all(p.grad is None for p in model.parameters())
    assert (directory / "recovery-step0" / "manifest.json").is_file()
    foreign = integrity.unseal(report)
    foreign["experiment"] = "MQ-001"
    with pytest.raises(ValueError, match="foreign"):
        native.verify_qualification(integrity.seal(foreign), binding)
    native_binding = dict(binding, fixture_only=False)
    with pytest.raises(ValueError):
        native.verify_qualification(report, native_binding)
    for key in ("diagnostics", "backend_events", "cache_check", "checkpoint_recovery"):
        missing = integrity.unseal(report)
        missing[key] = [] if isinstance(missing[key], list) else {}
        with pytest.raises(ValueError):
            native.verify_qualification(integrity.seal(missing), binding)


def test_noop_checkpoint_restore_cannot_pass_and_cleanup_preserves_source(tmp_path, monkeypatch):
    model, plan, binding = fixture(monkeypatch)
    named = engine.select_trainables(model, integrity.unseal(plan))
    before, buffers, rng = native.parameter_digest(model), native.buffer_digest(model), torch.get_rng_state().clone()
    monkeypatch.setattr(engine, "restore_checkpoint", lambda *a, **kwargs:
        {"updates": 0, "next_group": 0, "consumed_rows": 0, "consumed_row_ids": [],
         "processed_tokens": 0, "supervised_tokens": 0})
    monkeypatch.setattr(engine.CpuAdamW, "step", lambda *a, **kwargs: pytest.fail("Sentinels cannot call optimizer.step"))
    with pytest.raises(RuntimeError, match="sentinel recovery failed"):
        engine.checkpoint_roundtrip_zero(model, named, Design(), tmp_path / "noop-recovery", binding)
    assert native.parameter_digest(model) == before and native.buffer_digest(model) == buffers
    assert torch.equal(torch.get_rng_state(), rng)


def test_first_entry_only_restore_cannot_certify_full_recovery(tmp_path, monkeypatch):
    model, plan, binding = fixture(monkeypatch)
    named = engine.select_trainables(model, integrity.unseal(plan))
    before = native.parameter_digest(model)
    def partial(directory, optimizer, scope):
        meta = engine.validate_checkpoint(directory, optimizer.named, scope)
        entry, (name, parameter) = meta["entries"][0], optimizer.named[0]
        state = engine.load_file(str(Path(directory) / entry["file"]), device="cpu")
        with torch.no_grad():
            parameter.copy_(state["parameter"])
            optimizer.state[name][0].copy_(state["exp_avg"])
            optimizer.state[name][1].copy_(state["exp_avg_sq"])
        optimizer.step_count = 0
        engine._restore_rng(meta["rng"], True)
        return meta["progress"]
    monkeypatch.setattr(engine, "restore_checkpoint", partial)
    with pytest.raises(RuntimeError, match="sentinel recovery failed"):
        engine.checkpoint_roundtrip_zero(model, named, Design(), tmp_path / "partial-recovery", binding)
    assert native.parameter_digest(model) == before


@pytest.mark.parametrize("arm", ["whole_1e-5", "response_1e-5"])
def test_training_consumes_each_row_once_with_final_only_checkpoint_and_no_dev(tmp_path, monkeypatch, arm):
    model, plan, binding = fixture(monkeypatch)
    frozen = native.freeze_state_digest(model)
    directory = tmp_path / arm
    directory.mkdir()
    stages = []
    result = integrity.unseal(engine.train_cell(model, plan, arm, directory, binding,
        guard=lambda **kwargs: stages.append(kwargs) or {"fixture": True}))
    assert result["status"] == "TRAINING_COMPLETE_DEV_NOT_EVALUATED" and result["dev_evaluated"] is False
    assert result["progress"]["consumed_rows"] == 3 and result["progress"]["processed_tokens"] == 18
    assert result["progress"]["consumed_row_ids"] == [f"fixture-{i}" for i in range(3)]
    assert len(result["trace"]) == 3
    assert all(entry["update_l2"] > 0 and entry["preclip_gradient_l2"] >= 0 for entry in result["trace"])
    assert native.freeze_state_digest(model) == frozen
    assert sum(s.get("input_tokens", 0) for s in stages) == 18
    assert sum(s.get("forward_calls", 0) for s in stages) == 3
    assert not (directory / "baseline.json").exists() and not (directory / "final_dev.json").exists()
    assert (directory / "checkpoint-00003" / "manifest.json").is_file()
    state_binding = dict(binding, arm=arm, objective=arm.split("_")[0])
    metadata = engine.validate_checkpoint(directory / "checkpoint-00003", engine.select_trainables(model, integrity.unseal(plan)), state_binding)
    assert metadata["experiment"] == EXPERIMENT and metadata["progress"]["updates"] == 3
    assert all(p.grad is None for p in model.parameters())


def test_partial_failure_keeps_raw_rows_and_requires_new_attempt(tmp_path, monkeypatch):
    model, plan, binding = fixture(monkeypatch)
    directory = tmp_path / "failed"
    directory.mkdir()
    def guard(**values):
        if values["stage"] == "optimizer_step_after":
            raise RuntimeError("fixture sampled memory floor")
        return values
    with pytest.raises(RuntimeError, match="fixture sampled memory floor"):
        engine.train_cell(model, plan, "response_1e-5", directory, binding, guard=guard)
    failure = integrity.unseal(integrity.read_json(directory / "ENGINE_FAILED.json"))
    assert failure["automatic_retry"] is False and failure["uncheckpointed_model_invalid"] is True
    assert "row_backward_completed" in (directory / "raw-steps.jsonl").read_text()
    assert all(p.grad is None for p in model.parameters())
    assert not (directory / "train-result.json").exists()


def test_checkpoint_corruption_refused_before_restore_changes_parameters(tmp_path, monkeypatch):
    model, plan, binding = fixture(monkeypatch)
    named = engine.select_trainables(model, integrity.unseal(plan))
    optimizer = engine.CpuAdamW(named, Design())
    progress = engine.expected_progress(integrity.unseal(plan), "response", 0)
    path = tmp_path / "checkpoint"
    engine.save_checkpoint(path, optimizer, binding, progress)
    before = native.parameter_digest(model)
    with (path / "tensor-0000.safetensors").open("ab") as stream:
        stream.write(b"corrupt fixture")
    with pytest.raises(ValueError, match="hash"):
        engine.restore_checkpoint(path, optimizer, binding)
    assert native.parameter_digest(model) == before


def test_measured_optimizer_matches_audited_equations_and_state_exactly():
    torch.manual_seed(71)
    left = [("a", nn.Parameter(torch.randn(3, 4))), ("b", nn.Parameter(torch.randn(5)))]
    right = [(name, nn.Parameter(parameter.detach().clone())) for name, parameter in left]
    audited, measured = engine.CpuAdamW(left, Design()), engine.MeasuredCpuAdamW(right, Design())
    for lr in (1e-5, 5e-6, 1e-6):
        before = {name: parameter.detach().clone() for name, parameter in right}
        for (_, original), (_, candidate) in zip(left, right, strict=True):
            gradient = torch.randn_like(original)
            original.grad, candidate.grad = gradient.clone(), gradient.clone()
        audited.step(lr)
        update = measured.step(lr)
        expected = sum(float((parameter.detach().double() - before[name].double()).square().sum())
                       for name, parameter in right) ** .5
        assert update == pytest.approx(expected, rel=1e-15, abs=1e-18)
        for (name, original), (_, candidate) in zip(left, right, strict=True):
            assert torch.equal(original, candidate)
            assert all(torch.equal(a, b) for a, b in zip(audited.state[name], measured.state[name], strict=True))
        assert audited.step_count == measured.step_count


def test_generation_uses_only_frozen_prefix_and_retains_termination_accounting(monkeypatch):
    model, plan, binding = fixture(monkeypatch)
    row = copy.deepcopy(integrity.unseal(plan)["dev"][0])
    original = native.generate(model, row, tokenizer=Tokenizer(), eos_ids=[15])
    row.update(reference="NEVER_INPUT", answer="NEVER_INPUT", ids=[10, 11, 15], answer_start=0)
    changed = native.generate(model, row, tokenizer=Tokenizer(), eos_ids=[15])
    assert original == changed
    assert original["stop_reason"] in {"eos", "length_limit"}
    assert original["response_tokens"] == len(original["response_ids"])
    assert original["cost"]["forward_calls"] == original["response_tokens"]
    assert original["prompt_ids"] == row["prompt_ids"]


def test_generation_failure_preserves_partial_tokens_prediction_and_attempted_cost(monkeypatch):
    model, plan, binding = fixture(monkeypatch)
    row = integrity.unseal(plan)["dev"][0]
    with torch.no_grad(), native.math_attention():
        first_token = int(model(input_ids=torch.tensor([row["prompt_ids"]])).logits[0, -1].argmax())
    count = 0
    def guard(**values):
        nonlocal count
        if values["stage"] == "generation_token_before":
            count += 1
            if count == 2:
                raise RuntimeError("fixture generation floor")
    with pytest.raises(RuntimeError, match="fixture generation floor") as caught:
        native.generate(model, row, tokenizer=Tokenizer(), eos_ids=[(first_token + 1) % 16], guard=guard)
    assert len(caught.value.partial_ids) == 1
    assert caught.value.partial_prediction == Tokenizer().decode(caught.value.partial_ids)
    assert caught.value.partial_cost["forward_calls"] == 1
