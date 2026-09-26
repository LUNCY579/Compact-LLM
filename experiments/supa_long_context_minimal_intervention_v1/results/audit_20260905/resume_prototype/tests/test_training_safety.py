"""CPU-only failure-path tests for matched-dose training and state restoration."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from slcmi.training import _batch, evaluate_loss, train


class TinyAdapter(torch.nn.Module):
    def __init__(self, dtype=torch.bfloat16, *, checkpointing_fails=False, validation_fails=False):
        super().__init__()
        self.frozen = torch.nn.Parameter(torch.ones(1, dtype=dtype), requires_grad=False)
        self.lora_A = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.ones_(self.lora_A.weight)
        self.config = SimpleNamespace(abstain_logit_bias=5.15625, use_cache=True)
        self.is_gradient_checkpointing = False
        self.checkpointing_fails = checkpointing_fails
        self.validation_fails = validation_fails
        self.observed_gates = []

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs):
        assert gradient_checkpointing_kwargs == {"use_reentrant": False}
        if self.checkpointing_fails:
            raise RuntimeError("unsupported checkpointing")
        self.is_gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.is_gradient_checkpointing = False

    def enable_input_require_grads(self):
        self._require_grads_hook = object()

    def disable_input_require_grads(self):
        del self._require_grads_hook

    def forward(self, input_ids, labels, **kwargs):
        self.observed_gates.append((torch.is_grad_enabled(), self.training, self.config.abstain_logit_bias))
        if torch.is_grad_enabled():
            assert self.config.abstain_logit_bias == 0
        else:
            assert self.config.abstain_logit_bias == 5.15625
            if self.validation_fails:
                raise RuntimeError("validation failure")
        value = input_ids[0, 0].item()
        multiplier = float("inf") if value < 0 else float(value)
        return SimpleNamespace(loss=self.lora_A.weight.square().sum() * multiplier)


def row(value=1):
    return {"input_ids": [value], "labels": [value], "n_answer_tokens": 1}


def quiet(**kwargs):
    return {"device": "cpu", "max_steps": 2, "grad_accum": 1, "log": lambda _: None, **kwargs}


def assert_restored(model):
    assert not model.training
    assert model.config.abstain_logit_bias == 5.15625
    assert model.config.use_cache
    assert not model.is_gradient_checkpointing
    assert not hasattr(model, "_require_grads_hook")


def test_bf16_nonfinite_is_skipped_without_corrupting_weights_or_advancing_lr():
    model = TinyAdapter().eval()
    progress = []
    result = train(model, [row(-1), row(1)], warmup_ratio=1, progress_callback=progress.append, **quiet())
    assert result["optimizer_attempts"] == 2
    assert result["examples_consumed"] == 2
    assert result["steps_applied"] == 1
    assert result["steps_skipped_by_safety_check"] == 1
    assert result["steps_skipped_by_scaler"] == 0
    assert result["examples_in_applied_updates"] == 1
    assert result["answer_tokens_in_applied_updates"] == 1
    assert result["trajectory"][0]["lr_used"] == result["trajectory"][1]["lr_used"]
    assert result["trajectory"][0]["lr"] == result["trajectory"][0]["lr_used"]
    assert torch.isfinite(model.lora_A.weight).all()
    assert model.lora_A.weight.item() != 1
    assert [r["event"] for r in progress] == ["optimizer_attempt", "optimizer_attempt"]
    assert_restored(model)


def test_scaler_skips_overflow_and_counts_actual_dose_on_cpu(monkeypatch):
    original_scaler = torch.amp.GradScaler
    # Exercise the actual torch scaler's skip path, with its CPU implementation.
    monkeypatch.setattr(torch.amp, "GradScaler", lambda *args, **kwargs: original_scaler("cpu", init_scale=kwargs["init_scale"], enabled=True))
    model = TinyAdapter(dtype=torch.float16).eval()
    result = train(model, [row(-1), row(1)], warmup_ratio=1, **quiet())
    assert result["steps_skipped_by_scaler"] == 1
    assert result["steps_skipped_by_safety_check"] == 0
    assert result["steps_applied"] == 1
    assert result["trajectory"][0]["step_skipped_by_scaler"]
    assert result["trajectory"][0]["lr_used"] == result["trajectory"][1]["lr_used"]
    assert torch.isfinite(model.lora_A.weight).all()
    assert_restored(model)


@pytest.mark.parametrize("value", [-1, 0])
def test_entire_run_without_finite_nonzero_updates_fails_with_partial_receipt(value):
    model = TinyAdapter().eval()
    progress = []
    original = model.lora_A.weight.detach().clone()
    with pytest.raises(RuntimeError, match="no finite nonzero gradient"):
        train(model, [row(value)], progress_callback=progress.append, **quiet())
    assert torch.equal(model.lora_A.weight, original)
    assert len(progress) == 2
    assert all(not r["optimizer_update_applied"] for r in progress)
    assert_restored(model)


def test_checkpointing_failure_is_fatal_and_restores_state():
    model = TinyAdapter(checkpointing_fails=True).eval()
    with pytest.raises(RuntimeError, match="checkpointing could not be enabled"):
        train(model, [row()], **quiet())
    assert not model.observed_gates
    assert_restored(model)


def test_silent_checkpointing_noop_is_refused():
    model = TinyAdapter().eval()
    model.gradient_checkpointing_enable = lambda **kwargs: None
    with pytest.raises(RuntimeError, match="not active"):
        train(model, [row()], **quiet())
    assert_restored(model)


def test_validation_gate_is_on_and_training_gate_is_off():
    model = TinyAdapter().eval()
    result = train(model, [row()], eval_set=[row()], eval_every=1, **quiet())
    assert len(result["validation"]) == 2
    assert model.observed_gates == [(True, True, 0), (False, False, 5.15625)] * 2
    assert_restored(model)


def test_validation_exception_restores_gate_mode_checkpointing_and_cache():
    model = TinyAdapter(validation_fails=True).eval()
    progress = []
    with pytest.raises(RuntimeError, match="validation failure"):
        train(model, [row()], eval_set=[row()], eval_every=1, progress_callback=progress.append, **quiet())
    assert len(progress) == 1
    assert progress[0]["optimizer_update_applied"]
    assert_restored(model)


def test_evaluate_loss_preserves_incoming_mode_even_on_failure():
    model = TinyAdapter(validation_fails=True).train()
    with pytest.raises(RuntimeError, match="validation failure"):
        evaluate_loss(model, [row()], "cpu")
    assert model.training
    assert model.config.abstain_logit_bias == 5.15625


def test_dense_trainable_parameter_is_refused_before_updates():
    model = TinyAdapter().eval()
    model.frozen.requires_grad_(True)
    with pytest.raises(RuntimeError, match="only actual LoRA"):
        train(model, [row()], **quiet())
    assert not model.observed_gates
    assert_restored(model)


def test_detached_adapter_path_is_reported_before_scaler_update():
    model = TinyAdapter().eval()
    # A checkpointed activation can require grad while every adapter is detached.
    model.forward = lambda **kwargs: SimpleNamespace(loss=torch.ones((), requires_grad=True))
    with pytest.raises(RuntimeError, match="no gradient reached any LoRA"):
        train(model, [row()], **quiet())
    assert_restored(model)


def test_answer_only_logits_preserve_full_input_and_supervise_terminal_eos():
    example = {"input_ids": [10, 11, 20, 2], "labels": [-100, -100, 20, 2], "n_answer_tokens": 2}
    batch = _batch(example, "cpu")
    assert batch["input_ids"].tolist() == [[10, 11, 20, 2]]
    assert batch["labels"].tolist() == [[-100, 20, 2]]
    assert batch["logits_to_keep"] == 3
    shifted = torch.nn.functional.pad(batch["labels"], (0, 1), value=-100)[:, 1:]
    assert shifted.tolist() == [[20, 2, -100]]


@pytest.mark.parametrize("labels,n_answer", [([-100, 11, 20, 2], 2), ([-100, -100, -100, 2], 2), ([-100, -100, 20, 2], 0)])
def test_answer_only_logits_reject_mismatched_or_noncontiguous_supervision(labels, n_answer):
    with pytest.raises(ValueError):
        _batch({"input_ids": [10, 11, 20, 2], "labels": labels, "n_answer_tokens": n_answer}, "cpu")
