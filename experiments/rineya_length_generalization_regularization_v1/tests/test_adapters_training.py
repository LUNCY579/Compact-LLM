from dataclasses import replace
import copy
import pytest
import torch

from experiments.rineya_length_generalization_regularization_v1.config import Plan
from experiments.rineya_length_generalization_regularization_v1.fixtures import TinyLM
from experiments.rineya_length_generalization_regularization_v1.adapters import (
    adapter_state, inject_lora, load_adapters, merge_adapters)
from experiments.rineya_length_generalization_regularization_v1.training import (
    train_bounded, bounded_prefix, completion_loss, lr_multiplier)
from experiments.rineya_length_generalization_regularization_v1.regularization import norm_targets


def row(i=0, size=8):
    return dict(id=f"r{i}", parent_id=f"p{i}", split="train", task="fixture", role="acquisition",
                prompt_ids=[5, 8] * (size // 2), target_ids=[9, 2], answer="x", answerable=True,
                synthetic=True, fixture=True, length=size, depth=0.5)


def plan(**kwargs):
    return replace(Plan(expected_layers=2, rank=2, lora_alpha=4, dtype="float32", gradient_checkpointing=False,
                        accumulation=2, max_steps=2, train_context=32, max_tokens=40), **kwargs)


def test_lora_identity_merge_and_roundtrip():
    torch.manual_seed(3)
    model = TinyLM()
    ids = torch.tensor([[2, 6, 8, 11]])
    baseline = model(ids).logits
    inject_lora(model, plan())
    assert torch.equal(model(ids).logits, baseline)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith(".B"):
                p.normal_(0, 0.1)
    state = adapter_state(model)
    clone = copy.deepcopy(model)
    load_adapters(clone, state)
    assert torch.equal(model(ids).logits, clone(ids).logits)
    before = model(ids).logits
    assert len(merge_adapters(model)) == 4
    assert not adapter_state(model)
    assert torch.allclose(before, model(ids).logits, atol=2e-6, rtol=1e-5)


def test_invalid_adapter_load_atomic():
    model = TinyLM()
    inject_lora(model, plan())
    original = adapter_state(model)
    wrong = dict(original)
    wrong[next(iter(wrong))] = torch.zeros(1)
    with pytest.raises(ValueError):
        load_adapters(model, wrong)
    assert all(torch.equal(v, adapter_state(model)[k]) for k, v in original.items())
    with pytest.raises(ValueError):
        inject_lora(model, plan())


@pytest.mark.parametrize("mode", ["none", "pre_linear", "vpad", "pre_norm_control"])
def test_training_bounds_freeze_and_snapshots(mode):
    p = plan(mode=mode, probability=0 if mode == "none" else 0.1)
    torch.manual_seed(7)
    model = TinyLM()
    inject_lora(model, p)
    frozen = {n: v.detach().clone() for n, v in model.named_parameters() if not v.requires_grad}
    original = adapter_state(model)
    snapshots = []
    result = train_bounded(model, [row(i) for i in range(10)], p,
                           on_checkpoint=lambda meta, state: snapshots.append((meta, state)))
    assert result["tokens"] == 40 and result["steps"] == 2 and result["examples"] == 4
    assert result["supervised_tokens"] == 8
    assert [s[0]["fraction"] for s in snapshots] == [0, 0.25, 0.5, 1]
    assert snapshots[-1][0]["tokens"] == 40
    assert any(not torch.equal(v, adapter_state(model)[k]) for k, v in original.items())
    assert all(torch.equal(v, dict(model.named_parameters())[n]) for n, v in frozen.items())
    assert all("forward" not in m.__dict__ for _, m, _ in norm_targets(model, 2))


def test_prefix_no_truncation_or_hidden_repeats():
    assert len(bounded_prefix([row(i) for i in range(3)], plan(max_tokens=25))) == 2
    assert len(bounded_prefix([row(0)], plan(max_tokens=1000))) == 1
    with pytest.raises(ValueError):
        bounded_prefix([row()], plan(max_tokens=1))
    with pytest.raises(ValueError):
        bounded_prefix([row(size=64)], plan())
    wrong = row(); wrong["split"] = "dev"
    with pytest.raises(ValueError):
        bounded_prefix([wrong], plan())


def test_failure_restores_and_does_not_retry():
    model = TinyLM(); p = plan(mode="vpad", probability=0.1)
    inject_lora(model, p)
    model.eval()
    calls = []
    def stop():
        calls.append(1)
        raise RuntimeError("resource gate")
    with pytest.raises(RuntimeError):
        train_bounded(model, [row()], p, guard=stop)
    assert len(calls) == 1 and not model.training
    assert len(norm_targets(model, 2)) == 4


def test_loss_masks_prompt_and_shifts():
    logits = torch.randn(1, 5, 12, requires_grad=True)
    labels = torch.tensor([[-100, -100, -100, 4, 5]])
    loss = completion_loss(logits, labels)
    expected = torch.nn.functional.cross_entropy(logits[0, 2:4], labels[0, 3:])
    assert torch.equal(loss, expected)
    loss.backward()
    assert logits.grad[0, :2].abs().sum() == 0 and logits.grad[0, 4].abs().sum() == 0
    with pytest.raises(ValueError):
        completion_loss(logits, torch.full_like(labels, -100))


def test_lr_schedule_and_scope():
    assert lr_multiplier(1, 10, 0.2) == 0.5
    assert lr_multiplier(2, 10, 0.2) == 1.0
    assert lr_multiplier(10, 10, 0.2) > 0
    with pytest.raises(ValueError):
        train_bounded(TinyLM(), [row()], plan())
