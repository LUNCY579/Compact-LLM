from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import Qwen3Config, Qwen3ForCausalLM

from armr.hooks import Reapplication, SafetyError
from armr.metrics import logit_metrics, output_flags
from armr.provenance import cases_from, state_hash, write_json
from armr.reporting import summarize


@pytest.fixture
def model():
    torch.manual_seed(17)
    torch.set_num_threads(2)
    c = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=24, num_hidden_layers=5,
                   num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                   pad_token_id=0, eos_token_id=2, layer_types=["full_attention"]*5)
    c._attn_implementation = "eager"
    return Qwen3ForCausalLM(c).eval().requires_grad_(False)


@pytest.mark.parametrize("apps", [1, 2, 3])
def test_reapplication_counts_positions_restore(model, apps):
    ids = torch.tensor([[1, 4, 6]])
    block = model.model.layers[3]
    original = block.forward
    observed = []
    def spy(*args, **kwargs):
        observed.append(kwargs)
        return original(*args, **kwargs)
    block.forward = spy
    before = state_hash(model)
    with torch.inference_mode():
        base = model(ids, use_cache=False).logits
        observed.clear()
        with Reapplication(model, 3, apps) as wrapper:
            got = model(ids, use_cache=False).logits
            assert wrapper.calls == apps
        assert len(observed) == apps
        assert all(k["position_ids"] is observed[0]["position_ids"] and k["position_embeddings"] is observed[0]["position_embeddings"]
                   and k["attention_mask"] is observed[0]["attention_mask"] for k in observed)
        assert all(k["past_key_values"] is None for k in observed)
        assert block.forward is spy
        assert torch.equal(base, model(ids, use_cache=False).logits)
        assert torch.equal(base, got) if apps == 1 else not torch.equal(base, got)
    assert before == state_hash(model)
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())


def test_disabled_identity_and_cleanup_on_exception(model):
    ids = torch.tensor([[1, 4]])
    block = model.model.layers[3]
    with torch.inference_mode():
        base = model(ids, use_cache=False).logits
        with Reapplication(model, 3, 3, enabled=False):
            assert torch.equal(base, model(ids, use_cache=False).logits)
        with pytest.raises(RuntimeError, match="injected"):
            with Reapplication(model, 3, 2):
                raise RuntimeError("injected")
        assert "forward" not in block.__dict__
        assert not model._forward_pre_hooks
        assert torch.equal(base, model(ids, use_cache=False).logits)


@pytest.mark.parametrize("kwargs", [{"use_cache": True}, {}, {"use_cache": False, "past_key_values": object()}])
def test_cache_rejected_before_layer_execution(model, kwargs):
    with Reapplication(model, 3, 2) as wrapper, torch.inference_mode():
        with pytest.raises(SafetyError):
            model(torch.tensor([[1, 4]]), **kwargs)
        assert wrapper.calls == 0


def test_skip_is_actual_identity(model):
    ids = torch.tensor([[1, 4]])
    captured = []
    handle = model.model.layers[3].register_forward_hook(lambda m, args, output: captured.append((args[0], output)))
    with Reapplication(model, 3, skip=True), torch.inference_mode():
        model(ids, use_cache=False)
    handle.remove()
    assert captured[0][0] is captured[0][1]


@pytest.mark.parametrize("layer,apps", [(0,2),(1,3),(2,2),(5,2),(3,4),(3,True)])
def test_invalid_configuration_rejected(model, layer, apps):
    with pytest.raises(ValueError):
        Reapplication(model, layer, apps)


def test_no_training(model):
    with Reapplication(model, 3, 2):
        with pytest.raises(SafetyError, match="inference"):
            model(torch.tensor([[1,4]]), use_cache=False)


def test_overlap_rejected(model):
    with Reapplication(model, 3, 2):
        with pytest.raises(SafetyError, match="overlapping"):
            with Reapplication(model, 3, 2):
                pass
    assert not model._forward_pre_hooks


def test_nonfinite_stops_and_restores(model):
    block = model.model.layers[3]
    original = block.forward
    def bad(*args, **kwargs):
        return original(*args, **kwargs) * float("nan")
    block.forward = bad
    with pytest.raises(FloatingPointError), Reapplication(model, 3, 2), torch.inference_mode():
        model(torch.tensor([[1,4]]), use_cache=False)
    assert block.forward is bad


def test_logits_known_distribution():
    a = torch.tensor([1., 2., 3.])
    m = logit_metrics(a,a,k=2,gold=1)
    assert m["kl_base_to_arm"] == 0 and m["top1_agreement"] == 1
    assert m["gold_rank"] == 2 and m["topk_overlap"] == 1
    with pytest.raises(FloatingPointError):
        logit_metrics(a, a * float("inf"))


@pytest.mark.parametrize("text,ids,flag", [("",[],"empty"),("abc\ufffd",[1],"invalid"),
                                           ("abc",[1,2,1,2,1,2],"repetition_loop"),
                                           ("абвгдеж",[1,2],"script_corruption")])
def test_output_failure_flags(text, ids, flag):
    assert output_flags(text,ids)[flag]


def test_sealed_and_unknown_inputs_denied(tmp_path):
    for path in [tmp_path/"test2_answer_key.jsonl", tmp_path/"anything.jsonl",
                 tmp_path/"Rineya_Invisible_Limit_Lab_v1/data/test2_answer_key.jsonl"]:
        with pytest.raises((ValueError, RuntimeError)):
            cases_from(path)


def test_evidence_cannot_overwrite(tmp_path):
    path = tmp_path/"receipt.json"
    write_json(path,{"status":"FAIL"})
    with pytest.raises(FileExistsError):
        write_json(path,{"status":"PASS"})


def test_gate_fails_for_empty_or_corrupt():
    import json
    config=json.loads((Path(__file__).resolve().parents[1]/"configs/reapplication_screen.json").read_text())
    assert not summarize([],[],config)["go_fixed_smoke"]
