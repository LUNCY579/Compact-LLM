"""Mechanical CPU tests on tiny RANDOM models; no pretrained model evidence."""
from __future__ import annotations
from dataclasses import asdict, replace
from io import BytesIO
import copy
import pytest
import torch
from experiments.architecture_benchmark_001.budget import Config, budget, presets
from experiments.architecture_benchmark_001.model import DenseReference, SelfBlock

torch.set_num_threads(1)

def tiny(kind: str = "ced") -> Config:
    return Config(vocab_size=67, hidden_size=32, intermediate_size=96,
                  num_attention_heads=8, num_key_value_heads=4, head_dim=8,
                  encoder_layers=2 if kind == "ced" else 4,
                  decoder_layers=2 if kind == "ced" else 0, architecture=kind)

def model(kind: str = "ced") -> DenseReference:
    torch.manual_seed(19)
    return DenseReference(tiny(kind)).eval()

@pytest.mark.parametrize("kind", ["plain", "ced"])
def test_tiny_parameter_count(kind):
    m = model(kind)
    assert sum(p.numel() for p in m.parameters()) == budget(m.config)["parameters"]

@pytest.mark.parametrize("name", list(presets()))
def test_full_shape_parameter_counts_on_meta_device(name):
    # Shapes only. No real weights allocated, loaded, or run.
    c = presets()[name]
    with torch.device("meta"):
        m = DenseReference(c)
    assert sum(p.numel() for p in m.parameters()) == budget(c)["parameters"]

@pytest.mark.parametrize("kind", ["plain", "ced"])
def test_future_token_invariance(kind):
    m = model(kind)
    ids = torch.tensor([[3, 5, 7, 11, 13, 17]])
    changed = ids.clone(); changed[:, 3:] = torch.tensor([41, 43, 47])
    with torch.inference_mode():
        a, b = m(ids).logits, m(changed).logits
    torch.testing.assert_close(a[:, :3], b[:, :3], atol=2e-6, rtol=2e-5)
    assert not torch.allclose(a[:, -1], b[:, -1])

@pytest.mark.parametrize("kind", ["plain", "ced"])
@pytest.mark.parametrize("chunks", [[1, 1, 1, 1, 1, 1], [2, 3, 1]])
def test_full_vs_cached_chunks(kind, chunks):
    m = model(kind)
    ids = torch.tensor([[3, 5, 7, 11, 13, 17], [19, 23, 29, 31, 37, 41]])
    with torch.inference_mode():
        full = m(ids).logits
        offset, past, outputs = 0, None, []
        for size in chunks:
            out = m(ids[:, offset:offset + size], past=past, use_cache=True)
            past = out.cache; outputs.append(out.logits); offset += size
        torch.testing.assert_close(full, torch.cat(outputs, 1), atol=3e-6, rtol=3e-5)
        tensors = [z for pair in past.encoder for z in pair]
        if past.memory is not None:
            tensors += list(past.memory)
        actual_bf16_bytes = sum(z.numel() for z in tensors) * 2
        assert actual_bf16_bytes == budget(m.config)["kv_bytes_per_token_batch1"] * ids.numel()


def test_ced_tail_prefill_equivalence_and_continuation():
    m = model()
    ids = torch.tensor([[2, 3, 5, 7, 11]])
    with torch.inference_mode():
        full = m(ids, use_cache=True)
        tail = m(ids, use_cache=True, last_position_only=True)
        torch.testing.assert_close(full.logits[:, -1:], tail.logits, atol=3e-6, rtol=3e-5)
        new = torch.tensor([[13]])
        a = m(new, past=full.cache, use_cache=True).logits
        b = m(new, past=tail.cache, use_cache=True).logits
        c = m(torch.cat((ids, new), 1)).logits[:, -1:]
        torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)
        torch.testing.assert_close(a, c, atol=3e-6, rtol=3e-5)

@pytest.mark.parametrize("kind", ["plain", "ced"])
def test_left_padding_and_fully_masked_rows(kind):
    m = model(kind)
    plain = torch.tensor([[3, 5, 7]])
    padded = torch.tensor([[0, 0, 3, 5, 7]])
    mask = torch.tensor([[0, 0, 1, 1, 1]])
    with torch.inference_mode():
        a = m(plain).logits
        b = m(padded, attention_mask=mask, use_cache=True)
        assert torch.isfinite(b.logits).all()
        torch.testing.assert_close(a, b.logits[:, 2:], atol=3e-6, rtol=3e-5)
        c = m(torch.tensor([[11]]), past=b.cache, use_cache=True).logits
        d = m(torch.tensor([[3, 5, 7, 11]])).logits[:, -1:]
        torch.testing.assert_close(c, d, atol=3e-6, rtol=3e-5)

@pytest.mark.parametrize("kind", ["plain", "ced"])
def test_packed_document_isolation(kind):
    m = model(kind)
    ids = torch.tensor([[3, 5, 7, 11, 13, 17]])
    seg = torch.tensor([[0, 0, 0, 1, 1, 1]])
    pos = torch.tensor([[0, 1, 2, 0, 1, 2]])
    altered = ids.clone(); altered[:, :3] = torch.tensor([37, 41, 43])
    with torch.inference_mode():
        a = m(ids, segment_ids=seg, position_ids=pos).logits
        b = m(altered, segment_ids=seg, position_ids=pos).logits
        c = m(ids[:, 3:]).logits
        torch.testing.assert_close(a[:, 3:], b[:, 3:], atol=3e-6, rtol=3e-5)
        torch.testing.assert_close(a[:, 3:], c, atol=3e-6, rtol=3e-5)


def test_training_gradient_reaches_shared_memory_and_encoder():
    m = model().train()
    ids = torch.tensor([[3, 5, 7, 11, 13]])
    logits = m(ids).logits[:, :-1]
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 67), ids[:, 1:].reshape(-1))
    loss.backward()
    for p in [m.memory.k_proj.weight, m.memory.v_proj.weight,
              m.encoder[0].q_proj.weight, m.decoder[1].q_proj.weight,
              m.embed_tokens.weight]:
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    assert torch.isfinite(loss)


def test_identity_initialized_extra_block():
    m = model("plain")
    deeper = copy.deepcopy(m)
    added = SelfBlock(m.config)
    with torch.no_grad():
        added.o_proj.weight.zero_()
        added.mlp.down_proj.weight.zero_()
    deeper.encoder.insert(2, added)
    deeper.config = replace(m.config, encoder_layers=5)
    ids = torch.tensor([[2, 3, 5, 7]])
    with torch.inference_mode():
        torch.testing.assert_close(m(ids).logits, deeper(ids).logits, rtol=0, atol=0)


def test_serialization_preserves_tying_and_outputs():
    m = model()
    stream = BytesIO()
    torch.save({"config": asdict(m.config), "state_dict": m.state_dict()}, stream)
    stream.seek(0)
    data = torch.load(stream, weights_only=True)
    restored = DenseReference(Config(**data["config"])).eval()
    restored.load_state_dict(data["state_dict"], strict=True)
    assert sum(p.numel() for p in restored.parameters()) == budget(restored.config)["parameters"]
    ids = torch.tensor([[3, 5, 7]])
    with torch.inference_mode():
        torch.testing.assert_close(m(ids).logits, restored(ids).logits, rtol=0, atol=0)


def test_fail_closed_invalid_configs_and_cache_use():
    with pytest.raises(ValueError):
        Config(num_attention_heads=13)
    with pytest.raises(ValueError):
        Config(head_dim=127)
    m = model()
    ids = torch.tensor([[3, 5, 7]])
    with pytest.raises(ValueError):
        m(ids, use_cache=True)  # missing inference_mode
    with torch.inference_mode():
        cache = m(ids, use_cache=True).cache
        with pytest.raises(ValueError):
            model()(ids, past=cache, use_cache=True)
        with pytest.raises(ValueError):
            m(ids, past=cache)
        with pytest.raises(ValueError):
            m(ids, attention_mask=torch.tensor([[1, 1, 0]]), last_position_only=True)


@pytest.mark.parametrize("kind", ["plain", "ced"])
def test_causality_even_when_rotary_positions_repeat(kind):
    m = model(kind)
    ids = torch.tensor([[3, 5, 7, 11, 13]])
    changed = ids.clone(); changed[:, 2:] = torch.tensor([41, 43, 47])
    pos = torch.zeros_like(ids)
    with torch.inference_mode():
        a = m(ids, position_ids=pos).logits
        b = m(changed, position_ids=pos).logits
    torch.testing.assert_close(a[:, :2], b[:, :2], atol=3e-6, rtol=3e-5)
