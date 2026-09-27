"""The BF16-storage / FP32-compute path must compute EXACTLY the FP32 function (tiny random Qwen3).

A tiny Qwen3ForCausalLM is initialised, its weights rounded to BF16-representable values and
saved in BF16 (as Supa is). Loading it (a) in FP32 the ordinary way and (b) in BF16 + the CLARE
patches must give bit-identical logits and bit-identical gradients on CPU.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from clare.footprint import qwen3_mlp_footprint
from clare.masks import SkillMask
from clare.native import greedy_generate, make_fp32_compute, make_loss_fn, teacher_forced
from clare.training import DeltaHost


def tiny_config():
    return transformers.Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=48, num_hidden_layers=4,
                                    num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                                    max_position_embeddings=128, tie_word_embeddings=True, rope_theta=10000.0)


@pytest.fixture(scope="module")
def saved_tiny(tmp_path_factory):
    torch.manual_seed(0)
    cfg = tiny_config()
    m = transformers.Qwen3ForCausalLM(cfg)
    with torch.no_grad():
        for p in m.parameters():
            p.copy_(p.to(torch.bfloat16).to(torch.float32))  # BF16-representable values
            if p.ndim == 1:
                p.add_(0.01 * torch.randn_like(p)).copy_(p.to(torch.bfloat16).to(torch.float32))
    d = tmp_path_factory.mktemp("tiny_qwen3")
    m.to(torch.bfloat16).save_pretrained(d)
    return d


def load(d, dtype):
    return transformers.AutoModelForCausalLM.from_pretrained(str(d), dtype=dtype, attn_implementation="eager")


def footprint():
    return qwen3_mlp_footprint(range(1, 4), hidden=32, intermediate=48)


def patched(d):
    m = load(d, torch.bfloat16)
    fp = footprint()
    make_fp32_compute(m, skip=[s.name[:-len(".weight")] for s in fp])
    host = DeltaHost(m, fp)
    return m, host


def row(n=19, start=11):
    ids = [int(i) for i in torch.randint(0, 97, (n,), generator=torch.Generator().manual_seed(n))]
    labels = [t if i >= start else -100 for i, t in enumerate(ids)]
    return {"ids": ids, "labels": labels, "supervised_tokens": n - start}


def test_logits_bit_identical(saved_tiny):
    ref = load(saved_tiny, torch.float32).eval()
    m, host = patched(saved_tiny)
    m.eval()
    ids = torch.tensor([row()["ids"]])
    with torch.no_grad():
        a = ref(input_ids=ids).logits
        b = m(input_ids=ids).logits
    assert b.dtype == torch.float32
    assert torch.equal(a, b), float((a - b).abs().max())
    assert m.model.rotary_emb.inv_freq.dtype == torch.float32
    assert torch.equal(m.model.rotary_emb.inv_freq, ref.model.rotary_emb.inv_freq)
    assert m.get_output_embeddings().weight.dtype == torch.bfloat16, "storage stays BF16"


def test_gradients_bit_identical_and_masked(saved_tiny):
    r = row()
    ref = load(saved_tiny, torch.float32)
    fp = footprint()
    for n, p in ref.named_parameters():
        p.requires_grad_(n in fp)
    loss_ref, n_ref = make_loss_fn(ref, "cpu")(r)
    loss_ref.backward()
    m, host = patched(saved_tiny)
    params = host.start_dense(None)
    loss, n = make_loss_fn(m, "cpu")(r)
    assert n == n_ref == r["supervised_tokens"]
    assert torch.equal(loss, loss_ref)
    loss.backward()
    ref_params = dict(ref.named_parameters())
    for slot, p in zip(fp, params):
        assert torch.equal(p.grad, ref_params[slot.name].grad), slot.name
    host.stop_trainable()
    # masked dense: gradient is exactly zero outside the trainable mask
    mask = SkillMask.from_bool(fp, {s.name: (torch.arange(s.numel) % 3 == 0).numpy() for s in fp})
    params = host.start_dense(mask)
    loss, _ = make_loss_fn(m, "cpu")(r)
    loss.backward()
    for slot, p in zip(fp, params):
        g = p.grad.reshape(-1)
        allowed = torch.from_numpy(mask.bool(slot.name))
        assert torch.all(g[~allowed] == 0)
        assert torch.equal(g[allowed], ref_params[slot.name].grad.reshape(-1)[allowed])
    host.stop_trainable()


def test_frozen_delta_equals_materialised_weights(saved_tiny):
    """Evaluating base + frozen delta through the patch == an FP32 model whose weights hold W0 + D."""
    import numpy as np

    from clare.deltas import DenseDelta

    fp = footprint()
    rng = np.random.default_rng(0)
    delta = DenseDelta(fp, {s.name: (1e-2 * rng.standard_normal(s.numel)).astype(np.float32) for s in fp})
    ref = load(saved_tiny, torch.float32).eval()
    with torch.no_grad():
        params = dict(ref.named_parameters())
        for s in fp:
            params[s.name].add_(torch.from_numpy(delta.dense(s.name).reshape(s.shape)))
    m, host = patched(saved_tiny)
    m.eval()
    host.set_frozen(delta)
    ids = torch.tensor([row()["ids"]])
    with torch.no_grad():
        assert torch.equal(ref(input_ids=ids).logits, m(input_ids=ids).logits)


def test_greedy_matches_full_recompute_and_teacher_forced(saved_tiny):
    m, host = patched(saved_tiny)
    m.eval()
    prompt = row(12, 12)["ids"]
    out = greedy_generate(m, prompt, "cpu", max_new_tokens=8, eos_ids=(96,))
    seq = list(prompt)
    for t in out["ids"]:
        with torch.no_grad():
            nxt = int(m(input_ids=torch.tensor([seq])).logits[0, -1].argmax())
        assert nxt == t
        seq.append(t)
    r = row()
    tf = teacher_forced(m, r, "cpu", keep_vectors=True)
    assert tf["logsm"].shape == (r["supervised_tokens"], 97)
    loss, _ = make_loss_fn(m, "cpu")(r)
    assert abs(-tf["sum_logprob"] - float(loss)) < 1e-4


def test_eval_weight_caches_are_bit_identical_and_invalidated(saved_tiny):
    import numpy as np

    from clare.deltas import DenseDelta
    from clare.native import set_eval_weight_cache

    fp = footprint()
    rng = np.random.default_rng(1)
    delta = DenseDelta(fp, {s.name: (1e-2 * rng.standard_normal(s.numel)).astype(np.float32) for s in fp})
    m, host = patched(saved_tiny)
    m.eval()
    host.set_frozen(delta)
    ids = torch.tensor([row()["ids"]])
    with torch.no_grad():
        plain = m(input_ids=ids).logits
    n = set_eval_weight_cache(m, True, skip=[s.name[:-len(".weight")] for s in fp])
    host.cache_effective(True)
    assert n > 0 and host._eval_cache is not None
    with torch.no_grad():
        cached = m(input_ids=ids).logits
    assert torch.equal(plain, cached)
    g1 = greedy_generate(m, row(12, 12)["ids"], "cpu", max_new_tokens=6, eos_ids=(96,))
    host.set_frozen(None)  # invalidates the effective-weight cache
    assert host._eval_cache is None
    host.set_frozen(delta)
    set_eval_weight_cache(m, False)
    g2 = greedy_generate(m, row(12, 12)["ids"], "cpu", max_new_tokens=6, eos_ids=(96,))
    assert g1 == g2
