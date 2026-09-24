"""Linear compensation (Amendment 2, H-LC): the fold must be inert at identity, exact against a direct
computation, fully restored on exit, refused inside keep_layers, and the fit must recover a KNOWN
answer (a dropped block that is exactly the identity map must give T ~ I and ~zero residual).

Runs on an in-memory tiny random Qwen3 (4 blocks, hidden 32): no files, no GPU, seconds."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
import linear_compensation as lc  # noqa: E402
from layer_ablation import keep_layers  # noqa: E402


def tiny(seed=0):
    torch.manual_seed(seed)
    cfg = transformers.Qwen3Config(vocab_size=256, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                                   num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                                   tie_word_embeddings=True, max_position_embeddings=128, max_window_layers=4)
    m = transformers.Qwen3ForCausalLM._from_config(cfg, attn_implementation="eager", torch_dtype=torch.float32)
    return m.eval()


def items(n=12, seed=1):
    g = torch.Generator().manual_seed(seed)
    return [{"ids": torch.randint(1, 256, (int(torch.randint(8, 40, (1,), generator=g)),), generator=g).tolist()}
            for _ in range(n)]


def hidden(m, ids):
    with torch.no_grad():
        return m.model(input_ids=torch.tensor([ids]), use_cache=False).last_hidden_state


def test_runs():
    assert lc.runs([0, 1, 2, 3], 4) == []
    assert lc.runs([0, 3], 4) == [{"p": 0, "s": 1, "e": 2}]
    assert lc.runs([1, 3], 4) == [{"p": None, "s": 0, "e": 0}, {"p": 1, "s": 2, "e": 2}]
    assert lc.runs([0, 2], 5) == [{"p": 0, "s": 1, "e": 1}, {"p": 2, "s": 3, "e": 4}]


def test_identity_fold_is_bitwise_inert_and_restored():
    m = tiny()
    ids = items(1)[0]["ids"]
    W0 = m.model.layers[1].mlp.down_proj.weight.detach().clone()
    with keep_layers(m, [0, 1, 3]):
        ref = hidden(m, ids)
    with lc.folded(m, {1: torch.eye(32)}), keep_layers(m, [0, 1, 3]):
        got = hidden(m, ids)
    assert torch.equal(got, ref)
    with lc.folded(m, {1: torch.randn(32, 32)}):
        assert not torch.equal(m.model.layers[1].mlp.down_proj.weight, W0)
    assert torch.equal(m.model.layers[1].mlp.down_proj.weight, W0), "fold not restored exactly"


def test_fold_equals_direct_computation():
    """Block p's output under the fold must equal y_p + m_p T computed from the UNfolded block."""
    m = tiny()
    ids = torch.tensor([items(1)[0]["ids"]])
    T = torch.randn(32, 32) * 0.3 + torch.eye(32)
    cap = {}
    blk = m.model.layers[1]
    h1 = blk.post_attention_layernorm.register_forward_pre_hook(lambda mod, i: cap.__setitem__("y", i[0]))
    h2 = blk.mlp.register_forward_hook(lambda mod, i, o: cap.__setitem__("m", o))
    h3 = blk.register_forward_hook(lambda mod, i, o: cap.__setitem__("out", o[0] if isinstance(o, tuple) else o))
    with torch.no_grad():
        m.model(input_ids=ids, use_cache=False)
        want = cap["y"] + cap["m"] @ T
        with lc.folded(m, {1: T}):
            m.model(input_ids=ids, use_cache=False)
    for h in (h1, h2, h3):
        h.remove()
    assert torch.allclose(cap["out"], want, rtol=1e-4, atol=1e-5)


def test_fold_refuses_inside_keep_layers():
    m = tiny()
    with keep_layers(m, [0, 1, 3]):
        with pytest.raises(RuntimeError):
            with lc.folded(m, {1: torch.eye(32)}):
                pass


def test_fit_recovers_identity_for_an_identity_block():
    """POSITIVE CONTROL. Zero block 2's o_proj and down_proj: it becomes the identity map, so the
    teacher's residual after block 2 IS block 1's output, the optimal T is I and the residual is 0.
    The fit must find that (up to the ridge shrinkage)."""
    m = tiny()
    with torch.no_grad():
        m.model.layers[2].self_attn.o_proj.weight.zero_()
        m.model.layers[2].mlp.down_proj.weight.zero_()
    folds, rec = lc.fit_compensation(m, [0, 1, 3], items(), pad_id=0, device="cpu")
    (r,) = rec
    assert r["compensated"] and r["p"] == 1 and r["e"] == 2
    assert r["fit_residual_rel_identity"] < 1e-10                   # identity is exact here
    assert r["fit_residual_rel_fitted"] < 1e-4
    assert torch.allclose(folds[1], torch.eye(32), atol=0.05)


def test_fit_never_worse_than_identity_on_its_own_objective_and_run0_uncompensated():
    m = tiny(seed=3)
    folds, rec = lc.fit_compensation(m, [1, 3], items(), pad_id=0, device="cpu")
    assert rec[0] == {"p": None, "s": 0, "e": 0, "compensated": False, "reason": "run starts at block 0"}
    r = rec[1]
    assert r["compensated"] and set(folds) == {1}
    assert r["fit_residual_rel_fitted"] <= r["fit_residual_rel_identity"] + 1e-9
    assert torch.equal(m.model.layers[1].mlp.down_proj.weight, tiny(seed=3).model.layers[1].mlp.down_proj.weight), \
        "fitting must leave the model's weights untouched"
