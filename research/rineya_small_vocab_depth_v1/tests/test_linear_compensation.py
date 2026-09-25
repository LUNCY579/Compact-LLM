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
    assert r["fit_residual_rel_fitted"] < 1e-8
    # Amendment 4: the ridge shrinks toward I, so an identity-optimal fit is I to solver precision
    # (the first, shrink-toward-0 rule gave I - lambda (X^T X)^-1: "close to I" at atol 0.05)
    assert torch.allclose(folds[1], torch.eye(32), atol=1e-6)


def test_fit_never_worse_than_identity_on_its_own_objective_and_run0_uncompensated():
    m = tiny(seed=3)
    folds, rec = lc.fit_compensation(m, [1, 3], items(), pad_id=0, device="cpu")
    assert rec[0] == {"p": None, "s": 0, "e": 0, "compensated": False, "reason": "run starts at block 0"}
    r = rec[1]
    assert r["compensated"] and set(folds) == {1}
    assert r["fit_residual_rel_fitted"] <= r["fit_residual_rel_identity"] + 1e-9
    assert torch.equal(m.model.layers[1].mlp.down_proj.weight, tiny(seed=3).model.layers[1].mlp.down_proj.weight), \
        "fitting must leave the model's weights untouched"


def test_ridge_shrinks_toward_identity_not_zero():
    """Amendment 4. With an overwhelming penalty the fit must collapse to T = I (plain pruning, the
    free null) -- the textbook ||T||^2 penalty would collapse it to T = 0, i.e. DELETE block p's MLP."""
    m = tiny(seed=5)
    folds, rec = lc.fit_compensation(m, [0, 1, 3], items(), pad_id=0, device="cpu", ridge_rel=1e12)
    assert torch.allclose(folds[1], torch.eye(32), atol=1e-4)
    sp = rec[0]["spectrum"]
    assert sp["eig_min"] <= sp["eig_median"] <= sp["eig_max"] and sp["participation_ratio"] >= 1.0
    assert abs(sp["lambda_over_median"] - 1e12) / 1e12 < 1e-9


def test_scaled_identity_fold_is_bitwise_exact():
    """Review (c): an IDENTITY fold can only detect state corruption. T = 2I is a power-of-two
    scale, exact in floating point, so block p's output must equal y_p + 2 m_p BITWISE -- a check
    that the fold lands on the right block's MLP contribution with the right scale."""
    m = tiny()
    ids = torch.tensor([items(1)[0]["ids"]])
    cap = {}
    blk = m.model.layers[2]
    h1 = blk.post_attention_layernorm.register_forward_pre_hook(lambda mod, i: cap.__setitem__("y", i[0]))
    h2 = blk.mlp.register_forward_hook(lambda mod, i, o: cap.__setitem__("m", o))
    h3 = blk.register_forward_hook(lambda mod, i, o: cap.__setitem__("out", o[0] if isinstance(o, tuple) else o))
    with torch.no_grad():
        m.model(input_ids=ids, use_cache=False)
        want = cap["y"] + 2 * cap["m"]
        with lc.folded(m, {2: 2 * torch.eye(32)}):
            m.model(input_ids=ids, use_cache=False)
    for h in (h1, h2, h3):
        h.remove()
    assert torch.equal(cap["out"], want)


def test_fold_refuses_non_fp32_weights():
    m = tiny().to(torch.bfloat16)
    with pytest.raises(RuntimeError):
        with lc.folded(m, {1: torch.eye(32)}):
            pass


def test_torch_free_verdict_objective_equals_the_study_objective():
    """lc_verdict.cat_mean_kl must be the §4 objective exactly as the layer study computes it."""
    import random
    import layer_ablation as la
    import lc_verdict as lv
    import run_layer_study as rls
    rng = random.Random(3)
    rows = [{"id": f"r{i}", "category": f"c{i % 4}", "n": rng.randint(5, 400), "kl": rng.random() * 3,
             "top1": 0.5, "top5": 0.9, "nll": None, "nll_teacher": None} for i in range(40)]
    assert abs(lv.cat_mean_kl(rows) - rls.category_mean_kl(la.summarise(rows))) < 1e-12



def _raw(**kw):
    import lc_verdict as lv
    v = {"verdict": "INCONCLUSIVE", "size_precise": None, "seed": 17, "resamples": 2000,
         "rows_per_category": dict(lv.CALIBRATED_FOR["rows_per_category"]), "support_bar": 0.20,
         "relative_reduction_r": 0.3, "boot_frac_le_0": 0.5, "boot_frac_le_precise": 0.5, "s_ratio_hat": 0.1}
    v.update(kw)
    return v


def test_calibrated_verdict_branches():
    """Amendment 5: every branch of the registered verdict, including the binding constraint."""
    import lc_verdict as lv
    c = lv.calibrated
    assert c(_raw(verdict="UNDERPOWERED"))["verdict"] == "UNDERPOWERED"
    assert c(_raw(seed=18, boot_frac_le_0=0.0))["verdict"] == "UNCALIBRATED"          # binding: seed
    assert c(_raw(resamples=1000, boot_frac_le_0=0.0))["verdict"] == "UNCALIBRATED"   # binding: resamples
    rows = dict(lv.CALIBRATED_FOR["rows_per_category"]); rows["text_python"] = 6
    assert c(_raw(rows_per_category=rows, boot_frac_le_0=0.0))["verdict"] == "UNCALIBRATED"
    assert c(_raw(s_ratio_hat=1.0, boot_frac_le_0=0.0))["verdict"] == "UNCALIBRATED"  # above 0.8
    assert c(_raw(s_ratio_hat=2.0, boot_frac_le_0=0.0))["verdict"] == "UNCALIBRATED"  # beyond the table
    v = c(_raw(boot_frac_le_0=0.002, boot_frac_le_precise=0.001))
    assert v["verdict"] == "SUPPORTED" and v["size_precise"] is True and v["calibration_band"] == 0.2
    assert c(_raw(boot_frac_le_0=0.002, boot_frac_le_precise=0.2))["size_precise"] is False
    assert c(_raw(boot_frac_le_0=0.004))["verdict"] == "INCONCLUSIVE"          # 0.004 >= alpha 0.003
    assert c(_raw(s_ratio_hat=0.3, boot_frac_le_0=0.002))["verdict"] == "INCONCLUSIVE"   # band 0.45: alpha 0.001
    assert c(_raw(relative_reduction_r=0.1, boot_frac_le_0=0.001))["verdict"] == "REAL_BUT_SMALL"
    assert c(_raw(relative_reduction_r=-0.5, boot_frac_le_0=0.995))["verdict"] == "HARMFUL"   # mass above 0 = 0.005
    assert c(_raw(relative_reduction_r=-0.5, boot_frac_le_0=0.985))["verdict"] == "INCONCLUSIVE"


def test_code_rule_reproduces_the_calibration_analysis():
    """CANARY: re-derive the as-applied worst-cell rates from the 16,200 committed raw draws through
    lc_verdict.calibrated itself; they must equal the numbers design/HLC_CALIBRATION.json recorded.
    A mismatch means the code does not implement the rule that was calibrated."""
    import gzip
    import json
    from collections import defaultdict
    import lc_verdict as lv
    d = HERE / "design"
    rec = json.loads((d / "HLC_CALIBRATION.json").read_text(encoding="utf-8"))
    g = defaultdict(list)
    for f in sorted(d.glob("band_*.jsonl.gz")):
        for line in gzip.open(f, "rt", encoding="utf-8"):
            x = json.loads(line)
            v = lv.calibrated(_raw(verdict=x["verdict"], relative_reduction_r=x["relative_reduction_r"],
                                   boot_frac_le_0=x["boot_frac_le_0"], boot_frac_le_precise=x["boot_frac_le_precise"],
                                   s_ratio_hat=x["s_ratio_hat"]))
            g[(x["band_s_ratio"], x["s_kl"], x["kind"])].append(v)
    assert sum(len(v) for v in g.values()) == 16200
    rate = lambda xs, f: sum(1 for x in xs if f(x)) / len(xs)
    fb = max(rate(v, lambda y: y["verdict"] in ("SUPPORTED", "REAL_BUT_SMALL")) for k, v in g.items() if k[2] == "null")
    fh = max(rate(v, lambda y: y["verdict"] == "HARMFUL") for k, v in g.items() if k[2] == "null")
    fs = max(rate(v, lambda y: bool(y["size_precise"])) for k, v in g.items() if k[2] == "boundary")
    assert fb == rec["max_null_false_benefit_as_applied"]
    assert fh == rec["max_null_false_harm_as_applied"]
    assert fs == rec["max_boundary_false_size_as_applied"]
    assert fb <= 0.02 and fh <= 0.02 and fs <= 0.02
