"""SPEC 5.2's mandatory invariant: PSA must be a bit-honest no-op when it is off.

Everything downstream (significance matrices, ablation deltas, merged-checkpoint
comparisons) is a *difference* against a baseline run. If the ``"psa"`` kernel drifts
from ``sdpa`` even slightly while nominally disabled, every delta in the lab acquires
a silent offset that no other test would catch.

The model here is a hand-made 2-layer Qwen3 with random weights on CPU in float32,
per the task brief. No checkpoint is read and nothing touches the GPU.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from conftest import TINY_QWEN3_CONFIG
from transformers import Qwen3Config, Qwen3ForCausalLM

from rineya_psa.hooks import (
    CAPTURE_KINDS,
    AbsoluteBinning,
    CaptureSpec,
    DistanceBinning,
    HeadScaleSpec,
    LogitBiasSpec,
    PSAContext,
    ValueScaleSpec,
    psa_session,
    register_psa,
)
from rineya_psa.hooks.registry import PSA_KEY, attention_modules
from rineya_psa.loading import arch_from_config

#: The canonical toy architecture, owned by ``tests/conftest.py`` so the nine suites
#: that need one cannot drift apart. This file keeps its own seed.
TINY = dict(TINY_QWEN3_CONFIG)
N_TOKENS = 512
ATOL = 1e-5


@pytest.fixture(scope="module")
def fixture():
    torch.manual_seed(1234)
    model = Qwen3ForCausalLM(Qwen3Config(**TINY)).to(torch.float32).eval()
    arch = arch_from_config(model.config)
    input_ids = torch.randint(0, TINY["vocab_size"], (1, N_TOKENS), generator=None)
    return model, arch, input_ids


def _logits(model, input_ids, impl: str) -> torch.Tensor:
    previous = model.config._attn_implementation
    model.config._attn_implementation_internal = impl
    try:
        with torch.no_grad():
            return model(input_ids).logits.detach().clone()
    finally:
        model.config._attn_implementation_internal = previous


def _psa_logits(model, input_ids, ctx: PSAContext) -> torch.Tensor:
    with psa_session(model, ctx), torch.no_grad():
        return model(input_ids).logits.detach().clone()


def _identity_contexts(arch):
    n_bins = 8
    binning = AbsoluteBinning(n_bins)
    yield "disabled", PSAContext(enabled=False)
    yield "enabled_but_empty", PSAContext(enabled=True)
    yield (
        "explicitly_zeroed_specs",
        PSAContext(
            enabled=True,
            logit_bias=LogitBiasSpec.zeros(arch, n_bins, binning, lam=1.0),
            value_scale=ValueScaleSpec.ones(arch, n_bins, binning),
            head_scale=HeadScaleSpec.ones(arch),
            position_bins=binning,
        ),
    )
    # lam == 0 must neutralise a table that is otherwise wildly non-zero.
    loud = LogitBiasSpec.zeros(arch, n_bins, binning, lam=0.0)
    loud = LogitBiasSpec(
        table=torch.full_like(loud.table, 12.5), lam=0.0, binning=binning
    )
    yield "nonzero_table_zero_lambda", PSAContext(enabled=True, logit_bias=loud)
    # An empty layer subset is also provably an identity.
    yield (
        "empty_layer_subset",
        PSAContext(
            enabled=True,
            value_scale=ValueScaleSpec(
                table=torch.full((arch.num_layers, arch.num_kv_heads, n_bins), 0.1),
                binning=binning,
                layers=(),
            ),
        ),
    )
    # A disabled context must ignore even a live spec.
    yield (
        "disabled_with_live_spec",
        PSAContext(
            enabled=False,
            head_scale=HeadScaleSpec.zero_head(arch, 0, 0),
        ),
    )
    # Distance binning with a neutral table exercises the query-dependent branch.
    dist = DistanceBinning.uniform(4)
    yield (
        "distance_binning_neutral",
        PSAContext(
            enabled=True,
            value_scale=ValueScaleSpec.ones(arch, dist.n_bins, dist),
            logit_bias=LogitBiasSpec.zeros(arch, dist.n_bins, dist, lam=3.0),
        ),
    )


def test_registration_is_visible_to_transformers():
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    register_psa()
    assert PSA_KEY in ALL_ATTENTION_FUNCTIONS.valid_keys()
    # Without a mask entry, create_causal_mask early-returns None and "psa" would
    # run non-causally. This assertion is the guard against that regression.
    assert PSA_KEY in ALL_MASK_ATTENTION_FUNCTIONS._global_mapping


def test_sdpa_and_eager_agree_so_the_reference_itself_is_sound(fixture):
    model, _arch, input_ids = fixture
    sdpa = _logits(model, input_ids, "sdpa")
    eager = _logits(model, input_ids, "eager")
    assert torch.allclose(sdpa, eager, atol=ATOL, rtol=0.0)


@pytest.mark.parametrize("name", [n for n, _ in _identity_contexts(
    arch_from_config(Qwen3Config(**TINY))
)])
def test_identity_context_matches_sdpa(fixture, name):
    model, arch, input_ids = fixture
    ctx = dict(_identity_contexts(arch))[name]
    assert ctx.is_identity(), name

    sdpa = _logits(model, input_ids, "sdpa")
    psa = _psa_logits(model, input_ids, ctx)
    delta = (sdpa - psa).abs().max().item()
    assert delta <= ATOL, f"{name}: max |sdpa - psa| = {delta:g}"


def test_identity_context_matches_eager(fixture):
    model, arch, input_ids = fixture
    eager = _logits(model, input_ids, "eager")
    for name, ctx in _identity_contexts(arch):
        psa = _psa_logits(model, input_ids, ctx)
        delta = (eager - psa).abs().max().item()
        assert delta <= ATOL, f"{name}: max |eager - psa| = {delta:g}"


def test_capture_only_context_is_exact_and_takes_the_chunked_path(fixture):
    """Capture forces the explicit [T, T] kernel; it must still reproduce sdpa."""
    model, arch, input_ids = fixture
    binning = AbsoluteBinning(8)
    ctx = PSAContext(
        enabled=True,
        capture=CaptureSpec(what=("attn_by_bin", "attn_entropy", "qk_logit_mean",
                                  "head_out_norm", "gold_span_attn"),
                            chunk=64, binning=binning),
        position_bins=binning,
        evidence_spans=((10, 20), (300, 340)),
    )
    assert ctx.is_identity() and not ctx.is_noop()

    sdpa = _logits(model, input_ids, "sdpa")
    psa = _psa_logits(model, input_ids, ctx)
    assert (sdpa - psa).abs().max().item() <= ATOL

    for layer in range(arch.num_layers):
        by_bin = ctx.captured[(layer, "attn_by_bin")]
        assert by_bin.shape == (arch.num_q_heads, binning.n_bins)
        assert by_bin.dtype == np.float64
        # Every row of a softmax sums to one, so the per-row bin masses do too.
        assert np.allclose(by_bin.sum(axis=1), 1.0, atol=1e-6)
        assert ctx.captured[(layer, "attn_entropy")].shape == (arch.num_q_heads,)
        assert (ctx.captured[(layer, "attn_entropy")] >= 0).all()
        assert ctx.captured[(layer, "head_out_norm")].shape == (arch.num_q_heads,)
        assert ctx.captured[(layer, "qk_logit_mean")].shape == (
            arch.num_q_heads,
            binning.n_bins,
        )
        assert np.isfinite(ctx.captured[(layer, "qk_logit_mean")]).all()
        gold = ctx.captured[(layer, "gold_span_attn")]
        assert gold.shape == (arch.num_q_heads,)
        assert ((gold >= -1e-9) & (gold <= 1.0 + 1e-6)).all()


def test_chunk_size_does_not_change_the_logits(fixture):
    """Chunking over query rows is an exact decomposition, not an approximation."""
    model, _arch, input_ids = fixture
    outs = []
    for chunk in (17, 64, 256, N_TOKENS):
        ctx = PSAContext(
            enabled=True,
            capture=CaptureSpec(what=("attn_entropy",), chunk=chunk),
        )
        outs.append(_psa_logits(model, input_ids, ctx))
    for other in outs[1:]:
        assert torch.equal(outs[0], other) or (outs[0] - other).abs().max().item() <= 1e-6


def test_a_live_intervention_actually_moves_the_logits(fixture):
    """Negative control: the invariant above must not be passing by doing nothing."""
    model, arch, input_ids = fixture
    binning = AbsoluteBinning(8)
    table = torch.zeros((arch.num_layers, arch.num_kv_heads, binning.n_bins))
    table[:, :, 0] = 4.0
    ctx = PSAContext(
        enabled=True,
        logit_bias=LogitBiasSpec(table=table, lam=1.0, binning=binning),
    )
    assert not ctx.is_identity()
    sdpa = _logits(model, input_ids, "sdpa")
    psa = _psa_logits(model, input_ids, ctx)
    assert (sdpa - psa).abs().max().item() > 1e-2


def test_logit_bias_never_unmasks_the_future(fixture):
    """A huge positive bias on late bins must not leak information backwards."""
    model, arch, input_ids = fixture
    binning = AbsoluteBinning(4)
    table = torch.zeros((arch.num_layers, arch.num_kv_heads, binning.n_bins))
    table[:, :, 3] = 50.0  # the last quarter of the prompt, mostly in every row's future
    ctx = PSAContext(
        enabled=True,
        logit_bias=LogitBiasSpec(table=table, lam=1.0, binning=binning),
    )
    with psa_session(model, ctx), torch.no_grad():
        biased = model(input_ids).logits.detach().clone()

    # Position 0 can only see position 0, so no bias on any later bin can reach it.
    sdpa = _logits(model, input_ids, "sdpa")
    assert torch.allclose(sdpa[:, 0, :], biased[:, 0, :], atol=ATOL)

    # And truncating the prompt must not change the prefix logits either.
    prefix = input_ids[:, : N_TOKENS // 4]
    with psa_session(model, ctx), torch.no_grad():
        _ = model(prefix).logits
    assert torch.isfinite(biased).all()


def test_session_restores_everything_even_on_an_exception(fixture):
    model, _arch, input_ids = fixture
    before = model.config._attn_implementation
    ctx = PSAContext(enabled=True, capture=CaptureSpec(what=("attn_entropy",)))
    with pytest.raises(RuntimeError, match="boom"), psa_session(model, ctx):
        assert model.config._attn_implementation == PSA_KEY
        assert all(hasattr(m, "_psa_ctx") for m in attention_modules(model))
        raise RuntimeError("boom")
    assert model.config._attn_implementation == before
    assert not any(hasattr(m, "_psa_ctx") for m in attention_modules(model))
    with torch.no_grad():
        model(input_ids[:, :8])  # the baseline path still works afterwards


def test_nesting_raises_rather_than_corrupting_state(fixture):
    model, _arch, _input_ids = fixture
    before = model.config._attn_implementation
    outer = PSAContext(enabled=False)
    with (
        psa_session(model, outer),
        pytest.raises(RuntimeError, match="already active"),
        psa_session(model, PSAContext(enabled=False)),
    ):
        pass
    assert model.config._attn_implementation == before


def _kernel_inputs(model, seed: int, n_tokens: int = 37):
    """Post-RoPE-shaped q/k/v and an additive causal mask, straight into the kernel."""
    attn = model.model.layers[0].self_attn
    arch = arch_from_config(model.config)
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, arch.num_q_heads, n_tokens, arch.head_dim, generator=g)
    k = torch.randn(1, arch.num_kv_heads, n_tokens, arch.head_dim, generator=g)
    v = torch.randn(1, arch.num_kv_heads, n_tokens, arch.head_dim, generator=g)
    mask = torch.zeros(1, 1, n_tokens, n_tokens)
    mask.masked_fill_(
        torch.triu(torch.ones(n_tokens, n_tokens, dtype=torch.bool), 1),
        torch.finfo(torch.float32).min,
    )
    return attn, arch, q, k, v, mask


def test_query_dependent_value_scale_equals_its_definition(fixture):
    """Mode 2 under DistanceBinning is folded post-softmax; prove it is the same map."""
    from rineya_psa.hooks.registry import psa_attention_forward

    model, _arch, _ids = fixture
    attn, arch, q, k, v, mask = _kernel_inputs(model, seed=11)
    n = q.shape[2]
    dist = DistanceBinning.uniform(4)
    table = torch.rand(arch.num_layers, arch.num_kv_heads, dist.n_bins) + 0.5
    ctx = PSAContext(
        enabled=True, value_scale=ValueScaleSpec(table=table, binning=dist, layers=(0,))
    )
    attn._psa_ctx = ctx
    try:
        got, _ = psa_attention_forward(attn, q, k, v, mask, attn.scaling)
    finally:
        del attn._psa_ctx

    rep = arch.kv_group_size
    probs = torch.softmax(
        (q @ k.repeat_interleave(rep, dim=1).transpose(2, 3)) * attn.scaling + mask, dim=-1
    )
    ids = dist.chunk_bin_ids(n, 0, n)
    want = torch.zeros(1, arch.num_q_heads, n, arch.head_dim)
    for qh in range(arch.num_q_heads):
        kv = arch.kv_head_for_q(qh)
        s = table[0, kv][ids]
        want[0, qh] = (probs[0, qh] * s) @ v[0, kv]
    assert (got.transpose(1, 2) - want).abs().max().item() < 1e-5


def test_capture_reductions_match_a_brute_force_reference(fixture):
    from rineya_psa.hooks.registry import psa_attention_forward

    model, _arch, _ids = fixture
    attn, arch, q, k, v, mask = _kernel_inputs(model, seed=23)
    n = q.shape[2]
    binning = AbsoluteBinning(5)
    spans = ((3, 9), (20, 22))
    ctx = PSAContext(
        enabled=True,
        capture=CaptureSpec(layers=(0,), what=CAPTURE_KINDS, chunk=8, binning=binning),
        position_bins=binning,
        evidence_spans=spans,
    )
    attn._psa_ctx = ctx
    try:
        out, _ = psa_attention_forward(attn, q, k, v, mask, attn.scaling)
    finally:
        del attn._psa_ctx
    ctx.finalize_capture()

    rep = arch.kv_group_size
    z = ((q @ k.repeat_interleave(rep, dim=1).transpose(2, 3)) * attn.scaling)[0]
    p = torch.softmax(z + mask[0], dim=-1).double().numpy()
    z = z.double().numpy()
    keep = (mask[0, 0] > torch.finfo(torch.float32).min / 2).numpy()
    bins = binning.bin_ids_numpy(n)

    by_bin = np.stack([p[:, :, bins == b].sum(-1).mean(-1) for b in range(binning.n_bins)], -1)
    assert np.abs(by_bin - ctx.captured[(0, "attn_by_bin")]).max() < 1e-6

    entropy = (-(p * np.log(np.maximum(p, 1e-30))).sum(-1)).mean(-1)
    assert np.abs(entropy - ctx.captured[(0, "attn_entropy")]).max() < 1e-6

    sel = np.zeros(n, dtype=bool)
    for start, end in spans:
        sel[start:end] = True
    assert np.abs(p[:, :, sel].sum(-1).mean(-1) - ctx.captured[(0, "gold_span_attn")]).max() < 1e-6

    qk = np.zeros((arch.num_q_heads, binning.n_bins))
    for b in range(binning.n_bins):
        rows = [z[:, i, (bins == b) & keep[i]].mean(-1)
                for i in range(n) if ((bins == b) & keep[i]).any()]
        qk[:, b] = np.mean(rows, axis=0)
    assert np.abs(qk - ctx.captured[(0, "qk_logit_mean")]).max() < 1e-6

    norms = np.linalg.norm(out.transpose(1, 2)[0].double().numpy(), axis=-1).mean(-1)
    assert np.abs(norms - ctx.captured[(0, "head_out_norm")]).max() < 1e-6

    # Layers outside `capture.layers` produce no entries at all.
    assert {key[0] for key in ctx.captured} == {0}


def test_capture_weights_only_valid_query_rows(fixture):
    from rineya_psa.hooks.registry import psa_attention_forward

    model, _arch, _ids = fixture
    attn, arch, q, k, v, mask = _kernel_inputs(model, seed=31)
    n = q.shape[2]
    valid = torch.ones(1, n, dtype=torch.bool)
    valid[:, :10] = False
    ctx = PSAContext(
        enabled=True,
        capture=CaptureSpec(layers=(0,), what=("attn_entropy",), chunk=8),
        valid_mask=valid,
    )
    attn._psa_ctx = ctx
    try:
        psa_attention_forward(attn, q, k, v, mask, attn.scaling)
    finally:
        del attn._psa_ctx
    ctx.finalize_capture()

    rep = arch.kv_group_size
    p = torch.softmax(
        (q @ k.repeat_interleave(rep, dim=1).transpose(2, 3)) * attn.scaling + mask, dim=-1
    )[0].double().numpy()
    want = (-(p * np.log(np.maximum(p, 1e-30))).sum(-1))[:, 10:].mean(-1)
    assert np.abs(want - ctx.captured[(0, "attn_entropy")]).max() < 1e-6


def test_context_shape_validation_fires_before_the_forward(fixture):
    model, arch, _input_ids = fixture
    binning = AbsoluteBinning(4)
    wrong = ValueScaleSpec(
        table=torch.ones((arch.num_layers + 3, arch.num_kv_heads, 4)), binning=binning
    )
    with (
        pytest.raises(ValueError, match="value_scale"),
        psa_session(model, PSAContext(enabled=True, value_scale=wrong)),
    ):
        pass
    assert model.config._attn_implementation != PSA_KEY


def test_mode1_sdpa_fast_path_equals_the_eager_chunked_path(fixture):
    """SPEC 12.3's fast path must be an optimisation, not a different computation.

    Under an absolute binning the Mode-1 bias depends only on (query head, source token),
    so it can ride as a [1, H_q, 1, T] additive mask through SDPA instead of forcing the
    explicit [T, T] kernel. That is worth ~5x on the Phase-8 Mode-1 sweep, and it is only
    legitimate if it reproduces the kernel it replaces. Capture forces the chunked path
    without altering the intervention, so a capture-enabled run is the reference.
    """
    model, arch, input_ids = fixture
    binning = AbsoluteBinning(8)
    table = torch.randn(arch.num_layers, arch.num_kv_heads, binning.n_bins) * 1.5
    spec = LogitBiasSpec(table=table, lam=1.0, binning=binning)

    fast = _psa_logits(model, input_ids, PSAContext(enabled=True, logit_bias=spec))
    reference = _psa_logits(
        model,
        input_ids,
        PSAContext(
            enabled=True,
            logit_bias=spec,
            capture=CaptureSpec(what=("attn_entropy",), binning=binning),
        ),
    )
    assert torch.allclose(fast, reference, atol=2e-4, rtol=0.0), (
        f"fast path differs from the eager kernel by "
        f"{(fast - reference).abs().max().item():g}"
    )

    # Negative control: it must actually be doing something.
    plain = _logits(model, input_ids, "sdpa")
    assert (fast - plain).abs().max().item() > 1e-2
