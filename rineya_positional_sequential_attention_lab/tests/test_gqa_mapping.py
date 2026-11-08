"""GQA head bookkeeping (SPEC 5.5).

The classic failure is writing ``kv = g % n_kv`` instead of ``kv = g // group``.
Both are plausible-looking, both give a valid permutation, and the resulting
significance matrix is a transposed lie that no shape check catches. The tests here
pin the arithmetic *and* prove empirically that the registry's Mode 2 path moves
exactly the query heads ``Arch.q_heads_for_kv`` names, by zeroing one KV head's
values in a real Qwen3 attention module and watching which head slices of ``o_proj``'s
input go to zero.
"""

from __future__ import annotations

import pytest
import torch
from conftest import TINY_QWEN3_CONFIG
from transformers import Qwen3Config, Qwen3ForCausalLM

from rineya_psa.hooks import (
    AbsoluteBinning,
    PSAContext,
    ValueScaleSpec,
    psa_session,
)
from rineya_psa.loading import arch_from_config
from rineya_psa.types import Arch

PANEL_ARCH = Arch(
    num_layers=28,
    hidden_size=1024,
    num_q_heads=16,
    num_kv_heads=8,
    head_dim=128,
    vocab_size=151936,
)


# --------------------------------------------------------------------------
# the arithmetic
# --------------------------------------------------------------------------


def test_spec_pins_q_heads_for_kv_three():
    assert PANEL_ARCH.kv_group_size == 2
    assert PANEL_ARCH.q_heads_for_kv(3) == (6, 7)


def test_mapping_partitions_the_query_heads():
    seen: list[int] = []
    for kv in range(PANEL_ARCH.num_kv_heads):
        group = PANEL_ARCH.q_heads_for_kv(kv)
        assert len(group) == PANEL_ARCH.kv_group_size
        assert all(PANEL_ARCH.kv_head_for_q(g) == kv for g in group)
        seen.extend(group)
    assert sorted(seen) == list(range(PANEL_ARCH.num_q_heads))


def test_mapping_is_not_the_modulo_variant():
    # For the panel arch the two conventions genuinely disagree, so a test that
    # passes under both would be worthless.
    for kv in range(PANEL_ARCH.num_kv_heads):
        n_q, n_kv = PANEL_ARCH.num_q_heads, PANEL_ARCH.num_kv_heads
        modulo = tuple(g for g in range(n_q) if g % n_kv == kv)
        assert PANEL_ARCH.q_heads_for_kv(kv) != modulo


def test_mapping_bounds():
    with pytest.raises(IndexError):
        PANEL_ARCH.q_heads_for_kv(PANEL_ARCH.num_kv_heads)
    with pytest.raises(IndexError):
        PANEL_ARCH.kv_head_for_q(PANEL_ARCH.num_q_heads)


# --------------------------------------------------------------------------
# the empirical check, on a real (tiny) Qwen3
# --------------------------------------------------------------------------

#: The canonical toy architecture, owned by ``tests/conftest.py`` so the nine suites
#: that need one cannot drift apart. This file keeps its own seed.
TINY = dict(TINY_QWEN3_CONFIG)


def _tiny_model(seed: int = 0):
    torch.manual_seed(seed)
    model = Qwen3ForCausalLM(Qwen3Config(**TINY)).to(torch.float32).eval()
    return model, arch_from_config(model.config)


def _head_outputs(model, input_ids, layer: int, arch: Arch, ctx: PSAContext | None):
    """Capture ``o_proj``'s input for one layer, reshaped to ``[B, T, H_q, D]``.

    That tensor is the per-head attention output the whole GQA question is about.
    """
    grabbed: dict[str, torch.Tensor] = {}
    attn = model.model.layers[layer].self_attn

    def hook(_module, args):
        grabbed["x"] = args[0].detach().clone()

    handle = attn.o_proj.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            if ctx is None:
                model(input_ids)
            else:
                with psa_session(model, ctx):
                    model(input_ids)
    finally:
        handle.remove()
    x = grabbed["x"]
    return x.view(x.shape[0], x.shape[1], arch.num_q_heads, arch.head_dim)


@pytest.mark.parametrize("kv_head", [0, 1])
def test_zeroing_one_kv_head_moves_exactly_the_predicted_query_heads(kv_head):
    model, arch = _tiny_model()
    assert (arch.num_q_heads, arch.num_kv_heads) == (4, 2)
    input_ids = torch.randint(0, TINY["vocab_size"], (1, 48))
    layer = 1

    binning = AbsoluteBinning(1)  # one bin: the whole sequence, so this is a pure head test
    table = torch.ones((arch.num_layers, arch.num_kv_heads, 1))
    table[layer, kv_head, 0] = 0.0
    ctx = PSAContext(
        enabled=True,
        value_scale=ValueScaleSpec(table=table, binning=binning, layers=(layer,)),
    )
    assert not ctx.is_identity()

    base = _head_outputs(model, input_ids, layer, arch, None)
    hit = _head_outputs(model, input_ids, layer, arch, ctx)

    expected = set(arch.q_heads_for_kv(kv_head))
    transposed = {g for g in range(arch.num_q_heads) if g % arch.num_kv_heads == kv_head}
    assert expected != transposed, "the tiny arch must distinguish the two conventions"

    zeroed = {g for g in range(arch.num_q_heads) if torch.count_nonzero(hit[:, :, g, :]) == 0}
    assert zeroed == expected, (
        f"zeroing kv {kv_head} hit {sorted(zeroed)}, expected {sorted(expected)}"
    )

    untouched = set(range(arch.num_q_heads)) - expected
    for g in untouched:
        assert torch.allclose(base[:, :, g, :], hit[:, :, g, :], atol=1e-6), g
        assert torch.count_nonzero(base[:, :, g, :]) > 0


def test_logit_bias_reaches_the_query_heads_of_its_kv_row_only():
    """Mode 1 shares the same kv -> q expansion; check it independently of Mode 2."""
    from rineya_psa.hooks import LogitBiasSpec

    model, arch = _tiny_model(seed=3)
    input_ids = torch.randint(0, TINY["vocab_size"], (1, 48))
    layer = 0
    kv_head = 1
    n_bins = 4
    binning = AbsoluteBinning(n_bins)

    table = torch.zeros((arch.num_layers, arch.num_kv_heads, n_bins))
    table[layer, kv_head, 0] = 6.0  # a big pull toward the first quarter of the prompt
    ctx = PSAContext(
        enabled=True,
        logit_bias=LogitBiasSpec(table=table, lam=1.0, binning=binning, layers=(layer,)),
    )

    base = _head_outputs(model, input_ids, layer, arch, None)
    hit = _head_outputs(model, input_ids, layer, arch, ctx)
    moved = {
        g
        for g in range(arch.num_q_heads)
        if not torch.allclose(base[:, :, g, :], hit[:, :, g, :], atol=1e-6)
    }
    assert moved == set(arch.q_heads_for_kv(kv_head))


def test_head_scale_indexes_query_heads_not_kv_heads():
    from rineya_psa.hooks import HeadScaleSpec

    model, arch = _tiny_model(seed=5)
    input_ids = torch.randint(0, TINY["vocab_size"], (1, 32))
    layer = 1
    q_head = 3

    base = _head_outputs(model, input_ids, layer, arch, None)
    ctx = PSAContext(enabled=True, head_scale=HeadScaleSpec.zero_head(arch, layer, q_head))
    hit = _head_outputs(model, input_ids, layer, arch, ctx)

    assert torch.count_nonzero(hit[:, :, q_head, :]) == 0
    for g in range(arch.num_q_heads):
        if g != q_head:
            assert torch.allclose(base[:, :, g, :], hit[:, :, g, :], atol=1e-6), g


def test_attenuate_cell_touches_one_layer_head_bin():
    model, arch = _tiny_model(seed=7)
    n_bins = 4
    binning = AbsoluteBinning(n_bins)
    spec = ValueScaleSpec.attenuate_cell(arch, n_bins, binning, layer=1, kv_head=0, bin_index=2)
    assert spec.layers == (1,)
    assert not spec.is_identity()
    changed = (spec.table != 1.0).nonzero(as_tuple=False).tolist()
    assert changed == [[1, 0, 2]]
    assert float(spec.table[1, 0, 2]) == pytest.approx(0.75)

    input_ids = torch.randint(0, TINY["vocab_size"], (1, 40))
    ctx = PSAContext(enabled=True, value_scale=spec)
    base = _head_outputs(model, input_ids, 1, arch, None)
    hit = _head_outputs(model, input_ids, 1, arch, ctx)
    moved = {
        g
        for g in range(arch.num_q_heads)
        if not torch.allclose(base[:, :, g, :], hit[:, :, g, :], atol=1e-6)
    }
    assert moved == set(arch.q_heads_for_kv(0))
