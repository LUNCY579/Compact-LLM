"""The ``"psa"`` attention interface (SPEC 5.1/5.3/5.5).

Why an interface and not a monkeypatch
--------------------------------------
``Qwen3Attention.forward`` does not call a kernel directly. It resolves one::

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )
    attn_output, attn_weights = attention_interface(
        self, query_states, key_states, value_states, attention_mask,
        dropout=..., scaling=self.scaling, sliding_window=..., **kwargs)

``ALL_ATTENTION_FUNCTIONS`` is a ``GeneralInterface`` — a ``MutableMapping`` whose
``register(key, fn)`` classmethod writes into a *class-level* ``_global_mapping``, so
one registration is visible to every model file in the process. Registering ``"psa"``
there and flipping ``config._attn_implementation`` therefore buys us:

* the ``module`` itself (hence ``layer_idx``, ``num_key_value_groups``, ``scaling``),
  which a ``forward`` monkeypatch would have to reconstruct;
* post-RoPE, post-``q_norm``/``k_norm``, post-cache ``query``/``key``/``value``, i.e.
  exactly the tensors SPEC 5.3 defines its interventions on;
* perfect reversibility — restoring ``"sdpa"`` restores the upstream kernel object,
  with no patched function left behind if an exception unwinds mid-run. A
  monkeypatch of ``Qwen3Attention.forward`` is global, survives exceptions, and
  would silently follow the model into any other process that imports it.

The one non-obvious consequence: ``masking_utils._preprocess_mask_arguments`` early
-returns ``None`` for the mask when ``config._attn_implementation`` is absent from
``ALL_MASK_ATTENTION_FUNCTIONS._global_mapping``. An unregistered ``"psa"`` would
therefore run **without a causal mask** and score ~0 while looking healthy. So
``register_psa`` also registers ``"psa"`` in the mask interface, bound to
``eager_mask``: an explicit ``[B, 1, T, T]`` additive float mask, never ``None``,
never the ``is_causal`` shortcut, which is what an intervention kernel needs.

DEVIATION (SPEC 5.3, Mode 2): a *query-dependent* binning has no source-only value
scale, so ``V'[h,j,:] = S[l,h,b(i,j)] V[h,j,:]`` cannot be applied before
``repeat_kv``. For ``DistanceBinning`` the algebraically identical form is used
instead — the post-softmax weights are multiplied by ``S[l, kv(g), b(i,j)]`` before
the value matmul, which yields the same ``O`` without materialising a per-query copy
of ``V``. Absolute binnings still take the pre-``repeat_kv`` path SPEC mandates.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

import torch
from torch import nn
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, eager_mask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward, repeat_kv

from rineya_psa.hooks.binning import DEFAULT_CHUNK_ROWS
from rineya_psa.hooks.capture import ensure_accumulator, finalize_captures
from rineya_psa.hooks.specs import PSAContext
from rineya_psa.loading import arch_from_config

#: Key under which this lab's kernel lives in both transformers interfaces.
PSA_KEY = "psa"

_CTX_ATTR = "_psa_ctx"
_SESSION_ATTR = "_psa_session_active"

try:  # pragma: no cover - sdpa is present in every supported torch build
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
except ImportError:  # pragma: no cover
    sdpa_attention_forward = None


# ---------------------------------------------------------------------------
# mask helpers
# ---------------------------------------------------------------------------


def _slice_mask(mask: torch.Tensor | None, q_start: int, q_end: int) -> torch.Tensor | None:
    """Take query rows ``[q_start, q_end)`` out of a 4D mask, honouring broadcast rows."""
    if mask is None:
        return None
    if mask.ndim != 4:
        raise ValueError(f"psa expects a 4D attention mask, got {tuple(mask.shape)}")
    if mask.shape[2] == 1:
        return mask
    return mask[:, :, q_start:q_end, :]


def _keep_from_mask(mask: torch.Tensor | None) -> torch.Tensor | None:
    """Bool tensor of cells the mask *allows*. ``None`` means 'everything allowed'.

    Additive masks use ``finfo.min`` (not ``-inf``) for blocked cells, so a plain
    ``isfinite`` would classify them as live. Half of ``finfo.min`` separates the two
    populations with enormous margin for every float dtype in play.
    """
    if mask is None:
        return None
    if mask.dtype == torch.bool:
        return mask
    return mask > (torch.finfo(mask.dtype).min / 2)


# ---------------------------------------------------------------------------
# the kernel
# ---------------------------------------------------------------------------


def _baseline_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Identity path: hand straight back to the stock kernel, no PSA tensors built."""
    if sdpa_attention_forward is not None:
        return sdpa_attention_forward(
            module, query, key, value, attention_mask,
            dropout=dropout, scaling=scaling, **kwargs,
        )
    return eager_attention_forward(
        module, query, key, value, attention_mask,
        scaling=scaling, dropout=dropout, **kwargs,
    )


def psa_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``eager_attention_forward`` plus the three PSA modes and capture.

    Same contract as upstream: ``query`` is ``[B, H_q, T_q, D]``, ``key``/``value``
    are ``[B, H_kv, T_kv, D]``, the return is ``(attn_output [B, T_q, H_q, D],
    attn_weights | None)``.
    """
    ctx: PSAContext | None = getattr(module, _CTX_ATTR, None)
    if ctx is None or ctx.is_noop():
        return _baseline_forward(
            module, query, key, value, attention_mask, scaling, dropout, **kwargs
        )

    layer = int(getattr(module, "layer_idx", 0))
    n_rep = int(getattr(module, "num_key_value_groups", 1))
    q_len = query.shape[2]
    kv_len = key.shape[2]
    # Query row 0 sits at absolute position kv_len - q_len once a cache is in play.
    q_offset = kv_len - q_len
    src_valid = _source_valid(ctx, kv_len, query.device)

    # -- Mode 2, absolute binning: fold into V before repeat_kv (SPEC 5.3) ------
    post_softmax_scale_spec = None
    vs = ctx.value_scale
    if vs is not None and not vs.is_identity() and vs.applies_to(layer):
        if vs.binning.is_query_dependent:
            post_softmax_scale_spec = vs  # see the DEVIATION note in the module docstring
        else:
            s = vs.binning.gather_source(
                vs.table[layer].to(device=value.device, dtype=value.dtype),
                kv_len,
                valid_mask=src_valid,
                fill=1.0,
            )  # [H_kv, T] or [B, H_kv, T]
            value = value * (s.unsqueeze(0) if s.ndim == 2 else s).unsqueeze(-1)

    # -- Mode 1 table, expanded from KV heads to query heads -------------------
    bias_table_q = None
    lb = ctx.logit_bias
    if lb is not None and not lb.is_identity() and lb.applies_to(layer):
        # repeat_interleave mirrors repeat_kv exactly: query head g reads KV head
        # g // n_rep. Using .repeat() here instead would transpose the mapping.
        bias_table_q = lb.table[layer].to(device=query.device, dtype=query.dtype)
        bias_table_q = bias_table_q.repeat_interleave(n_rep, dim=0) * float(lb.lam)

    scale_table_q = None
    if post_softmax_scale_spec is not None:
        scale_table_q = post_softmax_scale_spec.table[layer].to(
            device=query.device, dtype=query.dtype
        )
        scale_table_q = scale_table_q.repeat_interleave(n_rep, dim=0)

    # -- Mode 3 -----------------------------------------------------------------
    gamma = None
    hs = ctx.head_scale
    if hs is not None and not hs.is_identity() and hs.applies_to(layer):
        gamma = hs.gamma[layer].to(device=query.device, dtype=query.dtype)

    acc = ensure_accumulator(ctx)
    capturing = acc is not None and ctx.capture.applies_to(layer)

    if bias_table_q is not None and scale_table_q is None and not capturing:
        # SPEC 12.3 fast path. Under an ABSOLUTE binning the Mode-1 bias depends only on
        # (query head, source token), never on the query row, so it is an additive mask of
        # shape [1, H_q, 1, T] that broadcasts over rows -- and SDPA takes an explicit
        # additive mask at full speed once the backend is pinned (measured 0.36 s / 1.31 GB
        # at T=4096, against 1.63 s / 4.02 GB for the eager chunked path). Folding it into
        # the causal mask here is what makes the Phase-8 Mode-1 sweep affordable: it drops
        # that arm from ~6x a plain forward to ~1.2x.
        #
        # The bias is added to the *already-masked* tensor, so -inf entries stay -inf and
        # the future cannot be unmasked. `attention_mask` may be None when the caller took
        # the is_causal shortcut, in which case an explicit causal mask is built.
        bias_mask = bias_table_q.to(query.dtype)  # [H_q, n_bins] -> gathered below
        src_bins = lb.binning.gather_source(
            bias_mask, kv_len, valid_mask=src_valid, fill=0.0
        )  # [H_q, T]
        # gather_source returns [H_q, T] unpadded, or [B, H_q, T] when a per-row
        # valid_mask is supplied. Both must broadcast over the query axis.
        additive = (
            src_bins.unsqueeze(0).unsqueeze(2) if src_bins.ndim == 2 else src_bins.unsqueeze(2)
        )  # -> [1 or B, H_q, 1, T]
        if attention_mask is not None:
            base = attention_mask
            if base.dtype != query.dtype:
                base = base.to(query.dtype)
            combined = base + additive
        else:
            neg = torch.finfo(query.dtype).min
            causal = torch.full((q_len, kv_len), neg, device=query.device, dtype=query.dtype)
            causal = causal.triu(1 + q_offset)
            combined = causal.view(1, 1, q_len, kv_len) + additive
        out, weights = _baseline_forward(
            module, query, key, value, combined, scaling, dropout, **kwargs
        )
        if gamma is not None:
            out = out * gamma.view(1, 1, -1, 1)
        return out, weights

    if bias_table_q is None and scale_table_q is None and not capturing:
        # Mode 2 is already baked into `value` and Mode 3 is a per-head rescale of the
        # output, so nothing here needs the explicit [T, T] matrix. This is the path
        # the causal-significance sweep of SPEC 5.3 takes on every one of its cells.
        out, weights = _baseline_forward(
            module, query, key, value, attention_mask, scaling, dropout, **kwargs
        )
        if gamma is not None:
            out = out * gamma.view(1, 1, -1, 1)  # out is [B, T_q, H_q, D] here
        return out, weights

    key_states = repeat_kv(key, n_rep)
    value_states = repeat_kv(value, n_rep)
    gamma_bhqd = None if gamma is None else gamma.view(1, -1, 1, 1)

    chunk = ctx.capture.chunk if ctx.capture is not None else DEFAULT_CHUNK_ROWS
    chunk = max(1, min(int(chunk), q_len))

    row_valid_full = _query_valid(ctx, kv_len, q_len, query.device)
    outputs: list[torch.Tensor] = []
    weight_chunks: list[torch.Tensor] = []
    keep_all = q_len <= chunk  # only then is returning the weight matrix affordable

    for q_start in range(0, q_len, chunk):
        q_end = min(q_start + chunk, q_len)
        q_chunk = query[:, :, q_start:q_end, :]

        # Raw scaled logits, before mask and before bias: capture wants these, and
        # they are finite everywhere so `keep` can filter them cleanly.
        z = torch.matmul(q_chunk, key_states.transpose(2, 3)) * scaling

        mask_chunk = _slice_mask(attention_mask, q_start, q_end)
        keep = _keep_from_mask(mask_chunk)
        scores = z if mask_chunk is None else z + mask_chunk

        if bias_table_q is not None:
            bias = lb.binning.gather_chunk(
                bias_table_q, kv_len, q_start, q_end, q_offset=q_offset,
                valid_mask=src_valid, fill=0.0,
            )  # [H_q, q|1, T] or [B, H_q, q|1, T]
            bias = bias if bias.ndim == 4 else bias.unsqueeze(0)
            # Never unmask: blocked cells keep their masked score untouched.
            scores = scores + bias if keep is None else torch.where(keep, scores + bias, scores)

        probs = nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        probs = nn.functional.dropout(probs, p=dropout, training=module.training)

        weighted = probs
        if scale_table_q is not None:
            sc = post_softmax_scale_spec.binning.gather_chunk(
                scale_table_q, kv_len, q_start, q_end, q_offset=q_offset,
                valid_mask=src_valid, fill=1.0,
            )
            sc = sc if sc.ndim == 4 else sc.unsqueeze(0)
            weighted = probs * sc

        out = torch.matmul(weighted, value_states)
        if gamma_bhqd is not None:
            out = out * gamma_bhqd

        if capturing:
            acc.observe(
                layer,
                probs=probs,
                logits=z,
                keep=keep,
                head_out=out,
                row_valid=None if row_valid_full is None
                else row_valid_full[:, q_start:q_end],
                evidence_spans=ctx.evidence_spans,
            )

        outputs.append(out)
        if keep_all:
            weight_chunks.append(probs)

    attn_output = (outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=2))
    attn_output = attn_output.transpose(1, 2).contiguous()
    # Chunking exists so the [T, T] matrix never lives all at once; returning it
    # would undo that, so long prompts get None (same as the sdpa kernel does).
    attn_weights = weight_chunks[0] if keep_all and weight_chunks else None
    return attn_output, attn_weights


def _source_valid(
    ctx: PSAContext, kv_len: int, device: torch.device
) -> torch.Tensor | None:
    mask = ctx.valid_mask
    if mask is None:
        return None
    m = mask.to(device=device, dtype=torch.bool)
    if m.shape[-1] != kv_len:
        raise ValueError(f"ctx.valid_mask has length {m.shape[-1]} but T is {kv_len}")
    return m


def _query_valid(
    ctx: PSAContext, kv_len: int, q_len: int, device: torch.device
) -> torch.Tensor | None:
    """``[B, q_len]`` bool over query rows, sliced from the tail of ``valid_mask``."""
    mask = ctx.valid_mask
    if mask is None:
        return None
    m = mask.to(device=device, dtype=torch.bool)
    if m.ndim == 1:
        m = m.unsqueeze(0)
    return m[:, kv_len - q_len:]


# ---------------------------------------------------------------------------
# registration and session management
# ---------------------------------------------------------------------------


def register_psa(force: bool = False) -> None:
    """Install ``"psa"`` into both transformers interfaces. Idempotent.

    ``GeneralInterface.register`` is a classmethod writing the class-level
    ``_global_mapping``, so this is process-wide and survives the creation of new
    ``AttentionInterface`` instances inside individual model files.
    """
    if force or PSA_KEY not in ALL_ATTENTION_FUNCTIONS._global_mapping:
        ALL_ATTENTION_FUNCTIONS.register(PSA_KEY, psa_attention_forward)
    if force or PSA_KEY not in ALL_MASK_ATTENTION_FUNCTIONS._global_mapping:
        ALL_MASK_ATTENTION_FUNCTIONS.register(PSA_KEY, eager_mask)


def is_registered() -> bool:
    return (
        PSA_KEY in ALL_ATTENTION_FUNCTIONS._global_mapping
        and PSA_KEY in ALL_MASK_ATTENTION_FUNCTIONS._global_mapping
    )


def attention_modules(model: Any) -> list[nn.Module]:
    """Every self-attention submodule that dispatches through the interface.

    Duck-typed rather than ``isinstance(m, Qwen3Attention)`` so Supa-A1.1's
    ``trust_remote_code`` subclass, which defines its own attention class object, is
    still found.
    """
    found = []
    for module in model.modules():
        if (
            hasattr(module, "layer_idx")
            and hasattr(module, "num_key_value_groups")
            and hasattr(module, "q_proj")
            and hasattr(module, "o_proj")
        ):
            found.append(module)
    return found


def _config_objects(model: Any) -> list[Any]:
    """Model config plus every distinct submodel / sub-config carrying the flag."""
    seen: dict[int, Any] = {}

    def visit(cfg: Any) -> None:
        if cfg is None or id(cfg) in seen:
            return
        if not hasattr(cfg, "_attn_implementation"):
            return
        seen[id(cfg)] = cfg
        for key in getattr(cfg, "sub_configs", {}) or {}:
            visit(getattr(cfg, key, None))

    visit(getattr(model, "config", None))
    for module in model.modules():
        visit(getattr(module, "config", None))
    return list(seen.values())


@contextlib.contextmanager
def psa_session(model: Any, ctx: PSAContext) -> Iterator[PSAContext]:
    """Run ``model`` through the PSA kernel, then put everything back.

    Nesting raises instead of stacking: two contexts on one module would make the
    inner exit restore the wrong ``_attn_implementation`` and leave a stale
    ``_psa_ctx`` behind, which is exactly the class of silent corruption the
    interface mechanism exists to avoid.
    """
    if getattr(model, _SESSION_ATTR, False):
        raise RuntimeError(
            "psa_session is already active on this model; nest configurations by "
            "building one PSAContext, not by nesting sessions"
        )
    register_psa()
    ctx.validate_against(arch_from_config(model.config))

    modules = attention_modules(model)
    if not modules:
        raise RuntimeError("no attention modules found; is this a Qwen3-family model?")
    configs = _config_objects(model)
    saved = [(cfg, cfg._attn_implementation) for cfg in configs]

    setattr(model, _SESSION_ATTR, True)
    for module in modules:
        setattr(module, _CTX_ATTR, ctx)
    for cfg in configs:
        cfg._attn_implementation_internal = PSA_KEY
    try:
        yield ctx
    finally:
        try:
            finalize_captures(ctx)
        finally:
            for cfg, previous in saved:
                cfg._attn_implementation_internal = previous
            for module in modules:
                if hasattr(module, _CTX_ATTR):
                    delattr(module, _CTX_ATTR)
            if hasattr(model, _SESSION_ATTR):
                delattr(model, _SESSION_ATTR)
