"""Staged attention for Qwen3, installed by module swap at inference time.

What this replaces
------------------
``Qwen3Attention.forward`` does, in order: ``q_proj``/``q_norm``, ``k_proj``/``k_norm``,
``v_proj``, RoPE on ``(q, k)``, a kernel resolved through
``ALL_ATTENTION_FUNCTIONS.get_interface(config._attn_implementation, ...)``, then
``reshape`` and ``o_proj``. :class:`SequentialQwen3Attention` reproduces that pipeline
per *stage*, over a subset of query heads, with the hidden state (arms B-D) or the
query (arm E) carried forward between stages.

Parameters are **shared by reference**. The wrapper holds the original
``Qwen3Attention`` as a submodule and calls its ``q_proj``/``k_proj``/``v_proj``/
``o_proj``/``q_norm``/``k_norm`` directly, so no weight is copied, no weight is
trainable through this path, and uninstalling restores the exact object graph. The
only visible change while installed is that parameter *names* gain an ``.inner``
segment (``model.layers.3.self_attn.inner.q_proj.weight``); nothing in this lab
saves a state dict from an installed model.

Why ``X`` is the post-layernorm tensor
--------------------------------------
``Qwen3DecoderLayer`` computes ``residual + self_attn(input_layernorm(residual))``.
The attention module therefore never sees the residual stream, only its normalised
image. The brief's recurrence ``X1 = X + a1 G1(X)`` is implemented on that normalised
image, with **no re-normalisation between stages** — that is the literal reading, and
it is precisely why ``stability.py`` is mandatory rather than optional: nothing
constrains ``X_s`` to stay in the range ``input_layernorm`` would have produced.

Arm E's refinement operator ``U_s``
-----------------------------------
``q_{s+1} = q_s + alpha_s * U_s r_s`` with

    U_s  =  q_norm . reshape . q_proj . o_proj[:, cols(G_s)]

i.e. the stage's head-space output is pushed into the residual space by the same
``o_proj`` columns that would have written it there, then read back out as a query by
the layer's own ``q_proj`` and ``q_norm``. Justification for choosing this over a
scaled identity in head space: value space and query space are different learned
bases of the same ``head_dim``, so an identity between them asserts an alignment the
model was never trained to have, whereas ``q_proj . o_proj`` is the model's own
"what I just read -> what I should ask next" map and introduces **no new parameters**.
``q_norm`` is kept on the refinement so the update lives on the same scale as the
query it modifies, which is what makes ``alpha_s`` interpretable. The refinement is
applied **before** RoPE, exactly where ``q_norm`` sits in the original forward, so the
positional phase is never added twice. A *learned* ``U_s`` is explicitly deferred to
the tiny-adaptation phase; this is the inference-only, parameter-free stand-in.

DEVIATION: incremental decoding with a KV cache is refused, not silently degraded.
Stage ``s`` of a ``recompute_kv`` arm needs keys and values derived from ``X_{s-1}``
for the *whole* context, which a cache populated by earlier steps cannot supply; a
cache would quietly mix stage-0 keys for old tokens with stage-s keys for new ones.
``loading.load_model`` sets ``config.use_cache = False``, so teacher-forced scoring
and greedy generation both work — the latter by recomputing the prefix each step.
Passing a non-``None`` ``past_key_values`` raises.

DEVIATION: a partial stage renumbers query heads (stage heads become ``0..len(G_s)-1``
inside the kernel call), so a live ``PSAContext`` carrying head-indexed tables would
be applied to the wrong heads. Rather than silently mis-indexing, a partial stage
raises when an active PSA intervention or capture is attached to the module. Arm A,
whose single stage holds every head in order, composes with ``psa_session`` normally.
"""

from __future__ import annotations

import contextlib
import sys
from collections.abc import Iterable, Iterator, Sequence
from typing import Any

import torch
from torch import nn
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3.modeling_qwen3 import (
    apply_rotary_pos_emb as _qwen3_apply_rotary_pos_emb,
)
from transformers.models.qwen3.modeling_qwen3 import (
    eager_attention_forward as _qwen3_eager_attention_forward,
)

from rineya_psa.sequential_attention.arms import SequentialArm

_PSA_CTX_ATTR = "_psa_ctx"


def _module_of(obj: Any) -> Any:
    """The python module a class was defined in, or ``None``.

    Supa-A1.1 arrives through ``trust_remote_code`` and therefore defines its own
    copies of ``apply_rotary_pos_emb`` and ``eager_attention_forward``. Resolving
    them from the attention class's defining module keeps the wrapper bit-faithful
    to whichever variant the loaded checkpoint actually runs.
    """
    return sys.modules.get(getattr(type(obj), "__module__", ""), None)


class _StageView:
    """Non-owning stand-in for the attention module, with GQA already expanded.

    The upstream kernels read ``num_key_value_groups`` off ``module`` and call
    ``repeat_kv`` themselves. A partial stage selects query heads individually, so
    its keys and values are pre-expanded to one per query head and the kernel must be
    told the expansion factor is 1. Everything else (``config``, ``layer_idx``,
    ``training``, ``is_causal``) is forwarded to the real module.
    """

    __slots__ = ("_inner", "num_key_value_groups")

    def __init__(self, inner: nn.Module) -> None:
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "num_key_value_groups", 1)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_inner"), name)


class SequentialQwen3Attention(nn.Module):
    """Staged replacement for one ``Qwen3Attention``. Inference only, reversible."""

    def __init__(self, attn: nn.Module, arm: SequentialArm) -> None:
        super().__init__()
        for required in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm"):
            if not hasattr(attn, required):
                raise TypeError(
                    f"{type(attn).__name__} has no {required!r}; this wrapper targets "
                    "Qwen3-family attention modules"
                )
        self.inner = attn  # submodule assignment shares parameters, never copies
        self.arm = arm
        self.head_dim = int(attn.head_dim)
        self.num_q_heads = int(attn.q_proj.out_features) // self.head_dim
        arm_heads = set(arm.order)
        if arm_heads != set(range(self.num_q_heads)):
            raise ValueError(
                f"arm {arm.name!r} covers {sorted(arm_heads)} but this layer has "
                f"{self.num_q_heads} query heads"
            )
        self._full_stage = tuple(
            tuple(g) == tuple(range(self.num_q_heads)) for g in arm.groups
        )
        # Cached per-device index tensors; built lazily so the wrapper can be
        # constructed on CPU and moved, and rebuilt if the model changes device.
        self._index_cache: dict[tuple[int, str, Any], torch.Tensor] = {}
        self._rope = getattr(
            _module_of(attn), "apply_rotary_pos_emb", _qwen3_apply_rotary_pos_emb
        )
        self._eager_default = getattr(
            _module_of(attn), "eager_attention_forward", _qwen3_eager_attention_forward
        )

    # -- small helpers ------------------------------------------------------

    def _q_index(self, stage: int, device: torch.device) -> torch.Tensor:
        key = (stage, "q", device)
        cached = self._index_cache.get(key)
        if cached is None:
            cached = torch.tensor(
                [int(h) for h in self.arm.groups[stage]], dtype=torch.long, device=device
            )
            self._index_cache[key] = cached
        return cached

    def _kv_index(self, stage: int, device: torch.device) -> torch.Tensor:
        """KV head index per query head of the stage — the ``repeat_kv`` mapping."""
        key = (stage, "kv", device)
        cached = self._index_cache.get(key)
        if cached is None:
            n_rep = int(self.inner.num_key_value_groups)
            cached = torch.tensor(
                [int(h) // n_rep for h in self.arm.groups[stage]],
                dtype=torch.long,
                device=device,
            )
            self._index_cache[key] = cached
        return cached

    def _o_cols(self, stage: int, device: torch.device) -> torch.Tensor:
        key = (stage, "o", device)
        cached = self._index_cache.get(key)
        if cached is None:
            d = self.head_dim
            cols = [h * d + k for h in self.arm.groups[stage] for k in range(d)]
            cached = torch.tensor(cols, dtype=torch.long, device=device)
            self._index_cache[key] = cached
        return cached

    def _project_out(self, stage: int, flat: torch.Tensor) -> torch.Tensor:
        """``o_proj`` restricted to this stage's head columns.

        For the full-head stage the real ``o_proj`` is called, which is what makes
        Arm A bit-identical rather than merely close. Any bias (Qwen3 sets
        ``attention_bias=False``, but the wrapper does not assume it) is added on
        stage 0 only, so the partitioned stages still sum to ``o_proj(full)``.
        """
        o_proj = self.inner.o_proj
        if self._full_stage[stage]:
            return o_proj(flat)
        weight = o_proj.weight.index_select(1, self._o_cols(stage, flat.device))
        bias = o_proj.bias if (stage == 0 and o_proj.bias is not None) else None
        return nn.functional.linear(flat, weight, bias)

    def _interface(self):
        config = self.inner.config
        return ALL_ATTENTION_FUNCTIONS.get_interface(
            config._attn_implementation, self._eager_default
        )

    def _guard_partial_stage(self, stage: int) -> None:
        if self._full_stage[stage]:
            return
        ctx = getattr(self.inner, _PSA_CTX_ATTR, None)
        if ctx is not None and not ctx.is_noop():
            raise RuntimeError(
                "a partial sequential stage renumbers query heads, so the head-indexed "
                f"PSA modes {ctx.active_modes()} would be applied to the wrong heads. "
                "Run PSA interventions against Arm A, or the sequential arms without a "
                "live PSAContext."
            )

    def _stage_attention(
        self,
        stage: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        kwargs: dict[str, Any],
    ) -> torch.Tensor:
        """One stage's attention, returning ``[B, T, hidden]`` after ``o_proj``.

        ``query`` is the full ``[B, H_q, T, D]`` tensor; ``key``/``value`` are
        ``[B, H_kv, T, D]``. Selection happens here so the caller does not have to
        know about GQA.
        """
        self._guard_partial_stage(stage)
        inner = self.inner
        if self._full_stage[stage]:
            module: Any = inner
            q, k, v = query, key, value
        else:
            module = _StageView(inner)
            q = query.index_select(1, self._q_index(stage, query.device))
            kv_idx = self._kv_index(stage, key.device)
            k = key.index_select(1, kv_idx)
            v = value.index_select(1, kv_idx)
        attn_output, _ = self._interface()(
            module,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not inner.training else inner.attention_dropout,
            scaling=inner.scaling,
            sliding_window=getattr(inner, "sliding_window", None),
            **kwargs,
        )
        flat = attn_output.reshape(*attn_output.shape[:2], -1).contiguous()
        return self._project_out(stage, flat)

    # -- forward ------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Any = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        if past_key_values is not None:
            raise RuntimeError(
                "sequential attention does not support an incremental KV cache; see the "
                "DEVIATION note in wrapper.py. Run with config.use_cache = False."
            )
        if position_embeddings is None:
            raise ValueError("position_embeddings are required (Qwen3 computes RoPE outside)")
        if self.arm.query_refinement:
            return self._forward_query_refinement(
                hidden_states, position_embeddings, attention_mask, kwargs
            )
        return self._forward_staged(
            hidden_states, position_embeddings, attention_mask, kwargs
        )

    def _project_qkv(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pre-RoPE ``q``, ``k``, ``v`` as ``[B, H, T, D]``, exactly as upstream does."""
        inner = self.inner
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        q = inner.q_norm(inner.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        k = inner.k_norm(inner.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        v = inner.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        return q, k, v

    def _project_q(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Pre-RoPE ``q`` alone, for stages that reuse cached keys and values."""
        inner = self.inner
        hidden_shape = (*hidden_states.shape[:-1], -1, self.head_dim)
        return inner.q_norm(inner.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)

    def _rope_q(
        self, q_pre: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """RoPE on the query only. ``apply_rotary_pos_emb`` is a two-tensor API, so the
        query is passed twice and the key half of the result discarded — cheaper than
        re-deriving the sin/cos algebra and guaranteed to match upstream exactly."""
        return self._rope(q_pre, q_pre, cos, sin)[0]

    def _forward_staged(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, None]:
        """Arms A-D: ``X_s = X_{s-1} + alpha_s G_s(X_{s-1})``."""
        cos, sin = position_embeddings
        arm = self.arm
        h = hidden_states
        total: torch.Tensor | None = None
        cached_kv: tuple[torch.Tensor, torch.Tensor] | None = None

        for stage in range(arm.n_stages):
            if arm.recompute_kv or cached_kv is None:
                # The expensive, scientifically interesting path: this stage's keys and
                # values come from X_{s-1}, so the context is re-read through the
                # representation the earlier stages produced.
                q_pre, k_pre, v = self._project_qkv(h)
                q, k = self._rope(q_pre, k_pre, cos, sin)
                if not arm.recompute_kv:
                    cached_kv = (k, v)
            else:
                # Cheap control: only the query evolves; the context stays as stage 0
                # saw it, so K/V projection, k_norm and RoPE-on-k are all skipped.
                q = self._rope_q(self._project_q(h), cos, sin)
                k, v = cached_kv
            contribution = self._stage_attention(
                stage, q, k, v, attention_mask, kwargs
            )
            alpha = float(arm.alphas[stage])
            if alpha != 1.0:
                contribution = contribution * alpha
            total = contribution if total is None else total + contribution
            if stage + 1 < arm.n_stages:
                h = h + contribution

        assert total is not None  # n_stages >= 1 is enforced by SequentialArm
        return total, None

    def _forward_query_refinement(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        kwargs: dict[str, Any],
    ) -> tuple[torch.Tensor, None]:
        """Arm E: the hidden state is fixed, the query evolves."""
        cos, sin = position_embeddings
        inner = self.inner
        arm = self.arm
        q_pre, k_pre, v = self._project_qkv(hidden_states)
        q_roped, k = self._rope(q_pre, k_pre, cos, sin)

        total: torch.Tensor | None = None
        delta: torch.Tensor | None = None  # accumulated pre-RoPE query update
        for stage in range(arm.n_stages):
            query = q_roped if delta is None else self._rope_q(q_pre + delta, cos, sin)
            contribution = self._stage_attention(
                stage, query, k, v, attention_mask, kwargs
            )
            alpha = float(arm.alphas[stage])
            scaled = contribution if alpha == 1.0 else contribution * alpha
            total = scaled if total is None else total + scaled
            if stage + 1 < arm.n_stages:
                # U_s = q_norm . reshape . q_proj, applied to the stage's residual write.
                update = inner.q_norm(
                    inner.q_proj(contribution).view(
                        *contribution.shape[:-1], -1, self.head_dim
                    )
                ).transpose(1, 2)
                if alpha != 1.0:
                    update = update * alpha
                delta = update if delta is None else delta + update

        assert total is not None
        return total, None

    # -- attributes other lab code duck-types on ----------------------------

    @property
    def layer_idx(self) -> int:
        return self.inner.layer_idx

    @property
    def config(self) -> Any:
        return self.inner.config

    @property
    def scaling(self) -> float:
        return self.inner.scaling

    @property
    def num_key_value_groups(self) -> int:
        return self.inner.num_key_value_groups

    @property
    def is_causal(self) -> bool:
        return self.inner.is_causal

    @property
    def q_proj(self) -> nn.Module:
        return self.inner.q_proj

    @property
    def o_proj(self) -> nn.Module:
        return self.inner.o_proj

    def extra_repr(self) -> str:
        return (
            f"arm={self.arm.name}, n_stages={self.arm.n_stages}, "
            f"alphas={self.arm.alphas}, recompute_kv={self.arm.recompute_kv}"
        )


# ---------------------------------------------------------------------------
# installation
# ---------------------------------------------------------------------------


def decoder_layers(model: Any) -> Sequence[nn.Module]:
    """``model.model.layers`` for a ``*ForCausalLM``, ``model.layers`` for a bare backbone."""
    for holder in (getattr(model, "model", None), model):
        layers = getattr(holder, "layers", None)
        if layers is not None:
            return layers
    raise AttributeError(
        f"{type(model).__name__} exposes no `.model.layers`; sequential attention "
        "installs by swapping `layers[i].self_attn`"
    )


def resolve_layers(model: Any, layers: Iterable[int] | str | None) -> tuple[int, ...]:
    """``None``/``"all"`` -> every layer; negative indices count from the end."""
    n = len(decoder_layers(model))
    if layers is None or layers == "all":
        return tuple(range(n))
    resolved = tuple(sorted({int(i) if int(i) >= 0 else n + int(i) for i in layers}))
    for i in resolved:
        if not 0 <= i < n:
            raise IndexError(f"layer {i} out of range [0,{n})")
    return resolved


class InstalledSequential:
    """Handle for an active installation. Also usable as a context manager.

    ``uninstall()`` is idempotent and restores the original module objects, not
    copies of them, so ``model.model.layers[i].self_attn is original`` holds again
    afterwards.
    """

    def __init__(
        self,
        model: Any,
        layers: tuple[int, ...],
        arm: SequentialArm,
        wrappers: dict[int, SequentialQwen3Attention],
        originals: dict[int, nn.Module],
    ) -> None:
        self.model = model
        self.layers = layers
        self.arm = arm
        self.wrappers = wrappers
        self._originals = originals
        self._active = True

    @property
    def active(self) -> bool:
        return self._active

    def uninstall(self) -> None:
        if not self._active:
            return
        blocks = decoder_layers(self.model)
        for idx, original in self._originals.items():
            blocks[idx].self_attn = original
        self._active = False

    def describe(self) -> dict[str, Any]:
        return {"layers": list(self.layers), "arm": self.arm.describe()}

    def __enter__(self) -> InstalledSequential:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.uninstall()


def install_sequential(
    model: Any,
    layers: Iterable[int] | str | None,
    arm_config: SequentialArm,
) -> InstalledSequential:
    """Swap ``layers[i].self_attn`` for a staged wrapper on the given layers.

    Returns a handle that is also a context manager, so both spellings work::

        handle = install_sequential(model, [12, 13], arm); ...; handle.uninstall()
        with install_sequential(model, [12, 13], arm):     ...

    Arm A on every layer is the mandatory baseline and is a provable no-op; the
    other arms are not, which is why ``stability.py`` exists.
    """
    blocks = decoder_layers(model)
    indices = resolve_layers(model, layers)
    originals: dict[int, nn.Module] = {}
    wrappers: dict[int, SequentialQwen3Attention] = {}
    try:
        for idx in indices:
            current = blocks[idx].self_attn
            if isinstance(current, SequentialQwen3Attention):
                raise RuntimeError(
                    f"layer {idx} already carries a sequential wrapper "
                    f"(arm {current.arm.name!r}); uninstall before installing again"
                )
            wrapper = SequentialQwen3Attention(current, arm_config)
            originals[idx] = current
            wrappers[idx] = wrapper
            blocks[idx].self_attn = wrapper
    except Exception:
        for idx, original in originals.items():
            blocks[idx].self_attn = original
        raise
    return InstalledSequential(model, indices, arm_config, wrappers, originals)


@contextlib.contextmanager
def sequential_session(
    model: Any,
    layers: Iterable[int] | str | None,
    arm_config: SequentialArm,
) -> Iterator[InstalledSequential]:
    """Install for the duration of the block, restore on exit including on exception."""
    handle = install_sequential(model, layers, arm_config)
    try:
        yield handle
    finally:
        handle.uninstall()
