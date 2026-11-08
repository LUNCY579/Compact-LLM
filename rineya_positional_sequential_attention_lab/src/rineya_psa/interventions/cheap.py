"""Cheap, reversible, inference-time positional corrections — brief section 33.

Four families, A-D. None of them touches a weight on disk, none survives the ``with``
block, and each is an exact identity at its neutral setting:

A  :func:`position_logit_correction`      ``z' = z + lam * B[l, kv(g), b(j)]``
B  :func:`position_mean_centering`        ``x' = x - lam * mu[l, b(t)]`` on the Q/K input
C  :func:`low_rank_positional_projection` ``x' = x - lam * (x U) U^T`` on the Q/K input
D  :func:`rope_scaling_schedule`          per-layer ``inv_freq' = s_l * inv_freq``

**Why B and C hook ``q_proj`` / ``k_proj`` and not the residual stream.** In Qwen3 the
same post-``input_layernorm`` tensor feeds ``q_proj``, ``k_proj`` *and* ``v_proj``, and
it is also the thing the residual branch carries forward. Editing it in place would
change the values the layer writes and everything downstream reads — a destructive
edit dressed up as a probe. A ``forward_pre_hook`` on ``q_proj`` and ``k_proj``
individually changes only what the position-sensitive dot product sees, which is what
section 33 asks for. ``q_proj`` and ``k_proj`` receive the *same* tensor object, so one
estimate per layer covers both pathways.

**Position bookkeeping (a real restriction, stated rather than hidden).** B and C need
to know each token's absolute position, and a ``forward_pre_hook`` on a projection sees
only ``[batch, T, hidden]``. Under SPEC 12.2 the lab runs batch 1, unpadded, no KV
cache, so position ``t`` is row ``t`` and the sequence length is ``T`` — exact. During
incremental decoding (``T == 1``) that inference is wrong, so the hooks raise instead
of silently applying the wrong bin. Pass ``strict=False`` to skip the correction on
single-token steps, and say so in the results if you do.

DEVIATION: SPEC section 2 permits an import-time ``torch`` only under ``hooks/``,
``loading.py``, ``sequential_attention/``, ``activation_analysis/`` and
``interventions/runtime*.py``. This module is none of those and the task's file list
has no ``runtime*.py``, so ``torch`` is imported inside functions.

DEVIATION: the brief writes ``position_logit_correction(B, lam)``. Every correction
here has to attach to a specific model to be a context manager at all, so ``model`` is
the first parameter of all four; the remaining signature is as written.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch
    from torch import nn

__all__ = [
    "CHEAP_LAMBDAS",
    "LOW_RANK_RANKS",
    "ROPE_SCHEDULES",
    "PositionChannelMeans",
    "estimate_position_channel_means",
    "layerwise_rope_scaling",
    "low_rank_positional_projection",
    "position_logit_correction",
    "position_mean_centering",
    "positional_subspace",
    "rope_scale_values",
    "rope_scaling_schedule",
]

#: Dose grid of brief section 33A. 0.0 is not in it; it is the untouched model.
CHEAP_LAMBDAS: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)

#: Ranks of brief section 33C.
LOW_RANK_RANKS: tuple[int, ...] = (1, 2, 4)

#: Which projections the Q/K-pathway corrections attach to.
QK_TARGETS: tuple[str, ...] = ("q_proj", "k_proj")


# ---------------------------------------------------------------------------
# shared plumbing
# ---------------------------------------------------------------------------


def _attention_sites(model: Any, layers: Sequence[int] | None = None) -> list[tuple[int, Any]]:
    """``[(layer_idx, attention_module), ...]`` in layer order, optionally filtered."""
    from rineya_psa.hooks.registry import attention_modules

    sites = sorted(
        ((int(m.layer_idx), m) for m in attention_modules(model)), key=lambda pair: pair[0]
    )
    if not sites:
        raise RuntimeError("no attention modules found; is this a Qwen3-family model?")
    if layers is None:
        return sites
    wanted = {int(x) for x in layers}
    unknown = wanted - {i for i, _ in sites}
    if unknown:
        raise IndexError(f"layers {sorted(unknown)} are not present in this model")
    return [(i, m) for i, m in sites if i in wanted]


def _target_modules(attn: Any, targets: Sequence[str]) -> list[Any]:
    out = []
    for name in targets:
        module = getattr(attn, name, None)
        if module is None:
            raise AttributeError(f"attention module has no {name!r}; cannot hook the Q/K pathway")
        out.append(module)
    return out


def _absolute_bin_ids(n_bins: int, seq_len: int) -> np.ndarray:
    """``b(j) = clamp(j * n_bins // T, 0, n_bins - 1)``, matching ``AbsoluteBinning``.

    Duplicated in integer arithmetic here rather than importing ``hooks.binning``
    because this path runs on the CPU side of a hook and must not depend on a device.
    """
    j = np.arange(int(seq_len), dtype=np.int64)
    return np.clip((j * int(n_bins)) // max(int(seq_len), 1), 0, int(n_bins) - 1)


def _positions_or_raise(x: Any, n_bins: int, strict: bool, what: str) -> np.ndarray | None:
    seq_len = int(x.shape[-2])
    if seq_len > 1:
        return _absolute_bin_ids(n_bins, seq_len)
    if strict:
        raise RuntimeError(
            f"{what}: a single-token step carries no absolute position information in a "
            "forward_pre_hook. Run prefill-only (SPEC 12.2: batch 1, unpadded, no cache) "
            "or pass strict=False to skip the correction on decode steps."
        )
    return None


@contextlib.contextmanager
def _hook_guard(handles: list[Any]) -> Iterator[None]:
    """Remove every registered handle on the way out, exception or not."""
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
        handles.clear()


# ---------------------------------------------------------------------------
# A. position logit correction (section 33A)
# ---------------------------------------------------------------------------


def position_logit_correction(
    model: Any,
    bias: Any,
    lam: float,
    *,
    binning: Any = None,
    layers: Sequence[int] | None = None,
) -> Any:
    """Mode-1 additive bias ``z' = z + lam * B[l, kv(g), b(j)]`` as a context manager.

    ``bias`` may be ``[n_bins]`` (one profile for the whole model), ``[L, n_bins]``
    (per layer) or ``[L, H_kv, n_bins]``; the first two are tiled, explicitly, here.
    ``lam`` is the section-33A dose, :data:`CHEAP_LAMBDAS`; ``lam = 0`` is an exact
    identity because ``LogitBiasSpec`` short-circuits on it.
    """
    import torch

    from rineya_psa.hooks.binning import AbsoluteBinning
    from rineya_psa.hooks.registry import psa_session
    from rineya_psa.hooks.specs import LogitBiasSpec, PSAContext
    from rineya_psa.loading import arch_from_config

    arch = arch_from_config(model.config)
    table = np.asarray(_to_numpy(bias), dtype=np.float64)
    if table.ndim == 1:
        table = np.broadcast_to(table, (arch.num_layers, arch.num_kv_heads, table.shape[0]))
    elif table.ndim == 2:
        table = np.repeat(table[:, None, :], arch.num_kv_heads, axis=1)
    elif table.ndim != 3:
        raise ValueError(f"bias must be 1-D, 2-D or 3-D; got shape {table.shape}")
    table = np.ascontiguousarray(table)

    bins = binning if binning is not None else AbsoluteBinning(int(table.shape[-1]))
    spec = LogitBiasSpec(
        table=torch.as_tensor(table, dtype=torch.float32),
        lam=float(lam),
        binning=bins,
        layers=None if layers is None else tuple(sorted({int(x) for x in layers})),
    )
    ctx = PSAContext(enabled=True, logit_bias=spec, position_bins=bins)
    return psa_session(model, ctx)


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().to("cpu").float().numpy()
    return np.asarray(value, dtype=np.float64)


# ---------------------------------------------------------------------------
# B. position mean centering (section 33B)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PositionChannelMeans:
    """``[n_layers, n_bins, hidden]`` per-position channel means of the Q/K input.

    Estimated on *control* contexts — prompts that carry no evidence — so the means
    describe the model's positional prior rather than the content of an evaluation
    item. ``counts`` records how many token positions landed in each bin so a caller
    can see whether a bin is under-sampled before trusting its mean.
    """

    values: np.ndarray  # [L, n_bins, hidden]
    counts: np.ndarray  # [L, n_bins]
    n_bins: int
    n_contexts: int
    layers: tuple[int, ...]
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        v = np.asarray(self.values, dtype=np.float64)
        c = np.asarray(self.counts, dtype=np.int64)
        if v.ndim != 3:
            raise ValueError(f"values must be [L, n_bins, hidden]; got {v.shape}")
        if c.shape != v.shape[:2]:
            raise ValueError(f"counts {c.shape} does not match values {v.shape[:2]}")
        object.__setattr__(self, "values", np.ascontiguousarray(v))
        object.__setattr__(self, "counts", np.ascontiguousarray(c))

    @property
    def hidden_size(self) -> int:
        return int(self.values.shape[2])

    def for_layer(self, layer: int) -> np.ndarray:
        try:
            return self.values[self.layers.index(int(layer))]
        except ValueError:
            raise KeyError(f"no channel means estimated for layer {layer}") from None

    def describe(self) -> dict[str, Any]:
        return {
            "shape": list(self.values.shape),
            "n_bins": self.n_bins,
            "n_contexts": self.n_contexts,
            "layers": list(self.layers),
            "min_bin_count": int(self.counts.min()) if self.counts.size else 0,
            "meta": dict(self.meta),
        }


def estimate_position_channel_means(
    model: Any,
    contexts: Iterable[Any],
    *,
    n_bins: int = 16,
    layers: Sequence[int] | None = None,
    device: Any = None,
) -> PositionChannelMeans:
    """Average the Q/K-projection input per ``(layer, position bin, channel)``.

    One forward per control context, no LM head (the base model is called directly, so
    the ``[T, vocab]`` logit tensor of SPEC 11.1 is never built). Accumulation is in
    float64 numpy: a 32k prompt contributes 32 768 terms per bin and float32 loses the
    tail of that sum.
    """
    import torch

    sites = _attention_sites(model, layers)
    layer_ids = tuple(i for i, _ in sites)
    slot_of = {layer: slot for slot, layer in enumerate(layer_ids)}
    hidden = int(model.config.hidden_size)

    sums = np.zeros((len(sites), int(n_bins), hidden), dtype=np.float64)
    counts = np.zeros((len(sites), int(n_bins)), dtype=np.int64)

    def make_hook(slot: int) -> Any:
        def hook(module: nn.Module, args: tuple[Any, ...]) -> None:
            x = args[0]
            seq_len = int(x.shape[-2])
            ids = _absolute_bin_ids(int(n_bins), seq_len)
            flat = x.detach().reshape(-1, x.shape[-1]).to(torch.float32).cpu().numpy()
            batch = flat.shape[0] // seq_len
            tiled = np.tile(ids, batch)
            np.add.at(sums[slot], tiled, flat.astype(np.float64))
            np.add.at(counts[slot], tiled, 1)

        return hook

    handles: list[Any] = []
    n_contexts = 0
    base = getattr(model, "model", model)
    target_device = device if device is not None else next(model.parameters()).device
    with _hook_guard(handles):
        for layer, attn in sites:
            # q_proj and k_proj see the same tensor, so hooking q_proj alone is enough
            # and halves the copies out of VRAM.
            handles.append(attn.q_proj.register_forward_pre_hook(make_hook(slot_of[layer])))
        with torch.no_grad():
            for context in contexts:
                ids = _as_input_ids(context, target_device)
                base(input_ids=ids)
                n_contexts += 1

    if n_contexts == 0:
        raise ValueError("estimate_position_channel_means needs at least one control context")
    empty = counts == 0
    means = np.where(empty[..., None], 0.0, sums / np.maximum(counts, 1)[..., None])
    return PositionChannelMeans(
        values=means,
        counts=counts,
        n_bins=int(n_bins),
        n_contexts=n_contexts,
        layers=layer_ids,
        meta={"hidden_size": hidden, "empty_bins": int(empty.sum())},
    )


def _as_input_ids(context: Any, device: Any) -> torch.Tensor:
    import torch

    if hasattr(context, "dim"):
        ids = context
    elif hasattr(context, "input_ids"):
        ids = torch.as_tensor(list(context.input_ids), dtype=torch.long)
    else:
        ids = torch.as_tensor(list(context), dtype=torch.long)
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    return ids.to(device=device, dtype=torch.long)


@contextlib.contextmanager
def position_mean_centering(
    model: Any,
    means: PositionChannelMeans,
    lam: float = 1.0,
    *,
    layers: Sequence[int] | None = None,
    targets: Sequence[str] = QK_TARGETS,
    strict: bool = True,
) -> Iterator[dict[str, Any]]:
    """Subtract ``lam * mu[l, b(t)]`` from the Q/K-projection input. Reversible.

    Yields a small record of what was installed so a caller can log it. ``lam = 0``
    installs no hooks at all and is therefore bit-exact, not merely close.
    """
    import torch

    sites = _attention_sites(model, layers)
    lam_f = float(lam)
    handles: list[Any] = []
    installed = 0

    with _hook_guard(handles):
        if lam_f != 0.0:
            for layer, attn in sites:
                mu = torch.as_tensor(means.for_layer(layer), dtype=torch.float32)
                for module in _target_modules(attn, targets):
                    handles.append(
                        module.register_forward_pre_hook(
                            _centering_hook(mu, means.n_bins, lam_f, strict)
                        )
                    )
                    installed += 1
        yield {
            "correction": "position_mean_centering",
            "lam": lam_f,
            "layers": [i for i, _ in sites],
            "targets": list(targets),
            "n_hooks": installed,
            "n_bins": means.n_bins,
        }


def _centering_hook(mu: torch.Tensor, n_bins: int, lam: float, strict: bool) -> Any:
    import torch

    cache: dict[tuple[int, str, str], torch.Tensor] = {}

    def hook(module: nn.Module, args: tuple[Any, ...]) -> tuple[Any, ...]:
        x = args[0]
        ids = _positions_or_raise(x, n_bins, strict, "position_mean_centering")
        if ids is None:
            return args
        key = (int(x.shape[-2]), str(x.device), str(x.dtype))
        if key not in cache:
            index = torch.as_tensor(ids, device=x.device, dtype=torch.long)
            cache[key] = mu.to(device=x.device, dtype=x.dtype)[index]  # [T, hidden]
        return (x - lam * cache[key], *args[1:])

    return hook


# ---------------------------------------------------------------------------
# C. low-rank positional projection (section 33C)
# ---------------------------------------------------------------------------


def positional_subspace(
    means: PositionChannelMeans, rank: int, *, per_layer: bool = True
) -> np.ndarray:
    """Top-``rank`` right singular vectors of the per-position mean matrix.

    The matrix whose row space we want is ``M[l] = mu[l] - mean_b mu[l, b]``: the part
    of the Q/K input that *varies with position*. Its leading right singular vectors
    span the directions along which position is written into the residual stream, so
    ``I - U U^T`` deletes position while leaving content directions untouched to first
    order.

    Returns ``[L, hidden, rank]`` when ``per_layer`` (the default, one subspace per
    layer) and ``[hidden, rank]`` otherwise (one shared subspace fitted on all layers
    stacked). Columns are orthonormal by construction.
    """
    r = int(rank)
    if r < 1:
        raise ValueError(f"rank must be >= 1, got {rank}")
    centred = means.values - means.values.mean(axis=1, keepdims=True)  # [L, n_bins, hidden]
    if not per_layer:
        stacked = centred.reshape(-1, centred.shape[-1])
        return _top_right_singular(stacked, r)
    return np.stack([_top_right_singular(centred[i], r) for i in range(centred.shape[0])])


def _top_right_singular(matrix: np.ndarray, rank: int) -> np.ndarray:
    """``[hidden, rank]`` orthonormal basis; zero-padded if the matrix is rank-deficient."""
    hidden = int(matrix.shape[-1])
    if rank > hidden:
        raise ValueError(f"rank {rank} exceeds hidden size {hidden}")
    _, singular, vt = np.linalg.svd(np.asarray(matrix, dtype=np.float64), full_matrices=False)
    basis = np.zeros((hidden, rank), dtype=np.float64)
    floor = 1e-12 * max(singular.max(initial=0.0), 1e-30)
    keep = min(rank, int(np.count_nonzero(singular > floor)))
    if keep:
        basis[:, :keep] = vt[:keep].T
    return basis


@contextlib.contextmanager
def low_rank_positional_projection(
    model: Any,
    basis: Any,
    lam: float = 1.0,
    *,
    layers: Sequence[int] | None = None,
    targets: Sequence[str] = QK_TARGETS,
    orthonormalize: bool = True,
) -> Iterator[dict[str, Any]]:
    """Apply ``P = I - lam * U U^T`` to the Q/K-projection input. Reversible.

    ``basis`` is ``[hidden, rank]`` (shared) or ``[L, hidden, rank]`` (per layer), as
    produced by :func:`positional_subspace`. ``lam = 1`` is the full projection,
    ``lam = 0`` installs nothing and is bit-exact. ``orthonormalize`` runs a QR on each
    basis so ``U U^T`` really is a projector even if the caller supplied raw columns.
    """
    import torch

    sites = _attention_sites(model, layers)
    lam_f = float(lam)
    array = np.asarray(_to_numpy(basis), dtype=np.float64)
    if array.ndim == 2:
        per_layer_basis = {layer: array for layer, _ in sites}
    elif array.ndim == 3:
        per_layer_basis = {layer: array[layer] for layer, _ in sites}
    else:
        raise ValueError(f"basis must be [hidden, rank] or [L, hidden, rank]; got {array.shape}")

    handles: list[Any] = []
    installed = 0
    rank = int(array.shape[-1])
    with _hook_guard(handles):
        if lam_f != 0.0:
            for layer, attn in sites:
                u = per_layer_basis[layer]
                if orthonormalize:
                    u = _orthonormalize(u)
                u_t = torch.as_tensor(u, dtype=torch.float32)
                for module in _target_modules(attn, targets):
                    handles.append(module.register_forward_pre_hook(_projection_hook(u_t, lam_f)))
                    installed += 1
        yield {
            "correction": "low_rank_positional_projection",
            "lam": lam_f,
            "rank": rank,
            "layers": [i for i, _ in sites],
            "targets": list(targets),
            "n_hooks": installed,
            "shared_basis": array.ndim == 2,
        }


def _orthonormalize(basis: np.ndarray) -> np.ndarray:
    """QR with the all-zero columns of a rank-deficient basis left as zeros.

    ``np.linalg.qr`` happily returns an arbitrary orthonormal direction for a zero
    column, which would make ``P`` delete a direction nobody measured.
    """
    q, r = np.linalg.qr(np.asarray(basis, dtype=np.float64))
    dead = np.abs(np.diag(r)) <= 1e-12
    q[:, dead] = 0.0
    return q


def _projection_hook(basis: torch.Tensor, lam: float) -> Any:
    cache: dict[tuple[str, str], torch.Tensor] = {}

    def hook(module: nn.Module, args: tuple[Any, ...]) -> tuple[Any, ...]:
        x = args[0]
        key = (str(x.device), str(x.dtype))
        if key not in cache:
            cache[key] = basis.to(device=x.device, dtype=x.dtype)
        u = cache[key]
        return (x - lam * ((x @ u) @ u.transpose(-1, -2)), *args[1:])

    return hook


# ---------------------------------------------------------------------------
# D. layerwise RoPE scaling schedules (section 33D)
# ---------------------------------------------------------------------------


def _t_grid(n_layers: int) -> list[float]:
    n = int(n_layers)
    if n < 1:
        raise ValueError(f"n_layers must be >= 1, got {n_layers}")
    return [0.0] * n if n == 1 else [i / (n - 1) for i in range(n)]


def _triangle(t: float, peak: float) -> float:
    """0 at the far end of the sequence, 1 at ``peak``, linear on both sides."""
    if t <= peak:
        return 1.0 if peak <= 0.0 else t / peak
    return 1.0 if peak >= 1.0 else (1.0 - t) / (1.0 - peak)


def _flat(n: int, lo: float, hi: float, value: float) -> list[float]:
    return [float(value)] * int(n)


def _increasing(n: int, lo: float, hi: float, value: float) -> list[float]:
    return [lo + (hi - lo) * t for t in _t_grid(n)]


def _decreasing(n: int, lo: float, hi: float, value: float) -> list[float]:
    return [hi - (hi - lo) * t for t in _t_grid(n)]


def _middle_hump(n: int, lo: float, hi: float, value: float) -> list[float]:
    return [lo + (hi - lo) * math.sin(math.pi * t) for t in _t_grid(n)]


def _late_hump(n: int, lo: float, hi: float, value: float) -> list[float]:
    return [lo + (hi - lo) * _triangle(t, 0.75) for t in _t_grid(n)]


def _middle_valley(n: int, lo: float, hi: float, value: float) -> list[float]:
    return [hi - (hi - lo) * math.sin(math.pi * t) for t in _t_grid(n)]


def _early_late(n: int, lo: float, hi: float, value: float) -> list[float]:
    # A step, not a smooth valley: outer thirds at ``hi``, middle third at ``lo``.
    # Deliberately distinct from ``middle_valley`` so the two schedules test different
    # hypotheses (a graded dip versus a hard partition of the stack).
    return [lo if (1.0 / 3.0) <= t < (2.0 / 3.0) else hi for t in _t_grid(n)]


#: Section 33D's seven schedules. Each maps ``(n_layers, lo, hi, value) -> [scale] * L``.
ROPE_SCHEDULES: dict[str, Any] = {
    "flat": _flat,
    "increasing": _increasing,
    "decreasing": _decreasing,
    "middle_hump": _middle_hump,
    "late_hump": _late_hump,
    "middle_valley": _middle_valley,
    "early_late": _early_late,
}


def rope_scale_values(
    schedule: str,
    n_layers: int,
    *,
    lo: float = 0.75,
    hi: float = 1.25,
    value: float = 1.0,
) -> tuple[float, ...]:
    """Per-layer RoPE frequency multipliers for a named schedule.

    ``flat`` uses ``value`` and ignores ``lo``/``hi``; every other schedule interpolates
    between ``lo`` and ``hi``. A scale of ``s`` multiplies every inverse frequency by
    ``s``, which is equivalent to multiplying every position index by ``s``: ``s < 1``
    stretches the context (position interpolation), ``s > 1`` compresses it.
    """
    try:
        factory = ROPE_SCHEDULES[schedule]
    except KeyError:
        raise ValueError(
            f"unknown RoPE schedule {schedule!r}; known: {sorted(ROPE_SCHEDULES)}"
        ) from None
    return tuple(float(x) for x in factory(int(n_layers), float(lo), float(hi), float(value)))


def _find_rotary(model: Any) -> Any:
    """The single rotary-embedding module whose ``inv_freq`` every layer shares."""
    found = [
        m
        for m in model.modules()
        if hasattr(m, "inv_freq") and callable(getattr(m, "forward", None))
    ]
    if not found:
        raise RuntimeError("no rotary embedding module with an inv_freq buffer was found")
    if len({id(m) for m in found}) > 1:
        # Qwen3 has exactly one; a per-layer-rotary architecture would need a different
        # swap strategy and should fail loudly rather than silently scale only the first.
        raise RuntimeError(
            f"expected one shared rotary module, found {len(found)}; "
            "per-layer rotary embeddings need their own swap path"
        )
    return found[0]


class _RopeCache:
    """Per-forward memo of ``(scale, dtype) -> (cos, sin)``.

    Positions are identical across layers within one forward, so the cache key is the
    scale alone once the ``position_ids`` object has been checked for identity. Cleared
    whenever a new ``position_ids`` tensor shows up, which is exactly once per forward.
    """

    def __init__(self, rotary: Any) -> None:
        self.rotary = rotary
        self._position_ids: Any = None
        self._entries: dict[tuple[float, str], tuple[Any, Any]] = {}

    def get(self, scale: float, position_ids: Any, x: Any) -> tuple[Any, Any]:
        if position_ids is not self._position_ids:
            self._position_ids = position_ids
            self._entries = {}
        key = (float(scale), str(x.dtype))
        entry = self._entries.get(key)
        if entry is None:
            entry = _rotary_with_scale(self.rotary, x, position_ids, scale)
            self._entries[key] = entry
        return entry


def _rotary_with_scale(rotary: Any, x: Any, position_ids: Any, scale: float) -> tuple[Any, Any]:
    """Call the model's own rotary forward with ``inv_freq`` temporarily scaled.

    Going through the real module rather than reimplementing the trig is what makes
    ``scale = 1.0`` *bit*-exact: ``inv_freq * 1.0`` is ``inv_freq`` to the last bit, and
    every subsequent operation is the one the model would have performed anyway.
    ``original_inv_freq`` is swapped too, because the ``dynamic_rope_update`` decorator
    restores ``inv_freq`` from it for dynamic and longrope RoPE types.
    """
    saved_inv = rotary.inv_freq
    saved_original = getattr(rotary, "original_inv_freq", None)
    scaled = saved_inv * float(scale)
    rotary.inv_freq = scaled
    if saved_original is not None:
        rotary.original_inv_freq = scaled
    try:
        return rotary(x, position_ids)
    finally:
        rotary.inv_freq = saved_inv
        if saved_original is not None:
            rotary.original_inv_freq = saved_original


@contextlib.contextmanager
def layerwise_rope_scaling(
    model: Any, scales: Sequence[float]
) -> Iterator[dict[str, Any]]:
    """Give layer ``l`` rotary frequencies ``scales[l] * inv_freq``. Reversible.

    Implemented as a ``forward_pre_hook`` on each attention module that replaces the
    ``position_embeddings`` kwarg, because ``Qwen3Model.forward`` computes ``cos``/``sin``
    once and hands the same pair to all 28 layers — swapping the shared buffer would
    scale the whole stack, not a schedule.
    """
    sites = _attention_sites(model)
    if len(scales) != len(sites):
        raise ValueError(
            f"got {len(scales)} scales for {len(sites)} attention layers"
        )
    scale_by_layer = {layer: float(s) for (layer, _), s in zip(sites, scales, strict=True)}
    cache = _RopeCache(_find_rotary(model))

    def hook(module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        position_embeddings = kwargs.get("position_embeddings")
        position_ids = kwargs.get("position_ids")
        if position_embeddings is None:
            raise RuntimeError(
                "layerwise_rope_scaling expects position_embeddings as a keyword argument; "
                "this transformers version passes them positionally"
            )
        if position_ids is None:
            raise RuntimeError(
                "layerwise_rope_scaling needs position_ids to recompute cos/sin, and this "
                "call did not supply them"
            )
        reference = kwargs.get("hidden_states", args[0] if args else None)
        if reference is None:
            raise RuntimeError("layerwise_rope_scaling could not find the hidden_states input")
        kwargs["position_embeddings"] = cache.get(
            scale_by_layer[int(module.layer_idx)], position_ids, reference
        )
        return args, kwargs

    handles = [
        attn.register_forward_pre_hook(hook, with_kwargs=True) for _, attn in sites
    ]
    with _hook_guard(handles):
        yield {
            "correction": "layerwise_rope_scaling",
            "scales": [scale_by_layer[i] for i, _ in sites],
            "n_layers": len(sites),
        }


@contextlib.contextmanager
def rope_scaling_schedule(
    model: Any,
    schedule: str,
    *,
    lo: float = 0.75,
    hi: float = 1.25,
    value: float = 1.0,
) -> Iterator[dict[str, Any]]:
    """:func:`layerwise_rope_scaling` driven by one of :data:`ROPE_SCHEDULES`."""
    sites = _attention_sites(model)
    scales = rope_scale_values(schedule, len(sites), lo=lo, hi=hi, value=value)
    with layerwise_rope_scaling(model, scales) as record:
        record["schedule"] = schedule
        record["lo"] = float(lo)
        record["hi"] = float(hi)
        record["value"] = float(value)
        yield record
