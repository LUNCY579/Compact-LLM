"""Positional Significance Attention — brief section 32.

PSA injects a significance matrix ``S[l, kv_head, bin]`` into *ordinary parallel*
attention, through the three hook modes of SPEC 5.3. It is the mandatory control for
the sequential arm: if PSA alone recovers the effect, sequentiality contributed
nothing and the study says so.

The three modes are deliberately parameterised so that ``lam = 0`` is an exact
identity in every one of them, which is what makes a lambda sweep interpretable as a
dose-response curve anchored at the untouched model:

===========  ==================================================  ==================
mode         what the matrix becomes                             identity at
===========  ==================================================  ==================
``logit``    ``z' = z + lam * S_hat[l, kv(g), b(j)]``             ``lam = 0``
``value``    ``V' = (1 + lam * S_hat[l, h, b(j)]) * V``           ``lam = 0``
``head``     ``O'[g] = (1 + lam * g_hat[l, g]) * O[g]``           ``lam = 0``
===========  ==================================================  ==================

``S_hat`` is the *standardised* matrix: each ``(layer, kv_head)`` profile is centred
over bins and the whole table is divided by its global peak deviation. Centring is
what makes ``logit`` a reallocation of attention mass rather than a temperature
change, and the single global divisor (rather than a per-row one) preserves the
relative importance of heads, which is the thing the significance matrix measures.

Mode 3 has no bin axis, so the bin axis has to be reduced. The neutral reduction is
used — mean over bins, then centred across the heads of a layer — and the resulting
per-KV-head vector is expanded to query heads with ``Arch.q_heads_for_kv`` semantics.

DEVIATION: ``significance/`` is empty at the time of writing, so there is no
``SignificanceMatrix`` class to import. :func:`coerce_significance` duck-types
whatever that module eventually ships (an ``.values`` / ``.table`` / ``.matrix``
attribute, a bare ``numpy`` array, or a torch tensor) into the local
:class:`SignificanceTable`. When ``significance/`` lands, nothing here needs to
change unless it names its payload something else again.

DEVIATION: SPEC section 2 permits an import-time ``torch`` only under ``hooks/``,
``loading.py``, ``sequential_attention/``, ``activation_analysis/`` and
``interventions/runtime*.py``. This module is not ``runtime*.py`` and the file list
for this task does not include one, so ``torch`` and ``rineya_psa.hooks`` are
imported inside functions instead — the same pattern ``evaluation/runner.py`` uses
and documents.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from rineya_psa.types import Arch

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from rineya_psa.hooks.binning import PositionBinning
    from rineya_psa.hooks.specs import PSAContext

__all__ = [
    "PSA_LAMBDA_GRID",
    "PSA_MODES",
    "PSAVariant",
    "SignificanceTable",
    "broadcast_over_heads",
    "build_psa_context",
    "canonical_mode",
    "coerce_significance",
    "head_gamma_from_matrix",
    "layer_band",
    "psa_lambda_sweep",
    "psa_mode_sweep",
    "psa_sweep",
    "standardize_significance",
]

#: Canonical mode names. ``logit``/``value``/``head`` are SPEC 5.3's Modes 1/2/3.
PSA_MODES: tuple[str, ...] = ("logit", "value", "head")

#: Dose grid for the lambda sweep. Anchored at 0 so every curve starts at baseline.
PSA_LAMBDA_GRID: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)

_MODE_ALIASES: dict[str, str] = {
    "logit": "logit",
    "logit_bias": "logit",
    "mode1": "logit",
    "1": "logit",
    "value": "value",
    "value_scale": "value",
    "mode2": "value",
    "2": "value",
    "head": "head",
    "head_scale": "head",
    "head_out": "head",
    "mode3": "head",
    "3": "head",
}

#: Names understood by :func:`layer_band`. Thirds of the stack, low index first.
LAYER_BANDS: tuple[str, ...] = ("early", "middle", "late", "all")


def canonical_mode(mode: str) -> str:
    """Fold the aliases a caller might reasonably type into one of :data:`PSA_MODES`."""
    key = str(mode).strip().lower().replace("-", "_")
    try:
        return _MODE_ALIASES[key]
    except KeyError:
        raise ValueError(
            f"unknown PSA mode {mode!r}; expected one of {list(PSA_MODES)} "
            "(or the aliases mode1/mode2/mode3)"
        ) from None


def layer_band(n_layers: int, name: str) -> tuple[int, ...] | None:
    """Layer subset for a named band. ``"all"`` returns ``None`` (= every layer).

    Thirds rather than a fixed count, so the same band name means the same fraction
    of the stack on a 2-layer test model and on the 28-layer panel.
    """
    if name not in LAYER_BANDS:
        raise ValueError(f"unknown layer band {name!r}; known: {list(LAYER_BANDS)}")
    if name == "all":
        return None
    n = int(n_layers)
    if n < 1:
        raise ValueError(f"n_layers must be >= 1, got {n_layers}")
    lo = (n * {"early": 0, "middle": 1, "late": 2}[name]) // 3
    hi = (n * {"early": 1, "middle": 2, "late": 3}[name]) // 3
    return tuple(range(lo, max(hi, lo + 1)))


# ---------------------------------------------------------------------------
# the matrix
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignificanceTable:
    """``[n_layers, n_kv_heads, n_bins]`` significance, plus the grid it was measured on.

    Held as float64 numpy rather than a tensor so the whole module stays torch-free at
    import time (SPEC section 2) and so a matrix can be manipulated, permuted and
    hashed on a CPU-only interpreter.
    """

    values: np.ndarray
    binning: Any = None  # PositionBinning | None; resolved lazily, see ``resolve_binning``
    name: str = "significance"
    source: str = "unknown"
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        v = np.asarray(self.values, dtype=np.float64)
        if v.ndim != 3:
            raise ValueError(
                f"SignificanceTable needs [n_layers, n_kv_heads, n_bins]; got {v.shape}"
            )
        if not np.all(np.isfinite(v)):
            raise ValueError(f"{self.name}: significance matrix contains non-finite entries")
        object.__setattr__(self, "values", np.ascontiguousarray(v))
        if self.binning is not None and int(getattr(self.binning, "n_bins", -1)) != v.shape[2]:
            raise ValueError(
                f"{self.name}: matrix has {v.shape[2]} bins but "
                f"{getattr(self.binning, 'name', self.binning)} has "
                f"{getattr(self.binning, 'n_bins', '?')}"
            )

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(x) for x in self.values.shape)  # type: ignore[return-value]

    @property
    def n_layers(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_kv_heads(self) -> int:
        return int(self.values.shape[1])

    @property
    def n_bins(self) -> int:
        return int(self.values.shape[2])

    def resolve_binning(self) -> PositionBinning:
        """The binning, defaulting to ``AbsoluteBinning(n_bins)``.

        Imported here rather than at module scope because ``hooks.binning`` pulls in
        torch, which SPEC section 2 forbids at import time outside ``hooks/``.
        """
        if self.binning is not None:
            return self.binning
        from rineya_psa.hooks.binning import AbsoluteBinning

        return AbsoluteBinning(self.n_bins)

    def with_values(self, values: np.ndarray, *, name: str | None = None) -> SignificanceTable:
        return SignificanceTable(
            values=values,
            binning=self.binning,
            name=self.name if name is None else name,
            source=self.source,
            meta=dict(self.meta),
        )

    def matches(self, arch: Arch) -> bool:
        return (self.n_layers, self.n_kv_heads) == (arch.num_layers, arch.num_kv_heads)

    def assert_matches(self, arch: Arch) -> None:
        if not self.matches(arch):
            raise ValueError(
                f"{self.name}: matrix is [{self.n_layers}, {self.n_kv_heads}, {self.n_bins}] "
                f"but the model is [{arch.num_layers}, {arch.num_kv_heads}, *]"
            )

    def describe(self) -> dict[str, Any]:
        v = self.values
        return {
            "name": self.name,
            "source": self.source,
            "shape": list(self.shape),
            "mean": float(v.mean()),
            "sd": float(v.std()),
            "abs_max": float(np.abs(v).max()) if v.size else 0.0,
            "binning": (
                None if self.binning is None else self.binning.describe()
            ),
            "meta": dict(self.meta),
        }


_PAYLOAD_ATTRS: tuple[str, ...] = ("values", "table", "matrix", "significance", "data")


def _payload_of(obj: Any) -> np.ndarray | None:
    if isinstance(obj, np.ndarray):
        return obj
    if hasattr(obj, "detach") and hasattr(obj, "cpu"):  # torch.Tensor, without importing torch
        # ``.float()`` first: numpy has no bfloat16 and would raise on a bf16 tensor.
        return np.asarray(obj.detach().to("cpu").float().numpy(), dtype=np.float64)
    if isinstance(obj, (list, tuple)):
        return np.asarray(obj, dtype=np.float64)
    for attr in _PAYLOAD_ATTRS:
        inner = getattr(obj, attr, None)
        if inner is None or callable(inner):
            continue
        payload = _payload_of(inner)
        if payload is not None:
            return payload
    return None


def coerce_significance(
    matrix: Any,
    *,
    binning: Any = None,
    name: str | None = None,
    n_kv_heads: int | None = None,
) -> SignificanceTable:
    """Accept whatever ``significance/`` hands over and return a :class:`SignificanceTable`.

    Recognised: a ``SignificanceTable``, a numpy array, a torch tensor, a nested
    sequence, or any object carrying the payload on ``.values`` / ``.table`` /
    ``.matrix`` / ``.significance`` / ``.data``. A 2-D ``[n_layers, n_bins]`` payload is
    broadcast across ``n_kv_heads`` when that is supplied, because a layer-level
    profile is a legitimate coarse matrix and refusing it would force every caller to
    tile by hand.
    """
    if isinstance(matrix, SignificanceTable):
        if binning is None and name is None:
            return matrix
        return SignificanceTable(
            values=matrix.values,
            binning=matrix.binning if binning is None else binning,
            name=matrix.name if name is None else name,
            source=matrix.source,
            meta=dict(matrix.meta),
        )

    payload = _payload_of(matrix)
    if payload is None:
        raise TypeError(
            f"cannot read a significance matrix out of {type(matrix).__name__}; expected an "
            f"array or an object with one of {list(_PAYLOAD_ATTRS)}"
        )
    values = np.asarray(payload, dtype=np.float64)
    if values.ndim == 2 and n_kv_heads is not None:
        values = np.repeat(values[:, None, :], int(n_kv_heads), axis=1)

    resolved_binning = binning
    if resolved_binning is None:
        candidate = getattr(matrix, "binning", None)
        if candidate is not None and hasattr(candidate, "n_bins"):
            resolved_binning = candidate

    return SignificanceTable(
        values=values,
        binning=resolved_binning,
        name=str(name or getattr(matrix, "name", None) or "significance"),
        source=str(getattr(matrix, "source", None) or type(matrix).__name__),
        meta=dict(getattr(matrix, "meta", {}) or {}),
    )


def broadcast_over_heads(values: Any, n_kv_heads: int) -> np.ndarray:
    """``[L, B] -> [L, H_kv, B]``. Explicit helper so the tiling is never implicit."""
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 2:
        raise ValueError(f"expected a 2-D [n_layers, n_bins] profile, got {v.shape}")
    return np.repeat(v[:, None, :], int(n_kv_heads), axis=1)


def standardize_significance(table: SignificanceTable) -> SignificanceTable:
    """Centre each ``(layer, kv_head)`` profile over bins; divide by the global peak.

    The result has zero mean along the bin axis and ``max |value| == 1`` over the whole
    table (unless the table is exactly uniform, in which case it is all zeros and every
    mode collapses to the identity — which is the honest answer for a matrix that says
    no position matters).
    """
    dev = table.values - table.values.mean(axis=-1, keepdims=True)
    peak = float(np.abs(dev).max()) if dev.size else 0.0
    scaled = np.zeros_like(dev) if peak <= 0.0 else dev / peak
    return table.with_values(scaled, name=f"{table.name}:standardized")


def head_gamma_from_matrix(table: SignificanceTable, arch: Arch) -> np.ndarray:
    """``[n_layers, n_q_heads]`` centred head weights for Mode 3.

    Mode 3 acts on the per-head output and has no bin axis, so the bin axis is reduced
    by its mean, the result is centred across the heads *within a layer* (so the mode is
    a reallocation between heads, not a global gain), and the whole thing is divided by
    its global peak so ``lam`` means the same dose it means in the other two modes.
    Expansion to query heads uses ``repeat`` semantics identical to
    ``Arch.q_heads_for_kv``: query head ``g`` reads KV head ``g // kv_group_size``.
    """
    table.assert_matches(arch)
    per_head = table.values.mean(axis=-1)  # [L, H_kv]
    dev = per_head - per_head.mean(axis=-1, keepdims=True)
    peak = float(np.abs(dev).max()) if dev.size else 0.0
    dev = dev / peak if peak > 0.0 else np.zeros_like(dev)
    return np.repeat(dev, arch.kv_group_size, axis=1)


# ---------------------------------------------------------------------------
# context construction
# ---------------------------------------------------------------------------


def build_psa_context(
    matrix: Any,
    mode: str,
    lam: float,
    layers: Sequence[int] | None = None,
    *,
    arch: Arch | None = None,
    binning: Any = None,
    standardize: bool = True,
    clamp_nonnegative: bool = True,
    capture: Any = None,
) -> PSAContext:
    """Build the SPEC 5.2 context that applies ``matrix`` through ``mode`` at strength ``lam``.

    ``arch`` is required for ``mode="head"`` (the gamma table is per *query* head, and
    the KV->Q expansion needs the group size) and is otherwise used only to validate
    shapes early. ``standardize=False`` feeds the raw matrix straight through, which is
    what a caller wants when the matrix is already a calibrated bias in logit units.
    """
    from rineya_psa.hooks.specs import (
        HeadScaleSpec,
        LogitBiasSpec,
        PSAContext,
        ValueScaleSpec,
    )

    canonical = canonical_mode(mode)
    lam_f = float(lam)
    table = coerce_significance(
        matrix, binning=binning, n_kv_heads=None if arch is None else arch.num_kv_heads
    )
    if arch is not None:
        table.assert_matches(arch)
    prepared = standardize_significance(table) if standardize else table
    bins = table.resolve_binning()
    layer_tuple = None if layers is None else tuple(sorted({int(x) for x in layers}))

    torch = _torch()
    if canonical == "logit":
        spec = LogitBiasSpec(
            table=torch.as_tensor(prepared.values, dtype=torch.float32),
            lam=lam_f,
            binning=bins,
            layers=layer_tuple,
        )
        return PSAContext(
            enabled=True, logit_bias=spec, position_bins=bins, capture=capture
        )

    if canonical == "value":
        scale = 1.0 + lam_f * prepared.values
        if clamp_nonnegative:
            # A negative value scale flips the sign of a head's contribution, which is
            # not an attenuation and is not what the matrix means. Only reachable at
            # lam > 1, so the identity at lam = 0 is untouched.
            scale = np.maximum(scale, 0.0)
        spec = ValueScaleSpec(
            table=torch.as_tensor(scale, dtype=torch.float32),
            binning=bins,
            layers=layer_tuple,
        )
        return PSAContext(
            enabled=True, value_scale=spec, position_bins=bins, capture=capture
        )

    if arch is None:
        raise ValueError("mode='head' needs arch: gamma is per query head, matrix is per KV head")
    gamma = 1.0 + lam_f * head_gamma_from_matrix(prepared, arch)
    if clamp_nonnegative:
        gamma = np.maximum(gamma, 0.0)
    spec = HeadScaleSpec(
        gamma=torch.as_tensor(gamma, dtype=torch.float32), layers=layer_tuple
    )
    return PSAContext(enabled=True, head_scale=spec, position_bins=bins, capture=capture)


def _torch() -> Any:
    import torch

    return torch


# ---------------------------------------------------------------------------
# sweeps
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PSAVariant:
    """One cell of a PSA sweep: the label an experiment row carries, plus its context."""

    label: str
    mode: str
    lam: float
    layers: tuple[int, ...] | None
    ctx: Any  # PSAContext; typed loosely so this module stays torch-free at import
    matrix_name: str = "significance"

    def describe(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "mode": self.mode,
            "lam": self.lam,
            "layers": None if self.layers is None else list(self.layers),
            "matrix": self.matrix_name,
            "is_identity": bool(self.ctx.is_identity()),
        }


def _variant(
    table: SignificanceTable,
    mode: str,
    lam: float,
    layers: Sequence[int] | None,
    kwargs: Mapping[str, Any],
    *,
    label_prefix: str = "psa",
) -> PSAVariant:
    canonical = canonical_mode(mode)
    layer_tuple = None if layers is None else tuple(sorted({int(x) for x in layers}))
    span = "all" if layer_tuple is None else f"L{layer_tuple[0]}-{layer_tuple[-1]}"
    return PSAVariant(
        label=f"{label_prefix}:{canonical}:lam{float(lam):g}:{span}",
        mode=canonical,
        lam=float(lam),
        layers=layer_tuple,
        ctx=build_psa_context(table, canonical, lam, layer_tuple, **kwargs),
        matrix_name=table.name,
    )


def psa_lambda_sweep(
    matrix: Any,
    mode: str,
    lams: Iterable[float] = PSA_LAMBDA_GRID,
    layers: Sequence[int] | None = None,
    **kwargs: Any,
) -> tuple[PSAVariant, ...]:
    """One mode, the whole dose grid. ``lam = 0`` is included so the curve is anchored."""
    table = coerce_significance(matrix, binning=kwargs.get("binning"))
    return tuple(_variant(table, mode, lam, layers, kwargs) for lam in lams)


def psa_mode_sweep(
    matrix: Any,
    lam: float,
    modes: Iterable[str] = PSA_MODES,
    layers: Sequence[int] | None = None,
    **kwargs: Any,
) -> tuple[PSAVariant, ...]:
    """One dose, all three modes. Answers 'which channel carries the effect?'."""
    table = coerce_significance(matrix, binning=kwargs.get("binning"))
    return tuple(_variant(table, mode, lam, layers, kwargs) for mode in modes)


def psa_sweep(
    matrix: Any,
    modes: Iterable[str] = PSA_MODES,
    lams: Iterable[float] = PSA_LAMBDA_GRID,
    layers: Sequence[int] | None = None,
    **kwargs: Any,
) -> tuple[PSAVariant, ...]:
    """The full mode x lambda grid, in a deterministic order (mode outer, lambda inner)."""
    table = coerce_significance(matrix, binning=kwargs.get("binning"))
    lam_list = list(lams)
    return tuple(
        _variant(table, mode, lam, layers, kwargs) for mode in modes for lam in lam_list
    )
