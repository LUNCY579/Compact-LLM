"""Intervention and capture specs (SPEC 5.2 / 5.3 / 5.4).

Every spec is frozen so a run configuration cannot drift halfway through a sweep.
They carry ``eq=False`` because their payload is a ``torch.Tensor``: dataclass
equality would call ``Tensor.__eq__`` and return a tensor, which silently poisons
``==`` and ``in`` checks. Identity semantics are what the lab actually needs.

DEVIATION: SPEC 5.2 lists six fields on ``PSAContext``. Three runtime-only fields
are appended here — ``valid_mask``, ``evidence_spans`` and ``captured`` — because
padding and gold-span bookkeeping change per input while the specs do not, and
SPEC 5.4 requires ``ctx.captured`` to exist. All six spec fields keep their exact
names and defaults, so any code written against SPEC 5.2 constructs a valid context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from rineya_psa.hooks.binning import (
    DEFAULT_CHUNK_ROWS,
    AbsoluteBinning,
    PositionBinning,
)
from rineya_psa.types import Arch

#: Reductions ``CaptureSpec.what`` may request (SPEC 5.4).
CAPTURE_KINDS: tuple[str, ...] = (
    "attn_by_bin",
    "attn_entropy",
    "gold_span_attn",
    "qk_logit_mean",
    "head_out_norm",
)

#: Default attenuation for the causal-significance probe of SPEC 5.3.
DEFAULT_ATTENUATION_EPS: float = 0.25


def _as_table(values: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    t = torch.as_tensor(values, dtype=dtype)
    return t.detach().clone().contiguous()


def _normalise_layers(layers: Any) -> tuple[int, ...] | None:
    if layers is None:
        return None
    return tuple(sorted({int(x) for x in layers}))


@dataclass(frozen=True, eq=False)
class LogitBiasSpec:
    """Mode 1: ``z'[g,i,j] = z[g,i,j] + lam * S[l, kv(g), b(j)]``."""

    table: torch.Tensor  # [n_layers, n_kv_heads, n_bins]
    lam: float
    binning: PositionBinning
    layers: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        t = _as_table(self.table)
        if t.ndim != 3:
            raise ValueError(f"LogitBiasSpec.table must be 3D [L, H_kv, B], got {tuple(t.shape)}")
        if t.shape[-1] != self.binning.n_bins:
            raise ValueError(
                f"table has {t.shape[-1]} bins but {self.binning.name} has {self.binning.n_bins}"
            )
        object.__setattr__(self, "table", t)
        object.__setattr__(self, "lam", float(self.lam))
        object.__setattr__(self, "layers", _normalise_layers(self.layers))

    @property
    def n_layers(self) -> int:
        return int(self.table.shape[0])

    @property
    def n_kv_heads(self) -> int:
        return int(self.table.shape[1])

    @property
    def n_bins(self) -> int:
        return int(self.table.shape[2])

    def is_identity(self) -> bool:
        """True only when the spec provably cannot move a single logit."""
        if self.lam == 0.0:
            return True
        if self.layers is not None and not self.layers:
            return True
        active = self.table if self.layers is None else self.table[list(self.layers)]
        return bool(torch.count_nonzero(active) == 0)

    def applies_to(self, layer: int) -> bool:
        return self.layers is None or int(layer) in self.layers

    def describe(self) -> dict[str, Any]:
        return {
            "mode": "logit_bias",
            "lam": self.lam,
            "shape": list(self.table.shape),
            "layers": None if self.layers is None else list(self.layers),
            "binning": self.binning.describe(),
            "abs_max": float(self.table.abs().max()) if self.table.numel() else 0.0,
            "is_identity": self.is_identity(),
        }

    @classmethod
    def zeros(
        cls,
        arch: Arch,
        n_bins: int,
        binning: PositionBinning,
        *,
        lam: float = 1.0,
        layers: Any = None,
    ) -> LogitBiasSpec:
        if int(n_bins) != binning.n_bins:
            raise ValueError(f"n_bins {n_bins} != binning.n_bins {binning.n_bins}")
        table = torch.zeros((arch.num_layers, arch.num_kv_heads, int(n_bins)), dtype=torch.float32)
        return cls(table=table, lam=float(lam), binning=binning, layers=layers)


@dataclass(frozen=True, eq=False)
class ValueScaleSpec:
    """Mode 2: ``V'[h,j,:] = S[l,h,b(j)] * V[h,j,:]``, applied before ``repeat_kv``."""

    table: torch.Tensor  # [n_layers, n_kv_heads, n_bins], multiplicative, 1.0 = no-op
    binning: PositionBinning
    layers: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        t = _as_table(self.table)
        if t.ndim != 3:
            raise ValueError(f"ValueScaleSpec.table must be 3D [L, H_kv, B], got {tuple(t.shape)}")
        if t.shape[-1] != self.binning.n_bins:
            raise ValueError(
                f"table has {t.shape[-1]} bins but {self.binning.name} has {self.binning.n_bins}"
            )
        object.__setattr__(self, "table", t)
        object.__setattr__(self, "layers", _normalise_layers(self.layers))

    @property
    def n_layers(self) -> int:
        return int(self.table.shape[0])

    @property
    def n_kv_heads(self) -> int:
        return int(self.table.shape[1])

    @property
    def n_bins(self) -> int:
        return int(self.table.shape[2])

    def is_identity(self) -> bool:
        if self.layers is not None and not self.layers:
            return True
        active = self.table if self.layers is None else self.table[list(self.layers)]
        return bool(torch.all(active == 1.0))

    def applies_to(self, layer: int) -> bool:
        return self.layers is None or int(layer) in self.layers

    def describe(self) -> dict[str, Any]:
        return {
            "mode": "value_scale",
            "shape": list(self.table.shape),
            "layers": None if self.layers is None else list(self.layers),
            "binning": self.binning.describe(),
            "min": float(self.table.min()) if self.table.numel() else 1.0,
            "max": float(self.table.max()) if self.table.numel() else 1.0,
            "is_identity": self.is_identity(),
        }

    @classmethod
    def ones(
        cls, arch: Arch, n_bins: int, binning: PositionBinning, *, layers: Any = None
    ) -> ValueScaleSpec:
        if int(n_bins) != binning.n_bins:
            raise ValueError(f"n_bins {n_bins} != binning.n_bins {binning.n_bins}")
        table = torch.ones((arch.num_layers, arch.num_kv_heads, int(n_bins)), dtype=torch.float32)
        return cls(table=table, binning=binning, layers=layers)

    @classmethod
    def attenuate_cell(
        cls,
        arch: Arch,
        n_bins: int,
        binning: PositionBinning,
        layer: int,
        kv_head: int,
        bin_index: int,
        eps: float = DEFAULT_ATTENUATION_EPS,
    ) -> ValueScaleSpec:
        """The causal-significance probe of SPEC 5.3: knock down one ``(l, h, b)`` cell.

        ``S = 1 - eps * 1[bin == b]`` on that one layer/KV-head row. ``layers`` is
        pinned to the single layer so the registry skips the other 27 entirely.
        """
        spec = cls.ones(arch, n_bins, binning)
        if not 0 <= int(layer) < arch.num_layers:
            raise IndexError(f"layer {layer} out of range [0,{arch.num_layers})")
        if not 0 <= int(kv_head) < arch.num_kv_heads:
            raise IndexError(f"kv_head {kv_head} out of range [0,{arch.num_kv_heads})")
        if not 0 <= int(bin_index) < int(n_bins):
            raise IndexError(f"bin {bin_index} out of range [0,{n_bins})")
        table = spec.table.clone()
        table[int(layer), int(kv_head), int(bin_index)] = 1.0 - float(eps)
        return cls(table=table, binning=binning, layers=(int(layer),))


@dataclass(frozen=True, eq=False)
class HeadScaleSpec:
    """Mode 3: ``O'[g] = gamma[l,g] * O[g]`` on the per-head output before ``o_proj``."""

    gamma: torch.Tensor  # [n_layers, n_q_heads]
    layers: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        g = _as_table(self.gamma)
        if g.ndim != 2:
            raise ValueError(f"HeadScaleSpec.gamma must be 2D [L, H_q], got {tuple(g.shape)}")
        object.__setattr__(self, "gamma", g)
        object.__setattr__(self, "layers", _normalise_layers(self.layers))

    @property
    def n_layers(self) -> int:
        return int(self.gamma.shape[0])

    @property
    def n_q_heads(self) -> int:
        return int(self.gamma.shape[1])

    def is_identity(self) -> bool:
        if self.layers is not None and not self.layers:
            return True
        active = self.gamma if self.layers is None else self.gamma[list(self.layers)]
        return bool(torch.all(active == 1.0))

    def applies_to(self, layer: int) -> bool:
        return self.layers is None or int(layer) in self.layers

    def describe(self) -> dict[str, Any]:
        return {
            "mode": "head_scale",
            "shape": list(self.gamma.shape),
            "layers": None if self.layers is None else list(self.layers),
            "min": float(self.gamma.min()) if self.gamma.numel() else 1.0,
            "max": float(self.gamma.max()) if self.gamma.numel() else 1.0,
            "is_identity": self.is_identity(),
        }

    @classmethod
    def ones(cls, arch: Arch, *, layers: Any = None) -> HeadScaleSpec:
        return cls(gamma=torch.ones((arch.num_layers, arch.num_q_heads), dtype=torch.float32),
                   layers=layers)

    @classmethod
    def zero_head(cls, arch: Arch, layer: int, q_head: int) -> HeadScaleSpec:
        """Ablate one query head. Convenience for head-importance sweeps."""
        spec = cls.ones(arch)
        gamma = spec.gamma.clone()
        gamma[int(layer), int(q_head)] = 0.0
        return cls(gamma=gamma, layers=(int(layer),))


@dataclass(frozen=True, eq=False)
class CaptureSpec:
    """SPEC 5.4. ``layers=None`` means every layer.

    DEVIATION: ``binning`` and ``chunk`` are explicit fields rather than implicit
    constants, so a capture run records exactly which bin grid produced
    ``attn_by_bin`` / ``qk_logit_mean``. The binning must be query-independent —
    those two reductions are defined per *absolute source bin*.
    """

    layers: tuple[int, ...] | None = None
    what: tuple[str, ...] = CAPTURE_KINDS
    chunk: int = DEFAULT_CHUNK_ROWS
    binning: PositionBinning | None = None

    def __post_init__(self) -> None:
        what = tuple(dict.fromkeys(str(w) for w in self.what))
        unknown = [w for w in what if w not in CAPTURE_KINDS]
        if unknown:
            raise ValueError(f"unknown capture kinds {unknown}; known: {list(CAPTURE_KINDS)}")
        if int(self.chunk) < 1:
            raise ValueError(f"chunk must be >= 1, got {self.chunk}")
        if self.binning is not None and self.binning.is_query_dependent:
            raise ValueError(
                "capture bins absolute source positions; "
                f"{self.binning.name} is query-dependent"
            )
        object.__setattr__(self, "what", what)
        object.__setattr__(self, "chunk", int(self.chunk))
        object.__setattr__(self, "layers", _normalise_layers(self.layers))

    def is_active(self) -> bool:
        return bool(self.what) and (self.layers is None or bool(self.layers))

    def applies_to(self, layer: int) -> bool:
        return self.layers is None or int(layer) in self.layers

    def wants(self, kind: str) -> bool:
        return kind in self.what

    def describe(self) -> dict[str, Any]:
        return {
            "layers": None if self.layers is None else list(self.layers),
            "what": list(self.what),
            "chunk": self.chunk,
            "binning": None if self.binning is None else self.binning.describe(),
        }


@dataclass
class PSAContext:
    """Runtime state read by ``psa_attention_forward`` (SPEC 5.2)."""

    enabled: bool = False
    logit_bias: LogitBiasSpec | None = None
    value_scale: ValueScaleSpec | None = None
    head_scale: HeadScaleSpec | None = None
    capture: CaptureSpec | None = None
    position_bins: PositionBinning | None = None

    # -- runtime extensions (see module docstring) ---------------------------
    #: ``[T]`` or ``[batch, T]`` bool; False marks padding. Padding source tokens
    #: are excluded from interventions, padding query rows from capture averages.
    valid_mask: torch.Tensor | None = None
    #: ``[start, end)`` token ranges of the gold evidence, for ``gold_span_attn``.
    evidence_spans: tuple[tuple[int, int], ...] = ()
    #: Finalised reductions, keyed ``(layer, tensor_name)`` (SPEC 5.4).
    captured: dict[tuple[int, str], np.ndarray] = field(default_factory=dict)

    #: Live accumulator, owned by ``hooks.capture``. Not part of the public surface.
    _acc: Any = field(default=None, repr=False, compare=False)

    def is_identity(self) -> bool:
        """True when no intervention can change a logit (SPEC 5.2's invariant key).

        Capture is deliberately *not* consulted: it reads the attention pattern and
        never writes to it, so a capture-only context is still an identity on the
        model output. Use ``is_noop`` for the dispatch fast path.
        """
        if not self.enabled:
            return True
        return all(
            spec is None or spec.is_identity()
            for spec in (self.logit_bias, self.value_scale, self.head_scale)
        )

    def capture_active(self) -> bool:
        return bool(self.enabled and self.capture is not None and self.capture.is_active())

    def is_noop(self) -> bool:
        """True when the custom kernel has nothing to do and can delegate to SDPA."""
        return self.is_identity() and not self.capture_active()

    def active_modes(self) -> tuple[str, ...]:
        if not self.enabled:
            return ()
        modes = []
        if self.logit_bias is not None and not self.logit_bias.is_identity():
            modes.append("logit_bias")
        if self.value_scale is not None and not self.value_scale.is_identity():
            modes.append("value_scale")
        if self.head_scale is not None and not self.head_scale.is_identity():
            modes.append("head_scale")
        if self.capture_active():
            modes.append("capture")
        return tuple(modes)

    def capture_binning(self) -> PositionBinning:
        """The absolute grid used by ``attn_by_bin`` / ``qk_logit_mean``."""
        if self.capture is not None and self.capture.binning is not None:
            return self.capture.binning
        if self.position_bins is not None and not self.position_bins.is_query_dependent:
            return self.position_bins
        return AbsoluteBinning(16)

    def reset_capture(self) -> None:
        self.captured = {}
        self._acc = None

    def finalize_capture(self) -> dict[tuple[int, str], np.ndarray]:
        """Turn running sums into means and publish them on ``captured``.

        Imported lazily: ``capture`` depends on ``specs`` for ``CaptureSpec``, so a
        module-level import here would close the cycle.
        """
        from rineya_psa.hooks.capture import finalize_captures

        return finalize_captures(self)

    def validate_against(self, arch: Arch) -> None:
        """Fail loudly at session entry rather than mid-forward with a shape error."""
        for spec, kind in ((self.logit_bias, "logit_bias"), (self.value_scale, "value_scale")):
            if spec is None:
                continue
            want = (arch.num_layers, arch.num_kv_heads)
            got = (spec.n_layers, spec.n_kv_heads)
            if got != want:
                raise ValueError(f"{kind} table is {got} but the model is {want}")
            if spec.layers is not None and any(
                not 0 <= i < arch.num_layers for i in spec.layers
            ):
                raise ValueError(f"{kind}.layers out of range for {arch.num_layers} layers")
        if self.head_scale is not None:
            want = (arch.num_layers, arch.num_q_heads)
            got = (self.head_scale.n_layers, self.head_scale.n_q_heads)
            if got != want:
                raise ValueError(f"head_scale gamma is {got} but the model is {want}")
        if (
            self.capture is not None
            and self.capture.layers is not None
            and any(not 0 <= i < arch.num_layers for i in self.capture.layers)
        ):
            raise ValueError(f"capture.layers out of range for {arch.num_layers} layers")

    def describe(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "active_modes": list(self.active_modes()),
            "is_identity": self.is_identity(),
            "logit_bias": None if self.logit_bias is None else self.logit_bias.describe(),
            "value_scale": None if self.value_scale is None else self.value_scale.describe(),
            "head_scale": None if self.head_scale is None else self.head_scale.describe(),
            "capture": None if self.capture is None else self.capture.describe(),
            "position_bins": None if self.position_bins is None else self.position_bins.describe(),
            "n_evidence_spans": len(self.evidence_spans),
        }
