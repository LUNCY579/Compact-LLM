"""Position binnings (SPEC 5.1/5.3).

Two binnings exist and are deliberately kept apart:

``AbsoluteBinning``
    Depends on the *source* index only, ``b(j) = floor(j / T * n_bins)``. Because it
    is query-independent it can be folded into ``value`` before ``repeat_kv``
    (Mode 2) and into a ``[n_heads, 1, T]`` bias row (Mode 1) with no per-query cost.

``DistanceBinning``
    Depends on ``(i - j) / T`` and is therefore query-dependent. It only exists as a
    chunked accessor.

**Chunk memory contract.** ``chunk_bin_ids(kv_length, q_start, q_end)`` is the only
entry point that can allocate a two-dimensional tensor. Its cost is exactly
``(q_end - q_start) * kv_length * bytes_per_entry()`` where ``bytes_per_entry()`` is
0 for ``AbsoluteBinning`` (it returns a size-1 broadcast row, storage ``kv_length``)
and 12 for ``DistanceBinning`` (a transient float32 distance grid plus the int64
result). Callers must size chunks with ``max_chunk_rows`` or call
``assert_chunk_within_budget``; the lab's default of 256 query rows costs 12 MB at
T = 4096 and 25 MB at T = 8192, well under the 200 MB ceiling of SPEC section 0.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

#: Ceiling for a single chunked bin-id allocation, per SPEC section 0.
CHUNK_BUDGET_BYTES: int = 200 * 1024 * 1024

#: Query rows per chunk everywhere in the lab unless a caller overrides it.
DEFAULT_CHUNK_ROWS: int = 256


class PositionBinning(abc.ABC):
    """Maps token indices to a small number of position bins.

    Subclasses are frozen dataclasses so a binning can be hashed into a provenance
    record and shared between specs without aliasing hazards.
    """

    #: True when the bin of a source token depends on the query row.
    is_query_dependent: bool = False

    #: Number of bins. A field on ``AbsoluteBinning``, a derived property on
    #: ``DistanceBinning``; every subclass exposes it as a plain int attribute.
    n_bins: int

    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @abc.abstractmethod
    def describe(self) -> dict[str, Any]:
        """JSON-serialisable provenance record."""

    @abc.abstractmethod
    def bytes_per_entry(self) -> int:
        """Peak bytes per ``(query, source)`` cell of a ``chunk_bin_ids`` call."""

    @abc.abstractmethod
    def chunk_bin_ids(
        self,
        kv_length: int,
        q_start: int,
        q_end: int,
        *,
        q_offset: int = 0,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Bin ids for query rows ``[q_start, q_end)`` against all ``kv_length`` sources.

        Returns a ``[q_rows, kv_length]`` int64 tensor, or a ``[1, kv_length]``
        broadcast row when the binning is query-independent. ``q_offset`` is the
        absolute position of query row 0 (``kv_length - q_length`` when a KV cache is
        in play), so the distance ``i - j`` stays correct during decoding.
        """

    # -- shared helpers -------------------------------------------------------

    def max_chunk_rows(self, kv_length: int, budget_bytes: int = CHUNK_BUDGET_BYTES) -> int:
        per_row = self.bytes_per_entry() * max(int(kv_length), 1)
        if per_row <= 0:
            return int(budget_bytes)  # query-independent: no per-row cost at all
        return max(1, int(budget_bytes) // per_row)

    def assert_chunk_within_budget(
        self, kv_length: int, n_rows: int, budget_bytes: int = CHUNK_BUDGET_BYTES
    ) -> None:
        cost = self.bytes_per_entry() * max(int(kv_length), 1) * int(n_rows)
        if cost > budget_bytes:
            raise ValueError(
                f"{self.name}: a chunk of {n_rows} query rows at T={kv_length} would "
                f"allocate {cost / 2**20:.1f} MB, over the {budget_bytes / 2**20:.0f} MB "
                f"budget; use at most {self.max_chunk_rows(kv_length, budget_bytes)} rows"
            )

    def gather_chunk(
        self,
        table: torch.Tensor,
        kv_length: int,
        q_start: int,
        q_end: int,
        *,
        q_offset: int = 0,
        valid_mask: torch.Tensor | None = None,
        fill: float = 0.0,
    ) -> torch.Tensor:
        """Expand ``table[..., n_bins]`` to ``[..., q_rows_or_1, kv_length]``.

        ``valid_mask`` (``[kv_length]`` or ``[batch, kv_length]``) marks real tokens;
        padding sources receive ``fill`` (0.0 for an additive bias, 1.0 for a
        multiplicative scale) so a padded batch cannot be nudged by an intervention
        aimed at content positions.
        """
        if table.shape[-1] != self.n_bins:
            raise ValueError(
                f"table last dim {table.shape[-1]} != n_bins {self.n_bins} for {self.name}"
            )
        ids = self.chunk_bin_ids(
            kv_length, q_start, q_end, q_offset=q_offset, device=table.device
        )
        out = table[..., ids]
        return _apply_valid_fill(out, valid_mask, fill)

    def gather_source(
        self,
        table: torch.Tensor,
        kv_length: int,
        *,
        valid_mask: torch.Tensor | None = None,
        fill: float = 0.0,
    ) -> torch.Tensor:
        """Expand ``table[..., n_bins]`` to ``[..., kv_length]``. Absolute binnings only.

        This is the accessor Mode 2 uses before ``repeat_kv``; a query-dependent
        binning has no source-only answer and raises.
        """
        if self.is_query_dependent:
            raise ValueError(
                f"{self.name} is query-dependent; it has no source-only bin assignment. "
                "Use gather_chunk (Mode 2 then folds into the post-softmax weights)."
            )
        if table.shape[-1] != self.n_bins:
            raise ValueError(
                f"table last dim {table.shape[-1]} != n_bins {self.n_bins} for {self.name}"
            )
        ids = self.bin_ids(kv_length, device=table.device)
        out = table[..., ids]
        return _apply_valid_fill(out, valid_mask, fill)

    def one_hot(
        self, kv_length: int, *, device: torch.device | str | None = None, dtype=torch.float32
    ) -> torch.Tensor:
        """``[kv_length, n_bins]`` indicator matrix. Absolute binnings only.

        Capture reduces ``probs @ one_hot`` rather than scattering, which keeps the
        per-chunk work a single matmul.
        """
        if self.is_query_dependent:
            raise ValueError(f"{self.name} is query-dependent; one_hot is undefined")
        ids = self.bin_ids(kv_length, device=device)
        out = torch.zeros((kv_length, self.n_bins), device=ids.device, dtype=dtype)
        out.scatter_(1, ids.view(-1, 1), 1.0)
        return out

    def bin_ids(
        self, kv_length: int, *, device: torch.device | str | None = None
    ) -> torch.Tensor:
        """``[kv_length]`` int64 source-bin ids. Absolute binnings only."""
        raise ValueError(f"{self.name} is query-dependent; bin_ids(T) is undefined")

    def bin_ids_numpy(self, kv_length: int) -> np.ndarray:
        """``[kv_length]`` int64 source-bin ids for torch-free CPU analysis."""
        raise ValueError(f"{self.name} is query-dependent; bin_ids_numpy(T) is undefined")


def _apply_valid_fill(
    out: torch.Tensor, valid_mask: torch.Tensor | None, fill: float
) -> torch.Tensor:
    if valid_mask is None:
        return out
    v = valid_mask.to(device=out.device, dtype=torch.bool)
    fill_t = torch.as_tensor(fill, device=out.device, dtype=out.dtype)
    if v.ndim == 1:
        if v.shape[0] != out.shape[-1]:
            raise ValueError(f"valid_mask length {v.shape[0]} != kv length {out.shape[-1]}")
        return torch.where(v, out, fill_t)
    if v.ndim != 2:
        raise ValueError(f"valid_mask must be 1D [T] or 2D [B, T]; got {tuple(v.shape)}")
    if v.shape[-1] != out.shape[-1]:
        raise ValueError(f"valid_mask length {v.shape[-1]} != kv length {out.shape[-1]}")
    # [B, T] -> [B, 1, ..., 1, T] so it broadcasts against a leading batch axis.
    v = v.view(v.shape[0], *([1] * (out.ndim - 1)), v.shape[-1])
    return torch.where(v, out.unsqueeze(0), fill_t)


@dataclass(frozen=True)
class AbsoluteBinning(PositionBinning):
    """``b(j) = clamp(floor(j / T * n_bins), 0, n_bins - 1)``.

    Computed in integer arithmetic (``j * n_bins // T``) so bin edges are exact and
    reproducible across devices; a float multiply would put ``j = T // 2`` on the
    wrong side of the midpoint for some ``T``.
    """

    n_bins: int = 16

    is_query_dependent = False

    def __post_init__(self) -> None:
        if int(self.n_bins) < 1:
            raise ValueError(f"n_bins must be >= 1, got {self.n_bins}")
        object.__setattr__(self, "n_bins", int(self.n_bins))

    @property
    def name(self) -> str:
        return f"absolute{self.n_bins}"

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "absolute",
            "name": self.name,
            "n_bins": self.n_bins,
            "domain": "source_index_over_sequence_length",
        }

    def bytes_per_entry(self) -> int:
        return 0  # broadcast row: storage is O(T), not O(q * T)

    def bin_ids(
        self, kv_length: int, *, device: torch.device | str | None = None
    ) -> torch.Tensor:
        T = int(kv_length)
        if T < 1:
            raise ValueError(f"kv_length must be >= 1, got {kv_length}")
        j = torch.arange(T, device=device, dtype=torch.int64)
        return torch.clamp((j * self.n_bins) // T, 0, self.n_bins - 1)

    def bin_ids_numpy(self, kv_length: int) -> np.ndarray:
        T = int(kv_length)
        if T < 1:
            raise ValueError(f"kv_length must be >= 1, got {kv_length}")
        j = np.arange(T, dtype=np.int64)
        return np.clip((j * self.n_bins) // T, 0, self.n_bins - 1)

    def chunk_bin_ids(
        self,
        kv_length: int,
        q_start: int,
        q_end: int,
        *,
        q_offset: int = 0,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        if q_end < q_start:
            raise ValueError(f"empty chunk [{q_start}, {q_end})")
        # Shape [1, T]: broadcasts over the query axis without allocating q * T.
        return self.bin_ids(kv_length, device=device).unsqueeze(0)


@dataclass(frozen=True)
class DistanceBinning(PositionBinning):
    """Bins the normalised look-back distance ``d = (i - j) / T``.

    ``edges`` are strictly increasing cut points; bin ``k`` covers
    ``[edges[k-1], edges[k])`` with ``edges[-1] = +inf`` and ``edges[-2..] = -inf``,
    i.e. ``n_bins = len(edges) + 1``.

    The masked upper triangle (``j > i``, hence ``d < 0``) lands in bin 0 by
    construction. That is defined and harmless: those cells are removed by the causal
    mask before softmax, and Mode 1 only adds bias where the mask is finite, so the
    bin-0 entry of the table never leaks into the output through the masked region.
    """

    edges: tuple[float, ...]

    is_query_dependent = True

    def __post_init__(self) -> None:
        e = tuple(float(x) for x in self.edges)
        if not e:
            raise ValueError("DistanceBinning needs at least one edge")
        if any(b <= a for a, b in zip(e, e[1:], strict=False)):
            raise ValueError(f"edges must be strictly increasing, got {e}")
        object.__setattr__(self, "edges", e)

    @classmethod
    def uniform(cls, n_bins: int) -> DistanceBinning:
        """Even cuts of ``d`` over ``[0, 1]``; matches ``floor(d * n_bins)`` on ``[0, 1)``."""
        if int(n_bins) < 2:
            raise ValueError("uniform DistanceBinning needs n_bins >= 2")
        n = int(n_bins)
        return cls(tuple(k / n for k in range(1, n)))

    @property
    def n_bins(self) -> int:
        return len(self.edges) + 1

    @property
    def name(self) -> str:
        return f"distance{self.n_bins}"

    def describe(self) -> dict[str, Any]:
        return {
            "kind": "distance",
            "name": self.name,
            "n_bins": self.n_bins,
            "edges": list(self.edges),
            "domain": "(query_index - source_index) / sequence_length",
        }

    def bytes_per_entry(self) -> int:
        # float32 distance grid (4) + int64 bucketize result (8), both live at once.
        return 12

    def chunk_bin_ids(
        self,
        kv_length: int,
        q_start: int,
        q_end: int,
        *,
        q_offset: int = 0,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        T = int(kv_length)
        if T < 1:
            raise ValueError(f"kv_length must be >= 1, got {kv_length}")
        if q_end < q_start:
            raise ValueError(f"empty chunk [{q_start}, {q_end})")
        self.assert_chunk_within_budget(T, q_end - q_start)
        i = torch.arange(q_start, q_end, device=device, dtype=torch.float32) + float(q_offset)
        j = torch.arange(T, device=device, dtype=torch.float32)
        d = (i[:, None] - j[None, :]) / float(T)
        bounds = torch.tensor(self.edges, device=d.device, dtype=d.dtype)
        # right=True gives bin k for edges[k-1] <= d < edges[k], matching floor() on a
        # uniform grid and sending every negative (masked) distance to bin 0.
        return torch.bucketize(d, bounds, right=True)

    def chunk_bin_ids_numpy(
        self, kv_length: int, q_start: int, q_end: int, *, q_offset: int = 0
    ) -> np.ndarray:
        T = int(kv_length)
        i = np.arange(q_start, q_end, dtype=np.float64) + float(q_offset)
        j = np.arange(T, dtype=np.float64)
        d = (i[:, None] - j[None, :]) / float(T)
        return np.searchsorted(np.asarray(self.edges, dtype=np.float64), d, side="right")


def binning_from_describe(record: dict[str, Any]) -> PositionBinning:
    """Rebuild a binning from its provenance record."""
    kind = record.get("kind")
    if kind == "absolute":
        return AbsoluteBinning(int(record["n_bins"]))
    if kind == "distance":
        return DistanceBinning(tuple(float(x) for x in record["edges"]))
    raise ValueError(f"unknown binning kind {kind!r}")
