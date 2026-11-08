"""Weight-space distances between two checkpoints, streamed in float64.

Deltas are accumulated, never materialised: a full ``embed_tokens`` delta in
float64 is 1.2 GB, and there are four of them. Everything below works on row
chunks and keeps only scalar accumulators, so peak RSS is a few tens of MB
regardless of checkpoint size. All arithmetic is float64 even when both stores are
bfloat16, because a bf16 accumulation of 155 M squared terms loses most of its
significant digits.

Head decomposition. Torch stores ``nn.Linear.weight`` as ``[out_features,
in_features]``, so for Qwen3-0.6B:

    q_proj  [n_q*head_dim, hidden]   -> 16 query heads of 128 **rows**
    k_proj  [n_kv*head_dim, hidden]  ->  8 KV heads of 128 **rows**
    v_proj  [n_kv*head_dim, hidden]  ->  8 KV heads of 128 **rows**
    o_proj  [hidden, n_q*head_dim]   -> 16 query heads of 128 **columns**

That orientation is *measured*, not assumed: ``resolve_orientation`` finds the
axis whose length equals ``n_heads * head_dim`` and requires q_proj and o_proj to
agree on which side of the matrix the head axis sits. k_proj and v_proj are square
for this architecture (``n_kv * head_dim == hidden == 1024``) so their own shapes
carry no information; they inherit the convention that q_proj and o_proj pinned
down, and the inference is recorded as ``evidence="inherited"``.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rineya_psa.model_audit.hashing import TensorLocation, decode_to_float64
from rineya_psa.types import Arch

# The families the brief aggregates by, in report order.
FAMILIES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "q_norm",
    "k_norm",
    "gate_proj",
    "up_proj",
    "down_proj",
    "input_layernorm",
    "post_attention_layernorm",
    "embed_tokens",
    "norm",
)

# DEVIATION (documented in __init__): two of the four cores materialise the tied
# ``lm_head.weight``. It is not one of the brief's 13 families, so it gets its own
# label rather than being silently folded into ``embed_tokens`` or dropped.
EXTRA_FAMILIES: tuple[str, ...] = ("lm_head",)
ALL_FAMILIES: tuple[str, ...] = FAMILIES + EXTRA_FAMILIES + ("other",)

# Sentinel layer index for tensors that live outside the decoder stack.
NO_LAYER: int = -1

# Default relative tolerance for "changed by more than storage noise". A bf16 store
# has ~3 decimal digits, so 1e-3 relative is roughly one ulp.
DEFAULT_REL_TOL: float = 1e-3

# ~64 MiB of float64 per side per chunk.
DEFAULT_CHUNK_BYTES: int = 64 << 20

_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")

# family -> (head space, axis under the nn.Linear [out, in] convention)
_HEAD_FAMILIES: dict[str, tuple[str, int]] = {
    "q_proj": ("q", 0),
    "k_proj": ("kv", 0),
    "v_proj": ("kv", 0),
    "o_proj": ("q", 1),
}
HEAD_KINDS: tuple[str, ...] = tuple(_HEAD_FAMILIES)


# --------------------------------------------------------------------------- #
# tensor naming
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TensorKey:
    """Where a tensor sits in the (layer, family) grid."""

    name: str
    layer: int
    family: str
    param: str  # "weight" | "bias" | ""

    @property
    def is_layer_local(self) -> bool:
        return self.layer != NO_LAYER


def parse_tensor_name(name: str) -> TensorKey:
    """Split a state-dict key into ``(layer, family, param)``.

    The family is the leaf module name, which for Qwen3 is exactly the brief's
    vocabulary (``q_proj`` ... ``post_attention_layernorm``) plus the two
    stack-level tensors (``embed_tokens``, ``norm``).
    """
    m = _LAYER_RE.match(name)
    layer, rest = (int(m.group(1)), m.group(2)) if m else (NO_LAYER, name)

    param = ""
    for candidate in ("weight", "bias"):
        if rest.endswith("." + candidate):
            param = candidate
            rest = rest[: -(len(candidate) + 1)]
            break

    leaf = rest.rsplit(".", 1)[-1]
    family = leaf if leaf in (*FAMILIES, *EXTRA_FAMILIES) else "other"
    return TensorKey(name=name, layer=layer, family=family, param=param)


def family_sort_key(family: str) -> int:
    try:
        return ALL_FAMILIES.index(family)
    except ValueError:
        return len(ALL_FAMILIES)


# --------------------------------------------------------------------------- #
# head layout
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class HeadLayout:
    """How to cut one projection matrix into per-head blocks."""

    family: str
    axis: int  # 0 -> blocks of rows, 1 -> blocks of columns
    n_heads: int
    head_dim: int
    space: str  # "q" | "kv"
    evidence: str  # "measured" | "inherited"

    @property
    def block(self) -> int:
        return self.head_dim


@dataclass(frozen=True)
class OrientationReport:
    """Result of measuring which axis of a projection carries the heads."""

    stored_out_first: bool
    evidence: tuple[str, ...]
    ambiguous: tuple[str, ...]
    observed: dict[str, list[int]]
    note: str

    def to_json(self) -> dict[str, Any]:
        return {
            "stored_out_first": self.stored_out_first,
            "evidence": list(self.evidence),
            "ambiguous": list(self.ambiguous),
            "observed_layer0_shapes": self.observed,
            "note": self.note,
        }


def resolve_orientation(
    shapes: Mapping[str, Sequence[int]], arch: Arch, *, layer: int = 0
) -> OrientationReport:
    """Measure the head axis from the checkpoint's own shapes.

    Returns ``stored_out_first=True`` when weights follow torch's
    ``[out_features, in_features]`` layout. Raises if q_proj and o_proj disagree,
    because then no single convention explains the file and every per-head number
    downstream would be silently wrong.
    """
    observed: dict[str, list[int]] = {}
    votes: dict[str, bool] = {}
    ambiguous: list[str] = []

    for family, (space, _conv_axis) in _HEAD_FAMILIES.items():
        name = f"model.layers.{layer}.self_attn.{family}.weight"
        shape = shapes.get(name)
        if shape is None or len(shape) != 2:
            continue
        shape = tuple(int(x) for x in shape)
        observed[family] = list(shape)
        width = (arch.num_q_heads if space == "q" else arch.num_kv_heads) * arch.head_dim
        hits = [ax for ax in (0, 1) if shape[ax] == width]
        if len(hits) != 1:
            ambiguous.append(family)
            continue
        head_axis = hits[0]
        # o_proj carries the head axis on its *input* side, so its vote is inverted.
        votes[family] = (head_axis == 0) if family != "o_proj" else (head_axis == 1)

    if not votes:
        raise ValueError(
            "cannot determine projection orientation: no unambiguous q/k/v/o_proj "
            f"shape at layer {layer} (observed {observed})"
        )
    distinct = set(votes.values())
    if len(distinct) > 1:
        raise ValueError(
            "projection shapes are mutually inconsistent; q_proj and o_proj disagree "
            f"about the head axis: {votes} from {observed}"
        )
    stored_out_first = distinct.pop()
    return OrientationReport(
        stored_out_first=stored_out_first,
        evidence=tuple(sorted(votes)),
        ambiguous=tuple(sorted(ambiguous)),
        observed=observed,
        note=(
            "measured from shapes; square projections (n_kv*head_dim == hidden_size) "
            "inherit the convention pinned by the unambiguous ones"
        ),
    )


def head_layout(
    family: str,
    shape: Sequence[int],
    arch: Arch,
    *,
    stored_out_first: bool = True,
) -> HeadLayout | None:
    """Per-head cut for one projection, or ``None`` if the family has no heads."""
    if family not in _HEAD_FAMILIES or len(shape) != 2:
        return None
    space, conv_axis = _HEAD_FAMILIES[family]
    n_heads = arch.num_q_heads if space == "q" else arch.num_kv_heads
    width = n_heads * arch.head_dim
    shape = tuple(int(x) for x in shape)

    axis = conv_axis if stored_out_first else (1 - conv_axis)
    hits = [ax for ax in (0, 1) if shape[ax] == width]
    if not hits:
        raise ValueError(
            f"{family}: neither axis of {shape} equals n_heads*head_dim={width}; "
            "the checkpoint does not match the declared architecture"
        )
    evidence = "measured" if len(hits) == 1 else "inherited"
    if len(hits) == 1 and hits[0] != axis:
        raise ValueError(
            f"{family}: shape {shape} puts the head axis at {hits[0]} but the "
            f"orientation convention says {axis}"
        )
    return HeadLayout(
        family=family,
        axis=axis,
        n_heads=n_heads,
        head_dim=arch.head_dim,
        space=space,
        evidence=evidence,
    )


# --------------------------------------------------------------------------- #
# accumulators
# --------------------------------------------------------------------------- #


@dataclass
class DeltaAccum:
    """Streaming accumulator for the metrics the brief asks for.

    Frobenius norms and the cosine are computed from sums of squares and a dot
    product, so a chunked pass gives bit-comparable results to a dense one (up to
    float64 summation order).
    """

    n: int = 0
    n_changed: int = 0
    n_changed_rel: int = 0
    sq_delta: float = 0.0
    sq_base: float = 0.0
    sq_other: float = 0.0
    dot: float = 0.0
    max_abs_delta: float = 0.0
    max_abs_base: float = 0.0

    def add_stats(self, stats: Mapping[str, float]) -> None:
        self.n += int(stats["n"])
        self.n_changed += int(stats["n_changed"])
        self.n_changed_rel += int(stats["n_changed_rel"])
        self.sq_delta += float(stats["sq_delta"])
        self.sq_base += float(stats["sq_base"])
        self.sq_other += float(stats["sq_other"])
        self.dot += float(stats["dot"])
        self.max_abs_delta = max(self.max_abs_delta, float(stats["max_abs_delta"]))
        self.max_abs_base = max(self.max_abs_base, float(stats["max_abs_base"]))

    def merge(self, other: DeltaAccum) -> None:
        self.n += other.n
        self.n_changed += other.n_changed
        self.n_changed_rel += other.n_changed_rel
        self.sq_delta += other.sq_delta
        self.sq_base += other.sq_base
        self.sq_other += other.sq_other
        self.dot += other.dot
        self.max_abs_delta = max(self.max_abs_delta, other.max_abs_delta)
        self.max_abs_base = max(self.max_abs_base, other.max_abs_base)

    def metrics(self) -> dict[str, Any]:
        fro = float(np.sqrt(self.sq_delta))
        base_fro = float(np.sqrt(self.sq_base))
        other_fro = float(np.sqrt(self.sq_other))
        denom = base_fro * other_fro
        return {
            "n_params": self.n,
            "frobenius": fro,
            "base_frobenius": base_fro,
            "other_frobenius": other_fro,
            # A zero base block has no scale to be relative to; NaN says "undefined"
            # rather than pretending the change was infinite or zero.
            "rel_frobenius": (fro / base_fro) if base_fro > 0.0 else float("nan"),
            "cosine": (self.dot / denom) if denom > 0.0 else float("nan"),
            "max_abs_delta": self.max_abs_delta,
            "max_abs_base": self.max_abs_base,
            "rmse": float(np.sqrt(self.sq_delta / self.n)) if self.n else float("nan"),
            "frac_changed": (self.n_changed / self.n) if self.n else float("nan"),
            "frac_changed_rel": (self.n_changed_rel / self.n) if self.n else float("nan"),
        }


def _block_stats(
    base: np.ndarray, other: np.ndarray, delta: np.ndarray, rel_tol: float
) -> dict[str, float]:
    """Scalar reductions for one whole chunk."""
    absd = np.abs(delta)
    absb = np.abs(base)
    return {
        "n": float(delta.size),
        "n_changed": float(np.count_nonzero(delta)),
        "n_changed_rel": float(np.count_nonzero(absd > rel_tol * absb)),
        "sq_delta": float(np.dot(delta.ravel(), delta.ravel())),
        "sq_base": float(np.dot(base.ravel(), base.ravel())),
        "sq_other": float(np.dot(other.ravel(), other.ravel())),
        "dot": float(np.dot(base.ravel(), other.ravel())),
        "max_abs_delta": float(absd.max()) if absd.size else 0.0,
        "max_abs_base": float(absb.max()) if absb.size else 0.0,
    }


def _grouped_stats(
    base: np.ndarray,
    other: np.ndarray,
    delta: np.ndarray,
    rel_tol: float,
    reduce_axes: tuple[int, ...],
) -> list[dict[str, float]]:
    """Same reductions, but keeping one axis (the head axis) intact."""
    keep = [ax for ax in range(delta.ndim) if ax not in reduce_axes]
    if len(keep) != 1:
        raise ValueError("grouped stats need exactly one surviving axis")
    per = int(np.prod([delta.shape[ax] for ax in reduce_axes])) if reduce_axes else 1
    absd = np.abs(delta)
    sq_delta = (delta * delta).sum(axis=reduce_axes)
    sq_base = (base * base).sum(axis=reduce_axes)
    sq_other = (other * other).sum(axis=reduce_axes)
    dot = (base * other).sum(axis=reduce_axes)
    changed = (delta != 0.0).sum(axis=reduce_axes)
    changed_rel = (absd > rel_tol * np.abs(base)).sum(axis=reduce_axes)
    max_delta = absd.max(axis=reduce_axes)
    max_base = np.abs(base).max(axis=reduce_axes)
    return [
        {
            "n": float(per),
            "n_changed": float(changed[i]),
            "n_changed_rel": float(changed_rel[i]),
            "sq_delta": float(sq_delta[i]),
            "sq_base": float(sq_base[i]),
            "sq_other": float(sq_other[i]),
            "dot": float(dot[i]),
            "max_abs_delta": float(max_delta[i]),
            "max_abs_base": float(max_base[i]),
        }
        for i in range(delta.shape[keep[0]])
    ]


# --------------------------------------------------------------------------- #
# streaming reader
# --------------------------------------------------------------------------- #


class MemmapPool:
    """One read-only ``np.memmap`` per file, reused across every tensor in it."""

    def __init__(self) -> None:
        self._maps: dict[Path, np.memmap] = {}

    def get(self, path: Path) -> np.memmap:
        path = Path(path)
        mm = self._maps.get(path)
        if mm is None:
            mm = np.memmap(path, dtype=np.uint8, mode="r")
            self._maps[path] = mm
        return mm

    def close(self) -> None:
        self._maps.clear()

    def __enter__(self) -> MemmapPool:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def _read_rows(pool: MemmapPool, loc: TensorLocation, r0: int, r1: int, n_cols: int) -> np.ndarray:
    mm = pool.get(loc.file)
    stride = n_cols * loc.itemsize
    raw = mm[loc.begin + r0 * stride : loc.begin + r1 * stride]
    return decode_to_float64(np.asarray(raw), loc.st_dtype).reshape(r1 - r0, n_cols)


def _rows_per_chunk(n_cols: int, chunk_bytes: int, *, align: int = 1) -> int:
    """How many rows fit in the byte budget, rounded to a multiple of ``align``.

    Aligning to ``head_dim`` when splitting rows means a chunk never straddles a
    head boundary, which keeps the per-head reduction a single reshape.
    """
    per_row = max(1, n_cols) * 8 * 3  # base + other + delta, all float64
    rows = max(1, int(chunk_bytes // per_row))
    if align > 1:
        rows = max(align, (rows // align) * align)
    return rows


def iter_row_chunks(
    pool: MemmapPool,
    base: TensorLocation,
    other: TensorLocation,
    *,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    align: int = 1,
) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield ``(row_offset, base_chunk, other_chunk)`` as float64 2-D arrays."""
    if base.shape != other.shape:
        raise ValueError(f"{base.name}: shape {base.shape} vs {other.shape}")
    shape = base.shape or (1,)
    n_rows = int(shape[0])
    n_cols = 1
    for d in shape[1:]:
        n_cols *= int(d)
    step = _rows_per_chunk(n_cols, chunk_bytes, align=align)
    for r0 in range(0, n_rows, step):
        r1 = min(r0 + step, n_rows)
        yield r0, _read_rows(pool, base, r0, r1, n_cols), _read_rows(pool, other, r0, r1, n_cols)


# --------------------------------------------------------------------------- #
# per-tensor and per-pair drivers
# --------------------------------------------------------------------------- #


@dataclass
class TensorDelta:
    key: TensorKey
    shape: tuple[int, ...]
    base_dtype: str
    other_dtype: str
    accum: DeltaAccum
    heads: dict[int, DeltaAccum] = field(default_factory=dict)
    layout: HeadLayout | None = None

    def metrics(self) -> dict[str, Any]:
        return self.accum.metrics()


def tensor_delta(
    pool: MemmapPool,
    base: TensorLocation,
    other: TensorLocation,
    *,
    arch: Arch | None = None,
    stored_out_first: bool = True,
    rel_tol: float = DEFAULT_REL_TOL,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
) -> TensorDelta:
    """Stream ``other - base`` for one tensor, with optional per-head breakdown."""
    key = parse_tensor_name(base.name)
    layout = (
        head_layout(key.family, base.shape, arch, stored_out_first=stored_out_first)
        if (arch is not None and key.param == "weight")
        else None
    )
    align = layout.head_dim if (layout is not None and layout.axis == 0) else 1

    accum = DeltaAccum()
    heads: dict[int, DeltaAccum] = {}
    for r0, b, o in iter_row_chunks(
        pool, base, other, chunk_bytes=chunk_bytes, align=align
    ):
        d = o - b
        accum.add_stats(_block_stats(b, o, d, rel_tol))
        if layout is None:
            continue
        if layout.axis == 0:
            n_blocks = b.shape[0] // layout.head_dim
            shaped = (n_blocks, layout.head_dim, b.shape[1])
            stats = _grouped_stats(
                b.reshape(shaped), o.reshape(shaped), d.reshape(shaped), rel_tol, (1, 2)
            )
            first = r0 // layout.head_dim
        else:
            shaped = (b.shape[0], layout.n_heads, layout.head_dim)
            stats = _grouped_stats(
                b.reshape(shaped), o.reshape(shaped), d.reshape(shaped), rel_tol, (0, 2)
            )
            first = 0
        for i, st in enumerate(stats):
            heads.setdefault(first + i, DeltaAccum()).add_stats(st)

    return TensorDelta(
        key=key,
        shape=base.shape,
        base_dtype=base.st_dtype,
        other_dtype=other.st_dtype,
        accum=accum,
        heads=heads,
        layout=layout,
    )


@dataclass
class PairDelta:
    """Everything the three delta CSVs need for one (base, other) pair."""

    base_key: str
    other_key: str
    rel_tol: float
    orientation: OrientationReport | None
    overall: DeltaAccum
    per_tensor: list[TensorDelta]
    by_layer_family: dict[tuple[int, str], DeltaAccum]
    by_head: dict[tuple[int, str, int], DeltaAccum]
    skipped: list[str]

    def overall_metrics(self) -> dict[str, Any]:
        return self.overall.metrics()


def pair_deltas(
    base_locs: Mapping[str, TensorLocation],
    other_locs: Mapping[str, TensorLocation],
    *,
    base_key: str,
    other_key: str,
    arch: Arch | None = None,
    rel_tol: float = DEFAULT_REL_TOL,
    chunk_bytes: int = DEFAULT_CHUNK_BYTES,
    per_head: bool = True,
) -> PairDelta:
    """Full weight-space comparison of two checkpoints over their shared tensors.

    Tensors present on one side only, or with mismatched shapes, are skipped and
    listed in ``PairDelta.skipped`` -- silently dropping them would make the
    aggregate norms incomparable between pairs.
    """
    shared = sorted(set(base_locs) & set(other_locs))
    skipped = sorted(set(base_locs) ^ set(other_locs))

    orientation: OrientationReport | None = None
    if per_head and arch is not None:
        shapes = {n: loc.shape for n, loc in base_locs.items()}
        try:
            orientation = resolve_orientation(shapes, arch)
        except ValueError:
            # No projections in this checkpoint (or an unreadable layout): fall back
            # to whole-tensor metrics rather than guessing a head split.
            orientation = None

    overall = DeltaAccum()
    per_tensor: list[TensorDelta] = []
    by_layer_family: dict[tuple[int, str], DeltaAccum] = {}
    by_head: dict[tuple[int, str, int], DeltaAccum] = {}

    with MemmapPool() as pool:
        for name in shared:
            b, o = base_locs[name], other_locs[name]
            if b.shape != o.shape:
                skipped.append(name)
                continue
            td = tensor_delta(
                pool,
                b,
                o,
                arch=arch if (per_head and orientation is not None) else None,
                stored_out_first=orientation.stored_out_first if orientation else True,
                rel_tol=rel_tol,
                chunk_bytes=chunk_bytes,
            )
            per_tensor.append(td)
            overall.merge(td.accum)
            by_layer_family.setdefault((td.key.layer, td.key.family), DeltaAccum()).merge(td.accum)
            for head_index, acc in td.heads.items():
                by_head.setdefault((td.key.layer, td.key.family, head_index), DeltaAccum()).merge(
                    acc
                )

    return PairDelta(
        base_key=base_key,
        other_key=other_key,
        rel_tol=rel_tol,
        orientation=orientation,
        overall=overall,
        per_tensor=per_tensor,
        by_layer_family=by_layer_family,
        by_head=by_head,
        skipped=sorted(set(skipped)),
    )
