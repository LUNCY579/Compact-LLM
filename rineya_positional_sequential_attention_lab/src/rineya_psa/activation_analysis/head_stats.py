"""Per-head positional statistics (brief section 18).

One row per ``(layer, q_head)``, plus the same table rolled up to ``(layer, kv_head)``:

===========================  ===========================================================
``gold_span_attn``           mean attention mass on the evidence span
``edge_attn``                mass on the first and last ``edge_fraction`` of the context
``middle_attn``              mass within ``middle_halfwidth`` of the context midpoint
``recent_attn``              mass on the most recent tokens (see the caveat below)
``entropy``                  attention entropy in nats
``qk_logit_mean``            mean pre-softmax q.k logit, averaged over bins
``position_mean``            centre of mass of the attention over normalised position
``position_variance``        spread of that distribution -- a focused head scores low
``head_out_norm``            ``||O[g, i, :]||`` before ``o_proj``
``ablation_effect``          attached from ``significance/`` (KV-head resolution)
``patch_effect``             attached from ``patching.py`` (layer resolution)
===========================  ===========================================================

All the attention statistics come from ``hooks.capture`` through a normal ``psa_session``
with a capture-only :class:`PSAContext`. None of that reduction logic is re-derived here;
this module chooses the bin grid, drives the forwards, and turns per-bin captures into
the named scalars above.

Three honesty notes, all reflected in the emitted columns
---------------------------------------------------------
**Recency needs a query-dependent bin, and capture forbids one.** ``CaptureSpec`` rejects
``DistanceBinning`` outright -- ``attn_by_bin`` is defined over *absolute source* bins --
and SPEC 12.3 explains why a query-dependent grid is unaffordable above 2K anyway. So
"recent" is measured where absolute lateness and recency coincide: on the **final
``final_rows`` query rows**, where a source in the last bin is by construction within
about ``T / n_bins`` tokens of the query. That is a second capture pass, restricted to
those rows by setting ``PSAContext.valid_mask``. ``valid_mask``'s documented job is
marking padding, and it is read on both the source and the query side -- but the source
side is consulted *only* by Mode 1 and Mode 2, and this context carries neither
(:func:`_assert_capture_only` enforces it). Within a capture-only context the mask is
exactly a per-query-row weight, which is what is wanted. ``recent_scope`` records the row
window on every row of the table so the number is never read as an all-rows average.

**Ablation effect is a KV-head quantity.** Mode-2 attenuation scales ``V`` before
``repeat_kv``, so it hits every query head in the group at once. The ``q_head`` rows
inherit their KV head's value; ``ablation_scope="kv_head"`` says so. A difference between
two q-heads of the same KV head is never evidence about ablation.

**Patch effect is a layer quantity.** ``patching.py`` patches ``attn_out`` after
``o_proj``, where the heads have already been summed, so there is no per-head patch to
report. The layer's value is broadcast across its heads and ``patch_effect_scope="layer"``
says so.

DEVIATION: SPEC 12.1 puts the ``sdpa_kernel`` pin in ``evaluation/runner.py``. Capture
runs the chunked eager path by construction (SPEC 12.3), so the pin is inert here, but the
same context manager is entered via ``collect.forced_sdpa_backends`` for the parts of the
forward that do dispatch through SDPA.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rineya_psa.activation_analysis.collect import forced_sdpa_backends
from rineya_psa.activation_analysis.patching import record_field
from rineya_psa.hooks.binning import DEFAULT_CHUNK_ROWS, AbsoluteBinning, PositionBinning
from rineya_psa.hooks.registry import psa_session
from rineya_psa.hooks.specs import CaptureSpec, PSAContext
from rineya_psa.loading import arch_from_config
from rineya_psa.significance.matrices import bin_centers
from rineya_psa.types import Arch, TokenizedItem

__all__ = [
    "ABLATION_REDUCERS",
    "CAPTURE_WHAT",
    "HEAD_METRICS",
    "HeadStatRow",
    "HeadStatsConfig",
    "HeadStatsTable",
    "KVHeadStatRow",
    "capture_head_stats",
    "capture_item",
    "head_stats_to_dataframe",
    "write_head_stats",
]

#: Every reduction ``hooks.capture`` offers; all five feed a column of the table.
CAPTURE_WHAT: tuple[str, ...] = (
    "attn_by_bin",
    "attn_entropy",
    "gold_span_attn",
    "qk_logit_mean",
    "head_out_norm",
)

#: Metrics measured by this module, in table order. ``ablation_effect`` and
#: ``patch_effect`` are attached afterwards and are deliberately not in this tuple.
HEAD_METRICS: tuple[str, ...] = (
    "gold_span_attn",
    "edge_attn",
    "middle_attn",
    "recent_attn",
    "entropy",
    "qk_logit_mean",
    "position_mean",
    "position_variance",
    "head_out_norm",
)

ABLATION_REDUCERS: tuple[str, ...] = ("max", "mean", "max_abs", "sum_abs")

_ATTACHED: tuple[str, ...] = ("ablation_effect", "patch_effect")


# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class HeadStatsConfig:
    """Bin grid and the windows the named scalars are read off it."""

    n_bins: int = 16
    edge_fraction: float = 0.1
    middle_halfwidth: float = 0.1
    recent_fraction: float = 0.1
    final_rows: int = 32
    chunk: int = DEFAULT_CHUNK_ROWS

    def __post_init__(self) -> None:
        if int(self.n_bins) < 2:
            raise ValueError(f"n_bins must be >= 2, got {self.n_bins}")
        for name in ("edge_fraction", "middle_halfwidth", "recent_fraction"):
            value = float(getattr(self, name))
            if not 0.0 < value <= 0.5:
                raise ValueError(f"{name} must lie in (0, 0.5], got {value}")
        if int(self.final_rows) < 0:
            raise ValueError("final_rows must be non-negative (0 disables the recency pass)")

    def binning(self) -> AbsoluteBinning:
        return AbsoluteBinning(int(self.n_bins))

    def masks(self, centers: np.ndarray) -> dict[str, np.ndarray]:
        """Bin selections for the edge / middle / recent windows.

        A window that selects no bin is a configuration error, not something to paper
        over by silently widening it: the resulting column would read as a hard zero.
        """
        out = {
            "edge": (centers < self.edge_fraction) | (centers > 1.0 - self.edge_fraction),
            "middle": np.abs(centers - 0.5) <= self.middle_halfwidth,
            "recent": centers > 1.0 - self.recent_fraction,
        }
        for name, mask in out.items():
            if not mask.any():
                raise ValueError(
                    f"the {name} window selects none of {centers.size} bins; increase "
                    f"n_bins or widen the window (bin width is {1.0 / centers.size:.4f})"
                )
        return out

    def describe(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_CONFIG = HeadStatsConfig()


# --------------------------------------------------------------------------------------
# Rows
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class HeadStatRow:
    """One ``(layer, q_head)``. ``kv_head`` is always spelled out; see SPEC 5.5."""

    model_key: str | None
    layer: int
    q_head: int
    kv_head: int
    n_items: int
    gold_span_attn: float
    edge_attn: float
    middle_attn: float
    recent_attn: float
    entropy: float
    qk_logit_mean: float
    position_mean: float
    position_variance: float
    head_out_norm: float
    ablation_effect: float = float("nan")
    patch_effect: float = float("nan")
    ablation_scope: str = "unset"
    patch_effect_scope: str = "unset"
    recent_scope: str = "unset"

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class KVHeadStatRow:
    """One ``(layer, kv_head)``: the q-head rows of the GQA group, averaged."""

    model_key: str | None
    layer: int
    kv_head: int
    q_heads: tuple[int, ...]
    n_items: int
    gold_span_attn: float
    edge_attn: float
    middle_attn: float
    recent_attn: float
    entropy: float
    qk_logit_mean: float
    position_mean: float
    position_variance: float
    head_out_norm: float
    ablation_effect: float = float("nan")
    patch_effect: float = float("nan")
    ablation_scope: str = "unset"
    patch_effect_scope: str = "unset"
    recent_scope: str = "unset"

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["q_heads"] = list(self.q_heads)
        return d


# --------------------------------------------------------------------------------------
# Table
# --------------------------------------------------------------------------------------


@dataclass
class HeadStatsTable:
    """``[n_layers, n_q_heads]`` per metric, plus the bookkeeping to read it."""

    layers: tuple[int, ...]
    n_q_heads: int
    kv_group_size: int
    means: dict[str, np.ndarray]
    sds: dict[str, np.ndarray]
    n_items: int
    binning_desc: Mapping[str, Any] = field(default_factory=dict)
    config: Mapping[str, Any] = field(default_factory=dict)
    model_key: str | None = None
    recent_scope: str = "unset"
    ablation_scope: str = "unset"
    patch_effect_scope: str = "unset"

    def __post_init__(self) -> None:
        shape = (len(self.layers), int(self.n_q_heads))
        for name, values in self.means.items():
            if np.asarray(values).shape != shape:
                raise ValueError(f"means[{name!r}] is {np.shape(values)}, expected {shape}")
        if self.n_q_heads % max(1, self.kv_group_size):
            raise ValueError(
                f"{self.n_q_heads} query heads do not split into groups of "
                f"{self.kv_group_size}"
            )

    @property
    def n_kv_heads(self) -> int:
        return int(self.n_q_heads) // int(self.kv_group_size)

    def kv_head_of(self, q_head: int) -> int:
        """SPEC 5.5's mapping, in the one direction this table needs."""
        return int(q_head) // int(self.kv_group_size)

    def value(self, metric: str, layer: int, q_head: int) -> float:
        try:
            block = self.means[metric]
        except KeyError:
            raise KeyError(f"no metric {metric!r}; have {sorted(self.means)}") from None
        return float(block[self.layers.index(int(layer)), int(q_head)])

    def _get(self, metric: str) -> np.ndarray:
        return np.asarray(
            self.means.get(metric, np.full((len(self.layers), self.n_q_heads), np.nan)),
            dtype=np.float64,
        )

    # -- attaching external evidence --------------------------------------------------

    def with_ablation_effect(
        self, matrix: Any, *, reducer: str = "max"
    ) -> HeadStatsTable:
        """Attach a ``SignificanceMatrix``'s per-cell importance, reduced over bins.

        The matrix is ``[layer, kv_head, bin]`` and Mode-2 attenuation is a KV-head
        intervention, so every q-head of a group receives the same number. Cells the
        matrix never evaluated stay NaN rather than becoming zeros.
        """
        if reducer not in ABLATION_REDUCERS:
            raise ValueError(f"unknown reducer {reducer!r}; known: {list(ABLATION_REDUCERS)}")
        importance = np.asarray(matrix.importance(), dtype=np.float64)
        if importance.shape[1] != self.n_kv_heads:
            raise ValueError(
                f"matrix has {importance.shape[1]} KV heads but the model has "
                f"{self.n_kv_heads}"
            )
        # Cells the ladder never visited are NaN by design (an unmeasured cell is not a
        # zero). Masking before the reduction rather than calling ``nanmax`` keeps an
        # all-unvisited row NaN without an "All-NaN slice" warning on every call.
        finite = np.isfinite(importance)
        n_finite = finite.sum(axis=2)
        if reducer == "max":
            reduced = np.where(finite, importance, -np.inf).max(axis=2)
        elif reducer == "max_abs":
            reduced = np.where(finite, np.abs(importance), -np.inf).max(axis=2)
        elif reducer == "mean":
            reduced = np.where(finite, importance, 0.0).sum(axis=2) / np.maximum(n_finite, 1)
        else:
            reduced = np.where(finite, np.abs(importance), 0.0).sum(axis=2)
        reduced = np.where(n_finite == 0, np.nan, reduced)

        matrix_layers = {int(layer): i for i, layer in enumerate(matrix.layers)}
        out = np.full((len(self.layers), self.n_q_heads), np.nan, dtype=np.float64)
        for row, layer in enumerate(self.layers):
            index = matrix_layers.get(int(layer))
            if index is None:
                continue
            for q_head in range(self.n_q_heads):
                kv = self.kv_head_of(q_head)
                if kv in matrix.kv_heads:
                    out[row, q_head] = reduced[index, matrix.kv_heads.index(kv)]
        return self._attach(
            "ablation_effect",
            out,
            ablation_scope=f"kv_head:{matrix.objective}:{reducer}",
        )

    def with_patch_effect(
        self,
        records: Sequence[Any],
        *,
        site: str = "attn_out",
        alignment: str = "final_query",
        statistic: str = "delta_gold_logprob",
    ) -> HeadStatsTable:
        """Attach the mean patch effect of one ``(site, alignment)``, by layer.

        Layer resolution, not head resolution: ``attn_out`` is post-``o_proj``. Layers
        with no usable record stay NaN.
        """
        totals: dict[int, list[float]] = {}
        for record in records:
            if str(record_field(record, "status", "ok")) != "ok":
                continue
            if str(record_field(record, "site")) != site:
                continue
            if str(record_field(record, "alignment")) != alignment:
                continue
            value = record_field(record, statistic)
            if value is None or not np.isfinite(float(value)):
                continue
            totals.setdefault(int(record_field(record, "layer")), []).append(float(value))

        out = np.full((len(self.layers), self.n_q_heads), np.nan, dtype=np.float64)
        for row, layer in enumerate(self.layers):
            values = totals.get(int(layer))
            if values:
                out[row, :] = float(np.mean(values))
        return self._attach(
            "patch_effect",
            out,
            patch_effect_scope=f"layer:{site}:{alignment}:{statistic}",
        )

    def _attach(self, metric: str, values: np.ndarray, **scopes: str) -> HeadStatsTable:
        means = dict(self.means)
        means[metric] = np.asarray(values, dtype=np.float64)
        sds = dict(self.sds)
        sds.pop(metric, None)
        return replace(self, means=means, sds=sds, **scopes)

    # -- emission ----------------------------------------------------------------------

    def rows(self) -> list[HeadStatRow]:
        blocks = {name: self._get(name) for name in (*HEAD_METRICS, *_ATTACHED)}
        return [
            HeadStatRow(
                model_key=self.model_key,
                layer=int(layer),
                q_head=int(q_head),
                kv_head=self.kv_head_of(q_head),
                n_items=int(self.n_items),
                ablation_scope=self.ablation_scope,
                patch_effect_scope=self.patch_effect_scope,
                recent_scope=self.recent_scope,
                **{name: float(blocks[name][row, q_head]) for name in
                   (*HEAD_METRICS, *_ATTACHED)},
            )
            for row, layer in enumerate(self.layers)
            for q_head in range(self.n_q_heads)
        ]

    def kv_rows(self) -> list[KVHeadStatRow]:
        blocks = {name: self._get(name) for name in (*HEAD_METRICS, *_ATTACHED)}
        group = int(self.kv_group_size)
        out: list[KVHeadStatRow] = []
        for row, layer in enumerate(self.layers):
            for kv in range(self.n_kv_heads):
                members = tuple(range(kv * group, (kv + 1) * group))
                out.append(
                    KVHeadStatRow(
                        model_key=self.model_key,
                        layer=int(layer),
                        kv_head=int(kv),
                        q_heads=members,
                        n_items=int(self.n_items),
                        ablation_scope=self.ablation_scope,
                        patch_effect_scope=self.patch_effect_scope,
                        recent_scope=self.recent_scope,
                        **{
                            name: float(np.nanmean(blocks[name][row, list(members)]))
                            if np.isfinite(blocks[name][row, list(members)]).any()
                            else float("nan")
                            for name in (*HEAD_METRICS, *_ATTACHED)
                        },
                    )
                )
        return out

    def long_frame(self) -> Any:
        """``(layer, q_head, kv_head, metric, mean, sd, n_items)`` -- parquet-friendly."""
        import pandas as pd

        rows = []
        for metric in (*HEAD_METRICS, *_ATTACHED):
            block = self._get(metric)
            sd_block = self.sds.get(metric)
            for row, layer in enumerate(self.layers):
                for q_head in range(self.n_q_heads):
                    rows.append(
                        {
                            "model_key": self.model_key,
                            "layer": int(layer),
                            "q_head": int(q_head),
                            "kv_head": self.kv_head_of(q_head),
                            "metric": metric,
                            "mean": float(block[row, q_head]),
                            "sd": (
                                float(sd_block[row, q_head])
                                if sd_block is not None
                                else float("nan")
                            ),
                            "n_items": int(self.n_items),
                        }
                    )
        return pd.DataFrame(rows)

    def describe(self) -> dict[str, Any]:
        return {
            "model_key": self.model_key,
            "layers": list(self.layers),
            "n_q_heads": self.n_q_heads,
            "n_kv_heads": self.n_kv_heads,
            "kv_group_size": self.kv_group_size,
            "n_items": self.n_items,
            "metrics": list(self.means),
            "binning": dict(self.binning_desc),
            "config": dict(self.config),
            "recent_scope": self.recent_scope,
            "ablation_scope": self.ablation_scope,
            "patch_effect_scope": self.patch_effect_scope,
        }

# --------------------------------------------------------------------------------------
# Capture
# --------------------------------------------------------------------------------------


def _tokenized(entry: Any) -> TokenizedItem:
    if isinstance(entry, TokenizedItem):
        return entry
    if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[1], TokenizedItem):
        return entry[1]
    raise TypeError(
        "expected TokenizedItem or (BenchmarkItem, TokenizedItem); got "
        f"{type(entry).__name__}"
    )


def _assert_capture_only(ctx: PSAContext) -> None:
    """The valid_mask row-selection trick is only sound with no intervention present."""
    if not ctx.is_identity():
        raise RuntimeError(
            "head_stats drives a capture-only context: valid_mask is used to select "
            "query rows, and Mode 1/2 would additionally read it as a source mask"
        )


def _tail_mask(n_tokens: int, n_rows: int, device: Any) -> torch.Tensor:
    mask = torch.zeros(int(n_tokens), dtype=torch.bool, device=device)
    mask[max(0, int(n_tokens) - int(n_rows)) :] = True
    return mask


def _forward(model: Any, input_ids: Sequence[int], device: Any) -> None:
    ids = torch.tensor([int(i) for i in input_ids], dtype=torch.long, device=device)
    with torch.inference_mode(), forced_sdpa_backends(device):
        model(input_ids=ids.reshape(1, -1), use_cache=False, logits_to_keep=1)


def _per_item_metrics(
    captured: Mapping[tuple[int, str], np.ndarray],
    layers: Sequence[int],
    n_q_heads: int,
    centers: np.ndarray,
    masks: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Turn one item's captures into ``{metric: [n_layers, n_q_heads]}``."""
    shape = (len(layers), int(n_q_heads))
    out = {name: np.full(shape, np.nan, dtype=np.float64) for name in HEAD_METRICS}
    for row, layer in enumerate(layers):
        by_bin = np.asarray(captured[(int(layer), "attn_by_bin")], dtype=np.float64)
        qk = np.asarray(captured[(int(layer), "qk_logit_mean")], dtype=np.float64)
        out["gold_span_attn"][row] = captured[(int(layer), "gold_span_attn")]
        out["entropy"][row] = captured[(int(layer), "attn_entropy")]
        out["head_out_norm"][row] = captured[(int(layer), "head_out_norm")]
        out["edge_attn"][row] = by_bin[:, masks["edge"]].sum(axis=1)
        out["middle_attn"][row] = by_bin[:, masks["middle"]].sum(axis=1)
        out["recent_attn"][row] = by_bin[:, masks["recent"]].sum(axis=1)
        out["qk_logit_mean"][row] = np.nanmean(qk, axis=1)

        total = by_bin.sum(axis=1, keepdims=True)
        # Rows sum to 1 up to the mass on masked columns, which is ~0; renormalising
        # makes the moments well defined even when that residue is not exactly zero.
        p = np.divide(by_bin, np.where(total > 0.0, total, 1.0))
        mean = p @ centers
        out["position_mean"][row] = mean
        # E[c^2] - E[c]^2 is exact in real arithmetic and can land at -1e-17 in float;
        # a negative variance in the table would read as a bug rather than as round-off.
        out["position_variance"][row] = np.maximum(p @ (centers**2) - mean**2, 0.0)
    return out


def capture_item(
    model: Any,
    tok_item: TokenizedItem,
    ctx: PSAContext,
    *,
    layers: Sequence[int],
    n_q_heads: int,
    config: HeadStatsConfig,
    centers: np.ndarray,
    masks: Mapping[str, np.ndarray],
    device: Any,
) -> dict[str, np.ndarray]:
    """One item's ``{metric: [n_layers, n_q_heads]}``, inside an open ``psa_session``.

    Two forwards: an all-rows pass for everything, then a final-rows pass that overwrites
    ``recent_attn`` (see the module docstring). ``config.final_rows = 0`` skips the second
    pass and leaves ``recent_attn`` as the all-rows figure, which is *not* a recency
    measurement -- the caller is told so through ``recent_scope``.
    """
    _assert_capture_only(ctx)
    ctx.evidence_spans = tuple(tok_item.evidence_spans)

    ctx.valid_mask = None
    ctx.reset_capture()
    _forward(model, tok_item.input_ids, device)
    metrics = _per_item_metrics(
        ctx.finalize_capture(), layers, n_q_heads, centers, masks
    )

    if config.final_rows:
        ctx.valid_mask = _tail_mask(tok_item.n_tokens, config.final_rows, device)
        ctx.reset_capture()
        _forward(model, tok_item.input_ids, device)
        tail = _per_item_metrics(
            ctx.finalize_capture(), layers, n_q_heads, centers, masks
        )
        metrics["recent_attn"] = tail["recent_attn"]
        ctx.valid_mask = None
    return metrics


def capture_head_stats(
    model: Any,
    items: Sequence[Any],
    *,
    layers: Sequence[int] | None = None,
    config: HeadStatsConfig = DEFAULT_CONFIG,
    binning: PositionBinning | None = None,
    arch: Arch | None = None,
    model_key: str | None = None,
    progress: Any | None = None,
) -> HeadStatsTable:
    """Run every item through a capture-only session and reduce to a head table.

    ``items`` are ``TokenizedItem`` or ``(BenchmarkItem, TokenizedItem)`` pairs. Cost is
    one forward per item, or two when ``config.final_rows > 0``.
    """
    tok_items = [_tokenized(entry) for entry in items]
    if not tok_items:
        raise ValueError("no items to capture")
    if arch is None:
        arch = arch_from_config(model.config)
    use_layers = tuple(
        range(arch.num_layers) if layers is None else sorted({int(i) for i in layers})
    )
    bad = [i for i in use_layers if not 0 <= i < arch.num_layers]
    if bad:
        raise IndexError(f"layers {bad} out of range for a {arch.num_layers}-layer model")

    grid = binning if binning is not None else config.binning()
    if grid.is_query_dependent:
        raise ValueError(
            f"{grid.name} is query-dependent; capture bins absolute source positions "
            "(see the module docstring on recency)"
        )
    centers = bin_centers(grid.describe())
    masks = config.masks(centers)

    ctx = PSAContext(
        enabled=True,
        capture=CaptureSpec(
            layers=use_layers, what=CAPTURE_WHAT, chunk=config.chunk, binning=grid
        ),
        position_bins=grid,
    )
    device = next(model.parameters()).device

    stacks: dict[str, list[np.ndarray]] = {name: [] for name in HEAD_METRICS}
    with psa_session(model, ctx):
        for tok_item in tok_items:
            metrics = capture_item(
                model,
                tok_item,
                ctx,
                layers=use_layers,
                n_q_heads=arch.num_q_heads,
                config=config,
                centers=centers,
                masks=masks,
                device=device,
            )
            for name, values in metrics.items():
                stacks[name].append(values)
            if progress is not None:
                progress(1)

    means: dict[str, np.ndarray] = {}
    sds: dict[str, np.ndarray] = {}
    for name, blocks in stacks.items():
        stacked = np.stack(blocks, axis=0)  # [n_items, L, H]
        means[name] = np.nanmean(stacked, axis=0)
        sds[name] = (
            np.nanstd(stacked, axis=0, ddof=1)
            if stacked.shape[0] > 1
            else np.zeros(stacked.shape[1:], dtype=np.float64)
        )

    recent_scope = (
        f"final_{config.final_rows}_query_rows" if config.final_rows else "all_query_rows"
    )
    return HeadStatsTable(
        layers=use_layers,
        n_q_heads=int(arch.num_q_heads),
        kv_group_size=int(arch.kv_group_size),
        means=means,
        sds=sds,
        n_items=len(tok_items),
        binning_desc=dict(grid.describe()),
        config=config.describe(),
        model_key=model_key,
        recent_scope=recent_scope,
    )



def head_stats_to_dataframe(
    rows: Sequence[HeadStatRow | KVHeadStatRow],
) -> Any:
    """Wide table, one row per head. Works for either resolution."""
    import pandas as pd

    if not rows:
        return pd.DataFrame(columns=list(HeadStatRow.__dataclass_fields__))
    return pd.DataFrame([r.to_json() for r in rows])


def write_head_stats(
    table: HeadStatsTable,
    path: str | Path,
    *,
    kv: bool = False,
    config: Mapping[str, Any] | None = None,
) -> Path:
    """Write the head table plus its ``*.provenance.json`` sidecar (SPEC section 9)."""
    from rineya_psa.statistics.provenance import write_result

    rows = table.kv_rows() if kv else table.rows()
    payload = {**table.describe(), **dict(config or {})}
    return write_result(head_stats_to_dataframe(rows), path, config=payload)
