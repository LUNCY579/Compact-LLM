"""The thirteen figures of brief section 47, one function each.

Every function takes tidy data plus a save path and returns the paths written (PNG at 200
dpi and PDF, from :func:`rineya_psa.figures.style.save_figure`). "Tidy data" is deliberately
loose: :func:`_columns` accepts a pandas ``DataFrame``, a ``{column: sequence}`` mapping, or
a sequence of row mappings, so a caller with a parquet table and a caller with a dict of
numpy arrays both work and neither drags pandas into a test.

Matrix figures (4, 5, 6) take arrays of the shape their axes imply and validate it, because
a transposed ``[layer, kv_head]`` matrix produces a plausible-looking heatmap that is
completely wrong.

Every value axis goes through
:func:`rineya_psa.figures.style.assert_no_truncated_axis` before the figure is saved, so a
figure whose y-axis hides zero raises instead of being written. That guard is the whole
point of section 47 and it is enforced here, not left to review.

DEVIATION: section 47 names the figures but not their column names. Each function takes its
column names as keyword arguments with defaults matching the artifacts this lab already
writes (``results/weight_delta_by_layer.csv`` uses ``model_key``/``layer``/``family``/
``rel_frobenius``, and ``runner.RunRecord`` uses ``correct``/``gold_nll``), so the defaults
are right for the lab's own tables and overridable for anything else.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from rineya_psa.figures import style
from rineya_psa.figures.style import (
    assert_no_truncated_axis,
    assign_colors,
    assign_markers,
    figure_style,
    save_figure,
)

__all__ = [
    "FIGURES",
    "FIGURE_ORDER",
    "absolute_vs_distance_effects",
    "accuracy_vs_latency",
    "accuracy_vs_position",
    "architecture_comparison",
    "edge_middle_gap_by_model",
    "kv_head_position_heatmap",
    "layer_kv_head_significance",
    "layer_position_heatmap",
    "low_rank_explained_variance",
    "nll_vs_position",
    "significance_sequentiality_interaction",
    "weight_delta_by_layer",
    "worst_vs_mean_accuracy",
]


# --------------------------------------------------------------------------------------
# Tidy-data adapters
# --------------------------------------------------------------------------------------


def _columns(data: Any) -> dict[str, np.ndarray]:
    """Normalise a DataFrame / column mapping / row sequence into ``{name: ndarray}``."""
    if data is None:
        raise TypeError("no data given")
    if hasattr(data, "columns") and hasattr(data, "__getitem__"):
        return {str(c): np.asarray(data[c]) for c in list(data.columns)}
    if isinstance(data, Mapping):
        return {str(k): np.asarray(list(v)) for k, v in data.items()}
    rows = list(data)
    if rows and all(isinstance(r, Mapping) for r in rows):
        names: list[str] = []
        for row in rows:
            names.extend(k for k in row if k not in names)
        return {str(n): np.asarray([r.get(n) for r in rows]) for n in names}
    raise TypeError(
        "expected a DataFrame, a {column: values} mapping, or a sequence of row mappings, "
        f"got {type(data).__name__}"
    )


def _column(cols: Mapping[str, np.ndarray], name: str, *, numeric: bool = True) -> np.ndarray:
    try:
        raw = cols[name]
    except KeyError:
        raise KeyError(f"missing column {name!r}; available: {sorted(cols)}") from None
    return np.asarray(raw, dtype=np.float64) if numeric else np.asarray([str(x) for x in raw])


def _optional(
    cols: Mapping[str, np.ndarray], name: str | None, *, numeric: bool = True
) -> np.ndarray | None:
    if name is None or name not in cols:
        return None
    return _column(cols, name, numeric=numeric)


def _unique_in_order(values: Sequence[Any]) -> list[str]:
    """First-appearance order, so colour assignment is stable and not alphabetised."""
    seen: list[str] = []
    for v in values:
        s = str(v)
        if s not in seen:
            seen.append(s)
    return seen


def _matrix(values: Any, expected: tuple[int | None, int | None], what: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{what}: expected a 2-D matrix, got shape {arr.shape}")
    rows, cols = expected
    if rows is not None and arr.shape[0] != rows:
        raise ValueError(f"{what}: expected {rows} rows, got {arr.shape[0]} (transposed?)")
    if cols is not None and arr.shape[1] != cols:
        raise ValueError(f"{what}: expected {cols} columns, got {arr.shape[1]} (transposed?)")
    return arr


def _error_bars(
    values: np.ndarray, low: np.ndarray | None, high: np.ndarray | None
) -> np.ndarray | None:
    """Absolute CI bounds -> the ``[[below], [above]]`` offsets matplotlib wants."""
    if low is None or high is None:
        return None
    below = np.clip(values - low, 0.0, None)
    above = np.clip(high - values, 0.0, None)
    return np.vstack([below, above])


def _all_finite(*arrays: np.ndarray | None) -> np.ndarray:
    chunks = [np.asarray(a, dtype=np.float64).ravel() for a in arrays if a is not None]
    if not chunks:
        return np.zeros(1)
    stacked = np.concatenate(chunks)
    finite = stacked[np.isfinite(stacked)]
    return finite if finite.size else np.zeros(1)


def _heatmap(
    ax: Any,
    matrix: np.ndarray,
    *,
    xticks: Sequence[Any],
    yticks: Sequence[Any],
    xlabel: str,
    ylabel: str,
    diverging: bool,
    max_ticks: int = 16,
) -> Any:
    """Shared heatmap body for figures 4-6, including the sign-aware colour scale."""
    kwargs: dict[str, Any] = {"aspect": "auto", "origin": "lower", "interpolation": "nearest"}
    if diverging:
        vmin, vmax = style.symmetric_norm(matrix)
        kwargs.update(cmap=style.DIVERGING_CMAP, vmin=vmin, vmax=vmax)
    else:
        kwargs.update(cmap=style.SEQUENTIAL_CMAP)
    image = ax.imshow(matrix, **kwargs)

    x_step = max(1, len(xticks) // max_ticks)
    y_step = max(1, len(yticks) // max_ticks)
    ax.set_xticks(np.arange(len(xticks))[::x_step])
    ax.set_xticklabels([str(t) for t in xticks][::x_step], rotation=0)
    ax.set_yticks(np.arange(len(yticks))[::y_step])
    ax.set_yticklabels([str(t) for t in yticks][::y_step])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(False)
    return image


def _position_labels(positions: Sequence[float]) -> list[str]:
    return [f"{float(p):.2f}" for p in positions]


# --------------------------------------------------------------------------------------
# 1-2. Positional curves
# --------------------------------------------------------------------------------------


def _curve_panel(
    ax: Any,
    cols: Mapping[str, np.ndarray],
    *,
    model_col: str,
    position_col: str,
    value_col: str,
    ci_low_col: str | None,
    ci_high_col: str | None,
) -> np.ndarray:
    models = _column(cols, model_col, numeric=False)
    positions = _column(cols, position_col)
    values = _column(cols, value_col)
    low = _optional(cols, ci_low_col)
    high = _optional(cols, ci_high_col)

    labels = style.panel_order(_unique_in_order(models))
    colors = assign_colors(labels)
    markers = assign_markers(labels)
    for label in labels:
        mask = models == label
        order = np.argsort(positions[mask])
        x = positions[mask][order]
        y = values[mask][order]
        ax.plot(
            x,
            y,
            marker=markers[label],
            color=colors[label],
            label=style.display_name(label),
            zorder=3,
        )
        if low is not None and high is not None:
            ax.fill_between(
                x,
                low[mask][order],
                high[mask][order],
                color=colors[label],
                alpha=0.16,
                linewidth=0.0,
                zorder=2,
            )
    return _all_finite(values, low, high)


def accuracy_vs_position(
    data: Any,
    path: str | Path,
    *,
    model_col: str = "model",
    position_col: str = "position",
    value_col: str = "accuracy",
    ci_low_col: str | None = "ci_low",
    ci_high_col: str | None = "ci_high",
    title: str = "Figure 1 - accuracy vs normalized evidence position",
    ylabel: str = "accuracy",
) -> tuple[Path, ...]:
    """Figure 1: the U-curve itself, one line per model, on a full 0-1 accuracy axis."""
    cols = _columns(data)
    with figure_style():
        fig, ax = plt.subplots()
        drawn = _curve_panel(
            ax,
            cols,
            model_col=model_col,
            position_col=position_col,
            value_col=value_col,
            ci_low_col=ci_low_col,
            ci_high_col=ci_high_col,
        )
        style.position_axis(ax)
        style.tidy_titles(ax, title=title, xlabel=None, ylabel=ylabel)
        style.set_accuracy_ylim(ax, drawn)
        style.legend_outside(ax, title="model")
        assert_no_truncated_axis(ax)
        return save_figure(fig, path)


def nll_vs_position(
    data: Any,
    path: str | Path,
    *,
    model_col: str = "model",
    position_col: str = "position",
    value_col: str = "gold_nll",
    ci_low_col: str | None = "ci_low",
    ci_high_col: str | None = "ci_high",
    title: str = "Figure 2 - gold NLL vs normalized evidence position",
    ylabel: str = "gold NLL (nats / answer token)",
) -> tuple[Path, ...]:
    """Figure 2: the metric SPEC section 13.1 makes primary, on the same grid as figure 1.

    NLL is a non-negative magnitude, so the axis still starts at zero: the pilot's 0.27-nat
    edge-middle gap is a *large* effect on a 0-0.5 axis and a modest one against zero, and
    the honest rendering is the second.
    """
    cols = _columns(data)
    with figure_style():
        fig, ax = plt.subplots()
        drawn = _curve_panel(
            ax,
            cols,
            model_col=model_col,
            position_col=position_col,
            value_col=value_col,
            ci_low_col=ci_low_col,
            ci_high_col=ci_high_col,
        )
        style.position_axis(ax)
        style.tidy_titles(ax, title=title, xlabel=None, ylabel=ylabel)
        style.set_zero_based_ylim(ax, drawn)
        style.legend_outside(ax, title="model")
        assert_no_truncated_axis(ax)
        return save_figure(fig, path)


# --------------------------------------------------------------------------------------
# 3. Edge-middle gap
# --------------------------------------------------------------------------------------


def edge_middle_gap_by_model(
    data: Any,
    path: str | Path,
    *,
    model_col: str = "model",
    value_col: str = "edge_middle_gap",
    ci_low_col: str | None = "ci_low",
    ci_high_col: str | None = "ci_high",
    title: str = "Figure 3 - edge-middle gap by model (95% CI)",
    ylabel: str = "edge - middle score",
) -> tuple[Path, ...]:
    """Figure 3: the U-shape reduced to one signed number per model, with its interval.

    A signed quantity, so zero is drawn explicitly: a bar chart of differences whose axis
    started at 0.1 would make every model look U-shaped.
    """
    cols = _columns(data)
    models = _column(cols, model_col, numeric=False)
    values = _column(cols, value_col)
    low = _optional(cols, ci_low_col)
    high = _optional(cols, ci_high_col)
    labels = style.panel_order(_unique_in_order(models))
    order = [int(np.flatnonzero(models == label)[0]) for label in labels]
    colors = assign_colors(labels)

    with figure_style():
        fig, ax = plt.subplots()
        x = np.arange(len(labels), dtype=np.float64)
        heights = values[order]
        lo = None if low is None else low[order]
        hi = None if high is None else high[order]
        ax.bar(x, heights, width=0.62, color=[colors[label] for label in labels], zorder=3)
        err = _error_bars(heights, lo, hi)
        if err is not None:
            ax.errorbar(
                x, heights, yerr=err, fmt="none", ecolor="#333333", elinewidth=1.1, zorder=4
            )
        style.zero_line(ax)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [style.display_name(label) for label in labels], rotation=12, ha="right"
        )
        style.tidy_titles(ax, title=title, xlabel=None, ylabel=ylabel)
        style.set_zero_based_ylim(ax, _all_finite(heights, lo, hi))
        assert_no_truncated_axis(ax)
        return save_figure(fig, path)


# --------------------------------------------------------------------------------------
# 4-6. Heatmaps
# --------------------------------------------------------------------------------------


def layer_position_heatmap(
    matrix: Any,
    path: str | Path,
    *,
    positions: Sequence[float],
    layers: Sequence[int] | None = None,
    title: str = "Figure 4 - layer x position significance",
    value_label: str = "causal significance",
    diverging: bool | None = None,
) -> tuple[Path, ...]:
    """Figure 4: ``[n_layers, n_position_bins]``. Rows are layers, columns position bins."""
    arr = _matrix(matrix, (None, len(positions)), "layer x position")
    layers = list(range(arr.shape[0])) if layers is None else list(layers)
    if len(layers) != arr.shape[0]:
        raise ValueError(f"{len(layers)} layer labels for {arr.shape[0]} rows")
    signed = bool(np.nanmin(arr) < 0) if diverging is None else bool(diverging)
    with figure_style():
        fig, ax = plt.subplots(figsize=(6.4, 5.0))
        image = _heatmap(
            ax,
            arr,
            xticks=_position_labels(positions),
            yticks=layers,
            xlabel="normalized position bin",
            ylabel="decoder layer",
            diverging=signed,
        )
        ax.set_title(title)
        style.colorbar(fig, image, ax, label=value_label)
        return save_figure(fig, path)


def kv_head_position_heatmap(
    matrix: Any,
    path: str | Path,
    *,
    positions: Sequence[float],
    kv_heads: Sequence[int] | None = None,
    title: str = "Figure 5 - KV head x position significance",
    value_label: str = "causal significance",
    diverging: bool | None = None,
) -> tuple[Path, ...]:
    """Figure 5: ``[n_kv_heads, n_position_bins]``.

    KV heads, not query heads: SPEC section 5.5 is explicit that the two must never be
    conflated, and the Mode-2 gate this matrix comes from acts before ``repeat_kv``.
    """
    arr = _matrix(matrix, (None, len(positions)), "kv_head x position")
    kv_heads = list(range(arr.shape[0])) if kv_heads is None else list(kv_heads)
    if len(kv_heads) != arr.shape[0]:
        raise ValueError(f"{len(kv_heads)} KV-head labels for {arr.shape[0]} rows")
    signed = bool(np.nanmin(arr) < 0) if diverging is None else bool(diverging)
    with figure_style():
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        image = _heatmap(
            ax,
            arr,
            xticks=_position_labels(positions),
            yticks=kv_heads,
            xlabel="normalized position bin",
            ylabel="KV head",
            diverging=signed,
        )
        ax.set_title(title)
        style.colorbar(fig, image, ax, label=value_label)
        return save_figure(fig, path)


def layer_kv_head_significance(
    matrix: Any,
    path: str | Path,
    *,
    layers: Sequence[int] | None = None,
    kv_heads: Sequence[int] | None = None,
    title: str = "Figure 6 - layer x KV head causal significance",
    value_label: str = "causal significance",
    diverging: bool | None = None,
) -> tuple[Path, ...]:
    """Figure 6: ``[n_layers, n_kv_heads]``, position summed out."""
    arr = _matrix(matrix, (None, None), "layer x kv_head")
    layers = list(range(arr.shape[0])) if layers is None else list(layers)
    kv_heads = list(range(arr.shape[1])) if kv_heads is None else list(kv_heads)
    if len(layers) != arr.shape[0] or len(kv_heads) != arr.shape[1]:
        raise ValueError(
            f"labels {len(layers)}x{len(kv_heads)} do not match matrix {arr.shape}"
        )
    signed = bool(np.nanmin(arr) < 0) if diverging is None else bool(diverging)
    with figure_style():
        fig, ax = plt.subplots(figsize=(5.6, 5.0))
        image = _heatmap(
            ax,
            arr,
            xticks=kv_heads,
            yticks=layers,
            xlabel="KV head",
            ylabel="decoder layer",
            diverging=signed,
        )
        ax.set_title(title)
        style.colorbar(fig, image, ax, label=value_label)
        return save_figure(fig, path)


# --------------------------------------------------------------------------------------
# 7. Absolute position vs distance
# --------------------------------------------------------------------------------------


def absolute_vs_distance_effects(
    data: Any,
    path: str | Path,
    *,
    binning_col: str = "binning",
    bin_col: str = "bin_center",
    value_col: str = "effect",
    model_col: str | None = "model",
    ci_low_col: str | None = "ci_low",
    ci_high_col: str | None = "ci_high",
    title: str = "Figure 7 - absolute-position vs distance effects",
    ylabel: str = "effect on gold NLL (nats)",
) -> tuple[Path, ...]:
    """Figure 7: the two binnings side by side, never overlaid.

    SPEC section 5.3 keeps ``AbsoluteBinning`` (over ``j / T``) and ``DistanceBinning``
    (over ``(i - j) / T``) separate because they are different variables; plotting them on
    one x-axis would invite exactly the confusion the separation exists to prevent. Shared
    y so the *sizes* remain comparable, which is the question the figure answers.
    """
    cols = _columns(data)
    binnings = _column(cols, binning_col, numeric=False)
    centers = _column(cols, bin_col)
    values = _column(cols, value_col)
    low = _optional(cols, ci_low_col)
    high = _optional(cols, ci_high_col)
    models = _optional(cols, model_col, numeric=False)
    if models is None:
        models = np.array(["all"] * len(values))

    panels = _unique_in_order(binnings)
    labels = style.panel_order(_unique_in_order(models))
    colors = assign_colors(labels)
    markers = assign_markers(labels)

    with figure_style():
        fig, axes = plt.subplots(1, max(1, len(panels)), figsize=(8.6, 3.9), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, panel in zip(axes, panels, strict=False):
            for label in labels:
                mask = (binnings == panel) & (models == label)
                if not mask.any():
                    continue
                order = np.argsort(centers[mask])
                x, y = centers[mask][order], values[mask][order]
                ax.plot(
                    x, y, marker=markers[label], color=colors[label],
                    label=style.display_name(label), zorder=3,
                )
                if low is not None and high is not None:
                    ax.fill_between(
                        x, low[mask][order], high[mask][order],
                        color=colors[label], alpha=0.16, linewidth=0.0, zorder=2,
                    )
            style.zero_line(ax)
            ax.set_xlabel(
                "j / T (absolute)" if "abs" in panel.lower() else "(i - j) / T (distance)"
            )
            ax.set_title(panel)
        axes[0].set_ylabel(ylabel)
        drawn = _all_finite(values, low, high)
        for ax in axes:
            style.set_zero_based_ylim(ax, drawn)
            assert_no_truncated_axis(ax)
        fig.suptitle(title, fontsize=11, fontweight="bold")
        if len(labels) > 1:
            style.legend_outside(axes[-1], title="model")
        return save_figure(fig, path)


# --------------------------------------------------------------------------------------
# 8. Low-rank explained variance
# --------------------------------------------------------------------------------------


def _series_map(
    data: Any, *, label_col: str, value_col: str, index_col: str | None
) -> dict[str, np.ndarray]:
    """``{label: values}`` from a mapping of arrays, a bare array, or a tidy frame."""
    if isinstance(data, Mapping) and not data:
        raise ValueError("no explained-variance series to plot")
    if isinstance(data, Mapping) and data and all(
        not np.isscalar(v) and np.ndim(v) >= 1 for v in data.values()
    ):
        return {str(k): np.asarray(v, dtype=np.float64) for k, v in data.items()}
    arr = None
    if not isinstance(data, Mapping):
        try:
            arr = np.asarray(data, dtype=np.float64)
        except (TypeError, ValueError):
            arr = None
    if arr is not None and arr.ndim == 1:
        return {"all": arr}
    cols = _columns(data)
    labels = _column(cols, label_col, numeric=False)
    values = _column(cols, value_col)
    index = _optional(cols, index_col)
    out: dict[str, np.ndarray] = {}
    for label in _unique_in_order(labels):
        mask = labels == label
        order = np.argsort(index[mask]) if index is not None else np.arange(int(mask.sum()))
        out[label] = values[mask][order]
    return out


def low_rank_explained_variance(
    data: Any,
    path: str | Path,
    *,
    label_col: str = "model",
    value_col: str = "explained_variance_ratio",
    index_col: str | None = "component",
    threshold: float = 0.9,
    title: str = "Figure 8 - low-rank structure of the positional subspace",
    ylabel: str = "explained variance ratio",
) -> tuple[Path, ...]:
    """Figure 8: per-component ratio (bars, first series) and the cumulative curve (lines).

    The cumulative curve is what the rank claim rests on -- "k components explain 90%" --
    so the ``threshold`` line is drawn rather than left to the reader's arithmetic.
    """
    series = _series_map(data, label_col=label_col, value_col=value_col, index_col=index_col)
    if not series:
        raise ValueError("no explained-variance series to plot")
    labels = style.panel_order(list(series))
    colors = assign_colors(labels)
    markers = assign_markers(labels)

    with figure_style():
        fig, ax = plt.subplots()
        first = labels[0]
        components = np.arange(1, len(series[first]) + 1)
        ax.bar(
            components, series[first], width=0.6, color=colors[first], alpha=0.35,
            label=f"{style.display_name(first)} (per component)", zorder=2,
        )
        drawn: list[np.ndarray] = [np.asarray(series[first])]
        for label in labels:
            values = np.asarray(series[label], dtype=np.float64)
            cumulative = np.cumsum(values)
            ax.plot(
                np.arange(1, values.size + 1), cumulative, marker=markers[label],
                color=colors[label], label=f"{style.display_name(label)} (cumulative)", zorder=3,
            )
            drawn.append(cumulative)
        ax.axhline(threshold, color="#4D4D4D", linestyle="--", linewidth=0.9, zorder=1)
        ax.annotate(
            f"{threshold:.0%}", (0.5, threshold), textcoords="offset points",
            xytext=(0, 4), fontsize=8, color="#4D4D4D",
        )
        ax.set_xlabel("component")
        style.tidy_titles(ax, title=title, xlabel=None, ylabel=ylabel)
        style.set_accuracy_ylim(ax, np.concatenate(drawn))
        style.legend_outside(ax)
        assert_no_truncated_axis(ax)
        return save_figure(fig, path)


# --------------------------------------------------------------------------------------
# 9. Weight deltas
# --------------------------------------------------------------------------------------


def weight_delta_by_layer(
    data: Any,
    path: str | Path,
    *,
    model_col: str = "model_key",
    layer_col: str = "layer",
    family_col: str = "family",
    value_col: str = "rel_frobenius",
    families: Sequence[str] = ("q_proj", "k_proj", "v_proj"),
    title: str = "Figure 9 - Q/K/V weight-delta magnitude by layer",
    ylabel: str = "relative Frobenius delta vs base",
) -> tuple[Path, ...]:
    """Figure 9: one panel per projection, one line per model, over the decoder stack.

    Defaults match ``results/weight_delta_by_layer.csv`` as the Phase 1 audit writes it, so
    the figure builds straight off the artifact with no renaming step in between.
    """
    cols = _columns(data)
    models = _column(cols, model_col, numeric=False)
    layers = _column(cols, layer_col)
    fams = _column(cols, family_col, numeric=False)
    values = _column(cols, value_col)

    present = [f for f in families if (fams == f).any()]
    if not present:
        raise ValueError(f"none of {list(families)} appear in column {family_col!r}")
    labels = style.panel_order(_unique_in_order(models))
    colors = assign_colors(labels)
    markers = assign_markers(labels)

    with figure_style():
        width = 3.1 * len(present) + 1.4
        fig, axes = plt.subplots(1, len(present), figsize=(width, 3.7), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, family in zip(axes, present, strict=True):
            for label in labels:
                mask = (fams == family) & (models == label) & (layers >= 0)
                if not mask.any():
                    continue
                order = np.argsort(layers[mask])
                ax.plot(
                    layers[mask][order], values[mask][order], marker=markers[label],
                    markersize=3.5, color=colors[label], label=style.display_name(label),
                    zorder=3,
                )
            ax.set_title(family)
            ax.set_xlabel("decoder layer")
        axes[0].set_ylabel(ylabel)
        drawn = _all_finite(values[layers >= 0])
        for ax in axes:
            style.set_zero_based_ylim(ax, drawn)
            assert_no_truncated_axis(ax)
        fig.suptitle(title, fontsize=11, fontweight="bold")
        style.legend_outside(axes[-1], title="model")
        return save_figure(fig, path)


# --------------------------------------------------------------------------------------
# 10. Architecture comparison across arms
# --------------------------------------------------------------------------------------


def architecture_comparison(
    data: Any,
    path: str | Path,
    *,
    arm_col: str = "arm",
    value_col: str = "score",
    group_col: str | None = "metric",
    ci_low_col: str | None = "ci_low",
    ci_high_col: str | None = "ci_high",
    title: str = "Figure 10 - architecture comparison across arms",
    ylabel: str = "score",
) -> tuple[Path, ...]:
    """Figure 10: grouped bars over the sequential-attention arms (brief Phase 10 A-E)."""
    cols = _columns(data)
    arms = _column(cols, arm_col, numeric=False)
    values = _column(cols, value_col)
    low = _optional(cols, ci_low_col)
    high = _optional(cols, ci_high_col)
    groups = _optional(cols, group_col, numeric=False)
    if groups is None:
        groups = np.array(["score"] * len(values))

    arm_labels = _unique_in_order(arms)
    group_labels = _unique_in_order(groups)
    colors = assign_colors(group_labels)
    width = 0.8 / max(1, len(group_labels))

    with figure_style():
        fig, ax = plt.subplots(figsize=(7.4, 4.2))
        x = np.arange(len(arm_labels), dtype=np.float64)
        for gi, group in enumerate(group_labels):
            heights = np.full(len(arm_labels), np.nan)
            lo = np.full(len(arm_labels), np.nan)
            hi = np.full(len(arm_labels), np.nan)
            for ai, arm in enumerate(arm_labels):
                mask = (arms == arm) & (groups == group)
                if not mask.any():
                    continue
                heights[ai] = values[mask][0]
                if low is not None and high is not None:
                    lo[ai], hi[ai] = low[mask][0], high[mask][0]
            offset = (gi - (len(group_labels) - 1) / 2) * width
            drawable = np.nan_to_num(heights, nan=0.0)
            ax.bar(x + offset, drawable, width=width * 0.92, color=colors[group],
                   label=str(group), zorder=3)
            if np.isfinite(lo).any():
                err = _error_bars(drawable, np.nan_to_num(lo, nan=0.0), np.nan_to_num(hi, nan=0.0))
                ax.errorbar(x + offset, drawable, yerr=err, fmt="none", ecolor="#333333",
                            elinewidth=1.0, zorder=4)
        style.zero_line(ax)
        ax.set_xticks(x)
        ax.set_xticklabels(arm_labels, rotation=12, ha="right")
        style.tidy_titles(ax, title=title, xlabel=None, ylabel=ylabel)
        style.set_zero_based_ylim(ax, _all_finite(values, low, high))
        if len(group_labels) > 1:
            style.legend_outside(ax, title=str(group_col))
        assert_no_truncated_axis(ax)
        return save_figure(fig, path)


# --------------------------------------------------------------------------------------
# 11. Significance x sequentiality interaction
# --------------------------------------------------------------------------------------


def significance_sequentiality_interaction(
    data: Any,
    path: str | Path,
    *,
    sequentiality_col: str = "sequentiality",
    condition_col: str = "condition",
    value_col: str = "score",
    ci_low_col: str | None = "ci_low",
    ci_high_col: str | None = "ci_high",
    contrast: float | None = None,
    contrast_ci: tuple[float, float] | None = None,
    title: str = "Figure 11 - significance x sequentiality interaction",
    ylabel: str = "score",
) -> tuple[Path, ...]:
    """Figure 11: the brief's section 48 difference-in-differences, drawn as an interaction.

    Non-parallel lines *are* the interaction, so the point estimate and its interval are
    annotated on the figure: the eye reads a crossing as decisive far more readily than the
    numbers justify.
    """
    cols = _columns(data)
    seqs = _column(cols, sequentiality_col, numeric=False)
    conditions = _column(cols, condition_col, numeric=False)
    values = _column(cols, value_col)
    low = _optional(cols, ci_low_col)
    high = _optional(cols, ci_high_col)

    cond_labels = _unique_in_order(conditions)
    seq_labels = _unique_in_order(seqs)
    colors = assign_colors(seq_labels)
    markers = assign_markers(seq_labels)

    with figure_style():
        fig, ax = plt.subplots()
        x = np.arange(len(cond_labels), dtype=np.float64)
        for label in seq_labels:
            ys, los, his = [], [], []
            for cond in cond_labels:
                mask = (seqs == label) & (conditions == cond)
                ys.append(float(values[mask][0]) if mask.any() else np.nan)
                if low is not None and high is not None and mask.any():
                    los.append(float(low[mask][0]))
                    his.append(float(high[mask][0]))
            y = np.asarray(ys)
            err = (
                _error_bars(y, np.asarray(los), np.asarray(his))
                if len(los) == len(cond_labels)
                else None
            )
            ax.errorbar(
                x, y, yerr=err, marker=markers[label], color=colors[label],
                label=str(label), elinewidth=1.1, zorder=3,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(cond_labels)
        ax.set_xlim(-0.35, len(cond_labels) - 0.65)
        style.tidy_titles(ax, title=title, xlabel="condition", ylabel=ylabel)
        style.set_zero_based_ylim(ax, _all_finite(values, low, high))
        if contrast is not None:
            ci = "" if contrast_ci is None else f" [{contrast_ci[0]:+.3f}, {contrast_ci[1]:+.3f}]"
            ax.annotate(
                f"interaction contrast {contrast:+.3f}{ci}",
                xy=(0.02, 0.02), xycoords="axes fraction", fontsize=9, color="#333333",
            )
        style.legend_outside(ax, title=str(sequentiality_col))
        assert_no_truncated_axis(ax)
        return save_figure(fig, path)


# --------------------------------------------------------------------------------------
# 12-13. Trade-off scatters
# --------------------------------------------------------------------------------------


def accuracy_vs_latency(
    data: Any,
    path: str | Path,
    *,
    label_col: str = "model",
    latency_col: str = "latency_s",
    value_col: str = "accuracy",
    group_col: str | None = None,
    title: str = "Figure 12 - accuracy vs latency",
    xlabel: str = "median latency per item (s)",
    ylabel: str = "accuracy",
) -> tuple[Path, ...]:
    """Figure 12: the cost of every intervention, plotted against what it buys.

    Both axes include zero. Latency especially: a truncated x-axis turns "12% slower" into
    a visual doubling, which is precisely how an intervention gets accepted on a slide.
    """
    cols = _columns(data)
    labels_all = _column(cols, label_col, numeric=False)
    latency = _column(cols, latency_col)
    values = _column(cols, value_col)
    groups = _optional(cols, group_col, numeric=False)
    if groups is None:
        groups = labels_all

    group_labels = style.panel_order(_unique_in_order(groups))
    colors = assign_colors(group_labels)
    markers = assign_markers(group_labels)

    with figure_style():
        fig, ax = plt.subplots()
        for group in group_labels:
            mask = groups == group
            ax.scatter(
                latency[mask], values[mask], s=52, color=colors[group],
                marker=markers[group], label=style.display_name(group), zorder=3,
            )
        style.annotate_points(ax, latency, values, labels_all)
        style.tidy_titles(ax, title=title, xlabel=xlabel, ylabel=ylabel)
        style.set_accuracy_ylim(ax, values)
        ax.set_xlim(0.0, float(np.nanmax(latency)) * 1.18 + 1e-9)
        assert_no_truncated_axis(ax, axis="y")
        assert_no_truncated_axis(ax, axis="x")
        style.legend_outside(ax)
        return save_figure(fig, path)


def worst_vs_mean_accuracy(
    data: Any,
    path: str | Path,
    *,
    label_col: str = "model",
    mean_col: str = "mean_accuracy",
    worst_col: str = "worst_accuracy",
    group_col: str | None = None,
    title: str = "Figure 13 - worst-position vs mean accuracy",
    xlabel: str = "mean accuracy over the position grid",
    ylabel: str = "worst-position accuracy",
) -> tuple[Path, ...]:
    """Figure 13: the robustness view. Distance below the diagonal is positional fragility.

    Two systems with the same mean sit at different heights here, which is the whole
    argument for reporting worst-position accuracy alongside the mean.
    """
    cols = _columns(data)
    labels_all = _column(cols, label_col, numeric=False)
    mean = _column(cols, mean_col)
    worst = _column(cols, worst_col)
    groups = _optional(cols, group_col, numeric=False)
    if groups is None:
        groups = labels_all

    group_labels = style.panel_order(_unique_in_order(groups))
    colors = assign_colors(group_labels)
    markers = assign_markers(group_labels)

    with figure_style():
        fig, ax = plt.subplots(figsize=(5.4, 5.0))
        ax.plot([0, 1], [0, 1], color="#8C8C8C", linestyle="--", linewidth=0.9, zorder=1)
        for group in group_labels:
            mask = groups == group
            ax.scatter(
                mean[mask], worst[mask], s=56, color=colors[group], marker=markers[group],
                label=style.display_name(group), zorder=3,
            )
        style.annotate_points(ax, mean, worst, labels_all)
        style.tidy_titles(ax, title=title, xlabel=xlabel, ylabel=ylabel)
        style.set_accuracy_ylim(ax, worst)
        ax.set_xlim(0.0, max(1.0, float(np.nanmax(mean))) + 0.02)
        assert_no_truncated_axis(ax, axis="y")
        assert_no_truncated_axis(ax, axis="x")
        style.legend_outside(ax)
        return save_figure(fig, path)


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------

#: Canonical filename stem -> builder, in the order brief section 47 lists them. The stem is
#: the artifact name; ``save_figure`` appends ``.png`` and ``.pdf``.
FIGURE_ORDER: tuple[tuple[str, Callable[..., tuple[Path, ...]]], ...] = (
    ("fig01_accuracy_vs_position", accuracy_vs_position),
    ("fig02_nll_vs_position", nll_vs_position),
    ("fig03_edge_middle_gap", edge_middle_gap_by_model),
    ("fig04_layer_position_heatmap", layer_position_heatmap),
    ("fig05_kv_head_position_heatmap", kv_head_position_heatmap),
    ("fig06_layer_kv_head_significance", layer_kv_head_significance),
    ("fig07_absolute_vs_distance", absolute_vs_distance_effects),
    ("fig08_low_rank_explained_variance", low_rank_explained_variance),
    ("fig09_weight_delta_by_layer", weight_delta_by_layer),
    ("fig10_architecture_comparison", architecture_comparison),
    ("fig11_significance_sequentiality", significance_sequentiality_interaction),
    ("fig12_accuracy_vs_latency", accuracy_vs_latency),
    ("fig13_worst_vs_mean_accuracy", worst_vs_mean_accuracy),
)

FIGURES: dict[str, Callable[..., tuple[Path, ...]]] = dict(FIGURE_ORDER)
