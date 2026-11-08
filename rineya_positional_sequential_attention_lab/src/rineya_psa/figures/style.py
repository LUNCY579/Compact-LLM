"""One shared matplotlib style for every figure in brief section 47.

Three rules the brief makes non-negotiable, and how they are enforced rather than merely
recommended:

1. **No truncated axes.** :func:`assert_no_truncated_axis` raises
   :class:`TruncatedAxisError` when a value-carrying axis does not include zero, or when a
   drawn point falls outside the limits. Every accuracy-like plot in
   :mod:`rineya_psa.figures.plots` calls it before saving, so a figure with a magnified
   difference cannot be written to disk by accident. A study whose whole claim is
   "the middle is worse" must not exaggerate the drop with a y-axis starting at 0.4.
2. **Colourblind-safe categorical colour.** The palette is Okabe-Ito, minus its yellow,
   which is unreadable as a line on a white background. Each panel model keeps the same
   colour in every figure (:data:`MODEL_COLORS`), so a reader can track ``halo`` across
   thirteen plots without consulting a legend each time.
3. **Print-legible, light background.** 200 dpi PNG *and* vector PDF from one call
   (:func:`save_figure`), 9-11 pt type, no chartjunk, no dark theme.

The style is applied through :func:`figure_style`, a context manager, not by mutating the
global rcParams at import: a lab module that changes matplotlib's global state on import
would silently reformat any other plotting a caller does in the same process.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

# Agg unless the process has already committed to something else: the lab runs headless,
# but ``force=False`` means an interactive session keeps its own backend.
matplotlib.use("Agg", force=False)

import matplotlib.pyplot as plt  # noqa: E402 - must follow the backend selection

from rineya_psa.paths import MODELS, PANEL, assert_readonly_target

__all__ = [
    "DPI",
    "DIVERGING_CMAP",
    "MODEL_COLORS",
    "MODEL_MARKERS",
    "PALETTE",
    "RC",
    "SEQUENTIAL_CMAP",
    "TruncatedAxisError",
    "apply_style",
    "assign_colors",
    "assign_markers",
    "assert_no_truncated_axis",
    "color_for",
    "display_name",
    "figure_style",
    "marker_for",
    "save_figure",
    "set_accuracy_ylim",
    "set_zero_based_ylim",
]

#: Okabe-Ito without ``#F0E442`` (yellow), which fails as a 1.8 pt line on white.
PALETTE: tuple[str, ...] = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#000000",  # black
)

#: Fixed per-model colour, so ``halo`` is the same vermillion in all thirteen figures.
MODEL_COLORS: dict[str, str] = {
    "qwen3": "#0072B2",
    "halo": "#D55E00",
    "nimbus": "#009E73",
    "supa_a11": "#CC79A7",
}

#: Redundant encoding: colour alone is never the only channel carrying series identity.
MODEL_MARKERS: dict[str, str] = {
    "qwen3": "o",
    "halo": "s",
    "nimbus": "^",
    "supa_a11": "D",
}

MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X", "*")

DPI: int = 200
SEQUENTIAL_CMAP: str = "viridis"
DIVERGING_CMAP: str = "RdBu_r"

GRID_COLOR: str = "#D9D9D9"
TEXT_COLOR: str = "#1A1A1A"

RC: dict[str, Any] = {
    "figure.figsize": (7.0, 4.2),
    "figure.facecolor": "white",
    "figure.dpi": 110,
    "savefig.dpi": DPI,
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "axes.facecolor": "white",
    "axes.edgecolor": "#4D4D4D",
    "axes.labelcolor": TEXT_COLOR,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.prop_cycle": plt.cycler(color=list(PALETTE)),
    "grid.color": GRID_COLOR,
    "grid.linewidth": 0.7,
    "grid.alpha": 0.9,
    "lines.linewidth": 1.8,
    "lines.markersize": 5.0,
    "lines.markeredgewidth": 0.0,
    "font.size": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "legend.frameon": False,
    "legend.fontsize": 9,
    "legend.handlelength": 1.6,
    "text.color": TEXT_COLOR,
    "image.cmap": SEQUENTIAL_CMAP,
    "errorbar.capsize": 2.5,
}


def apply_style() -> None:
    """Install the style globally. For scripts and notebooks; library code uses the CM."""
    plt.rcParams.update(RC)


@contextmanager
def figure_style(**overrides: Any) -> Iterator[None]:
    """Scoped rcParams. Create the figure inside, or half the settings will not apply."""
    params = dict(RC)
    params.update(overrides)
    with plt.rc_context(params):
        yield


# --------------------------------------------------------------------------------------
# Colour assignment
# --------------------------------------------------------------------------------------


def display_name(key: str) -> str:
    """Panel display name if ``key`` is a model, otherwise the key unchanged."""
    spec = MODELS.get(str(key))
    return spec.display_name if spec is not None else str(key)


def color_for(label: str) -> str:
    """Stable colour for one series, hashed when the label is not a panel model.

    Hashing rather than counting so the same label gets the same colour across processes
    and figures. Prefer :func:`assign_colors` when plotting several series at once: it
    guarantees distinct colours, which a hash cannot.
    """
    key = str(label)
    if key in MODEL_COLORS:
        return MODEL_COLORS[key]
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return PALETTE[int.from_bytes(digest[:4], "big") % len(PALETTE)]


def marker_for(label: str) -> str:
    key = str(label)
    if key in MODEL_MARKERS:
        return MODEL_MARKERS[key]
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return MARKERS[int.from_bytes(digest[4:8], "big") % len(MARKERS)]


def assign_colors(labels: Sequence[str]) -> dict[str, str]:
    """Distinct colours for a set of series, with panel models pinned to theirs.

    Unknown labels take the first unused palette entry in the order given, so two arms in
    one figure can never collide -- which is the failure mode :func:`color_for` cannot rule
    out. Past seven series the palette wraps and colour stops being the distinguishing
    channel; the marker does the work instead.
    """
    labels = [str(x) for x in labels]
    out: dict[str, str] = {}
    used: set[str] = set()
    for label in labels:
        if label in MODEL_COLORS:
            out[label] = MODEL_COLORS[label]
            used.add(out[label])
    spare = [c for c in PALETTE if c not in used]
    cursor = 0
    for label in labels:
        if label in out:
            continue
        if spare:
            out[label] = spare[cursor % len(spare)]
        else:
            out[label] = PALETTE[cursor % len(PALETTE)]
        cursor += 1
    return out


def assign_markers(labels: Sequence[str]) -> dict[str, str]:
    labels = [str(x) for x in labels]
    out: dict[str, str] = {}
    cursor = 0
    for label in labels:
        if label in MODEL_MARKERS:
            out[label] = MODEL_MARKERS[label]
        else:
            out[label] = MARKERS[cursor % len(MARKERS)]
            cursor += 1
    return out


def panel_order(labels: Iterable[str]) -> list[str]:
    """Panel models first in :data:`~rineya_psa.paths.PANEL` order, then the rest as given."""
    labels = [str(x) for x in labels]
    known = [k for k in PANEL if k in labels]
    return known + [x for x in labels if x not in known]


# --------------------------------------------------------------------------------------
# The no-truncated-axis guard (brief section 47)
# --------------------------------------------------------------------------------------


class TruncatedAxisError(AssertionError):
    """Raised when an axis hides zero or clips data. Deliberately an ``AssertionError``.

    It is a violated invariant, not a bad argument: by the time it fires the figure is
    already drawn, and the only correct response is to fix the limits and redraw.
    """


def _axis_limits(ax: Any, axis: str) -> tuple[float, float]:
    lo, hi = ax.get_ylim() if axis == "y" else ax.get_xlim()
    return (float(min(lo, hi)), float(max(lo, hi)))


def _drawn_values(ax: Any, axis: str) -> np.ndarray:
    """Every value actually rendered on ``axis``: lines, scatters, and bar extents."""
    index = 1 if axis == "y" else 0
    chunks: list[np.ndarray] = []
    for line in ax.get_lines():
        data = line.get_ydata() if axis == "y" else line.get_xdata()
        arr = np.asarray(data, dtype=np.float64).ravel()
        if arr.size:
            chunks.append(arr)
    for coll in ax.collections:
        try:
            offsets = np.asarray(coll.get_offsets(), dtype=np.float64)
        except (AttributeError, ValueError, TypeError):
            continue
        if offsets.ndim == 2 and offsets.shape[1] == 2 and offsets.size:
            chunks.append(offsets[:, index])
    for patch in ax.patches:
        if not hasattr(patch, "get_height"):
            continue
        if axis == "y":
            base, extent = float(patch.get_y()), float(patch.get_height())
        else:
            base, extent = float(patch.get_x()), float(patch.get_width())
        chunks.append(np.array([base, base + extent], dtype=np.float64))
    if not chunks:
        return np.empty(0, dtype=np.float64)
    stacked = np.concatenate(chunks)
    return stacked[np.isfinite(stacked)]


def assert_no_truncated_axis(
    ax: Any,
    *,
    axis: str = "y",
    require_zero: bool = True,
    check_data: bool = True,
    tol: float = 1e-9,
) -> None:
    """Brief section 47's guard. Call it on every accuracy-like plot before saving.

    ``require_zero`` is the headline check: the axis must span zero, so bar heights and
    curve drops are read at their true relative size. ``check_data`` catches the other
    kind of truncation -- limits that cut a drawn point off the canvas.
    """
    if axis not in {"x", "y"}:
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
    lo, hi = _axis_limits(ax, axis)
    if require_zero and not (lo <= tol and hi >= -tol):
        raise TruncatedAxisError(
            f"{axis}-axis limits ({lo:.6g}, {hi:.6g}) exclude zero: brief section 47 forbids "
            "a truncated axis on a magnitude plot. Use set_zero_based_ylim / "
            "set_accuracy_ylim, or pass require_zero=False if this axis is not a magnitude."
        )
    if not check_data:
        return
    values = _drawn_values(ax, axis)
    if values.size == 0:
        return
    below = values.min() < lo - tol
    above = values.max() > hi + tol
    if below or above:
        raise TruncatedAxisError(
            f"{axis}-axis limits ({lo:.6g}, {hi:.6g}) clip drawn data spanning "
            f"({values.min():.6g}, {values.max():.6g})"
        )


def set_zero_based_ylim(
    ax: Any, values: Any, *, pad: float = 0.08, upper: float | None = None
) -> tuple[float, float]:
    """Limits that always contain zero and all of ``values``, with a little headroom."""
    arr = np.asarray(list(np.ravel(values)), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    lo = float(min(0.0, arr.min())) if arr.size else 0.0
    hi = float(max(0.0, arr.max())) if arr.size else 1.0
    span = hi - lo
    margin = pad * span if span > 0 else max(abs(hi), 1.0) * pad
    top = hi + margin if upper is None else float(upper)
    bottom = lo - (margin if lo < 0 else 0.0)
    ax.set_ylim(bottom, max(top, bottom + 1e-9))
    return ax.get_ylim()


def set_accuracy_ylim(ax: Any, values: Any = (), *, headroom: float = 0.02) -> tuple[float, float]:
    """``(0, 1 + headroom)`` for a proportion, widened if the data (or its CI) exceeds it."""
    arr = np.asarray(list(np.ravel(values)), dtype=np.float64) if np.size(values) else np.zeros(1)
    arr = arr[np.isfinite(arr)]
    top = max(1.0, float(arr.max()) if arr.size else 1.0) + headroom
    bottom = min(0.0, float(arr.min()) if arr.size else 0.0)
    ax.set_ylim(bottom, top)
    return ax.get_ylim()


# --------------------------------------------------------------------------------------
# Saving
# --------------------------------------------------------------------------------------


def save_figure(
    fig: Any,
    path: str | Path,
    *,
    dpi: int = DPI,
    formats: Sequence[str] = ("png", "pdf"),
    close: bool = True,
) -> tuple[Path, ...]:
    """Write ``path`` once per format -- 200 dpi raster for review, vector for print.

    Any suffix on ``path`` is treated as a stem hint and replaced, so callers can pass
    ``figures/fig01_accuracy.png`` and still get the PDF. Every target goes through
    ``paths.assert_readonly_target`` **before** the parent directory is created: a figure
    path that pointed into a checkpoint must not leave an empty directory behind on its way
    to raising.
    """
    if not formats:
        raise ValueError("save_figure needs at least one output format")
    stem = Path(path).with_suffix("")
    suffixes = [str(fmt).lstrip(".") for fmt in formats]
    targets = [(stem.with_suffix(f".{suffix}"), suffix) for suffix in suffixes]
    for out, _fmt in targets:
        assert_readonly_target(out)

    stem.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for out, fmt in targets:
        fig.savefig(out, dpi=dpi, format=fmt)
        written.append(out)
    if close:
        plt.close(fig)
    return tuple(written)


# --------------------------------------------------------------------------------------
# Small shared drawing helpers
# --------------------------------------------------------------------------------------


def position_axis(ax: Any, *, label: str = "normalized evidence position") -> None:
    """The x-axis every positional figure shares: 0 to 1, edges and middle ticked."""
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel(label)


def zero_line(ax: Any, *, axis: str = "y") -> None:
    """A hairline at zero, so a signed quantity is read against its reference."""
    if axis == "y":
        ax.axhline(0.0, color="#4D4D4D", linewidth=0.8, zorder=1)
    else:
        ax.axvline(0.0, color="#4D4D4D", linewidth=0.8, zorder=1)


def annotate_points(
    ax: Any, xs: Sequence[float], ys: Sequence[float], labels: Sequence[str], *, dx: float = 0.0
) -> None:
    for x, y, label in zip(xs, ys, labels, strict=True):
        ax.annotate(
            str(label),
            (float(x), float(y)),
            textcoords="offset points",
            xytext=(6 + dx, 4),
            fontsize=8,
            color=TEXT_COLOR,
        )


def legend_outside(ax: Any, *, title: str | None = None, ncol: int = 1) -> Any:
    """Legend to the right of the axes, where it cannot cover a data point."""
    return ax.legend(
        loc="upper left", bbox_to_anchor=(1.01, 1.0), title=title, ncol=ncol, borderaxespad=0.0
    )


def colorbar(fig: Any, mappable: Any, ax: Any, *, label: str) -> Any:
    bar = fig.colorbar(mappable, ax=ax, pad=0.02, fraction=0.046)
    bar.set_label(label, fontsize=9)
    bar.ax.tick_params(labelsize=8)
    return bar


def symmetric_norm(values: Any) -> tuple[float, float]:
    """``(-m, m)`` for a signed matrix, so a diverging colormap's white sits at zero."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    m = float(np.max(np.abs(arr))) if arr.size else 1.0
    m = m if m > 0 else 1.0
    return -m, m


def tidy_titles(ax: Any, *, title: str | None, xlabel: str | None, ylabel: str | None) -> None:
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def style_manifest() -> Mapping[str, Any]:
    """The style as data, for the provenance sidecar of a figure bundle."""
    return {
        "palette": list(PALETTE),
        "model_colors": dict(MODEL_COLORS),
        "model_markers": dict(MODEL_MARKERS),
        "dpi": DPI,
        "formats": ["png", "pdf"],
        "sequential_cmap": SEQUENTIAL_CMAP,
        "diverging_cmap": DIVERGING_CMAP,
        "no_truncated_axes": True,
        "matplotlib": matplotlib.__version__,
    }
