"""The ``[layer, kv_head, bin]`` container every downstream phase consumes.

Why a container and not a bare ``ndarray``: a significance matrix is only meaningful
next to the binning that produced it, the objective it was measured on, and the
checkpoint it came from. Losing any of the three turns a legitimate intervention
table into a plausible-looking wrong one, and nothing downstream would notice.

The **control matrices of brief section 37** live here rather than in a separate
``controls.py`` on purpose. A positive result from ``to_logit_bias_spec`` is
uninterpretable without the matched controls, so they sit on the same object, one
method call away, and cannot be forgotten when a sweep is written in a hurry.

Torch-free by design (SPEC section 2): ``rineya_psa.hooks`` is imported lazily inside
``to_logit_bias_spec`` / ``to_value_scale_spec``, the only two methods that need it.

DEVIATION: SPEC section 9 routes artifacts through ``statistics.provenance.write_result``,
which handles ``.json/.jsonl/.csv/.parquet`` but not ``.npz``. ``save_npz`` therefore
writes the archive itself and emits the same ``*.provenance.json`` sidecar by calling
``stamp()`` directly, so the artifact contract of SPEC section 9 still holds.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rineya_psa.paths import assert_readonly_target
from rineya_psa.statistics.provenance import json_default, stamp

__all__ = [
    "NORMALISE_MODES",
    "OBJECTIVES",
    "OBJECTIVE_SIGN",
    "SignificanceMatrix",
    "bin_centers",
    "delta_matrices",
    "write_npz_with_provenance",
]

#: Objectives a significance matrix can be measured on (brief section 11).
OBJECTIVES: tuple[str, ...] = ("gold_nll", "margin", "correct")

#: ``importance = OBJECTIVE_SIGN[objective] * Delta``.
#:
#: ``Delta = L_normal - L_perturbed`` is the brief's literal definition, so for a
#: *loss* objective an important cell produces a **negative** Delta. Every ranking in
#: the lab goes through :meth:`SignificanceMatrix.importance`, which applies this sign,
#: so "large" always means "matters" regardless of objective.
OBJECTIVE_SIGN: Mapping[str, float] = {"gold_nll": -1.0, "margin": 1.0, "correct": 1.0}

NORMALISE_MODES: tuple[str, ...] = (
    "none",
    "center",
    "zscore",
    "max_abs",
    "per_head_center",
    "per_head_zscore",
    "per_layer_zscore",
    "rank",
)


def bin_centers(binning_desc: Mapping[str, Any]) -> np.ndarray:
    """Normalised centre of every bin described by ``binning_desc``.

    Absolute bins tile ``[0, 1]`` evenly. Distance bins are defined by cut points with
    two unbounded outer bins; those get a centre one half-width outside the outermost
    finite edge, which keeps the sequence monotone and finite for plotting and for the
    middle/edge summaries.
    """
    kind = binning_desc.get("kind")
    if kind == "absolute":
        n = int(binning_desc["n_bins"])
        return (np.arange(n, dtype=np.float64) + 0.5) / float(n)
    if kind == "distance":
        edges = np.asarray([float(x) for x in binning_desc["edges"]], dtype=np.float64)
        if edges.size == 0:
            raise ValueError("distance binning with no edges has no centres")
        half = 0.5 if edges.size == 1 else 0.5 * float(np.min(np.diff(edges)))
        inner = 0.5 * (edges[:-1] + edges[1:]) if edges.size > 1 else np.empty(0)
        return np.concatenate([[edges[0] - half], inner, [edges[-1] + half]])
    raise ValueError(f"cannot place bin centres for binning kind {kind!r}")


def _as_values(values: Any, shape: tuple[int, int, int]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"values must be {shape}, got {array.shape}")
    return np.ascontiguousarray(array)


def _index_tuple(items: Sequence[int] | None, n: int, name: str) -> tuple[int, ...]:
    if items is None:
        return tuple(range(n))
    out = tuple(int(x) for x in items)
    if len(out) != n:
        raise ValueError(f"{name} has {len(out)} entries but the axis is {n} long")
    if len(set(out)) != len(out):
        raise ValueError(f"{name} contains duplicates: {out}")
    return out


@dataclass(frozen=True, eq=False)
class SignificanceMatrix:
    """``values[l, h, b]`` plus everything needed to interpret or replay it.

    ``eq=False`` for the reason ``hooks.specs`` gives: the payload is an array, and a
    generated ``__eq__`` would return an array from ``==`` and silently poison ``in``.
    """

    values: np.ndarray  # [n_layers, n_heads, n_bins]
    layers: tuple[int, ...]
    kv_heads: tuple[int, ...]
    bins: tuple[int, ...]
    binning_desc: Mapping[str, Any]
    objective: str
    model_key: str
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim != 3:
            raise ValueError(f"values must be 3-D [L, H, B], got {values.shape}")
        object.__setattr__(self, "values", np.ascontiguousarray(values))
        object.__setattr__(
            self, "layers", _index_tuple(self.layers, values.shape[0], "layers")
        )
        object.__setattr__(
            self, "kv_heads", _index_tuple(self.kv_heads, values.shape[1], "kv_heads")
        )
        object.__setattr__(self, "bins", _index_tuple(self.bins, values.shape[2], "bins"))
        object.__setattr__(self, "binning_desc", dict(self.binning_desc))
        object.__setattr__(self, "provenance", dict(self.provenance))
        if int(self.binning_desc.get("n_bins", values.shape[2])) != values.shape[2]:
            raise ValueError(
                f"binning declares {self.binning_desc.get('n_bins')} bins but values has "
                f"{values.shape[2]}"
            )

    # -- basic accessors ------------------------------------------------------

    @property
    def n_layers(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_heads(self) -> int:
        return int(self.values.shape[1])

    @property
    def n_bins(self) -> int:
        return int(self.values.shape[2])

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.n_layers, self.n_heads, self.n_bins)

    @property
    def bin_centers(self) -> np.ndarray:
        return bin_centers(self.binning_desc)

    def importance(self) -> np.ndarray:
        """``Delta`` re-signed so that larger always means "this cell matters more"."""
        return float(OBJECTIVE_SIGN.get(self.objective, 1.0)) * self.values

    def unfold(self) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
        """``([l*h, b], row_keys)`` — the matrix brief section 21 takes the SVD of."""
        rows = tuple((lay, h) for lay in self.layers for h in self.kv_heads)
        return self.values.reshape(self.n_layers * self.n_heads, self.n_bins).copy(), rows

    def replace(self, **changes: Any) -> SignificanceMatrix:
        """Copy with fields overridden. Frozen dataclasses need an explicit one here
        because ``dataclasses.replace`` would re-run validation on a shared array."""
        fields: dict[str, Any] = {
            "values": self.values,
            "layers": self.layers,
            "kv_heads": self.kv_heads,
            "bins": self.bins,
            "binning_desc": self.binning_desc,
            "objective": self.objective,
            "model_key": self.model_key,
            "provenance": self.provenance,
        }
        fields.update(changes)
        return SignificanceMatrix(**fields)

    def _derived(self, values: np.ndarray, tag: str, **extra: Any) -> SignificanceMatrix:
        prov = dict(self.provenance)
        prov["derived_from"] = {
            "model_key": self.model_key,
            "objective": self.objective,
            "transform": tag,
            "source_sha256": self.content_hash(),
            **extra,
        }
        return self.replace(values=values, provenance=prov)

    def content_hash(self) -> str:
        """sha256 of the values plus the identifying metadata. Cheap identity for logs."""
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.values, dtype=np.float64).tobytes())
        h.update(
            json.dumps(
                {
                    "layers": list(self.layers),
                    "kv_heads": list(self.kv_heads),
                    "bins": list(self.bins),
                    "binning": dict(self.binning_desc),
                    "objective": self.objective,
                    "model_key": self.model_key,
                },
                sort_keys=True,
            ).encode("utf-8")
        )
        return h.hexdigest()

    def summary(self) -> dict[str, Any]:
        imp = self.importance()
        flat = imp.reshape(-1)
        finite = flat[np.isfinite(flat)]
        best = (
            np.unravel_index(int(np.nanargmax(imp)), imp.shape) if finite.size else (0, 0, 0)
        )
        return {
            "model_key": self.model_key,
            "objective": self.objective,
            "shape": list(self.shape),
            "binning": dict(self.binning_desc),
            "n_finite": int(finite.size),
            "mean": float(finite.mean()) if finite.size else float("nan"),
            "sd": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
            "max_importance": float(finite.max()) if finite.size else float("nan"),
            "argmax_cell": {
                "layer": int(self.layers[int(best[0])]),
                "kv_head": int(self.kv_heads[int(best[1])]),
                "bin": int(self.bins[int(best[2])]),
            },
            "content_sha256": self.content_hash(),
        }

    def top_cells(self, k: int = 16) -> list[tuple[tuple[int, int, int], float]]:
        """The ``k`` cells with the largest importance, as ``((layer, kv_head, bin), value)``."""
        imp = self.importance()
        flat = imp.reshape(-1)
        order = np.argsort(np.where(np.isfinite(flat), flat, -np.inf))[::-1][: int(k)]
        out: list[tuple[tuple[int, int, int], float]] = []
        for idx in order:
            li, hi, bi = np.unravel_index(int(idx), imp.shape)
            if not np.isfinite(flat[idx]):
                continue
            cell = (int(self.layers[li]), int(self.kv_heads[hi]), int(self.bins[bi]))
            out.append((cell, float(flat[idx])))
        return out

    # -- normalisation --------------------------------------------------------

    def normalise(self, mode: str = "zscore") -> SignificanceMatrix:
        """Rescale ``values`` without changing what the matrix means.

        ``per_head_*`` modes are the ones to use before an SVD when the interest is the
        *shape* of each head's positional profile rather than its amplitude; the global
        modes preserve the relative weight of heads and are what an intervention table
        wants. NaNs (unevaluated cells in a coarse-to-fine ladder) are ignored by every
        statistic and propagate unchanged.
        """
        if mode not in NORMALISE_MODES:
            raise ValueError(f"unknown normalise mode {mode!r}; known: {list(NORMALISE_MODES)}")
        v = self.values
        if mode == "none":
            out = v.copy()
        elif mode == "center":
            out = v - np.nanmean(v)
        elif mode == "zscore":
            sd = np.nanstd(v)
            out = (v - np.nanmean(v)) / (sd if sd > 0 else 1.0)
        elif mode == "max_abs":
            peak = np.nanmax(np.abs(v)) if np.isfinite(v).any() else 0.0
            out = v / (peak if peak > 0 else 1.0)
        elif mode == "per_head_center":
            out = v - np.nanmean(v, axis=2, keepdims=True)
        elif mode == "per_head_zscore":
            mu = np.nanmean(v, axis=2, keepdims=True)
            sd = np.nanstd(v, axis=2, keepdims=True)
            out = (v - mu) / np.where(sd > 0, sd, 1.0)
        elif mode == "per_layer_zscore":
            mu = np.nanmean(v, axis=(1, 2), keepdims=True)
            sd = np.nanstd(v, axis=(1, 2), keepdims=True)
            out = (v - mu) / np.where(sd > 0, sd, 1.0)
        else:  # "rank"
            out = _rank_normalise(v)
        return self._derived(out, f"normalise:{mode}")

    # -- intervention specs ---------------------------------------------------

    def _binning_object(self) -> Any:
        from rineya_psa.hooks.binning import binning_from_describe

        return binning_from_describe(dict(self.binning_desc))

    def _dense_table(self, arch: Any | None) -> np.ndarray:
        """Scatter this (possibly partial) matrix into a full ``[L, H_kv, B]`` table.

        Cells the matrix does not cover, and cells it covers with NaN, stay at zero:
        an intervention must never be driven by a number that was never measured.
        """
        if arch is None:
            n_layers = max(self.layers) + 1
            n_heads = max(self.kv_heads) + 1
        else:
            n_layers, n_heads = int(arch.num_layers), int(arch.num_kv_heads)
            if max(self.layers) >= n_layers or max(self.kv_heads) >= n_heads:
                raise ValueError(
                    f"matrix indexes layer {max(self.layers)} / kv_head {max(self.kv_heads)} "
                    f"but the model has {n_layers} layers and {n_heads} KV heads"
                )
        table = np.zeros((n_layers, n_heads, self.n_bins), dtype=np.float64)
        block = np.nan_to_num(self.values, nan=0.0, posinf=0.0, neginf=0.0)
        for li, layer in enumerate(self.layers):
            for hi, head in enumerate(self.kv_heads):
                table[layer, head, :] = block[li, hi, :]
        return table

    def to_logit_bias_spec(
        self,
        lam: float = 1.0,
        *,
        arch: Any | None = None,
        normalise_mode: str = "max_abs",
        layers: Sequence[int] | None = None,
    ) -> Any:
        """Mode 1 spec: ``z' = z + lam * S[l, kv(g), b(j)]``.

        The matrix is normalised first (``max_abs`` by default) so ``lam`` is in nats of
        logit and means the same thing across objectives and checkpoints. Pass
        ``normalise_mode="none"`` to use raw Deltas.

        ``importance`` and not ``values``, for the same reason
        :meth:`to_value_scale_spec` gives: ``Delta = L_normal - L_perturbed`` is negative
        at an important cell on a loss objective, so scattering the raw Deltas into a
        logit bias *penalises* attention to exactly the bins the matrix says matter most.
        This method used to do that, which made it point in the opposite direction to its
        Mode-2 sibling and turned ``sign_flip()`` — a negative control — into the
        correctly-signed intervention. ``tests/test_significance.py`` now pins the
        direction against ``top_cells``.
        """
        from rineya_psa.hooks.specs import LogitBiasSpec

        source = self if normalise_mode == "none" else self.normalise(normalise_mode)
        signed = source.replace(values=source.importance(), objective="importance")
        table = signed._dense_table(arch)
        return LogitBiasSpec(
            table=table,
            lam=float(lam),
            binning=self._binning_object(),
            layers=None if layers is None else tuple(int(x) for x in layers),
        )

    def to_value_scale_spec(
        self,
        alpha: float = 0.25,
        *,
        arch: Any | None = None,
        normalise_mode: str = "max_abs",
        clip: tuple[float, float] = (0.0, 2.0),
        layers: Sequence[int] | None = None,
    ) -> Any:
        """Mode 2 spec: ``S = clip(1 + alpha * normalised_importance)``.

        ``importance`` and not ``values``: the gate must *boost* the cells that matter,
        and for a loss objective the raw Delta is negative there. ``clip`` keeps the gate
        non-negative — a negative value scale flips the sign of a head's contribution,
        which is a different experiment and should be asked for explicitly.
        """
        from rineya_psa.hooks.specs import ValueScaleSpec

        source = self if normalise_mode == "none" else self.normalise(normalise_mode)
        signed = source.replace(values=source.importance(), objective="importance")
        table = 1.0 + float(alpha) * signed._dense_table(arch)
        lo, hi = float(clip[0]), float(clip[1])
        if lo > hi:
            raise ValueError(f"clip bounds are inverted: {clip}")
        return ValueScaleSpec(
            table=np.clip(table, lo, hi),
            binning=self._binning_object(),
            layers=None if layers is None else tuple(int(x) for x in layers),
        )

    # -- brief section 37 controls -------------------------------------------

    def shuffle_bins(self, seed: int) -> SignificanceMatrix:
        """Permute the bin axis **independently within each (layer, head) row**.

        Preserves every row's value multiset exactly — so head amplitude, layer profile
        and the global distribution are untouched — and destroys only the association
        between a value and the position it was measured at. This is the control that
        answers "is the *positional pattern* doing the work, or just the magnitudes?".
        """
        rng = np.random.default_rng(int(seed))
        out = self.values.copy()
        for li in range(self.n_layers):
            for hi in range(self.n_heads):
                out[li, hi, :] = out[li, hi, rng.permutation(self.n_bins)]
        return self._derived(out, "shuffle_bins", seed=int(seed))

    def shuffle_heads(self, seed: int) -> SignificanceMatrix:
        """Permute the head axis **within each layer**, keeping whole head profiles intact.

        Each layer keeps exactly the same set of positional profiles and therefore the
        same per-layer, per-bin marginals; only which KV head owns which profile changes.
        The control for "is head *identity* load-bearing, or just the layer's mix?".
        """
        rng = np.random.default_rng(int(seed))
        out = self.values.copy()
        for li in range(self.n_layers):
            out[li] = out[li][rng.permutation(self.n_heads), :]
        return self._derived(out, "shuffle_heads", seed=int(seed))

    def sign_flip(self) -> SignificanceMatrix:
        """Negate every value: same magnitudes, inverted preferences.

        An intervention built on this should *hurt* by as much as the real one helps. If
        it does not, the effect is an artefact of perturbing attention at all rather than
        of perturbing it in the measured direction.
        """
        return self._derived(-self.values, "sign_flip")

    def uniform_middle_boost(
        self,
        amount: float | None = None,
        *,
        center: float = 0.5,
        width: float = 0.5,
        objective: str | None = None,
    ) -> SignificanceMatrix:
        """A structureless matrix that is constant on the middle bins and zero elsewhere.

        Every ``(layer, head)`` row is identical, so the only information it carries is
        "the middle of the context deserves more weight". ``amount`` defaults to the real
        matrix's own mean *importance* over those bins, which matches the two
        interventions in scale; the resulting matrix is written in importance units and
        its objective is relabelled so a caller cannot accidentally re-apply the sign.
        """
        if not 0.0 < width <= 1.0:
            raise ValueError(f"width must lie in (0, 1], got {width}")
        centers = self.bin_centers
        lo, hi = center - width / 2.0, center + width / 2.0
        middle = (centers >= lo) & (centers <= hi)
        if not middle.any():  # a coarse grid can straddle the window entirely
            middle = np.zeros(self.n_bins, dtype=bool)
            middle[int(np.argmin(np.abs(centers - center)))] = True
        if amount is None:
            block = self.importance()[:, :, middle]
            amount = float(np.nanmean(block)) if np.isfinite(block).any() else 0.0
        out = np.zeros_like(self.values)
        out[:, :, middle] = float(amount)
        derived = self._derived(
            out,
            "uniform_middle_boost",
            amount=float(amount),
            center=float(center),
            width=float(width),
            middle_bins=[int(b) for b in np.flatnonzero(middle)],
        )
        return derived.replace(objective=objective or "importance")

    def controls(self, seed: int) -> dict[str, SignificanceMatrix]:
        """All four brief-section-37 controls, so a sweep gets them by construction."""
        return {
            "shuffle_bins": self.shuffle_bins(seed),
            "shuffle_heads": self.shuffle_heads(seed + 1),
            "sign_flip": self.sign_flip(),
            "uniform_middle_boost": self.uniform_middle_boost(),
        }

    # -- IO -------------------------------------------------------------------

    def _meta(self) -> dict[str, Any]:
        return {
            "layers": list(self.layers),
            "kv_heads": list(self.kv_heads),
            "bins": list(self.bins),
            "binning_desc": dict(self.binning_desc),
            "objective": self.objective,
            "model_key": self.model_key,
            "provenance": dict(self.provenance),
        }

    def save_npz(
        self, path: str | Path, *, config: Mapping[str, Any] | None = None
    ) -> Path:
        """Lossless round trip, plus the SPEC section 9 provenance sidecar."""
        target = Path(path)
        payload = {
            "values": self.values,
            "meta": np.asarray(json.dumps(self._meta(), default=json_default)),
        }
        return write_npz_with_provenance(payload, target, config=config)

    def save_csv(self, path: str | Path) -> Path:
        """Tidy one-row-per-cell CSV for pandas and for eyeballing.

        ``binning_json`` repeats on every row. That is redundant, and it is the price of
        a CSV that reloads into an identical object without a companion file.
        """
        target = Path(path)
        assert_readonly_target(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        centers = self.bin_centers
        binning_json = json.dumps(dict(self.binning_desc), sort_keys=True)
        lines = [
            "model_key,objective,layer,kv_head,bin,bin_center,value,importance,binning_json"
        ]
        sign = float(OBJECTIVE_SIGN.get(self.objective, 1.0))
        for li, layer in enumerate(self.layers):
            for hi, head in enumerate(self.kv_heads):
                for bi, b in enumerate(self.bins):
                    # float(), not the numpy scalar: numpy 2's repr is "np.float64(0.5)",
                    # which round-trips through nothing. repr of a Python float is exact.
                    v = float(self.values[li, hi, bi])
                    lines.append(
                        f"{self.model_key},{self.objective},{layer},{head},{b},"
                        f"{centers[bi]:.6f},{v!r},{sign * v!r},"
                        f"{json.dumps(binning_json)}"
                    )
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return target

    @classmethod
    def load(cls, path: str | Path) -> SignificanceMatrix:
        """Load from ``.npz`` (lossless) or ``.csv`` (values and binning, no provenance)."""
        target = Path(path)
        if target.suffix.lower() == ".csv":
            return cls.load_csv(target)
        with np.load(target, allow_pickle=False) as archive:
            values = np.asarray(archive["values"], dtype=np.float64)
            meta = json.loads(str(archive["meta"]))
        return cls(
            values=values,
            layers=tuple(meta["layers"]),
            kv_heads=tuple(meta["kv_heads"]),
            bins=tuple(meta["bins"]),
            binning_desc=meta["binning_desc"],
            objective=meta["objective"],
            model_key=meta["model_key"],
            provenance=meta.get("provenance", {}),
        )

    @classmethod
    def load_csv(cls, path: str | Path) -> SignificanceMatrix:
        """Rebuild from :meth:`save_csv`. ``provenance`` is not carried by the CSV form."""
        rows = Path(path).read_text(encoding="utf-8").strip().splitlines()
        header = rows[0].split(",")
        idx = {name: i for i, name in enumerate(header)}
        layers: list[int] = []
        heads: list[int] = []
        bins: list[int] = []
        records: list[tuple[int, int, int, float]] = []
        model_key = objective = ""
        binning_desc: Mapping[str, Any] = {}
        for line in rows[1:]:
            # binning_json is the last field and is JSON-quoted, so it holds no bare comma.
            parts = line.split(",", len(header) - 1)
            model_key = parts[idx["model_key"]]
            objective = parts[idx["objective"]]
            layer, head, b = (
                int(parts[idx["layer"]]),
                int(parts[idx["kv_head"]]),
                int(parts[idx["bin"]]),
            )
            records.append((layer, head, b, float(parts[idx["value"]])))
            binning_desc = json.loads(json.loads(parts[idx["binning_json"]]))
            for seq, value in ((layers, layer), (heads, head), (bins, b)):
                if value not in seq:
                    seq.append(value)
        layers.sort()
        heads.sort()
        bins.sort()
        li = {v: i for i, v in enumerate(layers)}
        hi = {v: i for i, v in enumerate(heads)}
        bi = {v: i for i, v in enumerate(bins)}
        values = np.full((len(layers), len(heads), len(bins)), np.nan, dtype=np.float64)
        for layer, head, b, value in records:
            values[li[layer], hi[head], bi[b]] = value
        return cls(
            values=values,
            layers=tuple(layers),
            kv_heads=tuple(heads),
            bins=tuple(bins),
            binning_desc=binning_desc,
            objective=objective,
            model_key=model_key,
            provenance={"source_csv": str(path)},
        )


def _rank_normalise(values: np.ndarray) -> np.ndarray:
    """Map finite entries onto an evenly spaced grid in ``[-1, 1]``, ties averaged."""
    out = np.full(values.shape, np.nan, dtype=np.float64)
    flat = values.reshape(-1)
    finite = np.isfinite(flat)
    n = int(finite.sum())
    if n == 0:
        return out
    if n == 1:
        out.reshape(-1)[finite] = 0.0
        return out
    subset = flat[finite]
    order = np.argsort(subset, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    # Average the ranks of tied values so a plateau does not acquire a fake gradient.
    unique, inverse = np.unique(subset, return_inverse=True)
    if unique.size < n:
        sums = np.zeros(unique.size, dtype=np.float64)
        counts = np.zeros(unique.size, dtype=np.float64)
        np.add.at(sums, inverse, ranks)
        np.add.at(counts, inverse, 1.0)
        ranks = (sums / counts)[inverse]
    out.reshape(-1)[finite] = 2.0 * ranks / (n - 1) - 1.0
    return out


def write_npz_with_provenance(
    arrays: Mapping[str, Any],
    path: str | Path,
    *,
    config: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """``np.savez`` with the SPEC section 9 sidecar and the read-only-checkpoint guard."""
    target = Path(path)
    sidecar = target.with_name(target.name + ".provenance.json")
    assert_readonly_target(target)
    assert_readonly_target(sidecar)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **dict(arrays))
    payload = stamp(config, extra=extra)
    payload["artifact"] = {
        "path": str(target),
        "bytes": target.stat().st_size,
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "arrays": sorted(str(k) for k in arrays),
    }
    sidecar.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )
    return target


def delta_matrices(a: SignificanceMatrix, b: SignificanceMatrix) -> SignificanceMatrix:
    """``b - a`` cell by cell — the cross-checkpoint comparison of brief section 19.

    Refuses to subtract matrices that are not on the same grid, the same binning or the
    same objective. Two checkpoints measured on different bin counts produce a
    numerically valid but meaningless difference, and that is precisely the mistake this
    guard exists to make impossible.
    """
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    if (a.layers, a.kv_heads, a.bins) != (b.layers, b.kv_heads, b.bins):
        raise ValueError("layer / head / bin indices differ; the matrices are not aligned")
    if dict(a.binning_desc) != dict(b.binning_desc):
        raise ValueError(
            f"binning mismatch: {a.binning_desc.get('name')} vs {b.binning_desc.get('name')}"
        )
    if a.objective != b.objective:
        raise ValueError(f"objective mismatch: {a.objective!r} vs {b.objective!r}")
    return SignificanceMatrix(
        values=b.values - a.values,
        layers=a.layers,
        kv_heads=a.kv_heads,
        bins=a.bins,
        binning_desc=a.binning_desc,
        objective=a.objective,
        model_key=f"{b.model_key}-minus-{a.model_key}",
        provenance={
            "kind": "delta",
            "minuend": {"model_key": b.model_key, "sha256": b.content_hash()},
            "subtrahend": {"model_key": a.model_key, "sha256": a.content_hash()},
        },
    )
