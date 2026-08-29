"""Low-rank structure of ``I`` and ``B`` (brief section 21).

The question section 21 asks is whether a ``[layer x head, bin]`` matrix — 224 rows by
8-16 columns on this panel — is really 224 independent positional profiles, or a handful
of shapes that every head reuses with its own gain. If it is the latter, an intervention
table does not need 224 free rows, and a *compact parameterisation* becomes both a
cheaper artefact and a stronger claim.

Three layers of answer, in increasing strength:

1. **Spectrum.** SVD of the unfolded matrix, cumulative explained variance at ranks
   1, 2, 4, 8, plus the rank-k reconstructions themselves.
2. **Compact model.** ``S[h, p] = a_h * f(p) + c_h``: one shared profile ``f`` over
   position, one gain and one offset per head. Fitted by alternating least squares.
3. **Comparison.** The compact model's explained variance *against the free matrix*,
   which is what says whether the compression is honest.

**Centring.** ``center="mean"`` subtracts the column mean before the decomposition, i.e.
this is PCA over heads rather than a raw SVD. That is the right default here: the mean
positional profile is shared by construction and would otherwise eat the first singular
direction and inflate every "rank 1 explains 95%" headline. ``center="none"`` is
available and recorded.

**A cross-check worth knowing.** For this particular model the ALS fixed point is not
merely a local optimum: with ``c_h`` free, the optimal ``c_h`` is the row mean, and the
remaining problem is exactly the best rank-1 approximation of the *row-centred* matrix.
So converged ALS must agree with the leading singular pair of that matrix, and
``tests/test_significance.py`` pins the agreement. A discrepancy means the ALS is broken,
not that it found something the SVD missed.

Unevaluated cells (NaN, from the coarse-to-fine ladder of ``causal.py``) cannot go into
an SVD. Rows that are not fully evaluated are **dropped**, and the count is recorded in
``n_rows_dropped``; imputing them would invent structure. The ALS fit does handle NaN
per cell, because a masked least-squares update is exact, so it runs on the full matrix
and reports its own coverage.

Pure numpy — no torch anywhere, at import time or otherwise.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rineya_psa.paths import RESULTS
from rineya_psa.significance.matrices import SignificanceMatrix, write_npz_with_provenance
from rineya_psa.statistics.provenance import write_result

__all__ = [
    "CENTER_MODES",
    "DEFAULT_RANKS",
    "LOW_RANK_NPZ_NAME",
    "SVD_JSON_NAME",
    "LowRankAnalysis",
    "LowRankSpectrum",
    "ProfileModel",
    "analyse_low_rank",
    "fit_profile_model",
    "low_rank_spectrum",
    "unfold_source",
]

#: Ranks brief section 21 asks for explicitly.
DEFAULT_RANKS: tuple[int, ...] = (1, 2, 4, 8)

CENTER_MODES: tuple[str, ...] = ("mean", "none")

SVD_JSON_NAME = "significance_svd.json"
LOW_RANK_NPZ_NAME = "significance_low_rank_models.npz"


# ======================================================================================
# unfolding
# ======================================================================================


def unfold_source(source: Any) -> tuple[np.ndarray, tuple[tuple[int, int], ...], dict[str, Any]]:
    """``([rows, bins], row_keys, meta)`` for a matrix, a bias object, or a bare array.

    ``row_keys`` are ``(layer, head)`` pairs in row order, so a reconstruction can be
    folded back into model space without guessing the ordering.
    """
    from rineya_psa.significance.bias import PositionalBias

    if isinstance(source, SignificanceMatrix):
        values, rows = source.unfold()
        return values, rows, {
            "kind": "significance_matrix",
            "model_key": source.model_key,
            "objective": source.objective,
            "binning": dict(source.binning_desc),
            "shape": list(source.shape),
            "content_sha256": source.content_hash(),
        }
    if isinstance(source, PositionalBias):
        return unfold_source(source.to_significance_matrix("attn_by_bin"))
    array = np.asarray(source, dtype=np.float64)
    if array.ndim == 3:
        n_l, n_h, n_b = array.shape
        rows = tuple((lay, h) for lay in range(n_l) for h in range(n_h))
        return array.reshape(n_l * n_h, n_b).copy(), rows, {
            "kind": "array3d",
            "shape": list(array.shape),
        }
    if array.ndim == 2:
        rows = tuple((0, r) for r in range(array.shape[0]))
        return array.copy(), rows, {"kind": "array2d", "shape": list(array.shape)}
    raise ValueError(f"cannot unfold an array of shape {array.shape}; want [L, H, B] or [R, B]")


# ======================================================================================
# spectrum
# ======================================================================================


@dataclass(frozen=True, eq=False)
class LowRankSpectrum:
    """SVD/PCA of one ``[l*h, b]`` unfolding, with its rank-k reconstructions."""

    name: str
    values: np.ndarray  # [R, B] the matrix actually decomposed (complete rows only)
    row_keys: tuple[tuple[int, int], ...]
    column_mean: np.ndarray  # [B]; zeros when center="none"
    u: np.ndarray  # [R, K]
    singular_values: np.ndarray  # [K]
    vt: np.ndarray  # [K, B]
    explained_variance_ratio: np.ndarray  # [K]
    cumulative_explained_variance: np.ndarray  # [K]
    ranks: tuple[int, ...]
    center: str
    n_rows_dropped: int
    source_meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_meta", dict(self.source_meta))

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.values.shape[0]), int(self.values.shape[1]))

    @property
    def max_rank(self) -> int:
        return int(self.singular_values.size)

    def reconstruct(self, k: int) -> np.ndarray:
        """Best rank-``k`` reconstruction, back in the units of :attr:`values`."""
        k = max(0, min(int(k), self.max_rank))
        base = np.broadcast_to(self.column_mean, self.values.shape).copy()
        if k == 0:
            return base
        return base + (self.u[:, :k] * self.singular_values[:k]) @ self.vt[:k]

    def reconstruction_error(self, k: int, *, relative: bool = True) -> float:
        """Frobenius error of :meth:`reconstruct`.

        When ``relative``, the denominator is the norm of the **centred** matrix, i.e.
        the same quantity the explained-variance ratio is computed on, so
        ``reconstruction_error(k) == sqrt(1 - cumulative_at(k))`` holds exactly.

        Normalising by ``||values||`` instead — which this used to do — silently
        reintroduces the shared mean profile that centring exists to remove, and the two
        numbers then sit side by side in the same JSON block disagreeing by an order of
        magnitude. Measured on a 60x8 matrix with a strong shared profile: rank-1
        explained variance 0.2116 next to a rank-1 "relative error" of 0.0122, a 73x
        discrepancy with no note that the denominators differed.
        """
        residual = float(np.linalg.norm(self.values - self.reconstruct(k)))
        if not relative:
            return residual
        scale = float(np.linalg.norm(self.values - self.column_mean))
        return residual / scale if scale > 0 else 0.0

    def cumulative_at(self, k: int) -> float:
        """Cumulative explained variance at rank ``k``; 1.0 once ``k >= max_rank``."""
        k = int(k)
        if k <= 0:
            return 0.0
        if k >= self.max_rank:
            return float(self.cumulative_explained_variance[-1]) if self.max_rank else 1.0
        return float(self.cumulative_explained_variance[k - 1])

    def reconstruct_matrix(self, k: int, template: SignificanceMatrix) -> SignificanceMatrix:
        """Fold a rank-``k`` reconstruction back into a :class:`SignificanceMatrix`.

        Only defined when no rows were dropped: with a partially evaluated ladder matrix
        there is no honest value to put in the missing rows, and writing zeros there
        would hand an intervention table numbers that were never measured.
        """
        if self.n_rows_dropped:
            raise ValueError(
                f"{self.name}: {self.n_rows_dropped} incomplete rows were dropped, so a "
                "rank-k reconstruction does not cover the template's grid"
            )
        if tuple(template.shape[:2]) != (len(template.layers), len(template.kv_heads)):
            raise ValueError("template shape is inconsistent with its own index tuples")
        recon = self.reconstruct(k).reshape(template.shape)
        return template.replace(
            values=recon,
            provenance={
                **dict(template.provenance),
                "derived_from": {
                    "transform": f"rank{int(k)}_reconstruction",
                    "spectrum": self.name,
                    "center": self.center,
                    "source_sha256": template.content_hash(),
                },
            },
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "center": self.center,
            "n_rows_dropped": int(self.n_rows_dropped),
            "max_rank": self.max_rank,
            "singular_values": self.singular_values.tolist(),
            "explained_variance_ratio": self.explained_variance_ratio.tolist(),
            "cumulative_explained_variance": self.cumulative_explained_variance.tolist(),
            "ranks": list(self.ranks),
            "cumulative_at_ranks": {str(k): self.cumulative_at(k) for k in self.ranks},
            "relative_reconstruction_error": {
                str(k): self.reconstruction_error(k) for k in self.ranks
            },
            "source": dict(self.source_meta),
        }


def low_rank_spectrum(
    source: Any,
    *,
    name: str = "matrix",
    ranks: Sequence[int] = DEFAULT_RANKS,
    center: str = "mean",
) -> LowRankSpectrum:
    """SVD (``center="none"``) or PCA over heads (``center="mean"``) of an unfolding."""
    if center not in CENTER_MODES:
        raise ValueError(f"unknown center mode {center!r}; known: {list(CENTER_MODES)}")
    values, row_keys, meta = unfold_source(source)
    complete = np.isfinite(values).all(axis=1)
    n_dropped = int((~complete).sum())
    kept = values[complete]
    kept_keys = tuple(k for k, ok in zip(row_keys, complete, strict=True) if ok)
    if kept.shape[0] < 1:
        raise ValueError(f"{name}: every row contains an unevaluated cell; nothing to decompose")

    column_mean = kept.mean(axis=0) if center == "mean" else np.zeros(kept.shape[1])
    centred = kept - column_mean
    u, s, vt = np.linalg.svd(centred, full_matrices=False)
    total = float((s**2).sum())
    evr = (s**2) / total if total > 0 else np.zeros_like(s)
    return LowRankSpectrum(
        name=name,
        values=kept,
        row_keys=kept_keys,
        column_mean=column_mean,
        u=u,
        singular_values=s,
        vt=vt,
        explained_variance_ratio=evr,
        cumulative_explained_variance=np.cumsum(evr),
        ranks=tuple(int(k) for k in ranks),
        center=center,
        n_rows_dropped=n_dropped,
        source_meta=meta,
    )


# ======================================================================================
# the compact parameterisation  S[h, p] = a_h f(p) + c_h
# ======================================================================================


@dataclass(frozen=True, eq=False)
class ProfileModel:
    """``S[h, p] ~= a_h * f(p) + c_h``, fitted by alternating least squares.

    ``f`` is pinned by ``sum_p f(p) = 0`` and ``||f||_2 = 1`` (the additive freedom goes
    into ``c_h``, the multiplicative freedom into ``a_h``), with the sign chosen so the
    largest-magnitude gain is positive. Without those three constraints ``(a, f, c)`` is
    only identified up to a two-parameter family and two runs would print different
    numbers for the same fit.
    """

    profile: np.ndarray  # f(p), [B]
    gain: np.ndarray  # a_h, [R]
    offset: np.ndarray  # c_h, [R]
    row_keys: tuple[tuple[int, int], ...]
    r2_total: float  # against the grand mean of the free matrix
    r2_shape: float  # against each row's own mean, i.e. the shape content only
    r2_per_row: np.ndarray  # [R], against each row's own mean
    n_iterations: int
    converged: bool
    n_observed: int
    n_free_parameters: int
    n_model_parameters: int
    source_meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_meta", dict(self.source_meta))

    def fitted(self) -> np.ndarray:
        """``[R, B]`` reconstruction."""
        return self.gain[:, None] * self.profile[None, :] + self.offset[:, None]

    @property
    def compression_ratio(self) -> float:
        return float(self.n_free_parameters) / float(max(self.n_model_parameters, 1))

    def to_json(self) -> dict[str, Any]:
        return {
            "profile": self.profile.tolist(),
            "r2_total": self.r2_total,
            "r2_shape": self.r2_shape,
            "r2_shape_median_per_row": float(np.nanmedian(self.r2_per_row))
            if self.r2_per_row.size
            else float("nan"),
            "gain_mean": float(np.nanmean(self.gain)) if self.gain.size else float("nan"),
            "gain_sd": float(np.nanstd(self.gain, ddof=1)) if self.gain.size > 1 else 0.0,
            "n_iterations": self.n_iterations,
            "converged": self.converged,
            "n_observed_cells": int(self.n_observed),
            "n_free_parameters": int(self.n_free_parameters),
            "n_model_parameters": int(self.n_model_parameters),
            "compression_ratio": self.compression_ratio,
            "identifiability": (
                "sum_p f(p) = 0, ||f||_2 = 1, sign fixed by the largest |a_h|"
            ),
            "source": dict(self.source_meta),
        }


def fit_profile_model(
    source: Any,
    *,
    max_iter: int = 500,
    tol: float = 1e-12,
) -> ProfileModel:
    """Alternating least squares for ``S[h, p] = a_h f(p) + c_h``.

    NaN cells are handled by masking each least-squares update, so a partially evaluated
    matrix still yields a fit — with ``n_observed`` recorded so a reader can see how much
    of the grid it rests on. Rows with fewer than two observed bins cannot constrain a
    gain and get ``a_h = 0`` (the row's own mean), which is the honest degenerate answer.
    """
    values, row_keys, meta = unfold_source(source)
    mask = np.isfinite(values)
    x = np.where(mask, values, 0.0)
    w = mask.astype(np.float64)
    n_rows, n_bins = x.shape
    if n_bins < 2:
        raise ValueError(f"a profile model needs at least 2 bins, got {n_bins}")

    row_count = w.sum(axis=1)
    safe_row_count = np.where(row_count > 0, row_count, 1.0)
    offset = (w * x).sum(axis=1) / safe_row_count
    residual = np.where(mask, x - offset[:, None], 0.0)

    # Initialise f from the leading right singular vector of the row-centred matrix; a
    # random start converges to the same place but takes far more sweeps on flat data.
    _, _, vt = np.linalg.svd(residual, full_matrices=False)
    profile = vt[0].astype(np.float64, copy=True)
    profile = _normalise_profile(profile, n_bins)
    gain = np.zeros(n_rows, dtype=np.float64)

    converged = False
    used_iter = 0
    previous = np.zeros_like(x)
    for used_iter in range(1, int(max_iter) + 1):  # noqa: B007 - read after the loop
        denom = (w * profile[None, :] ** 2).sum(axis=1)
        gain = np.where(
            denom > 1e-15, (w * (x - offset[:, None]) * profile[None, :]).sum(axis=1) / np.where(
                denom > 1e-15, denom, 1.0
            ), 0.0
        )
        offset = (w * (x - gain[:, None] * profile[None, :])).sum(axis=1) / safe_row_count
        denom_f = (w * gain[:, None] ** 2).sum(axis=0)
        profile = np.where(
            denom_f > 1e-15,
            (w * gain[:, None] * (x - offset[:, None])).sum(axis=0)
            / np.where(denom_f > 1e-15, denom_f, 1.0),
            profile,
        )
        # Re-impose the constraints; the shift moves into c_h and the scale into a_h, so
        # the fitted surface is untouched and only the printed parameters change.
        shift = float(profile.mean())
        profile = profile - shift
        offset = offset + gain * shift
        norm = float(np.linalg.norm(profile))
        if norm > 1e-15:
            profile = profile / norm
            gain = gain * norm
        fitted = gain[:, None] * profile[None, :] + offset[:, None]
        if np.max(np.abs(fitted - previous)) <= float(tol):
            converged = True
            break
        previous = fitted

    if gain.size:
        pivot = int(np.argmax(np.abs(gain)))
        if gain[pivot] < 0:
            gain, profile = -gain, -profile

    fitted = gain[:, None] * profile[None, :] + offset[:, None]
    ss_res = float((w * (x - fitted) ** 2).sum())
    n_observed = int(w.sum())
    grand = (w * x).sum() / max(n_observed, 1)
    ss_total = float((w * (x - grand) ** 2).sum())
    row_mean = (w * x).sum(axis=1) / safe_row_count
    ss_shape = float((w * (x - row_mean[:, None]) ** 2).sum())
    per_row_res = (w * (x - fitted) ** 2).sum(axis=1)
    per_row_tot = (w * (x - row_mean[:, None]) ** 2).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r2_per_row = np.where(per_row_tot > 0, 1.0 - per_row_res / per_row_tot, np.nan)

    return ProfileModel(
        profile=profile,
        gain=gain,
        offset=offset,
        row_keys=row_keys,
        r2_total=float(1.0 - ss_res / ss_total) if ss_total > 0 else float("nan"),
        r2_shape=float(1.0 - ss_res / ss_shape) if ss_shape > 0 else float("nan"),
        r2_per_row=r2_per_row,
        n_iterations=int(used_iter),
        converged=bool(converged),
        n_observed=n_observed,
        n_free_parameters=int(n_rows * n_bins),
        # a_h and c_h per row, plus f minus its two constraints.
        n_model_parameters=int(2 * n_rows + n_bins - 2),
        source_meta=meta,
    )


def _normalise_profile(profile: np.ndarray, n_bins: int) -> np.ndarray:
    """Centre and unit-normalise ``f``; fall back to a linear ramp if it collapses."""
    f = np.asarray(profile, dtype=np.float64) - float(np.mean(profile))
    norm = float(np.linalg.norm(f))
    if norm <= 1e-15:
        f = np.arange(n_bins, dtype=np.float64)
        f -= f.mean()
        norm = float(np.linalg.norm(f))
    return f / norm


# ======================================================================================
# the packaged analysis and its two artifacts
# ======================================================================================


@dataclass(frozen=True, eq=False)
class LowRankAnalysis:
    """Everything brief section 21 asks for, ready to write."""

    spectra: Mapping[str, LowRankSpectrum]
    profiles: Mapping[str, ProfileModel]
    ranks: tuple[int, ...]
    center: str

    def to_json(self) -> dict[str, Any]:
        return {
            "ranks": list(self.ranks),
            "center": self.center,
            "sources": sorted(self.spectra),
            "spectra": {name: sp.to_json() for name, sp in self.spectra.items()},
            "profile_models": {name: pm.to_json() for name, pm in self.profiles.items()},
            "reading": (
                "cumulative_at_ranks is the share of the head-centred variance captured "
                "by the leading k directions; profile_models.r2_total is what the "
                "two-parameters-per-head compact form recovers of the free matrix"
            ),
        }

    def arrays(self) -> dict[str, np.ndarray]:
        """Flat ``name__field`` array bundle for the ``.npz`` artifact."""
        out: dict[str, np.ndarray] = {}
        for name, sp in self.spectra.items():
            out[f"{name}__values"] = sp.values
            out[f"{name}__u"] = sp.u
            out[f"{name}__singular_values"] = sp.singular_values
            out[f"{name}__vt"] = sp.vt
            out[f"{name}__column_mean"] = sp.column_mean
            out[f"{name}__row_keys"] = np.asarray(sp.row_keys, dtype=np.int64).reshape(-1, 2)
            out[f"{name}__explained_variance_ratio"] = sp.explained_variance_ratio
            for k in self.ranks:
                out[f"{name}__reconstruction_rank{int(k)}"] = sp.reconstruct(int(k))
        for name, pm in self.profiles.items():
            out[f"{name}__profile_f"] = pm.profile
            out[f"{name}__profile_gain"] = pm.gain
            out[f"{name}__profile_offset"] = pm.offset
            out[f"{name}__profile_r2_per_row"] = pm.r2_per_row
        return out

    def write(
        self,
        results_dir: str | Path | None = None,
        *,
        config: Mapping[str, Any] | None = None,
    ) -> dict[str, Path]:
        """Write ``significance_svd.json`` and ``significance_low_rank_models.npz``.

        Both go through the SPEC section 9 writers, so both get a provenance sidecar.
        """
        root = Path(results_dir) if results_dir is not None else RESULTS
        root.mkdir(parents=True, exist_ok=True)
        json_path = write_result(self.to_json(), root / SVD_JSON_NAME, config=config)
        npz_path = write_npz_with_provenance(
            self.arrays(), root / LOW_RANK_NPZ_NAME, config=config
        )
        return {"svd_json": json_path, "low_rank_npz": npz_path}


def analyse_low_rank(
    sources: Mapping[str, Any],
    *,
    ranks: Sequence[int] = DEFAULT_RANKS,
    center: str = "mean",
) -> LowRankAnalysis:
    """Spectrum plus compact model for each named source (typically ``I`` and ``B``)."""
    if not sources:
        raise ValueError("analyse_low_rank needs at least one source matrix")
    ranks = tuple(int(k) for k in ranks)
    spectra: dict[str, LowRankSpectrum] = {}
    profiles: dict[str, ProfileModel] = {}
    for name, source in sources.items():
        spectra[str(name)] = low_rank_spectrum(
            source, name=str(name), ranks=ranks, center=center
        )
        profiles[str(name)] = fit_profile_model(source)
    return LowRankAnalysis(spectra=spectra, profiles=profiles, ranks=ranks, center=center)
