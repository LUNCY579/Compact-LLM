"""Relative-distance profile ``D[l, h, r]`` and the additive decomposition (brief §14).

Brief section 14 asks two things at once:

1. **Is there a distance effect at all**, independent of where in the context a token
   sits? ``D[l,h,r]`` is the head's attention profile over normalised look-back
   distance ``(i - j) / T``.
2. **Does absolute position plus distance explain the joint pattern**, i.e. is
   ``S[l,h,j] = A[l,h,b_abs(j)] + D[l,h,b_dist(i,j)]`` an adequate model? That is
   answered numerically by :func:`fit_additive_decomposition`, whose
   ``variance_explained`` block is the direct answer to the brief's question 7.

**Identifiability.** ``mu + A[a] + D[r]`` is unchanged by moving a constant between the
three terms, so the fit is constrained by ``sum_a A[a] = 0`` and ``sum_r D[r] = 0``
(unweighted sums over bins) with the level carried by ``mu``. That is the standard
sum-to-zero ANOVA parameterisation; it is a *choice*, it does not change the fitted
surface or any variance-explained number, and it is recorded on the result so a reader
never has to guess which convention produced a printed ``A``.

**How the joint table is measured.** SPEC 5.4 fixes ``CaptureSpec`` to absolute source
bins and explicitly rejects a query-dependent binning, so ``hooks/capture.py`` cannot
produce a joint ``(absolute, distance)`` table. Rather than edit that module, this one
supplies its own accumulator, :class:`JointBinAccumulator`, and installs it on the
context before the session opens; ``hooks.capture.ensure_accumulator`` only builds the
stock accumulator when ``ctx._acc`` is still ``None``, so this is an extension point and
not a monkeypatch. The reduction stays chunked over query rows — the ``[T, T]`` matrix
is never materialised.

DEVIATION: ``JointBinAccumulator`` reconstructs the chunk's query offset by counting
rows, because ``CaptureAccumulator.observe`` receives no ``q_start``. That is exact for
full-sequence forwards with ``use_cache=False``, which is the only way this lab runs a
capture pass; a KV-cached decode step would break it and the accumulator raises rather
than silently mis-binning.

DEVIATION: SPEC section 12.3 restricts *distance-binned interventions* to T <= 2048.
Distance-binned **capture** has no such restriction — it costs one chunked pass and no
``[1, H, T, T]`` mask — so this module runs at any length the forward itself supports.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rineya_psa.significance.matrices import SignificanceMatrix, write_npz_with_provenance

__all__ = [
    "DEFAULT_ABSOLUTE_BINS",
    "DEFAULT_DISTANCE_BINS",
    "AdditiveFit",
    "DistanceProfile",
    "JointBinAccumulator",
    "default_distance_binning",
    "fit_additive_decomposition",
    "measure_distance_profile",
    "variance_explained_summary",
]

DEFAULT_ABSOLUTE_BINS: int = 8
DEFAULT_DISTANCE_BINS: int = 8


def default_distance_binning(n_bins: int = DEFAULT_DISTANCE_BINS) -> Any:
    """Even cuts of ``(i - j) / T`` over ``[0, 1]``."""
    from rineya_psa.hooks.binning import DistanceBinning

    return DistanceBinning.uniform(int(n_bins))


# ======================================================================================
# the joint accumulator
# ======================================================================================


class JointBinAccumulator:
    """Chunked ``[layer, q_head, abs_bin, dist_bin]`` attention mass and cell counts.

    Duck-types :class:`rineya_psa.hooks.capture.CaptureAccumulator`: ``observe`` per
    query-row chunk, ``finalize`` at session exit. Two quantities are accumulated because
    one alone is uninterpretable:

    ``mass``
        Attention probability summed over the cells of each ``(abs, dist)`` pair,
        averaged over valid query rows. Sums to ~1 over the whole table.
    ``counts``
        How many *causally allowed* source cells each pair contains, averaged over the
        same rows. ``mass / counts`` is the mean attention weight per source token, which
        is the quantity that is comparable across pairs of wildly different size — the
        near-diagonal distance bins contain a handful of cells, the far ones thousands.
    """

    def __init__(
        self,
        spec: Any,
        absolute: Any,
        distance: Any,
        *,
        q_heads: int,
    ) -> None:
        if absolute.is_query_dependent:
            raise ValueError(f"{absolute.name} must be an absolute (query-independent) binning")
        if not distance.is_query_dependent:
            raise ValueError(f"{distance.name} must be a query-dependent distance binning")
        self.spec = spec
        self.absolute = absolute
        self.distance = distance
        self.n_abs = int(absolute.n_bins)
        self.n_dist = int(distance.n_bins)
        self.n_q_heads = int(q_heads)
        self._mass: dict[int, np.ndarray] = {}
        self._counts: dict[int, np.ndarray] = {}
        self._rows: dict[int, float] = {}
        self._cursor: dict[int, int] = {}
        self._abs_ids: dict[tuple[int, str], Any] = {}

    # -- internals ------------------------------------------------------------

    def _abs_for(self, kv_len: int, device: Any) -> Any:
        key = (int(kv_len), str(device))
        cached = self._abs_ids.get(key)
        if cached is None:
            cached = self.absolute.bin_ids(kv_len, device=device)
            self._abs_ids[key] = cached
        return cached

    def _q_start(self, layer: int, q_rows: int, kv_len: int) -> int:
        cursor = self._cursor.get(int(layer), 0)
        if cursor >= kv_len:
            cursor = 0  # a fresh forward pass over the same layer
        if cursor + q_rows > kv_len:
            raise RuntimeError(
                f"layer {layer}: chunk of {q_rows} rows starting at {cursor} overruns a "
                f"{kv_len}-token key axis. JointBinAccumulator assumes full-sequence "
                "forwards with use_cache=False; a cached decode step is not supported."
            )
        self._cursor[int(layer)] = cursor + q_rows
        return cursor

    # -- the per-chunk entry point -------------------------------------------

    def observe(
        self,
        layer: int,
        *,
        probs: Any,
        logits: Any,
        keep: Any,
        head_out: Any,
        row_valid: Any,
        evidence_spans: tuple[tuple[int, int], ...] = (),
    ) -> None:
        import torch

        if self.spec is not None and not self.spec.applies_to(layer):
            return
        B, n_q, q_rows, kv_len = probs.shape
        if n_q != self.n_q_heads:
            raise RuntimeError(f"expected {self.n_q_heads} query heads, saw {n_q}")
        q_start = self._q_start(int(layer), int(q_rows), int(kv_len))
        device, work = probs.device, torch.float32

        if row_valid is None:
            row_w = torch.ones((B, 1, q_rows, 1), device=device, dtype=work)
            n_rows = float(B * q_rows)
        else:
            row_w = row_valid.to(device=device, dtype=work).view(B, 1, q_rows, 1)
            n_rows = float(row_w.sum().item())
        if n_rows == 0.0:
            return

        abs_ids = self._abs_for(int(kv_len), device)  # [T]
        dist_ids = self.distance.chunk_bin_ids(
            int(kv_len), q_start, q_start + int(q_rows), q_offset=0, device=device
        )  # [q, T]
        joint = (abs_ids.view(1, -1) * self.n_dist + dist_ids).reshape(-1)  # [q*T]
        n_joint = self.n_abs * self.n_dist

        allowed = (
            torch.ones((1, 1, q_rows, kv_len), device=device, dtype=work)
            if keep is None
            else keep.to(work).expand(B, 1, q_rows, kv_len)
        )
        p = probs.to(work) * row_w * allowed

        mass = torch.zeros((B, n_q, n_joint), device=device, dtype=work)
        mass.index_add_(2, joint, p.reshape(B, n_q, -1))
        # ``allowed * row_w`` broadcasts the (possibly batch-free) mask up to B rows, so
        # the counts denominator is per batch element exactly like the mass numerator.
        weighted_allowed = (allowed * row_w).expand(B, 1, q_rows, kv_len).reshape(B, 1, -1)
        counts = torch.zeros((B, 1, n_joint), device=device, dtype=work)
        counts.index_add_(2, joint, weighted_allowed.contiguous())

        key = int(layer)
        mass_np = mass.sum(dim=0).double().cpu().numpy()
        counts_np = counts.sum(dim=(0, 1)).double().cpu().numpy()
        if key in self._mass:
            self._mass[key] += mass_np
            self._counts[key] += counts_np
            self._rows[key] += n_rows
        else:
            self._mass[key] = mass_np
            self._counts[key] = counts_np
            self._rows[key] = n_rows

    # -- finalisation ---------------------------------------------------------

    def finalize(self) -> dict[tuple[int, str], np.ndarray]:
        out: dict[tuple[int, str], np.ndarray] = {}
        for layer, total in self._mass.items():
            rows = max(self._rows[layer], 1.0)
            out[(layer, "joint_mass")] = (total / rows).reshape(
                self.n_q_heads, self.n_abs, self.n_dist
            )
            out[(layer, "joint_cell_count")] = (self._counts[layer] / rows).reshape(
                self.n_abs, self.n_dist
            )
        return out

    def row_counts(self) -> dict[int, float]:
        return dict(self._rows)


# ======================================================================================
# the additive decomposition
# ======================================================================================


@dataclass(frozen=True, eq=False)
class AdditiveFit:
    """Weighted least-squares ``y[a,r] ~ mu + A[a] + D[r]`` with sum-to-zero effects."""

    mu: np.ndarray  # [...]
    absolute: np.ndarray  # [..., n_abs], sums to zero on the last axis
    distance: np.ndarray  # [..., n_dist], sums to zero on the last axis
    r2_absolute_only: np.ndarray  # [...]
    r2_distance_only: np.ndarray  # [...]
    r2_additive: np.ndarray  # [...]
    ss_total: np.ndarray  # [...]
    n_iterations: int
    converged: bool
    identifiability: str = (
        "sum_a A[a] = 0 and sum_r D[r] = 0 (unweighted over bins); mu carries the level"
    )
    weighting: str = "cell_counts"

    @property
    def n_abs(self) -> int:
        return int(self.absolute.shape[-1])

    @property
    def n_dist(self) -> int:
        return int(self.distance.shape[-1])

    def fitted(self) -> np.ndarray:
        """``[..., n_abs, n_dist]`` reconstruction from the fitted effects."""
        return (
            self.mu[..., None, None]
            + self.absolute[..., :, None]
            + self.distance[..., None, :]
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "identifiability": self.identifiability,
            "weighting": self.weighting,
            "n_iterations": self.n_iterations,
            "converged": self.converged,
            "n_abs": self.n_abs,
            "n_dist": self.n_dist,
            **variance_explained_summary(self),
        }


def fit_additive_decomposition(
    table: Any,
    weights: Any | None = None,
    *,
    max_iter: int = 200,
    tol: float = 1e-12,
    weighting_label: str | None = None,
) -> AdditiveFit:
    """Fit ``mu + A[a] + D[r]`` to ``table[..., n_abs, n_dist]`` by weighted LS.

    Uniform weights admit a closed form (grand mean plus centred row and column means);
    non-uniform weights do not, because the design is no longer orthogonal. The general
    case is solved by block coordinate descent over ``(mu, A, D)``, each block solved
    exactly — the objective is a convex quadratic, so this converges to the global
    optimum rather than to a nearby stationary point.

    Cells with zero weight (an ``(absolute, distance)`` pair no causal ``(i, j)`` can
    occupy) are excluded from every sum instead of being scored as zeros, which would
    drag the fit toward a structural artefact of the triangular mask.
    """
    y = np.asarray(table, dtype=np.float64)
    if y.ndim < 2:
        raise ValueError(
            f"table must have at least 2 dimensions [.., n_abs, n_dist], got {y.shape}"
        )
    n_abs, n_dist = int(y.shape[-2]), int(y.shape[-1])
    lead = y.shape[:-2]
    flat = y.reshape(-1, n_abs, n_dist)

    if weights is None:
        w = np.ones_like(flat)
        label = weighting_label or "uniform"
    else:
        w = np.broadcast_to(np.asarray(weights, dtype=np.float64), y.shape)
        w = np.ascontiguousarray(w).reshape(-1, n_abs, n_dist)
        label = weighting_label or "explicit"
    w = np.where(np.isfinite(w) & (w > 0) & np.isfinite(flat), w, 0.0)
    values = np.where(w > 0, np.nan_to_num(flat, nan=0.0), 0.0)

    total_w = w.sum(axis=(1, 2))
    safe_total = np.where(total_w > 0, total_w, 1.0)
    mu = (w * values).sum(axis=(1, 2)) / safe_total
    alpha = np.zeros((flat.shape[0], n_abs), dtype=np.float64)
    delta = np.zeros((flat.shape[0], n_dist), dtype=np.float64)

    row_w = w.sum(axis=2)
    col_w = w.sum(axis=1)
    safe_row = np.where(row_w > 0, row_w, 1.0)
    safe_col = np.where(col_w > 0, col_w, 1.0)

    converged = False
    used_iter = 0
    for used_iter in range(1, int(max_iter) + 1):  # noqa: B007 - read after the loop
        previous = np.concatenate([mu[:, None], alpha, delta], axis=1)
        resid = values - mu[:, None, None] - delta[:, None, :]
        alpha = np.where(row_w > 0, (w * resid).sum(axis=2) / safe_row, 0.0)
        resid = values - mu[:, None, None] - alpha[:, :, None]
        delta = np.where(col_w > 0, (w * resid).sum(axis=1) / safe_col, 0.0)
        resid = values - alpha[:, :, None] - delta[:, None, :]
        mu = (w * resid).sum(axis=(1, 2)) / safe_total
        # Sum-to-zero re-parameterisation: shifts constants into mu, fitted surface fixed.
        shift_a = alpha.mean(axis=1)
        shift_d = delta.mean(axis=1)
        alpha = alpha - shift_a[:, None]
        delta = delta - shift_d[:, None]
        mu = mu + shift_a + shift_d
        current = np.concatenate([mu[:, None], alpha, delta], axis=1)
        if np.max(np.abs(current - previous)) <= float(tol):
            converged = True
            break

    weighted_mean = (w * values).sum(axis=(1, 2)) / safe_total
    ss_total = (w * (values - weighted_mean[:, None, None]) ** 2).sum(axis=(1, 2))
    safe_ss = np.where(ss_total > 0, ss_total, 1.0)

    row_mean = np.where(row_w > 0, (w * values).sum(axis=2) / safe_row, 0.0)
    col_mean = np.where(col_w > 0, (w * values).sum(axis=1) / safe_col, 0.0)
    ss_abs = (w * (values - row_mean[:, :, None]) ** 2).sum(axis=(1, 2))
    ss_dist = (w * (values - col_mean[:, None, :]) ** 2).sum(axis=(1, 2))
    fitted = mu[:, None, None] + alpha[:, :, None] + delta[:, None, :]
    ss_add = (w * (values - fitted) ** 2).sum(axis=(1, 2))

    def r2(ss_res: np.ndarray) -> np.ndarray:
        out = 1.0 - ss_res / safe_ss
        return np.where(ss_total > 0, out, np.nan)

    return AdditiveFit(
        mu=mu.reshape(lead),
        absolute=alpha.reshape(*lead, n_abs),
        distance=delta.reshape(*lead, n_dist),
        r2_absolute_only=r2(ss_abs).reshape(lead),
        r2_distance_only=r2(ss_dist).reshape(lead),
        r2_additive=r2(ss_add).reshape(lead),
        ss_total=ss_total.reshape(lead),
        n_iterations=int(used_iter),
        converged=bool(converged),
        weighting=label,
    )


def variance_explained_summary(fit: AdditiveFit) -> dict[str, Any]:
    """The numeric answer to the brief's question 7.

    Reports both an unweighted mean over ``(layer, head)`` and a
    ``ss_total``-weighted pooled figure. They differ whenever heads with a flat joint
    table (nothing to explain, ``R^2`` noisy) sit next to heads with a strong one, and
    quoting only the unweighted mean is how such a table gets over-read.
    """
    total = np.asarray(fit.ss_total, dtype=np.float64)
    weight = np.where(np.isfinite(total) & (total > 0), total, 0.0)
    denom = float(weight.sum())

    def pooled(name: str) -> float:
        r2 = np.asarray(getattr(fit, name), dtype=np.float64)
        ok = np.isfinite(r2) & (weight > 0)
        if not ok.any() or denom <= 0:
            return float("nan")
        # Pooling R^2 through the residual sums, not by averaging ratios.
        ss_res = (1.0 - r2[ok]) * weight[ok]
        return float(1.0 - ss_res.sum() / weight[ok].sum())

    out: dict[str, Any] = {"n_cells_fitted": int(np.isfinite(total).sum())}
    for name, label in (
        ("r2_absolute_only", "absolute_only"),
        ("r2_distance_only", "distance_only"),
        ("r2_additive", "additive"),
    ):
        r2 = np.asarray(getattr(fit, name), dtype=np.float64)
        finite = r2[np.isfinite(r2)]
        out[f"{label}_mean_r2"] = float(finite.mean()) if finite.size else float("nan")
        out[f"{label}_median_r2"] = float(np.median(finite)) if finite.size else float("nan")
        out[f"{label}_pooled_r2"] = pooled(name)
    out["additive_gain_over_absolute"] = (
        out["additive_pooled_r2"] - out["absolute_only_pooled_r2"]
    )
    out["additive_gain_over_distance"] = (
        out["additive_pooled_r2"] - out["distance_only_pooled_r2"]
    )
    out["residual_after_additive"] = 1.0 - out["additive_pooled_r2"]
    out["reading"] = (
        "residual_after_additive is the share of the joint (absolute x distance) "
        "variance that neither an absolute-position effect nor a distance effect "
        "captures, i.e. genuine interaction"
    )
    return out


# ======================================================================================
# result container
# ======================================================================================


@dataclass(frozen=True, eq=False)
class DistanceProfile:
    """Joint attention table over ``(absolute source bin, look-back distance bin)``."""

    joint_mass: np.ndarray  # [L, H_q, n_abs, n_dist], row-mean attention mass
    cell_counts: np.ndarray  # [n_abs, n_dist], row-mean count of allowed source cells
    layers: tuple[int, ...]
    q_heads: tuple[int, ...]
    n_kv_heads: int
    absolute_desc: Mapping[str, Any]
    distance_desc: Mapping[str, Any]
    model_key: str
    n_contexts: int
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mass = np.asarray(self.joint_mass, dtype=np.float64)
        counts = np.asarray(self.cell_counts, dtype=np.float64)
        n_abs = int(self.absolute_desc["n_bins"])
        n_dist = int(self.distance_desc["n_bins"])
        want = (len(self.layers), len(self.q_heads), n_abs, n_dist)
        if mass.shape != want:
            raise ValueError(f"joint_mass is {mass.shape}, expected {want}")
        if counts.shape != (n_abs, n_dist):
            raise ValueError(f"cell_counts is {counts.shape}, expected {(n_abs, n_dist)}")
        object.__setattr__(self, "joint_mass", np.ascontiguousarray(mass))
        object.__setattr__(self, "cell_counts", np.ascontiguousarray(counts))
        object.__setattr__(self, "absolute_desc", dict(self.absolute_desc))
        object.__setattr__(self, "distance_desc", dict(self.distance_desc))
        object.__setattr__(self, "provenance", dict(self.provenance))
        if len(self.q_heads) % int(self.n_kv_heads):
            raise ValueError(
                f"{len(self.q_heads)} query heads do not group into {self.n_kv_heads} KV heads"
            )

    @property
    def n_abs(self) -> int:
        return int(self.joint_mass.shape[2])

    @property
    def n_dist(self) -> int:
        return int(self.joint_mass.shape[3])

    @property
    def weight_per_cell(self) -> np.ndarray:
        """``[L, H_q, n_abs, n_dist]`` mean attention weight per allowed source token.

        NaN where the pair is structurally empty. This, not ``joint_mass``, is the
        quantity the additive model is fitted on: raw mass is dominated by how many
        source tokens a pair happens to contain.
        """
        counts = self.cell_counts[None, None, :, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(counts > 0, self.joint_mass / np.where(counts > 0, counts, 1.0), np.nan)

    def distance_marginal(self) -> np.ndarray:
        """``D[l, h, r]`` — attention mass by look-back distance, summed over absolute bins."""
        return self.joint_mass.sum(axis=2)

    def absolute_marginal(self) -> np.ndarray:
        """``A[l, h, a]`` — attention mass by absolute source bin, summed over distances."""
        return self.joint_mass.sum(axis=3)

    def to_kv_heads(self, array: np.ndarray) -> np.ndarray:
        """Average the query heads of each GQA group. Same approximation as ``bias.py``."""
        g = len(self.q_heads) // int(self.n_kv_heads)
        head_axis = 1
        shape = (
            *array.shape[:head_axis],
            int(self.n_kv_heads),
            g,
            *array.shape[head_axis + 1 :],
        )
        return array.reshape(shape).mean(axis=head_axis + 1)

    def to_distance_matrix(self, *, kv_heads: bool = True) -> SignificanceMatrix:
        """``D[l, h, r]`` as a :class:`SignificanceMatrix` over the distance grid."""
        values = self.distance_marginal()
        heads: tuple[int, ...] = self.q_heads
        granularity = "q_head"
        if kv_heads:
            values = self.to_kv_heads(values)
            heads = tuple(range(int(self.n_kv_heads)))
            granularity = "kv_head_mean_over_query_heads"
        return SignificanceMatrix(
            values=values,
            layers=self.layers,
            kv_heads=heads,
            bins=tuple(range(self.n_dist)),
            binning_desc=self.distance_desc,
            objective="attn_mass_by_distance",
            model_key=self.model_key,
            provenance={
                "kind": "distance_profile",
                "head_granularity": granularity,
                "n_contexts": int(self.n_contexts),
                **dict(self.provenance),
            },
        )

    def to_absolute_matrix(self, *, kv_heads: bool = True) -> SignificanceMatrix:
        """``A[l, h, a]`` as a :class:`SignificanceMatrix` over the absolute grid."""
        values = self.absolute_marginal()
        heads: tuple[int, ...] = self.q_heads
        granularity = "q_head"
        if kv_heads:
            values = self.to_kv_heads(values)
            heads = tuple(range(int(self.n_kv_heads)))
            granularity = "kv_head_mean_over_query_heads"
        return SignificanceMatrix(
            values=values,
            layers=self.layers,
            kv_heads=heads,
            bins=tuple(range(self.n_abs)),
            binning_desc=self.absolute_desc,
            objective="attn_mass_by_absolute_bin",
            model_key=self.model_key,
            provenance={
                "kind": "distance_profile_absolute_marginal",
                "head_granularity": granularity,
                **dict(self.provenance),
            },
        )

    def fit_additive(self, *, transform: str = "log", floor: float = 1e-12) -> AdditiveFit:
        """Fit ``S = A[a] + D[r]`` to :attr:`weight_per_cell`.

        ``transform="log"`` by default. Attention weights are the output of a softmax, so
        an *additive* effect on the pre-softmax logit — which is what Mode 1 applies and
        what the brief's ``S`` parameterises — is a *multiplicative* effect on the weight.
        Fitting an additive model to raw probabilities would therefore reject additivity
        for a model that is exactly additive in the natural parameter. ``"identity"``
        fits the probabilities directly for callers who want that instead; both are
        recorded, neither is a default that hides a modelling choice.
        """
        if transform not in ("log", "identity"):
            raise ValueError(f"transform must be 'log' or 'identity', got {transform!r}")
        y = self.weight_per_cell
        if transform == "log":
            y = np.log(np.maximum(y, float(floor)))
        weights = np.broadcast_to(self.cell_counts, y.shape)
        return fit_additive_decomposition(
            y, weights, weighting_label=f"cell_counts;transform={transform}"
        )

    def summary(self, *, transform: str = "log") -> dict[str, Any]:
        fit = self.fit_additive(transform=transform)
        return {
            "model_key": self.model_key,
            "shape": list(self.joint_mass.shape),
            "absolute_binning": dict(self.absolute_desc),
            "distance_binning": dict(self.distance_desc),
            "n_contexts": int(self.n_contexts),
            "distance_marginal_pooled": self.distance_marginal().mean(axis=(0, 1)).tolist(),
            "absolute_marginal_pooled": self.absolute_marginal().mean(axis=(0, 1)).tolist(),
            "cell_counts": self.cell_counts.tolist(),
            "additive_decomposition": fit.to_json(),
            "provenance": dict(self.provenance),
        }

    def save_npz(self, path: str | Path, *, config: Mapping[str, Any] | None = None) -> Path:
        import json

        from rineya_psa.statistics.provenance import json_default

        payload = {
            "joint_mass": self.joint_mass,
            "cell_counts": self.cell_counts,
            "meta": np.asarray(
                json.dumps(
                    {
                        "layers": list(self.layers),
                        "q_heads": list(self.q_heads),
                        "n_kv_heads": int(self.n_kv_heads),
                        "absolute_desc": dict(self.absolute_desc),
                        "distance_desc": dict(self.distance_desc),
                        "model_key": self.model_key,
                        "n_contexts": int(self.n_contexts),
                        "provenance": dict(self.provenance),
                    },
                    default=json_default,
                )
            ),
        }
        return write_npz_with_provenance(payload, path, config=config)

    @classmethod
    def load(cls, path: str | Path) -> DistanceProfile:
        import json

        with np.load(Path(path), allow_pickle=False) as archive:
            meta = json.loads(str(archive["meta"]))
            return cls(
                joint_mass=np.asarray(archive["joint_mass"], dtype=np.float64),
                cell_counts=np.asarray(archive["cell_counts"], dtype=np.float64),
                layers=tuple(meta["layers"]),
                q_heads=tuple(meta["q_heads"]),
                n_kv_heads=int(meta["n_kv_heads"]),
                absolute_desc=meta["absolute_desc"],
                distance_desc=meta["distance_desc"],
                model_key=meta["model_key"],
                n_contexts=int(meta["n_contexts"]),
                provenance=meta.get("provenance", {}),
            )


# ======================================================================================
# measurement
# ======================================================================================


def measure_distance_profile(
    model: Any,
    tok: Any = None,
    contexts: Iterable[Any] | None = None,
    *,
    layers: Sequence[int] | None = None,
    n_absolute_bins: int = DEFAULT_ABSOLUTE_BINS,
    n_distance_bins: int = DEFAULT_DISTANCE_BINS,
    absolute: Any | None = None,
    distance: Any | None = None,
    chunk: int = 256,
    query_tail_fraction: float = 1.0,
    model_key: str = "unknown",
    device: Any | None = None,
    force_backend: bool = True,
    progress: Any | None = None,
) -> DistanceProfile:
    """Measure the joint ``(absolute, distance)`` attention table (brief section 14).

    ``contexts`` are :class:`rineya_psa.significance.bias.BiasContext` objects (or
    anything with ``input_ids``); ``None`` builds the default content-free set from
    ``tok``, so the distance profile is measured on the same material as ``B``.

    ``query_tail_fraction`` defaults to ``1.0`` here, unlike in ``bias.py``. The
    distance axis is *defined* relative to the query row, so early rows are not a
    confound to be excluded — they are the short-context end of the distance
    distribution, and their limited reach is already carried by ``cell_counts``.
    """
    import torch

    from rineya_psa.hooks.binning import AbsoluteBinning
    from rineya_psa.hooks.registry import psa_session
    from rineya_psa.hooks.specs import CaptureSpec, PSAContext
    from rineya_psa.loading import arch_from_config
    from rineya_psa.significance.bias import default_bias_contexts, tail_query_mask
    from rineya_psa.significance.causal import forced_sdpa_backend

    arch = arch_from_config(model.config)
    absolute = absolute if absolute is not None else AbsoluteBinning(int(n_absolute_bins))
    distance = distance if distance is not None else default_distance_binning(n_distance_bins)

    if contexts is None:
        if tok is None:
            raise ValueError("measure_distance_profile needs either `contexts` or a tokenizer")
        contexts = default_bias_contexts(tok)
    context_list = list(contexts)
    if not context_list:
        raise ValueError("no contexts to measure")

    layer_ids = tuple(range(arch.num_layers)) if layers is None else tuple(int(x) for x in layers)
    if any(not 0 <= lay < arch.num_layers for lay in layer_ids):
        raise IndexError(f"layers out of range for {arch.num_layers} layers: {layer_ids}")
    if device is None:
        device = next(model.parameters()).device

    n_abs, n_dist = int(absolute.n_bins), int(distance.n_bins)
    mass = np.zeros((len(context_list), len(layer_ids), arch.num_q_heads, n_abs, n_dist))
    counts = np.zeros((len(context_list), n_abs, n_dist))

    # ``what`` only has to be non-empty for ``capture_active()``; the reductions it names
    # are never computed because our accumulator replaces the stock one entirely.
    spec = CaptureSpec(
        layers=layer_ids, what=("attn_by_bin",), chunk=int(chunk), binning=absolute
    )
    for ci, context in enumerate(context_list):
        input_ids = tuple(int(x) for x in context.input_ids)
        ids = torch.tensor(list(input_ids), dtype=torch.long, device=device).unsqueeze(0)
        ctx = PSAContext(enabled=True, capture=spec, position_bins=absolute)
        if query_tail_fraction < 1.0:
            ctx.valid_mask = tail_query_mask(len(input_ids), query_tail_fraction).to(device)
        ctx._acc = JointBinAccumulator(
            spec, absolute, distance, q_heads=int(arch.num_q_heads)
        )
        with psa_session(model, ctx), torch.inference_mode(), forced_sdpa_backend(
            force_backend, device
        ):
            model(input_ids=ids, use_cache=False, logits_to_keep=1)
        captured = ctx.captured
        for li, layer in enumerate(layer_ids):
            block = captured.get((int(layer), "joint_mass"))
            cell = captured.get((int(layer), "joint_cell_count"))
            if block is None or cell is None:
                raise RuntimeError(
                    f"no joint capture for layer {layer}; the psa kernel did not run on it"
                )
            mass[ci, li] = block
            if li == 0:
                counts[ci] = cell
        if progress is not None:
            progress(1)

    return DistanceProfile(
        joint_mass=mass.mean(axis=0),
        cell_counts=counts.mean(axis=0),
        layers=layer_ids,
        q_heads=tuple(range(arch.num_q_heads)),
        n_kv_heads=int(arch.num_kv_heads),
        absolute_desc=absolute.describe(),
        distance_desc=distance.describe(),
        model_key=model_key,
        n_contexts=len(context_list),
        provenance={
            "query_tail_fraction": float(query_tail_fraction),
            "chunk": int(chunk),
            "context_ids": [
                getattr(c, "context_id", f"ctx{i}") for i, c in enumerate(context_list)
            ],
            "context_lengths": [len(c.input_ids) for c in context_list],
            "accumulator": "significance.distance.JointBinAccumulator",
            "caveat": (
                "distance bin 0 also collects the causally masked upper triangle, whose "
                "attention mass is zero and whose allowed-cell count is zero, so it does "
                "not contribute to either accumulator"
            ),
        },
    )
