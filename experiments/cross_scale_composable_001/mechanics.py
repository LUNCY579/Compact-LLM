"""NumPy-only algebra for offline, provenance-gated skill-direction fixtures.

Read-only implementation references inspected before design:
* XSG 3059e576555afbeb735d79e129435c12caaf4359 alignment.py: centered
  affine transport, residual pushforward excludes the intercept, heldout metrics.
* Parent 07e53b0e7d8ed1dd3243493b3a25b97b3988f4fd soup compatibility/merge:
  keyed shape validation and immutable source checkpoint semantics.
* The parent's compose_lora_adapters_v22.py: weighted same-anchor deltas.

This is a small dense fixture implementation of ordinary least squares, not the
streaming ridge implementation in XSG. Capture identity, data partition and
source/receipt authorization MUST be checked by the receipt layer before use.
No tensor cosine, fit diagnostic or orthogonal projection establishes behavior.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np


RCOND = 1e-12


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _scalar(value: float, name: str) -> float:
    _require(not isinstance(value, (bool, np.bool_)), f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid {name}") from exc
    _require(math.isfinite(result), f"nonfinite {name}")
    return result


def _array(value, name: str, *, ndim: int | None = None,
           allow_empty_columns: bool = False) -> np.ndarray:
    raw = np.asarray(value)
    _require(raw.dtype.kind in "fiu", f"{name} must contain real numbers")
    _require(raw.ndim > 0 and (ndim is None or raw.ndim == ndim), f"bad {name} dimensions")
    if allow_empty_columns:
        _require(raw.ndim == 2 and raw.shape[0] > 0, f"bad {name} shape")
    else:
        _require(raw.size > 0 and all(d > 0 for d in raw.shape), f"empty {name}")
    out = np.array(raw, dtype=np.float64, order="C", copy=True)
    _require(bool(np.isfinite(out).all()), f"nonfinite {name}")
    return out


def _frozen(value: np.ndarray) -> np.ndarray:
    out = _array(value, "result")
    out.flags.writeable = False
    return out


def _tensors(values: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    _require(isinstance(values, Mapping) and bool(values), "empty or invalid tensor mapping")
    _require(all(isinstance(k, str) and bool(k) for k in values), "tensor keys must be nonempty strings")
    return {k: _array(values[k], k) for k in sorted(values)}


def _compatible(a: Mapping[str, np.ndarray], b: Mapping[str, np.ndarray]) -> None:
    _require(a.keys() == b.keys(), "tensor key sets differ")
    for key in a:
        _require(a[key].shape == b[key].shape, f"tensor shape differs: {key}")


def _norm(a: np.ndarray) -> float:
    # Scaling avoids the avoidable overflow of sum(a*a) on finite inputs.
    largest = float(np.max(np.abs(a))) if a.size else 0.0
    result = largest * float(np.linalg.norm(a / largest)) if largest else 0.0
    _require(math.isfinite(result), "norm exceeds float64 range")
    return result


def _joint_norm(tensors: Mapping[str, np.ndarray]) -> float:
    result = math.hypot(*(_norm(v) for v in tensors.values()))
    _require(math.isfinite(result), "joint norm exceeds float64 range")
    return result


def _cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    na, nb = _norm(a), _norm(b)
    if na == 0.0 or nb == 0.0:
        return None
    return float(np.clip(np.vdot(a / na, b / nb), -1.0, 1.0))


def tensor_digest(tensors: Mapping[str, np.ndarray]) -> str:
    """Canonical value digest: sorted UTF-8 keys, shapes and little-endian FP64.

    Original checkpoint byte hashes are separate; float32/float64 arrays with
    identical numeric values intentionally have the same canonical digest.
    """
    tensors = _tensors(tensors)
    digest = hashlib.sha256(b"cross-scale-tensor-values-v1\0")
    for key, value in tensors.items():
        descriptor = json.dumps({"name": key, "shape": list(value.shape), "dtype": "<f8"},
                                sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        data = value.astype("<f8", copy=False).tobytes(order="C")
        digest.update(len(descriptor).to_bytes(8, "little"))
        digest.update(descriptor)
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return digest.hexdigest()


def normalized_depth_map(donor_layers: int, student_layers: int) -> tuple[int, ...]:
    """Student index per donor layer; exact rational ties choose the lower index.

    Singleton stacks are placed at depth zero. This convention is geometry only,
    not an assumption of functional equivalence. Duplicate destinations are kept.
    """
    for count in (donor_layers, student_layers):
        _require(type(count) is int and count > 0, "layer counts must be positive integers")
    if donor_layers == 1 or student_layers == 1:
        return (0,) * donor_layers
    return tuple(min(range(student_layers),
                     key=lambda j: (abs(i * (student_layers - 1) - j * (donor_layers - 1)), j))
                 for i in range(donor_layers))


def extract_delta(base: Mapping[str, np.ndarray], skilled: Mapping[str, np.ndarray]) -> dict:
    base, skilled = _tensors(base), _tensors(skilled)
    _compatible(base, skilled)
    return {k: _frozen(skilled[k] - base[k]) for k in base}


def lora_delta(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    """Standard LoRA only: A [rank,in], B [out,rank], alpha/rank * B@A.

    No rsLoRA, DoRA, fan_in_fan_out transpose, bias, or scaling inference.
    """
    a, b = _array(a, "LoRA A", ndim=2), _array(b, "LoRA B", ndim=2)
    _require(a.shape[0] == b.shape[1], "LoRA inner rank mismatch")
    alpha = _scalar(alpha, "LoRA alpha")
    _require(alpha > 0, "LoRA alpha must be positive")
    return _frozen((b @ a) * (alpha / a.shape[0]))


@dataclass(frozen=True)
class FrozenTransport:
    weight: np.ndarray  # [donor width, student width], row-vector convention
    bias: np.ndarray
    rank: int
    singular_values: np.ndarray
    fit_rows: int
    rcond: float

    def __post_init__(self) -> None:
        w, b = _array(self.weight, "transport weight", ndim=2), _array(self.bias, "transport bias", ndim=1)
        _require(w.shape[1] == b.shape[0], "transport bias width mismatch")
        s = _array(self.singular_values, "transport singular values", ndim=1)
        _require(type(self.fit_rows) is int and self.fit_rows >= 2, "invalid fit row count")
        _require(type(self.rank) is int and 0 <= self.rank <= min(self.fit_rows - 1, w.shape[0]),
                 "invalid transport rank")
        _require(s.size == min(self.fit_rows, w.shape[0]) and bool((s >= 0).all()),
                 "invalid transport singular values")
        _require(0 < _scalar(self.rcond, "rcond") < 1, "rcond must lie in (0,1)")
        for key, value in (("weight", w), ("bias", b), ("singular_values", s)):
            object.__setattr__(self, key, _frozen(value))

    def predict(self, donor_base: np.ndarray) -> np.ndarray:
        x = _array(donor_base, "donor base", ndim=2)
        _require(x.shape[1] == self.weight.shape[0], "donor width differs")
        return _frozen(x @ self.weight + self.bias)

    def residual(self, donor_change: np.ndarray) -> np.ndarray:
        """T(h_skill)-T(h_base) = delta_h @ weight: never add bias here."""
        x = _array(donor_change, "donor residual", ndim=2)
        _require(x.shape[1] == self.weight.shape[0], "donor residual width differs")
        return _frozen(x @ self.weight)


def fit_transport(donor_base: np.ndarray, student_base: np.ndarray, *,
                  rcond: float = RCOND) -> FrozenTransport:
    """Minimum-norm centered least squares on already-gated open FIT rows.

    Dense offline fixtures only. Native-width streaming/memory qualification is
    a later stage; this function neither captures nor opens any checkpoint.
    """
    x, y = _array(donor_base, "donor base", ndim=2), _array(student_base, "student base", ndim=2)
    _require(x.shape[0] == y.shape[0] and x.shape[0] >= 2, "unaligned or insufficient fit rows")
    rcond = _scalar(rcond, "rcond")
    _require(0 < rcond < 1, "rcond must lie in (0,1)")
    mx, my = x.mean(0), y.mean(0)
    xc, yc = _array(x - mx, "centered donor"), _array(y - my, "centered student")
    weight, _, rank, singular = np.linalg.lstsq(xc, yc, rcond=rcond)
    return FrozenTransport(weight, my - mx @ weight, int(rank), singular, x.shape[0], rcond)


def transport_diagnostics(fit: FrozenTransport, donor_check: np.ndarray,
                          student_check: np.ndarray) -> dict:
    """Heldout checks; the receipt layer must enforce identity and disjointness."""
    y = _array(student_check, "student check", ndim=2)
    pred = fit.predict(donor_check)
    _require(y.shape == pred.shape, "check sample/feature mismatch")
    error_norm, actual_norm = _norm(y - pred), _norm(y)
    centered_norm = _norm(y - y.mean(0))
    cosines = [_cosine(a, b) for a, b in zip(y, pred)]
    defined = [c for c in cosines if c is not None]
    mse = (error_norm / math.sqrt(y.size)) ** 2
    _require(math.isfinite(mse), "diagnostic MSE exceeds float64 range")
    return {"mse": mse, "relative_rmse": error_norm / actual_norm if actual_norm else None,
            "r2": 1 - (error_norm / centered_norm) ** 2 if centered_norm else None,
            "mean_cosine": float(np.mean(defined)) if defined else None,
            "undefined_cosine_rows": len(cosines) - len(defined), "check_rows": y.shape[0],
            "fit_rank": fit.rank, "functional_transfer_established": False}


def orthonormal_basis(columns: np.ndarray, *, rcond: float = RCOND) -> np.ndarray:
    """SVD basis with numerical null directions removed; stable on duplicates."""
    columns = _array(columns, "basis columns", ndim=2, allow_empty_columns=True)
    rcond = _scalar(rcond, "rcond")
    _require(0 < rcond < 1, "rcond must lie in (0,1)")
    if columns.shape[1] == 0:
        basis = columns.copy()
    else:
        u, s, _ = np.linalg.svd(columns, full_matrices=False)
        rank = int(np.sum(s > rcond * s[0])) if s[0] > 0 else 0
        basis = u[:, :rank].copy()
        # Deterministic sign convention; rotations in degenerate singular spaces
        # remain implementation-dependent, but the projected subspace is unique.
        for j in range(rank):
            if basis[np.argmax(np.abs(basis[:, j])), j] < 0:
                basis[:, j] *= -1
    basis.flags.writeable = False
    return basis


def _matrix_rank(value: np.ndarray) -> int | None:
    if value.ndim != 2:
        return None
    return orthonormal_basis(value).shape[1]


def delta_report(delta: Mapping[str, np.ndarray], base: Mapping[str, np.ndarray] | None = None) -> dict:
    delta = _tensors(delta)
    if base is not None:
        base = _tensors(base)
        _compatible(delta, base)
    norm = _joint_norm(delta)
    base_norm = _joint_norm(base) if base is not None else None
    modules = {}
    for key, value in delta.items():
        bn = _norm(base[key]) if base is not None else None
        rank = _matrix_rank(value)
        modules[key] = {"shape": list(value.shape), "l2": _norm(value), "effective_rank": rank,
                        "target_parameter_count": value.size,
                        "nonzero_parameter_count": int(np.count_nonzero(value)),
                        "minimal_rank_factor_parameter_count": rank * sum(value.shape) if rank is not None else None,
                        "relative_l2_to_base": _norm(value) / bn if bn else None}
    return {"l2": norm, "relative_l2_to_base": norm / base_norm if base_norm else None,
            "target_parameter_count": sum(v.size for v in delta.values()),
            "nonzero_parameter_count": sum(int(np.count_nonzero(v)) for v in delta.values()),
            "modules": modules, "rank_rcond": RCOND,
            "parameter_count_note": "Dense support and minimal mathematical factor count; not an assertion of training parameter budget."}


def project_delta(delta: Mapping[str, np.ndarray], protected_skills: Sequence[Mapping[str, np.ndarray]],
                  base: Mapping[str, np.ndarray], *, module_layers: Mapping[str, str] | None = None) -> tuple[dict, dict, dict]:
    """Project against actual skill deltas in a fixed layer-balanced metric.

    Each module is divided by its BASE Frobenius norm and sqrt(modules in its
    layer). This is fixed linear weighting; normalizing each skill independently
    per module would destroy linear projection semantics. Every base module must
    be nonzero. With no layer grouping each module gets equal relative-energy
    weight. Orthogonality is measured in this metric, not necessarily raw cosine.
    """
    delta, base = _tensors(delta), _tensors(base)
    _compatible(delta, base)
    _require(isinstance(protected_skills, Sequence) and not isinstance(protected_skills, (str, bytes)),
             "protected skills must be a sequence")
    protected = [_tensors(p) for p in protected_skills]
    for skill in protected:
        _compatible(delta, skill)
    groups = {k: k for k in delta} if module_layers is None else dict(module_layers)
    _require(groups.keys() == delta.keys() and all(isinstance(v, str) and v for v in groups.values()),
             "invalid layer grouping")
    counts = {group: list(groups.values()).count(group) for group in set(groups.values())}
    scales = {}
    for key in delta:
        norm = _norm(base[key])
        _require(norm > 0, f"zero base module cannot define relative metric: {key}")
        scales[key] = 1.0 / norm / math.sqrt(counts[groups[key]])
        _require(math.isfinite(scales[key]) and scales[key] > 0, "metric scale outside float64 range")

    def flatten(values):
        return _array(np.concatenate([values[k].ravel() * scales[k] for k in delta]), "weighted vector")

    d = flatten(delta)
    columns = np.column_stack([flatten(p) for p in protected]) if protected else np.empty((d.size, 0))
    basis = orthonormal_basis(columns)
    removed_flat = basis @ (basis.T @ d)
    before_norm = _norm(d)
    # Do not let norm matching amplify a roundoff-only remainder into a skill.
    annihilated = before_norm > 0 and _norm(d - removed_flat) <= RCOND * before_norm
    if annihilated:
        removed_flat = d.copy()
    projected, removed = {}, {}
    start = 0
    for key, value in delta.items():
        stop = start + value.size
        removed[key] = _frozen(value if annihilated else removed_flat[start:stop].reshape(value.shape) / scales[key])
        projected[key] = _frozen(np.zeros_like(value) if annihilated else value - removed[key])
        start = stop
    after_norm, removed_norm = _norm(flatten(projected)), _norm(removed_flat)
    ranks_before = {k: _matrix_rank(v) for k, v in delta.items()}
    ranks_after = {k: _matrix_rank(v) for k, v in projected.items()}
    return projected, removed, {
        "basis_rank": basis.shape[1], "protected_skill_count": len(protected),
        "protected_tensor_digests": [tensor_digest(p) for p in protected],
        "metric": "base_module_relative_l2_equal_layer_v1", "module_layers": groups,
        "module_scales": scales, "weighted_norm_before": before_norm,
        "weighted_norm_after": after_norm, "weighted_removed_norm": removed_norm,
        "projection_numerically_annihilated": annihilated, "annihilation_relative_tolerance": RCOND,
        "projection_energy_removed": (removed_norm / before_norm) ** 2 if before_norm else None,
        "weighted_basis_residual_norm": _norm(basis.T @ flatten(projected)),
        "rank_before": ranks_before, "rank_after": ranks_after,
        "rank_increased_modules": [k for k in delta if ranks_before[k] is not None and ranks_after[k] > ranks_before[k]],
        "rank_preservation_guaranteed": False,
        "limitation": "Subtracting protected module matrices can increase rank; rank truncation would be a separately registered transform and may reintroduce overlap.",
    }


def norm_match(delta: Mapping[str, np.ndarray], target_norm: float) -> dict:
    delta = _tensors(delta)
    target_norm = _scalar(target_norm, "target norm")
    _require(target_norm >= 0, "negative target norm")
    norm = _joint_norm(delta)
    _require(norm > 0 or target_norm == 0, "cannot scale zero delta to a positive norm")
    return {k: _frozen((v / norm) * target_norm if norm else np.zeros_like(v)) for k, v in delta.items()}


def compose_deltas(a: Mapping[str, np.ndarray], b: Mapping[str, np.ndarray], *,
                   scale_a: float = 1.0, scale_b: float = 1.0) -> dict:
    """Pure same-base algebra; receipt-gated wrapper enforces base identity."""
    a, b = _tensors(a), _tensors(b)
    _compatible(a, b)
    scale_a, scale_b = _scalar(scale_a, "scale A"), _scalar(scale_b, "scale B")
    return {k: _frozen(scale_a * a[k] + scale_b * b[k]) for k in a}


def random_matched_delta(reference: Mapping[str, np.ndarray], seed: int) -> dict:
    """Random orthogonal left/right bases preserve per-module spectrum and norm.

    Effective numerical rank, shapes and dense target-module budget are matched.
    Declared adapter rank/training budget MUST additionally be checked by receipts;
    algebra cannot recover those from a dense update. Requires matrix modules.
    """
    reference = _tensors(reference)
    _require(type(seed) is int and seed >= 0, "seed must be a nonnegative integer")
    rng, result = np.random.default_rng(seed), {}
    for key, value in reference.items():
        _require(value.ndim == 2, "random matched control requires matrix modules")
        singular = np.linalg.svd(value, compute_uv=False)
        rank = int(np.sum(singular > RCOND * singular[0])) if singular[0] else 0
        if rank == 0:
            result[key] = _frozen(np.zeros_like(value))
            continue
        left, _ = np.linalg.qr(rng.normal(size=(value.shape[0], rank)), mode="reduced")
        right, _ = np.linalg.qr(rng.normal(size=(value.shape[1], rank)), mode="reduced")
        random = (left * singular[:rank]) @ right.T
        result[key] = _frozen(random / _norm(random) * _norm(value))
    return result


def parameter_overlap(a: Mapping[str, np.ndarray], b: Mapping[str, np.ndarray]) -> dict:
    a, b = _tensors(a), _tensors(b)
    _compatible(a, b)
    local = {k: _cosine(a[k].ravel(), b[k].ravel()) for k in a}
    defined = [v for v in local.values() if v is not None]
    return {"global_cosine": _cosine(np.concatenate([v.ravel() for v in a.values()]),
                                     np.concatenate([v.ravel() for v in b.values()])),
            "per_module_cosine": local,
            "equal_module_mean_cosine": float(np.mean(defined)) if defined else None,
            "equal_module_mean_absolute_cosine": float(np.mean(np.abs(defined))) if defined else None,
            "undefined_modules": [k for k, v in local.items() if v is None],
            "functional_independence_established": False}


def _subspace_geometry(a: np.ndarray, b: np.ndarray) -> dict:
    _require(a.shape[0] == b.shape[0], "subspace ambient dimensions differ")
    qa, qb = orthonormal_basis(a), orthonormal_basis(b)
    ra, rb = qa.shape[1], qb.shape[1]
    cosines = np.clip(np.linalg.svd(qa.T @ qb, compute_uv=False), 0, 1) if ra and rb else np.array([])
    na, nb = _norm(a), _norm(b)
    return {"rank_a": ra, "rank_b": rb,
            "principal_cosines": cosines.tolist(), "principal_angles_radians": np.arccos(cosines).tolist(),
            "rank_local_overlap": float(np.mean(cosines ** 2)) if cosines.size else None,
            "a_energy_in_b": (_norm(qb.T @ a) / na) ** 2 if na else None,
            "b_energy_in_a": (_norm(qa.T @ b) / nb) ** 2 if nb else None}


def subspace_overlap(a: np.ndarray, b: np.ndarray) -> dict:
    a, b = _array(a, "subspace A", ndim=2), _array(b, "subspace B", ndim=2)
    _require(a.shape == b.shape, "module shapes differ")
    return {"output_space": _subspace_geometry(a, b), "input_space": _subspace_geometry(a.T, b.T),
            "singular_values_a": np.linalg.svd(a, compute_uv=False).tolist(),
            "singular_values_b": np.linalg.svd(b, compute_uv=False).tolist(), "rank_rcond": RCOND}


def activation_overlap(a: np.ndarray, b: np.ndarray) -> dict:
    """Matched-row residual changes: raw cosine, feature subspace, centered CKA.

    Mean residual directions are retained in subspace analysis; only CKA centers
    across rows. Uses feature Gram products, not an n-by-n sample Gram matrix.
    """
    a, b = _array(a, "activation A", ndim=2), _array(b, "activation B", ndim=2)
    _require(a.shape == b.shape and a.shape[0] >= 2, "activation rows/features differ or insufficient")
    ac, bc = a - a.mean(0), b - b.mean(0)
    # Common positive scaling does not affect CKA and bounds intermediate values.
    an, bn = _norm(ac), _norm(bc)
    cka = None
    if an and bn:
        ac, bc = ac / an, bc / bn
        denominator = _norm(ac.T @ ac) * _norm(bc.T @ bc)
        cka = float(np.clip(_norm(ac.T @ bc) ** 2 / denominator, 0, 1)) if denominator else None
    return {"raw_residual_cosine": _cosine(a.ravel(), b.ravel()), "centered_linear_cka": cka,
            "feature_subspace": _subspace_geometry(a.T, b.T),
            "functional_independence_established": False}
