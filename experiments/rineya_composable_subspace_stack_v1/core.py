"""Bounded, deterministic matrix mechanics. Geometry is not capability evidence.

All decompositions are per weight matrix in common BASE coordinates. Raw LoRA
factors must first be composed as scale * B @ A, never averaged independently.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from itertools import product
import math
import numpy as np

MAX_ELEMENTS = 4_194_304


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def matrix(x, *, limit=MAX_ELEMENTS) -> np.ndarray:
    a = np.asarray(x)
    require(a.ndim == 2 and min(a.shape) > 0, "nonempty matrix required")
    require(a.size <= limit, "matrix exceeds workspace admission limit")
    require(a.dtype.kind == "f" and np.isfinite(a).all(), "finite floating matrix required")
    return np.array(a, dtype=np.float64, copy=True)


def rank_arg(r: int, shape) -> None:
    require(type(r) is int and 0 <= r <= min(shape), "invalid rank")


def svd(x, r):
    a = matrix(x)
    rank_arg(r, a.shape)
    if r == 0:
        return np.zeros((a.shape[0], 0)), np.zeros(0), np.zeros((0, a.shape[1]))
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    return u[:, :r], s[:r], vt[:r]


def truncate(x, r):
    u, s, vt = svd(x, r)
    return (u * s) @ vt


def basis(x, r):
    u, s, _ = svd(x, r)
    if not s.size:
        return u
    # Exclude null directions rather than inventing "skills" for zero updates.
    return u[:, s > max(x.shape) * np.finfo(float).eps * s[0]]


def effective_lora(a, b, scale=1.0):
    a, b = matrix(a), matrix(b)
    require(a.shape[0] == b.shape[1], "LoRA factor axes mismatch")
    require(math.isfinite(scale), "nonfinite LoRA scale")
    require(b.shape[0] * a.shape[1] <= MAX_ELEMENTS, "effective LoRA exceeds limit")
    return matrix(scale * (b @ a))


def protection(base, delta, k):
    w, d = matrix(base), matrix(delta)
    require(w.shape == d.shape, "base/delta shape mismatch")
    rank_arg(k, w.shape)
    u, _, vt = svd(w, k)
    # No large identity/projector matrices are materialized.
    safe = d - u @ (u.T @ d)
    safe = safe - (safe @ vt.T) @ vt
    removed = d - safe
    return safe, removed, {
        "removed_fraction": float(np.linalg.norm(removed) / max(np.linalg.norm(d), 1e-300)),
        "left_residual": float(np.linalg.norm(u.T @ safe)),
        "right_residual": float(np.linalg.norm(safe @ vt.T)),
        "removed_is_harmful": None,
    }


def shared_private(deltas, rank, agreement=0.75):
    """Share-inspired OFFLINE spectral-consensus analysis, not online Share.

    SVD of concatenated per-task bases estimates average projector agreement.
    'Private' means residual to the chosen joint bilinear subspace; it does not
    assert pairwise private-space orthogonality or functional independence.
    """
    ds = [matrix(d) for d in deltas]
    require(2 <= len(ds) <= 4 and all(d.shape == ds[0].shape for d in ds), "2-4 aligned deltas required")
    rank_arg(rank, ds[0].shape)
    require(0 < agreement <= 1, "agreement must lie in (0,1]")
    def consensus(transpose):
        bs = [basis(d.T if transpose else d, rank) for d in ds]
        z = np.concatenate(bs, axis=1)
        if z.shape[1] == 0:
            return z
        u, s, _ = np.linalg.svd(z, full_matrices=False)
        scores = s * s / len(ds)
        keep = np.flatnonzero(scores >= agreement - 1e-12)[:rank]
        return u[:, keep]
    u, v = consensus(False), consensus(True)
    shared = [u @ (u.T @ d @ v) @ v.T for d in ds]
    private = [d - s for d, s in zip(ds, shared)]
    return shared, private, {"left_rank": u.shape[1], "right_rank": v.shape[1],
                             "shared_energy": [float(np.linalg.norm(s)**2 / max(np.linalg.norm(d)**2, 1e-300))
                                               for s, d in zip(shared, ds)]}


def ctm_basis(deltas, rank, beta=0.0, iterations=6):
    """CtM dense-matrix Tucker-2: HOSVD initialization, HOOI refinement.

    Implements paper Eqs 2-7 reweighting: targets only determine the bases;
    merge coordinates ALWAYS come from original, unnormalized deltas.
    Dense effective deltas are bounded; LoRA core-space acceleration is not claimed.
    """
    ds = [matrix(d) for d in deltas]
    require(ds and len(ds) <= 4 and all(d.shape == ds[0].shape for d in ds), "aligned deltas required")
    rank_arg(rank, ds[0].shape)
    require(0 <= beta <= 1 and type(iterations) is int and 0 <= iterations <= 32, "invalid CtM settings")
    norms = np.array([np.linalg.norm(d) for d in ds])
    mean = float(norms.mean())
    xs = [d * ((beta*n + (1-beta)*mean)/n) if n else np.zeros_like(d) for d, n in zip(ds, norms)]
    def leading(parts, rows):
        # The stacked work array can exceed a single-matrix limit; bound it separately.
        z = np.concatenate(parts, axis=1)
        require(z.size <= 4 * MAX_ELEMENTS, "CtM workspace exceeds limit")
        if rank == 0:
            return np.zeros((rows, 0))
        return np.linalg.svd(z, full_matrices=False)[0][:, :rank]
    u = leading(xs, ds[0].shape[0])
    v = leading([x.T for x in xs], ds[0].shape[1])
    def objective():
        return float(sum(np.linalg.norm(u.T @ x @ v)**2 for x in xs))
    history = [objective()]
    for _ in range(iterations):
        u = leading([x @ v for x in xs], ds[0].shape[0])
        v = leading([x.T @ u for x in xs], ds[0].shape[1])
        history.append(objective())
    return u, v, history


def merge(deltas, coefficients):
    ds = [matrix(d) for d in deltas]
    require(ds and len(ds) == len(coefficients), "coefficient count mismatch")
    require(all(d.shape == ds[0].shape for d in ds), "delta axes mismatch")
    require(all(type(a) in (int, float) and math.isfinite(a) for a in coefficients), "invalid coefficients")
    return matrix(sum((a*d for a, d in zip(coefficients, ds)), np.zeros_like(ds[0])))


@dataclass(frozen=True)
class Arm:
    name: str
    protect: bool = False
    share: bool = False
    compression: str = "none"
    norm_match: bool = False

    def validate(self):
        require(isinstance(self.name, str) and self.name and all(c.isalnum() or c == '_' for c in self.name), "invalid arm name")
        require(all(type(v) is bool for v in (self.protect, self.share, self.norm_match)), "arm switches must be boolean")
        require(self.compression in ("none", "ctm", "mtc", "individual"), "unknown compression")
        return self


def design():
    """Eight UNIQUE factorial cells; leave-one-out aliases reuse those cells."""
    arms = [Arm(f"P{p}S{s}C{c}", bool(p), bool(s), "ctm" if c else "none")
            for p, s, c in product((0, 1), repeat=3)]
    arms += [Arm("merge_then_compress", compression="mtc"),
             Arm("individual_then_merge", compression="individual"),
             Arm("full_norm_matched", True, True, "ctm", True)]
    return arms


def compose(base, deltas, coefficients, arm: Arm, *, rank=16, protected_rank=16,
            shared_rank=16, agreement=0.75, private_gain=0.5, beta=0.0, iterations=6):
    arm.validate()
    w = matrix(base)
    ds = [matrix(d) for d in deltas]
    require(2 <= len(ds) <= 4 and all(d.shape == w.shape for d in ds), "2-4 common-base deltas required")
    for r in (rank, protected_rank, shared_rank):
        rank_arg(r, w.shape)
    require(type(private_gain) in (int, float) and math.isfinite(private_gain) and 0 <= private_gain <= 1, "invalid private gain")
    raw = merge(ds, coefficients)
    report = {"arm": asdict(arm), "raw_norm": float(np.linalg.norm(raw)), "protection": []}
    if arm.protect:
        protected = [protection(w, d, protected_rank) for d in ds]
        ds = [p[0] for p in protected]
        report["protection"] = [p[2] for p in protected]
    if arm.share:
        ss, pp, report["shared"] = shared_private(ds, shared_rank, agreement)
        ds = [s + private_gain*p for s, p in zip(ss, pp)]
    if arm.compression == "ctm":
        u, v, history = ctm_basis(ds, rank, beta, iterations)
        out = u @ merge([u.T @ d @ v for d in ds], coefficients) @ v.T if rank else np.zeros_like(w)
        report["ctm_objective"] = history
    elif arm.compression == "mtc":
        out = truncate(merge(ds, coefficients), rank)
    elif arm.compression == "individual":
        out = merge([truncate(d, rank) for d in ds], coefficients)
    else:
        out = merge(ds, coefficients)
    if arm.norm_match:
        target, actual = np.linalg.norm(raw), np.linalg.norm(out)
        require(actual > 1e-14 or target == 0, "cannot norm-match erased update")
        out *= target/actual if actual else 1.0
    out = matrix(out)
    report.update({"output_norm": float(np.linalg.norm(out)), "geometry_only": True,
                   "rank_bound": rank if arm.compression in ("ctm", "mtc") else None})
    return out, report


def geometry(deltas):
    ds = [matrix(d) for d in deltas]
    require(ds and all(d.shape == ds[0].shape for d in ds), "geometry shape mismatch")
    norms = [float(np.linalg.norm(d)) for d in ds]
    spectra, spaces = [], []
    for d in ds:
        u, s, vt = np.linalg.svd(d, full_matrices=False)
        energy = s*s
        total = float(energy.sum())
        p = energy/total if total else energy
        tol = max(d.shape)*np.finfo(float).eps*(s[0] if s.size else 0)
        nonzero = s > tol
        spaces.append((u[:, nonzero], vt[nonzero].T))
        spectra.append({"numerical_rank": int(nonzero.sum()),
                        "effective_rank_energy": float(np.exp(-sum(float(x*np.log(x)) for x in p if x > 0))) if total else 0.0,
                        "energy_at_rank": {str(r): float(energy[:r].sum()/total) if total else 0.0 for r in (4, 8, 16, 32)}})
    pairs = []
    for i in range(len(ds)):
        for j in range(i+1, len(ds)):
            ni, nj = norms[i], norms[j]
            angles = {}
            for idx, side in enumerate(("left", "right")):
                a, b = spaces[i][idx], spaces[j][idx]
                cs = np.linalg.svd(a.T @ b, compute_uv=False)
                angles[side] = np.arccos(np.clip(cs, 0, 1)).tolist()
            pairs.append({"i": i, "j": j, "cosine": float(np.vdot(ds[i], ds[j])/(ni*nj)) if ni and nj else None,
                          "principal_angles_radians": angles})
    return {"norms": norms, "spectra": spectra, "pairs": pairs, "functional_independence": None}
