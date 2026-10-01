"""TRISKILL-GEO-001: compact, objective-free activation interaction falsifier.

CPU preparation only. No model forward, training data, delta_C, or dense d*d
projector. Actual sources are immutable; output consists of a <=8D rotation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
from pathlib import Path

for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_name] = "1"  # This process only; shared CPU lane stays untouched.
import numpy as np

HERE = Path(__file__).resolve().parent
CORES = Path("C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores")
SOURCES = {
    "BASE": (CORES / "Rineya-Supa-1.7.0-devfreeze/model.safetensors",
             "962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b"),
    "T": (CORES / "Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze/model.safetensors",
          "a28629c9d3454fbf2e884fd7ca442c1b406e7caf4b36d34b2c0614d0add31284"),
    "TH": (CORES / "Rineya-Titan-Transition-1.0.0-devfreeze/model.safetensors",
           "a4d73353db9a91cbba7b4cd7c8fe4fe6822670d10fe523b26a6d651bb9509567"),
}
LAYER = 14
KEY = f"model.layers.{LAYER}.self_attn.o_proj.weight"
RANK = 4
ETA = 0.1
SEED = 20261001


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


class TensorFile:
    """Seek/read one tensor, without an accumulating model mmap working set."""
    def __init__(self, path: Path):
        self.stream = path.open("rb")
        size = struct.unpack("<Q", self.stream.read(8))[0]
        if not 0 < size < 16 << 20:
            self.stream.close()
            raise ValueError("invalid safetensors header size")
        self.header = json.loads(self.stream.read(size))
        self.header.pop("__metadata__", None)
        self.offset = 8 + size

    def tensor(self, key: str) -> np.ndarray:
        info = self.header[key]
        first, last = info["data_offsets"]
        self.stream.seek(self.offset + first)
        raw = self.stream.read(last - first)
        if len(raw) != last - first:
            raise ValueError(f"short tensor read: {key}")
        if info["dtype"] == "F32":
            result = np.frombuffer(raw, "<f4")
        elif info["dtype"] == "BF16":
            bits = np.frombuffer(raw, "<u2").astype("<u4") << 16
            result = bits.view("<f4")
        else:
            raise ValueError(f"unqualified dtype {info['dtype']}")
        result = result.reshape(info["shape"]).astype(np.float64)
        if not np.isfinite(result).all():
            raise ValueError(f"nonfinite tensor: {key}")
        return result

    def close(self):
        self.stream.close()


def left_subspace(delta: np.ndarray, rank: int, seed: int):
    """Frozen randomized range finder, 8 oversamples, one power step."""
    if delta.ndim != 2 or min(delta.shape) < rank or not np.isfinite(delta).all():
        raise ValueError("invalid delta matrix")
    rng = np.random.default_rng(seed)
    width = min(rank + 8, min(delta.shape))
    q, _ = np.linalg.qr(delta @ rng.standard_normal((delta.shape[1], width)))
    q, _ = np.linalg.qr(delta @ (delta.T @ q))
    u, values, _ = np.linalg.svd(q.T @ delta, full_matrices=False)
    if values[rank - 1] <= 1e-12:
        raise ValueError("parent write direction has insufficient numerical rank")
    return q @ u[:, :rank], values[:rank]


def construct(u: np.ndarray, v: np.ndarray, eta: float = ETA):
    """Projector commutator and Cayley transform in their compact union."""
    if u.shape[0] != v.shape[0] or not np.isfinite(u).all() or not np.isfinite(v).all():
        raise ValueError("invalid parent bases")
    for basis in (u, v):
        if not np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-10):
            raise ValueError("parent bases must be orthonormal")
    if not 0 <= eta <= ETA:
        raise ValueError("coefficient outside prospective envelope")
    left, values, _ = np.linalg.svd(np.concatenate((u, v), axis=1), full_matrices=False)
    b = left[:, values > 1e-10]
    ut, vt = b.T @ u, b.T @ v
    pt, ph = ut @ ut.T, vt @ vt.T
    k = pt @ ph - ph @ pt
    identity = np.eye(b.shape[1])
    r = np.linalg.solve(identity - eta * k / 2, identity + eta * k / 2)
    return b, r, k


def apply_rotation(hidden: np.ndarray, basis: np.ndarray, rotation: np.ndarray):
    # Vectors in hidden are rows; transpose R-I to implement Q on columns.
    return hidden + ((hidden @ basis) @ (rotation - np.eye(len(rotation))).T) @ basis.T


def operator_metrics(b: np.ndarray, r: np.ndarray, k: np.ndarray):
    change = r - np.eye(len(r))
    return {
        "union_rank": len(r), "intervention_rank": int(np.linalg.matrix_rank(change, tol=1e-10)),
        "basis_orthogonality_error": float(np.linalg.norm(b.T @ b - np.eye(len(r)))),
        "rotation_orthogonality_error": float(np.linalg.norm(r.T @ r - np.eye(len(r)))),
        "commutator_spectral_norm": float(np.linalg.norm(k, ord=2)),
        "operator_change_frobenius": float(np.linalg.norm(change)),
        "operator_change_spectral": float(np.linalg.norm(change, ord=2)),
        "noncommuting": bool(np.linalg.norm(k) > 1e-10),
    }


def prepare(out: Path):
    # Native weight arithmetic is CPU-only, with original conservative reserves.
    import psutil
    snapshot = {"ram_free_gib": psutil.virtual_memory().available / 2**30,
                "disk_free_gib": psutil.disk_usage(str(out.parent)).free / 2**30}
    if snapshot["ram_free_gib"] < 5 or snapshot["disk_free_gib"] < 6:
        raise RuntimeError(f"CPU preparation reserve gate failed: {snapshot}")
    if out.exists() or out.with_suffix(".json").exists():
        raise FileExistsError("create-only geometry output already exists")
    for name, (path, expected) in SOURCES.items():
        if sha256(path) != expected:
            raise ValueError(f"immutable {name} weight hash mismatch")
    files = {name: TensorFile(path) for name, (path, _) in SOURCES.items()}
    try:
        base, t, th = [files[name].tensor(KEY) for name in ("BASE", "T", "TH")]
    finally:
        for stream in files.values():
            stream.close()
    if not base.shape == t.shape == th.shape == (1024, 2048):
        raise ValueError(f"unqualified source shape: {base.shape}, {t.shape}, {th.shape}")
    dt, dh = t - base, th - t
    u, st = left_subspace(dt, RANK, SEED)
    v, sh = left_subspace(dh, RANK, SEED + 1)
    b, r, k = construct(u, v)
    random_b, _ = np.linalg.qr(np.random.default_rng(SEED + 2).standard_normal(b.shape))
    metrics = operator_metrics(b, r, k)
    if metrics["rotation_orthogonality_error"] > 1e-10 or metrics["commutator_spectral_norm"] > 0.500001:
        raise ValueError("rotation numerical canary failed")
    with out.open("xb") as stream:
        np.savez(stream, basis=b, rotation=r, random_basis=random_b, commutator=k)
    receipt = {
        "experiment": "TRISKILL-GEO-001", "source_commit": "01f47694988a85bdedab7831d1652f0b9bf07e9f",
        "source_weights": {name: {"path": str(path), "sha256": pin} for name, (path, pin) in SOURCES.items()},
        "key": KEY, "hook": "block14 output residual, all token positions; downstream weights fixed",
        "rank_per_parent": RANK, "eta": ETA, "seed": SEED, "metrics": metrics,
        "parent_write_norms": {"T": float(np.linalg.norm(dt)), "H_effective": float(np.linalg.norm(dh))},
        "top_singular_values": {"T": st.tolist(), "H_effective": sh.tolist()},
        "principal_cosines": np.linalg.svd(u.T @ v, compute_uv=False).tolist(),
        "resource_snapshot": snapshot, "geometry_sha256": sha256(out), "code_sha256": sha256(Path(__file__)),
        "model_forwards": 0, "training_examples": 0, "delta_C_used": False,
        "random_match": "same compact R, operator spectrum/norm/rank/compute; displacement can differ",
        "status": "PREPARED" if metrics["noncommuting"] else "STOP_NO_GEOMETRIC_INTERACTION",
        "quality_claim": False, "confirmation": False,
        "limitation": "No objective identifies this rotation with a third skill. Geometry is not functional independence.",
    }
    with out.with_suffix(".json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "geometry.npz")
    args = parser.parse_args()
    print(json.dumps(prepare(args.out), indent=2))
