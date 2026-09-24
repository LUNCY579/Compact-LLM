"""CONFIRM-001 — build RAND_MASK: Supa + dSparse (masked), DPLOC-001's control.

A Bernoulli per-element mask over dSparse's non-zero elements removes the same squared norm that
SPARSE_ATTN removes, but from RANDOM elements rather than from a module. The target norm is
||dSparse|ATTN|| = 0.4400615543769171, to within 0.1% (PREREG.md).

Reproducible and order-independent: each tensor's mask comes from its own RNG, seeded by
(SEED, tensor name), consumed row-chunk by row-chunk in a fixed order. One deterministic retry
rescales p if the realised norm misses the target by more than 0.1%.

Streaming FP32 writer; never holds more than one chunk. Create-only output. Sources read-only.
"""
from __future__ import annotations

import argparse, hashlib, json, math, os, shutil, struct, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "delta_subspace_geometry_v1"))
from subspace import ST, ROOTS  # noqa: E402

SEED = 20260924
NORM_SPARSE = 0.6940550846962167
TARGET_NORM = 0.4400615543769171                     # ||dSparse|ATTN||
TOL = 1e-3                                           # 0.1% on the norm
ASSETS = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "generation_config.json")
CHUNK = 4_000_000


def tensor_rng(name: str) -> np.random.Generator:
    h = int.from_bytes(hashlib.sha256(f"{SEED}:{name}".encode()).digest()[:8], "little")
    return np.random.default_rng(h)


def masked_delta_chunks(base, sp, key, p):
    """Yield (lo, hi, w0, masked_delta) for one tensor. Deterministic in (SEED, key, p)."""
    shape = base.info(key)[1]
    per_row = math.prod(shape[1:]) if len(shape) > 1 else 1
    step = max(1, CHUNK // per_row)
    rng = tensor_rng(key)
    for lo in range(0, shape[0], step):
        hi = min(lo + step, shape[0])
        w0 = base.rows(key, lo, hi).astype(np.float32)
        d = sp.rows(key, lo, hi).astype(np.float32) - w0
        u = rng.random(d.shape, dtype=np.float32)       # drawn for EVERY element so the stream
        keep = (u < np.float32(p)) & (d != 0)           # does not depend on which are non-zero
        yield lo, hi, w0, np.where(keep, d, np.float32(0))


def realised_norm(p: float) -> float:
    base = ST(Path(ROOTS["supa"]) / "model.safetensors"); sp = ST(Path(ROOTS["titan_sparse"]) / "model.safetensors")
    sq = 0.0
    for key in base.keys():
        for _, _, _, md in masked_delta_chunks(base, sp, key, p):
            sq += float((md.astype(np.float64) ** 2).sum())
    base.close(); sp.close()
    return math.sqrt(sq)


def build(out_dir: Path, p: float) -> dict:
    out_dir = out_dir.resolve()
    if out_dir.exists():
        raise SystemExit(f"refusing to overwrite {out_dir}")
    supa_root = Path(ROOTS["supa"])
    base = ST(supa_root / "model.safetensors"); sp = ST(Path(ROOTS["titan_sparse"]) / "model.safetensors")
    keys = base.keys()
    header, off = {"__metadata__": {"format": "pt", "experiment": "CONFIRM-001", "arm": "RAND_MASK"}}, 0
    for k in keys:
        shape = base.info(k)[1]; nb = math.prod(shape) * 4
        header[k] = {"dtype": "F32", "shape": shape, "data_offsets": [off, off + nb]}; off += nb
    enc = json.dumps(header, separators=(",", ":"), ensure_ascii=True).encode()
    enc += b" " * (-len(enc) % 8)
    out_dir.mkdir(parents=False, exist_ok=False)
    (out_dir / "INCOMPLETE.json").write_bytes(b'{"arm": "RAND_MASK"}\n')
    tmp = out_dir / "model.safetensors.partial"
    sq = 0.0; kept = nz = 0
    with open(tmp, "xb") as f:
        f.write(struct.pack("<Q", len(enc))); f.write(enc)
        for k in keys:
            for _, _, w0, md in masked_delta_chunks(base, sp, k, p):
                sq += float((md.astype(np.float64) ** 2).sum())
                kept += int((md != 0).sum())
                o = w0 + md
                if not np.isfinite(o).all():
                    raise RuntimeError(f"nonfinite in {k}")
                f.write(o.astype("<f4", copy=False).tobytes(order="C"))
        f.flush(); os.fsync(f.fileno())
    base.close(); sp.close()
    if tmp.stat().st_size != 8 + len(enc) + off:
        raise RuntimeError("byte count mismatch")
    for a in ASSETS:
        shutil.copyfile(supa_root / a, out_dir / a)
    cfg = json.loads((supa_root / "config.json").read_text(encoding="utf-8"))
    cfg["torch_dtype"] = "float32"
    if "dtype" in cfg:
        cfg["dtype"] = "float32"
    (out_dir / "config.json").write_bytes((json.dumps(cfg, indent=2) + "\n").encode())
    tmp.rename(out_dir / "model.safetensors")
    h = hashlib.sha256()
    with open(out_dir / "model.safetensors", "rb") as fh:
        for blk in iter(lambda: fh.read(8 << 20), b""): h.update(blk)
    rec = {"experiment": "CONFIRM-001", "arm": "RAND_MASK",
           "construction": f"Supa + dSparse * Bernoulli(p={p!r}) over dSparse non-zeros, per-tensor RNG seeded by ({SEED}, name), FP32",
           "p": p, "seed": SEED, "target_norm": TARGET_NORM,
           "realised_norm": math.sqrt(sq), "relative_error": math.sqrt(sq) / TARGET_NORM - 1,
           "kept_nonzeros": kept, "weight_sha256": h.hexdigest(),
           "weight_bytes": (out_dir / "model.safetensors").stat().st_size,
           "status": "MATERIALIZED_NOT_EVALUATED", "quality_claim": False}
    (out_dir / "BUILD.json").write_bytes((json.dumps(rec, indent=2) + "\n").encode())
    (out_dir / "INCOMPLETE.json").unlink()
    return rec


def verify(out_dir: Path, p: float) -> None:
    """Regenerate the mask independently for the first 256 rows of every tensor; require bitwise equality."""
    base = ST(Path(ROOTS["supa"]) / "model.safetensors"); sp = ST(Path(ROOTS["titan_sparse"]) / "model.safetensors")
    got = ST(out_dir / "model.safetensors")
    bad = checked = 0
    for k in base.keys():
        for lo, hi, w0, md in masked_delta_chunks(base, sp, k, p):
            g = got.rows(k, lo, hi)
            exp = (w0 + md).astype(np.float32)
            bad += int((g.view(np.uint32) != exp.view(np.uint32)).sum()); checked += g.size
            break                                                    # first chunk of each tensor
    base.close(); sp.close(); got.close()
    print(f"  verify RAND_MASK: {checked:,} elements re-derived, {bad} mismatches", flush=True)
    if bad:
        raise SystemExit("VERIFY FAILED")


def choose_p() -> float:
    p0 = (TARGET_NORM / NORM_SPARSE) ** 2
    n0 = realised_norm(p0)
    err0 = n0 / TARGET_NORM - 1
    print(f"  p0={p0:.6f} realised norm {n0:.8f} (error {err0:+.3e})", flush=True)
    if abs(err0) <= TOL:
        return p0
    p1 = p0 * (TARGET_NORM / n0) ** 2                                 # one deterministic retry
    n1 = realised_norm(p1); err1 = n1 / TARGET_NORM - 1
    print(f"  p1={p1:.6f} realised norm {n1:.8f} (error {err1:+.3e})", flush=True)
    if abs(err1) > TOL:
        raise SystemExit("could not hit the target norm within 0.1% in two passes")
    return p1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    p = choose_p()
    r = build(a.output, p)
    print(json.dumps({k: r[k] for k in ("p", "realised_norm", "relative_error", "kept_nonzeros", "weight_sha256")}, indent=2))
    verify(a.output, p)
