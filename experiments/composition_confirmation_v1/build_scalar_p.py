"""CONFIRM-001 Amendment 1 — build SCALAR_P: Supa + p·dSparse, where p is RAND_MASK's keep rate.

A Bernoulli mask M with keep rate p has E[M ⊙ dSparse] = p·dSparse exactly. SCALAR_P is therefore
the expected value of RAND_MASK, and RAND_MASK vs SCALAR_P (C6) asks whether a random mask of this
delta acts like its expectation. The deviation M⊙d − p·d is unstructured, with norm
sqrt(p(1−p))·||dSparse||. p is read from RAND_MASK/BUILD.json, whose weight hash is re-checked first,
so this arm can only be built after RAND_MASK and from the p RAND_MASK actually used.

Streaming FP32 writer, same layout as build_rand_mask.py. Create-only output. Sources read-only.
"""
from __future__ import annotations

import argparse, hashlib, json, math, os, shutil, struct
from pathlib import Path
import numpy as np

from build_rand_mask import ST, ROOTS, ASSETS, CHUNK, NORM_SPARSE

TOL = 1e-3


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(8 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def read_p(mask_dir: Path) -> float:
    rec = json.loads((mask_dir / "BUILD.json").read_text(encoding="utf-8"))
    if (mask_dir / "INCOMPLETE.json").exists() or rec.get("status") != "MATERIALIZED_NOT_EVALUATED":
        raise SystemExit("RAND_MASK is not a completed build")
    if sha256(mask_dir / "model.safetensors") != rec["weight_sha256"]:
        raise SystemExit("RAND_MASK weights do not match their BUILD.json hash")
    p = float(rec["p"])
    if not 0.0 < p < 1.0:
        raise SystemExit(f"implausible p {p}")
    return p


def scaled_chunks(base, sp, key, p):
    shape = base.info(key)[1]
    per_row = math.prod(shape[1:]) if len(shape) > 1 else 1
    step = max(1, CHUNK // per_row)
    for lo in range(0, shape[0], step):
        hi = min(lo + step, shape[0])
        w0 = base.rows(key, lo, hi).astype(np.float32)
        d = sp.rows(key, lo, hi).astype(np.float32) - w0
        yield lo, hi, w0, d * np.float32(p)


def build(out_dir: Path, p: float, mask_sha: str) -> dict:
    out_dir = out_dir.resolve()
    if out_dir.exists():
        raise SystemExit(f"refusing to overwrite {out_dir}")
    supa_root = Path(ROOTS["supa"])
    base = ST(supa_root / "model.safetensors"); sp = ST(Path(ROOTS["titan_sparse"]) / "model.safetensors")
    keys = base.keys()
    header, off = {"__metadata__": {"format": "pt", "experiment": "CONFIRM-001", "arm": "SCALAR_P"}}, 0
    for k in keys:
        shape = base.info(k)[1]; nb = math.prod(shape) * 4
        header[k] = {"dtype": "F32", "shape": shape, "data_offsets": [off, off + nb]}; off += nb
    enc = json.dumps(header, separators=(",", ":"), ensure_ascii=True).encode()
    enc += b" " * (-len(enc) % 8)
    out_dir.mkdir(parents=False, exist_ok=False)
    (out_dir / "INCOMPLETE.json").write_bytes(b'{"arm": "SCALAR_P"}\n')
    tmp = out_dir / "model.safetensors.partial"
    sq = 0.0
    with open(tmp, "xb") as f:
        f.write(struct.pack("<Q", len(enc))); f.write(enc)
        for k in keys:
            for _, _, w0, md in scaled_chunks(base, sp, k, p):
                sq += float((md.astype(np.float64) ** 2).sum())
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
    norm = math.sqrt(sq); target = p * NORM_SPARSE
    if abs(norm / target - 1) > TOL:
        raise SystemExit(f"realised norm {norm} misses p*||dSparse|| = {target}")
    rec = {"experiment": "CONFIRM-001", "arm": "SCALAR_P",
           "construction": f"Supa + p*dSparse, p={p!r} read from RAND_MASK/BUILD.json, FP32",
           "p": p, "rand_mask_weight_sha256": mask_sha,
           "realised_norm": norm, "expected_norm": target, "relative_error": norm / target - 1,
           "weight_sha256": sha256(out_dir / "model.safetensors"),
           "weight_bytes": (out_dir / "model.safetensors").stat().st_size,
           "status": "MATERIALIZED_NOT_EVALUATED", "quality_claim": False}
    (out_dir / "BUILD.json").write_bytes((json.dumps(rec, indent=2) + "\n").encode())
    (out_dir / "INCOMPLETE.json").unlink()
    return rec


def verify(out_dir: Path, p: float) -> None:
    """Re-derive the first chunk of every tensor independently; require bitwise equality."""
    base = ST(Path(ROOTS["supa"]) / "model.safetensors"); sp = ST(Path(ROOTS["titan_sparse"]) / "model.safetensors")
    got = ST(out_dir / "model.safetensors")
    bad = checked = 0
    for k in base.keys():
        for lo, hi, w0, md in scaled_chunks(base, sp, k, p):
            g = got.rows(k, lo, hi)
            exp = (w0 + md).astype(np.float32)
            bad += int((g.view(np.uint32) != exp.view(np.uint32)).sum()); checked += g.size
            break
    base.close(); sp.close(); got.close()
    print(f"  verify SCALAR_P: {checked:,} elements re-derived, {bad} mismatches", flush=True)
    if bad:
        raise SystemExit("VERIFY FAILED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mask", type=Path, required=True, help="the RAND_MASK arm directory")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    p = read_p(a.mask)
    mask_sha = json.loads((a.mask / "BUILD.json").read_text(encoding="utf-8"))["weight_sha256"]
    r = build(a.output, p, mask_sha)
    print(json.dumps({k: r[k] for k in ("p", "realised_norm", "expected_norm", "relative_error", "weight_sha256")}, indent=2))
    verify(a.output, p)
