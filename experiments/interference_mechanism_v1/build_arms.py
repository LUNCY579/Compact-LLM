"""INTERFERENCE-MECH-001 step 1 — build the exploratory arms, FP32 streaming, create-only.

All arms are W = BASE + A(dS) + B(dO), where dS = SPARSE - BASE and dO = ADD-100 - SPARSE are read
from the existing FP32 arms in titan_opallo_skill_vector_v1/arms (read-only). A and B are scalings,
optionally with a Bernoulli mask that is rescaled by 1/keep (drop-and-rescale, "DARE").

  SPARSE-125      1.25*dS
  ADD-A125        1.25*dS + dO
  DARE-O50        dS + M50(dO)/0.5        M50 keeps each non-zero of dO with prob 0.5
  DARE-O10        dS + M10(dO)/0.1        keeps 10%
  DISJOINT-DARE   M(dS)/0.5 + (1-M)(dO)/0.5, with ONE mask M over every element, so the two
                  donors never share a coordinate

Masks are deterministic. Each tensor draws from its own RNG seeded by (SEED, arm, tensor name),
consumed row-chunk by row-chunk in a fixed order. A uniform draw is made for EVERY element, so the
stream does not depend on which elements are non-zero.

POSITIVE CONTROLS, required before anything is built. The same recipe with (A=1, B=1), (A=1, B=0)
and (A=0.5, B=0.5), no masks, must reproduce ADD-100, SPARSE and AVG within 1e-6.
"""
from __future__ import annotations

import argparse, hashlib, json, math, os, shutil, struct, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP / "titan_model_soup_mini_v1" / "local-runs"))
from verify_independent import ST  # noqa: E402

SRC = EXP / "titan_opallo_skill_vector_v1" / "arms"
ASSETS = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "generation_config.json", "config.json")
CHUNK = 4_000_000
SEED = 20260925
TOL = 1e-6

# spec: (a, b, mask_kind, keep)
ARMS = {
    "SPARSE-125": (1.25, 0.0, None, None),
    "ADD-A125": (1.25, 1.0, None, None),
    "DARE-O50": (1.0, 1.0, "dO", 0.5),
    "DARE-O10": (1.0, 1.0, "dO", 0.1),
    "DISJOINT-DARE": (1.0, 1.0, "disjoint", 0.5),
}
CONTROLS = {"ADD-100": (1.0, 1.0, None, None), "SPARSE": (1.0, 0.0, None, None), "AVG": (0.5, 0.5, None, None)}


def rng_for(arm: str, key: str) -> np.random.Generator:
    h = int.from_bytes(hashlib.sha256(f"{SEED}:{arm}:{key}".encode()).digest()[:8], "little")
    return np.random.default_rng(h)


def chunks(base, sp, ad, key, arm, spec):
    a, b, kind, keep = spec
    shape = base.info(key)[1]
    per_row = math.prod(shape[1:]) if len(shape) > 1 else 1
    step = max(1, CHUNK // per_row)
    rng = rng_for(arm, key) if kind else None
    for lo in range(0, shape[0], step):
        hi = min(lo + step, shape[0])
        w0 = base.rows(key, lo, hi); s = sp.rows(key, lo, hi); o = ad.rows(key, lo, hi)
        dS = s - w0; dO = o - s
        if kind is None:
            w = w0 + np.float32(a) * dS + np.float32(b) * dO
        else:
            u = rng.random(dS.shape, dtype=np.float32)
            m = u < np.float32(keep)
            if kind == "dO":
                w = w0 + np.float32(a) * dS + np.where(m & (dO != 0), dO / np.float32(keep), np.float32(0))
            else:  # disjoint: dS on m, dO on ~m, each rescaled so its expectation is unchanged
                w = w0 + np.where(m, dS / np.float32(keep), np.float32(0)) + np.where(~m, dO / np.float32(1 - keep), np.float32(0))
        yield lo, hi, w.astype(np.float32, copy=False)


def open_sources():
    return (ST(SRC / "BASE" / "model.safetensors"), ST(SRC / "SPARSE" / "model.safetensors"),
            ST(SRC / "ADD-100" / "model.safetensors"))


def positive_controls(keys_subset=None) -> dict:
    base, sp, ad = open_sources()
    keys = keys_subset or base.keys()
    worst = {}
    for name, spec in CONTROLS.items():
        tgt = ST(SRC / name / "model.safetensors")
        w_ = 0.0
        for k in keys:
            for lo, hi, w in chunks(base, sp, ad, k, name, spec):
                w_ = max(w_, float(np.abs(w - tgt.rows(k, lo, hi)).max()))
        tgt.close(); worst[name] = w_
    for s in (base, sp, ad):
        s.close()
    return worst


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(8 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def build(arm: str, out: Path, keys_subset=None) -> dict:
    spec = ARMS[arm]
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    base, sp, ad = open_sources()
    keys = keys_subset or base.keys()
    header, off = {"__metadata__": {"format": "pt", "experiment": "INTERFERENCE-MECH-001", "arm": arm}}, 0
    for k in keys:
        shape = base.info(k)[1]; nb = math.prod(shape) * 4
        header[k] = {"dtype": "F32", "shape": shape, "data_offsets": [off, off + nb]}; off += nb
    enc = json.dumps(header, separators=(",", ":"), ensure_ascii=True).encode()
    enc += b" " * (-len(enc) % 8)
    out.mkdir(parents=True, exist_ok=False)
    tmp = out / "model.safetensors.partial"
    sq_s = sq_o = 0.0
    with open(tmp, "xb") as f:
        f.write(struct.pack("<Q", len(enc))); f.write(enc)
        for k in keys:
            for lo, hi, w in chunks(base, sp, ad, k, arm, spec):
                if not np.isfinite(w).all():
                    raise RuntimeError(f"nonfinite in {k}")
                d = (w - base.rows(k, lo, hi)).astype(np.float64)
                sq_s += float((d ** 2).sum())
                f.write(w.astype("<f4", copy=False).tobytes(order="C"))
        f.flush(); os.fsync(f.fileno())
    for s in (base, sp, ad):
        s.close()
    if tmp.stat().st_size != 8 + len(enc) + off:
        raise RuntimeError("byte count mismatch")
    for a in ASSETS:
        shutil.copyfile(SRC / "SPARSE" / a, out / a)
    tmp.rename(out / "model.safetensors")
    rec = {"experiment": "INTERFERENCE-MECH-001 step 1", "arm": arm, "spec": dict(zip(("a", "b", "mask", "keep"), spec)),
           "seed": SEED, "delta_norm_vs_BASE": math.sqrt(sq_s),
           "weight_sha256": sha256(out / "model.safetensors"), "quality_claim": False}
    (out / "BUILD.json").write_bytes((json.dumps(rec, indent=2) + "\n").encode())
    return rec


def verify(arm: str, out: Path) -> None:
    base, sp, ad = open_sources()
    got = ST(out / "model.safetensors")
    bad = 0
    for k in got.keys():
        for lo, hi, w in chunks(base, sp, ad, k, arm, ARMS[arm]):
            bad += int((got.rows(k, lo, hi).view(np.uint32) != w.view(np.uint32)).sum())
            break
    for s in (base, sp, ad, got):
        s.close()
    print(f"  verify {arm}: {bad} mismatches on first chunks", flush=True)
    if bad:
        raise SystemExit("VERIFY FAILED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--outdir", type=Path, default=HERE / "arms")
    ap.add_argument("--smoke-keys", nargs="*", default=None, help="restrict to these tensors (smoke test only)")
    a = ap.parse_args()
    worst = positive_controls(a.smoke_keys)
    print("positive controls (max abs error):", {k: f"{v:.2e}" for k, v in worst.items()}, flush=True)
    if any(v > TOL for v in worst.values()):
        raise SystemExit("POSITIVE CONTROL FAILED")
    for arm in a.arms:
        out = a.outdir / arm
        print(json.dumps(build(arm, out, a.smoke_keys)), flush=True)
        verify(arm, out)
