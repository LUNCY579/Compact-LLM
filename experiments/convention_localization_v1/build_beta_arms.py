"""CONVENTION-LOC-001 step 1 — build ADD-O{beta} = SPARSE + beta*(ADD-100 - SPARSE), FP32, streaming.

ADD-100 - SPARSE is dOpallo up to one FP32 rounding of the stored sums, so these arms are Supa +
dSparse + beta*dOpallo to within ~1 ulp of each weight. POSITIVE CONTROL: the same recipe at
beta = 0.75 is compared against the existing ADD-O75 arm (built by a different code path,
titan_opallo_skill_vector_v1/compose.py); the maximum absolute difference must be <= 1e-6, or
nothing is built.

Sources (read-only): titan_opallo_skill_vector_v1/arms/{SPARSE, ADD-100, ADD-O75}.
Output: convention_localization_v1/arms/ADD-O{50,60,90}. Create-only.
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
CONTROL_TOL = 1e-6


def chunks(a, b, key, beta):
    shape = a.info(key)[1]
    per_row = math.prod(shape[1:]) if len(shape) > 1 else 1
    step = max(1, CHUNK // per_row)
    for lo in range(0, shape[0], step):
        hi = min(lo + step, shape[0])
        s = a.rows(key, lo, hi)
        yield lo, hi, s + np.float32(beta) * (b.rows(key, lo, hi) - s)


def positive_control() -> float:
    sp, ad, o75 = (ST(SRC / n / "model.safetensors") for n in ("SPARSE", "ADD-100", "ADD-O75"))
    assert sp.keys() == ad.keys() == o75.keys()
    worst = 0.0
    for k in sp.keys():
        for lo, hi, w in chunks(sp, ad, k, 0.75):
            worst = max(worst, float(np.abs(w - o75.rows(k, lo, hi)).max()))
    for s in (sp, ad, o75):
        s.close()
    return worst


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(8 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def build(beta: float, out: Path) -> dict:
    if out.exists():
        raise SystemExit(f"refusing to overwrite {out}")
    sp, ad = ST(SRC / "SPARSE" / "model.safetensors"), ST(SRC / "ADD-100" / "model.safetensors")
    keys = sp.keys()
    header, off = {"__metadata__": {"format": "pt", "experiment": "CONVENTION-LOC-001", "arm": out.name}}, 0
    for k in keys:
        shape = sp.info(k)[1]; nb = math.prod(shape) * 4
        header[k] = {"dtype": "F32", "shape": shape, "data_offsets": [off, off + nb]}; off += nb
    enc = json.dumps(header, separators=(",", ":"), ensure_ascii=True).encode()
    enc += b" " * (-len(enc) % 8)
    out.mkdir(parents=True, exist_ok=False)
    tmp = out / "model.safetensors.partial"
    with open(tmp, "xb") as f:
        f.write(struct.pack("<Q", len(enc))); f.write(enc)
        for k in keys:
            for _, _, w in chunks(sp, ad, k, beta):
                if not np.isfinite(w).all():
                    raise RuntimeError(f"nonfinite in {k}")
                f.write(w.astype("<f4", copy=False).tobytes(order="C"))
        f.flush(); os.fsync(f.fileno())
    sp.close(); ad.close()
    if tmp.stat().st_size != 8 + len(enc) + off:
        raise RuntimeError("byte count mismatch")
    for a in ASSETS:
        shutil.copyfile(SRC / "SPARSE" / a, out / a)
    tmp.rename(out / "model.safetensors")
    rec = {"experiment": "CONVENTION-LOC-001 step 1", "arm": out.name, "beta": beta,
           "construction": "SPARSE + beta*(ADD-100 - SPARSE), FP32, streaming",
           "weight_sha256": sha256(out / "model.safetensors"), "quality_claim": False}
    (out / "BUILD.json").write_bytes((json.dumps(rec, indent=2) + "\n").encode())
    return rec


def verify(beta: float, out: Path) -> None:
    sp, ad, got = ST(SRC / "SPARSE" / "model.safetensors"), ST(SRC / "ADD-100" / "model.safetensors"), ST(out / "model.safetensors")
    bad = 0
    for k in sp.keys():
        for lo, hi, w in chunks(sp, ad, k, beta):
            bad += int((got.rows(k, lo, hi).view(np.uint32) != w.view(np.uint32)).sum())
            break
    for s in (sp, ad, got):
        s.close()
    print(f"  verify {out.name}: {bad} mismatches on first chunks", flush=True)
    if bad:
        raise SystemExit("VERIFY FAILED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--betas", type=float, nargs="+", default=[0.5, 0.6, 0.9])
    ap.add_argument("--control-only", action="store_true")
    a = ap.parse_args()
    worst = positive_control()
    print(f"positive control: max |SPARSE + 0.75*(ADD-100 - SPARSE) - ADD-O75| = {worst:.3e} (tol {CONTROL_TOL})", flush=True)
    if worst > CONTROL_TOL:
        raise SystemExit("POSITIVE CONTROL FAILED: the recipe does not reproduce ADD-O75")
    if a.control_only:
        raise SystemExit(0)
    for b in a.betas:
        out = HERE / "arms" / f"ADD-O{round(b * 100):02d}"
        print(json.dumps(build(b, out)), flush=True)
        verify(b, out)
