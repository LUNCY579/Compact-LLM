"""CONVENTION-LOC-001 step 1 — build ADD-O{beta} = SPARSE + beta*(ADD-100 - SPARSE), FP32, streaming,
and the module-restricted arms H1 (dOpallo on MLP tensors only) and H1c (dOpallo on attention +
layer-norm tensors only), proposed by the Titan-Opallo audit session
(titan_opallo_exact_endpoint_v3/proposal_offline/FOLLOWUP_PROPOSAL.md, section 2).

ADD-100 - SPARSE is dOpallo up to one FP32 rounding of the stored sums, so these arms are Supa +
dSparse + beta*dOpallo to within ~1 ulp of each weight. POSITIVE CONTROL: the same recipe at
beta = 0.75 is compared against the existing ADD-O75 arm (built by a different code path,
titan_opallo_skill_vector_v1/compose.py); the maximum absolute difference must be <= 1e-6, or
nothing is built.

Sources (read-only): titan_opallo_skill_vector_v1/arms/{SPARSE, ADD-100, ADD-O75}.
Output: convention_localization_v1/arms/{ADD-O50, ADD-O60, ADD-O90, H1, H1c}. Create-only.
A partition check requires dOpallo to be zero outside MLP and attention/layer-norm tensors, so H1 and
H1c split it exactly.
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


def module_of(key: str) -> str:
    """MLP / ATTN_LN / OTHER for a Qwen-namespace tensor name."""
    if ".mlp." in key:
        return "MLP"
    if ".self_attn." in key or "layernorm" in key:
        return "ATTN_LN"
    return "OTHER"


def coef_fn(spec: str):
    if spec.startswith("beta:"):
        b = float(spec.split(":", 1)[1]); return lambda k: b
    if spec == "H1":
        return lambda k: 1.0 if module_of(k) == "MLP" else 0.0
    if spec == "H1c":
        return lambda k: 1.0 if module_of(k) == "ATTN_LN" else 0.0
    raise ValueError(spec)


def partition_check() -> dict:
    """Every tensor where ADD-100 differs from SPARSE must be MLP or ATTN_LN, never OTHER, so H1 and
    H1c partition dOpallo exactly. Returns the squared norm of the delta by module class."""
    sp, ad = ST(SRC / "SPARSE" / "model.safetensors"), ST(SRC / "ADD-100" / "model.safetensors")
    sq = {"MLP": 0.0, "ATTN_LN": 0.0, "OTHER": 0.0}
    for k in sp.keys():
        shape = sp.info(k)[1]
        per_row = math.prod(shape[1:]) if len(shape) > 1 else 1
        step = max(1, CHUNK // per_row)
        for lo in range(0, shape[0], step):                  # streamed: never a whole tensor in float64
            hi = min(lo + step, shape[0])
            d = (ad.rows(k, lo, hi) - sp.rows(k, lo, hi)).astype(np.float64)
            sq[module_of(k)] += float((d ** 2).sum())
    sp.close(); ad.close()
    if sq["OTHER"] != 0.0:
        raise SystemExit(f"PARTITION CHECK FAILED: dOpallo has squared norm {sq['OTHER']} outside MLP/ATTN_LN")
    return sq


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


def build(spec: str, out: Path) -> dict:
    cf = coef_fn(spec)
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
            for _, _, w in chunks(sp, ad, k, cf(k)):
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
    rec = {"experiment": "CONVENTION-LOC-001 step 1", "arm": out.name, "spec": spec,
           "construction": "SPARSE + c(tensor)*(ADD-100 - SPARSE), FP32, streaming; c from spec",
           "weight_sha256": sha256(out / "model.safetensors"), "quality_claim": False}
    (out / "BUILD.json").write_bytes((json.dumps(rec, indent=2) + "\n").encode())
    return rec


def verify(spec: str, out: Path) -> None:
    cf = coef_fn(spec)
    sp, ad, got = ST(SRC / "SPARSE" / "model.safetensors"), ST(SRC / "ADD-100" / "model.safetensors"), ST(out / "model.safetensors")
    bad = 0
    for k in sp.keys():
        for lo, hi, w in chunks(sp, ad, k, cf(k)):
            bad += int((got.rows(k, lo, hi).view(np.uint32) != w.view(np.uint32)).sum())
            break
    for s in (sp, ad, got):
        s.close()
    print(f"  verify {out.name}: {bad} mismatches on first chunks", flush=True)
    if bad:
        raise SystemExit("VERIFY FAILED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", default=["beta:0.5", "beta:0.6", "beta:0.9", "H1", "H1c"])
    ap.add_argument("--control-only", action="store_true", help="run the positive control and the partition check, then exit")
    ap.add_argument("--no-controls", action="store_true",
                    help="skip both checks; only for a driver that ran --control-only earlier in the SAME run")
    a = ap.parse_args()
    if not a.no_controls:
        worst = positive_control()
        print(f"positive control: max |SPARSE + 0.75*(ADD-100 - SPARSE) - ADD-O75| = {worst:.3e} (tol {CONTROL_TOL})", flush=True)
        if worst > CONTROL_TOL:
            raise SystemExit("POSITIVE CONTROL FAILED: the recipe does not reproduce ADD-O75")
        sq = partition_check()
        print(f"partition check: dOpallo squared norm MLP {sq['MLP']:.4f}, ATTN_LN {sq['ATTN_LN']:.4f}, OTHER 0 "
              f"(MLP share of energy {sq['MLP'] / (sq['MLP'] + sq['ATTN_LN']):.3f})", flush=True)
        if a.control_only:
            raise SystemExit(0)
    else:
        print("controls skipped (--no-controls): the calling driver ran them in this run", flush=True)
    for spec in a.arms:
        name = f"ADD-O{round(float(spec[5:]) * 100):02d}" if spec.startswith("beta:") else spec
        out = HERE / "arms" / name
        print(json.dumps(build(spec, out)), flush=True)
        verify(spec, out)
