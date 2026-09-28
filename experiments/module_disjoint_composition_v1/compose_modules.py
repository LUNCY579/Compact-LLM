"""MODULE-DISJOINT-001 — build Supa + dSparse|ATTN [+ dOpallo|MLP].

Streaming FP32 writer: never holds more than one tensor's rows. Sources are opened read-only
through the independent safetensors reader already used for the geometry work; no source
directory is written to, and the output is create-only.

Deltas come from the UNROUNDED FP32 reconstructions, both independently verified 220/220 by
BF16 round-trip against the published cores.
"""
from __future__ import annotations

import argparse, hashlib, json, math, os, shutil, struct, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "delta_subspace_geometry_v1"))
from subspace import ST, ROOTS, OPALLO_TRAINED, ENC2HF  # noqa: E402

ASSETS = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
          "generation_config.json")
ATTN_MARKERS = ("self_attn.", "input_layernorm", "post_attention_layernorm")

# Per-arm scalar coefficients, each norm-matching its own module-restricted partner exactly.
# Measured, not inferred: ||dSparse||=0.6940550846962167, |ATTN=0.4400615543769171,
# |MLP=0.5367106193770012.
SCALAR_ALPHAS = {
    "SCALAR_MATCHED":     0.4400615543769171 / 0.6940550846962167,  # matches SPARSE_ATTN
    "SCALAR_MATCHED_MLP": 0.5367106193770012 / 0.6940550846962167,  # matches SPARSE_MLP
}
SCALAR_ALPHA = SCALAR_ALPHAS["SCALAR_MATCHED"]  # back-compat for already-built arms

ARMS = {
    # name: (sparse_module_class or None, opallo_module_class or None)
    "SPARSE_ATTN": ("ATTN", None),
    "SCALAR_MATCHED": ("ALL_SCALED", None),
    "SCALAR_MATCHED_MLP": ("ALL_SCALED", None),
    "MODULE_DISJOINT": ("ATTN", "MLP"),
    # controls available if needed; not in the registered set
    "SPARSE_MLP": ("MLP", None),
    "OPALLO_MLP": (None, "MLP"),
}


def module_class(key: str) -> str:
    return "ATTN" if any(t in key for t in ATTN_MARKERS) else "MLP"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build(arm: str, out_dir: Path) -> dict:
    s_cls, o_cls = ARMS[arm]
    alpha = SCALAR_ALPHAS.get(arm, SCALAR_ALPHA)
    out_dir = out_dir.resolve()
    if out_dir.exists():
        raise SystemExit(f"refusing to overwrite {out_dir}")
    supa_root = Path(ROOTS["supa"])

    base = ST(supa_root / "model.safetensors")
    sp = ST(Path(ROOTS["titan_sparse"]) / "model.safetensors")
    ot = ST(OPALLO_TRAINED)
    omap = {f"model.layers.{k.split('.', 2)[1]}.{ENC2HF[k.split('.', 2)[2]]}": k for k in ot.keys()}

    keys = base.keys()
    header, offset = {"__metadata__": {"format": "pt", "experiment": "MODULE-DISJOINT-001",
                                       "arm": arm}}, 0
    for k in keys:
        shape = base.info(k)[1]
        nbytes = math.prod(shape) * 4
        header[k] = {"dtype": "F32", "shape": shape, "data_offsets": [offset, offset + nbytes]}
        offset += nbytes
    encoded = json.dumps(header, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)

    out_dir.mkdir(parents=False, exist_ok=False)
    (out_dir / "INCOMPLETE.json").write_text(json.dumps({"arm": arm}), encoding="utf-8")
    target = out_dir / "model.safetensors.partial"

    stats = {"tensors_touched_sparse": 0, "tensors_touched_opallo": 0,
             "sq_from_sparse": 0.0, "sq_from_opallo": 0.0, "changed": 0}
    with target.open("xb") as f:
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)
        for k in keys:
            shape = base.info(k)[1]
            per_row = math.prod(shape[1:]) if len(shape) > 1 else 1
            step = max(1, 4_000_000 // per_row)
            cls = module_class(k)
            scaled = s_cls == "ALL_SCALED"
            use_s = s_cls is not None and k in omap and (scaled or cls == s_cls)
            use_o = o_cls is not None and cls == o_cls and k in omap
            if use_s:
                stats["tensors_touched_sparse"] += 1
            if use_o:
                stats["tensors_touched_opallo"] += 1
            for lo in range(0, shape[0], step):
                hi = min(lo + step, shape[0])
                w = base.rows(k, lo, hi).astype(np.float32)
                out = w.copy()
                if use_s:
                    d = sp.rows(k, lo, hi).astype(np.float32) - w
                    if scaled:
                        d = d * np.float32(alpha)
                    out += d
                    stats["sq_from_sparse"] += float((d.astype(np.float64) ** 2).sum())
                if use_o:
                    d = ot.rows(omap[k], lo, hi).astype(np.float32) - w
                    out += d
                    stats["sq_from_opallo"] += float((d.astype(np.float64) ** 2).sum())
                if not np.isfinite(out).all():
                    raise RuntimeError(f"nonfinite output in {k}")
                stats["changed"] += int((out != w).sum())
                f.write(out.astype("<f4", copy=False).tobytes(order="C"))
        f.flush()
        os.fsync(f.fileno())
    if target.stat().st_size != 8 + len(encoded) + offset:
        raise RuntimeError("output byte count mismatch")

    for name in ASSETS:
        shutil.copyfile(supa_root / name, out_dir / name)
    cfg = json.loads((supa_root / "config.json").read_text(encoding="utf-8"))
    cfg["torch_dtype"] = "float32"
    if "dtype" in cfg:
        cfg["dtype"] = "float32"
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    target.rename(out_dir / "model.safetensors")
    for h in (base, sp, ot):
        h.close()

    receipt = {
        "experiment": "MODULE-DISJOINT-001", "arm": arm,
        "sparse_module_class": s_cls, "opallo_module_class": o_cls,
        "scalar_alpha": alpha if s_cls == "ALL_SCALED" else None,
        "construction": (
            f"Supa + {alpha:.16g}*dSparse (all 220 trained tensors), FP32" if s_cls == "ALL_SCALED" else
            " + ".join(["Supa"] + ([f"1.0*dSparse|{s_cls}"] if s_cls else []) + ([f"1.0*dOpallo|{o_cls}"] if o_cls else []))
            + ", FP32"),
        "delta_sources": {
            "sparse": "published Rineya-Titan-Sparse FP32 minus published Supa",
            "opallo": "B28last20 trainable.safetensors (unrounded FP32) minus published Supa, "
                      "encoder.* -> model.layers.* remap verified 220/220 by BF16 round-trip"},
        "tensors_touched_sparse": stats["tensors_touched_sparse"],
        "tensors_touched_opallo": stats["tensors_touched_opallo"],
        "norm_from_sparse": math.sqrt(stats["sq_from_sparse"]),
        "norm_from_opallo": math.sqrt(stats["sq_from_opallo"]),
        "changed_elements_vs_supa": stats["changed"],
        "weight_sha256": sha256(out_dir / "model.safetensors"),
        "weight_bytes": (out_dir / "model.safetensors").stat().st_size,
        "stored_dtype": "float32",
        "status": "MATERIALIZED_NOT_EVALUATED",
        "quality_claim": False, "release_permitted": False,
    }
    (out_dir / "BUILD.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    (out_dir / "INCOMPLETE.json").unlink()
    return receipt


def verify(arm: str, out_dir: Path) -> None:
    """Re-derive a sample of tensors independently and require bitwise equality."""
    s_cls, o_cls = ARMS[arm]
    alpha = SCALAR_ALPHAS.get(arm, SCALAR_ALPHA)
    base = ST(Path(ROOTS["supa"]) / "model.safetensors")
    sp = ST(Path(ROOTS["titan_sparse"]) / "model.safetensors")
    ot = ST(OPALLO_TRAINED)
    got = ST(out_dir / "model.safetensors")
    omap = {f"model.layers.{k.split('.', 2)[1]}.{ENC2HF[k.split('.', 2)[2]]}": k for k in ot.keys()}
    bad = checked = untouched_ok = 0
    for k in base.keys():
        shape = base.info(k)[1]
        rows = min(shape[0], 256)
        w = base.rows(k, 0, rows).astype(np.float32)
        exp = w.copy()
        cls = module_class(k)
        if s_cls and k in omap and (s_cls == "ALL_SCALED" or cls == s_cls):
            d = sp.rows(k, 0, rows).astype(np.float32) - w
            if s_cls == "ALL_SCALED":
                d = d * np.float32(alpha)
            exp = exp + d
        if o_cls and cls == o_cls and k in omap:
            exp = exp + (ot.rows(omap[k], 0, rows).astype(np.float32) - w)
        g = got.rows(k, 0, rows)
        n = int((g.view(np.uint32) != exp.view(np.uint32)).sum())
        bad += n
        checked += g.size
        if k not in omap and n == 0:
            untouched_ok += 1
    for h in (base, sp, ot, got):
        h.close()
    print(f"  verify {arm}: {checked:,} elements re-derived independently, {bad} mismatches; "
          f"{untouched_ok} untouched tensors identical to Supa")
    if bad:
        raise SystemExit(f"VERIFY FAILED for {arm}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    r = build(a.arm, a.output)
    print(json.dumps({k: r[k] for k in ("arm", "tensors_touched_sparse", "tensors_touched_opallo",
                                        "norm_from_sparse", "norm_from_opallo",
                                        "changed_elements_vs_supa", "weight_sha256")}, indent=2))
    verify(a.arm, a.output)
