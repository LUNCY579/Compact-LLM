"""Build `theta_base + a*(theta_parent - theta_base)` for an arbitrary positive `a`.

    python analysis/build_scaled_arm.py --base <supa> --parent <arm> --alpha 1.020245875 \
        --out <dir> [--parent-partial] [--expect-norm 0.849139]

## Why this exists instead of reusing an existing builder

Two builders already do this arithmetic and **both correctly refused my request**:

  * `int001-20260917/build_arm.py` pins alpha to INT-001's **registered grid**
    `{0, 0.25, 0.5, 0.75, 1.0}`. That guard is what keeps INT-001's arms on the ladder it
    registered.
  * `lwf001-20260918/scale_own_delta.py` pins alpha to **(0, 1)**, since it exists to scale a
    finished delta *down*.

**Neither guard is weakened, edited, or bypassed.** They protect other experiments' registrations
and are none of this experiment's business. DIRCTL-002 and ROTATION-002 need alphas those
instruments were never registered for — 0.836381727 (off INT-001's grid) and 1.020245875 (an
extrapolation past the parent) — so this builder carries its own discipline instead.

## The discipline it carries

  * **FP32 throughout.** Never routed through BF16: a 0.056%-relative update rounded to BF16
    loses ~60% of its elements, which this programme measured and which silently interpolates.
  * **Endpoints are COPIED, not computed.** At `a == 0` and `a == 1` the tensor is copied
    verbatim, because `base + a*(parent - base)` can round even at `a == 1`.
  * **Extrapolation is legal and is FLAGGED.** `a > 1` is exactly what DIRCTL-002 needs, and the
    receipt records `extrapolation: true` so no later reader mistakes it for an interpolant.
  * **`--expect-norm` makes the build verify itself.** The achieved ‖δ‖ is computed and compared;
    a mismatch beyond tolerance **aborts before writing**, so a wrong arm cannot be scored.
  * **`--parent-partial`** for parents that store only their changed tensors (INT-001's
    `trained_fp32.safetensors` holds 220 of 310; its receipt records the other 90 as untouched).
    Absent tensors are copied from the base, which is what "untouched" means. Opt-in only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

SIDECARS = ("config.json", "generation_config.json", "tokenizer_config.json",
            "tokenizer.json", "vocab.json", "merges.txt")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for blk in iter(lambda: f.read(8 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def resolve(p: Path) -> Path:
    return p if p.is_file() else p / "model.safetensors"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--parent", type=Path, required=True,
                    help="model dir, or a .safetensors file of trained tensors")
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--parent-partial", action="store_true",
                    help="parent stores only changed tensors; absent ones are copied from base. "
                         "Use ONLY where a receipt records those tensors as untouched.")
    ap.add_argument("--expect-norm", type=float, default=None)
    ap.add_argument("--tol", type=float, default=1e-4)
    a = ap.parse_args(argv)

    if a.alpha <= 0:
        raise SystemExit("--alpha must be positive")
    if a.out.exists():
        raise SystemExit(f"create-only output: {a.out} exists")

    import torch  # noqa: PLC0415
    from safetensors import safe_open  # noqa: PLC0415
    from safetensors.torch import save_file  # noqa: PLC0415

    bfile, pfile = resolve(a.base), resolve(a.parent)
    with safe_open(str(bfile), framework="pt", device="cpu") as bf:
        keys = sorted(bf.keys())
    with safe_open(str(pfile), framework="pt", device="cpu") as pf:
        pkeys = set(pf.keys())

    missing = [k for k in keys if k not in pkeys]
    if missing and not a.parent_partial:
        raise SystemExit(f"parent missing {len(missing)} tensors, e.g. {missing[:3]} -- pass "
                         f"--parent-partial ONLY if a receipt records them as untouched")

    out, sq, changed = {}, 0.0, 0
    with safe_open(str(bfile), framework="pt", device="cpu") as bf, \
         safe_open(str(pfile), framework="pt", device="cpu") as pf:
        for k in keys:
            b = bf.get_tensor(k)
            if k not in pkeys:
                out[k] = b.clone()            # untouched by construction
                continue
            p = pf.get_tensor(k)
            if b.dtype != torch.float32 or p.dtype != torch.float32:
                raise SystemExit(f"{k} is not fp32 ({b.dtype}/{p.dtype}); refusing to build")
            if a.alpha == 1.0:
                v = p.clone()                 # copied, never computed
            else:
                v = b + a.alpha * (p - b)
            d = v - b
            s = float(torch.sum(d * d, dtype=torch.float64))
            if s > 0:
                changed += 1
            sq += s
            out[k] = v

    achieved = math.sqrt(sq)
    if a.expect_norm is not None and abs(achieved - a.expect_norm) > a.tol:
        raise SystemExit(f"norm check FAILED: achieved {achieved:.6f} vs expected "
                         f"{a.expect_norm:.6f} (tol {a.tol}); refusing to write")

    a.out.mkdir(parents=True)
    save_file(out, str(a.out / "model.safetensors"), metadata={"format": "pt"})
    src_dir = a.base if a.base.is_dir() else a.base.parent
    for s in SIDECARS:
        f = src_dir / s
        if f.exists():
            shutil.copy2(f, a.out / s)

    receipt = {
        "record": "SCALED_ARM_BUILD",
        "formula": "theta_base + alpha*(theta_parent - theta_base)",
        "alpha": a.alpha,
        "extrapolation": a.alpha > 1.0,
        "extrapolation_note": ("alpha > 1 projects PAST the parent along the same ray; legal "
                               "here and flagged so no reader mistakes it for an interpolant"),
        "base": str(a.base), "parent": str(a.parent),
        "base_weight_sha256": sha256_file(bfile),
        "parent_weight_sha256": sha256_file(pfile),
        "out_weight_sha256": sha256_file(a.out / "model.safetensors"),
        "dtype": "float32", "never_routed_through_bf16": True,
        "endpoints_copied_not_computed": True,
        "parent_partial": a.parent_partial,
        "tensors_total": len(keys), "tensors_changed": changed,
        "tensors_copied_from_base_as_untouched": len(missing),
        "achieved_delta_fro": achieved,
        "expected_delta_fro": a.expect_norm,
        "norm_abs_err": (abs(achieved - a.expect_norm) if a.expect_norm is not None else None),
        "quality_claim": False, "release_permitted": False,
        "guards_not_weakened": ("int001 build_arm.py's registered-grid guard and "
                                "lwf001 scale_own_delta.py's (0,1) guard were neither edited "
                                "nor bypassed; this builder exists so they did not have to be"),
    }
    (a.out / "ARM_BUILD.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"built {a.out}")
    print(f"  alpha {a.alpha}{'  (EXTRAPOLATION past the parent)' if a.alpha > 1 else ''}")
    print(f"  ||delta|| achieved {achieved:.6f}"
          + (f"  expected {a.expect_norm:.6f}  err {abs(achieved-a.expect_norm):.2e}"
             if a.expect_norm is not None else ""))
    print(f"  tensors changed {changed}/{len(keys)}"
          + (f", {len(missing)} copied from base as untouched" if missing else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
