"""Frobenius norms and pairwise cosines of (arm - base), streamed one tensor at a time.

    python analysis/delta_norms.py --base <supa> \
        --arm l0=<...>/arm-lambda0 --arm l05=<...>/arm-lambda0.5 \
        --pair l0:owna05 --out <...>/DELTA_NORMS.json

Exists because the LWF-ALIAS test needs a **within-lane magnitude calibration**: without ‖δ‖
for each arm, "the anchor protected the skill" and "the anchor shrank the update" are the same
observation. This is the same confound the norm audit found in the composition lane, in a
different coordinate.

## Memory discipline

The host has 15.2 GiB of RAM and a GPU job may be running concurrently, so nothing is
materialised whole. `safe_open` reads one tensor at a time:

  * **norms pass** — peak is base + one arm tensor.
  * **cosine pass** — peak is base + two arm tensors, and only for explicitly requested pairs.

Accumulation is in float64 over float32 tensors, because a global Frobenius norm over 220
tensors of a 596M model sums ~6e8 squared terms and float32 accumulation loses digits that the
ratios here depend on.

## The self-check that validates the whole script

`ctrl-own-alpha0.5` is `arm-lambda0`'s own finished delta multiplied by 0.5. So for that pair
the script must report **cosine = 1.000000** and **norm ratio = 0.500000**. If it does not, the
reader is wrong and every number it produces is suspect. That check is run automatically
whenever both arms are present and its verdict is written into the receipt.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def open_st(d: Path):
    from safetensors import safe_open  # noqa: PLC0415
    f = d / "model.safetensors"
    if not f.exists():
        raise SystemExit(f"no model.safetensors in {d}")
    return safe_open(str(f), framework="pt", device="cpu")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--pair", action="append", default=[], metavar="A:B")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    import torch  # noqa: PLC0415

    arms = {}
    for spec in a.arm:
        name, _, p = spec.partition("=")
        arms[name] = Path(p)

    with open_st(a.base) as bf:
        keys = sorted(bf.keys())

    # ---- pass 1: per-arm norms, peak = base + one arm tensor ------------------------------
    norms: dict[str, dict] = {}
    for name, path in arms.items():
        tot = 0.0
        per = {}
        with open_st(a.base) as bf, open_st(path) as af:
            ak = set(af.keys())
            missing = [k for k in keys if k not in ak]
            if missing:
                raise SystemExit(f"arm {name} missing {len(missing)} tensors, e.g. {missing[:3]}")
            for k in keys:
                d = (af.get_tensor(k).to(torch.float64)
                     - bf.get_tensor(k).to(torch.float64))
                s = float(torch.sum(d * d))
                tot += s
                if s > 0:
                    per[k] = math.sqrt(s)
                del d
        norms[name] = {
            "fro": math.sqrt(tot),
            "tensors_changed": len(per),
            "tensors_total": len(keys),
            "largest_tensors": sorted(per.items(), key=lambda kv: -kv[1])[:5],
        }
        print(f"  {name:<12} ||delta|| = {norms[name]['fro']:.6f}   "
              f"changed {len(per)}/{len(keys)}", flush=True)

    # ---- pass 2: cosines, peak = base + two arm tensors -----------------------------------
    cos = {}
    for spec in a.pair:
        x, _, y = spec.partition(":")
        if x not in arms or y not in arms:
            raise SystemExit(f"pair {spec} names an arm that was not supplied")
        dot = 0.0
        with open_st(a.base) as bf, open_st(arms[x]) as xf, open_st(arms[y]) as yf:
            for k in keys:
                b = bf.get_tensor(k).to(torch.float64)
                dx = xf.get_tensor(k).to(torch.float64) - b
                dy = yf.get_tensor(k).to(torch.float64) - b
                dot += float(torch.sum(dx * dy))
                del b, dx, dy
        c = dot / (norms[x]["fro"] * norms[y]["fro"])
        cos[spec] = {"inner": dot, "cosine": c,
                     "norm_ratio_y_over_x": norms[y]["fro"] / norms[x]["fro"]}
        print(f"  cos({x},{y}) = {c:.6f}   ratio = {cos[spec]['norm_ratio_y_over_x']:.6f}",
              flush=True)

    # ---- the self-check -------------------------------------------------------------------
    check = None
    for spec, v in cos.items():
        x, _, y = spec.partition(":")
        if "owna05" in y and ("l0" == x or "lambda0" == x):
            ok = abs(v["cosine"] - 1.0) < 1e-6 and abs(v["norm_ratio_y_over_x"] - 0.5) < 1e-6
            check = {
                "pair": spec, "expected": "cosine 1.0 and norm ratio 0.5 by construction",
                "cosine": v["cosine"], "norm_ratio": v["norm_ratio_y_over_x"],
                "verdict": "PASS -- reader validated" if ok else
                           "FAIL -- the reader is wrong; every number here is suspect",
            }

    out = {
        "record": "DELTA_NORMS",
        "class": "AUDIT -- zero model calls, zero training; streamed tensor reads only",
        "base": str(a.base),
        "arms": {k: str(v) for k, v in arms.items()},
        "accumulation_dtype": "float64",
        "norms": {k: {kk: vv for kk, vv in v.items() if kk != "largest_tensors"}
                  for k, v in norms.items()},
        "largest_changed_tensors": {k: v["largest_tensors"] for k, v in norms.items()},
        "pairs": cos,
        "self_check": check,
    }
    a.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    if check:
        print(f"\nSELF-CHECK: {check['verdict']}")
    print(f"\nwrote {a.out}")
    return 0 if (check is None or check["verdict"].startswith("PASS")) else 3


if __name__ == "__main__":
    raise SystemExit(main())
