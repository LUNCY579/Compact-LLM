"""HDOSE-001 — build theta_supa + alpha * P16(delta_H), a uniform scalar on the gate delta.

    python probe/build_hdose.py --alpha 0.50 --out <records>/hdose001-20260920/arm-a050

Registered in `probe/HDOSE001_PREREG.md` before this file was run.

`SCALED_H` (= `s_t * P16(delta_H)`, `s_t` median 0.675) is the highest `J_decision` on record and
contains no `delta_T`. But 0.675 was never chosen for H -- it is whatever `s_t` happened to be,
a ratio computed from `delta_T`'s norms for an unrelated purpose. This sweeps the parameter.

## The joint canary, adopted from SCALED-H-001

A `delta_H`-only build has nothing of its own to check against, so this recomputes **both** of
MAG-001's recorded per-tensor norms:

    ||P16(delta_T,t)||                     == recorded norm_T
    ||P16(delta_T,t) + P16(delta_H,t)||    == recorded norm_combined

That is one pass checking the source, **both** trained checkpoints, the tensor mapping and the
SVD path together. It costs an extra SVD over `delta_T` which is otherwise unused here, and that
cost is the entire point.

The statistic is `sqrt(sum of top-16 squared singular values)` -- **not** the Frobenius norm of
the reconstructed `U.S.Vh`, which is lossy by ~6e-5 and failed 140/140 when SCALED-T-001 first
used it. Tolerance 2e-6, calibrated in SCALED-T-001 Amendment 1.

**Nothing is written unless every tensor passes both checks.**
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

BASE = Path("C:/Users/clclc/BusinessProject")
REC = BASE / "rineya-dataset-retention-records"
SRC_DIR = BASE / "Chat-Bubble/Rineya_Published_Cores/Rineya-Supa-1.7.0-devfreeze"
SRC = SRC_DIR / "model.safetensors"
TRAINED_T = REC / "int001-20260917/trained_fp32.safetensors"
TRAINED_H = REC / "compose001-20260918/trained_H_fp32.safetensors"
TH16S = REC / "mag001-20260918/arm-TH16s/MATERIALIZATION.json"
RANK = 16
TOL = 2e-6
SIDECARS = ("config.json", "generation_config.json", "tokenizer.json",
            "tokenizer_config.json", "merges.txt", "vocab.json")


def p16(delta: torch.Tensor):
    """Rank-16 truncation plus the norm computed the way MAG-001 computed it."""
    if delta.ndim != 2:
        return delta.clone(), float(torch.linalg.norm(delta.double()))
    d = delta.double()
    U, S, Vh = torch.linalg.svd(d, full_matrices=False)
    return ((U[:, :RANK] * S[:RANK]) @ Vh[:RANK]).float(), float((S[:RANK] ** 2).sum().sqrt())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)

    if a.out.exists():
        print(f"REFUSED: {a.out} exists. Arms are create-only.")
        return 2

    scales = json.loads(TH16S.read_text(encoding="utf-8"))["norm_bookkeeping"]["per_tensor_scales"]
    by = {e["tensor"]: e for e in scales}
    print(f"alpha = {a.alpha}   canary over {len(by)} recorded tensors (norm_T AND norm_combined)")

    t0 = time.time()
    out: dict[str, torch.Tensor] = {}
    worst_t = worst_c = 0.0
    fails = []
    total = 0.0

    with safe_open(str(SRC), framework="pt") as S, \
         safe_open(str(TRAINED_T), framework="pt") as T, \
         safe_open(str(TRAINED_H), framework="pt") as H:
        for i, name in enumerate(list(S.keys())):
            src = S.get_tensor(name)
            if name not in by:
                out[name] = src.clone()
                continue
            dt, nt = p16(T.get_tensor(name).float() - src.float())
            dh, _ = p16(H.get_tensor(name).float() - src.float())
            nc = float(torch.linalg.norm((dt + dh).double())) if dt.ndim != 2 else \
                float((torch.linalg.svdvals((dt + dh).double())[:RANK] ** 2).sum().sqrt())
            rt = abs(nt - by[name]["norm_T"]) / max(by[name]["norm_T"], 1e-12)
            rc = abs(nc - by[name]["norm_combined"]) / max(by[name]["norm_combined"], 1e-12)
            worst_t, worst_c = max(worst_t, rt), max(worst_c, rc)
            if rt >= TOL or rc >= TOL:
                fails.append((name, rt, rc))
            out[name] = (src.float() + a.alpha * dh).float()
            total += float((a.alpha * dh).double().pow(2).sum())
            del src, dt, dh
            if (i + 1) % 80 == 0:
                print(f"  ...{i+1} tensors, worst norm_T {worst_t:.2e}, norm_combined {worst_c:.2e}",
                      flush=True)

    print(f"\nCANARY: worst norm_T {worst_t:.3e}   worst norm_combined {worst_c:.3e}   (tol {TOL:.0e})")
    if fails:
        print(f"REFUSED: {len(fails)} tensors exceed tolerance. Nothing written.")
        for n, rt, rc in fails[:5]:
            print(f"   {n}  norm_T {rt:.2e}  norm_combined {rc:.2e}")
        return 1
    print(f"CANARY PASSED on both norms, all {len(by)} tensors")

    a.out.mkdir(parents=True)
    w = a.out / "model.safetensors"
    save_file(out, str(w), metadata={"format": "pt"})
    for s in SIDECARS:
        if (SRC_DIR / s).is_file():
            shutil.copyfile(SRC_DIR / s, a.out / s)
    digest = hashlib.sha256(w.read_bytes()).hexdigest()
    (a.out / "MATERIALIZATION.json").write_text(json.dumps({
        "schema": 1, "record": "CHECKPOINT_MATERIALIZATION", "experiment": "HDOSE-001",
        "layout": "huggingface", "alpha": a.alpha, "rank": RANK,
        "what_this_is": (f"The frozen source plus {a.alpha} * P16(delta_H), a UNIFORM scalar on "
                         f"the gate delta. No delta_T present. No model was run for these bytes."),
        "registration": "experiments/delta_multi_reasoning_v1/probe/HDOSE001_PREREG.md",
        "source": str(SRC_DIR), "trained_for_delta_H": str(TRAINED_H),
        "canary": {"statistic": "sqrt(sum top-16 squared singular values), fp64",
                   "tolerance_relative": TOL, "worst_norm_T": worst_t,
                   "worst_norm_combined": worst_c, "tensors": len(by), "passed": True},
        "arm_update_total_fro": total ** 0.5, "out_weight_sha256": digest,
        "wall_seconds": round(time.time() - t0, 1),
        "reclaim_policy": ("weights are reclaimed after scoring per HDOSE001_PREREG.md; the "
                           "construction and this hash make the arm rebuildable in ~80 s"),
        "quality_claim": False, "release_permitted": False,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {w}\n  sha256 {digest}\n  ||alpha*P16(dH)|| = {total ** 0.5:.12f}\n  {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
