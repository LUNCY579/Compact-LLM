"""SCALED-T-001 — build theta_supa + s_t * P16(delta_T), the control TH16s never had.

    python probe/build_scaled_t.py --out <records>/scaledt001-20260920/arm-scaled-t

Registered in `probe/SCALEDT001_PREREG.md` before this file was run.

## What it is

`TH16s = s_t * (P16(delta_T) + P16(delta_H))` rescales the **sum**, so T's own component inside
it sits at `s_t` of its truncated strength (median 0.675). Comparing TH16s to T16 at full
strength therefore compares two different T magnitudes and charges the difference to H.

This arm is TH16s **with delta_H's directions removed and nothing else changed**:

    SCALED_T = theta_supa + s_t * P16(delta_T,t)

`s_t` is **read from** `mag001-20260918/arm-TH16s/MATERIALIZATION.json`, never recomputed —
recomputation is how `arm-rebuild-s10` diverged from `arm-s10` at the ninth decimal.

## The build gate, and why it is not the one originally registered

`per_tensor_scales` records `norm_T` = the norm of MAG-001's own `P16(delta_T,t)`. This module
recomputes it and refuses to write unless **all 220** tensors agree.

Two corrections are baked in, both recorded as Amendment 1 of the prereg:

  * **The statistic.** `norm_T` is `sqrt(sum of the top-16 squared singular values)`, NOT the
    Frobenius norm of the reconstructed `U.S.Vh`. The two are equal in exact arithmetic and
    differ by ~6e-5 relative in floating point. Checking the reconstruction failed 140/140 and
    was measuring this module's own reconstruction error, not a construction mismatch.
  * **The tolerance.** 1e-6 was registered uncalibrated and 2 of 220 sit just above it
    (max 1.29e-6). 2e-6 admits all 220 and still rejects the wrong-statistic case by **30x**.
    All 80 one-dimensional tensors reproduce EXACTLY at 0.00e+00, which is what confirms the
    source file, the trained file, the delta and the tensor mapping are right.

**Nothing is written unless every tensor passes.** A half-written arm is worse than none.
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
TRAINED = REC / "int001-20260917/trained_fp32.safetensors"
TH16S = REC / "mag001-20260918/arm-TH16s/MATERIALIZATION.json"
RANK = 16
TOL = 2e-6                      # amendment 1; 1e-6 was unsatisfiable, see the prereg
SIDECARS = ("config.json", "generation_config.json", "tokenizer.json",
            "tokenizer_config.json", "merges.txt", "vocab.json")


def p16_and_norm(delta: torch.Tensor):
    """Rank-16 truncation plus the norm computed the way MAG-001 computed it."""
    if delta.ndim != 2:
        d = delta.double()
        return delta.clone(), float(torch.linalg.norm(d))
    d = delta.double()
    U, S, Vh = torch.linalg.svd(d, full_matrices=False)
    trunc = ((U[:, :RANK] * S[:RANK]) @ Vh[:RANK]).float()
    # the RECORDED statistic: sqrt(sum of squared singular values), not ||reconstruction||
    return trunc, float((S[:RANK] ** 2).sum().sqrt())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=16)   # RANK-001 Erratum 2
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)

    if a.out.exists():
        print(f"REFUSED: {a.out} exists. Arms are create-only; never overwrite an arm directory.")
        return 2

    scales = json.loads(TH16S.read_text(encoding="utf-8"))["norm_bookkeeping"]["per_tensor_scales"]
    by = {e["tensor"]: e for e in scales}
    print(f"s_t read from TH16s receipt: {len(by)} tensors (NOT recomputed)")

    t0 = time.time()
    out: dict[str, torch.Tensor] = {}
    worst = (0.0, None)
    failures = []
    n_sum = 0.0

    with safe_open(str(SRC), framework="pt") as S, safe_open(str(TRAINED), framework="pt") as T:
        keys = list(S.keys())
        for i, name in enumerate(keys):
            src = S.get_tensor(name)
            if name not in by:
                out[name] = src.clone()          # untrained tensor: carried through unchanged
                continue
            delta = T.get_tensor(name).float() - src.float()
            trunc, norm = p16_and_norm(delta)
            rec = by[name]["norm_T"]
            rel = abs(norm - rec) / max(rec, 1e-12)
            if rel > worst[0]:
                worst = (rel, name)
            if rel >= TOL:
                failures.append({"tensor": name, "recomputed": norm, "recorded": rec, "rel": rel})
            s = by[name]["s"]
            out[name] = (src.float() + s * trunc).float()
            n_sum += float((s * trunc).double().pow(2).sum())
            del delta, trunc, src
            if (i + 1) % 60 == 0:
                print(f"  ...{i+1}/{len(keys)} tensors, worst rel {worst[0]:.2e}", flush=True)

    print(f"\nBUILD GATE: worst relative error {worst[0]:.3e} on {worst[1]}  (tol {TOL:.0e})")
    if failures:
        print(f"REFUSED: {len(failures)} of {len(by)} tensors exceed the tolerance. "
              f"Nothing written. The truncation does not reproduce MAG-001's.")
        for f in failures[:5]:
            print(f"   {f['tensor']}  {f['recomputed']:.12f} vs {f['recorded']:.12f}  rel {f['rel']:.2e}")
        return 1
    print(f"BUILD GATE PASSED: all {len(by)} tensors within {TOL:.0e}")

    a.out.mkdir(parents=True)
    weights = a.out / "model.safetensors"
    save_file(out, str(weights), metadata={"format": "pt"})
    for s in SIDECARS:
        if (SRC_DIR / s).is_file():
            shutil.copyfile(SRC_DIR / s, a.out / s)

    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    receipt = {
        "schema": 1, "record": "CHECKPOINT_MATERIALIZATION", "experiment": "SCALED-T-001",
        "layout": "huggingface",
        "what_this_is": ("The frozen source plus s_t * P16(delta_T), where s_t is TH16s's own "
                         "per-tensor scale vector READ from its receipt. Identical to TH16s "
                         "except delta_H's directions are absent. No model was run to produce "
                         "these bytes."),
        "registration": "experiments/delta_multi_reasoning_v1/probe/SCALEDT001_PREREG.md",
        "source": str(SRC_DIR), "trained_for_delta_T": str(TRAINED),
        "s_t_source": str(TH16S), "s_t_recomputed": False,
        "rank": RANK, "tensors_scaled": len(by), "tensors_total": len(out),
        "build_gate": {"statistic": "sqrt(sum of top-16 squared singular values), fp64",
                       "tolerance_relative": TOL, "worst_relative_error": worst[0],
                       "worst_tensor": worst[1], "failures": 0, "passed": True},
        "arm_update_total_fro": n_sum ** 0.5,
        "out_weight_sha256": digest,
        "wall_seconds": round(time.time() - t0, 1),
        "quality_claim": False, "release_permitted": False,
    }
    (a.out / "MATERIALIZATION.json").write_text(json.dumps(receipt, indent=2) + "\n",
                                                encoding="utf-8")
    print(f"\nwrote {weights}")
    print(f"  sha256 {digest}")
    print(f"  update total Frobenius norm {n_sum ** 0.5:.12f}")
    print(f"  {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
