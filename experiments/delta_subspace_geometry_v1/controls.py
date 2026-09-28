"""Controls for the subspace-collision finding.

The raw finding is that dSparse concentrates ~3.5x chance inside dOpallo's top-k input
subspace while their elementwise cosine is ~0.003. Before that can mean "these two SKILLS
compete for a subspace", three rival explanations must be excluded:

  R1 BASE-INHERITED. Both deltas were trained on the same W0, so both may simply align with
     W0's own dominant directions. Control: collision(dSparse, W0).
  R2 GENERIC / ARCHITECTURAL. Any two deltas of this shape in this network collide this much.
     Control: cross-block pairing — dSparse from block i against dOpallo from block j, i != j,
     same sublayer kind. Matched in shape, rank, scale and training recipe; differs ONLY in
     whether the two matrices sit at the same depth.
  R3 LOW-RANK ARTEFACT. A rank-18 matrix in a random k-dim subspace trivially exceeds chance.
     Control: collision(dSparse, Haar-random orthonormal k-subspace), which is the null by
     construction and must come out at ~1.0 if the estimator is calibrated.

A finding survives only if same-block collision EXCEEDS all three controls.
Read-only; no forward pass, no writes outside this directory.
"""
from __future__ import annotations

import json, math, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from subspace import ST, ROOTS, TITAN_TRAINED, OPALLO_TRAINED, ENC2HF, randomized_subspace  # noqa: E402

K = 32
SEED = 20260922
KINDS = ["self_attn.o_proj", "mlp.down_proj", "self_attn.v_proj", "mlp.gate_proj",
         "self_attn.k_proj", "mlp.up_proj", "self_attn.q_proj"]


def energy_ratio(A: np.ndarray, V: np.ndarray, k: int, n: int) -> float:
    na2 = float((A.astype(np.float64) ** 2).sum())
    e = float((np.asarray(A @ V, dtype=np.float64) ** 2).sum()) / na2
    return e / (k / n)


def main():
    rng = np.random.default_rng(SEED)
    base = ST(Path(ROOTS["supa"]) / "model.safetensors")
    tt = ST(TITAN_TRAINED)
    ot = ST(OPALLO_TRAINED)
    sp = ST(Path(ROOTS["titan_sparse"]) / "model.safetensors")
    omap = {f"model.layers.{k.split('.', 2)[1]}.{ENC2HF[k.split('.', 2)[2]]}": k for k in ot.keys()}

    # cache per (kind, block): dSparse, and dOpallo's top-k right subspace, and W0's top-k
    store = {}
    for kind in KINDS:
        for b in range(8, 28):
            key = f"model.layers.{b}.{kind}.weight"
            m, n = tt.info(key)[1]
            W0 = base.rows(key, 0, m).astype(np.float32)
            dS = sp.rows(key, 0, m).astype(np.float32) - W0
            dO = ot.rows(omap[key], 0, m).astype(np.float32) - W0
            _, VO, _ = randomized_subspace(dO, K, rng)
            _, VW, _ = randomized_subspace(W0, K, rng)
            store[(kind, b)] = (dS, VO, VW, m, n)
        print(f"  cached {kind}", flush=True)

    out = {"k": K, "seed": SEED, "by_kind": {}}
    for kind in KINDS:
        same, cross, basec, haar = [], [], [], []
        for b in range(8, 28):
            dS, VO, VW, m, n = store[(kind, b)]
            kk = min(K, m, n)
            same.append(energy_ratio(dS, VO, kk, n))
            basec.append(energy_ratio(dS, VW, kk, n))
            Q, _ = np.linalg.qr(rng.standard_normal((n, kk), dtype=np.float32))
            haar.append(energy_ratio(dS, Q, kk, n))
            for b2 in range(8, 28):
                if b2 == b:
                    continue
                VO2 = store[(kind, b2)][1]
                cross.append(energy_ratio(dS, VO2, kk, n))
        f = lambda v: {"median": float(np.median(v)), "mean": float(np.mean(v)), "n": len(v)}
        out["by_kind"][kind] = {
            "same_block_dSparse_in_dOpallo": f(same),
            "cross_block_dSparse_in_dOpallo": f(cross),
            "dSparse_in_W0_topk": f(basec),
            "dSparse_in_haar_random": f(haar),
            "same_over_cross": float(np.median(same) / np.median(cross)),
            "same_over_base": float(np.median(same) / np.median(basec)),
        }
        c = out["by_kind"][kind]
        print(f"  {kind:<20} same={np.median(same):6.2f}  cross={np.median(cross):6.2f}  "
              f"W0={np.median(basec):6.2f}  haar={np.median(haar):5.2f}  "
              f"same/cross={c['same_over_cross']:5.2f}", flush=True)

    for h in (base, tt, ot, sp):
        h.close()
    allsame = [out["by_kind"][k]["same_block_dSparse_in_dOpallo"]["median"] for k in KINDS]
    allcross = [out["by_kind"][k]["cross_block_dSparse_in_dOpallo"]["median"] for k in KINDS]
    allbase = [out["by_kind"][k]["dSparse_in_W0_topk"]["median"] for k in KINDS]
    allhaar = [out["by_kind"][k]["dSparse_in_haar_random"]["median"] for k in KINDS]
    out["overall"] = {
        "median_same": float(np.median(allsame)), "median_cross": float(np.median(allcross)),
        "median_in_W0": float(np.median(allbase)), "median_haar_null": float(np.median(allhaar)),
        "verdict": ("SAME-BLOCK EXCEEDS ALL CONTROLS" if np.median(allsame) > max(np.median(allcross), np.median(allbase)) * 1.2
                    else "NOT SEPARABLE FROM CONTROLS"),
    }
    Path("CONTROLS.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out["overall"], indent=2))


if __name__ == "__main__":
    main()
