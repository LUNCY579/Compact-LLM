"""DELTA-SUBSPACE-GEOMETRY-001 — does elementwise orthogonality imply functional non-interference?

Motivation
----------
For a linear layer W = W0 + a*dS + b*dO the outputs add: W x = W0 x + a dS x + b dO x.
Under ISOTROPIC inputs (E[x x^T] = I) the expected overlap of the two perturbations is

    E[<dS x, dO x>] = tr(dS^T dO) = <dS, dO>_F

so the elementwise Frobenius cosine IS the right statistic — and it is ~0.003 here.
But real activations are strongly ANISOTROPIC. The expected overlap is really

    E[<dS x, dO x>] = tr(dS^T dO Sigma),   Sigma = E[x x^T]

which can be large even when tr(dS^T dO) ~ 0, if the two deltas align inside the
high-variance input directions. Elementwise cosine averages that away.

This module measures the structural part of that risk with NO forward pass and NO Sigma:
for each weight matrix it asks whether one delta concentrates its energy inside the
other delta's dominant singular subspace, against an explicit chance baseline.

    collision_in  = ||A V_B^k||_F^2 / ||A||_F^2   /  (k / n_in)
    collision_out = ||U_B^k^T A||_F^2 / ||A||_F^2 /  (k / n_out)

ratio ~ 1  => A is isotropic w.r.t. B's preferred directions (elementwise cosine tells
              the whole story; composition is structurally safe)
ratio >> 1 => A concentrates exactly where B is most active. Elementwise cosine can still
              be ~0 because signs cancel, but the two deltas are competing for the same
              subspace and anisotropic inputs will surface that.

Read-only. No model is loaded and no checkpoint is written.
"""
from __future__ import annotations

import argparse, json, math, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "titan_model_soup_mini_v1" / "local-runs"))
from verify_independent import ST, ROOTS  # noqa: E402

TITAN_TRAINED = r"C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/int001-20260917/trained_fp32.safetensors"
OPALLO_TRAINED = (
    r"C:/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification/experiments/"
    r"architecture_benchmark_001/outputs/arm-checkpoints/"
    r"B28last20_lr1e-05_b5200_seed17_wsd0.2-0.2_b20.95/trainable.safetensors"
)
ENC2HF = {
    "input_norm.weight": "input_layernorm.weight",
    "post_norm.weight": "post_attention_layernorm.weight",
    "q_norm.weight": "self_attn.q_norm.weight",
    "k_norm.weight": "self_attn.k_norm.weight",
    "q_proj.weight": "self_attn.q_proj.weight",
    "k_proj.weight": "self_attn.k_proj.weight",
    "v_proj.weight": "self_attn.v_proj.weight",
    "o_proj.weight": "self_attn.o_proj.weight",
    "mlp.gate_proj.weight": "mlp.gate_proj.weight",
    "mlp.up_proj.weight": "mlp.up_proj.weight",
    "mlp.down_proj.weight": "mlp.down_proj.weight",
}
RNG_SEED = 20260922


def randomized_subspace(M: np.ndarray, k: int, rng, power: int = 2):
    """Top-k right and left singular subspaces of M via a randomized range finder."""
    m, n = M.shape
    k = min(k, m, n)
    Omega = rng.standard_normal((n, k + 8), dtype=np.float32)
    Y = M @ Omega
    for _ in range(power):                       # power iterations sharpen the top subspace
        Y = M @ (M.T @ Y)
    Q, _ = np.linalg.qr(Y)                       # Q: m x (k+8), approx left subspace
    B = Q.T @ M                                  # (k+8) x n
    Ub, s, Vt = np.linalg.svd(B, full_matrices=False)
    U = (Q @ Ub)[:, :k]
    V = Vt[:k].T
    return U, V, s[:k]


def effective_rank(s: np.ndarray) -> float:
    """exp(entropy of the normalized singular-value spectrum)."""
    p = s ** 2
    tot = p.sum()
    if tot <= 0:
        return 0.0
    p = p / tot
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def analyse(k: int, out_path: Path):
    rng = np.random.default_rng(RNG_SEED)
    base = ST(Path(ROOTS["supa"]) / "model.safetensors")
    tt = ST(TITAN_TRAINED)
    ot = ST(OPALLO_TRAINED)
    sp = ST(Path(ROOTS["titan_sparse"]) / "model.safetensors")
    omap = {f"model.layers.{key.split('.', 2)[1]}.{ENC2HF[key.split('.', 2)[2]]}": key for key in ot.keys()}

    rows = []
    for key in sorted(tt.keys()):
        shape = tt.info(key)[1]
        if len(shape) != 2:
            continue                                     # norms are 1-D; no subspace to speak of
        m, n = shape
        W0 = base.rows(key, 0, m).astype(np.float32)
        A_full = tt.rows(key, 0, m).astype(np.float32) - W0      # dTitan (unrounded)
        Asp = sp.rows(key, 0, m).astype(np.float32) - W0         # dTitan-Sparse
        B = ot.rows(omap[key], 0, m).astype(np.float32) - W0     # dOpallo (unrounded)

        UB, VB, sB = randomized_subspace(B, k, rng)
        rec = {
            "tensor": key,
            "block": int(key.split(".")[2]),
            "kind": key.split(".", 3)[3].replace(".weight", ""),
            "shape": [m, n],
            "k": int(min(k, m, n)),
            "norm_dTitan": float(np.linalg.norm(A_full)),
            "norm_dSparse": float(np.linalg.norm(Asp)),
            "norm_dOpallo": float(np.linalg.norm(B)),
            "eff_rank_dOpallo": effective_rank(sB),
        }
        for label, A in (("dTitan", A_full), ("dSparse", Asp)):
            na2 = float((A.astype(np.float64) ** 2).sum())
            if na2 <= 0:
                continue
            UA, VA, sA = randomized_subspace(A, k, rng)
            kk = rec["k"]
            e_in = float((np.asarray(A @ VB, dtype=np.float64) ** 2).sum()) / na2
            e_out = float((np.asarray(UB.T @ A, dtype=np.float64) ** 2).sum()) / na2
            # principal angles between the two top-k subspaces
            pa_in = np.linalg.svd(VA.T @ VB, compute_uv=False)
            pa_out = np.linalg.svd(UA.T @ UB, compute_uv=False)
            rec[label] = {
                "elementwise_cos": float(
                    (A.astype(np.float64) * B.astype(np.float64)).sum()
                    / (math.sqrt(na2) * np.linalg.norm(B.astype(np.float64)))
                ),
                "energy_in_B_rowspace": e_in,
                "chance_in": kk / n,
                "collision_in": e_in / (kk / n),
                "energy_in_B_colspace": e_out,
                "chance_out": kk / m,
                "collision_out": e_out / (kk / m),
                "mean_cos2_principal_angles_in": float((pa_in ** 2).mean()),
                "mean_cos2_principal_angles_out": float((pa_out ** 2).mean()),
                "eff_rank": effective_rank(sA),
            }
        rows.append(rec)
        print(f"  {key:<52} coll_in={rec['dSparse']['collision_in']:6.2f} "
              f"coll_out={rec['dSparse']['collision_out']:6.2f} "
              f"elem_cos={rec['dSparse']['elementwise_cos']:+.4f}", flush=True)

    for h in (base, tt, ot, sp):
        h.close()

    def agg(label, field):
        v = [r[label][field] for r in rows if label in r]
        return {"mean": float(np.mean(v)), "median": float(np.median(v)),
                "min": float(np.min(v)), "max": float(np.max(v))}

    summary = {
        "experiment": "DELTA-SUBSPACE-GEOMETRY-001",
        "k": k,
        "n_matrices": len(rows),
        "seed": RNG_SEED,
        "interpretation": (
            "collision_* is energy-in-the-other-delta's-top-k-subspace divided by the chance value k/dim. "
            "~1 means isotropic relative to the other delta (elementwise cosine is sufficient); "
            ">>1 means the deltas compete for the same subspace despite a near-zero elementwise cosine."
        ),
        "aggregate": {
            lab: {f: agg(lab, f) for f in
                  ("elementwise_cos", "collision_in", "collision_out",
                   "mean_cos2_principal_angles_in", "mean_cos2_principal_angles_out", "eff_rank")}
            for lab in ("dTitan", "dSparse")
        },
        "per_tensor": rows,
        "caveat": (
            "Structural only. No activations, no Sigma, no forward pass. A low collision index bounds "
            "the structural risk; it does not prove functional non-interference, and a high one does not "
            "prove interference. This does not license any quality, retention or release claim."
        ),
    }
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--output", type=Path, default=Path("SUBSPACE.json"))
    a = ap.parse_args()
    s = analyse(a.k, a.output)
    print(json.dumps(s["aggregate"], indent=2))
