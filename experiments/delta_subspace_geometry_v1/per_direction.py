"""Per-direction alignment inside the contended subspace — correcting effective_cosine.py.

effective_cosine.py computed  cos(dS·V, dO·V)  over the whole k-dimensional subspace V. A peer
review pointed out that this aggregates over the k directions and can cancel WITHIN V — exactly
the failure it was meant to expose in the global cosine. One level of averaging was removed, not
all of them.

This measures the alignment DIRECTION BY DIRECTION:

    c_i = cos( dS·v_i , dO·v_i )   for each right singular vector v_i of dO, i = 1..k

and reports the spread of {c_i}, not just its aggregate. If the spread is wide, individual
directions are strongly aligned even though the subspace aggregate is ~0, and the
sign-cancellation concern survives after all. If the spread is narrow and small, it does not.

A null is computed the same way on Haar-random directions, so "wide" has a reference.
"""
from __future__ import annotations

import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from subspace import ST, ROOTS, OPALLO_TRAINED, ENC2HF, randomized_subspace  # noqa: E402

K = 32
SEED = 20260923
KINDS = ["self_attn.o_proj", "mlp.down_proj", "self_attn.v_proj", "mlp.gate_proj",
         "self_attn.k_proj", "mlp.up_proj", "self_attn.q_proj"]


def per_dir_cos(A, B, V):
    """cos(A v_i, B v_i) for each column v_i of V."""
    AV = np.asarray(A @ V, dtype=np.float64)
    BV = np.asarray(B @ V, dtype=np.float64)
    num = (AV * BV).sum(axis=0)
    den = np.linalg.norm(AV, axis=0) * np.linalg.norm(BV, axis=0)
    ok = den > 0
    return num[ok] / den[ok]


def main():
    rng = np.random.default_rng(SEED)
    base = ST(Path(ROOTS["supa"]) / "model.safetensors")
    ot = ST(OPALLO_TRAINED)
    sp = ST(Path(ROOTS["titan_sparse"]) / "model.safetensors")
    omap = {f"model.layers.{k.split('.', 2)[1]}.{ENC2HF[k.split('.', 2)[2]]}": k for k in ot.keys()}

    real, null, agg = [], [], []
    per_kind = {k: [] for k in KINDS}
    for kind in KINDS:
        for b in range(8, 28):
            key = f"model.layers.{b}.{kind}.weight"
            m, n = sp.info(key)[1]
            W0 = base.rows(key, 0, m).astype(np.float32)
            dS = sp.rows(key, 0, m).astype(np.float32) - W0
            dO = ot.rows(omap[key], 0, m).astype(np.float32) - W0
            kk = min(K, m, n)
            _, VO, _ = randomized_subspace(dO, kk, rng)
            Q, _ = np.linalg.qr(rng.standard_normal((n, kk), dtype=np.float32))
            c = per_dir_cos(dS, dO, VO)
            cn = per_dir_cos(dS, dO, Q)
            real.extend(c.tolist()); null.extend(cn.tolist())
            per_kind[kind].extend(np.abs(c).tolist())
            # the aggregate statistic effective_cosine.py reported, for direct comparison
            AV, BV = dS @ VO, dO @ VO
            agg.append(float((AV.astype(np.float64) * BV.astype(np.float64)).sum() /
                             (np.linalg.norm(AV.astype(np.float64)) * np.linalg.norm(BV.astype(np.float64)))))
        print(f"  done {kind}", flush=True)
    for h in (base, ot, sp):
        h.close()

    real = np.array(real); null = np.array(null); agg = np.abs(np.array(agg))

    def d(v):
        return {"n": int(v.size), "mean_abs": float(np.abs(v).mean()),
                "median_abs": float(np.median(np.abs(v))),
                "p90_abs": float(np.percentile(np.abs(v), 90)),
                "p99_abs": float(np.percentile(np.abs(v), 99)),
                "max_abs": float(np.abs(v).max()), "sd": float(v.std()),
                "frac_above_0.05": float((np.abs(v) > 0.05).mean()),
                "frac_above_0.10": float((np.abs(v) > 0.10).mean())}

    out = {"k": K, "seed": SEED,
           "per_direction_in_dOpallo_subspace": d(real),
           "per_direction_in_haar_null": d(null),
           "subspace_aggregate_for_comparison": {
               "n": int(agg.size), "mean_abs": float(agg.mean()), "median_abs": float(np.median(agg)),
               "max_abs": float(agg.max())},
           "by_kind_median_abs_per_direction": {k: float(np.median(v)) for k, v in per_kind.items()},
           "reading": (
               "If per-direction spread greatly exceeds the aggregate AND the Haar null, then the "
               "subspace aggregate was hiding direction-level alignment and the sign-cancellation "
               "concern stands. If per-direction tracks the null, it does not.")}
    Path("PER_DIRECTION.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print()
    print(f"  {'statistic':<42} {'median|c|':>10} {'p90':>8} {'p99':>8} {'max':>8} {'>0.05':>7} {'>0.10':>7}")
    for lab, v in (("per-direction inside dOpallo top-32", real), ("per-direction, Haar null", null)):
        s = d(v)
        print(f"  {lab:<42} {s['median_abs']:10.4f} {s['p90_abs']:8.4f} {s['p99_abs']:8.4f} "
              f"{s['max_abs']:8.4f} {s['frac_above_0.05']:7.3f} {s['frac_above_0.10']:7.3f}")
    print(f"  {'subspace AGGREGATE (what I reported)':<42} {np.median(agg):10.4f} "
          f"{np.percentile(agg,90):8.4f} {np.percentile(agg,99):8.4f} {agg.max():8.4f}")


if __name__ == "__main__":
    main()
