"""Is the alignment of dSparse and dOpallo different INSIDE the dominant subspace?

The global elementwise cosine is the interference expected under isotropic inputs. The real
interference is tr(dS^T dO Sigma). Without activations Sigma is unknown, but we can ask how the
alignment behaves when restricted to candidate high-variance subspaces:

    cos_eff(V) = <dS V, dO V> / (||dS V|| ||dO V||)

Candidate V (top-k right singular subspaces):
    dO      - the directions Opallo's update reads most strongly
    dS      - the directions Titan-Sparse's update reads most strongly
    W0      - the directions the BASE layer reads most strongly (crude Sigma proxy)
    random  - Haar null; must reproduce the global cosine

If cos_eff is ~the global cosine for every V, anisotropy is not a threat and elementwise cosine
was sufficient. If |cos_eff| is much larger inside dO/dS/W0 than at random, the near-zero global
cosine is sign cancellation across coordinates and the deltas really do contend.

Also reports the SIGNED per-matrix values so that cancellation ACROSS matrices is visible
separately from cancellation WITHIN a matrix.
"""
from __future__ import annotations

import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from subspace import ST, ROOTS, TITAN_TRAINED, OPALLO_TRAINED, ENC2HF, randomized_subspace  # noqa: E402

K = 32
SEED = 20260922
KINDS = ["self_attn.o_proj", "mlp.down_proj", "self_attn.v_proj", "mlp.gate_proj",
         "self_attn.k_proj", "mlp.up_proj", "self_attn.q_proj"]


def cos_of(A, B):
    a = A.astype(np.float64).ravel(); b = B.astype(np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def main():
    rng = np.random.default_rng(SEED)
    base = ST(Path(ROOTS["supa"]) / "model.safetensors")
    ot = ST(OPALLO_TRAINED)
    sp = ST(Path(ROOTS["titan_sparse"]) / "model.safetensors")
    omap = {f"model.layers.{k.split('.', 2)[1]}.{ENC2HF[k.split('.', 2)[2]]}": k for k in ot.keys()}

    rows = []
    for kind in KINDS:
        for b in range(8, 28):
            key = f"model.layers.{b}.{kind}.weight"
            m, n = sp.info(key)[1]
            W0 = base.rows(key, 0, m).astype(np.float32)
            dS = sp.rows(key, 0, m).astype(np.float32) - W0
            dO = ot.rows(omap[key], 0, m).astype(np.float32) - W0
            kk = min(K, m, n)
            _, VO, _ = randomized_subspace(dO, kk, rng)
            _, VS, _ = randomized_subspace(dS, kk, rng)
            _, VW, _ = randomized_subspace(W0, kk, rng)
            Q, _ = np.linalg.qr(rng.standard_normal((n, kk), dtype=np.float32))
            rec = {"tensor": key, "kind": kind, "block": b, "global_cos": cos_of(dS, dO)}
            for lab, V in (("in_dO", VO), ("in_dS", VS), ("in_W0", VW), ("in_random", Q)):
                rec[lab] = cos_of(dS @ V, dO @ V)
            rows.append(rec)
        print(f"  done {kind}", flush=True)
    for h in (base, ot, sp):
        h.close()

    def stats(f):
        v = np.array([r[f] for r in rows])
        return {"mean_signed": float(v.mean()), "median_signed": float(np.median(v)),
                "mean_abs": float(np.abs(v).mean()), "median_abs": float(np.median(np.abs(v))),
                "max_abs": float(np.abs(v).max()),
                "frac_above_0.05": float((np.abs(v) > 0.05).mean())}

    out = {"k": K, "seed": SEED, "n_matrices": len(rows),
           "overall": {f: stats(f) for f in ("global_cos", "in_random", "in_W0", "in_dS", "in_dO")},
           "by_kind": {}, "per_tensor": rows}
    for kind in KINDS:
        sub = [r for r in rows if r["kind"] == kind]
        out["by_kind"][kind] = {f: float(np.median([abs(r[f]) for r in sub]))
                                for f in ("global_cos", "in_random", "in_W0", "in_dS", "in_dO")}
    Path("EFFECTIVE_COSINE.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print("\n  median |cos| restricted to each subspace (140 matrices)")
    print("  subspace     median|cos|   mean|cos|    max|cos|   frac>0.05")
    for f in ("global_cos", "in_random", "in_W0", "in_dS", "in_dO"):
        s = out["overall"][f]
        print(f"  {f:<12} {s['median_abs']:10.4f} {s['mean_abs']:12.4f} {s['max_abs']:11.4f} {s['frac_above_0.05']:11.2f}")
    print("\n  by sublayer, median |cos| inside dO's top-k subspace:")
    for k in sorted(KINDS, key=lambda k: -out["by_kind"][k]["in_dO"]):
        c = out["by_kind"][k]
        print(f"   {k:<22} global={c['global_cos']:.4f}  random={c['in_random']:.4f}  in_dO={c['in_dO']:.4f}")


if __name__ == "__main__":
    main()
