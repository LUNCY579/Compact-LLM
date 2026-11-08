"""Why would a rank-4 truncation fail to rebuild byte-identically when rank-16 and rank-64 do?

RANK-001 Erratum 2 records exactly that asymmetry: r=16 and r=64 rebuilt byte-identical three
hours apart, r=4 did not, on identical code, host and provably identical hashed inputs. The
erratum states the cause is undiagnosed and declines to call it last-bit noise without
measuring it. This measures the mechanism it names as the most likely one.

THE HYPOTHESIS
--------------
`delta_r = U[:, :r] diag(S[:r]) V[:r, :]` is well defined only when the r-th and (r+1)-th
singular values are separated. When s_r ~= s_{r+1} the leading r-dimensional singular subspace
is not unique: any rotation within the tied block is an equally valid SVD, LAPACK's
divide-and-conquer driver may return a different basis on different runs, and the RECONSTRUCTION
changes even though the factorisation is correct both times.

So the prediction is specific and falsifiable: if this is the cause, the RELATIVE GAP
(s_r - s_{r+1}) / s_r should be systematically SMALLER at r=4 than at r=16 or r=64 across the
140 trained matrices. If instead r=4's gaps are comparable or larger, this mechanism is refuted
and the defect lies elsewhere in the builder.

Note what this does and does not settle. A tied subspace makes the BYTES non-unique; it does
not by itself make the SCORES differ, because the two reconstructions span the same energy.
This script therefore also reports the reconstruction sensitivity bound, so the erratum can say
how large a difference the ambiguity could produce rather than assuming it is negligible.

No model is run. Read-only. Writes one JSON receipt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SUPA = Path(r"C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Supa-1.7.0-devfreeze")
TRAINED = Path(r"C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/int001-20260917/trained_fp32.safetensors")
RANKS = (4, 16, 64)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    import torch
    from safetensors import safe_open

    torch.backends.cuda.matmul.allow_tf32 = False

    rows: list[dict] = []
    with safe_open(str(SUPA / "model.safetensors"), framework="pt") as fs, \
         safe_open(str(TRAINED), framework="pt") as ft:
        names = set(ft.keys())
        for name in fs.keys():
            if name not in names:
                continue
            src = fs.get_tensor(name)
            trn = ft.get_tensor(name)
            d = trn.to(torch.float32) - src.to(torch.float32)
            if d.dim() != 2:
                continue
            sv = torch.linalg.svdvals(d).to(torch.float64)
            rec: dict = {"name": name, "shape": list(d.shape)}
            tot = float(sv.pow(2).sum())
            for r in RANKS:
                if r >= sv.numel():
                    continue
                s_r, s_r1 = float(sv[r - 1]), float(sv[r])
                gap = (s_r - s_r1) / s_r if s_r > 0 else None
                rec[f"relgap_r{r}"] = gap
                # Energy sitting in the ambiguous tied block: how much of the reconstruction
                # could move if the basis inside a near-tied block came out differently.
                tied = [j for j in range(sv.numel())
                        if s_r > 0 and abs(float(sv[j]) - s_r) / s_r < 1e-6]
                rec[f"tied_block_size_r{r}"] = len(tied)
                rec[f"ambiguous_energy_share_r{r}"] = (
                    sum(float(sv[j]) ** 2 for j in tied) / tot if tot > 0 else None)
            rows.append(rec)

    def stats(key: str) -> dict:
        xs = sorted(r[key] for r in rows if r.get(key) is not None)
        if not xs:
            return {}
        n = len(xs)
        return {"n": n, "min": xs[0], "p10": xs[max(0, n // 10)],
                "p50": xs[n // 2], "max": xs[-1],
                "n_below_1e-3": sum(1 for x in xs if x < 1e-3),
                "n_below_1e-2": sum(1 for x in xs if x < 1e-2)}

    gap_stats = {f"r{r}": stats(f"relgap_r{r}") for r in RANKS}
    amb_stats = {f"r{r}": stats(f"ambiguous_energy_share_r{r}") for r in RANKS}

    p50 = {r: gap_stats[f"r{r}"].get("p50") for r in RANKS}
    ordered = (p50[4] is not None and p50[16] is not None and p50[64] is not None
               and p50[4] < p50[16] and p50[4] < p50[64])
    verdict = ("CONSISTENT_WITH_TIED_SUBSPACE_AT_R4" if ordered
               else "NOT_EXPLAINED_BY_SPECTRAL_DEGENERACY")

    receipt = {
        "schema": 1, "record": "DMRF_SPECTRAL_GAP_DIAGNOSTIC",
        "experiment": "DMRF-001",
        "addresses": "RANK-001 Erratum 2 (r=4 arm does not rebuild to its recorded hash)",
        "what_this_is": (
            "Relative singular-value gaps at the three registered truncation ranks, over all "
            "2D trained matrices. Tests whether r=4 sits in a less well separated part of the "
            "spectrum than r=16 or r=64, which would make its leading subspace non-unique and "
            "its reconstruction build-order dependent."),
        "hypothesis": "relgap at r=4 is systematically smaller than at r=16 and r=64",
        "relative_gap_definition": "(s_r - s_{r+1}) / s_r",
        "relative_gap_stats": gap_stats,
        "ambiguous_energy_stats": amb_stats,
        "median_relgap_by_rank": p50,
        "verdict": verdict,
        "what_this_does_not_settle": (
            "A non-unique basis changes the BYTES but spans the same energy, so it does not by "
            "itself imply the SCORES differ. The ambiguous energy share bounds how much of the "
            "reconstruction could move; it does not measure what did move, because the "
            "original r=4 bytes were reclaimed and cannot be diffed."),
        "model_calls": 0, "optimizer_steps": 0,
        "per_tensor": rows,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"median_relgap_by_rank": p50, "verdict": verdict,
                      "relative_gap_stats": gap_stats}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
