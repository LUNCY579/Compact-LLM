"""Targeted same-layer write-out subspace check for delta C against T and H.

It tests the 40 o_proj/down_proj tensors in trained blocks 8-27. Collision is
C's energy in a donor's top-16 INPUT singular subspace divided by chance k/n.
Haar and depth-shifted donor subspaces are controls. This is structural only:
there are no activations or model forwards, and it changes no Stage A verdict.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from delta_geometry import BASE, SPARSE, TT, C, EXPECTED_SHA, TensorFile, sha256

HERE = Path(__file__).resolve().parent
OUT = HERE / "DELTA_C_SUBSPACE.json"
SEED = 20260926
RANK = 16


def right_subspace(matrix: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    m, n = matrix.shape
    k = min(RANK, m, n)
    a = matrix.astype(np.float32)
    omega = rng.standard_normal((n, k + 8), dtype=np.float32)
    y = a @ omega
    for _ in range(2):
        y = a @ (a.T @ y)
    q, _ = np.linalg.qr(y)
    small = q.T @ a
    _, _, vh = np.linalg.svd(small, full_matrices=False)
    return vh[:k].T.astype(np.float32)


def collision(a: np.ndarray, basis: np.ndarray) -> float:
    n = a.shape[1]
    k = basis.shape[1]
    energy = float(np.sum(a.astype(np.float64)**2))
    projected = float(np.sum((a @ basis).astype(np.float64)**2))
    return (projected / energy) / (k / n)


def projected_cosine(a: np.ndarray, b: np.ndarray, basis: np.ndarray) -> float | None:
    av = a @ basis
    bv = b @ basis
    dot = float(np.sum(av.astype(np.float64) * bv.astype(np.float64)))
    a2 = float(np.sum(av.astype(np.float64)**2))
    b2 = float(np.sum(bv.astype(np.float64)**2))
    return dot / math.sqrt(a2 * b2) if a2 > 0 and b2 > 0 else None


def stats(values: list[float]) -> dict:
    return {"n": len(values), "median": float(np.median(values)),
            "mean": float(np.mean(values)), "p90": float(np.percentile(values, 90))}


def main() -> int:
    if OUT.exists():
        raise SystemExit("DELTA_C_SUBSPACE.json exists (create-only)")
    paths = {"BASE": BASE, "SPARSE": SPARSE, "TT": TT, "C": C}
    for name, path in paths.items():
        if sha256(path) != EXPECTED_SHA[name]:
            raise SystemExit(f"{name} source hash mismatch")
    files = {name: TensorFile(path) for name, path in paths.items()}
    try:
        keys = sorted(k for k in files["C"].header
                      if k.endswith(".self_attn.o_proj.weight") or k.endswith(".mlp.down_proj.weight"))
        if len(keys) != 40:
            raise SystemExit(f"expected 40 write-out matrices, got {len(keys)}")
        rng = np.random.default_rng(SEED)
        donor_bases = {}
        # Pass 1: donor singular subspaces, retained only as small n_input x 16 bases.
        for key in keys:
            base = files["BASE"].tensor(key)
            sparse = files["SPARSE"].tensor(key)
            dt = (sparse - base).astype(np.float32)
            dh = (files["TT"].tensor(key) - sparse).astype(np.float32)
            donor_bases[key] = {"T": right_subspace(dt, rng), "H": right_subspace(dh, rng)}
            del base, sparse, dt, dh
        rows = []
        for kind in ("self_attn.o_proj", "mlp.down_proj"):
            group = sorted(k for k in keys if f".{kind}." in k)
            for index, key in enumerate(group):
                base = files["BASE"].tensor(key)
                sparse = files["SPARSE"].tensor(key)
                a = (files["C"].tensor(key) - base).astype(np.float32)
                dt = (sparse - base).astype(np.float32)
                dh = (files["TT"].tensor(key) - sparse).astype(np.float32)
                vt = donor_bases[key]["T"]
                vh = donor_bases[key]["H"]
                other = group[(index + 1) % len(group)]
                haar, _ = np.linalg.qr(rng.standard_normal((a.shape[1], RANK), dtype=np.float32))
                row = {"tensor": key, "kind": kind, "n_input": a.shape[1],
                       "C_in_T_top16": collision(a, vt),
                       "C_in_H_top16": collision(a, vh),
                       "C_in_T_other_block_top16": collision(a, donor_bases[other]["T"]),
                       "C_in_H_other_block_top16": collision(a, donor_bases[other]["H"]),
                       "C_in_Haar_top16": collision(a, haar),
                       "cos_C_T_in_T_top16": projected_cosine(a, dt, vt),
                       "cos_C_H_in_H_top16": projected_cosine(a, dh, vh)}
                rows.append(row)
                print(f"{len(rows):02d}/40 {key} C/T={row['C_in_T_top16']:.2f} C/H={row['C_in_H_top16']:.2f}", flush=True)
                del base, sparse, a, dt, dh
        fields = ("C_in_T_top16", "C_in_H_top16", "C_in_T_other_block_top16",
                  "C_in_H_other_block_top16", "C_in_Haar_top16")
        summary = {field: stats([r[field] for r in rows]) for field in fields}
        by_kind = {kind: {field: stats([r[field] for r in rows if r["kind"] == kind]) for field in fields}
                   for kind in ("self_attn.o_proj", "mlp.down_proj")}
        cos = {field: stats([abs(r[field]) for r in rows if r[field] is not None])
               for field in ("cos_C_T_in_T_top16", "cos_C_H_in_H_top16")}
        result = {"experiment": "TRISKILL-001", "analysis": "C vs T/H write-out input subspaces",
                  "source_sha256": EXPECTED_SHA, "seed": SEED, "rank": RANK,
                  "summary": summary, "by_kind": by_kind, "projected_abs_cosines": cos,
                  "per_tensor": rows,
                  "limits": "No activation covariance or forward pass. Subspace overlap is descriptive, not causal proof or a decision rule."}
        OUT.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps({"summary": summary, "projected_abs_cosines": cos}, indent=1))
    finally:
        for file in files.values():
            file.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
