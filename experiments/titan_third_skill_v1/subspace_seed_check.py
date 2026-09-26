"""Robustness check: recompute C-in-T top-16 collision at two new SVD seeds."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from delta_geometry import BASE, SPARSE, C, EXPECTED_SHA, TensorFile, sha256
from delta_subspace_c import right_subspace, collision

HERE = Path(__file__).resolve().parent
OUT = HERE / "DELTA_C_SUBSPACE_SEED_CHECK.json"
SEEDS = (17, 20260927)


def main() -> int:
    if OUT.exists():
        raise SystemExit("seed check exists (create-only)")
    original = json.loads((HERE / "DELTA_C_SUBSPACE.json").read_text(encoding="utf-8"))
    paths = {"BASE": BASE, "SPARSE": SPARSE, "C": C}
    for name, path in paths.items():
        if sha256(path) != EXPECTED_SHA[name]:
            raise SystemExit(f"{name} hash mismatch")
    files = {name: TensorFile(path) for name, path in paths.items()}
    try:
        keys = sorted(r["tensor"] for r in original["per_tensor"])
        if len(keys) != 40:
            raise SystemExit("unexpected matrix count")
        values = {seed: [] for seed in SEEDS}
        for key in keys:
            base = files["BASE"].tensor(key)
            dt = (files["SPARSE"].tensor(key) - base).astype(np.float32)
            dc = (files["C"].tensor(key) - base).astype(np.float32)
            for seed in SEEDS:
                # Derive a per-key seed so key traversal never changes the draw.
                key_seed = seed + int(key.split(".")[2]) * 31 + (1 if ".mlp." in key else 0)
                basis = right_subspace(dt, np.random.default_rng(key_seed))
                values[seed].append(collision(dc, basis))
        medians = {str(seed): float(np.median(values[seed])) for seed in SEEDS}
        result = {"experiment": "TRISKILL-001", "analysis": "top-16 collision seed robustness",
                  "original_seed": original["seed"],
                  "original_median": original["summary"]["C_in_T_top16"]["median"],
                  "alternate_medians": medians, "n_matrices": 40,
                  "source_sha256": {name: EXPECTED_SHA[name] for name in paths},
                  "interpretation": "descriptive randomized-SVD stability, not a functional test"}
        OUT.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps(result, indent=1))
    finally:
        for file in files.values():
            file.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
