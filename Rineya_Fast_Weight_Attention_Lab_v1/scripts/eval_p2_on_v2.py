"""Score the EXISTING P2 adapters on the new non-ceilinged suite.

This isolates two explanations for P2's "B = C = D on every item" that the P2
run itself confounded:

* **Ceiling only.** If the P2 adapters now separate on a suite that can
  discriminate, then the ceiling alone explained the identical predictions and
  the short retention horizon was not the binding constraint.
* **Horizon too.** If they still tie, the 87.5-token half-life is doing the work
  and P2b's matched long-horizon init is the right correction.

Eval-only: no training, one model load, arms installed and uninstalled in place.
Cheap enough to run on a contended GPU (1K/2K peak ~1.8 GiB).

Note the P2 adapters were trained on the 10-colour synthetic corpus, so this is
an out-of-distribution transfer measurement, not a fair test of arm C. It is
reported as such.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rineya_fwa.arms import apply_arm  # noqa: E402
from rineya_fwa.data_longctx import build_position_suite_v2  # noqa: E402
from rineya_fwa.eval_longctx import mcnemar_exact, score_position_suite  # noqa: E402
from rineya_fwa.loading import load_supa, load_tokenizer  # noqa: E402
from rineya_fwa.paths import ARTIFACTS  # noqa: E402
from rineya_fwa.provenance import write_json  # noqa: E402
from rineya_fwa.train import load_adapter  # noqa: E402

P2 = ARTIFACTS / "p2_abcd"
# P1/P2 trained under the short-horizon init; the adapter carries the trained
# values, but the module must be constructed with the same shape/semantics.
SHORT_HORIZON = {"beta_raw_init": 0.0, "lambda_raw_init": -2.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[1024, 2048])
    ap.add_argument("--n-per-cell", type=int, default=8)
    ap.add_argument("--n-decoys", type=int, default=24)
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 22])
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--out", type=str, default=str(ARTIFACTS / "diagnostics" / "p2_adapters_on_v2.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_tokenizer()
    items = build_position_suite_v2(
        tok, lengths=args.lengths, n_per_cell=args.n_per_cell, n_decoys=args.n_decoys,
        n_candidates=32, seed=101, split="eval", key_mode="confusable", hops=args.hops,
    )
    print(f"[data] {len(items)} items, lengths={args.lengths}, hops={args.hops}", flush=True)

    model = load_supa(device=device, eval_mode=True)
    abst = int(getattr(model.config, "abstain_token_id"))

    plan: list[tuple[str, str, Path | None]] = [("A", "arm_A", None)]
    for seed in args.seeds:
        for arm in ("B", "C", "D"):
            plan.append((arm, f"arm_{arm}/seed_{seed}", P2 / f"arm_{arm}" / f"seed_{seed}" / "fwa_adapter.pt"))

    results: dict[str, Any] = {}
    for arm, tag, adapter in plan:
        if adapter is not None and not adapter.is_file():
            print(f"[skip] {tag}: no adapter", flush=True)
            continue
        handle = apply_arm(
            model, arm, layers=[args.layer], rank=args.rank, init_alpha=0.0, **SHORT_HORIZON
        )
        try:
            if handle is not None and adapter is not None:
                load_adapter(handle, adapter)
            t0 = time.perf_counter()
            r = score_position_suite(model, tok, items, abstain_token_id=abst)
            results[tag] = {
                "arm": arm,
                "alpha": float(handle.blocks()[0].alpha.detach()) if handle else None,
                "summary": {k: v for k, v in r.items() if k != "rows"},
                "rows": r["rows"],
            }
            lo, hi = r["exact_ci95"]
            print(
                f"[eval] {tag:18s} exact={r['exact']:.4f} [{lo:.3f},{hi:.3f}] "
                f"mid-edge={r['middle_edge_gap']:+.4f} margin={r['mean_margin']:+.2f} "
                f"({time.perf_counter()-t0:.0f}s)",
                flush=True,
            )
        finally:
            if handle is not None:
                handle.uninstall()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    contrasts = {}
    if "arm_A" in results:
        for seed in args.seeds:
            for arm in ("B", "C", "D"):
                tag = f"arm_{arm}/seed_{seed}"
                if tag in results:
                    contrasts[f"{arm}_vs_A_seed{seed}"] = mcnemar_exact(
                        results["arm_A"]["rows"], results[tag]["rows"]
                    )
    for seed in args.seeds:
        b, c, d = (f"arm_{x}/seed_{seed}" for x in "BCD")
        if b in results and c in results:
            contrasts[f"C_vs_B_seed{seed}"] = mcnemar_exact(results[b]["rows"], results[c]["rows"])
        if c in results and d in results:
            contrasts[f"D_vs_C_seed{seed}"] = mcnemar_exact(results[c]["rows"], results[d]["rows"])

    payload = {
        "note": "P2 adapters (trained on the 10-colour synthetic corpus, short horizon) "
                "scored on the P2b suite. Out-of-distribution transfer, not a fair test of arm C.",
        "config": vars(args),
        "summaries": {k: v["summary"] for k, v in results.items()},
        "alphas": {k: v["alpha"] for k, v in results.items()},
        "contrasts": contrasts,
    }
    write_json(Path(args.out), payload, config=vars(args))
    print(json.dumps(contrasts, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
