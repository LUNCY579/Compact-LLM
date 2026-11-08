"""Step 1 of Amendment 11/12: a powered test of H7 on early-position retrieval.

H7: short-horizon recurrent memory improves EARLY-position retrieval over the
parameter-matched static control.

Step 0 found C-short beating arm B 4/0 and 5/0 across two seeds with zero losses,
but on only 48 early items, which cannot reach p < 0.05. This scores the same
four adapters on **320 fresh early-position items** so the effect can be measured
rather than merely pointed at.

No training happens here. The adapters already exist and the training split is
disjoint from every eval suite by construction, so scoring them on new items is
exactly the relationship P2b's arms had to P2b's suite (Amendment 12).
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
from rineya_fwa.data_longctx import build_position_suite_v2, write_jsonl  # noqa: E402
from rineya_fwa.eval_longctx import mcnemar_exact, score_position_suite, wilson  # noqa: E402
from rineya_fwa.loading import load_supa, load_tokenizer  # noqa: E402
from rineya_fwa.paths import ARTIFACTS, DATA  # noqa: E402
from rineya_fwa.provenance import write_json  # noqa: E402
from rineya_fwa.train import load_adapter  # noqa: E402

SHORT_HORIZON = {"beta_raw_init": 0.0, "lambda_raw_init": -2.0}
LONG_HORIZON = {"beta_raw_init": -1.1, "lambda_raw_init": -5.9}
P2B = ARTIFACTS / "p2b_longctx"
OUT = ARTIFACTS / "h7_early"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", type=float, nargs="+", default=[0.02, 0.05, 0.10, 0.15])
    ap.add_argument("--lengths", type=int, nargs="+", default=[2048, 4096])
    ap.add_argument("--n-per-cell", type=int, default=40)
    ap.add_argument("--n-decoys", type=int, default=24)
    ap.add_argument("--hops", type=int, default=2)
    ap.add_argument("--eval-seed", type=int, default=808)
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 22])
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--rank", type=int, default=16)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    tok = load_tokenizer()

    items = build_position_suite_v2(
        tok, lengths=args.lengths, positions=tuple(args.positions),
        n_per_cell=args.n_per_cell, n_decoys=args.n_decoys, n_candidates=32,
        seed=args.eval_seed, split="eval_h7", key_mode="confusable", hops=args.hops,
    )
    write_jsonl(DATA / "h7_early" / "position_suite.jsonl", items)
    print(f"[data] {len(items)} items, positions={args.positions}, lengths={args.lengths}", flush=True)

    # arms to score: A (baseline for this new suite) + B/C-short at each seed
    plan: list[tuple[str, str, Path | None, dict[str, float] | None]] = [("A", "arm_A", None, None)]
    for s in args.seeds:
        plan.append(("B", f"arm_B/seed_{s}", P2B / f"arm_B/seed_{s}" / "fwa_adapter.pt", LONG_HORIZON))
        plan.append(("C", f"arm_Cshort/seed_{s}", P2B / f"arm_Cshort/seed_{s}" / "fwa_adapter.pt", SHORT_HORIZON))

    model = load_supa(device=device, eval_mode=True)
    abst = int(getattr(model.config, "abstain_token_id"))
    results: dict[str, Any] = {}

    for arm, tag, adapter, horizon in plan:
        if adapter is not None and not adapter.is_file():
            print(f"[eval] {tag}: adapter missing, skipping", flush=True)
            continue
        cached = OUT / tag / "eval.json"
        if cached.is_file():
            results[tag] = json.loads(cached.read_text(encoding="utf-8"))
            print(f"[eval] {tag} cached exact={results[tag]['position']['exact']:.4f}", flush=True)
            continue
        handle = apply_arm(model, arm, layers=[args.layer], rank=args.rank,
                           init_alpha=0.0, **(horizon or {}))
        try:
            if handle is not None and adapter is not None:
                load_adapter(handle, adapter)
            t0 = time.perf_counter()
            pos = score_position_suite(model, tok, items, abstain_token_id=abst)
            rec = {
                "arm": arm, "tag": tag, "horizon": horizon,
                "position": {k: v for k, v in pos.items() if k != "rows"},
                "position_rows": pos["rows"],
                "eval_seconds": time.perf_counter() - t0,
            }
            results[tag] = rec
            write_json(cached, rec, config=vars(args))
            k = int(round(pos["exact"] * len(items)))
            lo, hi = pos["exact_ci95"]
            print(f"[eval] {tag} {k}/{len(items)} = {pos['exact']:.4f} [{lo:.3f},{hi:.3f}] "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
        finally:
            if handle is not None:
                handle.uninstall()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ---- preregistered contrast: within seed, C-short vs B ----
    contrasts: dict[str, Any] = {}
    for s in args.seeds:
        b, c = results.get(f"arm_B/seed_{s}"), results.get(f"arm_Cshort/seed_{s}")
        if not (b and c):
            continue
        contrasts[f"Cshort_vs_B_seed{s}"] = mcnemar_exact(b["position_rows"], c["position_rows"])
        for p in args.positions:
            sub = lambda r: [x for x in r["position_rows"] if abs(x["position"] - p) < 1e-9]  # noqa: E731
            contrasts[f"Cshort_vs_B_seed{s}_pos{p:.2f}"] = mcnemar_exact(sub(b), sub(c))

    print("\n=== totals ===")
    for tag in sorted(results):
        r = results[tag]["position"]
        k = int(round(r["exact"] * len(items)))
        lo, hi = r["exact_ci95"]
        print(f"  {tag:22s} {k:3d}/{len(items)} = {r['exact']:.4f} [{lo:.3f},{hi:.3f}]")
        print(f"  {'':22s} by position: " +
              "  ".join(f"{kk}={vv['exact']:.3f}" for kk, vv in r["by_position"].items()))
    print("\n=== preregistered contrasts (C-short better / B better) ===")
    for k, v in contrasts.items():
        star = "  <-- p<0.05" if v["p_two_sided"] < 0.05 else ""
        print(f"  {k:36s} {v['b_better']:3d}/{v['a_better']:3d} disc={v['discordant']:3d} "
              f"p={v['p_two_sided']:.5f}{star}")

    verdict = all(
        contrasts.get(f"Cshort_vs_B_seed{s}", {}).get("p_two_sided", 1.0) < 0.05
        and contrasts[f"Cshort_vs_B_seed{s}"]["b_better"] > contrasts[f"Cshort_vs_B_seed{s}"]["a_better"]
        for s in args.seeds
        if f"Cshort_vs_B_seed{s}" in contrasts
    )
    print(f"\nH7 (needs p<0.05 at BOTH seeds): {'SUPPORTED' if verdict else 'NOT SUPPORTED'}")
    write_json(OUT / "summary.json",
               {"config": vars(args), "n_items": len(items),
                "results": {k: {kk: vv for kk, vv in v.items() if kk != "position_rows"}
                            for k, v in results.items()},
                "contrasts": contrasts, "h7_supported": verdict},
               config=vars(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
