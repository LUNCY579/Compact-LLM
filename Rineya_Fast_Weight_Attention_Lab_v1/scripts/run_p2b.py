"""P2b: matched A/B/C/D on a non-ceilinged, token-exact long-context suite.

Differences from P2, all forced by the P2 post-mortem (reports/MECHANISM_DIAGNOSTIC.md):

* eval is the v2 suite -- token-exact 1K/2K/4K contexts, real CC0 filler,
  confusable decoy keys, 552-wide single-token answer space, n_per_cell items
  per (length, position) cell instead of one;
* scoring is candidate-argmax in one forward pass, so the malformed-output
  artifact that produced P2's only arm-A "miss" cannot recur;
* training uses a long-context TRAIN split disjoint from eval by construction,
  not the 10-colour synthetic corpus, and is not front-truncated;
* the plasticity/forgetting init is matched across arms B/C/D and set to a
  horizon that can actually span the evaluated context (P1/P2 ran a 87.5-token
  half-life against contexts of thousands of tokens);
* enough optimizer steps that the merge gate is not still ramping when training
  stops (in P2, alpha moved +0.0016 of a possible +0.0024 and was still rising).

Arm C-short repeats arm C with the ORIGINAL P1/P2 short-horizon init, so the
horizon change is itself measured rather than assumed.
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
from rineya_fwa.data_longctx import (  # noqa: E402
    build_interference_battery_v2,
    build_position_suite_v2,
    build_train_rows,
    write_jsonl,
)
from rineya_fwa.eval_longctx import (  # noqa: E402
    mcnemar_exact,
    score_interference,
    score_position_suite,
)
from rineya_fwa.loading import load_supa, load_tokenizer  # noqa: E402
from rineya_fwa.paths import ARTIFACTS, DATA  # noqa: E402
from rineya_fwa.provenance import write_json  # noqa: E402
from rineya_fwa.train import load_adapter, train_arm  # noqa: E402

OUT = ARTIFACTS / "p2b_longctx"

# Long-horizon init: lambda = softplus(-5.9) = 2.74e-3, beta = 2*sigmoid(-1.1) = 0.499.
# With ||x||^2 = 16 that is eta = 0.0312, ridge decay 0.99991/token (half-life
# ~8100 tokens) and a 50% write along the key direction instead of the
# deadbeat full replacement at beta = 1.0.
LONG_HORIZON = {"beta_raw_init": -1.1, "lambda_raw_init": -5.9}
SHORT_HORIZON = {"beta_raw_init": 0.0, "lambda_raw_init": -2.0}


def free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 22])
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--eval-lengths", type=int, nargs="+", default=[1024, 2048, 4096])
    ap.add_argument("--n-per-cell", type=int, default=8)
    ap.add_argument("--n-decoys", type=int, default=48)
    ap.add_argument("--interference-length", type=int, default=2048)
    ap.add_argument("--n-per-category", type=int, default=8)
    ap.add_argument("--train-rows", type=int, default=512)
    ap.add_argument("--train-lengths", type=int, nargs="+", default=[1024])
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--init-alpha", type=float, default=0.05)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-short-ablation", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    layers = [args.layer]
    OUT.mkdir(parents=True, exist_ok=True)

    tok = load_tokenizer()

    # ---------------- data ----------------
    print("[data] building long-context TRAIN split (disjoint from eval)", flush=True)
    train_rows = build_train_rows(
        tok, n_rows=args.train_rows, lengths=args.train_lengths, n_decoys=args.n_decoys, seed=909
    )
    write_jsonl(DATA / "longctx" / "train.jsonl", train_rows)
    print(f"[data] {len(train_rows)} train rows -> data/longctx/train.jsonl", flush=True)

    print("[data] building eval suites", flush=True)
    pos_items = build_position_suite_v2(
        tok, lengths=args.eval_lengths, n_per_cell=args.n_per_cell,
        n_decoys=args.n_decoys, n_candidates=32, seed=101, split="eval",
        key_mode="confusable",
    )
    int_items = build_interference_battery_v2(
        tok, target_tokens=args.interference_length, n_per_category=args.n_per_category, seed=202
    )
    write_jsonl(DATA / "longctx" / "position_suite.jsonl", pos_items)
    write_jsonl(DATA / "longctx" / "interference_suite.jsonl", int_items)
    print(f"[data] position n={len(pos_items)}  interference n={len(int_items)}", flush=True)

    # sanity: no train/eval key collision
    train_keys = {r["id"] for r in train_rows}
    eval_keys = {i["id"] for i in pos_items}
    assert not (train_keys & eval_keys), "train/eval id collision"

    # ---------------- training ----------------
    runs: list[dict[str, Any]] = []
    plan: list[tuple[str, int, dict[str, float], str]] = []
    for seed in args.seeds:
        for arm in ("B", "C", "D"):
            plan.append((arm, seed, LONG_HORIZON, f"arm_{arm}/seed_{seed}"))
    if not args.skip_short_ablation:
        plan.append(("C", args.seeds[0], SHORT_HORIZON, f"arm_Cshort/seed_{args.seeds[0]}"))

    if not args.skip_train:
        for arm, seed, horizon, tag in plan:
            out_dir = OUT / tag
            if (out_dir / "fwa_adapter.pt").is_file():
                print(f"[train] {tag} already done, skipping", flush=True)
                continue
            print(f"[train] {tag} horizon={horizon}", flush=True)
            t0 = time.perf_counter()
            man = train_arm(
                arm=arm,
                layers=layers,
                train_file=DATA / "longctx" / "train.jsonl",
                output_dir=out_dir,
                seed=seed,
                rank=args.rank,
                max_steps=args.max_steps,
                max_sequence_length=max(args.train_lengths) + 128,
                learning_rate=args.lr,
                grad_accum=args.grad_accum,
                init_alpha=args.init_alpha,
                train_rows=train_rows,
                **horizon,
            )
            man["tag"] = tag
            man["horizon"] = horizon
            runs.append(man)
            print(
                f"[train] {tag} steps={man['optimizer_steps']} first={man['first_loss']:.4f} "
                f"last={man['last_loss']:.4f} alpha={man['final_alpha']:.5f} "
                f"gpu_h={man['gpu_hours']:.4f} peak={man['peak_vram_bytes']/2**30:.2f}GiB "
                f"({time.perf_counter()-t0:.0f}s)",
                flush=True,
            )
            free_cuda()

    # ---------------- evaluation ----------------
    print("[eval] loading model once for all arms", flush=True)
    model = load_supa(device=device, eval_mode=True)
    abst = int(getattr(model.config, "abstain_token_id"))
    evals: list[dict[str, Any]] = []

    eval_plan: list[tuple[str, str, int | None, dict[str, float] | None, Path | None]] = [
        ("A", "arm_A", None, None, None)
    ]
    for arm, seed, horizon, tag in plan:
        eval_plan.append((arm, tag, seed, horizon, OUT / tag / "fwa_adapter.pt"))

    for arm, tag, seed, horizon, adapter in eval_plan:
        if adapter is not None and not adapter.is_file():
            print(f"[eval] {tag}: no adapter, skipping", flush=True)
            continue
        print(f"[eval] {tag}", flush=True)
        handle = apply_arm(
            model, arm, layers=layers, rank=args.rank, init_alpha=0.0,
            **(horizon or {}),
        )
        try:
            if handle is not None and adapter is not None:
                load_adapter(handle, adapter)
            t0 = time.perf_counter()
            pos = score_position_suite(model, tok, pos_items, abstain_token_id=abst)
            inter = score_interference(model, tok, int_items)
            rec = {
                "arm": arm,
                "tag": tag,
                "seed": seed,
                "horizon": horizon,
                "alpha": float(handle.blocks()[0].alpha.detach()) if handle else None,
                "position": {k: v for k, v in pos.items() if k != "rows"},
                "position_rows": pos["rows"],
                "interference": {k: v for k, v in inter.items() if k != "rows"},
                "interference_rows": inter["rows"],
                "eval_seconds": time.perf_counter() - t0,
            }
            evals.append(rec)
            write_json(OUT / tag / "eval.json", rec, config=vars(args))
            print(
                f"[eval] {tag} position={pos['exact']:.4f} "
                f"CI{tuple(round(x,3) for x in pos['exact_ci95'])} "
                f"mid-edge={pos['middle_edge_gap']:+.4f} "
                f"interference={inter['exact']:.4f} ({rec['eval_seconds']:.0f}s)",
                flush=True,
            )
        finally:
            if handle is not None:
                handle.uninstall()
            free_cuda()

    # ---------------- contrasts ----------------
    def rows_for(pred) -> list[dict[str, Any]] | None:
        hits = [e for e in evals if pred(e)]
        return hits[0]["position_rows"] if hits else None

    contrasts = {}
    a_rows = rows_for(lambda e: e["arm"] == "A")
    for seed in args.seeds:
        b = rows_for(lambda e: e["tag"] == f"arm_B/seed_{seed}")
        c = rows_for(lambda e: e["tag"] == f"arm_C/seed_{seed}")
        d = rows_for(lambda e: e["tag"] == f"arm_D/seed_{seed}")
        if a_rows and c:
            contrasts[f"C_vs_A_seed{seed}"] = mcnemar_exact(a_rows, c)
        if b and c:
            contrasts[f"C_vs_B_seed{seed}"] = mcnemar_exact(b, c)
        if c and d:
            contrasts[f"D_vs_C_seed{seed}"] = mcnemar_exact(c, d)
    cs = rows_for(lambda e: e["tag"].startswith("arm_Cshort"))
    c0 = rows_for(lambda e: e["tag"] == f"arm_C/seed_{args.seeds[0]}")
    if cs and c0:
        contrasts["Clong_vs_Cshort"] = mcnemar_exact(cs, c0)

    summary = {
        "config": vars(args),
        "training": runs,
        "eval": [
            {k: v for k, v in e.items() if k not in ("position_rows", "interference_rows")}
            for e in evals
        ],
        "contrasts": contrasts,
        "training_gpu_hours": sum(r.get("gpu_hours") or 0.0 for r in runs),
    }
    write_json(OUT / "summary.json", summary, config=vars(args))
    print(json.dumps(contrasts, indent=2))
    print(f"wrote {OUT / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
