"""Step 0 of Amendment 11: does C-short's early-position edge replicate?

Trains and scores arm C-short at a second seed on the **identical P2b items**,
loaded from ``data/longctx/*.jsonl`` rather than regenerated. Regeneration is not
an option: Amendment 10 fixed a seeding bug, so the generator no longer produces
the items P2b scored. Those files are verified to be the scored items — their
id-only fingerprint reproduces the run's recorded ``dcdf2b6f1e7a5d06`` and an
exhaustive check of 1344 scored rows found 0 gold mismatches.

Preregistered stopping rule (Amendment 11): if C-short does not beat arm B at the
same seed *directionally* on positions {0.05, 0.10}, the signal is seed noise and
H7 is abandoned.
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
from rineya_fwa.eval_longctx import mcnemar_exact, score_position_suite  # noqa: E402
from rineya_fwa.loading import load_supa, load_tokenizer  # noqa: E402
from rineya_fwa.paths import ARTIFACTS, DATA  # noqa: E402
from rineya_fwa.provenance import write_json  # noqa: E402
from rineya_fwa.train import load_adapter, train_arm  # noqa: E402

SHORT_HORIZON = {"beta_raw_init": 0.0, "lambda_raw_init": -2.0}
EARLY = (0.05, 0.10)
OUT = ARTIFACTS / "p2b_longctx"          # same tree; this is a P2b arm
P2B = ARTIFACTS / "p2b_longctx"


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def subset(rows: list[dict[str, Any]], positions: tuple[float, ...]) -> list[dict[str, Any]]:
    return [r for r in rows if any(abs(r["position"] - p) < 1e-9 for p in positions)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=22)
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--init-alpha", type=float, default=0.05)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_tokenizer()

    pos_items = read_jsonl(DATA / "longctx" / "position_suite.jsonl")
    train_rows = read_jsonl(DATA / "longctx" / "train.jsonl")
    print(f"[data] loaded {len(pos_items)} position items and {len(train_rows)} train rows from disk", flush=True)

    tag = f"arm_Cshort/seed_{args.seed}"
    out_dir = OUT / tag
    if not (out_dir / "fwa_adapter.pt").is_file():
        print(f"[train] {tag} layer={args.layer} horizon={SHORT_HORIZON}", flush=True)
        t0 = time.perf_counter()
        man = train_arm(
            arm="C", layers=[args.layer], train_file=None, output_dir=out_dir,
            seed=args.seed, rank=args.rank, max_steps=args.max_steps,
            max_sequence_length=1152, learning_rate=args.lr, grad_accum=args.grad_accum,
            init_alpha=args.init_alpha, train_rows=train_rows, **SHORT_HORIZON,
        )
        print(f"[train] {tag} steps={man['optimizer_steps']} alpha={man['final_alpha']:.5f} "
              f"gpu_h={man['gpu_hours']:.3f} ({time.perf_counter()-t0:.0f}s)", flush=True)
    else:
        print(f"[train] {tag} adapter already present", flush=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = load_supa(device=device, eval_mode=True)
    abst = int(getattr(model.config, "abstain_token_id"))
    handle = apply_arm(model, "C", layers=[args.layer], rank=args.rank,
                       init_alpha=0.0, **SHORT_HORIZON)
    try:
        load_adapter(handle, out_dir / "fwa_adapter.pt")
        t0 = time.perf_counter()
        pos = score_position_suite(model, tok, pos_items, abstain_token_id=abst)
        rec = {
            "arm": "C", "tag": tag, "seed": args.seed, "layers": [args.layer],
            "horizon": SHORT_HORIZON,
            "alpha": float(handle.blocks()[0].alpha.detach()),
            "position": {k: v for k, v in pos.items() if k != "rows"},
            "position_rows": pos["rows"],
            "suite_fingerprint": "dcdf2b6f1e7a5d06",
            "items_source": "data/longctx/position_suite.jsonl (verified identical to P2b)",
            "eval_seconds": time.perf_counter() - t0,
        }
        write_json(out_dir / "eval.json", rec, config=vars(args))
    finally:
        handle.uninstall()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- preregistered contrast -------------------------------------------
    B = json.loads((P2B / f"arm_B/seed_{args.seed}" / "eval.json").read_text(encoding="utf-8"))
    CS11 = json.loads((P2B / "arm_Cshort/seed_11" / "eval.json").read_text(encoding="utf-8"))
    B11 = json.loads((P2B / "arm_B/seed_11" / "eval.json").read_text(encoding="utf-8"))

    out: dict[str, Any] = {"seed": args.seed}
    for name, (x, y) in {
        f"Cshort_vs_B_seed{args.seed}_ALL": (B, rec),
        "Cshort_vs_B_seed11_ALL": (B11, CS11),
    }.items():
        out[name] = mcnemar_exact(x["position_rows"], y["position_rows"])
    for name, (x, y) in {
        f"Cshort_vs_B_seed{args.seed}_EARLY": (B, rec),
        "Cshort_vs_B_seed11_EARLY": (B11, CS11),
    }.items():
        out[name] = mcnemar_exact(subset(x["position_rows"], EARLY), subset(y["position_rows"], EARLY))

    tot = lambda r: int(round(r["position"]["exact"] * len(pos_items)))  # noqa: E731
    early_acc = lambda r: (  # noqa: E731
        sum(z["exact"] for z in subset(r["position_rows"], EARLY)),
        len(subset(r["position_rows"], EARLY)),
    )
    print("\n=== totals ===")
    for n, r in ((f"B  seed{args.seed}", B), (f"Cshort seed{args.seed}", rec),
                 ("B  seed11", B11), ("Cshort seed11", CS11)):
        k, m = early_acc(r)
        print(f"  {n:18s} all {tot(r):3d}/{len(pos_items)}   early(0.05,0.10) {k:2d}/{m}")
    print("\n=== preregistered contrasts (2nd arm better / 1st better) ===")
    for k, v in out.items():
        if k == "seed":
            continue
        print(f"  {k:34s} {v['b_better']:2d}/{v['a_better']:2d}  disc={v['discordant']:2d}  p={v['p_two_sided']:.4f}")

    e = out[f"Cshort_vs_B_seed{args.seed}_EARLY"]
    replicated = e["b_better"] > e["a_better"]
    out["h7_replicated_directionally"] = replicated
    print(f"\nSTOPPING RULE (Amendment 11): early-position C-short vs B at seed {args.seed} "
          f"= {e['b_better']}/{e['a_better']} -> "
          f"{'REPLICATES directionally; step 1 is justified' if replicated else 'DOES NOT replicate; H7 abandoned'}")
    write_json(ARTIFACTS / "diagnostics" / f"cshort_replication_seed{args.seed}.json", out, config=vars(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
