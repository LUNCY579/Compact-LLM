"""Pick a suite difficulty where the frozen backbone is NOT at ceiling.

The P1/P2 suite put every arm at 14/14, so the preregistered C-B contrast was
untestable. Before running P2b we choose (length, n_decoys) such that arm A --
unmodified Supa-A1.1 -- lands inside a discriminative band, i.e. there is
headroom for a fast-memory pathway to win and room for it to lose.

This is calibration on ARM A ONLY. No FWA adapter is loaded and no arm is
compared here, so it cannot bias the A/B/C/D contrast.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rineya_fwa.data_longctx import build_position_suite_v2  # noqa: E402
from rineya_fwa.eval_longctx import score_position_suite  # noqa: E402
from rineya_fwa.loading import load_supa, load_tokenizer  # noqa: E402
from rineya_fwa.paths import ARTIFACTS  # noqa: E402
from rineya_fwa.provenance import write_json  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[1024, 2048])
    ap.add_argument("--decoys", type=int, nargs="+", default=[8, 24, 48])
    ap.add_argument("--n-per-cell", type=int, default=4)
    ap.add_argument("--key-mode", type=str, default="confusable")
    ap.add_argument("--hops", type=int, default=1)
    ap.add_argument("--out", type=str, default=str(ARTIFACTS / "diagnostics" / "calibration.json"))
    args = ap.parse_args()

    tok = load_tokenizer()
    model = load_supa(device="cuda" if torch.cuda.is_available() else "cpu", eval_mode=True)
    abst = int(getattr(model.config, "abstain_token_id"))

    grid = []
    for length in args.lengths:
        for n_dec in args.decoys:
            t0 = time.perf_counter()
            items = build_position_suite_v2(
                tok, lengths=[length], positions=(0.05, 0.50, 0.95),
                n_per_cell=args.n_per_cell, n_decoys=n_dec, n_candidates=32, seed=7, key_mode=args.key_mode, hops=args.hops,
            )
            build_s = time.perf_counter() - t0
            ctx = [i["n_context_tokens"] for i in items]
            t1 = time.perf_counter()
            r = score_position_suite(model, tok, items, abstain_token_id=abst)
            eval_s = time.perf_counter() - t1
            row = {
                "length": length,
                "n_decoys": n_dec, "key_mode": args.key_mode, "hops": args.hops,
                "ctx_tokens_min": min(ctx),
                "ctx_tokens_max": max(ctx),
                "prompt_tokens_max": max(x["n_prompt_tokens"] for x in r["rows"]),
                "exact": r["exact"],
                "n": r["n"],
                "exact_ci95": r["exact_ci95"],
                "unconstrained_exact": r["unconstrained_exact"],
                "abstain_argmax_rate": r["abstain_argmax_rate"],
                "mean_margin": r["mean_margin"],
                "by_position": {k: v["exact"] for k, v in r["by_position"].items()},
                "build_seconds": build_s,
                "eval_seconds": eval_s,
                "seconds_per_item": eval_s / max(r["n"], 1),
            }
            grid.append(row)
            print(
                f"len={length:5d} decoys={n_dec:3d} ctx={min(ctx)}-{max(ctx)} "
                f"prompt<={row['prompt_tokens_max']:5d} exact={r['exact']:.3f} "
                f"unconstrained={r['unconstrained_exact']:.3f} abstain={r['abstain_argmax_rate']:.3f} "
                f"margin={r['mean_margin']:+.2f} {row['seconds_per_item']:.2f}s/item",
                flush=True,
            )
            print(f"        by_position={ {k: round(v, 3) for k, v in row['by_position'].items()} }", flush=True)

    write_json(Path(args.out), {"grid": grid}, config=vars(args))
    print(json.dumps({"wrote": args.out}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
