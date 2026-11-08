"""Post-hoc interference analysis: the "blended" outcome the main scorer cannot see.

Brief section 14 asks whether older memories are *retained, overwritten, blended,
or incorrectly retrieved*. The main scorer (`score_interference`) reports only
retained / overwritten / wrong, because argmax over candidates collapses the
distribution and cannot distinguish

  * a confident wrong answer  (mass on one non-gold code), from
  * a blend               (mass smeared across several codes that appear in the
                           context, i.e. the memory has mixed them), from
  * a diffuse failure     (mass on codes that are not in the context at all,
                           i.e. no retrieval happened rather than interference).

Those are different failure modes and the brief asks us to tell them apart.
This pass re-scores the interference battery and keeps the full candidate
distribution, so it must run AFTER the main run rather than inside it.

Definitions used, stated up front:

  in-context codes   candidate codes that literally appear in this item's context
                     (the gold code plus its decoys, plus the superseded value
                     for the correction categories D and G)
  p_gold             softmax mass on the gold code, restricted to candidates
  p_in               mass on in-context codes other than gold
  p_out              mass on candidates that never appear in the context
  blend_ratio        p_in / (p_in + p_out) -- how much of the *error* mass lands
                     on things that were actually in the context

  label:
    retained     gold is the argmax
    overwritten  a correction item where the argmax is the superseded value
    blended      gold is not argmax, blend_ratio >= 0.5, and the error mass is
                 spread over an effective >= 1.5 codes (inverse Simpson) -- the
                 memory has mixed several things that were genuinely present
    misretrieved gold is not argmax and one in-context code dominates the error
    diffuse      gold is not argmax and the error mass is mostly out-of-context
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rineya_fwa.arms import apply_arm  # noqa: E402
from rineya_fwa.data_longctx import INTERFERENCE_CATEGORIES, build_interference_battery_v2  # noqa: E402
from rineya_fwa.eval_longctx import answer_logits  # noqa: E402
from rineya_fwa.loading import load_supa, load_tokenizer  # noqa: E402
from rineya_fwa.paths import ARTIFACTS  # noqa: E402
from rineya_fwa.provenance import write_json  # noqa: E402
from rineya_fwa.train import load_adapter  # noqa: E402

LONG_HORIZON = {"beta_raw_init": -1.1, "lambda_raw_init": -5.9}
SHORT_HORIZON = {"beta_raw_init": 0.0, "lambda_raw_init": -2.0}
CODE_RE = re.compile(r"\b([A-Z]{2})\b")


def classify(item: dict[str, Any], logits: torch.Tensor) -> dict[str, Any]:
    cand = item["candidate_token_ids"]
    codes = item["candidates"]
    gold = int(item["gold_token_id"])
    sup = item.get("superseded_token_id")
    context = item["messages"][1]["content"]
    present = set(CODE_RE.findall(context))

    sel = torch.tensor(cand, dtype=torch.long)
    probs = torch.softmax(logits[sel].float(), dim=-1)
    best_i = int(probs.argmax())
    best_id = int(sel[best_i])

    p_gold = 0.0
    p_in = 0.0
    p_out = 0.0
    per_code: list[tuple[str, float]] = []
    for i, (cid, code) in enumerate(zip(cand, codes)):
        p = float(probs[i])
        if int(cid) == gold:
            p_gold = p
            continue
        if code in present:
            p_in += p
        else:
            p_out += p
        per_code.append((code, p))

    err = p_in + p_out
    blend_ratio = (p_in / err) if err > 0 else 0.0
    top_err = max((p for _, p in per_code), default=0.0)
    top_err_share = (top_err / err) if err > 0 else 0.0

    # How many codes the error mass is really spread over (inverse Simpson).
    # A threshold on the top share is the wrong statistic: two near-equal codes
    # give a top share of ~0.5, so any cutoff at 0.5 classifies a clean two-way
    # blend as a single misretrieval depending on which side of the boundary it
    # lands. Effective count is scale-free -- 2 equal codes give 2.0, a 0.9/0.1
    # split gives 1.22, three equal codes give 3.0.
    if err > 0:
        shares = [p / err for _, p in per_code]
        effective_codes = 1.0 / sum(s * s for s in shares) if any(shares) else 0.0
    else:
        effective_codes = 0.0

    if best_id == gold:
        label = "retained"
    elif sup is not None and best_id == int(sup):
        label = "overwritten"
    elif blend_ratio >= 0.5 and effective_codes >= 1.5:
        label = "blended"
    elif blend_ratio >= 0.5:
        label = "misretrieved"
    else:
        label = "diffuse"

    return {
        "id": item["id"],
        "category": item["category"],
        "label": label,
        "p_gold": p_gold,
        "p_in_context_non_gold": p_in,
        "p_out_of_context": p_out,
        "blend_ratio": blend_ratio,
        "top_error_share": top_err_share,
        "effective_error_codes": effective_codes,
        "n_context_codes": len(present),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default=str(ARTIFACTS / "p2b_longctx"))
    ap.add_argument("--length", type=int, default=2048)
    ap.add_argument("--n-per-category", type=int, default=8)
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--out", type=str, default=str(ARTIFACTS / "diagnostics" / "interference_blend.json"))
    args = ap.parse_args()

    run = Path(args.run)
    tok = load_tokenizer()
    items = build_interference_battery_v2(
        tok, target_tokens=args.length, n_per_category=args.n_per_category, seed=202
    )
    model = load_supa(device="cuda" if torch.cuda.is_available() else "cpu", eval_mode=True)

    plan: list[tuple[str, str, Path | None, dict[str, float] | None]] = [("A", "arm_A", None, None)]
    for d in sorted(run.glob("arm_*/seed_*")):
        tag = f"{d.parent.name}/{d.name}"
        adapter = d / "fwa_adapter.pt"
        if not adapter.is_file():
            continue
        arm = "C" if d.parent.name == "arm_Cshort" else d.parent.name.replace("arm_", "")
        horizon = SHORT_HORIZON if d.parent.name == "arm_Cshort" else LONG_HORIZON
        plan.append((arm, tag, adapter, horizon))

    out: dict[str, Any] = {}
    for arm, tag, adapter, horizon in plan:
        handle = apply_arm(
            model, arm, layers=[args.layer], rank=args.rank, init_alpha=0.0, **(horizon or {})
        )
        try:
            if handle is not None and adapter is not None:
                load_adapter(handle, adapter)
            rows = []
            for item in items:
                logits, _ = answer_logits(model, tok, item["messages"])
                rows.append(classify(item, logits))
            by_cat: dict[str, dict[str, int]] = {}
            for c in INTERFERENCE_CATEGORIES:
                cell = [r for r in rows if r["category"] == c]
                counts: dict[str, int] = {}
                for r in cell:
                    counts[r["label"]] = counts.get(r["label"], 0) + 1
                by_cat[c] = counts
            out[tag] = {"rows": rows, "by_category": by_cat}
            tally: dict[str, int] = {}
            for r in rows:
                tally[r["label"]] = tally.get(r["label"], 0) + 1
            print(f"[blend] {tag:20s} {tally}", flush=True)
        finally:
            if handle is not None:
                handle.uninstall()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_json(Path(args.out), {"length": args.length, "arms": out}, config=vars(args))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
