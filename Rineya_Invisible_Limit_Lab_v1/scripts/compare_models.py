#!/usr/bin/env python3
"""Evaluate the pre-registered decision rule between two prediction sets.

Implements PREREGISTRATION.md §5 literally, so the Stage 8 verdict is a mechanical
consequence of the data rather than a judgement call made after seeing it:

  1. Rineya >= Qwen3-4B on EACH of base, paraphrase, counterfactual, unanswerable
  2. Rineya  > Qwen3-4B overall, paired exact McNemar p < 0.05
  3. configuration is compute-eligible (<= 6 forward passes) -- asserted by caller
  4. configuration frozen before TEST was scored -- asserted by caller

Ties satisfy condition 1; losses do not. Per-variant tests are Holm-corrected.
Correctness uses the suite's own scorer semantics, not a reimplementation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

VARIANTS = ["base", "paraphrase", "counterfactual", "unanswerable"]


def load_scorer(root: Path):
    spec = importlib.util.spec_from_file_location(
        "ev", root / "Rineya_Long_Context_Demo_v1/scripts/evaluate_predictions.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_jsonl(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            out[d["id"]] = d
    return out


def mcnemar_exact_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for i, (k, p) in enumerate(items):
        val = min(1.0, (m - i) * p)
        running = max(running, val)
        adjusted[k] = running
    return adjusted


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--key", type=Path, required=True)
    ap.add_argument("--a", type=Path, required=True, help="Rineya predictions")
    ap.add_argument("--b", type=Path, required=True, help="comparison predictions")
    ap.add_argument("--label-a", default="Rineya")
    ap.add_argument("--label-b", default="Qwen3-4B")
    ap.add_argument("--passes-a", type=float, default=1.0,
                    help="forward passes per item for A (compute-eligibility check)")
    ap.add_argument("--budget", type=float, default=6.0)
    ap.add_argument("--report", type=Path)
    ap.add_argument("--root", type=Path, default=Path("."))
    args = ap.parse_args()

    ev = load_scorer(args.root)
    key = load_jsonl(args.key)
    A, B = load_jsonl(args.a), load_jsonl(args.b)

    missing_a = [i for i in key if i not in A]
    missing_b = [i for i in key if i not in B]
    if missing_a or missing_b:
        print(f"VOID: coverage incomplete -- A missing {len(missing_a)}, B missing {len(missing_b)}")
        print("PREREGISTRATION §5 requires full coverage or the run is void.")
        return 2

    def correct(preds: dict[str, dict], rid: str) -> bool:
        k = key[rid]
        accepted = {ev.normalize(v) for v in k["accepted_answers"]}
        return ev.normalize(ev.prediction_text(preds[rid])) in accepted

    ids = sorted(key)
    okA = {i: correct(A, i) for i in ids}
    okB = {i: correct(B, i) for i in ids}

    def bootstrap_ci(subset: list[str], iters: int = 20000,
                     seed: int = 17) -> tuple[float, float]:
        """Percentile CI on the paired delta, resampling scenarios not rows.

        Rows are not independent -- the four variants of a scenario share a
        context -- so resampling rows would understate the interval. Scenarios
        are the independent unit here.
        """
        import random
        by_scenario: dict[str, list[str]] = {}
        for i in subset:
            by_scenario.setdefault(key[i]["scenario_id"], []).append(i)
        scenarios = sorted(by_scenario)
        if not scenarios:
            return (0.0, 0.0)
        rnd = random.Random(seed)
        deltas: list[float] = []
        for _ in range(iters):
            draw = [scenarios[rnd.randrange(len(scenarios))] for _ in scenarios]
            d = 0
            for s in draw:
                for i in by_scenario[s]:
                    d += int(okA[i]) - int(okB[i])
            deltas.append(d)
        deltas.sort()
        lo = deltas[int(0.025 * len(deltas))]
        hi = deltas[int(0.975 * len(deltas)) - 1]
        return (float(lo), float(hi))

    def stats(subset: list[str]) -> dict[str, Any]:
        a = sum(okA[i] for i in subset)
        b = sum(okB[i] for i in subset)
        bb = sum(1 for i in subset if okA[i] and not okB[i])
        cc = sum(1 for i in subset if okB[i] and not okA[i])
        lo, hi = bootstrap_ci(subset)
        return {"n": len(subset), "a": a, "b": b, "delta": a - b,
                "b_only_a": bb, "c_only_b": cc, "p": mcnemar_exact_p(bb, cc),
                "ci95_delta": [lo, hi]}

    overall = stats(ids)
    per: dict[str, dict[str, Any]] = {}
    for v in VARIANTS:
        per[v] = stats([i for i in ids if key[i]["variant"] == v])
    adj = holm({v: per[v]["p"] for v in VARIANTS})

    w = max(len(args.label_a), len(args.label_b), 8)
    print(f"{'split':16s} {args.label_a:>{w}s} {args.label_b:>{w}s} {'delta':>6s} "
          f"{'95% CI':>14s} {'b':>4s} {'c':>4s} {'p':>8s} {'p_holm':>8s}")
    print("-" * (16 + 2 * w + 50))
    o = overall
    ci = f"[{o['ci95_delta'][0]:+.0f},{o['ci95_delta'][1]:+.0f}]"
    print(f"{'OVERALL':16s} {o['a']:>{w}d} {o['b']:>{w}d} {o['delta']:>+6d} "
          f"{ci:>14s} {o['b_only_a']:>4d} {o['c_only_b']:>4d} {o['p']:>8.4f} {'-':>8s}")
    for v in VARIANTS:
        r = per[v]
        ci = f"[{r['ci95_delta'][0]:+.0f},{r['ci95_delta'][1]:+.0f}]"
        print(f"{v:16s} {r['a']:>{w}d} {r['b']:>{w}d} {r['delta']:>+6d} "
              f"{ci:>14s} {r['b_only_a']:>4d} {r['c_only_b']:>4d} {r['p']:>8.4f} {adj[v]:>8.4f}")

    # --- pre-registered conditions ---
    c1 = all(per[v]["delta"] >= 0 for v in VARIANTS)
    c2 = overall["delta"] > 0 and overall["p"] < 0.05
    c3 = args.passes_a <= args.budget
    losses = [v for v in VARIANTS if per[v]["delta"] < 0]

    print()
    print("pre-registered conditions (§5):")
    print(f"  1  uniform: >= on all four variants        {'PASS' if c1 else 'FAIL'}"
          + ("" if c1 else f"  (losses: {', '.join(losses)})"))
    print(f"  2  overall > 0 and p < 0.05                {'PASS' if c2 else 'FAIL'}"
          f"  (delta {overall['delta']:+d}, p={overall['p']:.4f})")
    print(f"  3  compute-eligible: {args.passes_a:g} <= {args.budget:g} passes        "
          f"{'PASS' if c3 else 'FAIL'}")
    print(f"  4  frozen before TEST scored               (asserted by operator)")
    print()
    verdict = "H1 ESTABLISHED" if (c1 and c2 and c3) else "H1 NOT ESTABLISHED"
    print(f"VERDICT: {verdict}")
    if not (c1 and c2 and c3):
        print("  Per §8 this leaves the invisible-limit hypothesis standing.")
        print("  It does NOT prove the hypothesis is a law.")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({
            "key": str(args.key), "a": str(args.a), "b": str(args.b),
            "label_a": args.label_a, "label_b": args.label_b,
            "overall": overall, "per_variant": per, "p_holm": adj,
            "passes_a": args.passes_a, "budget": args.budget,
            "conditions": {"uniform": c1, "overall_significant": c2,
                           "compute_eligible": c3},
            "verdict": verdict,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
