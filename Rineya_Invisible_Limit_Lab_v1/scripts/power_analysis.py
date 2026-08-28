#!/usr/bin/env python3
"""Operating characteristics of the pre-registered decision rule.

Two questions the pre-registration did not answer, both of which affect how the
Stage 8 verdict should be read:

1. **Power.** With TEST at 512 rows (128 per variant), how large must a real
   difference be before the rule can detect it? An underpowered rule that returns
   "H1 not established" tells you little.

2. **False-positive rate of the conjunction.** §5 condition 1 requires
   delta >= 0 on all four variants but attaches no significance test to it.
   Under the null, each variant is roughly a coin flip, so the conjunction alone
   has a false-positive rate far above 5%. Combined with condition 2 the joint
   rate is tighter, but it is not the clean alpha = 0.05 the document implies.
   This quantifies it by simulation rather than leaving it as a footnote.

Discordance rates are estimated from the observed DEV comparison, so the numbers
describe this benchmark rather than a textbook ideal.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

VARIANTS = ["base", "paraphrase", "counterfactual", "unanswerable"]


def mcnemar_exact_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n))


def mde(n_discordant: int, alpha: float = 0.05) -> int:
    """Smallest |b-c| reaching significance given a discordant total."""
    for d in range(0, n_discordant + 1):
        b = (n_discordant + d) // 2
        c = n_discordant - b
        if mcnemar_exact_p(b, c) < alpha:
            return b - c
    return n_discordant


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev-report", type=Path, required=True,
                    help="dev_headtohead.json from compare_models.py")
    ap.add_argument("--test-scale", type=float, default=2.0,
                    help="TEST rows / DEV rows (512/256 = 2)")
    ap.add_argument("--sims", type=int, default=200000)
    ap.add_argument("--report", type=Path)
    args = ap.parse_args()

    dev = json.loads(args.dev_report.read_text(encoding="utf-8"))
    rng = np.random.default_rng(17)

    print("=" * 76)
    print("1. POWER — minimum detectable effect on TEST")
    print("=" * 76)
    print("Discordance observed on DEV, scaled to TEST size:\n")
    print(f"  {'split':16s} {'n':>5s} {'discordant':>11s} {'MDE rows':>9s} {'MDE pp':>7s}")
    print("  " + "-" * 54)

    scaled: dict[str, int] = {}
    rows: list[dict] = []
    for name, d in [("OVERALL", dev["overall"])] + [(v, dev["per_variant"][v]) for v in VARIANTS]:
        disc = int(round((d["b_only_a"] + d["c_only_b"]) * args.test_scale))
        n = int(round(d["n"] * args.test_scale))
        m = mde(disc)
        scaled[name] = disc
        pp = 100.0 * m / n if n else 0.0
        print(f"  {name:16s} {n:>5d} {disc:>11d} {m:>9d} {pp:>6.1f}%")
        rows.append({"split": name, "n": n, "discordant": disc,
                     "mde_rows": m, "mde_pp": round(pp, 2)})

    print()
    print("  Read: a real difference smaller than MDE cannot be detected at")
    print("  alpha=0.05, so 'H1 not established' on a sub-MDE gap is uninformative.")

    print()
    print("=" * 76)
    print("2. FALSE-POSITIVE RATE of the §5 conjunction, under the null")
    print("=" * 76)
    print("Null model: the two systems are equally good, so each discordant pair")
    print("independently favours either side with probability 0.5.\n")

    disc_v = np.array([scaled[v] for v in VARIANTS])
    disc_all = scaled["OVERALL"]

    # Simulate: per-variant discordant pairs split binomially; overall is their sum
    # plus any discordance not attributed to a variant (should be zero).
    b_v = rng.binomial(disc_v[None, :].repeat(args.sims, 0), 0.5)
    delta_v = 2 * b_v - disc_v[None, :]           # b - c per variant
    cond1 = (delta_v >= 0).all(axis=1)

    b_all = b_v.sum(axis=1)
    c_all = disc_v.sum() - b_all
    delta_all = b_all - c_all
    # exact p for the overall test, vectorised via a lookup
    tot = int(disc_v.sum())
    ptab = np.array([mcnemar_exact_p(b, tot - b) for b in range(tot + 1)])
    p_all = ptab[b_all]
    cond2 = (delta_all > 0) & (p_all < 0.05)

    r1 = float(cond1.mean())
    r2 = float(cond2.mean())
    rj = float((cond1 & cond2).mean())

    print(f"  condition 1 alone (delta>=0 on all four)     {r1:6.2%}")
    print(f"  condition 2 alone (overall delta>0, p<0.05)  {r2:6.2%}")
    print(f"  BOTH (the actual verdict rule)               {rj:6.2%}")
    print()
    if rj <= 0.05:
        print(f"  The conjunction is conservative: {rj:.2%} <= 5%. The verdict rule")
        print("  does not inflate false positives; condition 2 dominates.")
    else:
        print(f"  WARNING: joint false-positive rate {rj:.2%} exceeds 5%.")
        print("  Condition 1 is a point-estimate screen with no error control and")
        print("  should not be read as a significance claim.")

    print()
    print("=" * 76)
    print("3. What a null result would and would not mean")
    print("=" * 76)
    print(f"  If the verdict is 'not established' with an overall gap under")
    print(f"  {mde(disc_all)} rows, the honest statement is that this design")
    print("  cannot resolve a difference that small -- not that none exists.")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({
            "power": rows,
            "false_positive_rate": {
                "condition1_only": r1, "condition2_only": r2, "joint": rj,
                "sims": args.sims,
                "null_model": "equal systems; each discordant pair a fair coin",
            },
        }, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
