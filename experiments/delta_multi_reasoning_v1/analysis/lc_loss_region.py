"""What are the rows that BOTH rotated arms lose? The first non-featureless subset here.

    python analysis/lc_loss_region.py --bank <root>/Rineya_Long_Context_Demo_v1/data/all_prompts.jsonl \
        --evidence <compose-lane>/.../retention-design-v1 \
        --baseline l0=LONGCTX_lwf_l0_lc.json \
        --arm l025=LONGCTX_lwf_l025_lc.json --arm ord23=LONGCTX_lwf_ord23_lc.json \
        --out <records>/LC_LOSS_REGION.json

Zero model calls — every input already exists. **The 256-row suite is SPENT / report-only, so
nothing here may promote a claim and it must be labelled report-only wherever quoted.**

## Why this is worth doing before the fourth conservation point

**Every "which rows?" question in this programme has come back featureless** — per-item headroom
ceilinged at the noise floor, the alias rows the delta loses carried no distinguishing feature,
and `ALIAS_RESOLUTION_INVESTIGATION.md` section 3 found nothing separating lost from kept rows.
Three independent attempts, three nulls.

The peer session then found that the two rotated arms lose the **same rows**: overlap 10 against
1.69 expected under independence, from an anchor loss and a row reshuffle — two mechanisms with
nothing in common. **A shared region is not noise, and regions can have properties.**

This script asks what those properties are, against the five stratifications the suite already
carries (`task_type`, `needle_profile`, `difficulty`, `split`, `variant`) plus `length_bucket`.

## The denominator, again

Losses are scored over **the rows the baseline got right**, and every attribute rate is computed
against that same population — not against all 256. A subgroup where the baseline is weak cannot
show losses, and comparing a loss count to the whole bank reports the baseline's own
distribution. That mistake killed two hypotheses in this programme already.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ATTRS = ("task_type", "needle_profile", "difficulty", "split", "variant", "length_bucket")


def fisher(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact on [[a,b],[c,d]]."""
    def lg(n):
        return math.lgamma(n + 1)

    def p_of(x):
        return math.exp(lg(a + b) + lg(c + d) + lg(a + c) + lg(b + d)
                        - lg(a + b + c + d) - lg(x) - lg(a + b - x)
                        - lg(a + c - x) - lg(d - a + x))
    n = a + b + c + d
    lo, hi = max(0, a + c - (c + d)), min(a + b, a + c)
    obs = p_of(a)
    return min(1.0, sum(p_of(x) for x in range(lo, hi + 1) if p_of(x) <= obs * (1 + 1e-9)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--evidence", type=Path, required=True)
    ap.add_argument("--baseline", required=True, metavar="NAME=FILE")
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=FILE")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    rows = {r["id"]: r for r in
            (json.loads(l) for l in a.bank.read_text(encoding="utf-8").splitlines() if l.strip())}

    def load(spec):
        name, _, f = spec.partition("=")
        j = json.loads((a.evidence / f).read_text(encoding="utf-8"))
        return name, {k: bool(v) for k, v in j["per_row_correct"].items()}, j

    bname, bcor, bj = load(a.baseline)
    arms = [load(s) for s in a.arm]

    base_right = {i for i, ok in bcor.items() if ok}
    losses = {n: {i for i in base_right if not c[i]} for n, c, _ in arms}
    names = [n for n, _, _ in arms]

    shared = set.intersection(*losses.values())
    expected = 1.0
    for n in names:
        expected *= len(losses[n]) / len(base_right)
    expected *= len(base_right)

    # ---- what distinguishes the shared region from the rest of the baseline-correct set? ----
    attr_stats = {}
    for at in ATTRS:
        levels = sorted({str(rows[i].get(at)) for i in base_right if i in rows})
        per = {}
        for lv in levels:
            in_lv = {i for i in base_right if i in rows and str(rows[i].get(at)) == lv}
            k = len(shared & in_lv)
            rest = len(shared) - k
            per[lv] = {
                "baseline_correct_rows": len(in_lv),
                "shared_loss_rows": k,
                "rate": round(k / len(in_lv), 4) if in_lv else None,
                "fisher_vs_rest": round(
                    fisher(k, len(in_lv) - k, rest, len(base_right) - len(in_lv) - rest), 4)
                if in_lv else None,
            }
        attr_stats[at] = per

    # ---- the alias_resolution sub-task, which this suite carries in its own right ----------
    alias_ids = [i for i in rows if rows[i].get("task_type") == "alias_resolution"]
    alias = {"n": len(alias_ids), f"{bname}_correct": sum(1 for i in alias_ids if bcor.get(i))}
    for n, c, j in arms:
        alias[f"{n}_correct"] = sum(1 for i in alias_ids if c.get(i))
    alias["note"] = ("this suite carries its own alias_resolution stratum -- an INDEPENDENT "
                     "fourth bank for the task ALIAS-001 measures, at n = 32 and therefore "
                     "underpowered; reported, not leaned on")

    out = {
        "record": "LC_LOSS_REGION",
        "class": "REPORT-ONLY -- the 256-row suite is SPENT; nothing here may promote a claim",
        "zero_model_calls": True,
        "baseline": bname,
        "baseline_correct_rows": len(base_right),
        "arm_losses": {n: len(losses[n]) for n in names},
        "shared_loss_rows": sorted(shared),
        "shared_count": len(shared),
        "expected_overlap_under_independence": round(expected, 2),
        "enrichment_x": round(len(shared) / expected, 2) if expected else None,
        "attributes": attr_stats,
        "alias_resolution_substratum": alias,
        "denominator_rule": ("losses and every attribute rate are scored over the rows the "
                             "BASELINE got right, never over all 256"),
        "caveats": [
            "SPENT instrument: report-only, non-confirmatory, whatever it shows",
            "n = 10 shared rows; per-attribute cells are tiny and Fisher p values are "
            "uncorrected across six attributes -- treat as hypothesis-generating",
            "two arms is not a survey of mechanisms; a third rotated arm would test whether the "
            "region is shared generally or only by these two",
        ],
    }
    a.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print(f"baseline {bname}: {len(base_right)}/256 correct")
    for n in names:
        print(f"  {n:<8} loses {len(losses[n])} of those")
    print(f"\nSHARED loss rows: {len(shared)}  expected {expected:.2f}  "
          f"enrichment {len(shared)/expected:.2f}x")
    print(f"  {sorted(shared)}")
    for at in ATTRS:
        print(f"\n{at}")
        for lv, v in attr_stats[at].items():
            if v["baseline_correct_rows"] == 0:
                continue
            print(f"   {lv:<26}{v['shared_loss_rows']:>3}/{v['baseline_correct_rows']:<4}"
                  f" rate {v['rate']:.3f}  fisher {v['fisher_vs_rest']:.3f}")
    print(f"\nalias_resolution sub-stratum (n={alias['n']}): "
          + "  ".join(f"{k}={v}" for k, v in alias.items()
                      if k.endswith('_correct')))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
