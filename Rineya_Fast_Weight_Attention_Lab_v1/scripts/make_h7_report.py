"""Turn the H7 early-position run into a report and figure.

H7 (PREREGISTRATION Amendment 11/12): short-horizon recurrent memory improves
EARLY-position retrieval over the parameter-matched static control.

Criterion fixed before the run: C-short beats B with p < 0.05 at BOTH seeds on
the paired two-sided exact McNemar over all 320 items. The by-position and
by-length tables are descriptive breakdowns, not additional tests.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rineya_fwa.eval_longctx import mcnemar_exact, wilson  # noqa: E402
from rineya_fwa.paths import ARTIFACTS, REPORTS  # noqa: E402

RUN = ARTIFACTS / "h7_early"
POSITIONS = (0.02, 0.05, 0.10, 0.15)
LENGTHS = (2048, 4096)


def load(tag: str) -> dict[str, Any] | None:
    p = RUN / tag / "eval.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def sub_pos(rec: dict[str, Any], p: float) -> list[dict[str, Any]]:
    return [x for x in rec["position_rows"] if abs(x["position"] - p) < 1e-9]


def sub_len(rec: dict[str, Any], L: int) -> list[dict[str, Any]]:
    return [x for x in rec["position_rows"] if x["target_length"] == L]


def figure(arms: dict[str, dict[str, Any]], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for tag, rec in arms.items():
        ys, los, his = [], [], []
        for p in POSITIONS:
            cell = sub_pos(rec, p)
            k, n = sum(x["exact"] for x in cell), len(cell)
            ys.append(k / n)
            lo, hi = wilson(k, n)
            los.append(k / n - lo)
            his.append(hi - k / n)
        ax.errorbar(POSITIONS, ys, yerr=[los, his], marker="o", capsize=3, label=tag)
    ax.set_xlabel("relative position of the decisive evidence")
    ax.set_ylabel("exact match")
    ax.set_ylim(0, 1.02)
    ax.set_title("H7: early-position retrieval, 320 items (Wilson 95% CI)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 22])
    ap.add_argument("--out", type=str, default=str(REPORTS / "RESULTS_H7.md"))
    args = ap.parse_args()

    arms: dict[str, dict[str, Any]] = {}
    for tag in ["arm_A"] + [f"arm_{a}/seed_{s}" for s in args.seeds for a in ("B", "Cshort")]:
        r = load(tag)
        if r:
            arms[tag] = r
    if "arm_A" not in arms:
        print("no arm_A; has the run finished?")
        return 1
    n = len(arms["arm_A"]["position_rows"])

    tot = ["| Arm | exact [95% CI] |", "|---|---|"]
    for tag, rec in arms.items():
        k = sum(x["exact"] for x in rec["position_rows"])
        lo, hi = wilson(k, n)
        tot.append(f"| {tag.replace('arm_','').replace('/seed_',' s')} | {k}/{n} = {k/n:.4f} [{lo:.3f},{hi:.3f}] |")

    prim = ["| Seed | C-short better | B better | discordant | p |", "|---|---:|---:|---:|---:|"]
    verdict = []
    for s in args.seeds:
        b, c = arms.get(f"arm_B/seed_{s}"), arms.get(f"arm_Cshort/seed_{s}")
        if not (b and c):
            continue
        m = mcnemar_exact(b["position_rows"], c["position_rows"])
        star = " **sig**" if m["p_two_sided"] < 0.05 else ""
        prim.append(f"| {s} | {m['b_better']} | {m['a_better']} | {m['discordant']} | {m['p_two_sided']:.5f}{star} |")
        verdict.append(m["p_two_sided"] < 0.05 and m["b_better"] > m["a_better"])

    bypos = ["| Seed | " + " | ".join(f"{p:.2f}" for p in POSITIONS) + " |", "|---" * (len(POSITIONS) + 1) + "|"]
    for s in args.seeds:
        b, c = arms.get(f"arm_B/seed_{s}"), arms.get(f"arm_Cshort/seed_{s}")
        if not (b and c):
            continue
        cells = []
        for p in POSITIONS:
            m = mcnemar_exact(sub_pos(b, p), sub_pos(c, p))
            bk, ck, cn = sum(x["exact"] for x in sub_pos(b, p)), sum(x["exact"] for x in sub_pos(c, p)), len(sub_pos(b, p))
            cells.append(f"{bk}->{ck} /{cn} ({m['b_better']}/{m['a_better']})")
        bypos.append(f"| {s} | " + " | ".join(cells) + " |")

    bylen = ["| Seed | " + " | ".join(str(L) for L in LENGTHS) + " |", "|---" * (len(LENGTHS) + 1) + "|"]
    for s in args.seeds:
        b, c = arms.get(f"arm_B/seed_{s}"), arms.get(f"arm_Cshort/seed_{s}")
        if not (b and c):
            continue
        cells = []
        for L in LENGTHS:
            m = mcnemar_exact(sub_len(b, L), sub_len(c, L))
            bk, ck, cn = sum(x["exact"] for x in sub_len(b, L)), sum(x["exact"] for x in sub_len(c, L)), len(sub_len(b, L))
            cells.append(f"{bk}->{ck} /{cn} ({m['b_better']}/{m['a_better']}) p={m['p_two_sided']:.3f}")
        bylen.append(f"| {s} | " + " | ".join(cells) + " |")

    figure(arms, REPORTS / "figures" / "h7_early_positions.png")
    supported = bool(verdict) and all(verdict)

    body = f"""# Results — H7, early-position retrieval

Preregistered in `PREREGISTRATION.md` Amendments 11 and 12.
Classification: **EXPLORATORY**. See the caveat at the bottom — this is not a
confirmatory result and two seeds do not make it one.

**H7.** Short-horizon recurrent memory (arm C-short) improves early-position
retrieval over the parameter-matched static control (arm B).

Suite: {n} items, positions {list(POSITIONS)}, lengths {list(LENGTHS)}, two-hop
`vault -> unit -> code`, 24 confusable decoys, fresh eval seed 808. No arm was
retrained; the adapters come from P2b and the training split is disjoint from
every eval suite by construction.

## Totals

{chr(10).join(tot)}

## Primary contrast — paired two-sided exact McNemar, within seed, all {n} items

{chr(10).join(prim)}

**Criterion fixed before the run: p < 0.05 at BOTH seeds.**
**Verdict: H7 {"SUPPORTED" if supported else "NOT SUPPORTED"}.**

## By position (descriptive: B -> C-short of cell, with discordant counts)

{chr(10).join(bypos)}

![h7](figures/h7_early_positions.png)

## By context length (descriptive)

{chr(10).join(bylen)}

## What this does and does not establish

**Does**, if supported: on a synthetic two-hop retrieval task at layer 14, a
bounded rank-16 recurrent state with an 87.5-token retention horizon recovers
early-position items that a parameter-matched static pathway does not.

**Does not**
- Overturn the P2b NO-GO. That contrast (C vs B over all positions, long horizon)
  was null at both seeds and stands.
- Show any efficiency benefit. Peak VRAM is unchanged, prefill is 1.5-4.2x
  slower, and an auxiliary design cannot convert a bounded state into a memory
  saving because full attention still runs.
- Generalise past layer 14, past this synthetic task, or to the long horizon.

## Why this is exploratory even at two seeds

The early-position effect was **found by mining P2b's residuals**, and this suite
was then built to oversample exactly where the effect had been seen. That is a
sound way to generate a hypothesis and an unsound way to confirm one. The two
seeds share the same 320 items, so they are not independent replications
(Amendment 9) and their p-values are not combined.

A confirmatory result needs a fresh suite, a third seed, and the analysis fixed
before the data is seen.
"""
    Path(args.out).write_text(body, encoding="utf-8")
    print(body)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
