#!/usr/bin/env python3
"""Phase 0, step 3: secondary analysis over the cached per-example records.

No model is loaded. Two questions the headline numbers cannot answer on their
own:

1. Is any part of the apparent benefit of context refinement just an
   **abstention shift**? Shorter contexts give the model less to point at, so it
   says UNKNOWN more often -- which scores as correct on the 64 unanswerable
   rows and as wrong everywhere else.
2. Which **task families** break under selection, and does the pattern have a
   mechanism behind it?

    python scripts/phase0_analyse.py
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slcmi.provenance import read_jsonl, write_json  # noqa: E402
from slcmi.stats import mcnemar_exact, paired_bootstrap_delta, summarise  # noqa: E402

UNKNOWN_FORMS = {"unknown", "not determinable", "cannot be determined"}


def is_abstention(record: dict) -> bool:
    return record["normalized_prediction"].strip() in UNKNOWN_FORMS


def frame(records_by_condition: dict[str, dict[str, dict]], ids: list[str]) -> dict:
    present = [c for c in records_by_condition if all(i in records_by_condition[c] for i in ids)]
    scores = {c: [records_by_condition[c][i]["correct"] for i in ids] for c in present}
    out = {
        "n_examples": len(ids),
        "headline": {c: summarise(scores[c]) for c in present},
        "abstention_rate": {
            c: round(
                sum(is_abstention(records_by_condition[c][i]) for i in ids) / len(ids), 4
            )
            for c in present
        },
        "contrasts": {},
    }
    for label, (base, arm) in {
        "B_minus_A": ("A_full", "B_query"),
        "B_minus_C": ("C_random", "B_query"),
        "D_minus_A": ("A_full", "D_oracle"),
        "D_minus_B": ("B_query", "D_oracle"),
    }.items():
        if base in present and arm in present:
            out["contrasts"][label] = {
                **paired_bootstrap_delta(scores[base], scores[arm]),
                "mcnemar": mcnemar_exact(scores[base], scores[arm]),
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", default="demo", choices=("demo", "lcr_dev"))
    args = parser.parse_args()

    primary = args.suite == "demo"
    results = ROOT / ("results/phase0_eval" if primary else f"results/phase0_eval_{args.suite}")
    conditions_dir = ROOT / (
        "results/phase0_conditions" if primary else f"results/phase0_conditions_{args.suite}"
    )
    records = read_jsonl(results / "per_example.jsonl")

    by_condition: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in records:
        by_condition[r["condition"]][r["example_id"]] = r

    a = by_condition["A_full"]
    all_ids = sorted(a)
    answerable_ids = [i for i in all_ids if a[i]["answerable"]]
    unanswerable_ids = [i for i in all_ids if not a[i]["answerable"]]
    oracle_ids = sorted(by_condition.get("D_oracle", {}))
    oracle_answerable = [i for i in oracle_ids if a[i]["answerable"]]

    analysis = {
        "suite": args.suite,
        "question_1_abstention_shift": {
            "note": (
                "The 64 unanswerable rows are scored correct precisely when the "
                "model says UNKNOWN. If refinement raises the abstention rate, it "
                "buys those rows at the cost of every answerable row, which is a "
                "response-bias shift rather than a reasoning gain."
            ),
            "frames": {
                "all": frame(by_condition, all_ids),
                "answerable_only": frame(by_condition, answerable_ids),
                "unanswerable_only": frame(by_condition, unanswerable_ids),
                "oracle_eligible": frame(by_condition, oracle_ids),
                "oracle_eligible_answerable": frame(by_condition, oracle_answerable),
            },
        },
        "question_2_task_families": {},
    }

    families: dict[str, list[str]] = defaultdict(list)
    for i in all_ids:
        families[a[i]["task_type"]].append(i)

    for family, ids in sorted(families.items()):
        present = [c for c in by_condition if all(i in by_condition[c] for i in ids)]
        scores = {c: [by_condition[c][i]["correct"] for i in ids] for c in present}
        entry = {
            "n": len(ids),
            "accuracy_points": {
                c: summarise(scores[c])["accuracy_points"] for c in present
            },
        }
        if "A_full" in present and "B_query" in present:
            entry["B_minus_A"] = paired_bootstrap_delta(scores["A_full"], scores["B_query"])
        analysis["question_2_task_families"][family] = entry

    # Retention actually achieved, read back from the cached conditions.
    conditions = read_jsonl(conditions_dir / "conditions.jsonl")
    retention: dict[str, list[float]] = defaultdict(list)
    for row in conditions:
        retention[row["condition"]].append(row["retention_ratio"])
    analysis["retention_ratio_mean"] = {
        c: round(sum(v) / len(v), 4) for c, v in sorted(retention.items())
    }

    write_json(results / "analysis.json", analysis, overwrite=True)

    print("=== abstention rate by condition ===")
    for fname in ("all", "answerable_only", "unanswerable_only"):
        f = analysis["question_1_abstention_shift"]["frames"][fname]
        rates = " ".join(f"{c}={v:.3f}" for c, v in f["abstention_rate"].items())
        print(f"  {fname:20s} n={f['n_examples']:3d}  {rates}")

    print("\n=== accuracy by frame ===")
    for fname, f in analysis["question_1_abstention_shift"]["frames"].items():
        line = "  ".join(
            f"{c.split('_')[0]}={v['accuracy_points']:5.1f}%" for c, v in f["headline"].items()
        )
        print(f"  {fname:28s} n={f['n_examples']:3d}  {line}")

    print("\n=== B - A by frame ===")
    for fname, f in analysis["question_1_abstention_shift"]["frames"].items():
        c = f["contrasts"].get("B_minus_A")
        if c:
            print(f"  {fname:28s} {c['delta_points']:+6.2f} pts "
                  f"CI [{c['ci_low_points']:+.2f}, {c['ci_high_points']:+.2f}]")

    print(f"\nwrote {results / 'analysis.json'}")


if __name__ == "__main__":
    main()
