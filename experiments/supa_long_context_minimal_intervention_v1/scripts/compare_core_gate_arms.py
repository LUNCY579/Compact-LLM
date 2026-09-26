#!/usr/bin/env python3
"""Amendment 01 analysis: does Phase 0's conclusion depend on the checkpoint or
on the abstention gate?

Reads the cached per-example records from every (core, gate) cell that has been
run. No model is loaded.

The question that motivated this: Phase 0's headline mechanism is an abstention
shift on answerable rows (1.0% -> 10.9% -> 26.0% across full / query / random).
The gate is a constant +5.15625 bias on the UNKNOWN token. If the gate *causes*
that shift, removing it should pull B - A toward zero.

    python scripts/compare_core_gate_arms.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slcmi.provenance import read_jsonl, write_json  # noqa: E402
from slcmi.stats import mcnemar_exact, paired_bootstrap_delta, summarise  # noqa: E402

UNKNOWN_FORMS = {"unknown", "not determinable", "cannot be determined"}

CELLS = [
    ("a11", "gated", "phase0_eval", "Supa-A1.1 fp16, gate ON (primary)"),
    ("a11", "ungated", "phase0_eval_a11_ungated", "Supa-A1.1 fp16, gate OFF"),
    ("a1", "gated", "phase0_eval_a1_gated", "Supa-A1 bf16, gate ON"),
    ("a1", "ungated", "phase0_eval_a1_ungated", "Supa-A1 bf16, gate OFF"),
]


def is_abstention(record: dict) -> bool:
    return record["normalized_prediction"].strip() in UNKNOWN_FORMS


def load_cell(stage: str) -> dict[str, dict[str, dict]] | None:
    path = ROOT / "results" / stage / "per_example.jsonl"
    if not path.is_file():
        return None
    by_condition: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in read_jsonl(path):
        by_condition[r["condition"]][r["example_id"]] = r
    return by_condition


def cell_report(by_condition: dict[str, dict[str, dict]]) -> dict:
    a = by_condition["A_full"]
    all_ids = sorted(a)
    answerable = [i for i in all_ids if a[i]["answerable"]]
    unanswerable = [i for i in all_ids if not a[i]["answerable"]]
    oracle_ids = sorted(by_condition.get("D_oracle", {}))

    def frame(ids: list[str]) -> dict:
        present = [c for c in by_condition if all(i in by_condition[c] for i in ids)]
        scores = {c: [by_condition[c][i]["correct"] for i in ids] for c in present}
        out = {
            "n": len(ids),
            "accuracy_points": {
                c: summarise(scores[c])["accuracy_points"] for c in sorted(present)
            },
            "abstention_rate": {
                c: round(sum(is_abstention(by_condition[c][i]) for i in ids) / len(ids), 4)
                for c in sorted(present)
            },
            "contrasts": {},
        }
        for label, (base, arm) in {
            "B_minus_A": ("A_full", "B_query"),
            "B_minus_C": ("C_random", "B_query"),
            "D_minus_A": ("A_full", "D_oracle"),
        }.items():
            if base in present and arm in present:
                out["contrasts"][label] = {
                    **paired_bootstrap_delta(scores[base], scores[arm]),
                    "mcnemar": mcnemar_exact(scores[base], scores[arm]),
                }
        return out

    return {
        "all": frame(all_ids),
        "answerable_only": frame(answerable),
        "unanswerable_only": frame(unanswerable),
        "oracle_eligible": frame(oracle_ids) if oracle_ids else None,
    }


def main() -> None:
    cells: dict[str, dict] = {}
    for core, gate, stage, label in CELLS:
        loaded = load_cell(stage)
        if loaded is None:
            print(f"  (skipped {label}: not run)")
            continue
        cells[f"{core}_{gate}"] = {
            "core": core,
            "gate": gate,
            "label": label,
            "stage": stage,
            "frames": cell_report(loaded),
        }

    analysis: dict = {
        "cells": cells,
        "note": (
            "All cells share the same 256 rows and the same cached A/B/C/D "
            "contexts, so they are not independent samples. Agreement is "
            "robustness, not confirmation."
        ),
    }

    # Interaction: does the gate change the sign or size of B - A?
    if "a11_gated" in cells and "a11_ungated" in cells:
        g = cells["a11_gated"]["frames"]["answerable_only"]["contrasts"]["B_minus_A"]
        u = cells["a11_ungated"]["frames"]["answerable_only"]["contrasts"]["B_minus_A"]
        analysis["gate_interaction_on_B_minus_A_answerable"] = {
            "gated_delta_points": g["delta_points"],
            "ungated_delta_points": u["delta_points"],
            "difference_points": round(u["delta_points"] - g["delta_points"], 3),
            "both_negative": bool(g["delta_points"] < 0 and u["delta_points"] < 0),
            "prediction": (
                "If the gate caused the abstention shift, the ungated B - A "
                "should move materially toward zero."
            ),
        }

    write_json(ROOT / "results/amendment_01_core_gate_comparison.json", analysis, overwrite=True)

    print("\n=== accuracy (all 256 rows) ===")
    print(f"  {'cell':34s} {'A':>7s} {'B':>7s} {'C':>7s} {'D':>7s}")
    for key, c in cells.items():
        f = c["frames"]["all"]["accuracy_points"]
        d = c["frames"]["oracle_eligible"]
        dv = d["accuracy_points"].get("D_oracle") if d else None
        print(f"  {c['label']:34s} {f.get('A_full', 0):6.1f}% {f.get('B_query', 0):6.1f}% "
              f"{f.get('C_random', 0):6.1f}% {(dv if dv is not None else float('nan')):6.1f}%")

    print("\n=== B - A, answerable rows only (the honest frame) ===")
    for key, c in cells.items():
        ct = c["frames"]["answerable_only"]["contrasts"].get("B_minus_A")
        if ct:
            print(f"  {c['label']:34s} {ct['delta_points']:+6.2f} pts  "
                  f"CI [{ct['ci_low_points']:+.2f}, {ct['ci_high_points']:+.2f}]")

    print("\n=== abstention rate on ANSWERABLE rows (saying UNKNOWN is always wrong here) ===")
    print(f"  {'cell':34s} {'A':>7s} {'B':>7s} {'C':>7s}")
    for key, c in cells.items():
        r = c["frames"]["answerable_only"]["abstention_rate"]
        print(f"  {c['label']:34s} {r.get('A_full', 0):6.1%} {r.get('B_query', 0):6.1%} "
              f"{r.get('C_random', 0):6.1%}")

    if "gate_interaction_on_B_minus_A_answerable" in analysis:
        i = analysis["gate_interaction_on_B_minus_A_answerable"]
        print(f"\n=== gate interaction on B - A (answerable) ===")
        print(f"  gated   {i['gated_delta_points']:+.2f} pts")
        print(f"  ungated {i['ungated_delta_points']:+.2f} pts")
        print(f"  difference {i['difference_points']:+.2f} pts")

    print(f"\nwrote {ROOT / 'results/amendment_01_core_gate_comparison.json'}")


if __name__ == "__main__":
    main()
