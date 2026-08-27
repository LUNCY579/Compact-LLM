#!/usr/bin/env python3
"""Build a coverage-safe scoreboard from Rineya Demo JSON reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

FIELDS = (
    "run",
    "hits",
    "strict_exact",
    "base_paraphrase_both_correct",
    "counterfactual_pair_both_correct_and_changed",
    "unanswerable_accuracy",
    "all_four_variants_correct",
    "short_response_rate_le_12_words",
    "covered_items",
    "key_items",
    "rank_eligible",
    "report",
)


def collect_reports(root: Path, expected_items: int = 64) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted(root.rglob("*_report.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        for run in report.get("runs", []):
            run_name = str(run["run"])
            if run_name in seen:
                raise ValueError(f"Duplicate run name {run_name!r} in {path}")
            seen.add(run_name)
            key_items = int(run.get("key_items", 0))
            covered_items = int(run.get("covered_items", 0))
            exact = float(run.get("strict_exact", 0.0))
            rank_eligible = key_items == expected_items and covered_items == expected_items
            packs = run.get("pack_metrics", {})
            rows.append(
                {
                    "run": run_name,
                    "hits": round(exact * key_items) if rank_eligible else None,
                    "strict_exact": exact,
                    "base_paraphrase_both_correct": float(
                        packs.get("base_paraphrase_both_correct", 0.0)
                    ),
                    "counterfactual_pair_both_correct_and_changed": float(
                        packs.get("counterfactual_pair_both_correct_and_changed", 0.0)
                    ),
                    "unanswerable_accuracy": float(
                        packs.get("unanswerable_accuracy", 0.0)
                    ),
                    "all_four_variants_correct": float(
                        packs.get("all_four_variants_correct", 0.0)
                    ),
                    "short_response_rate_le_12_words": float(
                        run.get("short_response_rate_le_12_words", 0.0)
                    ),
                    "covered_items": covered_items,
                    "key_items": key_items,
                    "rank_eligible": rank_eligible,
                    "report": str(path.resolve()),
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            not row["rank_eligible"],
            -(row["hits"] or -1),
            -row["all_four_variants_correct"],
            -row["base_paraphrase_both_correct"],
            -row["unanswerable_accuracy"],
            row["run"],
        ),
    )


def write_outputs(rows: list[dict[str, Any]], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "scoreboard.json").write_text(
        json.dumps({"schema_version": "1.0", "runs": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output / "scoreboard.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Rineya Showcase-64 Weight-Geometry Scoreboard",
        "",
        "Only complete 64/64 runs are rank-eligible.",
        "",
        "| Run | Hits | Exact | BP | CF | UNKNOWN | All four | Short | Coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        display = dict(row)
        display["hits_display"] = (
            str(row["hits"]) if row["rank_eligible"] else "—"
        )
        lines.append(
            "| {run} | {hits_display} | {strict_exact:.3f} | "
            "{base_paraphrase_both_correct:.3f} | "
            "{counterfactual_pair_both_correct_and_changed:.3f} | "
            "{unanswerable_accuracy:.3f} | {all_four_variants_correct:.3f} | "
            "{short_response_rate_le_12_words:.3f} | {covered_items}/{key_items} |".format(
                **display
            )
        )
    (output / "SCOREBOARD.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-items", type=int, default=64)
    args = parser.parse_args()
    rows = collect_reports(args.root.resolve(strict=True), args.expected_items)
    if not rows:
        raise SystemExit(f"No *_report.json files found under {args.root}")
    write_outputs(rows, args.output.resolve())
    print(f"Wrote {len(rows)} runs to {args.output.resolve()}")


if __name__ == "__main__":
    main()
