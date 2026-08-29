#!/usr/bin/env python3
"""Head-to-head exact McNemar matrix over prediction files on one split.

Each ``--system NAME=PATH`` is a prediction JSONL covering at least the split's
answer key. Correctness uses the published scorer's normalisation, so the counts
here match ``evaluate_predictions.py`` exactly.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gate_core as core  # noqa: E402


def parse_system(value: str) -> tuple[str, Path]:
    name, _, raw = value.partition("=")
    if not name or not raw:
        raise argparse.ArgumentTypeError(f"Expected NAME=PATH, got {value!r}")
    return name, Path(raw)


def correctness(path: Path, key_rows: list[dict[str, Any]]) -> dict[str, bool]:
    rows = {str(row["id"]): row for row in core.load_jsonl(path)}
    result: dict[str, bool] = {}
    for key in key_rows:
        row_id = str(key["id"])
        if row_id not in rows:
            raise ValueError(f"{path}: missing {row_id}")
        accepted = {
            core.SCORER.normalize(value)
            for value in key.get("accepted_answers", [key.get("answer", "")])
        }
        prediction = ""
        for field in core.SCORER.PREDICTION_FIELDS:
            if rows[row_id].get(field) is not None:
                prediction = str(rows[row_id][field])
                break
        result[row_id] = core.SCORER.normalize(prediction) in accepted
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--system", type=parse_system, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    key_rows = core.load_jsonl(args.key)
    ids = [str(row["id"]) for row in key_rows]
    systems = {name: correctness(path, key_rows) for name, path in args.system}

    totals = {name: sum(flags[i] for i in ids) for name, flags in systems.items()}
    pairs: list[dict[str, Any]] = []
    for left, right in combinations(systems, 2):
        b = sum(systems[right][i] and not systems[left][i] for i in ids)
        c = sum(systems[left][i] and not systems[right][i] for i in ids)
        pairs.append(
            {
                "baseline": left,
                "candidate": right,
                "candidate_only_correct_b": b,
                "baseline_only_correct_c": c,
                "delta": totals[right] - totals[left],
                "mcnemar_exact_p": round(core.mcnemar_exact_p(b, c), 6),
            }
        )

    payload = {
        "schema_version": "1.0",
        "kind": "system_comparison_matrix",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "key": str(args.key.resolve()),
        "rows": len(ids),
        "totals": totals,
        "pairs": pairs,
        "systems": {name: str(path.resolve()) for name, path in args.system},
    }
    core.write_json(args.output, payload)

    lines = [
        f"| comparison | correct | b | c | delta | exact McNemar p |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for entry in pairs:
        lines.append(
            f"| {entry['candidate']} vs {entry['baseline']} | "
            f"{totals[entry['candidate']]} vs {totals[entry['baseline']]} | "
            f"{entry['candidate_only_correct_b']} | {entry['baseline_only_correct_c']} | "
            f"{entry['delta']:+d} | {entry['mcnemar_exact_p']:.4f} |"
        )
    text = "\n".join(lines)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(text + "\n", encoding="utf-8", newline="\n")
    print(f"rows={len(ids)}  totals={totals}")
    print(text)


if __name__ == "__main__":
    main()

