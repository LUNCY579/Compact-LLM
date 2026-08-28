#!/usr/bin/env python3
"""Rank every confidence signal by how well it separates unanswerable rows.

Threshold-free: the ROC AUC of "row is unanswerable" against the signal, in its
abstention direction. AUC 0.5 means the signal carries no information about
answerability, whatever threshold is placed on it.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gate_core as core  # noqa: E402


def roc_auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Mann-Whitney U form, with ties counted as half."""

    pos = scores[positive]
    neg = scores[~positive]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=float)
    sorted_scores = scores[order]
    index = 0
    while index < sorted_scores.size:
        stop = index
        while stop + 1 < sorted_scores.size and sorted_scores[stop + 1] == sorted_scores[index]:
            stop += 1
        ranks[order[index : stop + 1]] = (index + stop) / 2.0 + 1.0
        index = stop + 1
    rank_sum = ranks[positive].sum()
    return float((rank_sum - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))


def diagnostics(rows: Sequence[core.GateRow]) -> list[dict[str, object]]:
    unanswerable = np.array([not row.answerable for row in rows], dtype=bool)
    output = []
    for signal in core.SIGNALS:
        values = np.array([row.signals[signal] for row in rows], dtype=float)
        direction = core.signal_direction(signal)
        # Orient so that "larger means more likely to deserve abstention".
        oriented = values if direction == "high" else -values
        auc = roc_auc(oriented, unanswerable)
        output.append(
            {
                "signal": signal,
                "direction": direction,
                "auc_unanswerable": round(auc, 4),
                "mean_answerable": round(float(values[~unanswerable].mean()), 4),
                "mean_unanswerable": round(float(values[unanswerable].mean()), 4),
            }
        )
    return sorted(output, key=lambda entry: -float(entry["auc_unanswerable"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confidence", type=Path, action="append", required=True)
    parser.add_argument("--key", type=Path, action="append", required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not len(args.confidence) == len(args.key) == len(args.label):
        parser.error("--confidence, --key and --label must be repeated equally")

    payload = {
        "schema_version": "1.0",
        "kind": "abstention_signal_diagnostics",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "splits": {},
    }
    for confidence, key, label in zip(args.confidence, args.key, args.label):
        rows = core.build_rows(core.load_jsonl(confidence), core.load_jsonl(key))
        payload["splits"][label] = {
            "rows": len(rows),
            "unanswerable": sum(not row.answerable for row in rows),
            "signals": diagnostics(rows),
        }

    core.write_json(args.output, payload)
    header = f"{'signal':32s}" + "".join(f"{label:>14s}" for label in args.label)
    print(header)
    order = [entry["signal"] for entry in payload["splits"][args.label[0]]["signals"]]
    for signal in order:
        line = f"{signal:32s}"
        for label in args.label:
            match = next(
                entry
                for entry in payload["splits"][label]["signals"]
                if entry["signal"] == signal
            )
            line += f"{match['auc_unanswerable']:14.3f}"
        print(line)


if __name__ == "__main__":
    main()

