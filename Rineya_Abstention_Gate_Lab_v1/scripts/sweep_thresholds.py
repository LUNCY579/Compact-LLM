#!/usr/bin/env python3
"""Sweep decode-time abstention thresholds on the tuning split.

Selection is confined to the split named by ``--tuning-split-name``; the script
refuses to run against anything else so a held-out split cannot be used to pick
a threshold. The pre-registered rule is documented in
``Rineya_Abstention_Gate_Lab_v1/CONFIRMATORY_PROTOCOL.md`` and implemented in
``select_primary``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gate_core as core  # noqa: E402


@dataclass(frozen=True)
class Sweep:
    """Every achievable partition for one signal, as a boolean abstention mask."""

    signal: str
    thresholds: np.ndarray  # (T,)
    masks: np.ndarray  # (T, N) bool


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def row_arrays(rows: Sequence[core.GateRow]) -> dict[str, np.ndarray]:
    return {
        "baseline_correct": np.array([row.baseline_correct for row in rows], bool),
        "abstain_correct": np.array([row.abstain_correct for row in rows], bool),
        "answerable": np.array([row.answerable for row in rows], bool),
        "model_abstained": np.array([row.abstained_by_model for row in rows], bool),
    }


def build_sweep(signal: str, rows: Sequence[core.GateRow]) -> Sweep:
    values = np.array([row.signals[signal] for row in rows], dtype=np.float64)
    direction = core.signal_direction(signal)
    thresholds = np.array(
        core.candidate_thresholds(values.tolist(), direction), dtype=np.float64
    )
    if direction == "low":
        masks = values[None, :] <= thresholds[:, None]
    else:
        masks = values[None, :] >= thresholds[:, None]
    return Sweep(signal=signal, thresholds=thresholds, masks=masks)


def score_masks(masks: np.ndarray, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Vectorised gate metrics for a stack of abstention masks."""

    flat = masks.reshape(-1, masks.shape[-1])
    base = arrays["baseline_correct"][None, :]
    gated_ok = np.where(flat, arrays["abstain_correct"][None, :], base)
    final_abstain = flat | arrays["model_abstained"][None, :]
    unanswerable = ~arrays["answerable"]

    abstentions = final_abstain.sum(axis=1)
    true_positive = (final_abstain & unanswerable[None, :]).sum(axis=1)
    false_positive = (final_abstain & arrays["answerable"][None, :]).sum(axis=1)
    added = (flat & ~arrays["model_abstained"][None, :]).sum(axis=1)
    gained = (gated_ok & ~base).sum(axis=1)
    lost = (base & ~gated_ok).sum(axis=1)

    total_unanswerable = int(unanswerable.sum())
    precision = np.divide(
        true_positive,
        abstentions,
        out=np.zeros(abstentions.shape, dtype=float),
        where=abstentions > 0,
    )
    recall = (
        true_positive / total_unanswerable
        if total_unanswerable
        else np.zeros(true_positive.shape, dtype=float)
    )

    shape = masks.shape[:-1]
    return {
        "gated_correct": gated_ok.sum(axis=1).reshape(shape),
        "abstentions": abstentions.reshape(shape),
        "added_abstentions": added.reshape(shape),
        "abstention_true_positives": true_positive.reshape(shape),
        "abstention_false_positives": false_positive.reshape(shape),
        "abstention_precision": np.asarray(precision, float).reshape(shape),
        "abstention_recall": np.asarray(recall, float).reshape(shape),
        "gained": gained.reshape(shape),
        "lost": lost.reshape(shape),
    }


def single_signal_table(
    sweeps: Sequence[Sweep], arrays: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for sweep in sweeps:
        scored = score_masks(sweep.masks, arrays)
        for index, threshold in enumerate(sweep.thresholds):
            gained = int(scored["gained"][index])
            lost = int(scored["lost"][index])
            table.append(
                {
                    "combine": "single",
                    "signal": sweep.signal,
                    "direction": core.signal_direction(sweep.signal),
                    "threshold": float(threshold),
                    "gated_correct": int(scored["gated_correct"][index]),
                    "abstentions": int(scored["abstentions"][index]),
                    "added_abstentions": int(scored["added_abstentions"][index]),
                    "abstention_true_positives": int(
                        scored["abstention_true_positives"][index]
                    ),
                    "abstention_false_positives": int(
                        scored["abstention_false_positives"][index]
                    ),
                    "abstention_precision": float(scored["abstention_precision"][index]),
                    "abstention_recall": float(scored["abstention_recall"][index]),
                    "gained": gained,
                    "lost": lost,
                    "mcnemar_exact_p": core.mcnemar_exact_p(gained, lost),
                }
            )
    return table


def pair_best(
    sweeps: Sequence[Sweep], arrays: dict[str, np.ndarray]
) -> list[dict[str, Any]]:
    """Best AND/OR conjunction per signal pair. Exploratory only."""

    results: list[dict[str, Any]] = []
    for left, right in combinations(sweeps, 2):
        for combine in ("and", "or"):
            if combine == "and":
                grid = left.masks[:, None, :] & right.masks[None, :, :]
            else:
                grid = left.masks[:, None, :] | right.masks[None, :, :]
            scored = score_masks(grid, arrays)
            correct = scored["gated_correct"]
            best = int(correct.max())
            candidates = np.argwhere(correct == best)
            # deterministic tie-break: fewest added abstentions, then lowest index
            added = scored["added_abstentions"]
            order = sorted(
                (int(added[i, j]), int(i), int(j)) for i, j in candidates
            )
            _, i, j = order[0]
            gained = int(scored["gained"][i, j])
            lost = int(scored["lost"][i, j])
            results.append(
                {
                    "combine": combine,
                    "signal": left.signal,
                    "direction": core.signal_direction(left.signal),
                    "threshold": float(left.thresholds[i]),
                    "signal_b": right.signal,
                    "direction_b": core.signal_direction(right.signal),
                    "threshold_b": float(right.thresholds[j]),
                    "gated_correct": best,
                    "abstentions": int(scored["abstentions"][i, j]),
                    "added_abstentions": int(added[i, j]),
                    "abstention_precision": float(scored["abstention_precision"][i, j]),
                    "abstention_recall": float(scored["abstention_recall"][i, j]),
                    "gained": gained,
                    "lost": lost,
                    "mcnemar_exact_p": core.mcnemar_exact_p(gained, lost),
                }
            )
    return results


def select_primary(table: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pre-registered rule, applied to single-signal gates only.

    1. maximise tuning-split correct answers
    2. tie-break on the fewest added abstentions (least behavioural change)
    3. tie-break on the highest abstention precision
    4. tie-break on signal name, then threshold, for determinism
    """

    singles = [entry for entry in table if entry["combine"] == "single"]
    if not singles:
        raise ValueError("No single-signal gates to select from")
    return min(
        singles,
        key=lambda entry: (
            -entry["gated_correct"],
            entry["added_abstentions"],
            -entry["abstention_precision"],
            entry["signal"],
            entry["threshold"],
        ),
    )


def spec_from_entry(entry: dict[str, Any]) -> core.GateSpec:
    return core.GateSpec(
        signal=entry["signal"],
        threshold=entry["threshold"],
        signal_b=entry.get("signal_b"),
        threshold_b=entry.get("threshold_b"),
        combine=entry["combine"],
    )


def frontier(table: Sequence[dict[str, Any]], signal: str) -> list[dict[str, Any]]:
    entries = [entry for entry in table if entry["signal"] == signal]
    return sorted(entries, key=lambda entry: entry["threshold"])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confidence", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument(
        "--tuning-split-name",
        default="showcase",
        help="The only split allowed to influence threshold selection.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    key_rows = core.load_jsonl(args.key)
    splits = {str(row.get("split")) for row in key_rows}
    if splits != {args.tuning_split_name}:
        raise SystemExit(
            f"Refusing to tune on splits {sorted(splits)}; "
            f"only {args.tuning_split_name!r} may select a threshold"
        )

    confidence_rows = core.load_jsonl(args.confidence)
    rows = core.build_rows(confidence_rows, key_rows)
    arrays = row_arrays(rows)

    baseline = core.evaluate_gate(core.IDENTITY_GATE, rows)
    ceiling = core.oracle_ceiling(rows)

    sweeps = [build_sweep(signal, rows) for signal in core.SIGNALS]
    table = single_signal_table(sweeps, arrays)
    pairs = pair_best(sweeps, arrays)

    primary_entry = select_primary(table)
    primary_spec = spec_from_entry(primary_entry)
    check = core.evaluate_gate(primary_spec, rows)
    if check.gated_correct != primary_entry["gated_correct"]:
        raise RuntimeError(
            "Vectorised sweep disagrees with gate_core.evaluate_gate: "
            f"{primary_entry['gated_correct']} vs {check.gated_correct}"
        )

    best_pair = min(
        pairs,
        key=lambda entry: (
            -entry["gated_correct"],
            entry["added_abstentions"],
            entry["signal"],
            entry["signal_b"],
        ),
    )

    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    core.write_jsonl(out / f"{args.tag}_single_signal_sweep.jsonl", table)
    core.write_jsonl(out / f"{args.tag}_pair_signal_best.jsonl", pairs)
    core.write_json(
        out / f"{args.tag}_selected_gate.json",
        {
            "schema_version": "1.0",
            "kind": "abstention_gate_spec",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "tuning_split": args.tuning_split_name,
            "selection_rule": select_primary.__doc__,
            "gate": primary_spec.to_dict(),
            "tuning_outcome": check.to_dict(),
            "tuning_inputs": {
                "confidence": str(args.confidence.resolve()),
                "confidence_sha256": sha256_file(args.confidence),
                "key": str(args.key.resolve()),
                "key_sha256": sha256_file(args.key),
            },
        },
    )

    per_signal_best = {}
    for signal in core.SIGNALS:
        entries = [entry for entry in table if entry["signal"] == signal]
        per_signal_best[signal] = min(
            entries,
            key=lambda entry: (
                -entry["gated_correct"],
                entry["added_abstentions"],
                entry["threshold"],
            ),
        )

    summary = {
        "schema_version": "1.0",
        "kind": "abstention_threshold_sweep",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tuning_split": args.tuning_split_name,
        "rows": len(rows),
        "baseline": baseline.to_dict(),
        "oracle_abstention_ceiling": ceiling,
        "single_signal_gates_evaluated": len(table),
        "signal_pairs_evaluated": len(pairs),
        "per_signal_best": per_signal_best,
        "primary_selection": primary_entry,
        "best_pair_exploratory": best_pair,
        "selected_gate_frontier": frontier(table, primary_spec.signal),
    }
    core.write_json(out / f"{args.tag}_sweep_summary.json", summary)

    print(json.dumps({
        "rows": len(rows),
        "baseline_correct": baseline.baseline_correct,
        "oracle_ceiling": ceiling,
        "selected": primary_spec.to_dict()["rule"],
        "tuning_correct": check.gated_correct,
        "tuning_delta": check.delta,
        "tuning_mcnemar_p": round(check.mcnemar_p, 6),
        "best_pair_correct": best_pair["gated_correct"],
    }, indent=2))


if __name__ == "__main__":
    main()

