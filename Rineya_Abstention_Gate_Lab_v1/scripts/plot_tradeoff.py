#!/usr/bin/env python3
"""Plot the abstention precision/recall tradeoff and the accuracy-vs-threshold curve.

The tuning panels are the ones the threshold was chosen from. The stability
panels are drawn after the gate was frozen and applied; they are a diagnostic,
not part of selection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gate_core as core  # noqa: E402
from sweep_thresholds import build_sweep, row_arrays, score_masks  # noqa: E402

HIGHLIGHT = "unknown_variant_logprob"
FAMILY_COLOURS = {
    "unknown": "#1f6feb",
    "first_token": "#d29922",
    "answer": "#8b949e",
}


def family(signal: str) -> str:
    if signal.startswith("unknown"):
        return "unknown"
    if signal.startswith("first_token"):
        return "first_token"
    return "answer"


def curves(rows: Sequence[core.GateRow]) -> dict[str, dict[str, np.ndarray]]:
    arrays = row_arrays(rows)
    result: dict[str, dict[str, np.ndarray]] = {}
    for signal in core.SIGNALS:
        sweep = build_sweep(signal, rows)
        scored = score_masks(sweep.masks, arrays)
        result[signal] = {
            "threshold": sweep.thresholds,
            "precision": scored["abstention_precision"],
            "recall": scored["abstention_recall"],
            "correct": scored["gated_correct"].astype(float),
        }
    return result


def load_rows(confidence: Path, key: Path) -> list[core.GateRow]:
    return core.build_rows(core.load_jsonl(confidence), core.load_jsonl(key))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tuning-confidence", type=Path, required=True)
    parser.add_argument("--tuning-key", type=Path, required=True)
    parser.add_argument("--holdout-confidence", type=Path, required=True)
    parser.add_argument("--holdout-key", type=Path, required=True)
    parser.add_argument("--gate-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    spec = core.GateSpec.from_dict(
        json.loads(args.gate_spec.read_text(encoding="utf-8"))["gate"]
    )
    tuning_rows = load_rows(args.tuning_confidence, args.tuning_key)
    holdout_rows = load_rows(args.holdout_confidence, args.holdout_key)
    tuning = curves(tuning_rows)
    holdout = curves(holdout_rows)

    tuning_base = core.evaluate_gate(core.IDENTITY_GATE, tuning_rows)
    holdout_base = core.evaluate_gate(core.IDENTITY_GATE, holdout_rows)
    tuning_gate = core.evaluate_gate(spec, tuning_rows)
    holdout_gate = core.evaluate_gate(spec, holdout_rows)

    figure, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))

    # Panel A: precision/recall frontier per signal, tuning split.
    ax = axes[0]
    for signal, data in tuning.items():
        order = np.argsort(data["recall"])
        ax.plot(
            data["recall"][order],
            data["precision"][order],
            color=FAMILY_COLOURS[family(signal)],
            alpha=0.95 if signal == HIGHLIGHT else 0.35,
            linewidth=2.4 if signal == HIGHLIGHT else 1.0,
            zorder=3 if signal == HIGHLIGHT else 1,
            label=signal if signal == HIGHLIGHT else None,
        )
    ax.scatter(
        [tuning_base.abstention_recall],
        [tuning_base.abstention_precision],
        marker="s",
        s=70,
        color="#000000",
        zorder=5,
        label="nimbus_low, ungated",
    )
    ax.scatter(
        [tuning_gate.abstention_recall],
        [tuning_gate.abstention_precision],
        marker="*",
        s=260,
        color="#cf222e",
        zorder=6,
        label="selected operating point",
    )
    ax.set_xlabel("abstention recall (unanswerable rows caught)")
    ax.set_ylabel("abstention precision")
    ax.set_title("A. tuning split (showcase64)\nprecision/recall over every threshold")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.axhline(0.25, color="#8b949e", linestyle=":", linewidth=1)
    ax.text(0.62, 0.265, "chance (25% unanswerable)", fontsize=8, color="#57606a")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=8)

    # Panel B: same frontier on the held-out split, gate already frozen.
    ax = axes[1]
    for signal, data in holdout.items():
        order = np.argsort(data["recall"])
        ax.plot(
            data["recall"][order],
            data["precision"][order],
            color=FAMILY_COLOURS[family(signal)],
            alpha=0.95 if signal == HIGHLIGHT else 0.35,
            linewidth=2.4 if signal == HIGHLIGHT else 1.0,
            zorder=3 if signal == HIGHLIGHT else 1,
            label=signal if signal == HIGHLIGHT else None,
        )
    ax.scatter(
        [holdout_base.abstention_recall],
        [holdout_base.abstention_precision],
        marker="s",
        s=70,
        color="#000000",
        zorder=5,
        label="nimbus_low, ungated",
    )
    ax.scatter(
        [holdout_gate.abstention_recall],
        [holdout_gate.abstention_precision],
        marker="*",
        s=260,
        color="#cf222e",
        zorder=6,
        label="frozen threshold, transferred",
    )
    ax.set_xlabel("abstention recall (unanswerable rows caught)")
    ax.set_ylabel("abstention precision")
    ax.set_title("B. held-out split (stability192)\nthreshold NOT re-tuned here")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.axhline(0.25, color="#8b949e", linestyle=":", linewidth=1)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=8)

    # Panel C: accuracy vs threshold for the selected signal on both splits.
    ax = axes[2]
    for label, data, rows_total, colour in (
        ("showcase64 (tuning)", tuning[spec.signal], len(tuning_rows), "#cf222e"),
        ("stability192 (held out)", holdout[spec.signal], len(holdout_rows), "#1f6feb"),
    ):
        ax.plot(
            data["threshold"],
            data["correct"] / rows_total,
            color=colour,
            linewidth=2.0,
            label=label,
        )
    ax.axvline(
        spec.threshold,
        color="#000000",
        linestyle="--",
        linewidth=1.4,
        label=f"frozen threshold {spec.threshold:.3f}",
    )
    ax.axhline(
        tuning_base.baseline_correct / len(tuning_rows),
        color="#cf222e",
        linestyle=":",
        linewidth=1.2,
    )
    ax.axhline(
        holdout_base.baseline_correct / len(holdout_rows),
        color="#1f6feb",
        linestyle=":",
        linewidth=1.2,
    )
    ax.set_xlim(-16.0, 2.0)
    ax.set_ylim(0.30, 0.90)
    ax.set_xlabel(f"{spec.signal} threshold (abstain at or above)")
    ax.set_ylabel("strict-exact accuracy")
    ax.set_title("C. accuracy vs threshold\ndotted = ungated nimbus_low")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)

    figure.suptitle(
        "Single-pass decode-time abstention gate on nimbus_low "
        "(Rineya long-context suite)",
        fontsize=13,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

