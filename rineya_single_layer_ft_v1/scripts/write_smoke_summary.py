#!/usr/bin/env python3
"""Combine puzzle + ablation dual metrics into reports/layer8_smoke_summary.json."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
LAB_ROOT = SCRIPTS.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common import atomic_write_json, load_yaml, resolve_repo_path  # noqa: E402


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=LAB_ROOT / "configs" / "layer8_lora_puzzle_smoke.yaml",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    output_root = args.output_dir or resolve_repo_path(cfg["train"]["output_dir"])

    baseline_puzzle = load_json(output_root / "eval" / "baseline" / "report.json")
    lora_puzzle = load_json(output_root / "eval" / "layer8_lora" / "report.json")
    baseline_abl = load_json(output_root / "ablation" / "baseline" / "layer8_skip.json")
    lora_abl = load_json(output_root / "ablation" / "layer8_lora" / "layer8_skip.json")

    puzzle_delta = float(lora_puzzle["accuracy"]) - float(baseline_puzzle["accuracy"])
    cos_delta = float(lora_abl["mean_logit_cosine_vs_intact"]) - float(
        baseline_abl["mean_logit_cosine_vs_intact"]
    )
    score_delta = float(lora_abl["catastrophe_score"]) - float(
        baseline_abl["catastrophe_score"]
    )

    puzzle_ok = puzzle_delta >= -0.01  # allow tiny noise; no material regression
    ablation_activated = cos_delta < -0.02 or score_delta > 0.05
    activated = puzzle_ok and ablation_activated

    summary = {
        "schema_version": "layer8_smoke_v1",
        "experiment": cfg.get("experiment", {}),
        "puzzle": {
            "baseline_accuracy": baseline_puzzle["accuracy"],
            "layer8_lora_accuracy": lora_puzzle["accuracy"],
            "delta": round(puzzle_delta, 6),
            "baseline_correct_total": [
                baseline_puzzle["correct"],
                baseline_puzzle["total"],
            ],
            "layer8_lora_correct_total": [lora_puzzle["correct"], lora_puzzle["total"]],
        },
        "ablation_layer8_skip": {
            "baseline": {
                "catastrophe_score": baseline_abl["catastrophe_score"],
                "mean_logit_cosine_vs_intact": baseline_abl["mean_logit_cosine_vs_intact"],
                "bucket": baseline_abl["bucket"],
                "critical": baseline_abl["critical"],
            },
            "layer8_lora": {
                "catastrophe_score": lora_abl["catastrophe_score"],
                "mean_logit_cosine_vs_intact": lora_abl["mean_logit_cosine_vs_intact"],
                "bucket": lora_abl["bucket"],
                "critical": lora_abl["critical"],
            },
            "cosine_delta": round(cos_delta, 6),
            "catastrophe_score_delta": round(score_delta, 6),
        },
        "gates": {
            "puzzle_no_material_regression": puzzle_ok,
            "ablation_more_sensitive": ablation_activated,
            "activated_claim_eligible": activated,
        },
        "interpretation": (
            "activated"
            if activated
            else (
                "puzzle_ok_ablation_still_null"
                if puzzle_ok and not ablation_activated
                else (
                    "ablation_moved_but_puzzle_regressed"
                    if ablation_activated and not puzzle_ok
                    else "no_signal"
                )
            )
        ),
        "caveat": (
            "Inference leave-one-out sensitivity is not training importance. "
            "No all-layer matched-budget control in this smoke."
        ),
    }

    report_path = LAB_ROOT / "reports" / "layer8_smoke_summary.json"
    atomic_write_json(report_path, summary)
    atomic_write_json(output_root / "layer8_smoke_summary.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
