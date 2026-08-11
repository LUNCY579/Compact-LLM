from __future__ import annotations

"""Evaluate every retained long-horizon checkpoint at a fixed capability dose."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase_9_student_distillation.grokking_utils_v231 import discover_adapters, read_json, write_csv, write_json


def run(args: list[str]) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def flatten(summary: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"step": summary["step"], "dose": summary["dose"], "adapter": summary.get("adapter")}
    for split_name, split in summary.get("splits", {}).items():
        for key in (
            "answerable_correctness",
            "hard_answerable_correctness",
            "abstention_correctness",
            "false_abstention_rate",
            "answer_coverage",
            "target_token_nll_mean",
            "target_token_margin_mean",
            "target_token_rank_mean",
            "first_target_margin_mean",
            "first_target_rank_mean",
            "first_target_gold_probability_mean",
            "first_target_entropy_mean",
        ):
            row[f"{split_name}_{key}"] = split.get(key)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-root", type=Path, required=True)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--train-probe", type=Path, required=True)
    ap.add_argument("--dev-probe", type=Path, required=True)
    ap.add_argument("--ood-probe", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--dose", type=float, default=1.0)
    ap.add_argument("--logit-limit", type=int, default=96)
    ap.add_argument("--limit-per-split", type=int)
    ap.add_argument("--checkpoint-stride", type=int, default=1)
    ap.add_argument("--require-cuda", action="store_true")
    ap.add_argument("--reuse-existing", action="store_true")
    a = ap.parse_args()
    checkpoints = discover_adapters(a.training_root)
    if len(checkpoints) < 2:
        raise RuntimeError(f"No retained checkpoints found under {a.training_root}; use --save-steps and --keep-checkpoints")
    kept: list[dict[str, Any]] = []
    nonzero = [item for item in checkpoints if int(item["step"]) > 0]
    for index, item in enumerate(checkpoints):
        if int(item["step"]) == 0 or item.get("kind") == "final" or index % max(1, a.checkpoint_stride) == 0:
            kept.append(item)
    # De-duplicate steps and prefer final adapter if present.
    by_step: dict[int, dict[str, Any]] = {}
    for item in kept:
        by_step[int(item["step"])] = item
    kept = [by_step[step] for step in sorted(by_step)]

    rows: list[dict[str, Any]] = []
    a.output_root.mkdir(parents=True, exist_ok=True)
    for item in kept:
        step = int(item["step"])
        dest = a.output_root / f"step_{step:06d}"
        summary_path = dest / "summary.json"
        if not (a.reuse_existing and summary_path.exists()):
            cmd = [
                sys.executable,
                "scripts/phase_9_student_distillation/evaluate_grokking_checkpoint_v231.py",
                "--base-model", a.base_model,
                "--dose", str(a.dose),
                "--step", str(step),
                "--train-probe", str(a.train_probe),
                "--dev-probe", str(a.dev_probe),
                "--ood-probe", str(a.ood_probe),
                "--output-dir", str(dest),
                "--logit-limit", str(a.logit_limit),
            ]
            if item.get("adapter"):
                cmd += ["--adapter", str(item["adapter"])]
            if a.limit_per_split:
                cmd += ["--limit-per-split", str(a.limit_per_split)]
            if a.require_cuda:
                cmd.append("--require-cuda")
            run(cmd)
        rows.append(flatten(read_json(summary_path)))

    payload = {
        "schema_version": "2.3.1",
        "training_root": str(a.training_root),
        "base_model": a.base_model,
        "dose": a.dose,
        "checkpoint_count": len(rows),
        "sealed_data_used": False,
        "rows": rows,
    }
    write_json(a.output_root / "training_time_curve.json", payload)
    write_csv(a.output_root / "training_time_curve.csv", rows)
    md = [
        "# v2.3.1 Training-Time Capability Curve",
        "",
        "| Step | Train ans | DEV ans | OOD ans | Train NLL | DEV NLL | OOD NLL | DEV margin | OOD margin |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md.append(
            f"| {row['step']} | {row.get('train_answerable_correctness')} | {row.get('dev_answerable_correctness')} | "
            f"{row.get('ood_answerable_correctness')} | {row.get('train_target_token_nll_mean')} | {row.get('dev_target_token_nll_mean')} | "
            f"{row.get('ood_target_token_nll_mean')} | {row.get('dev_first_target_margin_mean')} | {row.get('ood_first_target_margin_mean')} |"
        )
    (a.output_root / "TRAINING_TIME_CURVE.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
