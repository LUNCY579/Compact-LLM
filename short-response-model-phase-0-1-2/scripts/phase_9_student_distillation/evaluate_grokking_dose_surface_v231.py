from __future__ import annotations

"""Adaptive checkpoint × capability-dose surface for v2.3.1."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.phase_9_student_distillation.grokking_utils_v231 import (
    discover_adapters,
    nearest_steps,
    read_json,
    write_csv,
    write_json,
)


def run(args: list[str]) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def biggest_ood_jumps(rows: list[dict[str, Any]], count: int) -> list[int]:
    ranked: list[tuple[float, int]] = []
    rows = sorted(rows, key=lambda row: int(row["step"]))
    for left, right in zip(rows, rows[1:]):
        a = float(left.get("ood_answerable_correctness") or 0.0)
        b = float(right.get("ood_answerable_correctness") or 0.0)
        ranked.append((b - a, int(right["step"])))
    return [step for _, step in sorted(ranked, reverse=True)[: max(0, count)]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--training-root", type=Path, required=True)
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--time-curve", type=Path, required=True)
    ap.add_argument("--train-probe", type=Path, required=True)
    ap.add_argument("--dev-probe", type=Path, required=True)
    ap.add_argument("--ood-probe", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--require-cuda", action="store_true")
    ap.add_argument("--reuse-existing", action="store_true")
    args = ap.parse_args()
    cfg = read_json(args.config)
    checkpoints = discover_adapters(args.training_root)
    by_step = {int(item["step"]): item for item in checkpoints}
    available = sorted(by_step)
    max_step = max(available)
    time_rows = read_json(args.time_curve)["rows"]
    surface_cfg = cfg["dose_surface"]
    chosen = set(nearest_steps(available, surface_cfg["milestone_fractions"], max_step))
    jumps = biggest_ood_jumps(time_rows, int(surface_cfg["transition_neighbor_count"]))
    for jump in jumps:
        if jump not in available:
            continue
        idx = available.index(jump)
        for offset in range(-int(surface_cfg["transition_neighbor_count"]), int(surface_cfg["transition_neighbor_count"]) + 1):
            j = idx + offset
            if 0 <= j < len(available):
                chosen.add(available[j])
    steps = sorted(chosen)
    doses = [float(value) for value in surface_cfg["doses"]]
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for step in steps:
        item = by_step[step]
        for dose in doses:
            dest = args.output_root / f"step_{step:06d}" / ("dose_" + f"{dose:.2f}".replace(".", "p"))
            summary_path = dest / "summary.json"
            if not (args.reuse_existing and summary_path.exists()):
                cmd = [
                    sys.executable,
                    "scripts/phase_9_student_distillation/evaluate_grokking_checkpoint_v231.py",
                    "--base-model", args.base_model,
                    "--dose", str(dose),
                    "--step", str(step),
                    "--train-probe", str(args.train_probe),
                    "--dev-probe", str(args.dev_probe),
                    "--ood-probe", str(args.ood_probe),
                    "--output-dir", str(dest),
                    "--logit-limit", str(min(48, int(cfg["probes"]["logit_cases_per_split"]))),
                    "--limit-per-split", str(int(surface_cfg["probe_limit_per_split"])),
                ]
                if dose != 0.0 and item.get("adapter"):
                    cmd += ["--adapter", str(item["adapter"])]
                elif dose != 0.0 and not item.get("adapter"):
                    # Step zero has no residual delta, so every dose is identical.
                    pass
                if args.require_cuda:
                    cmd.append("--require-cuda")
                run(cmd)
            summary = read_json(summary_path)
            row: dict[str, Any] = {"step": step, "dose": dose}
            for split_name, split in summary["splits"].items():
                for key in (
                    "answerable_correctness",
                    "hard_answerable_correctness",
                    "false_abstention_rate",
                    "answer_coverage",
                    "target_token_nll_mean",
                    "first_target_margin_mean",
                    "first_target_rank_mean",
                    "first_target_gold_probability_mean",
                ):
                    row[f"{split_name}_{key}"] = split.get(key)
            rows.append(row)
    payload = {
        "schema_version": "2.3.1",
        "sealed_data_used": False,
        "steps": steps,
        "doses": doses,
        "rows": rows,
    }
    write_json(args.output_root / "dose_time_surface.json", payload)
    write_csv(args.output_root / "dose_time_surface.csv", rows)
    md = [
        "# v2.3.1 Step × Dose Surface",
        "",
        "| Step | Dose | Train ans | DEV ans | OOD ans | DEV margin | OOD margin |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md.append(
            f"| {row['step']} | {row['dose']} | {row.get('train_answerable_correctness')} | {row.get('dev_answerable_correctness')} | "
            f"{row.get('ood_answerable_correctness')} | {row.get('dev_first_target_margin_mean')} | {row.get('ood_first_target_margin_mean')} |"
        )
    (args.output_root / "DOSE_TIME_SURFACE.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
