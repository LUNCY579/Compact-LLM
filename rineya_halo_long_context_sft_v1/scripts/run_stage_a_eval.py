#!/usr/bin/env python3
"""Run Halo vs Halo+adapter long-context eval, then sealed scoring if the bank exists."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from common import (
    LAB_ROOT,
    REPO_ROOT,
    SEALED_SCORER,
    atomic_write_json,
    iter_jsonl,
    load_config,
    python_executable,
    resolve_repo_path,
    stage_paths,
    validate_adapter_directory,
    write_jsonl,
)

PREDICT = SCRIPTS / "predict_long_context.py"
SCORE_MAIN = SCRIPTS / "score_main.py"
BUILD_SEALED = REPO_ROOT / "rineya_long_context_cc0_v1_main" / "scripts" / "build_sealed_eval.py"
SEPARATE_SEALED = REPO_ROOT / "rineya_long_context_cc0_v1_main" / "scripts" / "separate_sealed_prompts.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage A long-context evaluation for Halo QLoRA.")
    parser.add_argument("--config", type=Path, default=LAB_ROOT / "configs" / "stage_a_qlora.yaml")
    parser.add_argument("--split", choices=("val", "test", "sealed", "all"), default="all")
    parser.add_argument("--arm", choices=("baseline", "adapter", "both"), default="both")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--build-sealed", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    completed = subprocess.run(command, cwd=str(REPO_ROOT))
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def sealed_paths(corpus_root: Path) -> tuple[Path, Path]:
    prompts = corpus_root / "sealed_eval" / "prompts" / "prompts.jsonl"
    answers = corpus_root / "sealed_eval" / "answer_key" / "answers.jsonl"
    return prompts, answers


def maybe_build_sealed(corpus_root: Path) -> None:
    print("Building sealed evaluation bank (prompt-only + private answer key)...")
    run([python_executable(), str(BUILD_SEALED), "--root", str(corpus_root)])
    run([python_executable(), str(SEPARATE_SEALED), "--root", str(corpus_root)])


def limited_records(records: Path, dest: Path, limit: int) -> Path:
    if limit <= 0:
        return records
    rows = []
    for row in iter_jsonl(records):
        rows.append(row)
        if len(rows) >= limit:
            break
    write_jsonl(dest, rows)
    return dest


def predict_arm(
    *,
    config_path: Path,
    records: Path,
    output: Path,
    adapter: Path | None,
    limit: int,
    resume: bool,
) -> None:
    command = [
        python_executable(),
        str(PREDICT),
        "--config",
        str(config_path),
        "--records",
        str(records),
        "--output",
        str(output),
    ]
    if adapter is not None:
        command.extend(["--adapter", str(adapter)])
    if limit:
        command.extend(["--limit", str(limit)])
    if resume:
        command.append("--resume")
    run(command)


def score_arm(records: Path, predictions: Path, output: Path, arm: str) -> dict:
    run([
        python_executable(),
        str(SCORE_MAIN),
        "--records",
        str(records),
        "--predictions",
        str(predictions),
        "--output",
        str(output),
        "--arm",
        arm,
    ])
    return json.loads(output.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = stage_paths(config)
    corpus_root = resolve_repo_path(config["paths"]["corpus_root"])
    model_id = resolve_repo_path(config["paths"]["model_id"])
    eval_dir = paths["eval_dir"]
    eval_dir.mkdir(parents=True, exist_ok=True)
    adapter = paths["adapter_dir"]
    arms: list[tuple[str, Path | None]] = []
    if args.arm in {"baseline", "both"}:
        arms.append(("baseline", None))
    if args.arm in {"adapter", "both"}:
        if not adapter.exists():
            raise FileNotFoundError(f"adapter missing: {adapter}. Finish or resume training first.")
        validate_adapter_directory(adapter)
        arms.append(("adapter", adapter))

    summary: dict = {
        "model_id": str(model_id),
        "adapter": str(adapter) if adapter.exists() else None,
        "splits": {},
        "sealed": None,
    }

    split_map = {
        "val": paths["val_mix"],
        "test": paths["test_mix"],
    }
    requested = ["val", "test"] if args.split in {"all", "val", "test"} else []
    if args.split == "val":
        requested = ["val"]
    elif args.split == "test":
        requested = ["test"]
    elif args.split == "sealed":
        requested = []

    for split in requested:
        records = split_map[split]
        if not records.exists():
            raise FileNotFoundError(f"{records} missing. Run build_stage_mix.py first.")
        if args.limit:
            records = limited_records(records, eval_dir / f"{split}_limit{args.limit}.jsonl", args.limit)
        split_report: dict = {}
        for arm_name, adapter_path in arms:
            pred_path = eval_dir / f"{arm_name}_{split}.jsonl"
            score_path = eval_dir / f"{arm_name}_{split}_score.json"
            predict_arm(
                config_path=args.config,
                records=records,
                output=pred_path,
                adapter=adapter_path,
                limit=args.limit,
                resume=args.resume,
            )
            split_report[arm_name] = score_arm(records, pred_path, score_path, f"{arm_name}_{split}")
        if "baseline" in split_report and "adapter" in split_report:
            split_report["delta_trim_accuracy"] = (
                split_report["adapter"]["trim"]["accuracy"] - split_report["baseline"]["trim"]["accuracy"]
            )
            split_report["delta_exact_accuracy"] = (
                split_report["adapter"]["exact"]["accuracy"] - split_report["baseline"]["exact"]["accuracy"]
            )
        summary["splits"][split] = split_report

    if args.split in {"all", "sealed"}:
        prompts, answers = sealed_paths(corpus_root)
        if not prompts.exists() or not answers.exists():
            if args.build_sealed:
                maybe_build_sealed(corpus_root)
                prompts, answers = sealed_paths(corpus_root)
        if not prompts.exists() or not answers.exists():
            message = (
                "sealed bank missing: expected "
                f"{prompts} and {answers}. Skipping sealed scoring. "
                "Pass --build-sealed to generate locally, or place the prompt/answer JSONL files first."
            )
            print(message)
            summary["sealed"] = {"status": "SKIPPED_SEALED_BANK_MISSING", "message": message}
        else:
            for arm_name, adapter_path in arms:
                pred_path = eval_dir / f"{arm_name}_sealed.jsonl"
                predict_arm(
                    config_path=args.config,
                    records=prompts,
                    output=pred_path,
                    adapter=adapter_path,
                    limit=args.limit,
                    resume=args.resume,
                )
                if args.limit:
                    summary.setdefault("sealed", {})
                    summary["sealed"][arm_name] = {
                        "status": "SKIPPED_PARTIAL_LIMIT",
                        "predictions": str(pred_path),
                        "note": "score_sealed.py requires exact ID coverage; omit --limit for a confirmatory score.",
                    }
                    continue
                score_path = eval_dir / f"{arm_name}_sealed_score.json"
                run([
                    python_executable(),
                    str(SEALED_SCORER),
                    "--prompts",
                    str(prompts),
                    "--answers",
                    str(answers),
                    "--predictions",
                    str(pred_path),
                    "--output",
                    str(score_path),
                    "--normalization",
                    str((config.get("decode") or {}).get("normalization") or "trim"),
                ])
                summary.setdefault("sealed", {})
                summary["sealed"][arm_name] = json.loads(score_path.read_text(encoding="utf-8"))

    out = eval_dir / "stage_a_eval_summary.json"
    atomic_write_json(out, summary)
    atomic_write_json(LAB_ROOT / "reports" / "stage_a_eval_summary.json", summary)
    print(json.dumps({"wrote": str(out), "splits": list(summary["splits"]), "sealed": summary["sealed"] is not None}, indent=2))


if __name__ == "__main__":
    main()
