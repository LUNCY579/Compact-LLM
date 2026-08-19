from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from . import __version__


def _print(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rineya-state-lab",
        description="Train and evaluate full-context versus compact-state reasoning adapters.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    data = sub.add_parser("data", help="Prepare and validate the paired dataset")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    prepare = data_sub.add_parser("prepare", help="Safely extract the bundled dataset")
    prepare.add_argument("--archive", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--overwrite", action="store_true")
    validate = data_sub.add_parser("validate", help="Validate alignment, splits, and transition chains")
    validate.add_argument("--data-root", required=True)
    validate.add_argument("--strict", action="store_true")
    validate.add_argument("--output")

    train = sub.add_parser("train", help="Train one LoRA/QLoRA adapter")
    train.add_argument("--config", required=True)
    train.add_argument("--set", dest="overrides", action="append", default=[], help="Override key=value")

    evaluate = sub.add_parser("evaluate", help="Run one configured evaluation matrix")
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--set", dest="overrides", action="append", default=[])
    evaluate.add_argument("--force", action="store_true")

    compare = sub.add_parser("compare", help="Run paired statistical comparison by problem ID")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--output", required=True)
    compare.add_argument("--baseline-protocol")
    compare.add_argument("--candidate-protocol")
    compare.add_argument("--metric", default="score.correct_strict")
    compare.add_argument("--noninferiority-margin", type=float, default=0.02)
    compare.add_argument("--bootstrap-samples", type=int, default=10000)
    compare.add_argument("--randomization-samples", type=int, default=20000)
    compare.add_argument("--seed", type=int, default=42)

    experiment = sub.add_parser("experiment", help="Run the complete staged experiment")
    experiment_sub = experiment.add_subparsers(dest="experiment_command", required=True)
    exp_run = experiment_sub.add_parser("run")
    exp_run.add_argument("--config", required=True)
    exp_run.add_argument("--stages", help="Comma-separated stage names")
    exp_run.add_argument("--force", action="store_true")

    doctor = sub.add_parser("doctor", help="Inspect Python, CUDA, dependencies, model, and dataset")
    doctor.add_argument("--model-path")
    doctor.add_argument("--data-root")
    doctor.add_argument("--output")
    doctor.add_argument("--require-cuda", action="store_true")

    smoke = sub.add_parser("smoke-test", help="Run a no-model reference evaluation and comparison")
    smoke.add_argument("--work-dir", default="outputs/smoke")
    smoke.add_argument(
        "--archive",
        default="assets/datasets/Rineya_Sequence_Independence_Dataset_v1.0_COMPLETE.zip",
    )
    smoke.add_argument("--limit", type=int, default=24)
    smoke.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        if args.command == "data" and args.data_command == "prepare":
            from .data.prepare import prepare_dataset

            _print(prepare_dataset(args.archive, args.output, overwrite=args.overwrite))
        elif args.command == "data" and args.data_command == "validate":
            from .data.validate import validate_dataset

            _print(validate_dataset(args.data_root, strict=args.strict, output=args.output))
        elif args.command == "train":
            from .training.run import run_training

            _print(run_training(args.config, args.overrides))
        elif args.command == "evaluate":
            from .evaluation.runner import run_evaluation

            _print(run_evaluation(args.config, args.overrides, force=args.force))
        elif args.command == "compare":
            from .evaluation.compare import compare_runs

            _print(
                compare_runs(
                    args.baseline,
                    args.candidate,
                    args.output,
                    baseline_protocol=args.baseline_protocol,
                    candidate_protocol=args.candidate_protocol,
                    metric=args.metric,
                    noninferiority_margin=args.noninferiority_margin,
                    bootstrap_samples=args.bootstrap_samples,
                    randomization_samples=args.randomization_samples,
                    seed=args.seed,
                )
            )
        elif args.command == "experiment" and args.experiment_command == "run":
            from .experiment import run_experiment

            stages = [part.strip() for part in args.stages.split(",") if part.strip()] if args.stages else None
            _print(run_experiment(args.config, stages=stages, force=args.force))
        elif args.command == "doctor":
            from .doctor import run_doctor

            report = run_doctor(
                model_path=args.model_path,
                data_root=args.data_root,
                output=args.output,
                require_cuda=args.require_cuda,
            )
            _print(report)
            if not report["passed"]:
                raise SystemExit(2)
        elif args.command == "smoke-test":
            from .smoke import run_smoke_test

            _print(run_smoke_test(work_dir=args.work_dir, archive=args.archive, limit=args.limit, force=args.force))
        else:
            raise RuntimeError("Unreachable command dispatch")
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
    except SystemExit:
        raise
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
