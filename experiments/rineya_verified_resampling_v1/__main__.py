"""Command-line interface; native imports occur only in supervised worker mode."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

from .contracts import PACKAGE, Blocked, read_json, write_new
from .datasets import prepare_centralized, prepare_fixture, freeze_study, load_study


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RVS-001 frozen-model resampling; no training or paid APIs")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare-centralized", help="generate audited open-role bridge data; no model/tokenizer")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--per-family", type=int, default=32)
    p.add_argument("--role", choices=["DEV", "TRAIN_SELECTION"], default="DEV")
    p.add_argument("--selection-seed", type=int, default=20260927)
    p = sub.add_parser("freeze", help="freeze inputs and policy, not native qualification")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--config", type=Path, default=PACKAGE / "configs" / "titan_pilot.json")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("demo", help="scripted CPU mechanics, never a Rineya model result")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--parents", type=int, default=16)
    p.add_argument("--scenario", choices=["minority", "flat", "all_correct", "vote_gain"], default="minority")
    p = sub.add_parser("evaluate", help="offline isolated oracle; no model imports")
    p.add_argument("--study", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--oracle", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    for command in ("qualify", "collect"):
        p = sub.add_parser(command, help="explicitly permitted Windows CUDA inference under existing watchdog")
        p.add_argument("--study", type=Path, required=True)
        p.add_argument("--model-dir", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--allow-model-load", action="store_true")
        if command == "collect":
            p.add_argument("--qualification", type=Path, required=True)
            p.add_argument("--allow-inference", action="store_true")
            p.add_argument("--resume", action="store_true")
            p.add_argument("--max-attempts", type=int)
            p.add_argument("--abandon-interrupted", action="store_true")
    p = sub.add_parser("doctor", help="inspect installed package metadata and OS; does not load a model")
    p = sub.add_parser("_native-worker", help=argparse.SUPPRESS)
    p.add_argument("--request", type=Path, required=True)
    p.add_argument("--go", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare-centralized":
            result = prepare_centralized(args.out, args.per_family, args.role, args.selection_seed)
        elif args.command == "freeze":
            result = freeze_study(args.data, args.config, args.out)
        elif args.command == "demo":
            from .runner import FixtureBackend, collect
            from .evaluation import evaluate
            args.out.mkdir(parents=True, exist_ok=False)
            prepare_fixture(args.out / "data", args.parents)
            freeze_study(args.out / "data", PACKAGE / "configs" / "titan_pilot.json", args.out / "study")
            study, public = load_study(args.out / "study")
            collect(study, public, FixtureBackend(args.scenario), args.out / "run")
            result = evaluate(args.out / "study", args.out / "run", args.out / "data" / "oracle.private.jsonl", args.out / "report")
        elif args.command == "evaluate":
            from .evaluation import evaluate
            result = evaluate(args.study, args.run, args.oracle, args.out)
        elif args.command == "doctor":
            import importlib.metadata
            import os
            from .native import VERSIONS
            actual = {}
            for name in VERSIONS:
                try:
                    actual[name] = importlib.metadata.version(name)
                except importlib.metadata.PackageNotFoundError:
                    actual[name] = "NOT_INSTALLED"
            result = {"status": "METADATA_ONLY_NOT_QUALIFICATION", "windows": os.name == "nt",
                      "expected": VERSIONS, "installed": actual, "native_model_loads": 0}
        elif args.command == "_native-worker":
            from .driver import native_worker
            result = native_worker(args.request, args.go)
        else:
            from .driver import native_launch
            result = native_launch(args.command, args.study, args.model_dir, args.out,
                                   qualification=getattr(args, "qualification", None), allow_model_load=args.allow_model_load,
                                   allow_inference=getattr(args, "allow_inference", False), resume=getattr(args, "resume", False),
                                   max_attempts=getattr(args, "max_attempts", None),
                                   abandon_interrupted=getattr(args, "abandon_interrupted", False))
        # Complete structured artifacts live on disk; keep CLI output short.
        summary = {k: v for k, v in result.items() if k not in {"curves", "config", "dataset", "checks", "run_binding", "caveats", "families_at_max_k"}}
        print(json.dumps(summary, indent=2, allow_nan=False))
        return 2 if result.get("status") == "BLOCKED" else 0
    except Exception as exc:
        failure = {"status": "BLOCKED", "command": args.command, "reason": f"{type(exc).__name__}: {exc}",
                   "efficacy": "UNMEASURED", "no_automatic_retry": True}
        if args.command == "_native-worker":
            try:
                write_new(args.request.parent / "worker_failure.json", failure)
            except OSError:
                pass
        elif hasattr(args, "out") and args.out.is_dir():
            write_new(args.out / f"blocked_{time.time_ns()}.json", failure)
        print(json.dumps(failure, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
