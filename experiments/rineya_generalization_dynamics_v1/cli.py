"""Run with python -m experiments.rineya_generalization_dynamics_v1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from .common import Blocked, read
from .pipeline import DEFAULT_NATIVE, DEFAULT_POLICY, confirm, evaluate_development, initialize, select, validate_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GD-001: isolated checkpoint generalization experiment")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "evaluate-dev", "select", "confirm", "report", "demo"):
        sub = commands.add_parser(name)
        sub.add_argument("--run", type=Path, required=True)
        if name in {"init", "demo"}:
            sub.add_argument("--pairs-per-family", type=int, default=24)
        if name == "init":
            sub.add_argument("--spec", type=Path, required=True)
            sub.add_argument("--policy", type=Path)
            sub.add_argument("--seed", type=int, default=17)
        if name in {"evaluate-dev", "confirm"}:
            sub.add_argument("--allow-native", action="store_true")
        if name == "confirm":
            sub.add_argument("--allow-confirmation", action="store_true")
            sub.add_argument("--seed", type=int, required=True)
    pre = commands.add_parser("preflight")
    pre.add_argument("--spec", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            output = initialize(args.run, read(args.spec), policy=read(args.policy) if args.policy else None,
                                seed=args.seed, per_family=args.pairs_per_family)
        elif args.command == "evaluate-dev":
            results = evaluate_development(args.run, args.allow_native)
            output = {"completed_checkpoints": [r["checkpoint_id"] for r in results], "evidence_kind": results[0]["evidence_kind"]}
        elif args.command == "select":
            output = select(args.run)
        elif args.command == "confirm":
            output = confirm(args.run, args.seed, allow_confirmation=args.allow_confirmation, allow_native=args.allow_native)
        elif args.command == "report":
            candidates = [args.run / name for name in ("confirmation_report.json", "development_report.json", "manifest.json")]
            output = read(next((p for p in candidates if p.is_file()), candidates[-1]))
        elif args.command == "preflight":
            from .native import preflight
            spec = read(args.spec) if args.spec else {}
            config = spec.get("runtime", DEFAULT_NATIVE)
            validate_config(DEFAULT_POLICY, config)
            output = preflight(config, any(c["kind"] == "peft" for c in spec.get("checkpoints", [])))
        else:
            initialize(args.run, mechanical=True, per_family=args.pairs_per_family)
            evaluate_development(args.run)
            select(args.run)
            output = confirm(args.run, 101, allow_confirmation=True)
        print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
        return 2 if output.get("status", "").startswith("BLOCKED") else 0
    except (Blocked, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "BLOCKED", "reason": str(exc)}, indent=2))
        return 2
