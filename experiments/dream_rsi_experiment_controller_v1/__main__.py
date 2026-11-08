"""CLI: python -m experiments.dream_rsi_experiment_controller_v1 --help."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .fixtures import create_bundle
from .replay import Plan
from .schema import Invalid, parse_world, read_json
from .study import evaluate_frozen, fit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RINEYA-DREAM-001 offline replay; no native execution")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="synthetic mechanical end-to-end qualification")
    demo.add_argument("--out", type=Path, required=True)
    fit_parser = commands.add_parser("fit", help="fit on train; select on validation; freeze without reading test")
    fit_parser.add_argument("--catalog", type=Path, required=True)
    fit_parser.add_argument("--plan", type=Path, required=True)
    fit_parser.add_argument("--out", type=Path, required=True)
    test = commands.add_parser("evaluate", help="one-shot frozen heldout evaluation")
    test.add_argument("--catalog", type=Path, required=True)
    test.add_argument("--run", type=Path, required=True)
    check = commands.add_parser("check-world", help="check one explicitly chosen world and source receipts")
    check.add_argument("--world", type=Path, required=True)
    check.add_argument("--source-root", type=Path, required=True)
    commands.add_parser("history-status", help="show pinned, read-only history admission audit")
    commands.add_parser("native", help="always blocked: no native runner is implemented or authorized")
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            args.out.mkdir(parents=True, exist_ok=False)
            catalog = create_bundle(args.out / "fixtures")
            fit(catalog, Plan(), args.out / "run")
            result = evaluate_frozen(catalog, args.out / "run")
        elif args.command == "fit":
            result = fit(args.catalog, Plan.parse(read_json(args.plan)), args.out)
        elif args.command == "evaluate":
            result = evaluate_frozen(args.catalog, args.run)
        elif args.command == "check-world":
            world = parse_world(read_json(args.world), args.source_root)
            result = {"id": world.key, "kind": world.kind, "nodes": len(world.nodes),
                      "content_sha256": world.content_sha256, "status": "SCHEMA_AND_BINDINGS_VALID",
                      "note": "not independent verification of historical truth or causal sufficiency"}
        elif args.command == "history-status":
            result = read_json(Path(__file__).parent / "HISTORY_AUDIT.json")
        else:
            raise Invalid("BLOCKED: RINEYA-DREAM-002/native execution needs a separate protocol and operator grant")
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (Invalid, OSError, TypeError, KeyError) as exc:
        print(json.dumps({"status": "BLOCKED", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
