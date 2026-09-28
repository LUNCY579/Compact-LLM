"""Run with python -m experiments.rineya_composable_subspace_stack_v1."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from .protocol import (read_json, write_new, default_plan, validate_plan, source_digest,
                       digest, audit_splits, freeze)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("init-plan"); q.add_argument("--output", type=Path, required=True)
    q = sub.add_parser("validate"); q.add_argument("--plan", type=Path, required=True)
    q = sub.add_parser("demo"); q.add_argument("--output", type=Path, required=True)
    q = sub.add_parser("audit-splits"); q.add_argument("--ledgers", type=Path, required=True); q.add_argument("--output", type=Path, required=True)
    for cmd in ("preflight", "build", "collect"):
        q = sub.add_parser(cmd)
        q.add_argument("--plan", type=Path, required=True)
        q.add_argument("--grant", type=Path, required=True)
        if cmd != "collect":
            q.add_argument("--manifest", type=Path, required=True)
        if cmd == "build":
            q.add_argument("--arm", required=True)
            q.add_argument("--output", type=Path, required=True)
        if cmd == "collect":
            q.add_argument("--model", type=Path, required=True)
            q.add_argument("--suite", type=Path, required=True)
            q.add_argument("--freeze", type=Path, required=True)
            q.add_argument("--selection", type=Path)
            q.add_argument("--output", type=Path, required=True)
    q = sub.add_parser("freeze")
    for arg in ("plan", "artifacts", "output"):
        q.add_argument("--"+arg, type=Path, required=True)
    for arg in ("cal-sha256", "check-sha256", "audit-sha256"):
        q.add_argument("--"+arg, required=True)
    q = sub.add_parser("compare")
    for arg in ("reference", "candidate", "output"):
        q.add_argument("--"+arg, type=Path, required=True)
    q = sub.add_parser("select")
    for arg in ("plan", "reference", "freeze", "output"):
        q.add_argument("--"+arg, type=Path, required=True)
    q.add_argument("--candidates", type=Path, nargs="+", required=True)
    q = sub.add_parser("confirm")
    for arg in ("run", "freeze", "selection", "output"):
        q.add_argument("--"+arg, type=Path, required=True)
    args = p.parse_args(argv)
    try:
        if args.command == "init-plan":
            result = default_plan(); write_new(args.output, result)
        elif args.command == "validate":
            plan = read_json(args.plan)
            result = {"status": "PLAN_VALID", "arms": len(validate_plan(plan)), "plan_sha256": digest(plan), "code_sha256": source_digest()}
        elif args.command == "demo":
            from .demo import run_demo
            result = run_demo(args.output)
        elif args.command == "audit-splits":
            result = audit_splits(read_json(args.ledgers)); write_new(args.output, result)
        elif args.command == "preflight":
            from .artifacts import preflight
            result = preflight(read_json(args.plan), args.manifest, read_json(args.grant))
        elif args.command == "build":
            from .artifacts import build
            result = build(read_json(args.plan), args.manifest, read_json(args.grant), args.arm, args.output)
        elif args.command == "collect":
            from .native import collect
            result = collect(read_json(args.plan), read_json(args.grant), args.model, args.suite, args.output,
                             read_json(args.freeze), read_json(args.selection) if args.selection else None)
        elif args.command == "freeze":
            result = freeze(read_json(args.plan), read_json(args.artifacts), cal_sha256=args.cal_sha256,
                            check_sha256=args.check_sha256, audit_sha256=args.audit_sha256)
            write_new(args.output, result)
        elif args.command == "compare":
            from .evaluation import compare
            result = compare(read_json(args.reference), read_json(args.candidate)); write_new(args.output, result)
        elif args.command == "select":
            from .evaluation import select_on_cal
            plan = read_json(args.plan); validate_plan(plan)
            frozen = read_json(args.freeze)
            if frozen["plan_sha256"] != digest(plan):
                raise ValueError("selection plan differs from frozen plan")
            result = select_on_cal(read_json(args.reference), [read_json(x) for x in args.candidates], frozen,
                                   retention_margin=plan["retention_margin"], min_gain=plan["min_capability_gain"])
            write_new(args.output, result)
        else:
            from .evaluation import confirm_check
            result = confirm_check(read_json(args.run), read_json(args.freeze), read_json(args.selection)); write_new(args.output, result)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError, ImportError, RuntimeError) as exc:
        print(json.dumps({"status": "REFUSED", "error": str(exc), "release_permitted": False}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
