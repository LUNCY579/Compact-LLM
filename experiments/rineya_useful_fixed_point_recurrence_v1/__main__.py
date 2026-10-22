"""Run via python -m experiments.rineya_useful_fixed_point_recurrence_v1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from .protocol import preflight, paired_report, new_run, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fixture = sub.add_parser("fixtures", help="CPU analytic and synthetic training mechanics only")
    fixture.add_argument("--out", type=Path, required=True)
    check = sub.add_parser("preflight", help="no model loading or native execution")
    check.add_argument("--permit", type=Path, required=True)
    native = sub.add_parser("native-screen", help="explicitly permitted Stage 0/1; never trains")
    for key in ("permit", "repo", "checkpoint", "cases", "out"):
        native.add_argument(f"--{key}", type=Path, required=True)
    report = sub.add_parser("report", help="one arm/seed/split of explicit paired outcomes")
    report.add_argument("--rows", type=Path, required=True)
    report.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "preflight":
        result = preflight(json.loads(args.permit.read_text()))
    elif args.command == "fixtures":
        from .fixtures import run_fixtures
        result = run_fixtures(args.out)
        result = {k: v for k, v in result.items() if k not in {"trajectories", "matched_training"}}
    elif args.command == "native-screen":
        from .native import run_native
        result = run_native(json.loads(args.permit.read_text()), repo=args.repo,
                            checkpoint=args.checkpoint, cases_path=args.cases, out=args.out)
    else:
        rows = [json.loads(line) for line in args.rows.read_text().splitlines() if line.strip()]
        result = paired_report(rows)
        write_json(new_run(args.out) / "paired_report.json", result)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 2 if result.get("status") in {"BLOCKED", "FAIL"} else 0


if __name__ == "__main__":
    raise SystemExit(main())
