"""Bounded CPU-only CLI; supplied input files are exposed historical artifacts."""
import argparse
import json
from pathlib import Path
import time

from .study import ROOT, PACKAGE, encoded, evaluate, prepare, require, workspace_output


def main(argv=None):
    started = time.monotonic()
    parser = argparse.ArgumentParser(description="Blinded historical abstention annotations; no inference")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("evaluate")
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--labels-a", type=Path, required=True)
    p.add_argument("--labels-b", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        require(Path.cwd().resolve() == ROOT, "run from this assigned worktree root")
        args.out = workspace_output(args.out)
        if args.command == "prepare":
            result = prepare(args.out)
        else:
            args.prepared = workspace_output(args.prepared)
            # Annotation files must be kept in this worktree's ignored run directory.
            args.labels_a = workspace_output(args.labels_a)
            args.labels_b = workspace_output(args.labels_b)
            result = evaluate(args.prepared, args.labels_a, args.labels_b, args.out)
        print(json.dumps({k: result[k] for k in ("status", "responses", "worlds", "freeze_id")}, indent=2))
        return 0
    except Exception as exc:
        failure = {"status": "BLOCKED", "command": args.command, "reason": f"{type(exc).__name__}: {exc}",
                   "elapsed_seconds_through_failure": time.monotonic() - started,
                   "model_loads": 0, "no_automatic_retry": True, "preserve_partial_outputs": True,
                   "resource_costs": "external supervisor receipt required for full failed-worker costs"}
        safe_out = args.out.resolve()
        if safe_out.is_relative_to(PACKAGE / "runs") and safe_out.exists() and safe_out.is_dir():
            target = safe_out / ("BLOCKED_" + str(time.time_ns()) + ".json")
            with target.open("xb") as stream:
                stream.write(encoded(failure))
        print(json.dumps(failure, indent=2), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
