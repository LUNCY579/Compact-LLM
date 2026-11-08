"""Aggregate explicit Track A immutable shards; refuse incomplete comparisons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROGRAM = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROGRAM))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROGRAM / "native_armr/configs/protocol.json")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--role", choices=["architecture_dev", "confirmatory_dev", "confirmatory_diagnostics"], default="confirmatory_dev")
    parser.add_argument("--run-list", type=Path, help="JSON array of explicit run directory paths; avoids Windows command-length limits")
    parser.add_argument("--primary-run-list", type=Path, help="Required for diagnostics: JSON array of all completed original-bank run directories")
    parser.add_argument("run_dirs", type=Path, nargs="*")
    args = parser.parse_args(argv)
    if args.run_list:
        if args.run_dirs:
            parser.error("provide either --run-list or positional run directories")
        from shared.guards import guard_non_sealed_path
        paths = json.loads(guard_non_sealed_path(args.run_list).read_text(encoding="utf-8"))
        if not isinstance(paths, list) or not paths or any(not isinstance(p, str) or not p for p in paths):
            parser.error("--run-list must contain a nonempty JSON array of directory paths")
        args.run_dirs = [Path(path) for path in paths]
    if not args.run_dirs:
        parser.error("explicit completed run directories are required")
    primary = None
    if args.primary_run_list:
        from shared.guards import guard_non_sealed_path
        values = json.loads(guard_non_sealed_path(args.primary_run_list).read_text(encoding="utf-8"))
        if not isinstance(values, list) or not values or any(not isinstance(p, str) or not p for p in values):
            parser.error("--primary-run-list must contain a nonempty JSON array of directory paths")
        primary = [Path(path) for path in values]
    from native_armr.src.audit import audit_runs
    output, result = audit_runs(config_path=args.config, dataset_manifest=args.manifest,
                               role=args.role, run_dirs=args.run_dirs, primary_run_dirs=primary)
    print(json.dumps({"audit_dir": str(output), **result}, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
