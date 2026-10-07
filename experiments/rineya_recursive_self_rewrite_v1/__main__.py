"""Offline commands. Native model loading/training is never triggered by this CLI."""
from __future__ import annotations

import argparse
from pathlib import Path

from .artifacts import verify_run, write_run
from .core import Limits, canonical, loads
from .data import compile_dataset
from .fixtures import catalog, fixture_backend
from .pipeline import discovery, reconstruct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    demo = sub.add_parser("demo")
    demo.add_argument("--out", required=True)
    demo.add_argument("--tasks", type=int, default=12, choices=range(1, 49))
    verify = sub.add_parser("verify")
    verify.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "plan":
        print(canonical(loads((Path(__file__).parent / "protocol.json").read_text())))
    elif args.command == "verify":
        print(canonical(verify_run(args.out)))
    else:
        tasks, ancestry = catalog(args.tasks)
        backend, limits = fixture_backend(), Limits()
        found = discovery(tasks, backend, limits)
        sources = [next(r for r in found["records"] if r["task"] == t.key and r["arm"] == "portfolio" and r["success"])
                   for t in tasks]
        by_task = {t.key: t for t in tasks}
        rewrites = [reconstruct(by_task[s["task"]], s, backend, backend, backend, limits, arm=arm)
                    for arm in ("one_pass", "recursive") for s in sources]
        datasets = [compile_dataset(tasks, ancestry, found["records"], rewrites, arm=arm,
                                    source_ids=[s["id"] for s in sources])
                    for arm in ("direct", "one_pass", "recursive")]
        write_run(args.out, tasks, ancestry, found, rewrites, datasets)
        print(canonical(verify_run(args.out)))


if __name__ == "__main__":
    main()
