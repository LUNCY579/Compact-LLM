"""Offline commands only. Native execution intentionally has no success path."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .core import calibrate, cell_metrics, grid, replay
from .fixture import bank


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "smoke", "native-preflight"))
    parser.add_argument("--out", type=Path, help="new output path; never overwrite a receipt")
    args = parser.parse_args()
    matrix = grid((1, 2, 4, 8), (1, 2, 4, 8))
    if args.command == "plan":
        result = {"experiment": "CDE-001", "cells": matrix,
                  "training_arms": ["full", "sampled", "all", "adjacent", "deepest"],
                  "native_enabled": False, "self_relabeling": "separate_training_data_intervention"}
    elif args.command == "smoke":
        dev, test = bank("dev"), bank("test")
        policy = calibrate(dev, (0.0, 0.01, 0.1), patience=2)
        result = {"experiment": "CDE-001", "evidence": "SCRIPTED_FIXTURE_ONLY",
                  "native_forwards": 0, "training_steps": 0, "policy": asdict(policy),
                  "heldout_fixture": replay(test, policy, heldout=True),
                  "cells": [cell_metrics(test, cell["depth"], cell["width"]) for cell in matrix]}
    else:
        result = {"experiment": "CDE-001", "status": "BLOCKED", "native_enabled": False,
                  "native_forwards": 0, "weights_loaded": False,
                  "blockers": ["Native CDE trace collector/training driver are not implemented.",
                               "Exact recurrent checkpoint/runtime/tokenizer binding is unqualified.",
                               "Per-depth full-prefix/incremental and cache parity are unqualified.",
                               "Actual memory guards, work accounting and retention gates must be inherited."]}
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        try:
            with args.out.open("x", encoding="utf-8") as handle:
                handle.write(text)
        except FileExistsError:
            parser.error(f"refusing to overwrite {args.out}")
    else:
        print(text, end="")
    return 2 if args.command == "native-preflight" else 0


if __name__ == "__main__":
    raise SystemExit(main())
