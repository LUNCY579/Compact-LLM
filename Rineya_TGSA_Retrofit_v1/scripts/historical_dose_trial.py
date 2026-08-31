#!/usr/bin/env python3
"""Preflight or build the frozen 1.0625 adapter-dose transfer trial."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1]
REPO = LAB.parent
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))

from rineya_tgsa.historical_dose import build_dose_trial, load_dose_registration, preflight_dose


def resolve_defaults(config: Path):
    value = load_dose_registration(config)
    resolve = lambda key: (REPO / value[key]["locator_from_retrofit_repo"]).resolve()
    output = lambda key: (LAB / value["outputs"][key]).resolve()
    return resolve("parent"), resolve("base"), resolve("adapter"), output("control"), output("candidate"), output("evidence")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "build"))
    parser.add_argument(
        "--config",
        type=Path,
        default=LAB / "configs" / "historical_dose_a11_1p0625_trial.json",
    )
    args = parser.parse_args()
    config = args.config.resolve(strict=True)
    parent, base, adapter, control, candidate, evidence = resolve_defaults(config)
    if args.command == "preflight":
        print(json.dumps(preflight_dose(config, parent, base, adapter), sort_keys=True))
        return 0
    print(
        json.dumps(
            build_dose_trial(
                config, parent, base, adapter, control, candidate, evidence
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
