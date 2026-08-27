#!/usr/bin/env python3
"""Audit the schema-v2 Phase-C qualification and write PASS or STOP evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rineya_pcsm.gate import GateInputs, audit_phase_c_gate, write_gate_artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--budget", type=Path, required=True)
    parser.add_argument("--sample-manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--gradient-summary", type=Path, required=True)
    parser.add_argument("--gradient-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = GateInputs(
        run=args.run,
        plans=args.plans,
        preregistration=args.preregistration,
        budget=args.budget,
        sample=args.sample_manifest,
        split=args.split_manifest,
        gradient_summary=args.gradient_summary,
        gradient_cache=args.gradient_cache,
    )
    result = audit_phase_c_gate(inputs)
    write_gate_artifact(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
