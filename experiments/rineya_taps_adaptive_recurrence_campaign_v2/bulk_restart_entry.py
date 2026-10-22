"""Explicit larger-bank GPU bootstrap; imports no model before containment."""
import argparse
import json
import math
import os
from .full_generation_restart_authority import validate_human_scope
from pathlib import Path
import runpy
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--admission", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    while not args.release.is_file():
        if time.monotonic() - started > 30:
            raise RuntimeError("BULK parent did not establish containment")
        time.sleep(.05)
    if json.loads(args.release.read_text(encoding="utf-8"))["contained"] is not True:
        raise RuntimeError("BULK containment not admitted")
    admission = json.loads(args.admission.read_text(encoding="utf-8"))
    validate_human_scope(admission)
    parent_clock = float(os.environ.get("BULK_RESTART_PARENT_STARTED_PERF_COUNTER", "nan"))
    elapsed = time.perf_counter() - parent_clock
    if not math.isfinite(parent_clock) or parent_clock <= 0 or elapsed < 0 or elapsed > 16000:
        raise RuntimeError("exact parent-owned perf clock required before native import")
    sys.argv = ["experiments.rineya_taps_adaptive_recurrence_campaign_v2.bulk_restart_native",
                "--source", str(args.source), "--out", str(args.out), "--admission", str(args.admission)]
    runpy.run_module("experiments.rineya_taps_adaptive_recurrence_campaign_v2.bulk_restart_native", run_name="__main__")


if __name__ == "__main__":
    main()
