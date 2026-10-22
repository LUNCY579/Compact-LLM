"""Wait for parent containment before executing the unchanged v1 entrypoint."""
import argparse
import json
from pathlib import Path
import runpy
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--release", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    started = time.monotonic()
    while not args.release.is_file():
        if time.monotonic() - started > 30:
            raise RuntimeError("parent did not establish containment")
        time.sleep(.05)
    admission = json.loads(args.release.read_text(encoding="utf-8"))
    if admission["contained"] is not True:
        raise RuntimeError("containment not admitted")
    sys.argv = ["experiments.rineya_taps_adaptive_recurrence_v1", "native-dev",
                "--source", str(args.source), "--out", str(args.out)]
    runpy.run_module("experiments.rineya_taps_adaptive_recurrence_v1", run_name="__main__")


if __name__ == "__main__":
    main()
