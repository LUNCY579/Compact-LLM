"""Contained bootstrap: no model import before parent release and exact review."""
import argparse
import json
import math
import os
from pathlib import Path
import time
from .fresh_scope import sha, verify_dispatch, V3, CAMPAIGN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("release", "source", "out", "admission"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    admission = verify_dispatch(args.admission)
    lock = json.loads((CAMPAIGN / "native_worker.lock").read_text(encoding="utf-8"))
    run_root = Path(lock["out"]).resolve(strict=True)
    if (V3 not in run_root.parents or args.source.resolve() != Path(admission["source"]).resolve()
            or args.out.resolve() != run_root / "result" or args.release.resolve() != run_root / "release.json"
            or lock.get("lane") != admission["lane"] or lock.get("parent_pid") != os.getppid()
            or type(lock.get("token")) is not str or not lock["token"]):
        raise PermissionError("canonical source/output/release/parent lock binding required")
    started = float(os.environ["FRESH_PARENT_STARTED_PERF_COUNTER"])
    elapsed = time.perf_counter() - started
    if not math.isfinite(started) or not math.isfinite(elapsed) or not 0 <= elapsed < admission["hard_deadline_seconds"]:
        raise PermissionError("finite nonfuture unexpired parent clock required before model imports")
    while not args.release.exists():
        if time.perf_counter() - started >= 30:
            raise TimeoutError("contained fresh parent release absent")
        time.sleep(.05)
    release = json.loads(args.release.read_text(encoding="utf-8"))
    if (release.get("contained") is not True or release.get("admission_sha256") != sha(args.admission)
            or release.get("study_id") != admission["study_id"] or release.get("token") != lock["token"]):
        raise PermissionError("exact contained fresh release required")
    from .fresh_native import run
    return run(args.source, args.out, args.admission)


if __name__ == "__main__":
    raise SystemExit(main())
