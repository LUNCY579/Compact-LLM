"""Durable producer-side receipts for one publication integrity run.

This wrapper measures no full native process envelope and grants no training
permission. An intended exit code is separate from an observed process exit.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

LIBRARY_SHA256 = "129378230a0077bfb06700963bac737382dd9dcdaddcb39eb21003295820a047"
CATALOG_SHA256 = "fc21b2294710e0f429439de3270097ca16990e923788d745918c1af61af3114e"


def write(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main():
    archive = Path(sys.argv[1]).absolute()
    output = Path(sys.argv[2]).absolute()
    library = Path(__file__).resolve().with_name("snapshot.py")
    if hashlib.sha256(library.read_bytes()).hexdigest() != LIBRARY_SHA256:
        raise RuntimeError("publication verifier library pin mismatch")
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    identity = {"executor_pid": os.getpid(), "parent_pid": os.getppid(),
                "runtime": sys.executable, "started_unix": started,
                "archive": str(archive), "library_sha256": LIBRARY_SHA256,
                "catalog_sha256": CATALOG_SHA256,
                "scope": "Publication integrity only, no dataset/native/model qualification"}
    write(output / "START.json", identity)
    try:
        import snapshot
        result = snapshot.verify(archive, CATALOG_SHA256)
        if hashlib.sha256(library.read_bytes()).hexdigest() != LIBRARY_SHA256:
            raise RuntimeError("publication verifier library changed")
        write(output / "RESULT.json", result)
        write(output / "PRODUCER_OUTCOME.json", dict(identity,
              status="VERIFICATION_COMPLETED", finished_unix=time.time(),
              intended_exit_code=0, observed_exit_code="REQUIRES_EXTERNAL_OBSERVATION",
              result_sha256=hashlib.sha256((output / "RESULT.json").read_bytes()).hexdigest()))
        print(json.dumps(result, sort_keys=True))
        return 0
    except BaseException as error:
        write(output / "PRODUCER_OUTCOME.json", dict(identity,
              status="VERIFICATION_FAILED_OR_INTERRUPTED", finished_unix=time.time(),
              error_type=type(error).__name__, intended_exit_code=1,
              completed_counts="UNKNOWN", full_verification_pass=False))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
