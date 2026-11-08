"""Contained bootstrap: no model import before parent release and exact review."""
import argparse
import json
import math
import os
from pathlib import Path
import time
import psutil
from .fresh_scope import sha, verify_dispatch, V3, CAMPAIGN


def validate_parent_binding(args, admission, lock):
    run_root = Path(lock["out"]).resolve(strict=True)
    if (V3 not in run_root.parents or args.source.resolve() != Path(admission["source"]).resolve()
            or args.out.resolve() != run_root / "result" or args.release.resolve() != run_root / "release.json"
            or lock.get("lane") != admission["lane"]
            or type(lock.get("parent_pid")) is not int or lock["parent_pid"] <= 0
            or type(lock.get("token")) is not str or not lock["token"]):
        raise PermissionError("canonical source/output/release/parent lock binding required")


def process_record(process):
    return {"pid": process.pid, "parent_pid": process.ppid(), "created": process.create_time(), "exe": process.exe()}


def validate_process_chain(lock, current, direct_parent, supervisor):
    """Direct interpreter or one owned Windows venv redirector, never arbitrary ancestry."""
    def same(record, pid, created, exe):
        return (record["pid"] == pid and record["created"] == created and
                os.path.normcase(record["exe"]) == os.path.normcase(exe))
    if not same(supervisor, lock["parent_pid"], lock["parent_created"], lock["runtime_exe"]):
        raise PermissionError("supervisor process identity changed")
    direct = (same(current, lock["launcher_pid"], lock["launcher_created"], lock["runtime_exe"])
              and current["parent_pid"] == lock["parent_pid"])
    redirected = (current["parent_pid"] == lock["launcher_pid"] and
                  same(direct_parent, lock["launcher_pid"], lock["launcher_created"], lock["launcher_exe"])
                  and direct_parent["parent_pid"] == lock["parent_pid"] and
                  os.path.normcase(current["exe"]) == os.path.normcase(lock["runtime_exe"]))
    if not direct and not redirected:
        raise PermissionError("bootstrap is not the owned launcher or its exact venv child")


def validate_clock(started, now, deadline):
    elapsed = now - started
    if (type(started) not in (int, float) or type(now) not in (int, float)
            or not math.isfinite(started) or not math.isfinite(elapsed) or not 0 <= elapsed < deadline):
        raise PermissionError("finite nonfuture unexpired parent clock required before model imports")


def validate_release(release, admission, lock, admission_sha):
    if (release.get("contained") is not True or release.get("admission_sha256") != admission_sha
            or release.get("study_id") != admission["study_id"] or release.get("token") != lock["token"]):
        raise PermissionError("exact contained fresh release required")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("release", "source", "out", "admission"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    admission = verify_dispatch(args.admission)
    lock = json.loads((CAMPAIGN / "native_worker.lock").read_text(encoding="utf-8"))
    validate_parent_binding(args, admission, lock)
    validate_process_chain(lock, process_record(psutil.Process()), process_record(psutil.Process(os.getppid())),
                           process_record(psutil.Process(lock["parent_pid"])))
    started = float(os.environ["FRESH_PARENT_STARTED_PERF_COUNTER"])
    validate_clock(started, time.perf_counter(), admission["hard_deadline_seconds"])
    while not args.release.exists():
        if time.perf_counter() - started >= 30:
            raise TimeoutError("contained fresh parent release absent")
        time.sleep(.05)
    release = json.loads(args.release.read_text(encoding="utf-8"))
    validate_release(release, admission, lock, sha(args.admission))
    from .fresh_native import run
    return run(args.source, args.out, args.admission)


if __name__ == "__main__":
    raise SystemExit(main())
