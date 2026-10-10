"""Run only this package's stdlib checks and retain a measured CPU receipt.

Usage from repository root:
python -B experiments/rineya_ppt_centralized_reasoning_v1/qualify_cpu.py
The resource targets are engineering checks, not native launch authorization.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import unittest

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]


def process_peaks() -> dict:
    if os.name == "nt":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage",
                    "PrivateUsage")]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return {"method": "Windows GetProcessMemoryInfo lifetime peaks",
                "peak_working_set_bytes": counters.PeakWorkingSetSize,
                "peak_private_commit_bytes": counters.PeakPagefileUsage,
                "resource_accounting_scope": "single test process; tests spawn no children"}
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"method": "getrusage lifetime peak RSS",
            "peak_working_set_bytes": int(peak * (1 if sys.platform == "darwin" else 1024)),
            "peak_private_commit_bytes": None,
            "resource_accounting_scope": "single test process; tests spawn no children"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", default="evidence/cpu_sampler_2026-10-03.json",
                        help="new receipt path relative to this package; existing receipts are preserved")
    args = parser.parse_args()
    destination = (PACKAGE / args.receipt).resolve()
    if not destination.is_relative_to(PACKAGE) or destination.suffix != ".json":
        parser.error("receipt must be a JSON file inside this package")
    if destination.exists():
        parser.error("receipt already exists; choose a new --receipt path")

    # Provenance metadata is collected before the measured in-process test run.
    # No remote command or inherited runtime is invoked.
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                         text=True, capture_output=True, check=True).stdout.strip()
    sources = [PACKAGE / "tiny_sampler.py", Path(__file__).resolve(),
               PACKAGE / "registration.json", *sorted((PACKAGE / "tests").glob("test_*.py"))]
    hashes_before = {str(path.relative_to(ROOT)).replace("\\", "/"):
                     hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    sys.path.insert(0, str(ROOT))
    started_cpu, started_wall = time.process_time(), time.perf_counter()
    output = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(PACKAGE / "tests"), pattern="test_*.py")
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    cpu_seconds, wall_seconds = time.process_time() - started_cpu, time.perf_counter() - started_wall
    peaks = process_peaks()
    hashes_after = {str(path.relative_to(ROOT)).replace("\\", "/"):
                    hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    measured_memory = max(peaks["peak_working_set_bytes"], peaks["peak_private_commit_bytes"] or 0)
    target_met = cpu_seconds < 60 and measured_memory < 256 * 1024 ** 2
    source_stable = hashes_before == hashes_after
    success = (result.wasSuccessful() and result.testsRun > 12 and not result.skipped
               and target_met and source_stable)
    receipt = {
        "schema_version": 1, "experiment_id": "PPT-001",
        "record_kind": "CPU_ONLY_TINY_ENUMERABLE_SAMPLER_QUALIFICATION",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_base_commit": git, "source_sha256": hashes_before,
        "sources_unchanged_during_tests": source_stable,
        "python_version": platform.python_version(), "platform": platform.platform(),
        "fixture": {"horizon": 2, "canonical_states": 7, "joint_states": 49,
                    "arithmetic": "exact fractions and integer random draws",
                    "scope": "supplied toy probabilities; no checkpoint or native adapter"},
        "test_result": {"passed": result.wasSuccessful(), "count": result.testsRun,
                        "failures": len(result.failures), "errors": len(result.errors),
                        "skipped": len(result.skipped), "output": output.getvalue()},
        "measurements": {"test_cpu_seconds": cpu_seconds, "test_wall_seconds": wall_seconds, **peaks},
        "proposed_engineering_target": {"cpu_seconds_strict_upper_bound": 60,
                                        "memory_bytes_strict_upper_bound": 256 * 1024 ** 2,
                                        "met_in_this_run": target_met,
                                        "native_admission_or_authorization": False},
        "qualified": success, "native_scoring_qualified": False,
        "native_benchmark_ready": False, "model_loads": 0, "training_runs": 0,
        "protected_banks_opened": False,
        "decision_criteria": ["A5 must beat identical A4 uncoupled replicas",
                              "A5 must also beat strong A2 independent sampling at matched total work"],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves previous receipts even if another run races.
    with destination.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"receipt": str(destination), "qualified": success,
                      "tests": result.testsRun, "cpu_seconds": cpu_seconds,
                      "wall_seconds": wall_seconds, "peak_memory_mib": measured_memory / 1024 ** 2}))
    if not result.wasSuccessful():
        print(output.getvalue(), file=sys.stderr)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
