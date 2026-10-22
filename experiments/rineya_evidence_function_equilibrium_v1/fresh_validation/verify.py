"""Create-only CPU fixture verification receipt; never qualifies a native model."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from .preflight import PACKAGE, ROOT, file_sha, write_new


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if out == ROOT or ROOT in out.parents:
        parser.error("use a new external receipt directory")
    out.mkdir(parents=True, exist_ok=False)
    modules = ["experiments.rineya_evidence_function_equilibrium_v1.fresh_validation.test_preparation",
               "experiments.rineya_evidence_function_equilibrium_v1.fresh_validation.test_native_adapter"]
    command = [sys.executable, "-B", "-m", "unittest", *modules, "-v"]
    import psutil
    if psutil.virtual_memory().available < int(1.5 * 2 ** 30):
        write_new(out / "CPU_TEST_RECEIPT.json", {"status": "BLOCKED_LOW_HOST_RAM",
                  "native_qualification": "NOT_RUN", "native_model_loads": 0})
        return 2
    peak, cpu_user, cpu_system = 0, 0.0, 0.0
    start, timed_out = time.monotonic(), False
    with (out / "unittest.log").open("x", encoding="utf-8") as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        process = psutil.Process(child.pid)
        while child.poll() is None:
            try:
                # uv's python.exe is a small launcher on this host; include the
                # actual interpreter descendants, not just its 4-MiB launcher.
                tree = [process, *process.children(recursive=True)]
                rss, user, system = 0, 0.0, 0.0
                for member in tree:
                    try:
                        rss += member.memory_info().rss
                        times = member.cpu_times()
                        user += times.user
                        system += times.system
                    except psutil.NoSuchProcess:
                        pass
                peak = max(peak, rss)
                cpu_user, cpu_system = max(cpu_user, user), max(cpu_system, system)
            except psutil.NoSuchProcess:
                pass
            if time.monotonic() - start > 120:
                # Only the child launched by this CPU test verifier is stopped.
                for member in reversed(process.children(recursive=True)):
                    member.kill()
                child.kill()
                child.wait()
                timed_out = True
                break
            time.sleep(.05)
    elapsed = time.monotonic() - start
    exit_code = child.wait()
    # Explicitly inspect the unittest footer: a skipped tensor runtime is incomplete.
    output = (out / "unittest.log").read_text(encoding="utf-8")
    passed = exit_code == 0 and "Ran 22 tests" in output and "\nOK\n" in output and "skipped=" not in output
    value = {"kind": "SYNTHETIC_CPU_MECHANICS_ONLY", "status": "PASS" if passed else "FAIL_OR_INCOMPLETE",
             "command": command, "exit_code": exit_code, "timed_out": timed_out,
             "wall_seconds": elapsed, "peak_child_tree_rss_bytes_sampled": peak,
             "sample_interval_seconds": .05, "child_cpu_user_seconds_sampled": cpu_user,
             "child_cpu_system_seconds_sampled": cpu_system,
             "native_model_loads": 0, "cuda_model_calls": 0, "native_qualification": "NOT_RUN",
             "fresh_candidate_outcomes_read": 0, "tests_expected": 22,
             "log_sha256": file_sha(out / "unittest.log"),
             "tested_code_sha256": {p.name: file_sha(p) for p in sorted(PACKAGE.glob("*.py"))},
             "protocol_sha256": file_sha(PACKAGE / "protocol.json")}
    write_new(out / "CPU_TEST_RECEIPT.json", value)
    print(json.dumps({"status": value["status"], "tests_expected": 22,
                      "wall_seconds": elapsed, "peak_child_tree_rss_bytes_sampled": peak,
                      "out": str(out), "native_qualification": "NOT_RUN"}, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
