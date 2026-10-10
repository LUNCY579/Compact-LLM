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


REQUIRED_TEST_IDS = frozenset(
    f"{module}.{class_name}.{method}"
    for module, class_name, methods in (
        ("test_registration", "RegistrationTests", (
            "test_complementary_non_expiring_scope",
            "test_documents_are_present_and_local_links_resolve",
            "test_future_native_values_remain_unset",
            "test_identity_and_design_status",
            "test_interaction_control_and_baseline",
            "test_no_execution_permissions",
            "test_no_native_or_sampler_result_claims",
            "test_owner_controlled_disposition",
            "test_parent_pin",
            "test_related_sources_are_pinned_not_replaced",
            "test_reversible_parking",
            "test_versioned_paper",
        )),
        ("test_tiny_sampler", "TinySamplerExactTests", (
            "test_all_joint_transitions_and_stationarity_after_composition",
            "test_all_local_acceptance_ratios",
            "test_all_suffix_scores_and_uniform_restart_support",
            "test_canonical_complete_records_and_base_probabilities",
            "test_default_restart_is_uniform_for_every_record_and_power",
            "test_deliberately_broken_truncation_is_rejected_by_exact_qualification",
            "test_eos_can_lengthen_and_shorten_only_when_prefix_is_restarted",
            "test_every_local_transition_normalization_balance_and_stationarity",
            "test_exchange_execution_recomputes_receiver_power_scores",
            "test_finite_initialized_chain_is_not_automatically_stationary",
            "test_invalid_power_and_restart_fail_closed",
            "test_power_targets_are_exact_sequence_targets",
            "test_random_restart_post_eos_and_seeded_replay",
            "test_rejected_work_contributes_to_self_transition",
            "test_sampled_step_all_proposal_boundaries_and_acceptance_branches",
            "test_stale_replica_scores_refuse_exchange_before_any_draw",
            "test_swap_orientation_and_detailed_balance",
        )),
        ("test_cpu_qualification", "QualificationGateTests", (
            "test_complete_inventory_allows_extra_tests",
            "test_loaded_but_unexecuted_checks_are_rejected",
            "test_missing_mandatory_checks_reject_unrelated_replacements",
            "test_result_records_executed_test_ids",
            "test_skips_and_expected_failures_are_rejected",
        )),
    )
    for method in methods
)


class QualificationResult(unittest.TextTestResult):
    """Record actual execution, independently of discovery and total count."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.executed_test_ids: list[str] = []

    def startTest(self, test):
        self.executed_test_ids.append(test.id())
        super().startTest(test)


def test_ids(suite):
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            yield from test_ids(test)
        else:
            yield test.id()


def qualification_issues(loaded_ids, result) -> list[str]:
    issues = []
    for phase, present in (("loaded", loaded_ids), ("executed", result.executed_test_ids)):
        missing = REQUIRED_TEST_IDS - set(present)
        if missing:
            issues.append(f"mandatory tests not {phase}: " + ", ".join(sorted(missing)))
    if not result.wasSuccessful():
        issues.append("test failures or errors")
    if result.skipped:
        issues.append("skipped tests")
    if result.expectedFailures:
        issues.append("expected failures cannot qualify mechanics")
    if result.testsRun != len(result.executed_test_ids):
        issues.append("execution inventory does not match result count")
    return issues


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
    loaded_ids = tuple(test_ids(suite))
    result = unittest.TextTestRunner(stream=output, verbosity=2,
                                    resultclass=QualificationResult).run(suite)
    cpu_seconds, wall_seconds = time.process_time() - started_cpu, time.perf_counter() - started_wall
    peaks = process_peaks()
    hashes_after = {str(path.relative_to(ROOT)).replace("\\", "/"):
                    hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    measured_memory = max(peaks["peak_working_set_bytes"], peaks["peak_private_commit_bytes"] or 0)
    target_met = cpu_seconds < 60 and measured_memory < 256 * 1024 ** 2
    source_stable = hashes_before == hashes_after
    issues = qualification_issues(loaded_ids, result)
    success = not issues and target_met and source_stable
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
                        "skipped": len(result.skipped),
                        "expected_failures": len(result.expectedFailures),
                        "required_test_ids": sorted(REQUIRED_TEST_IDS),
                        "loaded_test_ids": list(loaded_ids),
                        "executed_test_ids": result.executed_test_ids,
                        "qualification_issues": issues, "output": output.getvalue()},
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
