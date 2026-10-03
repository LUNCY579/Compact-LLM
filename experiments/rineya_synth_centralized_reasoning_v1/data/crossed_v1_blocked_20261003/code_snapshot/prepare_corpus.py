"""Preserve a new, immutable CPU-only candidate corpus and matching receipt.

Examples from the repository root:
  python experiments/rineya_synth_centralized_reasoning_v1/prepare_corpus.py --output new_leaf
  python experiments/rineya_synth_centralized_reasoning_v1/prepare_corpus.py --output data/new_leaf

Bare leaves and data/<leaf> are package-relative. Other relative paths must
resolve from the current directory to one direct child of this package's data
directory. No existing output is overwritten, removed, or reused. A matched CPU
schedule still grants no native, model, training, or protected-bank permission.
"""
from __future__ import annotations

import argparse
import copy
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from balanced_corpus import build_balanced_plan, compile_balanced_records, validate_balanced_records
from cpu_tokenizer import load_local_tokenizer
from matching_schedule import ARRAY_FIELDS, freeze_enrollment, match_schedule, validate_schedule
from rs_contract import digest, require

PACKAGE = Path(__file__).resolve().parent
VERSION = "synth-cpu-preparation-1"
CPU_PEAK_LIMIT_BYTES = 512 * 1024 * 1024
CPU_WALL_LIMIT_SECONDS = 600
FRAMEWORK_ROOTS = {"torch", "transformers", "tensorflow", "jax", "flax"}


def no_frameworks() -> None:
    require(not any(name.split(".", 1)[0] in FRAMEWORK_ROOTS for name in sys.modules),
            "CPU preparation process must contain no model frameworks")


def resolve_output(value: str | Path) -> Path:
    """Resolve one new direct child; reject redirected data roots and traversal."""
    require(isinstance(value, (str, Path)) and bool(str(value)), "output path required")
    path = Path(value)
    require(".." not in path.parts, "output traversal is not permitted")
    package = PACKAGE.resolve()
    data = (package / "data").resolve()
    require(data.parent == package, "data directory must resolve inside this package")
    if not path.is_absolute():
        if len(path.parts) == 1:
            path = data / path
        elif path.parts[0] == "data":
            path = package / path
        else:
            path = Path.cwd() / path
    target = path.resolve()
    require(target.parent == data and target.name not in ("", ".", ".."),
            "output must be exactly one child of this package's data directory")
    require(not target.exists(), "output already exists; choose a new leaf")
    return target


def create_bundle(value: str | Path) -> Path:
    target = resolve_output(value)
    target.parent.mkdir(exist_ok=True)
    # mkdir without exist_ok closes the check/create race without reusing data.
    target.mkdir()
    require(target.resolve().parent == (PACKAGE.resolve() / "data").resolve(),
            "created bundle escaped its data directory")
    return target


def write_json_once(path: Path, value) -> None:
    text = json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def write_jsonl_once(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False) + "\n")


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def peak_working_set_bytes() -> int:
    """Read this process's Windows peak working set, never GPU/host inventory."""
    if os.name == "nt":
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise OSError(ctypes.get_last_error(), "cannot read Windows process peak working set")
        return int(counters.PeakWorkingSetSize)
    # This supports CPU fixture checks on other hosts without claiming Windows
    # memory evidence there. Production on this workspace uses the branch above.
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


class ResourceWatch:
    def __init__(self):
        self.started = time.perf_counter()
        self.peak_bytes = None
        self.memory_error = None
        self.stage = "initialization"

    def observe(self) -> None:
        value = peak_working_set_bytes()
        require(type(value) is int and value > 0, "invalid process memory telemetry")
        self.peak_bytes = max(self.peak_bytes or 0, value)

    def check(self, stage: str) -> None:
        self.stage = stage
        no_frameworks()
        self.observe()
        require(time.perf_counter() - self.started <= CPU_WALL_LIMIT_SECONDS,
                "prospective CPU wall budget exceeded; native qualification remains closed")
        require(self.peak_bytes <= CPU_PEAK_LIMIT_BYTES,
                "prospective CPU peak working set budget exceeded; native qualification remains closed")

    def record(self) -> dict:
        try:
            self.observe()
        except Exception as exc:
            self.memory_error = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - self.started
        return {"wall_seconds": round(elapsed, 6), "peak_working_set_bytes": self.peak_bytes,
                "memory_measurement": "Windows_PROCESS_MEMORY_COUNTERS.PeakWorkingSetSize" if os.name == "nt"
                                      else "process_ru_maxrss",
                "memory_telemetry_error": self.memory_error, "last_stage": self.stage,
                "prospective_cpu_wall_limit_seconds": CPU_WALL_LIMIT_SECONDS,
                "prospective_cpu_peak_limit_bytes": CPU_PEAK_LIMIT_BYTES,
                "cooperative_checks": True, "hard_process_kill_watchdog": False,
                "cpu_budget_satisfied": self.peak_bytes is not None and self.memory_error is None and
                                        self.peak_bytes <= CPU_PEAK_LIMIT_BYTES and elapsed <= CPU_WALL_LIMIT_SECONDS,
                "native_resource_qualification": False}


def validate_policy(policy: dict) -> None:
    require(type(policy) is dict, "matching policy must be an object")
    expected = {"schema_version": 1, "source_seed": 20261003, "candidate_parents": 64,
                "enrolled_parents": 20, "quadruple_repeat_min": 1, "quadruple_repeat_max": 6,
                "solver_time_limit_seconds": 60, "position_residual_fraction_of_answer_slots_max": .04,
                "between_arm_position_difference_fraction_of_answer_slots_max": .02,
                "source_pool_replacement_after_freeze": False, "native_launch_authorized": False,
                "protected_banks_accessed": False, "training_admitted": False,
                "shared_repeat_weights_and_occurrence_order_across_arms": True,
                "equal_S_k_and_R_first_requested_depth_row_histograms": True}
    require(all(type(policy.get(key)) is type(value) and policy[key] == value
                for key, value in expected.items()), "frozen matching policy changed")


def validate_decisions(decisions: dict) -> None:
    require(type(decisions) is dict and type(decisions.get("permissions")) is dict,
            "pilot decision sections missing")
    require(all(decisions["permissions"].get(key) is False
                for key in ("native_inference", "training", "protected_bank_access", "model_upload", "paid_services")),
            "CPU preparation must not grant execution permissions")
    require(decisions["budget"]["ceiling_per_trained_arm_per_seed"] == 131072 and
            decisions["budget"]["minimum_consumed_per_arm_per_seed"] == 130560 and
            decisions["adaptation"]["gradient_accumulation_sequences"] == 20 and
            decisions["adaptation"]["context_cap_tokens"] == 512,
            "frozen dose, accumulation or context cap changed")


def source_paths() -> list[Path]:
    return sorted([*PACKAGE.glob("*.py"), PACKAGE / "MATCHING_POLICY.json",
                   PACKAGE / "PILOT_DECISIONS.json", PACKAGE / "pilot_trainable_inventory.json"])


def source_hashes() -> dict:
    return {path.name: file_sha256(path) for path in source_paths()}


def snapshot_sources(bundle: Path, expected: dict) -> dict:
    snapshot = bundle / "code_snapshot"
    snapshot.mkdir()
    hashes = {}
    for path in source_paths():
        content = path.read_bytes()
        require(hashlib.sha256(content).hexdigest() == expected[path.name], "source changed during snapshot")
        with (snapshot / path.name).open("xb") as handle:
            handle.write(content)
        hashes[path.name] = hashlib.sha256(content).hexdigest()
    return hashes


def git_head() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PACKAGE, check=True,
                                capture_output=True, text=True, timeout=10)
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def bundle_file_hashes(bundle: Path) -> dict:
    return {path.relative_to(bundle).as_posix(): {"sha256": file_sha256(path), "bytes": path.stat().st_size}
            for path in sorted(bundle.rglob("*")) if path.is_file()}


def prepare_bundle(output: str | Path, *, model_dir: str | Path | None = None) -> dict:
    """Materialize immutable candidates; every exit remains native-unqualified."""
    bundle = create_bundle(output)
    watch = ResourceWatch()
    started_utc = datetime.now(timezone.utc).isoformat()
    status, reason = "CPU_PREPARATION_FAILED", "preparation did not complete"
    result = None
    plan = enrollment = decisions = policy = None
    rows, selected, arrays = [], [], {}
    tokenizer = None
    before = after = None
    sources_before, sources_after, snapshots = {}, {}, {}
    array_files_written = False
    schedule_written = False
    stages = []
    try:
        watch.check("freeze_registration_before_tokenization")
        policy = json.loads((PACKAGE / "MATCHING_POLICY.json").read_text(encoding="utf-8"))
        decisions = json.loads((PACKAGE / "PILOT_DECISIONS.json").read_text(encoding="utf-8"))
        validate_policy(policy)
        validate_decisions(decisions)
        sources_before = source_hashes()
        snapshots = snapshot_sources(bundle, sources_before)
        plan = build_balanced_plan(seed=policy["source_seed"], parent_count=policy["candidate_parents"])
        enrollment = freeze_enrollment(plan)
        require(len(enrollment["parents"]) == policy["enrolled_parents"] and
                enrollment["policy_sha256"] == digest(policy), "pre-tokenization enrollment or policy drift")
        for name, value in (("plan.json", plan), ("enrollment.json", enrollment),
                            ("pilot_decisions.json", decisions), ("matching_policy.json", policy)):
            write_json_once(bundle / name, value)
        stages.append("source_plan_and_enrollment_frozen_before_tokenization")
        watch.check("compile_and_validate_full_catalog")
        rows = compile_balanced_records(plan)
        validate_balanced_records(rows, plan)
        expected_ids = set(enrollment["record_ids"]["M1"] + enrollment["record_ids"]["M2"])
        selected = [row for row in rows if row["row_id"] in expected_ids]
        require(len(selected) == 640 and {row["row_id"] for row in selected} == expected_ids and
                all(row["split"] == "train" for row in selected), "complete enrolled TRAIN candidate pool required")
        write_jsonl_once(bundle / "candidate_records.jsonl", selected)
        for arm in ("M1", "M2"):
            arm_rows = [row for row in selected if row["arm"] == arm]
            require(len(arm_rows) == 320, "paired catalog row count drift")
            # Explicit allowlist; lineage, verification and oracle lists stay in
            # the separate candidate sidecar, never in student prompt objects.
            examples = [{"messages": copy.deepcopy(row["messages"]), "target": row["target"]} for row in arm_rows]
            write_jsonl_once(bundle / f"student_pool_{arm}.jsonl", examples)
        stages.append("full_catalog_validated_and_atomic_TRAIN_exports_written")
        watch.check("load_pinned_local_tokenizer")
        tokenizer = load_local_tokenizer(decisions_path=bundle / "pilot_decisions.json", model_dir=model_dir)
        before = tokenizer.verify_binding()
        for number, row in enumerate(selected, 1):
            watch.check(f"tokenize_registered_row_{number}_of_640")
            array = tokenizer.encode_example(row["messages"], row["target"])
            require(type(array) is dict and set(array) == ARRAY_FIELDS, "tokenizer array schema changed")
            arrays[row["row_id"]] = array
        after = tokenizer.verify_binding()
        require(before == after, "tokenizer binding changed during materialization")
        for arm in ("M1", "M2"):
            write_jsonl_once(bundle / f"token_pool_{arm}.jsonl",
                             [{"row_id": row["row_id"], **arrays[row["row_id"]]}
                              for row in selected if row["arm"] == arm])
        array_files_written = True
        stages.append("640_exact_token_arrays_materialized_with_unchanged_bindings")
        watch.check("match_frozen_TRAIN_schedule")
        result = match_schedule(selected, arrays, enrollment, decisions)
        require(type(result) is dict and result.get("status") in ("MATCHED_CPU_SCHEDULE", "BLOCKED_MATCHING"),
                "matcher returned an unsupported outcome")
        write_json_once(bundle / "schedule_result.json", result)
        schedule_written = True
        if result["status"] == "MATCHED_CPU_SCHEDULE":
            validate_schedule(result, selected, arrays, enrollment, decisions)
            write_json_once(bundle / "training_occurrences.json", {
                "schedule_sha256": result["sha256"], "row_ids": copy.deepcopy(result["schedules"]),
                "arrays_are_referenced_without_duplication": True, "training_admitted": False,
                "native_launch_authorized": False})
            status, reason = "MATCHED_CPU_SCHEDULE", "independent exact schedule checks passed; native qualification pending"
        else:
            status, reason = "BLOCKED_MATCHING", result.get("reason", "matching not established")
        stages.append("matching_outcome_preserved_without_changing_constraints")
        require(before == tokenizer.verify_binding(), "tokenizer binding changed during matching")
        watch.check("final_source_and_CPU_budget_check")
    except (Exception, KeyboardInterrupt) as exc:
        status = "CPU_PREPARATION_INTERRUPTED" if isinstance(exc, KeyboardInterrupt) else "CPU_PREPARATION_FAILED"
        reason = f"{type(exc).__name__}: {exc}"
    # Finalization preserves partial work even when a cooperative budget check
    # stops production. It never overwrites a completed artifact or admits it.
    finalization_errors = []
    if tokenizer is not None:
        try:
            after = tokenizer.verify_binding()
            require(before is not None and before == after, "final tokenizer binding mismatch")
        except Exception as exc:
            finalization_errors.append(f"tokenizer_postcheck: {type(exc).__name__}: {exc}")
    if tokenizer is not None and not array_files_written:
        for arm in ("M1", "M2"):
            try:
                path = bundle / f"token_pool_{arm}.jsonl"
                if not path.exists():
                    write_jsonl_once(path, [{"row_id": row["row_id"], **arrays[row["row_id"]]}
                                           for row in selected if row["arm"] == arm and row["row_id"] in arrays])
            except Exception as exc:
                finalization_errors.append(f"partial_arrays_{arm}: {type(exc).__name__}: {exc}")
    if not schedule_written:
        try:
            result = result or {"status": status, "reason": reason, "training_admitted": False,
                                "native_launch_authorized": False}
            write_json_once(bundle / "schedule_result.json", result)
        except Exception as exc:
            finalization_errors.append(f"schedule_receipt: {type(exc).__name__}: {exc}")
    try:
        sources_after = source_hashes()
        if sources_before and sources_before != sources_after:
            finalization_errors.append("production source or frozen registration changed during preparation")
    except Exception as exc:
        finalization_errors.append(f"source_postcheck: {type(exc).__name__}: {exc}")
    resource_record = watch.record()
    if not resource_record["cpu_budget_satisfied"]:
        finalization_errors.append("CPU resource budget or memory telemetry not established")
    if finalization_errors:
        status, reason = "CPU_PREPARATION_FAILED", reason + "; " + "; ".join(finalization_errors)
    receipt = {"schema_version": 1, "implementation": VERSION, "experiment_id": "SYNTH-CRS-001",
               "status": status, "stop_reason": reason, "started_at_utc": started_utc,
               "finished_at_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
               "source_head": git_head(), "source_identity": "HEAD plus exact working-tree hashes and immutable code snapshot",
               "source_hashes_before": sources_before, "source_hashes_after": sources_after,
               "code_snapshot_hashes": snapshots, "completed_stages": stages, "resources": resource_record,
               "matching_status": result.get("status") if type(result) is dict else None,
               "counts": {"full_catalog_records_validated": len(rows), "enrolled_parents": len(enrollment["parents"]) if enrollment else 0,
                          "candidate_registered_rows": len(selected), "token_arrays_materialized": len(arrays),
                          "student_pool_registered_row_ids_by_arm": {arm: sum(row["arm"] == arm for row in selected) for arm in ("M1", "M2")},
                          "distinct_student_example_hashes_by_arm": {arm: len({digest({"messages": row["messages"], "target": row["target"]})
                                                                             for row in selected if row["arm"] == arm}) for arm in ("M1", "M2")}},
               "tokenizer_binding_before": before, "tokenizer_binding_after": after,
               "tokenizer_binding_verified_unchanged": before is not None and before == after,
               "framework_modules_imported": sorted(name for name in sys.modules if name.split(".", 1)[0] in FRAMEWORK_ROOTS),
               "teacher_calls": 0, "model_loaded": False, "native_inference_performed": False,
               "training_performed": False, "protected_banks_accessed": False, "training_admitted": False,
               "native_launch_authorized": False, "native_qualification": False, "finalization_errors": finalization_errors}
    try:
        receipt["files"] = bundle_file_hashes(bundle)
        receipt["sha256"] = digest(receipt)
        write_json_once(bundle / "preparation_receipt.json", receipt)
    except Exception as exc:
        return {"status": "CPU_PREPARATION_FAILED", "output": str(bundle), "exit_code": 2,
                "reason": reason + f"; cannot preserve final receipt: {type(exc).__name__}: {exc}",
                "training_admitted": False, "native_launch_authorized": False}
    return {"status": status, "output": str(bundle), "exit_code": 0 if status == "MATCHED_CPU_SCHEDULE" else 2,
            "matching_status": receipt["matching_status"], "reason": reason,
            "receipt": str(bundle / "preparation_receipt.json"), "receipt_sha256": receipt["sha256"],
            "training_admitted": False, "native_launch_authorized": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="new direct child of this package's data directory")
    parser.add_argument("--model-dir", type=Path, help="existing absolute local tokenizer directory; exact pins remain mandatory")
    args = parser.parse_args(argv)
    try:
        report = prepare_bundle(args.output, model_dir=args.model_dir)
    except (Exception, KeyboardInterrupt) as exc:
        report = {"status": "CPU_PREPARATION_FAILED", "reason": f"{type(exc).__name__}: {exc}",
                  "exit_code": 2, "training_admitted": False, "native_launch_authorized": False}
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
