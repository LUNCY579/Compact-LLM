"""Bounded executable-inventory experiment; no model operations or qualification.

The native source/registry remain unchanged. Framework imports are deferred to
one contained observation worker, after exact source/review/resource admission.
Unknown launch/exit responses retain their reservation. This diagnostic does
not reopen the quarantined native milestone.
"""
from __future__ import annotations

import argparse
import ctypes as c
from ctypes import wintypes as w
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

from . import native
from .operations import OperationsStore, canonical, host_state_path

PACKAGE = native.PACKAGE
ROOT = native.ROOT
CONTRACT = PACKAGE / "contracts/OWNED_RUNTIME_IDENTITY_DIAGNOSTIC_V1.json"
KEY = "CRS-OWNED-IDENTITY-001"
SCOPE = "NO_MODEL_FRAMEWORK_IMPORT_IDENTITY_OBSERVATION"
MAX_LOG = 16 * 1024 * 1024


def forbidden_call(module, name, event):
    if event == "call":
        if module.startswith("torch.cuda") and name in {
            "is_available", "is_bf16_supported", "init", "_lazy_init", "device_count",
            "current_device", "set_device", "get_device_properties", "get_device_name",
            "mem_get_info", "synchronize", "empty_cache", "memory_allocated", "memory_reserved"}:
            return True
        if name == "__init__" and (module.startswith("torch.nn.modules")
                or module.startswith("transformers.models.qwen3.configuration")
                or module.startswith("transformers.models.qwen3.modeling")
                or module.startswith("transformers.tokenization_utils")):
            return True
        if module.startswith("safetensors") and name in {"load_file", "load_model", "save_file", "save_model"}:
            return True
    if event == "c_call":
        if module.startswith("torch") and (name.startswith("_cuda_") or name in {
            "tensor", "as_tensor", "from_numpy", "empty", "empty_like", "empty_strided",
            "zeros", "zeros_like", "ones", "ones_like", "full", "full_like", "arange",
            "range", "linspace", "logspace", "rand", "randn", "randint", "randperm",
            "eye", "load", "save"}):
            return True
        if module.startswith("safetensors") and name in {"safe_open", "load", "save"}:
            return True
    return False


def import_boundary(path):
    """Detect preregistered Python/C-call APIs before their body executes.

    C-extension internal calls and native threads are outside Python profiling.
    This instrumentation adds timing; it never certifies zero hidden C effects.
    """
    violations = []
    def profile(frame, event, arg):
        if event == "call":
            module, name = frame.f_globals.get("__name__", ""), frame.f_code.co_name
        elif event == "c_call":
            module, name = getattr(arg, "__module__", "") or "", getattr(arg, "__name__", "")
        else: return
        if forbidden_call(module, name, event):
            value = {"module": module, "name": name, "profile_event": event,
                     "thread_id": threading.get_ident(), "body_execution_blocked": True,
                     "internal_C_side_effect_coverage": False}
            violations.append(value)
            append(path, value)
            raise native.NativeBlocked("probe forbidden API reached before execution: " + module + "." + name)
    sys.setprofile(profile); threading.setprofile(profile)
    return profile, violations


def append(path, value):
    data = canonical({"wall_time": time.time(), "monotonic_time": time.monotonic(), **value}) + b"\n"
    native.require(sum(f.stat().st_size for f in path.parent.iterdir()
                       if f.is_file() and f.suffix in {".jsonl", ".log"}) + len(data) <= MAX_LOG,
                   "probe log ceiling")
    with path.open("ab") as out:
        out.write(data); out.flush(); os.fsync(out.fileno())


def normalized(path):
    return path.replace("\\", "/").casefold()


def classify_identity(executable, actual_sha256, registry):
    """Observation only: never extends an executable permission registry."""
    expected = registry.get(normalized(executable))
    return {"normalized_path": normalized(executable), "path_registered": expected is not None,
            "expected_sha256": expected, "actual_sha256": actual_sha256,
            "hash_matches": expected is not None and actual_sha256 == expected,
            "registry_mutated": False}


def registry(cfg):
    return {normalized(path): expected for path, expected in cfg["runtime_file_sha256"].items()
            if path.lower().endswith(".exe")}


class JobNotifications:
    """Exact Job completion port, observational and explicitly non-exhaustive.

    Microsoft documents non-guaranteed creation/exit delivery and PID reuse.
    A notification is never itself a live process identity or exit proof.
    """
    def __init__(self, job):
        self.kernel = job.kernel
        self.kernel.CreateIoCompletionPort.argtypes = [w.HANDLE, w.HANDLE, c.c_size_t, w.DWORD]
        self.kernel.CreateIoCompletionPort.restype = w.HANDLE
        self.kernel.GetQueuedCompletionStatus.argtypes = [w.HANDLE, c.POINTER(w.DWORD),
                                                          c.POINTER(c.c_size_t), c.POINTER(c.c_void_p), w.DWORD]
        self.kernel.GetQueuedCompletionStatus.restype = w.BOOL
        self.handle = self.kernel.CreateIoCompletionPort(w.HANDLE(-1), None, 0, 1)
        native.require(self.handle, "completion port unavailable")
        self.key = 0x43525301
        class Association(c.Structure):
            _fields_ = [("CompletionKey", c.c_void_p), ("CompletionPort", w.HANDLE)]
        association = Association(self.key, self.handle)
        try:
            native.require(self.kernel.SetInformationJobObject(job.handle, 7, c.byref(association),
                                                               c.sizeof(association)), "Job notification association failed")
        except BaseException:
            self.close(); raise

    def poll(self):
        records = []
        for _ in range(64):
            message, key, value = w.DWORD(), c.c_size_t(), c.c_void_p()
            ok = self.kernel.GetQueuedCompletionStatus(self.handle, c.byref(message), c.byref(key),
                                                       c.byref(value), 0)
            if not ok:
                native.require(value.value is None and c.get_last_error() == 258,
                               "completion-port observation error")
                break
            native.require(key.value == self.key, "foreign Job notification")
            records.append({"message": message.value, "pid_value": value.value,
                            "exact_job_key": key.value, "identity_not_established_by_notification": True})
        return records

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle); self.handle = None


def inventory(job, cfg):
    """Capture raw identities before the original strict guard can reject them."""
    import psutil
    job.kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
    job.kernel.OpenProcess.restype = w.HANDLE
    job.kernel.IsProcessInJob.argtypes = [w.HANDLE, w.HANDLE, c.POINTER(w.BOOL)]
    job.kernel.IsProcessInJob.restype = w.BOOL
    permitted, rows, rss = registry(cfg), [], 0
    ids = native.job_process_ids(job)
    for pid in ids:
        row = {"pid": pid, "query_error": None, "observed_at": time.time()}
        handle = None
        try:
            process = psutil.Process(pid)
            created, executable = process.create_time(), process.exe()
            handle = job.kernel.OpenProcess(0x1000, False, pid)
            native.require(handle, "process handle unavailable")
            member = w.BOOL()
            native.require(job.kernel.IsProcessInJob(handle, job.handle, c.byref(member)), "membership query failed")
            actual = native.sha(executable)
            native.require(process.create_time() == created and process.exe() == executable,
                           "sample PID identity drift")
            member_rss = process.memory_info().rss
            row.update(creation_time=created, executable=executable,
                       exact_owned_job_member=bool(member.value), rss_bytes=member_rss,
                       **classify_identity(executable, actual, permitted))
            rss += member_rss
        except BaseException as error:
            row["query_error"] = type(error).__name__ + ": " + str(error)
            row["original_notification_identity_inferred"] = False
        finally:
            if handle: job.kernel.CloseHandle(handle)
        rows.append(row)
    return {"job_pid_list": ids, "members": rows, "observed_rss_bytes": rss,
            "coverage_complete": False, "sampling_and_notifications_can_miss_short_lived_identities": True}


def checked_contract(expected):
    native.require(native.sha(CONTRACT) == expected, "probe contract drift")
    cfg = native.read_json(CONTRACT)
    native.require(cfg["scope"] == SCOPE and cfg["limits"]["wall_seconds"] == 60
                   and cfg["limits"]["runs"] == 1, "registered single bounded observation required")
    native.require(native.sha(native.CONTRACT) == cfg["preserved_native"]["native_contract_sha256"]
                   and native.sha(native.__file__) == cfg["preserved_native"]["native_source_sha256"],
                   "quarantined native source changed")
    for path, expected_sha in cfg["source_sha256"].items():
        native.require(native.sha(ROOT / path) == expected_sha, "probe source drift: " + path)
    native_cfg = native.read_json(native.CONTRACT)
    native.require(hashlib.sha256(canonical(registry(native_cfg))).hexdigest() == cfg["registry_sha256"],
                   "unchanged runtime registry binding required")
    for path, expected_sha in cfg["runtime_file_sha256"].items():
        native.require(native.sha(path) == expected_sha, "installed import source drift")
    return cfg, native_cfg


def checked_reviews(paths, expected):
    native.require(len(paths) == 2, "independent operations and scientific reviews required")
    roles, reviewers = set(), set()
    for path in paths:
        review = native.read_json(path)
        native.require(review.get("status") == "APPROVED_FOR_OWNED_IDENTITY_PROBE"
                       and review.get("binding") == expected, "probe review/source binding mismatch")
        reviewer = review.get("reviewer")
        native.require(type(reviewer) is str and reviewer.strip() and reviewer.strip() != "/root",
                       "identified non-builder reviewer required")
        roles.add(review.get("role")); reviewers.add(reviewer)
    native.require(roles == {"operations", "science"} and len(reviewers) == 2,
                   "separate reviewer responsibilities required")


def child(args, cfg, native_cfg):
    go = native.read_json(args.go)
    handle, consumed = go.get("own_job_query_handle"), False
    try:
        native.require(go["contract_sha256"] == args.contract_sha256 and go["worker_tree_pid"] > 0,
                       "probe GO binding required")
        import psutil
        ancestors = [p.pid for p in psutil.Process().parents()]
        native.require(go["supervisor_pid"] in ancestors and go["worker_tree_pid"] in ancestors + [os.getpid()],
                       "probe contained ancestry mismatch")
        store = OperationsStore(host_state_path(), args.output / "receipts")
        store._require_lease(go["owner"], go["fence"])
        consumed = type(handle) is int and 0 < handle < 2 ** 64
        witness = native.current_windows_job_witness(handle,
                  go["query_handle_target_pid"], go["query_handle_target_creation_time"])
    finally:
        if not consumed and type(handle) is int and 0 < handle < 2 ** 64:
            native.close_unconsumed_query_handle(handle)
    native.write_new(args.output / "CHILD_CONTAINMENT.json", witness)
    events = args.output / "CHILD_EVENTS.jsonl"
    def event(phase, **detail):
        store._require_lease(go["owner"], go["fence"])
        append(events, {"phase": phase, **detail})
    event("idle_begin", forbidden_imports_present=[m for m in ("torch", "transformers", "tokenizers", "safetensors") if m in sys.modules])
    time.sleep(2)
    for index in range(10):
        event("source_begin", index=index)
        observed = native.committed_snapshot()
        native.require(observed == go["source"], "contained source snapshot drift")
        if index in (0, 9):
            native.verify_sources(native_cfg)
            native.require(native.runtime_metadata() == native_cfg["runtime"], "contained installed runtime drift")
        event("source_end", index=index, source=observed)
        time.sleep(.05)
    event("torch_import_begin")
    profile, violations = import_boundary(args.output / "BOUNDARY_REFUSALS.jsonl")
    import torch
    event("torch_import_end", cuda_initialized=vars(torch.cuda).get("_initialized"))
    native.require(vars(torch.cuda).get("_initialized") is False and not violations
                   and sys.getprofile() is profile and threading.getprofile() is profile, "import boundary changed or CUDA initialized")
    event("transformers_package_import_begin")
    import transformers
    event("transformers_package_import_end")
    event("qwen_symbol_import_begin")
    from transformers import Qwen3Config, Qwen3ForCausalLM
    event("qwen_symbol_import_end", config_symbol=Qwen3Config.__module__ + "." + Qwen3Config.__name__,
          model_symbol=Qwen3ForCausalLM.__module__ + "." + Qwen3ForCausalLM.__name__,
          cuda_initialized=vars(torch.cuda).get("_initialized"),
          transitive_modules=[m for m in sorted(sys.modules) if m.split(".")[0] in {"torch", "transformers", "safetensors", "tokenizers"}])
    native.require(vars(torch.cuda).get("_initialized") is False and not violations
                   and sys.getprofile() is profile and threading.getprofile() is profile, "symbol import boundary changed or CUDA initialized")
    time.sleep(2)
    native.require(not violations and vars(torch.cuda).get("_initialized") is False
                   and sys.getprofile() is profile and threading.getprofile() is profile,
                   "post-import boundary changed or refused")
    sys.setprofile(None); threading.setprofile(None)
    event("all_registered_phases_returned", model_instances_requested=0, model_binary_reads_requested=0,
          tokenizer_instances_requested=0, tensor_operations_requested=0, model_calls_requested=0,
          native_qualified=False, original_failure_cause_established=False)
    store.close()
    return 0


def supervise(args, cfg, native_cfg):
    """One registered observational attempt; no automatic repeat or promotion."""
    import psutil
    native.require(os.name == "nt" and sys.executable == native_cfg["runtime"]["executable"],
                   "pinned Windows runtime required")
    source = native.committed_snapshot()
    native.require(not source["dirty"], "clean probe source required")
    binding = {"source_commit": source["commit"], "contract_sha256": args.contract_sha256,
               "probe_sha256": native.sha(__file__), "native_sha256": native.sha(native.__file__),
               "scope": SCOPE}
    checked_reviews(args.review, binding)
    native.verify_sources(native_cfg)
    native.require(native.runtime_metadata() == native_cfg["runtime"], "probe runtime drift")
    output = args.output.resolve()
    native.require(output.is_relative_to(PACKAGE / "results") and not output.exists(), "new CRS probe output required")
    telemetry = native.collect_telemetry(PACKAGE, desktop_identities=native_cfg["gpu_process_registry"])
    native.require(not native.admission_reasons(native_cfg, telemetry), "probe host admission closed")
    time.sleep(2)
    second = native.collect_telemetry(PACKAGE, desktop_identities=native_cfg["gpu_process_registry"])
    native.require(not native.admission_reasons(native_cfg, second), "second probe admission closed")
    output.mkdir(parents=True)
    store = OperationsStore(host_state_path(), output / "receipts")
    owner, lease = "CRS-IDENTITY-" + uuid.uuid4().hex, None
    process = job = notifications = None
    attempted = dispatched = reserved = False
    outcome, observation, reason, exit_proof = "FAILED", "UNRESOLVED_OR_INCOMPLETE_COVERAGE", None, None
    started = time.monotonic()
    stdout = stderr = None
    peak_rss = 0
    try:
        store.register_job(KEY, binding, args.epoch)
        native.require(store.get_job(KEY)["state"] == "PENDING", "single registered probe already attempted")
        store.ready(KEY)
        lease = store.acquire_lease(owner, 120)
        store.reserve(KEY, owner, lease["fence"], 60)
        reserved = True
        command = [sys.executable, "-B", "-m", "experiments.rineya_centralized_reasoning_system_v1.owned_runtime_identity_probe",
                   "--child", "--contract-sha256", args.contract_sha256, "--output", str(output),
                   "--go", str(output / "GO.json")]
        native.write_new(output / "LAUNCH_INTENT.json", {"command": command, "binding": binding,
                          "preflight": telemetry, "second_observation": second, "job_key": KEY,
                          "lease": lease, "scope": SCOPE, "model_dispatch": False,
                          "declared_environment": {key: (":4096:8" if key == "CUBLAS_WORKSPACE_CONFIG" else os.environ.get(key))
                              for key in cfg["environment"]["selected_identity_variables"]}})
        stdout, stderr = (output / "stdout.log").open("xb"), (output / "stderr.log").open("xb")
        native.write_new(output / "LAUNCH_ATTEMPT.json", {"attempted_at": time.time(), "job_key": KEY,
                          "command_sha256": hashlib.sha256(canonical(command)).hexdigest(), "owner": owner,
                          "fence": lease["fence"], "outcome": "PENDING_CREATION_RESPONSE"})
        attempted = True
        process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                  env=dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1",
                           CUBLAS_WORKSPACE_CONFIG=":4096:8"),
                  creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | 4)
        dispatched = True
        worker_identity = {"pid": process.pid, "creation_time": str(psutil.Process(process.pid).create_time()),
                           "command_hash": hashlib.sha256(canonical(command)).hexdigest()}
        store.mark_running(KEY, owner, lease["fence"], worker_identity)
        from experiments.opallo_native_collector_v1.supervisor import WindowsJob
        job = WindowsJob(process)
        notifications = JobNotifications(job)
        initial = inventory(job, native_cfg)
        append(output / "OBSERVATIONS.jsonl", {"phase": "before_resume", "inventory": initial,
                                               "notifications": notifications.poll()})
        native.require(native.job_process_ids(job) == [process.pid] and not psutil.Process(process.pid).children(recursive=True),
                       "probe must be assigned suspended before children")
        native.job_member_witness(job, native_cfg)
        native.resume_suspended_primary_thread(process)
        deadline, target = time.monotonic() + 3, []
        while time.monotonic() < deadline:
            current = inventory(job, native_cfg)
            append(output / "OBSERVATIONS.jsonl", {"phase": "runtime_target", "inventory": current,
                                                   "notifications": notifications.poll()})
            members = native.job_member_witness(job, native_cfg)
            target = [m for m in members if Path(m["executable"]).resolve() == Path(sys._base_executable).resolve()]
            if len(target) == 1: break
            time.sleep(.02)
        native.require(len(target) == 1, "one exact probe CPython target required")
        transfer = native.duplicate_owned_job_query_handle(job, target[0])
        native.write_new(output / "GO.json", {"contract_sha256": args.contract_sha256,
                         "owner": owner, "fence": lease["fence"], "worker_tree_pid": process.pid,
                         "supervisor_pid": os.getpid(), "source": source, **transfer})
        next_guard = next_gpu = 0.0
        while True:
            store._require_lease(owner, lease["fence"])
            now = time.monotonic()
            native.require(now - started <= 60, "probe wall ceiling")
            native.require(sum(f.stat().st_size for f in output.iterdir()
                           if f.is_file() and f.suffix in {".jsonl", ".log"}) <= MAX_LOG, "probe aggregate log ceiling")
            native.require(not (output / "BOUNDARY_REFUSALS.jsonl").exists(), "probe forbidden API boundary refused")
            current = inventory(job, native_cfg)
            append(output / "OBSERVATIONS.jsonl", {"phase": "observer", "inventory": current,
                                                   "notifications": notifications.poll()})
            native.require(len(current["job_pid_list"]) <= 8 and current["observed_rss_bytes"] <= 2 * native.GIB,
                           "probe process/RSS ceiling")
            native.require(psutil.virtual_memory().available / native.GIB >= 1.25
                           and psutil.disk_usage(output).free / native.GIB >= 4, "probe RAM/disk floor")
            peak_rss = max(peak_rss, current["observed_rss_bytes"])
            if now >= next_guard:
                append(output / "OBSERVATIONS.jsonl", {"phase": "strict_guard_begin"})
                try:
                    native.owned_job_rss_snapshot(job, native_cfg)
                except BaseException as error:
                    append(output / "OBSERVATIONS.jsonl", {"phase": "strict_guard_refusal",
                           "error": type(error).__name__ + ": " + str(error),
                           "inventory_after_refusal": inventory(job, native_cfg),
                           "notifications_after_refusal": notifications.poll()})
                    observation = "OBSERVED_STRICT_GUARD_REFUSAL_IDENTITY_REQUIRES_RAW_REVIEW"
                    raise
                append(output / "OBSERVATIONS.jsonl", {"phase": "strict_guard_return"})
                next_guard = now + .1
            if now >= next_gpu:
                gpu = native.collect_telemetry(output, desktop_identities=native_cfg["gpu_process_registry"])
                append(output / "OBSERVATIONS.jsonl", {"phase": "gpu_observation", "snapshot": gpu})
                owned_ids = set(native.job_process_ids(job))
                native.require(not any(p["pid"] in owned_ids for p in gpu["processes"]), "probe owned GPU side effect")
                native.require(not native.admission_reasons(native_cfg, gpu,
                               postimport_rss_gib=current["observed_rss_bytes"] / native.GIB), "probe live telemetry gate")
                next_gpu = now + .5
            if process.poll() is not None and not native.job_process_ids(job):
                native.require(process.returncode == 0, "contained probe did not return all phases")
                events = [json.loads(line) for line in (output / "CHILD_EVENTS.jsonl").read_text().splitlines()]
                native.require(events[-1]["phase"] == "all_registered_phases_returned"
                               and [e.get("index") for e in events if e["phase"] == "source_end"] == list(range(10))
                               and any(e["phase"] == "qwen_symbol_import_end" and e["cuda_initialized"] is False for e in events),
                               "probe planned phases incomplete")
                outcome, observation = "SUCCEEDED", "COMPLETED_SPECIFIED_NO_MODEL_PATHS_SAMPLED_COVERAGE"
                break
            time.sleep(.02)
    except BaseException as error:
        reason = type(error).__name__ + ": " + str(error)
    finally:
        if job:
            try: exit_proof = native.drain_owned_job(job)
            except BaseException as error:
                reason = reason or type(error).__name__ + ": " + str(error)
                outcome = "OUTCOME_UNKNOWN"
            job.close()
        if notifications: notifications.close()
        if process is not None and process.poll() is None:
            process.kill(); process.wait(timeout=5)
        for pipe in (stdout, stderr):
            if pipe: pipe.close()
        elapsed = time.monotonic() - started
        uncertain = (attempted and not dispatched) or (dispatched and not exit_proof)
        if uncertain: outcome = "OUTCOME_UNKNOWN"
        result = {"status": outcome, "observation": observation, "reason": reason,
                  "binding": binding, "worker_dispatched": dispatched, "launch_attempted": attempted,
                  "owned_tree_exit": exit_proof, "wall_seconds": elapsed, "peak_tree_rss_bytes": peak_rss,
                  "model_dispatch": False, "model_measurements": 0, "native_qualified": False,
                  "reader_qualified": False, "training_qualified": False,
                  "original_failure_cause_established": False, "native_quarantine_reopened": False,
                  "coverage_complete": False, "notifications_delivery_guaranteed": False}
        try:
            native.write_new(output / "PROBE_RESULT.json", result)
            if reserved:
                row = store.get_job(KEY)
                if uncertain or (dispatched and row["worker"] is None):
                    store.mark_unknown(KEY, "probe creation/identity/exit uncertain; retain reservation")
                else:
                    actual = elapsed if dispatched else 0
                    proof = {"observer": "CRS_OWNED_IDENTITY_PROBE", "request_hash": row["request_hash"],
                             "owner": owner, "fence": lease["fence"], "elapsed_native_seconds": actual}
                    if dispatched:
                        proof.update(worker_identity=row["worker"], process_exited=True,
                                     exit_observed_at=time.time(), owned_tree_exit=exit_proof)
                    else: proof.update(never_dispatched=True, launch_attempted=False)
                    reference = store.write_receipt("native_outcome_observation", KEY, proof)
                    store.reconcile(KEY, outcome, actual, {"native_outcome_proof_ref": reference,
                                    "probe_result_sha256": native.sha(output / "PROBE_RESULT.json")},
                                    owner=owner, fence=lease["fence"])
                    store.release_lease(owner, lease["fence"])
            elif lease: store.release_lease(owner, lease["fence"])
        except BaseException:
            if reserved and store.get_job(KEY)["state"] in {"RESERVED", "RUNNING", "OUTCOME_UNKNOWN"}:
                store.mark_unknown(KEY, "probe publication/reconciliation failed; preserve reservation")
            raise
        finally: store.close()
    print(json.dumps(result), flush=True)
    return 0 if outcome == "SUCCEEDED" else 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--supervise", action="store_true")
    mode.add_argument("--child", action="store_true")
    p.add_argument("--contract-sha256", required=True)
    p.add_argument("--review", type=Path, action="append", default=[])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epoch", default="CRS-EPOCH-001")
    p.add_argument("--go", type=Path)
    args = p.parse_args()
    cfg, native_cfg = checked_contract(args.contract_sha256)
    if args.child:
        native.require(args.go is not None and args.output.resolve().is_relative_to(PACKAGE / "results")
                       and args.go.resolve().parent == args.output.resolve(), "probe child/GO path binding")
        deadline = time.monotonic() + 5
        while not args.go.exists() and time.monotonic() < deadline: time.sleep(.02)
        native.require(args.go.exists(), "probe contained GO unavailable")
        return child(args, cfg, native_cfg)
    return supervise(args, cfg, native_cfg)


if __name__ == "__main__":
    sys.exit(main())
