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
import math
import ntpath
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

from . import native
from .operations import OperationsStore, TERMINAL, canonical, host_state_path

PACKAGE = native.PACKAGE
ROOT = native.ROOT
CONTRACT = PACKAGE / "contracts/OWNED_RUNTIME_IDENTITY_DIAGNOSTIC_002_V4.json"
DESKTOP_AUTHORITY_SHA = "479b6c5284da978c491a6477a9064f21b585fa29e9991bf84d75960c380d18f0"
DESKTOP_DISPOSITION_SHA = "b2a64e3bd3471d42db805d06cba44c1c4ffed6af79ff19dc9c2f4bb94954f49e"
DESKTOP_CORRECTION_SHA = "84f56a6f566f988ce402e9342676f7db6988607194e3b12a37a4d22ff5a5f939"
KEY = "CRS-OWNED-IDENTITY-002"
SCOPE = "NO_MODEL_FRAMEWORK_IMPORT_IDENTITY_OBSERVATION"
MAX_LOG = 16 * 1024 * 1024
HASH_SINGLE = 64 * 1024 * 1024
HASH_DISTINCT = 256 * 1024 * 1024
FATAL_EXIT = os._exit  # Capture before any third-party imports; never catchable.
BOUNDARY_EXIT_CODE = 86


def forbidden_call(module, name, event):
    if event == "call":
        if module.startswith("torch.serialization") and name in {"load", "save", "_load", "_legacy_load", "_save", "_legacy_save"}:
            return True
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
    fatal = FATAL_EXIT
    def profile(frame, event, arg):
        if event == "call":
            module, name = frame.f_globals.get("__name__", ""), frame.f_code.co_name
        elif event == "c_call":
            module, name = getattr(arg, "__module__", "") or "", getattr(arg, "__name__", "")
        else: return
        if type(module) is not str or type(name) is not str:
            try: append(path, {"profile_error": "invalid callable identity", "body_execution_blocked": True})
            finally: fatal(BOUNDARY_EXIT_CODE)
        if forbidden_call(module, name, event):
            value = {"module": module, "name": name, "profile_event": event,
                     "thread_id": threading.get_ident(), "body_execution_blocked": True,
                     "internal_C_side_effect_coverage": False}
            violations.append(value)
            try:
                append(path, value)
            finally:
                # A callback exception unsets profiling and can be swallowed by
                # importing code. Terminate even when refusal publication fails.
                fatal(BOUNDARY_EXIT_CODE)
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


class HashBudget:
    """Bound every read; count distinct file identities, including failed reads."""
    def __init__(self, single=HASH_SINGLE, distinct=HASH_DISTINCT):
        self.single, self.distinct = single, distinct
        self.identities, self.distinct_bytes, self.read_bytes = set(), 0, 0

    def hash(self, executable):
        path = Path(executable)
        before = path.stat()
        native.require(before.st_size <= self.single, "observed executable exceeds64MiB hash bound")
        identity = (normalized(str(path.resolve())), before.st_size, before.st_mtime_ns, before.st_ino)
        if identity not in self.identities:
            native.require(self.distinct_bytes + before.st_size <= self.distinct,
                           "distinct executable bytes exceed256MiB hash bound")
            self.identities.add(identity); self.distinct_bytes += before.st_size
        h, remaining = hashlib.sha256(), before.st_size
        with path.open("rb") as stream:
            opened = os.fstat(stream.fileno())
            native.require((opened.st_size, opened.st_mtime_ns, opened.st_ino) ==
                           (before.st_size, before.st_mtime_ns, before.st_ino), "executable changed before hash")
            while remaining:
                part = stream.read(min(2 * 1024 * 1024, remaining))
                native.require(part, "executable truncated while hashing")
                self.read_bytes += len(part); remaining -= len(part); h.update(part)
            after = os.fstat(stream.fileno())
        fresh = path.stat()
        expected = (before.st_size, before.st_mtime_ns, before.st_ino)
        native.require((after.st_size, after.st_mtime_ns, after.st_ino) == expected
                       and (fresh.st_size, fresh.st_mtime_ns, fresh.st_ino) == expected,
                       "executable file identity drift while hashing")
        return h.hexdigest()

    def snapshot(self):
        return {"distinct_file_identities": len(self.identities), "distinct_bytes": self.distinct_bytes,
                "actual_read_bytes": self.read_bytes, "individual_limit": self.single,
                "distinct_limit": self.distinct}


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


def inventory(job, cfg, hash_budget=None):
    """Capture raw identities before the original strict guard can reject them."""
    import psutil
    job.kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
    job.kernel.OpenProcess.restype = w.HANDLE
    job.kernel.IsProcessInJob.argtypes = [w.HANDLE, w.HANDLE, c.POINTER(w.BOOL)]
    job.kernel.IsProcessInJob.restype = w.BOOL
    permitted, rows, rss = registry(cfg), [], 0
    hash_budget = hash_budget if hash_budget is not None else HashBudget()
    began = time.monotonic()
    ids = native.job_process_ids(job)
    for pid in ids:
        row = {"pid": pid, "query_error": None, "observed_at": time.time()}
        handle = None
        try:
            process = psutil.Process(pid)
            created, executable = process.create_time(), process.exe()
            row.update(creation_time=created, executable=executable,
                       identity_observed_at=time.time(), **classify_identity(executable, None, permitted))
            handle = job.kernel.OpenProcess(0x1000, False, pid)
            native.require(handle, "process handle unavailable")
            member = w.BOOL()
            native.require(job.kernel.IsProcessInJob(handle, job.handle, c.byref(member)), "membership query failed")
            row.update(exact_owned_job_member=bool(member.value), membership_observed_at=time.time())
            member_rss = process.memory_info().rss
            row.update(rss_bytes=member_rss, rss_observed_at=time.time()); rss += member_rss
            actual = hash_budget.hash(executable)
            row.update(actual_sha256=actual, hash_matches=actual == row["expected_sha256"], hash_observed_at=time.time())
            fresh = psutil.Process(pid)  # exe/create_time are cached on a Process instance.
            native.require(fresh.create_time() == created and fresh.exe() == executable,
                           "sample PID identity drift")
            row["fresh_identity_verified_at"] = time.time()
        except BaseException as error:
            row["query_error"] = type(error).__name__ + ": " + str(error)
            row["original_notification_identity_inferred"] = False
        finally:
            if handle:
                try:
                    native.require(job.kernel.CloseHandle(handle), "observational process handle close failed")
                except BaseException as error:
                    row["close_error"] = type(error).__name__ + ": " + str(error)
        rows.append(row)
    return {"job_pid_list": ids, "members": rows, "observed_rss_bytes": rss,
            "coverage_complete": False, "sampling_and_notifications_can_miss_short_lived_identities": True,
            "observation_began_monotonic": began, "observation_ended_monotonic": time.monotonic(),
            "hash_budget": hash_budget.snapshot()}


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
    correction = cfg["delegated_boundary_correction_006"]
    native.require(correction["authority_path"] == "authorization/DELEGATED_DESKTOP_BOUNDARY_CORRECTION_006.json"
                   and correction["authority_sha256"] == DESKTOP_CORRECTION_SHA
                   and native.sha(PACKAGE / correction["authority_path"]) == DESKTOP_CORRECTION_SHA
                   and correction["cumulative_source_review_round"] == 6, "finite sixth correction binding required")
    checked_desktop_profile(cfg, native_cfg)
    return cfg, native_cfg


def exact_windows_path(value):
    native.require(type(value) is str and ntpath.isabs(value) and ntpath.normpath(value) == value
                   and not any(part in value for part in ("*", "?", "/")), "exact absolute desktop path required")
    return normalized(value)


def unique_json(path):
    def pairs(values):
        result = {}
        for key, value in values:
            native.require(key not in result, "duplicate desktop JSON key")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs)


def checked_desktop_profile(cfg, native_cfg):
    """Probe-only cooperative policy. Never changes either native registry."""
    reference = cfg["desktop_profile"]
    native.require(reference["path"] == "contracts/DESKTOP_GPU_IDENTITIES_PROBE002_V1.json", "fixed probe-only profile required")
    path = PACKAGE / reference["path"]
    native.require(native.sha(path) == reference["sha256"], "desktop profile drift")
    profile = unique_json(path)
    native.require(profile["schema_version"] == 1 and profile["probe_key"] == KEY,
                   "desktop profile scope mismatch")
    authority = profile["authority"]
    native.require(authority == cfg["owner_exception_desktop_profile"]["authority"]
                   and authority["path"] == "authorization/OWNER_EXCEPTION_DESKTOP_PROFILE_GRANTED_2026-10-04.json"
                   and authority["sha256"] == DESKTOP_AUTHORITY_SHA, "actual desktop authority binding required")
    native.require(native.sha(PACKAGE / authority["path"]) == DESKTOP_AUTHORITY_SHA, "desktop authority drift")
    grant = unique_json(PACKAGE / authority["path"])
    native.require(grant.get("authorization_received") is True
                   and grant["approved_current_scope"]["updated_prior_desktop_roles"] == 3
                   and grant["approved_current_scope"]["new_Chrome_Edge_rows_authorized"] is False,
                   "actual three-role authority required")
    disposition_ref = profile["context_disposition"]
    native.require(disposition_ref == {"path": "evidence/desktop_profile_integration_001/source/COOPERATIVE_CONTEXT_DISPOSITION.json",
                                      "sha256": DESKTOP_DISPOSITION_SHA}
                   and native.sha(PACKAGE / disposition_ref["path"]) == DESKTOP_DISPOSITION_SHA, "cooperative context disposition drift")
    disposition = unique_json(PACKAGE / disposition_ref["path"])
    native.require(disposition["authority_receipt_sha256"] == DESKTOP_AUTHORITY_SHA
                   and disposition["actual_noncompute_workload_proved"] is False, "cooperative context authority mismatch")
    native.require(profile["original_native_contract_sha256"] == cfg["preserved_native"]["native_contract_sha256"]
                   and profile["original_owned_child_registry_sha256"] == cfg["registry_sha256"]
                   and profile["original_desktop_registry"] == native_cfg["gpu_process_registry"], "original registries must remain unchanged")
    rows = profile["updated_rows"]
    native.require(type(rows) is list and len(rows) == 3 and {r["role_id"] for r in rows} == {"webview", "codex", "swift"}, "exactly three prior-role updates required")
    paths = set()
    for row in rows:
        key = exact_windows_path(row["path"])
        native.require(key not in paths and row["path"].endswith({"webview":"msedgewebview2.exe", "codex":"ChatGPT.exe", "swift":"codex-computer-use-swift.exe"}[row["role_id"]]), "duplicate or wrong desktop role")
        paths.add(key)
        native.require(type(row["sha256"]) is str and len(row["sha256"]) == 64
                       and set(row["sha256"]) <= set("0123456789abcdef")
                       and len(row["dependency_sha256"]) == 1, "exact desktop executable/dependency pins required")
        for dep, digest in row["dependency_sha256"].items():
            exact_windows_path(dep)
            native.require(type(digest) is str and len(digest) == 64 and set(digest) <= set("0123456789abcdef"), "invalid desktop dependency pin")
        native.require(row["owner_roots"] and len(set(row["owner_roots"])) == len(row["owner_roots"]), "explicit owner roots required")
        native.require(set(row["allowed_ancestry_paths"]) == {row["path"], *row["owner_roots"]}, "exact role ancestry paths required")
    policy = profile["context_policy"]
    native.require(policy["max_depth"] == 8 and policy["freshness_seconds"] == 5
                   and policy["actual_noncompute_workload_proved"] is False
                   and policy["hidden_inprocess_GPU_operations"] == "UNMEASURED"
                   and policy["inspection_pins_are_permissions"] is False, "bounded cooperative context policy required")
    pins = profile["inspection_only_ancestor_sha256"]
    allowed_pins = {row["path"]: row["sha256"] for row in rows}
    for anchor in profile["inspection_only_original_anchors"]:
        native.require(anchor in native_cfg["gpu_process_registry"], "inspection anchor must be an unchanged original desktop identity")
        native.require(any(a["path"] == anchor["path"] and a["sha256"] == anchor["sha256"] for a in disposition["inspection_only_anchors"]), "undeclared inspection-only anchor")
        allowed_pins[anchor["path"]] = anchor["sha256"]
    native.require(pins == allowed_pins and len(profile["inspection_only_original_anchors"]) == 2, "inspection-only trust expansion refused")
    for ancestor, digest in pins.items():
        exact_windows_path(ancestor)
        native.require(type(digest) is str and len(digest) == 64 and set(digest) <= set("0123456789abcdef"), "invalid inspection-only ancestor pin")
    for row in rows:
        for root in row["owner_roots"]:
            native.require(root in pins, "owner root must have an inspection-only pin")
    return profile


def stable_desktop_hash(path, expected):
    """No verdict cache: every observation rehashes selected physical bytes."""
    before = Path(path).stat()
    actual = native.sha(path)
    after = Path(path).stat()
    native.require((before.st_size, before.st_mtime_ns, before.st_ino) ==
                   (after.st_size, after.st_mtime_ns, after.st_ino) and actual == expected, "desktop file hash/physical identity drift")
    return actual


def process_principal(pid):
    """Actual Windows token SID and session, without command line/environment access."""
    native.require(os.name == "nt", "Windows process token required")
    kernel, adv = c.WinDLL("kernel32", use_last_error=True), c.WinDLL("advapi32", use_last_error=True)
    kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]; kernel.OpenProcess.restype = w.HANDLE
    kernel.CloseHandle.argtypes = [w.HANDLE]; kernel.CloseHandle.restype = w.BOOL
    adv.OpenProcessToken.argtypes = [w.HANDLE, w.DWORD, c.POINTER(w.HANDLE)]; adv.OpenProcessToken.restype = w.BOOL
    adv.GetTokenInformation.argtypes = [w.HANDLE, c.c_int, c.c_void_p, w.DWORD, c.POINTER(w.DWORD)]; adv.GetTokenInformation.restype = w.BOOL
    adv.ConvertSidToStringSidW.argtypes = [c.c_void_p, c.POINTER(w.LPWSTR)]; adv.ConvertSidToStringSidW.restype = w.BOOL
    kernel.LocalFree.argtypes = [c.c_void_p]; kernel.LocalFree.restype = c.c_void_p
    kernel.ProcessIdToSessionId.argtypes = [w.DWORD, c.POINTER(w.DWORD)]; kernel.ProcessIdToSessionId.restype = w.BOOL
    handle = kernel.OpenProcess(0x1000, False, pid); token = w.HANDLE(); sid = w.LPWSTR()
    native.require(bool(handle), "process token owner unavailable")
    try:
        native.require(bool(adv.OpenProcessToken(handle, 8, c.byref(token))), "process token unavailable")
        needed = w.DWORD(); adv.GetTokenInformation(token, 1, None, 0, c.byref(needed))
        native.require(0 < needed.value <= 65536, "bounded token owner required")
        buffer = c.create_string_buffer(needed.value)
        native.require(bool(adv.GetTokenInformation(token, 1, buffer, needed, c.byref(needed))), "token owner query failed")
        native.require(bool(adv.ConvertSidToStringSidW(c.cast(buffer, c.POINTER(c.c_void_p))[0], c.byref(sid))), "token SID unavailable")
        session = w.DWORD()
        native.require(bool(kernel.ProcessIdToSessionId(pid, c.byref(session))), "process session unavailable")
        return {"sid_sha256": hashlib.sha256(sid.value.encode("utf-8")).hexdigest(), "session": session.value}
    finally:
        if sid: kernel.LocalFree(c.cast(sid, c.c_void_p))
        if token: kernel.CloseHandle(token)
        kernel.CloseHandle(handle)


def validate_desktop_context(profile, row, witness, principal, *, now):
    native.require(type(principal) is dict and set(principal) == {"sid_sha256", "session"}
                   and type(principal["sid_sha256"]) is str and len(principal["sid_sha256"]) == 64
                   and type(principal["session"]) is int and principal["session"] > 0, "interactive SID/session unavailable")
    native.require(witness.get("activity") == "PERMITTED_COOPERATIVE_DESKTOP_CONTEXT", "unknown or declared compute context refused")
    native.require(witness.get("profile_sha256") == hashlib.sha256(canonical(profile)).hexdigest()
                   and witness.get("authority_sha256") == DESKTOP_AUTHORITY_SHA, "desktop context provenance drift")
    began, completed = witness["began_at"], witness["completed_at"]
    native.require(all(type(v) in (int, float) and math.isfinite(v) for v in (began, completed, now))
                   and 0 <= now - began <= 5 and began <= completed <= now, "desktop context stale/future/overrun")
    chain = witness["chain"]
    native.require(type(chain) is list and 1 <= len(chain) <= 8 and len({p["pid"] for p in chain}) == len(chain), "complete bounded noncyclic desktop ancestry required")
    native.require(chain[0]["executable"] == row["path"] and chain[0]["sha256"] == row["sha256"], "wrong desktop context executable")
    pins = profile["inspection_only_ancestor_sha256"]
    for index, member in enumerate(chain):
        native.require(member["executable"] in row["allowed_ancestry_paths"], "unrelated desktop ancestry context")
        native.require(member["principal"] == principal and member["stable_after_checks"] is True, "desktop owner/session/PID/parent drift")
        native.require(type(member["creation_time"]) in (int, float) and math.isfinite(member["creation_time"]), "desktop creation identity missing")
        if index:
            child = chain[index-1]
            native.require(child["ppid"] == member["pid"] and member["creation_time"] <= child["creation_time"]
                           and pins.get(member["executable"]) == member["sha256"], "unknown/drifted desktop parent")
    native.require(chain[-1]["executable"] in row["owner_roots"] and pins.get(chain[-1]["executable"]) == chain[-1]["sha256"], "unowned desktop context")
    boundary = witness.get("owner_boundary")
    native.require(type(boundary) is dict and boundary.get("attained") is True
                   and boundary.get("owner_pid") == chain[-1]["pid"]
                   and boundary.get("owner_ppid") == chain[-1]["ppid"]
                   and boundary.get("relation_rechecked") is True, "actual stable owner boundary required")
    codex_path = next(r["path"] for r in profile["updated_rows"] if r["role_id"] == "codex")
    reason, external = boundary.get("reason"), boundary.get("external_parent")
    if chain[-1]["executable"] != codex_path:
        native.require(reason == "NAMED_OWNER_ROOT" and external is None, "named owner-root boundary mismatch")
    else:
        native.require(type(external) is dict and external.get("pid") == chain[-1]["ppid"]
                       and type(external["pid"]) is int and external["pid"] > 0, "Codex external-parent relation missing")
        if reason == "OBSERVED_ABSENT_EXTERNAL_PARENT":
            native.require(external.get("state") == "ABSENT" and external.get("absence_rechecked") is True,
                           "external parent absence not rechecked")
        else:
            native.require(reason == "OBSERVED_NONFAMILY_PARENT" and external.get("state") == "LIVE"
                           and type(external.get("executable")) is str and external["executable"] != codex_path
                           and type(external.get("creation_time")) in (int, float)
                           and math.isfinite(external["creation_time"])
                           and external["creation_time"] <= chain[-1]["creation_time"]
                           and external.get("stable_after_checks") is True, "complete Codex application boundary not observed")
    return True


def collect_desktop_context(profile, row, pid, principal):
    import psutil
    began, chain, boundary = time.time(), [], None
    pins = profile["inspection_only_ancestor_sha256"]
    for depth in range(8):
        process = psutil.Process(pid)
        created, executable, parent = process.create_time(), process.exe(), process.ppid()
        expected = row["sha256"] if depth == 0 else pins.get(executable)
        native.require(expected is not None, "unregistered inspection-only parent")
        actual = stable_desktop_hash(executable, expected)
        member = {"pid": pid, "creation_time": created, "executable": executable, "ppid": parent,
                  "sha256": actual, "principal": process_principal(pid), "stable_after_checks": False}
        chain.append(member)
        if executable in row["owner_roots"]:
            # Retain all same-executable Codex ancestors; stop at its app boundary.
            boundary = {"attained": True, "owner_pid": pid, "owner_ppid": parent,
                        "relation_rechecked": False, "external_parent": None}
            if executable != next(r["path"] for r in profile["updated_rows"] if r["role_id"] == "codex"):
                boundary["reason"] = "NAMED_OWNER_ROOT"; break
            native.require(parent > 0, "Codex application boundary unavailable")
            try:
                outside = psutil.Process(parent)
                outside_created, outside_executable = outside.create_time(), outside.exe()
            except psutil.NoSuchProcess:
                boundary.update(reason="OBSERVED_ABSENT_EXTERNAL_PARENT",
                                external_parent={"pid": parent, "state": "ABSENT", "absence_rechecked": False})
                break
            if outside_executable != executable:
                native.require(outside_created <= created, "external parent PID reused before owner-boundary observation")
                boundary.update(reason="OBSERVED_NONFAMILY_PARENT", external_parent={"pid": parent,
                                "state": "LIVE", "creation_time": outside_created, "executable": outside_executable,
                                "stable_after_checks": False})
                break
            boundary = None
        native.require(parent > 0, "orphaned desktop helper")
        pid = parent
    native.require(boundary is not None, "desktop ancestry depth exhausted before actual app boundary")
    checked_paths = {member["executable"] for member in chain}
    for candidate in profile["updated_rows"]:
        if candidate["path"] in checked_paths:
            for dep, digest in candidate["dependency_sha256"].items(): stable_desktop_hash(dep, digest)
    for member in chain:
        fresh = psutil.Process(member["pid"])
        member["stable_after_checks"] = (fresh.create_time() == member["creation_time"] and fresh.exe() == member["executable"]
                                        and fresh.ppid() == member["ppid"] and process_principal(member["pid"]) == member["principal"])
    external = boundary["external_parent"]
    if boundary["reason"] == "OBSERVED_ABSENT_EXTERNAL_PARENT":
        try: psutil.Process(external["pid"]).create_time()
        except psutil.NoSuchProcess: external["absence_rechecked"] = True
        else: native.require(False, "previously absent external parent PID reused during checks")
    elif boundary["reason"] == "OBSERVED_NONFAMILY_PARENT":
        fresh = psutil.Process(external["pid"])
        external["stable_after_checks"] = fresh.create_time() == external["creation_time"] and fresh.exe() == external["executable"]
    boundary["relation_rechecked"] = chain[-1]["stable_after_checks"]
    witness = {"began_at": began, "completed_at": time.time(), "chain": chain,
               "owner_boundary": boundary,
               "profile_sha256": hashlib.sha256(canonical(profile)).hexdigest(), "authority_sha256": DESKTOP_AUTHORITY_SHA,
               "activity": "PERMITTED_COOPERATIVE_DESKTOP_CONTEXT", "actual_noncompute_workload_proved": False,
               "hidden_inprocess_GPU_operations": "UNMEASURED"}
    validate_desktop_context(profile, row, witness, principal, now=time.time())
    return witness


def collect_probe_telemetry(output, cfg, native_cfg):
    import psutil
    profile = checked_desktop_profile(cfg, native_cfg)
    # Original helper sees only its preserved original registry. Updated rows are
    # removed from unclassified only after current complete ownership checks.
    telemetry = native.collect_telemetry(output, desktop_identities=native_cfg["gpu_process_registry"])
    principal = process_principal(os.getpid())
    rows = {normalized(r["path"]): r for r in profile["updated_rows"]}
    telemetry["desktop_context_witnesses"] = []
    telemetry["desktop_profile_sha256"] = cfg["desktop_profile"]["sha256"]
    for process in telemetry["processes"]:
        row = rows.get(normalized(process["name"]))
        if row is None: continue
        pid = process["pid"]
        try:
            native.require(pid not in telemetry["competing_gpu_pids"] and pid not in telemetry["unresolved_gpu_pids"], "known compute or unresolved desktop PID")
            witness = collect_desktop_context(profile, row, pid, principal)
            telemetry["desktop_context_witnesses"].append({"pid": pid, "witness": witness})
            process.update(role=row["role"], executable_sha256=row["sha256"], cooperative_context_verified=True)
            telemetry["unclassified_gpu_pids"] = [p for p in telemetry["unclassified_gpu_pids"] if p != pid]
        except (OSError, psutil.Error, native.NativeBlocked) as error:
            if pid not in telemetry["unclassified_gpu_pids"]: telemetry["unclassified_gpu_pids"].append(pid)
            telemetry["desktop_context_witnesses"].append({"pid": pid, "unavailable": type(error).__name__ + ": " + str(error)})
    telemetry["desktop_checks_completed_at"] = time.time()
    telemetry["actual_noncompute_workload_proved"] = False
    telemetry["hidden_inprocess_GPU_operations"] = "UNMEASURED"
    return telemetry


def reviewer_identity(value):
    """Canonical local reviewer leaf, with only bare or /root aliases."""
    native.require(type(value) is str, "reviewer identity must be text")
    parts = [part.strip().casefold() for part in value.strip().replace("\\", "/").split("/") if part.strip()]
    native.require(len(parts) == 1 or (len(parts) == 2 and parts[0] == "root"), "unknown reviewer namespace")
    native.require(parts and parts[-1] in {"exception_science", "exception_ops"},
                   "owner exception001 requires a registered non-builder reviewer")
    return parts[-1]


def checked_reviews(paths, expected, expected_roles=None):
    native.require(len(paths) == 2, "independent operations and scientific reviews required")
    expected_roles = expected_roles if expected_roles is not None else native.read_json(CONTRACT)["independent_reviewers_required"]
    native.require(set(expected_roles) == {"science", "operations"}, "frozen exact reviewer role map required")
    required = {role: reviewer_identity(name) for role, name in expected_roles.items()}
    native.require(required == {"science": "exception_science", "operations": "exception_ops"},
                   "owner exception001 exact reviewer policy required")
    roles, reviewers = set(), set()
    for path in paths:
        review = native.read_json(path)
        native.require(review.get("status") == "APPROVED_FOR_OWNED_IDENTITY_PROBE"
                       and review.get("binding") == expected, "probe review/source binding mismatch")
        reviewer = reviewer_identity(review.get("reviewer")); role = review.get("role")
        native.require(role in required and reviewer == required[role], "reviewer does not match frozen exact role")
        roles.add(role); reviewers.add(reviewer)
    native.require(roles == {"operations", "science"} and len(reviewers) == 2,
                   "separate reviewer responsibilities required")


def probe_binding(contract_sha256, source):
    return {"source_commit": source["commit"], "contract_sha256": contract_sha256,
            "probe_sha256": native.sha(__file__), "native_sha256": native.sha(native.__file__), "scope": SCOPE,
            "desktop_profile_sha256": native.read_json(CONTRACT)["desktop_profile"]["sha256"],
            "desktop_authority_sha256": DESKTOP_AUTHORITY_SHA}


def recover_finalization(output, store, expected_binding):
    """Recover publication only from terminal durable evidence; never dispatch."""
    output = Path(output)
    row = store.get_job(KEY)
    native.require(row["state"] in TERMINAL and row["actual_seconds"] is not None
                   and row["payload"] == expected_binding, "terminal known-cost same-binding job required")
    pending = native.read_json(output / "PROBE_RESULT.json")
    native.require(pending.get("binding") == expected_binding and pending.get("status") == row["state"],
                   "pending result and terminal scheduler differ")
    evidence = row["result"]["evidence"]
    native.require(evidence.get("probe_result_sha256") == native.sha(output / "PROBE_RESULT.json"), "terminal result bytes drift")
    proof = store.verify_receipt(evidence["native_outcome_proof_ref"])
    native.require(proof["kind"] == "native_outcome_observation" and proof["key"] == KEY
                   and proof["payload"]["request_hash"] == row["request_hash"]
                   and proof["payload"]["elapsed_native_seconds"] == row["actual_seconds"], "terminal exit/cost proof drift")
    lease = store.db.execute("SELECT * FROM leases WHERE kind='model'").fetchone()
    if lease is not None and (lease["owner"], lease["fence"]) == (row["owner"], row["fence"]) and lease["status"] == "LIVE":
        # Existing exit/cost authority only; this neither acquires a lease nor
        # admits another job. Expired/active-other-work cases reject normally.
        store.release_lease(row["owner"], row["fence"])
    released = [json.loads(event["detail"]) for event in store.db.execute(
        "SELECT detail FROM events WHERE kind='LEASE_RELEASED' AND key='model'")]
    native.require({"owner": row["owner"], "fence": row["fence"]} in released,
                   "original lease release evidence required; reconcile explicitly before recovery")
    payload = {"status": row["state"], "job_key": KEY, "binding": expected_binding,
               "scheduler_state": row["state"], "probe_result_ref": {"path": str(output / "PROBE_RESULT.json"),
                   "sha256": native.sha(output / "PROBE_RESULT.json")},
               "actual_charged_seconds": row["actual_seconds"], "native_outcome_proof_ref": evidence["native_outcome_proof_ref"],
               "recovery_dispatches": 0, "original_lease_reconciled": True, "model_qualification_granted": False}
    path = output / "FINALIZATION_RECEIPT.json"
    if path.exists(): native.require(native.read_json(path) == payload, "immutable finalization receipt differs")
    else: native.write_new(path, payload)
    return payload


def record_finalization_failure(output, store, binding, error, *, reserved):
    """Report current durable truth; historical admission is not held budget."""
    notes = [{"action": "result_or_accounting_publication", "error": type(error).__name__ + ": " + str(error)}]
    scheduler_state, actual_charge, held, lease_state = None, None, None, None
    try:
        row = store.get_job(KEY)
        if reserved and row["state"] in {"RESERVED", "RUNNING", "OUTCOME_UNKNOWN"}:
            store.mark_unknown(KEY, "probe publication/reconciliation failed; preserve reservation")
            row = store.get_job(KEY)
        scheduler_state, actual_charge = row["state"], row["actual_seconds"]
        held = row["state"] in {"RESERVED", "RUNNING", "OUTCOME_UNKNOWN"}
        lease_row = store.db.execute("SELECT * FROM leases WHERE kind='model'").fetchone()
        lease_state = None if lease_row is None else lease_row["status"]
    except BaseException as secondary:
        notes.append({"action": "read_or_mark_scheduler_state", "error": type(secondary).__name__ + ": " + str(secondary)})
    known = scheduler_state in TERMINAL and actual_charge is not None
    payload = {"status": "PUBLICATION_FAILED_KNOWN_SCHEDULER_OUTCOME" if known else "OUTCOME_UNKNOWN",
               "errors": notes, "binding": binding, "scheduler_state": scheduler_state,
               "actual_charged_seconds": actual_charge, "lease_status": lease_state,
               "reservation_retained": held, "recovery_dispatches": 0,
               "recovery": "recover existing finalization only; never retry worker"}
    try: native.write_new(Path(output) / "FINALIZATION_ERROR.json", payload)
    except BaseException as secondary:
        print("probe finalization evidence unavailable: " + type(secondary).__name__, file=sys.stderr, flush=True)
    return payload


def authority(go):
    fields = ("job_key", "epoch", "request_hash", "binding", "owner", "fence", "worker_identity",
              "runtime_identity", "source", "review_refs", "launch_intent_ref")
    return {k: go[k] for k in fields}


def require_probe_job(store, go, args):
    """A live lease alone never licenses this child's imports."""
    native.require(go.get("job_key") == KEY and go.get("epoch") == args.epoch
                   and go.get("contract_sha256") == args.contract_sha256,
                   "probe scheduled job/epoch/contract mismatch")
    store._require_lease(go["owner"], go["fence"])
    row = store.get_job(KEY)
    native.require(row["state"] == "RUNNING" and row["reserved_seconds"] == 60
                   and (row["owner"], row["fence"], row["epoch"]) == (go["owner"], go["fence"], go["epoch"])
                   and row["request_hash"] == go["request_hash"] and row["payload"] == go["binding"]
                   and row["worker"] == go["worker_identity"], "child is not this exact reserved RUNNING probe")
    native.require(go["source"].get("dirty") is False
                   and go["binding"] == probe_binding(args.contract_sha256, go["source"]), "probe source binding mismatch")
    return row


def verify_refs(refs):
    paths = []
    for ref in refs:
        native.require(type(ref) is dict and set(ref) == {"path", "sha256"}, "immutable reference required")
        path = Path(ref["path"]).resolve()
        native.require(path.is_relative_to(PACKAGE) and native.sha(path) == ref["sha256"], "bound reference drift")
        paths.append(path)
    return paths


def read_jsonl(path):
    path = Path(path)
    native.require(path.stat().st_size <= MAX_LOG, "bounded journal size required before read")
    with path.open("rb") as stream: raw = stream.read(MAX_LOG + 1)
    native.require(0 < len(raw) <= MAX_LOG and raw.endswith(b"\n"), "bounded complete journal required")
    def unique(pairs):
        value = {}
        for key, item in pairs:
            native.require(key not in value, "duplicate journal key"); value[key] = item
        return value
    values = [json.loads(line, object_pairs_hook=unique) for line in raw.splitlines()]
    canonical(values)
    return values


def boundary_state(torch, profile):
    return {"cuda_initialized": vars(torch.cuda).get("_initialized"),
            "profile_current_thread": sys.getprofile() is profile,
            "profile_new_python_threads": threading.getprofile() is profile,
            "internal_C_side_effect_coverage": False, "generic_C_type_constructor_coverage": False}


def validate_child_evidence(output, go, native_cfg):
    """Complete immutable phase receipt, not a few caller success markers."""
    output = Path(output); events = read_jsonl(output / "CHILD_EVENTS.jsonl")
    native.require(native.read_json(output / "GO.json") == go, "saved GO differs from admitted supervisor context")
    intent = native.read_json(output / "LAUNCH_INTENT.json")
    native.require(go["launch_intent_ref"] == {"path": str(output / "LAUNCH_INTENT.json"),
                                               "sha256": native.sha(output / "LAUNCH_INTENT.json")}
                   and intent.get("binding") == go["binding"] and intent.get("job_key") == KEY
                   and intent.get("review_refs") == go["review_refs"], "saved dispatch authority corrupt")
    expected_authority = hashlib.sha256(canonical(authority(go))).hexdigest()
    expected_phases = ["idle_begin", "idle_end"]
    for _ in range(10): expected_phases += ["source_begin", "source_end"]
    expected_phases += ["torch_import_begin", "boundary_armed", "torch_import_end",
                       "transformers_package_import_begin", "transformers_package_import_end",
                       "qwen_symbol_import_begin", "qwen_symbol_import_end",
                       "post_import_begin", "post_import_end", "all_registered_phases_returned"]
    native.require([e.get("phase") for e in events] == expected_phases, "complete ordered child schedule required")
    previous = None
    for event in events:
        native.require(event.get("authority_sha256") == expected_authority
                       and type(event.get("wall_time")) in (int, float)
                       and type(event.get("monotonic_time")) in (int, float)
                       and math.isfinite(event["wall_time"]) and math.isfinite(event["monotonic_time"]), "child authority/time corrupt")
        native.require(previous is None or event["monotonic_time"] >= previous, "child event order/time drift")
        previous = event["monotonic_time"]
    native.require(events[0]["forbidden_imports_present"] == []
                   and events[1]["monotonic_time"] - events[0]["monotonic_time"] >= 2, "clean idle baseline incomplete")
    for index in range(10):
        begin, end = events[2 + 2 * index:4 + 2 * index]
        native.require(type(begin.get("index")) is int and begin["index"] == index
                       and type(end.get("index")) is int and end["index"] == index
                       and end.get("source") == go["source"], "source schedule identity drift")
        checked = index in (0, 9)
        native.require(end.get("verify_sources_returned") is checked
                       and end.get("runtime_metadata") == (native_cfg["runtime"] if checked else None), "exact10/2/2 source evidence missing")
    for event in events:
        if event["phase"] in {"torch_import_end", "transformers_package_import_end", "qwen_symbol_import_end", "post_import_end"}:
            native.require(event.get("boundary") == {"cuda_initialized": False, "profile_current_thread": True,
                           "profile_new_python_threads": True, "internal_C_side_effect_coverage": False,
                           "generic_C_type_constructor_coverage": False}, "complete import boundary evidence missing")
    armed = events[23]
    native.require(armed.get("profile_current_thread") is True and armed.get("profile_new_python_threads") is True,
                   "profile not armed before import")
    symbol = next(e for e in events if e["phase"] == "qwen_symbol_import_end")
    native.require(symbol.get("config_symbol") == "transformers.models.qwen3.configuration_qwen3.Qwen3Config"
                   and symbol.get("model_symbol") == "transformers.models.qwen3.modeling_qwen3.Qwen3ForCausalLM"
                   and type(symbol.get("transitive_modules")) is list
                   and symbol["transitive_modules"] == sorted(set(symbol["transitive_modules"]))
                   and {"torch", "transformers"} <= set(symbol["transitive_modules"]), "pinned symbol/module evidence corrupt")
    post_begin, post_end = events[-3:-1]
    native.require(post_end["monotonic_time"] - post_begin["monotonic_time"] >= 2, "post-import observation incomplete")
    final = events[-1]
    for name in ("model_instances_requested", "model_binary_reads_requested", "tokenizer_instances_requested",
                 "tensor_operations_requested", "model_calls_requested"):
        native.require(type(final.get(name)) is int and final[name] == 0, "zero requested-operation evidence corrupt")
    native.require(final.get("native_qualified") is False and final.get("original_failure_cause_established") is False,
                   "probe cannot qualify or identify old cause")
    native.require(events[-1]["monotonic_time"] - events[0]["monotonic_time"] <= 60,
                   "child phase evidence exceeds registered wall envelope")
    containment = native.read_json(output / "CHILD_CONTAINMENT.json")
    native.require(containment.get("authority_sha256") == expected_authority
                   and containment.get("go_sha256") == native.sha(output / "GO.json")
                   and containment.get("pid") == go["runtime_identity"]["pid"]
                   and containment.get("creation_time") == go["runtime_identity"]["creation_time"]
                   and containment.get("query_handle_closed_before_framework") is True
                   and containment.get("query_handle_inheritable") is False
                   and containment.get("query_handle_access") == "JOB_OBJECT_QUERY_ONLY",
                   "bound child containment evidence missing")
    native.validate_owned_job_flags(containment.get("exact_owned_limit_flags"), containment.get("in_exact_supervisor_job"))
    native.require(not (output / "BOUNDARY_REFUSALS.jsonl").exists(), "boundary refusal cannot complete")
    refs = {name: {"path": str(output / name), "sha256": native.sha(output / name)} for name in
            ("CHILD_EVENTS.jsonl", "CHILD_CONTAINMENT.json", "GO.json", "LAUNCH_INTENT.json", "LAUNCH_ATTEMPT.json")}
    return {"authority_sha256": expected_authority, "artifact_refs": refs, "phase_count": len(events),
            "source_snapshot_calls": 10, "verify_sources_calls": 2, "runtime_metadata_calls": 2,
            "requested_model_calls": 0, "hidden_internal_operations_attested_zero": False}


class ResourceWatch:
    """Independent deadline and raw RSS watcher; never hashes or samples GPU."""
    def __init__(self, job, *, started, output, seconds=60, sample=None):
        self.job, self.started, self.output, self.seconds = job, started, Path(output), seconds
        self.sample = sample or self.raw_sample
        self.stop = threading.Event(); self.lock = threading.Lock()
        self.reason, self.peak, self.records = None, 0, []
        self.timer = threading.Timer(max(0, started + seconds - time.monotonic()), self.refuse, args=("PROBE_WALL_LIMIT",))
        self.timer.daemon = True
        self.thread = threading.Thread(target=self.run, daemon=True)

    def refuse(self, reason):
        with self.lock:
            if self.reason is None: self.reason = reason
        self.job.kill()  # Exact assigned Job only, independent of observer I/O.

    def raw_sample(self):
        import psutil
        ids, rss = native.job_process_ids(self.job), 0
        for pid in ids:
            try:
                p = psutil.Process(pid); created = p.create_time(); rss += p.memory_info().rss
                native.require(psutil.Process(pid).create_time() == created, "raw watchdog PID identity drift")
            except psutil.NoSuchProcess:
                native.confirm_owned_process_exit(self.job, pid)
        return {"rss_bytes": rss, "pid_count": len(ids), "host_available_gib": psutil.virtual_memory().available / native.GIB,
                "disk_free_gib": psutil.disk_usage(str(self.output)).free / native.GIB,
                "log_bytes": sum(f.stat().st_size for f in self.output.iterdir()
                                 if f.is_file() and f.suffix in {".jsonl", ".log"})}

    def run(self):
        while not self.stop.is_set():
            try:
                sample = self.sample(); sample.update(monotonic_time=time.monotonic(), wall_time=time.time())
                self.records.append(sample); self.peak = max(self.peak, sample["rss_bytes"])
                native.require(sample["rss_bytes"] <= 2 * native.GIB and sample["pid_count"] <= 8
                               and sample["host_available_gib"] >= 1.25 and sample["disk_free_gib"] >= 4
                               and sample["log_bytes"] <= MAX_LOG, "independent resource floor/ceiling")
                self.check()
            except BaseException as error:
                self.refuse("RESOURCE_WATCH: " + type(error).__name__ + ": " + str(error)); return
            self.stop.wait(.05)

    def start(self):
        self.timer.start(); self.thread.start()

    def check(self):
        if time.monotonic() - self.started >= self.seconds: self.refuse("PROBE_WALL_LIMIT")
        native.require(self.reason is None, self.reason or "resource watch refused")

    def close(self):
        self.stop.set(); self.timer.cancel(); self.timer.join(timeout=2); self.thread.join(timeout=2)
        native.require(not self.timer.is_alive() and not self.thread.is_alive(), "independent resource watcher did not stop")


def child(args, cfg, native_cfg):
    go = native.read_json(args.go)
    handle, consumed = go.get("own_job_query_handle"), False
    store = None
    try:
        native.require(go["contract_sha256"] == args.contract_sha256 and go["worker_tree_pid"] > 0,
                       "probe GO binding required")
        import psutil
        checked_contract(args.contract_sha256)
        ancestors = [p.pid for p in psutil.Process().parents()]
        native.require(go["supervisor_pid"] in ancestors and go["worker_tree_pid"] in ancestors + [os.getpid()],
                       "probe contained ancestry mismatch")
        store = OperationsStore(host_state_path(), args.output / "receipts")
        require_probe_job(store, go, args)
        review_paths = verify_refs(go["review_refs"])
        checked_reviews(review_paths, go["binding"])
        intent_path = verify_refs([go["launch_intent_ref"]])[0]
        intent = native.read_json(intent_path)
        native.require(intent["binding"] == go["binding"] and intent["job_key"] == KEY
                       and intent["review_refs"] == go["review_refs"], "bound dispatch/review proof mismatch")
        launcher = psutil.Process(go["worker_identity"]["pid"])
        native.require(str(launcher.create_time()) == go["worker_identity"]["creation_time"]
                       and go["worker_identity"]["pid"] == go["worker_tree_pid"]
                       and os.getpid() == go["runtime_identity"]["pid"]
                       and psutil.Process().create_time() == go["runtime_identity"]["creation_time"],
                       "exact scheduled launcher/runtime identity mismatch")
        consumed = type(handle) is int and 0 < handle < 2 ** 64
        witness = native.current_windows_job_witness(handle,
                  go["query_handle_target_pid"], go["query_handle_target_creation_time"])
    except BaseException:
        if store is not None: store.close()
        raise
    finally:
        if not consumed and type(handle) is int and 0 < handle < 2 ** 64:
            native.close_unconsumed_query_handle(handle)
    try:
        return child_phases(args, native_cfg, go, store, witness)
    finally:
        store.close()


def child_phases(args, native_cfg, go, store, witness):
    context_sha = hashlib.sha256(canonical(authority(go))).hexdigest()
    native.write_new(args.output / "CHILD_CONTAINMENT.json", {**witness,
                     "authority_sha256": context_sha, "go_sha256": native.sha(args.go)})
    events = args.output / "CHILD_EVENTS.jsonl"
    def event(phase, **detail):
        require_probe_job(store, go, args)
        append(events, {"phase": phase, "authority_sha256": context_sha, **detail})
    event("idle_begin", forbidden_imports_present=[m for m in ("torch", "transformers", "tokenizers", "safetensors") if m in sys.modules])
    time.sleep(2)
    event("idle_end")
    for index in range(10):
        event("source_begin", index=index)
        observed = native.committed_snapshot()
        native.require(observed == go["source"], "contained source snapshot drift")
        runtime_observation = None
        if index in (0, 9):
            native.verify_sources(native_cfg)
            runtime_observation = native.runtime_metadata()
            native.require(runtime_observation == native_cfg["runtime"], "contained installed runtime drift")
        checked = index in (0, 9)
        event("source_end", index=index, source=observed, verify_sources_returned=checked,
              runtime_metadata=runtime_observation)
        time.sleep(.05)
    event("torch_import_begin")
    profile, violations = import_boundary(args.output / "BOUNDARY_REFUSALS.jsonl")
    event("boundary_armed", profile_current_thread=sys.getprofile() is profile,
          profile_new_python_threads=threading.getprofile() is profile)
    import torch
    event("torch_import_end", boundary=boundary_state(torch, profile))
    native.require(vars(torch.cuda).get("_initialized") is False and not violations
                   and sys.getprofile() is profile and threading.getprofile() is profile, "import boundary changed or CUDA initialized")
    event("transformers_package_import_begin")
    import transformers
    event("transformers_package_import_end", boundary=boundary_state(torch, profile))
    event("qwen_symbol_import_begin")
    from transformers import Qwen3Config, Qwen3ForCausalLM
    event("qwen_symbol_import_end", config_symbol=Qwen3Config.__module__ + "." + Qwen3Config.__name__,
          model_symbol=Qwen3ForCausalLM.__module__ + "." + Qwen3ForCausalLM.__name__,
          boundary=boundary_state(torch, profile),
          transitive_modules=[m for m in sorted(sys.modules) if m.split(".")[0] in {"torch", "transformers", "safetensors", "tokenizers"}])
    native.require(vars(torch.cuda).get("_initialized") is False and not violations
                   and sys.getprofile() is profile and threading.getprofile() is profile, "symbol import boundary changed or CUDA initialized")
    event("post_import_begin")
    time.sleep(2)
    native.require(not violations and vars(torch.cuda).get("_initialized") is False
                   and sys.getprofile() is profile and threading.getprofile() is profile,
                   "post-import boundary changed or refused")
    event("post_import_end", boundary=boundary_state(torch, profile))
    sys.setprofile(None); threading.setprofile(None)
    event("all_registered_phases_returned", model_instances_requested=0, model_binary_reads_requested=0,
          tokenizer_instances_requested=0, tensor_operations_requested=0, model_calls_requested=0,
          native_qualified=False, original_failure_cause_established=False)
    return 0


def cleanup_runtime(job, notifications, process, pipes, watcher):
    """Every cleanup attempt survives another cleanup action's exception."""
    errors, exit_proof, code = [], None, None
    def attempt(label, function):
        try: return function()
        except BaseException as error:
            errors.append({"action": label, "error": type(error).__name__ + ": " + str(error)})
            return None
    if job:
        exit_proof = attempt("drain_exact_job", lambda: native.drain_owned_job(job))
    if watcher: attempt("stop_resource_watch", watcher.close)
    if job:
        attempt("close_exact_job", job.close)
    if notifications: attempt("close_notifications", notifications.close)
    if process is not None:
        code = attempt("poll_owned_launcher", process.poll)
        if code is None:
            attempt("kill_owned_launcher", process.kill)
            code = attempt("wait_owned_launcher", lambda: process.wait(timeout=5))
        else:
            code = attempt("reap_owned_launcher", lambda: process.wait(timeout=5))
    for index, pipe in enumerate(pipes):
        if pipe: attempt("close_pipe_" + str(index), pipe.close)
    return {"exit_proof": exit_proof, "exit_code": code, "errors": errors,
            "exit_verified": process is not None and code is not None
                             and type(exit_proof) is dict and exit_proof.get("owned_tree_exited") is True}


def supervise(args, cfg, native_cfg):
    """One registered observational attempt; no automatic repeat or promotion."""
    import psutil
    native.require(os.name == "nt" and sys.executable == native_cfg["runtime"]["executable"],
                   "pinned Windows runtime required")
    source = native.committed_snapshot()
    native.require(not source["dirty"], "clean probe source required")
    binding = probe_binding(args.contract_sha256, source)
    checked_reviews(args.review, binding)
    native.verify_sources(native_cfg)
    native.require(native.runtime_metadata() == native_cfg["runtime"], "probe runtime drift")
    output = args.output.resolve()
    native.require(output.is_relative_to(PACKAGE / "results") and not output.exists(), "new CRS probe output required")
    telemetry = collect_probe_telemetry(PACKAGE, cfg, native_cfg)
    native.require(not native.admission_reasons(native_cfg, telemetry), "probe host admission closed")
    time.sleep(2)
    second = collect_probe_telemetry(PACKAGE, cfg, native_cfg)
    native.require(not native.admission_reasons(native_cfg, second), "second probe admission closed")
    output.mkdir(parents=True)
    store = OperationsStore(host_state_path(), output / "receipts")
    owner, lease = "CRS-IDENTITY-" + uuid.uuid4().hex, None
    process = job = notifications = watcher = None
    attempted = dispatched = reserved = False
    outcome, observation, reason, exit_proof = "FAILED", "UNRESOLVED_OR_INCOMPLETE_COVERAGE", None, None
    started = time.monotonic()
    stdout = stderr = None
    peak_rss = 0
    hash_budget = HashBudget()
    completion = None
    observation_number = 0
    last_observation = last_guard = None
    def observe(phase):
        nonlocal observation_number, last_observation
        before = time.monotonic(); current = inventory(job, native_cfg, hash_budget)
        observation_number += 1
        value = {"phase": phase, "observation_id": observation_number, "inventory": current,
                 "notifications": notifications.poll(), "began_monotonic": before,
                 "ended_monotonic": time.monotonic(), "target_poll_seconds": .02,
                 "gap_since_previous_observation": None if last_observation is None else before - last_observation}
        append(output / "OBSERVATIONS.jsonl", value); last_observation = before
        return current, observation_number
    def guard(observation_id, phase):
        nonlocal last_guard
        began = time.monotonic()
        append(output / "OBSERVATIONS.jsonl", {"phase": "strict_guard_begin", "guard_stage": phase,
               "observation_id": observation_id, "began_monotonic": began, "target_guard_seconds": .1,
               "gap_since_previous_guard": None if last_guard is None else began - last_guard})
        last_guard = began
        try: value = native.owned_job_rss_snapshot(job, native_cfg)
        except BaseException as error:
            append(output / "OBSERVATIONS.jsonl", {"phase": "strict_guard_refusal", "guard_stage": phase,
                   "observation_id": observation_id, "ended_monotonic": time.monotonic(),
                   "error": type(error).__name__ + ": " + str(error),
                   "inventory_after_refusal": inventory(job, native_cfg, hash_budget),
                   "notifications_after_refusal": notifications.poll()})
            raise
        append(output / "OBSERVATIONS.jsonl", {"phase": "strict_guard_return", "guard_stage": phase,
               "observation_id": observation_id, "ended_monotonic": time.monotonic()})
        return value
    try:
        store.register_job(KEY, binding, args.epoch)
        native.require(store.get_job(KEY)["state"] == "PENDING", "single registered probe already attempted")
        store.ready(KEY)
        lease = store.acquire_lease(owner, 120)
        store.reserve(KEY, owner, lease["fence"], 60)
        reserved = True
        command = [sys.executable, "-B", "-m", "experiments.rineya_centralized_reasoning_system_v1.owned_runtime_identity_probe",
                   "--child", "--contract-sha256", args.contract_sha256, "--output", str(output),
                   "--go", str(output / "GO.json"), "--epoch", args.epoch]
        review_refs = [{"path": str(path.resolve()), "sha256": native.sha(path)} for path in args.review]
        native.write_new(output / "LAUNCH_INTENT.json", {"command": command, "binding": binding,
                          "preflight": telemetry, "second_observation": second, "job_key": KEY,
                          "lease": lease, "scope": SCOPE, "model_dispatch": False,
                          "review_refs": review_refs,
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
        watcher = ResourceWatch(job, started=started, output=output); watcher.start()
        notifications = JobNotifications(job)
        initial, initial_id = observe("before_resume"); watcher.check()
        native.require(native.job_process_ids(job) == [process.pid] and not psutil.Process(process.pid).children(recursive=True),
                       "probe must be assigned suspended before children")
        guard(initial_id, "before_resume"); watcher.check()
        native.resume_suspended_primary_thread(process)
        deadline, target = time.monotonic() + 3, []
        while time.monotonic() < deadline:
            current, observation_id = observe("runtime_target"); watcher.check()
            members = guard(observation_id, "runtime_target")["members"]; watcher.check()
            target = [m for m in members if Path(m["executable"]).resolve() == Path(sys._base_executable).resolve()]
            if len(target) == 1: break
            time.sleep(.02)
        native.require(len(target) == 1, "one exact probe CPython target required")
        transfer = native.duplicate_owned_job_query_handle(job, target[0])
        admitted_row = store.get_job(KEY)
        native.write_new(output / "GO.json", {"contract_sha256": args.contract_sha256,
                         "owner": owner, "fence": lease["fence"], "worker_tree_pid": process.pid,
                         "supervisor_pid": os.getpid(), "source": source, "binding": binding,
                         "job_key": KEY, "epoch": args.epoch, "request_hash": admitted_row["request_hash"],
                         "worker_identity": worker_identity, "runtime_identity": target[0], "review_refs": review_refs,
                         "launch_intent_ref": {"path": str(output / "LAUNCH_INTENT.json"),
                                               "sha256": native.sha(output / "LAUNCH_INTENT.json")}, **transfer})
        go = native.read_json(output / "GO.json"); watcher.check()
        next_guard = next_gpu = 0.0
        while True:
            store._require_lease(owner, lease["fence"])
            now = time.monotonic()
            watcher.check()
            native.require(sum(f.stat().st_size for f in output.iterdir()
                           if f.is_file() and f.suffix in {".jsonl", ".log"}) <= MAX_LOG, "probe aggregate log ceiling")
            native.require(not (output / "BOUNDARY_REFUSALS.jsonl").exists(), "probe forbidden API boundary refused")
            current, observation_id = observe("observer"); watcher.check()
            native.require(len(current["job_pid_list"]) <= 8 and current["observed_rss_bytes"] <= 2 * native.GIB,
                           "probe process/RSS ceiling")
            native.require(psutil.virtual_memory().available / native.GIB >= 1.25
                           and psutil.disk_usage(str(output)).free / native.GIB >= 4, "probe RAM/disk floor")
            peak_rss = max(peak_rss, current["observed_rss_bytes"])
            if now >= next_guard:
                try:
                    guard(observation_id, "observer"); watcher.check()
                except BaseException:
                    observation = "OBSERVED_STRICT_GUARD_REFUSAL_IDENTITY_REQUIRES_RAW_REVIEW"
                    raise
                next_guard = now + .1
            if now >= next_gpu:
                gpu = collect_probe_telemetry(output, cfg, native_cfg)
                watcher.check()
                append(output / "OBSERVATIONS.jsonl", {"phase": "gpu_observation", "snapshot": gpu})
                owned_ids = set(native.job_process_ids(job))
                native.require(not any(p["pid"] in owned_ids for p in gpu["processes"]), "probe owned GPU side effect")
                native.require(not native.admission_reasons(native_cfg, gpu,
                               postimport_rss_gib=current["observed_rss_bytes"] / native.GIB), "probe live telemetry gate")
                next_gpu = now + .5
            if process.poll() is not None and not native.job_process_ids(job):
                native.require(process.returncode == 0, "contained probe did not return all phases")
                watcher.check()
                completion = validate_child_evidence(output, go, native_cfg)
                watcher.check()
                native.verify_sources(native_cfg)
                native.require(native.committed_snapshot() == source, "probe source drift after phases")
                watcher.check()
                outcome, observation = "SUCCEEDED", "COMPLETED_SPECIFIED_NO_MODEL_PATHS_SAMPLED_COVERAGE"
                break
            time.sleep(.02)
    except BaseException as error:
        reason = type(error).__name__ + ": " + str(error)
    finally:
        cleanup = cleanup_runtime(job, notifications, process, (stdout, stderr), watcher)
        exit_proof = cleanup["exit_proof"]
        if cleanup["errors"]:
            reason = reason or "cleanup errors retained"
        elapsed = time.monotonic() - started
        if watcher and watcher.reason:
            reason = reason or watcher.reason
            if outcome == "SUCCEEDED": outcome = "FAILED"
        if elapsed > 60 and outcome == "SUCCEEDED":
            reason = "PROBE_WALL_LIMIT_LATE_COMPLETION"; outcome = "FAILED"
        uncertain = (attempted and not dispatched) or (dispatched and (not cleanup["exit_verified"] or bool(cleanup["errors"])))
        if uncertain: outcome = "OUTCOME_UNKNOWN"
        result = {"status": outcome, "observation": observation, "reason": reason,
                  "binding": binding, "worker_dispatched": dispatched, "launch_attempted": attempted,
                  "owned_tree_exit": exit_proof, "wall_seconds": elapsed, "peak_tree_rss_bytes": peak_rss,
                  "exit_code": cleanup["exit_code"], "cleanup_errors": cleanup["errors"],
                  "hash_budget": hash_budget.snapshot(), "completion": completion,
                  "resource_watch": None if watcher is None else {"reason": watcher.reason, "peak_rss_bytes": watcher.peak,
                                                                  "samples": watcher.records},
                  "model_dispatch": False, "model_measurements": 0, "native_qualified": False,
                  "reader_qualified": False, "training_qualified": False,
                  "original_failure_cause_established": False, "native_quarantine_reopened": False,
                  "coverage_complete": False, "notifications_delivery_guaranteed": False,
                  "accounting_finalization_in_this_record": False}
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
                                     exit_observed_at=time.time(), owned_tree_exit=exit_proof, owned_tree_exited=True)
                    else: proof.update(never_dispatched=True, launch_attempted=False)
                    reference = store.write_receipt("native_outcome_observation", KEY, proof)
                    store.reconcile(KEY, outcome, actual, {"native_outcome_proof_ref": reference,
                                    "probe_result_sha256": native.sha(output / "PROBE_RESULT.json")},
                                    owner=owner, fence=lease["fence"])
                    store.release_lease(owner, lease["fence"])
            elif lease: store.release_lease(owner, lease["fence"])
            if reserved and not uncertain:
                recover_finalization(output, store, binding)
        except BaseException as error:
            failure = record_finalization_failure(output, store, binding, error, reserved=reserved)
            outcome = failure["status"]
            result["return_status_after_finalization_error"] = outcome
        finally:
            try: store.close()
            except BaseException as error:
                outcome = "OUTCOME_UNKNOWN"
                print("probe store close failed: " + type(error).__name__, file=sys.stderr, flush=True)
    print(json.dumps(result), flush=True)
    return 0 if outcome == "SUCCEEDED" else 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--supervise", action="store_true")
    mode.add_argument("--child", action="store_true")
    mode.add_argument("--recover-finalization", action="store_true")
    p.add_argument("--contract-sha256", required=True)
    p.add_argument("--review", type=Path, action="append", default=[])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epoch", default="CRS-EPOCH-001")
    p.add_argument("--go", type=Path)
    args = p.parse_args()
    cfg, native_cfg = checked_contract(args.contract_sha256)
    if args.recover_finalization:
        output = args.output.resolve()
        native.require(output.is_relative_to(PACKAGE / "results"), "owned probe recovery output required")
        store = OperationsStore(host_state_path(), output / "receipts")
        try:
            pending = native.read_json(output / "PROBE_RESULT.json")
            native.require(pending["binding"]["contract_sha256"] == args.contract_sha256,
                           "recovery contract mismatch")
            print(json.dumps(recover_finalization(output, store, pending["binding"])), flush=True)
        finally: store.close()
        return 0
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
