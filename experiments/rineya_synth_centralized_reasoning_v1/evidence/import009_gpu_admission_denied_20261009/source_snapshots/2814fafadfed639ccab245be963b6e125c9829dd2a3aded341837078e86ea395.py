"""Collector-specific coherent watchdog checks; original supervisor unchanged.

This source is staged preparation, not admission or proof of batch04 causality.
No threshold, cadence, worker ownership or containment primitive is replaced.
"""
from copy import deepcopy
import math
from pathlib import Path
import sys
import threading

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1] if HERE.name == "short_ram_successor_work" and HERE.parent.name == "artifacts" else HERE
sys.path.insert(0, str(PACKAGE))
import native_supervisor as shared

DIAGNOSTIC_NAME = "SHORT_RAM_ABORT_DIAGNOSTIC.json"


def _safe(value):
    return repr(value) if type(value) is float and not math.isfinite(value) else value


class ShortRamMonitor(shared._Monitor):
    def __init__(self, *args, clock=None, phase_clock=None, persist=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.check_clock = shared.time.monotonic if clock is None else clock
        self.phase_clock = shared.time.perf_counter if phase_clock is None else phase_clock
        self.persist_diagnostic = shared._json_new if persist is None else persist
        self.cpu_phase = {"name": "not_started", "sampler_monotonic": None, "sampler_perf_counter": None}
        self.last_failed_checks = {}
        self.diagnostic_saved = self.diagnostic_persisted = False
        self.diagnostic_error = None
        self.abort_owner_lock = threading.Lock()

    def _phase(self, name):
        value = {"name": name, "sampler_monotonic": self.check_clock(), "sampler_perf_counter": self.phase_clock()}
        with self.lock:
            self.cpu_phase = value

    def check(self):
        # The selected snapshots are immutable publications. Capturing now after
        # selection prevents a later publication being judged against an earlier
        # now; a genuinely aging selected sample still fails the same predicate.
        with self.lock:
            cpu, gpu, reason = self.cpu, self.gpu, self.reason
            now = self.check_clock()
            phase = deepcopy(self.cpu_phase)
        if reason:
            return reason
        if now >= self.deadline:
            failure = "whole_scope_deadline"
        elif not shared._fresh_cpu(cpu, now):
            failure = "stale_external_CPU_watchdog"
        elif not shared._fresh_gpu(gpu, now):
            failure = "stale_external_GPU_observation"
        elif cpu["host_available_bytes"] < 1.25 * shared.GIB:
            failure = "runtime_system_reserve"
        elif cpu["worker_tree_rss_bytes"] > 2.0 * shared.GIB:
            failure = "runtime_owned_process_tree_RSS"
        elif gpu["gpu_free_bytes"] < .5 * shared.GIB:
            failure = "runtime_GPU_free"
        elif gpu["gpu_uuid"] != self.gpu_uuid:
            failure = "runtime_GPU_identity_changed"
        else:
            return None
        amount = all(type(cpu.get(k)) is int and cpu[k] >= 0 for k in
                     ("host_available_bytes", "disk_free_bytes", "worker_tree_rss_bytes"))
        started, completed = cpu["sample_started_monotonic"], cpu["sample_completed_monotonic"]
        age = now - started
        flags = {"amounts_nonnegative_int": amount, "age_nonnegative": 0 <= age,
            "age_within_200ms": age <= shared.CPU_FRESHNESS,
            "chronology": started <= completed <= now}
        finite = all(type(v) in (int, float) and math.isfinite(v) for v in (now, started, completed))
        branch = "nonnegative_integer_amounts" if not amount else "nonfinite_timestamp" if not finite else "negative_age" if not flags["age_nonnegative"] else \
                 "over_200ms_age" if not flags["age_within_200ms"] else "chronology" if not flags["chronology"] else "other_guard"
        detail = {"schema_version": 1, "failure": failure, "check_thread_id": threading.get_ident(),
            "check_monotonic": _safe(now), "selected_CPU_started_monotonic": _safe(started),
            "selected_CPU_completed_monotonic": _safe(completed), "selected_CPU_age_seconds": _safe(age),
            "selected_GPU_started_monotonic": _safe(gpu["gpu_sample_started_monotonic"]),
            "selected_GPU_completed_monotonic": _safe(gpu["gpu_sample_completed_monotonic"]),
            "CPU_predicate_flags": flags, "CPU_predicate_branch": branch, "CPU_phase": phase,
            "CPU_freshness_seconds": shared.CPU_FRESHNESS, "CPU_interval_seconds": shared.CPU_INTERVAL,
            "causal_OS_or_batch04_explanation_established": False}
        with self.lock:
            # Three supervisor/sampler callers exist; retain a bounded current
            # failed check per thread, with no file IO on the guard path.
            if len(self.last_failed_checks) < 8 or threading.get_ident() in self.last_failed_checks:
                self.last_failed_checks[threading.get_ident()] = detail
        return failure

    def _abort(self, reason):
        # A duplicate superclass abort returns as soon as reason is set, even
        # while the owner's terminate remains in flight. Only one owner may
        # perform containment and its subsequent diagnostic publication.
        if not self.abort_owner_lock.acquire(blocking=False):
            return
        try:
            self._owned_abort(reason)
        finally:
            self.abort_owner_lock.release()

    def _owned_abort(self, reason):
        with self.lock:
            detail = deepcopy(self.last_failed_checks.get(threading.get_ident(), {
                "schema_version": 1, "failure": reason, "check_timestamp_available": False,
                "CPU_phase": deepcopy(self.cpu_phase), "check_thread_id": threading.get_ident()}))
        # Never hold the lock through the original abort, and contain the owned
        # worker before any diagnostic encoding/file operation.
        super()._abort(reason)
        with self.lock:
            if self.diagnostic_saved or self.reason != reason:
                return
            self.diagnostic_saved = True
        try:
            detail["containment_invoked_before_diagnostic_IO"] = True
            self.persist_diagnostic(self.output / DIAGNOSTIC_NAME, shared._json_safe(detail))
            self.diagnostic_persisted = True
        except BaseException as error:
            self.diagnostic_error = (type(error).__name__ + ": " + str(error))[:512]

    def _cpu_loop(self):
        next_poll = shared.time.monotonic()
        while not self.stop.is_set():
            try:
                self._phase("sampling_CPU")
                snapshot = shared._cpu_snapshot(self.output, self.worker)
                with self.lock:
                    self.cpu = snapshot
                    self.observations.append({"kind": "CPU", **snapshot})
                self._phase("CPU_published_before_heartbeat_IO")
                self.ready.set()
                self._phase("heartbeat_IO")
                shared._heartbeat(self.output, shared.os.getpid())
                self._phase("CPU_check_after_heartbeat_IO")
                reason = self.check()
                if reason:
                    self._abort(reason)
                    return
            except BaseException as error:
                self._abort("CPU_watchdog_error:" + type(error).__name__ + ":" + str(error))
                return
            next_poll += shared.CPU_INTERVAL
            self._phase("CPU_cadence_wait")
            self.stop.wait(max(0, next_poll - shared.time.monotonic()))

    def diagnostic_receipt(self):
        return {"path": DIAGNOSTIC_NAME, "attempted": self.diagnostic_saved,
            "persisted": self.diagnostic_persisted, "write_error": self.diagnostic_error,
            "original_abort_reason_preserved": self.reason}
