"""Inference-only scoped dispatcher, never study training permission.

The dispatch/containment algorithm is adapted byte-for-byte except worker and
preflight authority from native_supervisor.supervise, qualified by Native7.
All shared resource/deadline/lease/Job/monitor/cleanup implementations are reused.
The inference-specific caller requires a fresh source-bound scope and native
qualification; this code alone does not authorize DEV scoring or training.
"""
from pathlib import Path as _Path
import sys as _sys
_HERE=_Path(__file__).resolve().parent
_PACKAGE=_HERE.parents[1] if _HERE.name=="encoder_runtime_v2_work" and _HERE.parent.name=="artifacts" else _HERE
_sys.path.insert(0,str(_PACKAGE));_sys.path.insert(0,str(_HERE))

from native_supervisor import (
    ADMISSION_DEADLINE,
    ADMISSION_INTERVAL,
    BOUNDS,
    CPU_FRESHNESS,
    CPU_INTERVAL,
    GPU_FRESHNESS,
    GPU_INTERVAL,
    GPU_TIMEOUT,
    PACKAGE,
    Path,
    _LaunchFailure,
    _Monitor,
    _WindowsOwnedWorker,
    _acquire_lease,
    _admission_failure,
    _bounds_valid,
    _cpu_snapshot,
    _gpu_pending_threads,
    _inventory,
    _json_new,
    _json_safe,
    _new_output,
    _pid_alive,
    _shutdown_gpu_probe,
    _utc,
    resource_snapshot,
    sys,
    time
)

ADAPTED_SUPERVISOR_SOURCE_SHA256 = 'edf2de5b0917150c2c79e6a555dc0399995d12859028e62a7f7ece86b767e691'

def _preflight(path):
    import long_native_contract
    return long_native_contract.preflight(path)


def supervise_long_native_qualification(config_path, output_dir, *, admission_only=False):
    """Return and preserve one terminal receipt, with no launch retry."""
    import long_native_contract
    if long_native_contract.STAGED: raise ValueError("staged native-long dispatch forbidden before APIs")
    started = time.monotonic()
    output = _new_output(output_dir)
    argv = [sys.executable, "-I", "-B", str(PACKAGE / "long_native_worker.py"),
            "--config", str(Path(config_path).absolute()), "--output-dir", str(output)]
    receipt = {"schema_version": 1, "started_utc": _utc(), "output_dir": str(output),
               "attempted_worker_argv": argv, "worker_attempted": False, "worker_launched": False,
               "worker_resumed": False,
               "admission_only": bool(admission_only), "bounds": dict(BOUNDS),
               "cadence": {"CPU_watchdog_seconds": CPU_INTERVAL, "CPU_freshness_seconds": CPU_FRESHNESS,
                           "GPU_query_interval_seconds": GPU_INTERVAL, "GPU_query_timeout_seconds": GPU_TIMEOUT,
                           "GPU_freshness_seconds": GPU_FRESHNESS, "admission_interval_seconds": ADMISSION_INTERVAL,
                           "admission_deadline_seconds": ADMISSION_DEADLINE, "admission_samples": 2},
               "deadline_monotonic": started + 600, "status": "ADMISSION_DENIED", "returncode": None,
               "gpu_telemetry_backend": "pinned_System32_NVML_read_only",
               "mechanical_qualification_established": False, "production_training_authorized": False,
               "protected_bank_access": False, "admission_observations": []}
    lease = worker = monitor = None
    try:
        config = _preflight(config_path)
        receipt["preflight"] = config
        receipt["config_sha256"] = config["config_sha256"]
        authorization = config.get("authorization", {})
        if (authorization.get("scope") != "TRAIN_long_split_fresh_native_encoder_stdlib_V2_qualification_only"
                or any(authorization.get(name) is not False for name in
                       ("production_training", "protected_bank_access", "paid_services"))):
            raise ValueError("supervisor admits inference qualification only")
        _bounds_valid(config)
        if Path(config["worker_path"]).resolve() != (PACKAGE / "long_native_worker.py").resolve():
            raise ValueError("worker must be the pinned package/long_native_worker.py")
        argv[3] = config["worker_path"]
        argv[5] = config["config_path"]
        if time.monotonic() >= receipt["deadline_monotonic"]:
            raise RuntimeError("whole scope elapsed during preflight")
        lease = _acquire_lease(output, config["config_sha256"])
        receipt["lease_path"] = str(lease)
        if not admission_only:
            # Byte checks can take longer than the GPU freshness window. Complete
            # them before the final current observation pair; the worker repeats
            # preflight before importing any model framework or loading weights.
            refreshed = _preflight(config_path)
            if refreshed != config:
                raise RuntimeError("source/config/runtime/data binding changed before admission")
            receipt["preflight_revalidated_before_current_observations"] = True
            if time.monotonic() >= receipt["deadline_monotonic"]:
                raise RuntimeError("whole scope elapsed during pin revalidation")
        admission_started = time.monotonic()
        for index in range(2):
            if index:
                time.sleep(ADMISSION_INTERVAL)
            observation = resource_snapshot(output)
            receipt["admission_observations"].append(observation)
            now = time.monotonic()
            if now - admission_started > ADMISSION_DEADLINE or now >= receipt["deadline_monotonic"]:
                raise RuntimeError("admission observation deadline")
            failure = _admission_failure(observation, now)
            if failure:
                raise RuntimeError(failure)
            if index:
                prior = receipt["admission_observations"][0]
                if (observation["gpu_observation_id"] == prior["gpu_observation_id"]
                        or observation["gpu_sample_started_monotonic"] <= prior["gpu_sample_started_monotonic"]
                        or observation["sample_started_monotonic"] <= prior["sample_started_monotonic"]
                        or observation["gpu_uuid"] != prior["gpu_uuid"]):
                    raise RuntimeError("admission requires two distinct current observations of the same GPU")
        receipt["status"] = "ADMISSION_PASSED_NO_WORKER"
        if not admission_only:
            dispatch_cpu = _cpu_snapshot(output)
            receipt["dispatch_observation"] = {**observation, **dispatch_cpu}
            if _admission_failure(receipt["dispatch_observation"], time.monotonic()):
                raise RuntimeError("admission became stale before dispatch")
            observation = receipt["dispatch_observation"]
            receipt["worker_attempted"] = True
            worker = _WindowsOwnedWorker(argv, output, config_sha256=config["config_sha256"],
                                         deadline_monotonic=receipt["deadline_monotonic"])
            receipt["worker_launched"] = True
            receipt["worker_identity"] = dict(worker.identity)
            receipt["owned_handle_alive_verified_after_launch"] = worker.is_alive()
            _json_new(lease / "worker.json", receipt["worker_identity"])
            monitor = _Monitor(worker, output, receipt["deadline_monotonic"], observation)
            monitor.start()
            reason = monitor.pre_resume_check()
            if reason:
                monitor._abort(reason)
                receipt["status"] = "OWNED_WORKER_ABORTED"
                receipt["failure"] = reason
            else:
                worker.resume()
                receipt["worker_resumed"] = True
                receipt["worker_resume_monotonic"] = time.monotonic()
                _json_new(lease / "resume.json", {
                    "worker_identity": receipt["worker_identity"], "resumed_monotonic": receipt["worker_resume_monotonic"],
                    "external_monitor_ready_before_resume": True, "deadline_monotonic": receipt["deadline_monotonic"]})
                while True:
                    reason = monitor.check()
                    if reason:
                        monitor._abort(reason)
                        receipt["status"] = "OWNED_WORKER_ABORTED"
                        receipt["failure"] = reason
                        break
                    returncode = worker.poll()
                    if returncode is not None:
                        receipt["returncode"] = returncode
                        if worker.process_identities():
                            monitor._abort("root_exited_with_live_owned_descendants")
                            receipt["status"] = "OWNED_WORKER_ABORTED"
                            receipt["failure"] = "root_exited_with_live_owned_descendants"
                        else:
                            receipt["status"] = ("SUPERVISED_WORKER_EXITED_ZERO" if returncode == 0
                                                 else "SUPERVISED_WORKER_FAILED")
                        break
                    time.sleep(CPU_INTERVAL)
    except BaseException as error:
        receipt["failure"] = type(error).__name__ + ": " + str(error)
        receipt["status"] = "OWNED_WORKER_ABORTED" if worker else "ADMISSION_DENIED"
        if isinstance(error, _LaunchFailure):
            receipt["status"] = "OWNED_WORKER_LAUNCH_FAILED"
            receipt["launch_failure"] = error.evidence
            receipt["returncode"] = error.evidence["returncode"]
            if lease and error.evidence.get("process_identity"):
                _json_new(lease / "worker.json", error.evidence["process_identity"])
            _json_new(output / "WORKER_LAUNCH_FAILURE.json", error.evidence)
        if worker:
            try:
                worker.terminate()
            except BaseException as containment_error:
                receipt["termination_error"] = type(containment_error).__name__ + ": " + str(containment_error)
    finally:
        unjoined = []
        if monitor:
            try:
                observations = monitor.close()
                unjoined.extend(getattr(monitor, "unjoined_threads", []))
                _json_new(output / "RESOURCE_OBSERVATIONS.json", observations)
            except BaseException as error:
                receipt["sampler_shutdown_error"] = type(error).__name__ + ": " + str(error)
                unjoined.append({"kind": "external_sampler_shutdown_unverified"})
        if worker:
            try:
                worker.close()
                receipt["returncode"] = worker.poll()
                receipt["job_closed_kill_on_close"] = True
                receipt["owned_worker_identity_currently_alive"] = _pid_alive(worker.identity)
            except BaseException as error:
                receipt["containment_failure"] = type(error).__name__ + ": " + str(error)
                receipt["status"] = "OWNED_WORKER_ABORTED"
        if not unjoined:
            try:
                _shutdown_gpu_probe()
                receipt["gpu_telemetry_cleanup_verified"] = True
            except BaseException as error:
                receipt["gpu_telemetry_cleanup_error"] = type(error).__name__ + ": " + str(error)
                receipt["gpu_telemetry_cleanup_verified"] = False
        unjoined.extend({"kind": "NVML_native_call", **item} for item in _gpu_pending_threads())
        unique = {}
        for item in unjoined:
            key = (item.get("kind"), item.get("name"), item.get("ident"), item.get("operation"))
            unique[key] = item
        unjoined = list(unique.values())
        receipt["unjoined_sampler_threads"] = unjoined
        receipt["sampler_threads_terminal_verified"] = not unjoined
        if unjoined or receipt.get("gpu_telemetry_cleanup_verified") is False:
            receipt["status"] = "SUPERVISOR_SHUTDOWN_UNVERIFIED"
        receipt["finished_utc"] = _utc()
        receipt["elapsed_seconds"] = time.monotonic() - started
        if (time.monotonic() >= receipt["deadline_monotonic"]
                and receipt["status"] == "SUPERVISED_WORKER_EXITED_ZERO"):
            receipt["status"] = "WHOLE_SCOPE_DEADLINE_EXCEEDED"
            receipt["failure"] = "whole_scope_deadline_during_shutdown"
        if unjoined:
            receipt["partial_outputs_frozen"] = False
            receipt["partial_outputs"] = [{"path": str(path.relative_to(output)), "frozen": False}
                                           for path in sorted(output.rglob("*")) if path.is_file() or path.is_symlink()]
        else:
            receipt["partial_outputs_frozen"] = True
            receipt["partial_outputs"] = _inventory(output)
        receipt["finished_utc"] = _utc()
        receipt["elapsed_seconds"] = time.monotonic() - started
        if (time.monotonic() >= receipt["deadline_monotonic"]
                and receipt["status"] == "SUPERVISED_WORKER_EXITED_ZERO"):
            receipt["status"] = "WHOLE_SCOPE_DEADLINE_EXCEEDED"
            receipt["failure"] = "whole_scope_deadline_during_inventory"
        receipt = _json_safe(receipt)
        if lease:
            _json_new(lease / "terminal.json", receipt)
        _json_new(output / "SUPERVISOR_RECEIPT.json", receipt)
    return receipt
