"""Deterministic stdlib-only controls; no host benchmark/model/native launch."""
import ast
from copy import deepcopy
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1] if HERE.name == "short_ram_successor_work" and HERE.parent.name == "artifacts" else HERE.parent
STAGE = PACKAGE/"artifacts/short_ram_successor_work"
CODE = STAGE if HERE == STAGE else PACKAGE
ORIGINAL = (STAGE/"original_source_preserved") if HERE == STAGE else PACKAGE/"evidence/short_ram_successor_reviewed_cpu_20261005/original_source_preserved"
sys.path.insert(0, str(CODE)); sys.path.append(str(PACKAGE))


def load_target(name, path, *, register=True):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    if register: sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = load_target("short_collection_contract", CODE/"short_collection_contract.py")
successor = load_target("short_ram_monitor", CODE/"short_ram_monitor.py")
assert Path(contract.__file__).resolve().parent == CODE.resolve()
assert Path(successor.__file__).resolve().parent == CODE.resolve()

shared = successor.shared


def snapshot(now=10.):
    return {"sample_started_monotonic": now-.1, "sample_completed_monotonic": now-.09,
        "host_available_bytes": 2*shared.GIB, "disk_free_bytes": 4*shared.GIB, "worker_tree_rss_bytes": shared.GIB,
        "gpu_sample_started_monotonic": now-.1, "gpu_sample_completed_monotonic": now-.09,
        "gpu_free_bytes": shared.GIB, "gpu_total_bytes": 8*shared.GIB, "GPU_index": 0,
        "gpu_observation_id": "CPU_FIXTURE_OBSERVATION_NO_API", "gpu_uuid": "CPU_FIXTURE_GPU_ID_NO_API"}


def monitor(now=10., persist=None, worker=None):
    events = []
    if worker is None:
        worker = SimpleNamespace(terminate=lambda: events.append("terminate"), job=SimpleNamespace(close=lambda: events.append("job_close")))
    result = successor.ShortRamMonitor(worker, HERE, 600., snapshot(now), clock=lambda: now,
        phase_clock=lambda: 123., persist=(lambda path, value: events.append((path.name, value))) if persist is None else persist)
    result.gpu_uuid = result.gpu["gpu_uuid"]
    return result, events


class PublishingLock:
    """Deterministically publish between the original clock and snapshot copy."""
    def __init__(self, target, published):
        self.target, self.published = target, published
        self.request = threading.Event()
        self.thread = threading.Thread(target=self.publish)
        self.thread.start()
    def publish(self):
        if not self.request.wait(2):
            return
        self.target.cpu = {**self.target.cpu, "sample_started_monotonic": 10.001, "sample_completed_monotonic": 10.002}
        self.published.set()
    def __enter__(self):
        self.request.set()
        if not self.published.wait(2):
            raise RuntimeError("CPU barrier did not publish")
        return self
    def __exit__(self, *args):
        return False


def original_check():
    source = (PACKAGE/"native_supervisor.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_Monitor")
    function = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "check")
    namespace = {"time": SimpleNamespace(monotonic=lambda: 10.), "GIB": shared.GIB,
                 "_fresh_cpu": shared._fresh_cpu, "_fresh_gpu": shared._fresh_gpu}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(PACKAGE/"native_supervisor.py"), "exec"), namespace)
    return namespace["check"], namespace


class MonitorControls(unittest.TestCase):
    def test_exact_original_barrier_race_and_successor_order(self):
        outcomes = []
        for original in (True, False):
            target, _ = monitor(); published = threading.Event()
            clock = lambda: 10.003 if published.is_set() else 10.
            lock = PublishingLock(target, published); target.lock = lock
            if original:
                check, ns = original_check(); ns["time"].monotonic = clock
                result = MethodType(check, target)()
            else:
                target.check_clock = clock; result = target.check()
            lock.thread.join(2); self.assertFalse(lock.thread.is_alive())
            outcomes.append(result)
        self.assertEqual(outcomes, ["stale_external_CPU_watchdog", None])

    def test_stale_negative_nonfinite_and_overlong_completion_rejected(self):
        controls = [("sample_started_monotonic", 9.799), ("sample_started_monotonic", 10.01),
            ("sample_started_monotonic", float("nan")), ("sample_completed_monotonic", 10.01),
            ("sample_started_monotonic", float("-inf"))]
        for field, value in controls:
            target, _ = monitor(); target.cpu = {**target.cpu, field: value}
            self.assertEqual(target.check(), "stale_external_CPU_watchdog")
            check, _ = original_check()
            self.assertEqual(MethodType(check, target)(), target.check())

    def test_registered_cpu_window_boundary_unchanged(self):
        target, _ = monitor(now=.2)
        target.cpu = {**target.cpu, "sample_started_monotonic": 0., "sample_completed_monotonic": .1}
        self.assertIsNone(target.check())
        target.check_clock = lambda: .2000001
        self.assertEqual(target.check(), "stale_external_CPU_watchdog")

    def test_all_original_floors_gpu_identity_and_deadline(self):
        cases = [("cpu", "host_available_bytes", int(1.25*shared.GIB)-1, "runtime_system_reserve"),
            ("cpu", "worker_tree_rss_bytes", 2*shared.GIB+1, "runtime_owned_process_tree_RSS"),
            ("gpu", "gpu_free_bytes", int(.5*shared.GIB)-1, "runtime_GPU_free"),
            ("gpu", "gpu_uuid", "CHANGED", "runtime_GPU_identity_changed"),
            ("gpu", "gpu_sample_started_monotonic", 7.999, "stale_external_GPU_observation"),
            ("cpu", "host_available_bytes", True, "stale_external_CPU_watchdog")]
        for kind, key, value, expected in cases:
            target, _ = monitor(); setattr(target, kind, {**getattr(target, kind), key: value})
            self.assertEqual(target.check(), expected)
            check, _ = original_check()
            self.assertEqual(MethodType(check, target)(), expected)
        target, _ = monitor(); target.deadline = 10.
        self.assertEqual(target.check(), "whole_scope_deadline")

    def test_diagnostic_after_containment_and_error_does_not_override_reason(self):
        events = []
        def fail_write(path, value):
            events.append("diagnostic_IO"); self.assertEqual(events[0], "terminate")
            raise OSError("injected diagnostic failure")
        worker = SimpleNamespace(terminate=lambda: events.append("terminate"), job=SimpleNamespace(close=lambda: events.append("job_close")))
        target, _ = monitor(persist=fail_write, worker=worker)
        target.cpu = {**target.cpu, "sample_started_monotonic": 9.7}
        reason = target.check(); target._abort(reason); target._abort(reason)
        self.assertEqual(events, ["terminate", "diagnostic_IO"])
        self.assertEqual(target.reason, "stale_external_CPU_watchdog")
        self.assertIn("OSError", target.diagnostic_receipt()["write_error"])

    def test_termination_failure_keeps_original_job_close_fallback(self):
        events = []
        def terminate():
            events.append("terminate"); raise OSError("injected terminate failure")
        worker = SimpleNamespace(terminate=terminate, job=SimpleNamespace(close=lambda: events.append("job_close")))
        target, _ = monitor(worker=worker, persist=lambda *args: events.append("diagnostic_IO"))
        target._abort("CPU_watchdog_error:fixture")
        self.assertEqual(events, ["terminate", "job_close", "diagnostic_IO"])
        self.assertEqual(target.reason, "CPU_watchdog_error:fixture")

    def test_concurrent_abort_cannot_publish_before_owner_termination_returns(self):
        events = []; entered = threading.Event(); release = threading.Event()
        def terminate():
            events.append("terminate_enter"); entered.set()
            if not release.wait(2): raise RuntimeError("CPU barrier timeout")
            events.append("terminate_return")
        worker = SimpleNamespace(terminate=terminate, job=SimpleNamespace(close=lambda: events.append("job_close")))
        target, _ = monitor(worker=worker, persist=lambda *args: events.append("diagnostic_write"))
        owner = threading.Thread(target=lambda: target._abort("CPU_FIXTURE_ABORT")); owner.start()
        self.assertTrue(entered.wait(2))
        duplicate = threading.Thread(target=lambda: target._abort("CPU_FIXTURE_ABORT")); duplicate.start(); duplicate.join(2)
        self.assertFalse(duplicate.is_alive()); self.assertEqual(events, ["terminate_enter"])
        release.set(); owner.join(2); self.assertFalse(owner.is_alive())
        self.assertEqual(events, ["terminate_enter", "terminate_return", "diagnostic_write"])
        self.assertEqual(target.reason, "CPU_FIXTURE_ABORT")

    def test_concurrent_abort_cannot_publish_before_Job_close_fallback_returns(self):
        events = []; entered = threading.Event(); release = threading.Event()
        def terminate(): events.append("terminate_failed"); raise OSError("CPU_FIXTURE_TERMINATION_FAILURE")
        def close():
            events.append("job_close_enter"); entered.set()
            if not release.wait(2): raise RuntimeError("CPU barrier timeout")
            events.append("job_close_return")
        worker = SimpleNamespace(terminate=terminate, job=SimpleNamespace(close=close))
        target, _ = monitor(worker=worker, persist=lambda *args: events.append("diagnostic_write"))
        owner = threading.Thread(target=lambda: target._abort("CPU_FIXTURE_ABORT")); owner.start()
        self.assertTrue(entered.wait(2)); target._abort("CPU_FIXTURE_ABORT")
        self.assertEqual(events, ["terminate_failed", "job_close_enter"])
        release.set(); owner.join(2); self.assertFalse(owner.is_alive())
        self.assertEqual(events, ["terminate_failed", "job_close_enter", "job_close_return", "diagnostic_write"])

    def test_sampler_phase_clock_and_selected_check_evidence(self):
        now = [20.]; captured = []
        target, _ = monitor(now=20., persist=lambda path, value: captured.append(value))
        target.check_clock = lambda: now[0]
        def heartbeat(*args): now[0] = 20.25
        with patch.object(shared, "_cpu_snapshot", return_value=snapshot(20.)), patch.object(shared, "_heartbeat", side_effect=heartbeat):
            target._cpu_loop()
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["CPU_predicate_branch"], "over_200ms_age")
        self.assertEqual(captured[0]["check_monotonic"], 20.25)
        self.assertEqual(captured[0]["CPU_phase"]["name"], "CPU_check_after_heartbeat_IO")
        self.assertEqual(captured[0]["CPU_phase"]["sampler_perf_counter"], 123.)
        self.assertEqual(captured[0]["selected_CPU_started_monotonic"], 19.9)
        self.assertFalse(captured[0]["causal_OS_or_batch04_explanation_established"])


class RecipeControls(unittest.TestCase):
    def test_runtime65_retains_all64_pins_and_rejects_omitted_or_substituted_AST(self):
        core = {"CPU_FIXTURE_RUNTIME_%02d" % i: "%064x" % i for i in range(64)}
        profile = {"id": "SHORT_M0_PHYSICAL_RAM_COHERENT_EXTERNAL_CHECK_V2", "runtime_extension_pins": {"CPU_FIXTURE/ast.py": "a"*64}}
        merged = contract.merge_runtime_inventory(core, profile)
        self.assertEqual(len(merged), 65); self.assertEqual({k: merged[k] for k in core}, core)
        for bad_core, bad_profile in ((dict(list(core.items())[:-1]), profile), (core, {**profile, "runtime_extension_pins": {}}),
                (core, {**profile, "runtime_extension_pins": {"CPU_FIXTURE/other.py": "a"*64}}),
                (core, {**profile, "id": "unknown"})):
            with self.assertRaises(ValueError): contract.merge_runtime_inventory(bad_core, bad_profile)

    def test_dispatch_algorithm_preserved_except_declared_post_join_diagnostic(self):
        module = load_target("successor_supervisor_algorithm_control", CODE/"short_collection_supervisor.py", register=False)
        self.assertEqual(Path(module.__file__).resolve().parent, CODE.resolve())
        expected = inspect.getsource(shared.supervise)
        for old, new in (("def supervise(", "def _supervise_registered_batch("), ("native_worker.py", "short_collection_worker.py"),
                ("mechanical_native_qualification_only", contract.PURPOSE), ("mechanical qualification only", "one fixed short M0 raw batch only")):
            expected = expected.replace(old, new)
        actual = inspect.getsource(module._supervise_registered_batch).replace(
            '        if monitor is not None:\n            receipt["short_RAM_monitor_diagnostic"] = monitor.diagnostic_receipt()\n', '')
        self.assertEqual(expected, actual)
        for name in ("_WindowsOwnedWorker", "_bounds_valid", "resource_snapshot", "_shutdown_gpu_probe"):
            self.assertIs(getattr(module, name), getattr(shared, name))
        self.assertTrue(issubclass(module._Monitor, shared._Monitor))

    def test_contract_bootstrap_portable_and_stage_preflight_refuses_first(self):
        self.assertEqual(contract.PACKAGE, PACKAGE)
        expected = "staged preparation" if contract.STAGED else "owned collector config"
        with self.assertRaisesRegex(ValueError, expected):
            contract.preflight(HERE/"NO_CONFIG_NO_NATIVE")
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            package = Path(directory)/"copied_package"; package.mkdir()
            source = package/"short_collection_contract.py"; source.write_bytes((CODE/"short_collection_contract.py").read_bytes())
            spec = importlib.util.spec_from_file_location("portable_short_contract_control", source)
            module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            self.assertEqual(module.PACKAGE, package.resolve()); self.assertFalse(module.STAGED)

    def test_ram_callback_fresh_exact_types_errors_and_unchanged_worker_guard_floors(self):
        import worker_ram_probe
        from inference_qual_worker import make_guard
        calls = []
        status = {"available_bytes": 2*shared.GIB, "total_bytes": 16*shared.GIB}
        def read(): calls.append(1); return deepcopy(status)
        callback = lambda: worker_ram_probe.available_bytes(status_call=read)
        rss = [shared.GIB]
        guard = make_guard(HERE, {"job_name": "CPU_FIXTURE_NO_JOB_API", "supervisor_pid": 1}, 0., lambda: None,
            owned_resources=lambda name: {"tree_rss_bytes": rss[0]}, available=callback,
            monotonic=lambda: 1., wall=lambda: 1., heartbeat_reader=lambda path: {"unix_seconds": 1., "supervisor_pid": 1})
        guard("CPU1"); guard("CPU2"); self.assertEqual(len(calls), 2)
        status["available_bytes"] = int(1.25*shared.GIB)-1
        with self.assertRaisesRegex(ValueError, "host resource guard"): guard("RAM_floor")
        status["available_bytes"] = 2*shared.GIB; rss[0] = 2*shared.GIB+1
        with self.assertRaisesRegex(ValueError, "host resource guard"): guard("RSS_floor")
        for value in (True, float("nan"), -1, 17*shared.GIB):
            status["available_bytes"] = value
            with self.assertRaises(ValueError): callback()
        def error(): raise OSError("injected status error")
        with self.assertRaisesRegex(OSError, "injected status error"):
            worker_ram_probe.available_bytes(status_call=error)

    def test_helper_bytepins_and_proven_callback_profile(self):
        profile = contract.successor_profile()
        self.assertEqual(profile["helper_pins"], contract.HELPER_PINS)
        self.assertEqual(profile["CPU_freshness_seconds"], .2)
        self.assertFalse(profile["native_launch_authorized"])

    def test_source_pin_drift_rejects_profile(self):
        with patch.object(contract.native, "filehash", return_value="0"*64):
            with self.assertRaisesRegex(ValueError, "helper/proven callback byte pin"):
                contract.successor_profile()

    def test_original_plan_row_order_and_dose_methods_unchanged(self):
        def functions(path):
            return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)}
        old = functions(ORIGINAL/"short_collection_contract.py")
        new = functions(CODE/"short_collection_contract.py")
        for name in ("plan", "select_batch", "_closed"):
            self.assertEqual(old[name], new[name])

    def test_staged_dispatch_and_worker_reject_before_any_native_operation(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            fake_stage = Path(directory)/"package/artifacts/short_ram_successor_work"
            fake_stage.mkdir(parents=True)
            for name in ("short_collection_contract.py", "short_collection_worker.py", "short_collection_supervisor.py", "short_ram_monitor.py"):
                (fake_stage/name).write_bytes((CODE/name).read_bytes())
            staged_contract = load_target("explicit_stage_contract_control", fake_stage/"short_collection_contract.py", register=False)
            staged_monitor = load_target("explicit_stage_monitor_control", fake_stage/"short_ram_monitor.py", register=False)
            with patch.dict(sys.modules, {"short_collection_contract": staged_contract, "short_ram_monitor": staged_monitor}):
                for name, call in (("short_collection_worker", "run"), ("short_collection_supervisor", "dispatch")):
                    module = load_target("explicit_stage_"+name, fake_stage/(name+".py"), register=False)
                    self.assertIs(module.contract, staged_contract)
                    self.assertEqual(Path(module.contract.__file__).resolve().parent, fake_stage.resolve())
                    with self.assertRaisesRegex(ValueError, "staged preparation"):
                        getattr(module, call)("NO_CONFIG", "NO_OUTPUT")
        self.assertFalse(any(n.split(".", 1)[0] in {"torch", "transformers", "tensorflow", "jax"} for n in sys.modules))

    def test_only_worker_available_callback_changed_from_original(self):
        old = (ORIGINAL/"short_collection_worker.py").read_text()
        new = (CODE/"short_collection_worker.py").read_text()
        restored = new.replace("import worker_ram_probe\n", "").replace(
            "make_guard(output,owner,start,lambda:torch,available=worker_ram_probe.available_bytes)",
            "make_guard(output,owner,start,lambda:torch)")
        self.assertEqual(ast.dump(ast.parse(restored), include_attributes=False), ast.dump(ast.parse(old), include_attributes=False))


if __name__ == "__main__":
    unittest.main()
