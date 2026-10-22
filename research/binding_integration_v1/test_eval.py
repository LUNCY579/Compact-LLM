"""Synthetic auxiliary-probe custody tests; no production model/weight access."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

WORKTREE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKTREE))
from research.binding_integration_v1 import eval_common as c, eval_probe as p, run_probe
from research.binding_diagnostic_v1 import score as diagnostic_score


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name).resolve()
        self.patch_out = patch.object(c, "OUT", self.root); self.patch_out.start()
        (self.root / "eval").mkdir(); (self.root / "posttrain/reservations").mkdir(parents=True)
        self.grant_path = self.root / "AUX_PROBE_GRANT.json"; self.gate_path = self.root / "BUILD_GATE.json"
        self.put(self.grant_path, {"fixture": "grant"}); self.put(self.gate_path, {"status": "PASS"})
        self.envelope = {"deadline_unix": 1000000, "free_disk_floor_bytes": 0, "new_records_disk_bytes": 2**30,
                         "probe_workers_all_attempts": 7, "probe_worker_seconds_all_attempts": 6300,
                         "probe_rows_reserved_all_attempts": 3500, "probe_generation_tokens_reserved_all_attempts": 200000,
                         "probe_retries_total": 1}
        self.put(self.root / "SESSION_ENVELOPE.json", self.envelope)

    def tearDown(self):
        self.patch_out.stop(); self.temp.cleanup()

    def put(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    def inputs(self, kind="binding", *, text="Synthetic fixture"):
        row_count, geometry = c.KINDS[kind]
        rows = []; parent_index = 0
        for per_parent, parents in geometry.items():
            for _ in range(parents):
                parent = f"p{parent_index}"; parent_index += 1
                for index in range(per_parent):
                    rows.append({"id": f"{parent}-{index}", "parent_id": parent,
                                 "prompt": [{"role": "system", "content": "Synthetic fixture"}, {"role": "user", "content": text}],
                                 "prompt_ids": [1, 2, 3]})
        self.assertEqual(len(rows), row_count)
        path = self.root / (kind + "-inputs.jsonl")
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        spec = {"input_path": str(path), "input_sha256": c.sha(path), "rows": row_count,
                "parent_counts": {str(key): value for key, value in geometry.items()}}
        return rows, {"datasets": {kind: spec}}

    def global_reservation(self, suffix, attempt=1):
        counts = {"dose": (384, 128), "select": (384, 128), "retention": (150, 24),
                  "long_context": (256, 24), "binding": (384, 24), "diagnostic": (896, 24)}
        n, maximum = counts[suffix]; slot = "integrated-" + suffix + ("-retry1" if attempt == 2 else "")
        path = self.root / "posttrain/reservations" / (slot + ".json")
        self.put(path, {"suffix": suffix, "attempt": attempt, "out": str(self.root / "eval" / slot),
                        "reserved_rows": n, "reserved_generation_tokens": n * maximum, "reserved_worker_seconds": 900,
                        "session_envelope_sha256": c.sha(self.root / "SESSION_ENVELOPE.json")})
        return path

    def local_reservation(self, kind, attempt=1, *, owner=None, rows=None):
        name = c.slot(kind, attempt); n = c.KINDS[kind][0] if rows is None else len(rows)
        prefills = 3 * n if rows is None else sum(len(row["prompt_ids"]) for row in rows)
        parent = self.global_reservation(kind, attempt)
        value = {"kind": kind, "attempt": attempt, "grant_sha256": c.sha(self.grant_path),
                 "rows": n, "reserved_generated_tokens": n * 24, "forward_calls": n * 24,
                 "processed_input_tokens": prefills + n * 24, "parent_reservation_sha256": c.sha(parent)}
        if owner is not None:
            value.update(controller_owner=owner, controller_lease_sha256=c.sha(self.root / "eval/AUX_CONTROLLER_LEASE.json"))
        self.put(p.reservation_path(kind, attempt), value)
        return value


class InputAndBudgetTests(Fixture):
    def test_input_geometry_and_gold_separation(self):
        for kind in c.KINDS:
            rows, grant = self.inputs(kind)
            self.assertEqual(c.load_inputs(grant, kind), rows)
        rows, grant = self.inputs()
        rows[0]["reference"] = "forbidden worker field"
        path = Path(grant["datasets"]["binding"]["input_path"])
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        grant["datasets"]["binding"]["input_sha256"] = c.sha(path)
        with self.assertRaisesRegex(ValueError, "gold/extra"):
            c.load_inputs(grant, "binding")

    def test_utf8_prompt_decoding_does_not_depend_on_windows_locale(self):
        rows, grant = self.inputs(text="Café Ω")
        original = Path.read_text
        def locale_read(path, encoding=None, errors=None, **kwargs):
            return original(path, encoding="cp1252" if encoding is None else encoding, errors=errors, **kwargs)
        with patch.object(Path, "read_text", locale_read):
            observed = c.load_inputs(grant, "binding")
        self.assertEqual(observed, rows)

    def test_local_allowance_is_two_initial_legs_plus_one_global_aux_retry(self):
        self.local_reservation("binding"); self.local_reservation("diagnostic"); self.local_reservation("diagnostic", 2)
        total = c.costs({}, self.grant_path)["totals"]
        self.assertEqual((total["workers"], total["rows"], total["reserved_generated_tokens"], total["retries"]), (3, 2176, 52224, 1))
        self.local_reservation("binding", 2)
        with self.assertRaisesRegex(ValueError, "subcap"):
            c.costs({}, self.grant_path)

    def test_six_leg_global_budget_keeps_every_retry_and_failure(self):
        for leg in ("dose", "select", "retention", "long_context", "binding", "diagnostic"):
            self.global_reservation(leg)
        total = c.global_costs()
        self.assertEqual((total["attempts"], total["rows"], total["generation_tokens"]), (6, 2454, 138768))
        self.put(self.root / "eval/integrated-binding/supervisor.json", {"status": "EXITED", "returncode": 2, "wall_seconds": 117.25})
        self.assertEqual(c.global_costs()["worker_seconds"], 5 * 900 + 117.25)
        self.global_reservation("diagnostic", 2)
        self.assertEqual((c.global_costs()["attempts"], c.global_costs()["rows"]), (7, 3350))
        self.global_reservation("binding", 2)
        with self.assertRaisesRegex(ValueError, "global all-attempt"):
            c.global_costs()

    def test_expiry_refuses_a_worker_before_any_disk_or_model_work(self):
        with patch.object(c.time, "time", return_value=101), patch.object(c.shutil, "disk_usage") as disk:
            with self.assertRaisesRegex(ValueError, "deadline"):
                c.phase_guard({"deadline_unix": 1000}, 900)
            disk.assert_not_called()

    def test_inherited_supervisor_guard_already_enforces_the_900_second_cap(self):
        terminal = {"status": "COMPLETE", "complete": True, "failure": None, "before": {}, "after": {}}
        for wall in (900.01, float("inf"), float("nan"), -1, True):
            with self.subTest(wall=wall):
                with self.assertRaises(ValueError):
                    c.original.verify_terminal({"status": "EXITED", "returncode": 0, "wall_seconds": wall}, terminal)
        c.original.verify_terminal({"status": "EXITED", "returncode": 0, "wall_seconds": 900}, terminal)


class LeaseAndWorkerTests(Fixture):
    def test_concurrent_aux_controller_cannot_enter(self):
        with c.controller_lease(self.grant_path):
            original = (self.root / "eval/AUX_CONTROLLER_LEASE.json").read_bytes()
            def contend():
                with c.controller_lease(self.grant_path):
                    self.fail("second controller entered")
            with ThreadPoolExecutor(max_workers=1) as pool:
                with self.assertRaises(FileExistsError):
                    pool.submit(contend).result()
            self.assertEqual((self.root / "eval/AUX_CONTROLLER_LEASE.json").read_bytes(), original)
        self.assertFalse((self.root / "eval/AUX_CONTROLLER_LEASE.json").exists())

    def test_changed_lease_is_preserved(self):
        path = self.root / "eval/AUX_CONTROLLER_LEASE.json"
        with self.assertRaisesRegex(ValueError, "owner changed"):
            with c.controller_lease(self.grant_path):
                self.put(path, {"owner_pid": 987, "nonce": "another owner"})
        self.assertEqual(c.read(path)["owner_pid"], 987)

    def test_worker_requires_the_current_grant_in_its_owner_lease(self):
        rows, grant = self.inputs()
        args = SimpleNamespace(kind="binding", attempt=1, grant=self.grant_path)
        out = self.root / "eval" / c.slot("binding", 1); out.mkdir()
        owner = {"owner_pid": 55, "nonce": "fixture", "grant_sha256": "0" * 64}
        self.put(self.root / "eval/AUX_CONTROLLER_LEASE.json", owner)
        self.local_reservation("binding", owner=owner, rows=rows)
        self.put(out / "launch.json", {"supervisor_pid": 55}); self.put(out / "worker_claim.json", {"supervisor_pid": 55})
        with self.assertRaisesRegex(ValueError, "ownership|reservation|grant"):
            p.verify_reservation(args, grant, rows, out)

    def test_parent_reservation_is_required_before_worker_runtime(self):
        rows, grant = self.inputs()
        args = SimpleNamespace(kind="binding", attempt=1, grant=self.grant_path)
        out = self.root / "eval" / c.slot("binding", 1); out.mkdir()
        with c.controller_lease(self.grant_path) as owner:
            self.local_reservation("binding", owner=owner, rows=rows)
            self.put(out / "launch.json", {"supervisor_pid": owner["owner_pid"]})
            self.put(out / "worker_claim.json", {"supervisor_pid": owner["owner_pid"]})
            p.verify_reservation(args, grant, rows, out)
            (self.root / "posttrain/reservations" / (out.name + ".json")).unlink()
            with self.assertRaises((OSError, ValueError)):
                p.verify_reservation(args, grant, rows, out)

    def test_bad_reservation_precedes_resource_guard_and_weight_snapshot(self):
        out = self.root / "eval/integrated-binding"; out.mkdir()
        args = SimpleNamespace(kind="binding", attempt=1, grant=self.grant_path, build_gate=self.gate_path, out=out)
        supervisor = ModuleType("experiments.model_quality_gpu_v1.supervisor"); supervisor.claim_worker = lambda value: None
        resources = ModuleType("experiments.model_quality_gpu_v1.resources")
        resources.guard = lambda **kwargs: self.fail("resource/CUDA guard ran before refused reservation")
        resources.lease = lambda value: nullcontext()
        with patch.dict(sys.modules, {supervisor.__name__: supervisor, resources.__name__: resources}), \
             patch.object(p, "setup"), patch.object(c, "RABPY", Path(sys.executable)), \
             patch.object(c, "context", return_value=({}, {})), patch.object(c, "load_inputs", return_value=[]), \
             patch.object(p, "verify_reservation", side_effect=ValueError("missing reservation")), \
             patch.object(c, "snapshot", side_effect=AssertionError("weight snapshot ran")):
            with self.assertRaisesRegex(ValueError, "missing reservation"):
                p.worker(args)
        self.assertEqual(c.read(out / "worker_terminal.json")["status"], "FAILED")


class GrantSchemaTests(Fixture):
    def context_fixture(self):
        pin = "a" * 64; source = str(self.root / "source")
        runtime = {name: "fixture" for name in ("python", "torch", "transformers", "safetensors")}
        sidecars = {name: pin for name in c.original.SIDECARS}; pilot = {"qualified.py": pin}
        datasets = {kind: {"input_path": str(self.root / (kind + ".jsonl")), "input_sha256": pin, "rows": n,
                           "parent_counts": {str(k): v for k, v in parents.items()}} for kind, (n, parents) in c.KINDS.items()}
        grant = {"schema": 1, "record": "BINDING_INTEGRATION_AUX_GRANT", "experiment": "BINDING-INTEGRATION-001",
                 "decision": "APPROVED", "authorization_reference": "synthetic fixture", "limits": c.LIMITS,
                 "arm_path": str(self.root / "arms/integrated"), "arm_weight_sha256": pin,
                 "arm_receipts_sha256": {"ARM.json": pin, "MATERIALIZATION.json": pin},
                 "source_dir": source, "sidecars_sha256": sidecars, "build_gate_path": str(self.gate_path),
                 "build_gate_sha256": pin, "data_use_admission_sha256": pin, "datasets": datasets,
                 "code_sha256": {name: pin for name in c.WORKER_FILES}, "core_code_sha256": {name: pin for name in c.CORE_FILES},
                 "pilot_code_sha256": pilot, "runtime_versions": runtime, "python_path": str(c.RABPY),
                 "output_root": str(self.root / "eval"), "deadline_unix": self.envelope["deadline_unix"]}
        gate = {"status": "PASS_FIXTURE", "source": source, "data_use_admission_sha256": pin}
        baseline = {"runtime_versions": runtime, "source_dir": source, "sidecars_sha256": sidecars,
                    "code_sha256": grant["core_code_sha256"], "pilot_code_sha256": pilot}
        admission = {"status": "ADMITTED_FOR_NAMED_TRAIN_DATA_AND_EXPLORATORY_EVALUATION_SCOPE",
                     "experiment": "BINDING-INTEGRATION-001", "evaluation_scope": {
                         "legacy_binding384": {"input_gold_and_baseline_report_pins": {"DEV_INPUTS.jsonl": {
                             "path": datasets["binding"]["input_path"], "sha256": pin}}},
                         "controlled_diagnostic896": {"input_gold_and_report_pins": {"candidate-v1/DIAGNOSTIC_INPUTS.jsonl": {
                             "path": datasets["diagnostic"]["input_path"], "sha256": pin}}}}}
        values = {self.grant_path.name: grant, self.gate_path.name: gate, "BASELINE_PLAN.json": baseline,
                  "DATA_USE_ADMISSION.json": admission, "SESSION_ENVELOPE.json": self.envelope}
        return grant, gate, values, pin, pilot

    def test_strict_grant_rejects_a_top_level_gold_field(self):
        grant, gate, values, pin, pilot = self.context_fixture()
        with patch.object(c, "read", side_effect=lambda path: values[Path(path).name]), patch.object(c, "sha", return_value=pin), \
             patch.object(c.original, "pilot_code_files", return_value=pilot):
            c.context(self.grant_path, self.gate_path)
            grant["gold_path"] = "forbidden.jsonl"
            with self.assertRaisesRegex(ValueError, "grant/gate"):
                c.context(self.grant_path, self.gate_path)

    def test_approved_grant_cannot_rescue_a_failed_build_gate(self):
        grant, gate, values, pin, pilot = self.context_fixture(); gate["status"] = "FAILED"
        with patch.object(c, "read", side_effect=lambda path: values[Path(path).name]), patch.object(c, "sha", return_value=pin), \
             patch.object(c.original, "pilot_code_files", return_value=pilot):
            with self.assertRaisesRegex(ValueError, "grant/gate"):
                c.context(self.grant_path, self.gate_path)


class CompletionPublicationTests(Fixture):
    def run_fake_controller(self, *, cleanup_failure=False, final_phase_failure=False):
        rows, _ = self.inputs(); before = {"identity": "synthetic"}; training = {"training": "synthetic"}
        arm = self.root / "arm"; self.put(arm / "ARM.json", {})
        self.global_reservation("binding")
        args = SimpleNamespace(kind="binding", attempt=1, grant=self.grant_path, build_gate=self.gate_path)
        supervisor_module = ModuleType("experiments.model_quality_gpu_v1.supervisor")
        out = self.root / "eval/integrated-binding"
        def bounded_child(command, directory, timeout):
            self.assertEqual(timeout, 900); directory.mkdir()
            self.put(directory / "GENERATIONS.json", {"fixture": True})
            self.put(directory / "supervisor.json", {"status": "EXITED", "returncode": 0, "wall_seconds": 1})
            self.put(directory / "worker_terminal.json", {"status": "COMPLETE", "complete": True, "failure": None,
                "before": before, "after": before, "kind": "binding", "grant_sha256": c.sha(self.grant_path),
                "build_gate_sha256": c.sha(self.gate_path), "generations_sha256": c.sha(directory / "GENERATIONS.json")})
            if cleanup_failure:
                path = self.root / "eval/AUX_CONTROLLER_LEASE.json"; owner = c.read(path); owner["nonce"] = "changed"; self.put(path, owner)
            return 0
        supervisor_module.bounded_child = bounded_child
        def phase(*args, **kwargs):
            if final_phase_failure and not (self.root / "eval/AUX_CONTROLLER_LEASE.json").exists():
                raise ValueError("late phase failure")
        original_write = c.write
        def publish(path, value):
            if Path(path).name == "AUX_BINDING.json":
                self.assertFalse((self.root / "eval/AUX_CONTROLLER_LEASE.json").exists())
            return original_write(path, value)
        with patch.dict(sys.modules, {supervisor_module.__name__: supervisor_module}), patch.object(p, "setup"), \
             patch.object(c, "context", return_value=({"arm_path": str(arm)}, {})), patch.object(c, "RABPY", Path(sys.executable)), \
             patch.object(c, "load_inputs", return_value=rows), patch.object(c, "phase_guard", side_effect=phase), \
             patch.object(c, "costs", return_value={"totals": {key: 0 for key in c.LIMITS}}), patch.object(c, "global_costs", return_value={}), \
             patch.object(c, "snapshot", return_value=before), patch.object(run_probe, "arm_snapshot", return_value=training), \
             patch.object(c, "verify_launch", return_value={}), patch.object(p.tempfile, "gettempdir", return_value=str(self.root)), \
             patch.object(c, "write", side_effect=publish), patch("builtins.print"):
            return p.controller(args)

    def test_complete_is_published_only_after_owned_cleanup(self):
        self.assertEqual(self.run_fake_controller(), 0)
        self.assertEqual(c.read(self.root / "eval/integrated-binding/AUX_BINDING.json")["status"], "COMPLETE")

    def test_changed_lease_cannot_leave_a_complete_aux_binding(self):
        with self.assertRaisesRegex(ValueError, "owner changed"):
            self.run_fake_controller(cleanup_failure=True)
        self.assertFalse((self.root / "eval/integrated-binding/AUX_BINDING.json").exists())

    def test_late_phase_failure_cannot_publish_complete(self):
        with self.assertRaisesRegex(ValueError, "late phase"):
            self.run_fake_controller(final_phase_failure=True)
        self.assertFalse((self.root / "eval/integrated-binding/AUX_BINDING.json").exists())


class CompletedReadSetTests(Fixture):
    def artifacts(self):
        out = self.root / "eval/integrated-binding"; out.mkdir()
        arm = self.root / "arm"; source = self.root / "source"
        self.put(arm / "ARM.json", {}); self.put(source / "tokenizer.json", {}); self.put(source / "tokenizer_config.json", {})
        before = {"identity": "synthetic"}; training = {"training": "synthetic"}
        rows = [{"id": "r", "parent_id": "p", "prompt_ids": [1]}]
        grant = {"arm_path": str(arm), "source_dir": str(source)}
        owner = {"owner_pid": 55, "nonce": "fixture", "grant_sha256": c.sha(self.grant_path)}
        parent = self.root / "posttrain/reservations/integrated-binding.json"; self.put(parent, {})
        reservation = {"kind": "binding", "attempt": 1, "grant_sha256": c.sha(self.grant_path), "rows": 1,
                       "forward_calls": 24, "reserved_generated_tokens": 24, "processed_input_tokens": 25,
                       "parent_reservation_sha256": c.sha(parent), "controller_owner": owner,
                       "controller_lease_sha256": hashlib.sha256((json.dumps(owner, indent=2, allow_nan=False) + "\n").encode()).hexdigest()}
        rp = p.reservation_path("binding", 1); self.put(rp, reservation)
        prediction = {"id": "r", "parent_id": "p", "input_tokens": 1, "generated_ids_including_eos": [2],
                      "terminated": True, "prediction": "ok", "generated_tokens": 1, "forward_calls": 1, "processed_input_tokens": 1}
        bundle = {"schema": 1, "record": "BINDING_INTEGRATION_AUX_GENERATIONS", "kind": "binding", "rows": 1,
                  "predictions": [prediction], "eos_ids": [2], "numerics": c.NUMERICS, "gold_file_read": False,
                  "training_permission": False, "extra_warmup_or_qualification_calls": 0,
                  "before": before, "after": before, "grant_sha256": c.sha(self.grant_path), "build_gate_sha256": c.sha(self.gate_path),
                  "actual_budget_charges": {"rows": 1, "generated_tokens_including_eos": 1, "forward_calls": 1, "processed_input_tokens": 1}}
        self.put(out / "GENERATIONS.json", bundle)
        self.put(out / "supervisor.json", {"status": "EXITED", "returncode": 0, "wall_seconds": 1})
        self.put(out / "worker_terminal.json", {"status": "COMPLETE", "complete": True, "failure": None,
            "kind": "binding", "before": before, "after": before, "grant_sha256": c.sha(self.grant_path),
            "build_gate_sha256": c.sha(self.gate_path), "generations_sha256": c.sha(out / "GENERATIONS.json")})
        self.put(out / "PRECHECK.json", {"status": "PRECHECK_PASSED", "kind": "binding", "before": before,
            "numerics": c.NUMERICS, "gold_file_read": False, "model_input_fields": ["prompt_ids"],
            "worker_input_fields": ["id", "parent_id", "prompt", "prompt_ids"]})
        for name in ("launch.json", "spawn.json", "worker_claim.json"):
            self.put(out / name, {"supervisor_pid": 55})
        receipt = {"status": "COMPLETE", "complete": True, "kind": "binding", "attempt": 1, "controller_returncode": 0,
            "worker_returncode": 0, "grant_sha256": c.sha(self.grant_path), "build_gate_sha256": c.sha(self.gate_path),
            "failure": None, "before": before, "after": before, "training_before": training, "training_after": training,
            "arm_receipt_sha256": c.sha(arm / "ARM.json"), "reservation_sha256": c.sha(rp),
            **{stem + "_sha256": c.sha(out / name) for stem, name in (("supervisor", "supervisor.json"),
                ("worker_terminal", "worker_terminal.json"), ("generations", "GENERATIONS.json"))}}
        self.put(out / "AUX_BINDING.json", receipt)
        return out, grant, rows, before, training

    @contextmanager
    def offline_verifier(self, *, mutate=False):
        out, grant, rows, before, training = self.artifacts()
        def snapshot(*args):
            if mutate:
                value = c.read(out / "GENERATIONS.json"); value["late_change"] = True; self.put(out / "GENERATIONS.json", value)
            return before
        class Decoder:
            def __init__(self, *args): pass
            def decode(self, ids): return "ok"
        with patch.object(c, "context", return_value=(grant, {})), patch.object(c, "snapshot", side_effect=snapshot), \
             patch.object(c, "load_inputs", return_value=rows), patch.object(c, "verify_launch", return_value={}), \
             patch.object(c, "phase_guard"), patch.object(c, "costs", return_value={}), patch.object(c, "global_costs", return_value={}), \
             patch.object(run_probe, "arm_snapshot", return_value=training), \
             patch.object(diagnostic_score, "ByteLevelDecoder", Decoder), patch.object(diagnostic_score, "expected_eos", return_value=[2]):
            yield out

    def test_completed_fixture_returns_only_verified_generations_and_custody(self):
        with self.offline_verifier():
            bundle, custody = c.verify_completed(self.root, "binding")
        self.assertEqual(bundle["rows"], 1)
        self.assertNotIn("endpoints", custody)

    def test_late_artifact_change_cannot_become_the_returned_custody_pin(self):
        with self.offline_verifier(mutate=True):
            with self.assertRaises(ValueError):
                c.verify_completed(self.root, "binding")

    def test_retry_directory_blocks_initial_reporter(self):
        with self.offline_verifier():
            (self.root / "eval/integrated-binding-retry1").mkdir()
            with self.assertRaisesRegex(ValueError, "retry reporting"):
                c.verify_completed(self.root, "binding")


if __name__ == "__main__":
    unittest.main()
