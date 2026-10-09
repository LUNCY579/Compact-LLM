"""Small authored admission/source/terminal controls; no heavy native audit."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import ast
import hashlib
import json
import sys
import tempfile
import time
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arithmetic_raw_auditor as A


def fixture():
    inventories = {name: {"AUTHORED_" + name + "_%02d" % n: "a" * 64 for n in range(count)}
                   for name, count in (("source", 3), ("runtime", 65), ("decoder", 17), ("data", 4), ("model", 7))}
    row = {"row_id": "a" * 64, "prompt_sha256": "b" * 64, "prompt_ids": [1], "prompt_tokens": 1, "attention_mask": [1]}
    rows = [{**row, "row_id": A.digest(["AUTHORED", n])} for n in range(64)]
    profile = {"id": "SHORT_M0_PHYSICAL_RAM_COHERENT_EXTERNAL_CHECK_V2"}
    scope = {"contract": "synth-short-M0-physical-RAM-coherent-monitor-staged-2", "rows": rows,
             "short_RAM_monitor_profile": profile, "batch_selector": {"source_kind": A.KIND, "ordinal": 1},
             "source_packet_sha256": A.digest(rows), "model_role": "fresh_dense_M0_zero_B_only_no_STATE1_or_study_checkpoint",
             "dose": {"rows": 64, "maximum_forward_calls": 2050, "context_cap": 512, "maximum_generated_tokens": 32},
             **{key: deepcopy(inventories[name]) for name, key in A.SECTIONS.items()}}
    registration = {"id": A.REGISTRATION_ID, "schema_version": 1, "source_commit": "a" * 40,
                    "profile_body_sha256": A.digest(profile), "inventories": inventories,
                    "counts": {name: len(rows) for name, rows in inventories.items()}}
    registration["sha256"] = A.digest(registration)
    plan = {"fixture_only": False, "rows": 512, "parent_components": 128,
            "batches": {A.KIND: [{"rows": rows, "source_packet_sha256": A.digest(rows)} for _ in range(8)]}}
    plan["sha256"] = A.digest(plan)
    return scope, plan, registration


class Controls(unittest.TestCase):
    def test_existing_receipt_replay_allows_only_valid_UTC_and_seal_changes_never_writes(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            path = Path(directory) / "authored_saved_audit.json"
            saved = {"kind": "AUTHORED_AUDIT_ONLY", "fixture_only": True, "checks_passed": 12,
                     "source_packet_sha256": "f" * 64, "recorded_at_utc": "2026-10-09T00:00:00+00:00"}
            def write(value):
                value = deepcopy(value); value["sha256"] = A.digest({k: v for k, v in value.items() if k != "sha256"})
                raw = A.canonical(value).encode("ascii"); path.write_bytes(raw)
                return value, {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            saved, ref = write(saved); before = path.read_bytes(); calls = []
            request = {"config_path": str(HERE / "AUTHORED_config.json"), "raw_directory": str(HERE / "AUTHORED_raw"),
                       "preserved_directory": str(HERE / "AUTHORED_preserved"), "expected_selector": [A.KIND, 1],
                       "expected_config_file_sha256": "a" * 64, "expected_preservation_body_sha256": "b" * 64,
                       "inventory_registration_path": str(HERE / "AUTHORED_inventory.json"),
                       "expected_inventory_registration_file_sha256": "c" * 64, "expected_profile_body_sha256": "d" * 64,
                       "expected_plan_file_sha256": "e" * 64, "expected_source_commit": "a" * 40,
                       "terminal_preservation_signal": A.SIGNAL}
            def fresh(**kwargs):
                calls.append(kwargs); self.assertIsNone(kwargs["output_path"])
                value = {**saved, "recorded_at_utc": "2026-10-09T01:00:00+00:00"}
                value["sha256"] = A.digest({k: v for k, v in value.items() if k != "sha256"}); return value
            self.assertEqual(saved, A._CPU_audit_existing(ref, request, fresh, capability=A._CPU_capability()))
            self.assertEqual(before, path.read_bytes()); self.assertEqual(1, len(calls))
            self.assertEqual(saved, A._CPU_audit_existing({**ref, "value_sha256": A.digest(saved)}, request, fresh,
                                                        capability=A._CPU_capability()))
            with self.assertRaisesRegex(ValueError, "whole audit value pin"):
                A._CPU_audit_existing({**ref, "value_sha256": "0" * 64}, request,
                                     lambda **kwargs: (_ for _ in ()).throw(AssertionError("actor before value pin")),
                                     capability=A._CPU_capability())
            for field, value in (("checks_passed", 13), ("source_packet_sha256", "0" * 64),
                                 ("recorded_at_utc", 0), ("recorded_at_utc", "not-a-date"),
                                 ("recorded_at_utc", "2026-10-09T01:00:00+01:00")):
                def changed(**kwargs):
                    result = fresh(**kwargs); result[field] = value
                    result["sha256"] = A.digest({k: v for k, v in result.items() if k != "sha256"}); return result
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    A._CPU_audit_existing(ref, request, changed, capability=A._CPU_capability())
                self.assertEqual(before, path.read_bytes())
            forbidden = lambda **kwargs: (_ for _ in ()).throw(AssertionError("actor before selected request/ref"))
            with self.assertRaises(ValueError):
                A._CPU_audit_existing(ref, {**request, "output_path": str(path)}, forbidden, capability=A._CPU_capability())
            with self.assertRaises(ValueError):
                A._CPU_audit_existing({**ref, "sha256": "0" * 64}, request, forbidden, capability=A._CPU_capability())
            with self.assertRaisesRegex(ValueError, "staged arithmetic"):
                A.audit_existing(ref, request)

    def test_exact_admission_does_not_infer_native_authority_and_refuses_source_kind_order_and_inventory_drift(self):
        scope, plan, registration = fixture()
        args = {"selected": (A.KIND, 1), "source_commit": "a" * 40,
                "profile_sha256": registration["profile_body_sha256"]}
        value = A.admit(scope, plan, registration, deepcopy(scope), **args)
        self.assertFalse(value["native_PASS"])
        for selected in (("task_guard_DEV", 1), (A.KIND, 9), (A.KIND, True)):
            with self.subTest(selected=selected), self.assertRaises(ValueError):
                A.admit(scope, plan, registration, scope, **{**args, "selected": selected})
        for change in (lambda s: s["rows"].reverse(), lambda s: s["runtime_pins"].pop(next(iter(s["runtime_pins"]))),
                       lambda s: s["dose"].update(maximum_forward_calls=2049)):
            altered = deepcopy(scope); change(altered)
            with self.subTest(change=change), self.assertRaises(ValueError):
                A.admit(altered, plan, registration, scope, **args)

    def test_full_original_raw_trace_decode_dose_model_resources_ownership_block_is_identical(self):
        tree, changes, text = A.prepared_source()
        original = A.ORIGINAL.read_text()
        start = " events=jsonl(leaf/'events.jsonl')"; stop = " def compact(values):"
        self.assertEqual(original[original.index(start):original.index(stop)], text[text.index(start):text.index(stop)])
        self.assertEqual(8, len(changes))
        self.assertEqual(5, len(tree.body))
        self.assertIn("counts.get(\"data\")==4", text)
        self.assertIn("str(CAPTURE/'arithmetic_capture_owned.py')", text)

    def test_original_early_terminal_checks_refuse_partial_failures_oracle_fields_and_false_source_commit(self):
        tree, _, _ = A.prepared_source()
        node = next(n for n in tree.body if n.name == "_early_terminal_checks")
        ns = {"digest": A.digest, "ROW_FIELDS": {"row_id", "prompt_sha256", "prompt_ids", "prompt_tokens", "attention_mask"},
              "CONTRACT_VERSION": "synth-short-M0-physical-RAM-coherent-monitor-staged-2",
              "contract": SimpleNamespace(AUTHORIZATION={"scope": "AUTHORED_ARITHMETIC"})}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(A.ORIGINAL), "exec"), ns)
        scope, _, _ = fixture(); scope.update(authorization=ns["contract"].AUTHORIZATION, profile="BF16_RESIDENT_FP32_ACTIVATIONS_QV8_V1")
        worker = {"status": "COMPLETE_SCOPED_SHORT_M0_RAW_COLLECTION_UNSCORED", "error": None, "source_commit": "a" * 40,
                  "collection": {"rows": [{"row_id": r["row_id"]} for r in scope["rows"]], "source_packet_sha256": scope["source_packet_sha256"]}}
        sup = {"status": "SUPERVISED_WORKER_EXITED_ZERO", "returncode": 0, "partial_outputs_frozen": True,
               "preflight": {"source_commit": "a" * 40}}
        preservation = {"source_commit": "a" * 40}
        def call(s, w, u, p):
            ns["_early_terminal_checks"](s, w, u, p, (A.KIND, 1), A.require)
        call(scope, worker, sup, preservation)
        for mutate in (lambda s, w, u, p: w.update(error={"error": "failed"}),
                       lambda s, w, u, p: w["collection"]["rows"].pop(),
                       lambda s, w, u, p: s["rows"][0].update(labels=[1]),
                       lambda s, w, u, p: p.update(source_commit="b" * 40),
                       lambda s, w, u, p: u.update(status="ADMISSION_DENIED")):
            values = [deepcopy(v) for v in (scope, worker, sup, preservation)]; mutate(*values)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError): call(*values)

    def test_missing_signal_anchors_or_integration_denies_before_any_evidence_or_package_execution(self):
        kwargs = {"expected_selector": (A.KIND, 1), "expected_config_file_sha256": "a" * 64,
                  "expected_preservation_body_sha256": "b" * 64, "inventory_registration_path": "unused",
                  "expected_inventory_registration_file_sha256": "c" * 64, "expected_profile_body_sha256": "d" * 64,
                  "expected_plan_file_sha256": "e" * 64, "expected_source_commit": "a" * 40}
        with patch.object(Path, "open", side_effect=AssertionError("unexpected evidence read")):
            with self.assertRaisesRegex(ValueError, "signal before reads"):
                A.audit_terminal("unused", "unused", "unused", **kwargs)
            with self.assertRaisesRegex(ValueError, "anchors required"):
                A.audit_terminal("unused", "unused", "unused", **{**kwargs, "expected_plan_file_sha256": None}, terminal_preservation_signal=A.SIGNAL)
            with self.assertRaisesRegex(ValueError, "staged arithmetic"):
                A.audit_terminal("unused", "unused", "unused", **kwargs, terminal_preservation_signal=A.SIGNAL)
        self.assertFalse(any(name in sys.modules for name in ("torch", "transformers", "tokenizers", "ctypes", "psutil")))

    def test_selected_JSON_descriptor_drift_and_nonfinite_are_refused(self):
        with tempfile.TemporaryDirectory(dir=HERE) as directory:
            path = Path(directory) / "authored.json"; raw = b'{"value":1}'
            path.write_bytes(raw); ref = {"path": str(path), "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
            self.assertEqual({"value": 1}, A.read_ref(ref))
            path.write_bytes(raw + b" ")
            with self.assertRaises(ValueError): A.read_ref(ref)
            for raw in (b'{"value":1,"value":2}', b'{"value":1e999}', b'{"value":NaN}'):
                path.write_bytes(raw); ref.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
                with self.assertRaises(ValueError): A.read_ref(ref)
            def write(name, value):
                file = Path(directory) / name; data = A.canonical(value).encode("ascii"); file.write_bytes(data)
                return {"path": str(file), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            data_refs = {name: write(name + ".json", {"AUTHORED": name}) for name in ("projection", "prompt", "policy", "ledger")}
            plan = {"measurement_reference": data_refs["projection"], "prompt_reference": data_refs["prompt"],
                    "policy_reference": data_refs["policy"], "ledger_reference": data_refs["ledger"]}
            plan_ref = write("plan.json", plan); plan_ref["value_sha256"] = A.digest(plan)
            joined = A.arithmetic_references(plan, plan_ref, plan_ref["sha256"])
            self.assertEqual(A.REF_FIELDS, set(joined["plan"]))
            with self.assertRaises(ValueError):
                A.arithmetic_references(plan, {**plan_ref, "value_sha256": "0" * 64}, plan_ref["sha256"])
            Path(data_refs["prompt"]["path"]).write_bytes(b"REPLACED_AUTHORED")
            with self.assertRaises(ValueError): A.arithmetic_references(plan, plan_ref, plan_ref["sha256"])
            installed = Path(directory) / "capture"; installed.mkdir()
            for name in ("arithmetic_capture_contract.py", "arithmetic_capture_owned.py", "arithmetic_capture_pipeline.py"):
                (installed / name).write_bytes(b"raise AssertionError('source must not execute')\n")
            sources = {str(file.resolve()): hashlib.sha256(file.read_bytes()).hexdigest() for file in installed.iterdir()}
            dependencies = {str(Path(A.__file__).resolve()): hashlib.sha256(Path(A.__file__).read_bytes()).hexdigest(),
                            str(A.ORIGINAL.resolve()): A.ORIGINAL_SHA256}
            registered = {"source_commit": "a" * 40, "inventories": {"source": sources}, "auditor_dependency_pins": dependencies}
            with patch.object(A, "CAPTURE", installed):
                pin = A.selected_sources(registered, "a" * 40)
                self.assertEqual(sources[str((installed / "arithmetic_capture_contract.py").resolve())], pin)
                with patch.object(A, "HERE", A.EXPECTED):
                    with self.assertRaisesRegex(ValueError, "bytes before execution"):
                        A._capture_module("0" * 64)
                (installed / "arithmetic_capture_owned.py").write_bytes(b"CHANGED")
                with self.assertRaisesRegex(ValueError, "source before imports"):
                    A.selected_sources(registered, "a" * 40)
            package = Path(directory) / "publication_fixture"; (package / "evidence").mkdir(parents=True)
            result = {"kind": "AUTHORED_POST_COMPOSITION_ONLY", "native_PASS": False,
                      "arithmetic_data_reference_closure": joined}
            result["sha256"] = A.digest(result)
            target = package / "evidence/authored_raw_independent_audit_001.json"
            with patch.object(A, "PACKAGE", package):
                returned = A.publish_result(target, "authored_raw", result)
                self.assertEqual(returned, json.loads(target.read_bytes()))
                self.assertEqual(A.digest({k: v for k, v in returned.items() if k != "sha256"}), returned["sha256"])
                with self.assertRaises(ValueError): A.publish_result(target, "authored_raw", result)
                with self.assertRaises(ValueError): A.publish_result(Path(directory) / "wrong.json", "authored_raw", result)


if __name__ == "__main__":
    files = ("arithmetic_raw_auditor.py", "test_arithmetic_raw_auditor.py")
    before = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in files}
    started = time.monotonic(); result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls))
    value = {"kind": "AUTHORED_ARITHMETIC_RAW_AUDITOR_CPU_CONTROLS001", "tests": result.testsRun,
             "success": result.wasSuccessful(), "errors": len(result.errors), "failures": len(result.failures),
             "seconds": time.monotonic() - started, "source_before": before,
             "source_after": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in files},
             "heavy_audit_or_decoder_model_or_native_API_executed": False, "native_PASS": False,
             "limits": ["Pure admission and source-preserved early64 terminal controls only; no real capture exists.",
                        "The complete inherited heavy audit is conditionally implemented but unexecuted."]}
    value["sha256"] = A.digest(value)
    (HERE / "CPU_CONTROLS001.json").write_bytes((A.canonical(value) + "\n").encode("ascii"))
    raise SystemExit(0 if result.wasSuccessful() else 1)
