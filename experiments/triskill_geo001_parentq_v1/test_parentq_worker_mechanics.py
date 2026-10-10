"""Model-free worker fixtures. No Torch import, model load, or capability evidence.

Run only in an owned hardware test slot with TMP/TEMP under BusinessProject outside
the checkout and PYTHONDONTWRITEBYTECODE=1. The existing tiny-model suite is separate.
"""
import hashlib
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import parentq_worker as worker


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def passing_receipt():
    return {"passed": True, "validation_profile": "FULL_CPU_REGISTRATION", "mechanical_checks_passed": True,
            "design_ready": True, "counts": {"T": 24, "H_ON": 32, "H_OFF": 32}, "canary_n": 8,
            "canary_eligible_rows": 56, "canary_pairwise_different_before_length_filter": 58,
            "delta_H_training_rows": {"screen_contract_rows": 10, "screen_contract_rows_with_abstention_gold": 0,
                "native_prompt_counts": {name: {"rows": 28, "exact_abstention_replies": 28}
                                         for name in ("native_a", "native_b")}}}


def fixture_grant(freeze_sha, attempt_number=1):
    return json.dumps({"experiment": "TRISKILL-GEO-001-PARENTQ", "freeze_sha256": freeze_sha,
                       "attempt_number": attempt_number, "operator_grant": "fixture grant only",
                       "hardware_slot": "model-free fixture"})


class WorkerMechanicsTests(unittest.TestCase):
    def test_preflight_uses_provisional_floor_without_claiming_native_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / "lease.json"
            lock.write_text(json.dumps({"pid": 123, "label": worker.LEASE_LABEL, "token": "t"}), encoding="utf-8")
            geo = types.SimpleNamespace(LOCK=lock, ancestor_pids=lambda: {123},
                preflight=lambda *a, **k: {"host_ram_free_gib": 6.49, "blockers": []})
            record = worker.preflight(geo, "t")
            self.assertEqual(record["status"], "BLOCKED")
            self.assertIn("provisional 6.5", record["blockers"][0])
            self.assertFalse(record["parentq_preload_ram_floor"]["native_lane_measurement"])
            geo.preflight = lambda *a, **k: {"host_ram_free_gib": 6.5, "blockers": []}
            self.assertEqual(worker.preflight(geo, "t")["status"], "RESOURCES_CLEAR")

    def test_bind_geo_hashes_and_parses_one_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            geo_dir = Path(directory)
            freeze = {"dependency_binding": {"roles": {"binding": {"sha256": "a" * 64}}}}
            path = geo_dir / worker.GEO_FREEZE_NAME
            path.write_text(json.dumps(freeze), encoding="utf-8")
            bound = types.SimpleNamespace(load_worker=lambda: geo)
            geo = types.SimpleNamespace(_validate_freeze=lambda *args: (None, bound, "geometry"))
            binding = types.SimpleNamespace(validate_dependencies=lambda value: bound if value == freeze["dependency_binding"] else None)
            pin = digest(path)
            with patch.object(worker, "_exec_pinned", return_value=binding), patch.object(worker, "sha256", side_effect=AssertionError("reopened freeze")):
                self.assertEqual(worker.bind_geo(geo_dir, pin), (geo, bound, "geometry"))

    def test_bootstrap_refusal_persists_before_binding_or_model_import(self):
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory)
            start = {"lease_token": "t", "operator_grant": fixture_grant("f" * 64), "freeze_sha256": "f" * 64,
                     "attempt_number": 1, "supervisor_pid": 123, "experiment": "TRISKILL-GEO-001-PARENTQ"}
            (attempt / worker.START_NAME).write_text(json.dumps(start), encoding="utf-8")
            with patch.object(worker, "_worker_identity"), patch.object(worker, "validate_freeze", side_effect=ValueError("STOP_FIXTURE_PIN")):
                result = worker.run(attempt / worker.OUT_NAME, attempt / "freeze.json", "f" * 64, "t", attempt)
            self.assertEqual(result["status"], "STOP_FIXTURE_PIN")
            self.assertEqual(result["model_load_attempts"], 0)
            self.assertEqual(json.loads((attempt / worker.BLOCKED_NAME).read_bytes()), result)
            self.assertFalse((attempt / worker.OUT_NAME).exists())

    def test_entire_preload_refusal_is_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory)
            with patch.object(worker, "_run_impl", side_effect=RuntimeError("STOP_FIXTURE_PRELOAD")):
                result = worker._run_bound(attempt / worker.OUT_NAME, attempt / "freeze.json", "f" * 64, "t", attempt)
            self.assertEqual(result["status"], "STOP_FIXTURE_PRELOAD")
            self.assertTrue((attempt / worker.BLOCKED_NAME).exists())

    def test_load_marker_survives_failed_load_and_spends_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory)
            worker._write_json(attempt / worker.LOAD_BEGUN_NAME, {"fixture": True})
            with patch.object(worker, "_run_impl", side_effect=RuntimeError("fixture load failed")):
                result = worker._run_bound(attempt / worker.OUT_NAME, attempt / "freeze.json", "f" * 64, "t", attempt)
            self.assertTrue(result["load_begun"])
            self.assertEqual(result["model_load_attempts"], 1)
            self.assertTrue((attempt / worker.INCOMPLETE_NAME).exists())
            self.assertFalse((attempt / worker.BLOCKED_NAME).exists())
            with self.assertRaises(FileExistsError):
                worker._write_json(attempt / worker.LOAD_BEGUN_NAME, {"replacement": True})

    def _real_worker_fixture(self, *, fail_padding=False, overlap_failure=None):
        """Exercise the real worker through a raising fake loader; zero models/forwards."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            attempt, packet, geo_dir = root / "attempt", root / "packet", root / "geo"
            for folder in (attempt, packet, geo_dir):
                folder.mkdir()
            source = worker.HERE / "parentq_worker.py"
            module = worker._exec_pinned(source, digest(source), "mechanical_no_model")
            (packet / "parentq_worker.py").write_bytes(source.read_bytes())
            module.HERE, module.GEO_DIR = packet, geo_dir
            module._parentq_source_binding = (str(packet / "parentq_worker.py"), digest(packet / "parentq_worker.py"))
            module._check_output_location = worker._check_output_location
            rows = [types.SimpleNamespace(row_id=name, to_dict=lambda name=name: {"row_id": name})
                    for name in ("t", "on", "off")]
            bank = dict(zip(("T", "H_ON", "H_OFF"), ([row] for row in rows)))
            (packet / worker.BANK_JSON).write_text(json.dumps([row.to_dict() for row in rows]), encoding="utf-8")
            (packet / worker.CANARY_JSON).write_text(json.dumps({"rows": [{"row_id": "c"}]}), encoding="utf-8")
            small_weight = geo_dir / "fixture.bin"
            small_weight.write_bytes(b"not model weights")
            archive_path = geo_dir / "geometry.npz"
            archive_path.write_bytes(b"fake archive, no native geometry")
            (geo_dir / "geometry.json").write_text(json.dumps({"status": "PREPARED", "geometry_sha256": digest(archive_path)}), encoding="utf-8")
            row_module = types.SimpleNamespace(build_bank=lambda benchmark: bank, all_rows=lambda bank: rows)
            decide = types.SimpleNamespace(decide=lambda *a, **k: {
                "status": "PARENTS_QUALIFIED" if overlap_failure is not None else "STOPPED_BEFORE:padding"})
            safety = types.SimpleNamespace(validate_runtime_root=lambda path: path.resolve(),
                                           RUNTIME_ROOT=root,
                                           validate_readiness=lambda *a: {"fixture": True},
                                           check_runtime_ownership=lambda *a: None)
            module._exec_pinned = lambda path, sha, name: {"rows": row_module, "decide": decide, "safety": safety}[name]
            if overlap_failure is not None:
                def compare(*args):
                    if isinstance(overlap_failure, Exception):
                        raise overlap_failure
                    return {"status": overlap_failure, "differences": [], "diagnostic_errors": ["fixture audit unavailable"],
                            "select_outcome": False, "retry_policy": "PENDING_OPERATOR_DECISION"}
                module.compare_attempt_overlap = compare
            caps = worker.allocation(sum(len(ids) for _, _, ids in worker.schedule_units(bank, [{"row_id": "c"}])), 288, 32)
            module.validate_freeze = lambda *a: {"files": {name: "fixture" for name in ("parentq_rows.py", "parentq_decide.py", "parentq_safety.py")}, "native_allocation": caps, "runtime_root": str(root)}
            module.preflight = lambda *a: {"blockers": [], "lease": {"pid": os.getpid()}}
            geo = types.SimpleNamespace(INPUT_SLOTS=288, OUTPUT_SLOTS=32, GEOMETRY_JSON="geometry.json",
                                        GEOMETRY_NPZ="geometry.npz", GPU_MEMORY_FRACTION=0.78, PRELOAD_GPU_GIB=6,
                                        ACTIVE_GPU_GIB=0.5, ACTIVE_RAM_GIB=1.5, EOS_TOKEN_IDS=[99])
            benchmark = types.SimpleNamespace(build_development=lambda: [types.SimpleNamespace(row_id="fixture-mechanics", system="fixture", user="fixture")])
            bound = types.SimpleNamespace(load_local=lambda role: benchmark, verify_current=lambda: None,
                load_renderer=lambda: types.SimpleNamespace(template_selftest=lambda: None, render=lambda *a: "fixture"), receipt=lambda: {"fixture": True})
            geometry = types.SimpleNamespace(SOURCES={"BASE": (small_weight, digest(small_weight))}, LAYER=14)
            start = {"lease_token": "t", "operator_grant": fixture_grant("f"), "freeze_sha256": "f",
                     "attempt_number": 1, "supervisor_pid": os.getpid(), "runtime_root": str(root),
                     "experiment": "TRISKILL-GEO-001-PARENTQ"}
            worker._write_json(attempt / worker.START_NAME, start)
            class Archive:
                def __enter__(self): return {"basis": []}
                def __exit__(self, *args): pass
            fake_numpy = types.ModuleType("numpy")
            fake_numpy.load = lambda *a, **k: Archive()
            fake_torch = types.ModuleType("torch")
            fake_torch.__version__, fake_torch.float32 = "fixture", "float32"
            fake_torch.set_num_threads = lambda *a: None
            fake_torch.use_deterministic_algorithms = lambda *a: None
            fake_torch.as_tensor = lambda *a, **k: []
            fake_torch.backends = types.SimpleNamespace(cuda=types.SimpleNamespace(matmul=types.SimpleNamespace()), cudnn=types.SimpleNamespace())
            fake_torch.cuda = types.SimpleNamespace(init=lambda: None, set_per_process_memory_fraction=lambda *a: None,
                                                   mem_get_info=lambda: (8 << 30, 8 << 30), is_initialized=lambda: False)
            fake_transformers = types.ModuleType("transformers")
            fake_transformers.__version__ = "fixture"
            called = []
            def fake_load(*a, **k):
                called.append((attempt / worker.LOAD_BEGUN_NAME).is_file())
                if fail_padding:
                    model = types.SimpleNamespace(generation_config=types.SimpleNamespace(eos_token_id=[99]),
                        parameters=lambda: iter([types.SimpleNamespace(device="fixture")]),
                        model=types.SimpleNamespace(layers={14: types.SimpleNamespace(register_forward_hook=lambda hook: types.SimpleNamespace(remove=lambda: None))}))
                    model.eval = lambda: model
                    return model
                raise RuntimeError("fixture loader stopped, no model exists")
            fake_transformers.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=fake_load)
            class Tokenizer:
                eos_token_id = 99
                def __call__(self, *a, **k): return {"input_ids": [1]}
            fake_transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **k: Tokenizer())
            geo.HookState, geo.make_hook = lambda *a: types.SimpleNamespace(), lambda *a: "fake hook"
            def decode(*a, **k):
                tokens = [1, 2, 5, 4] if k["padded"] else [1, 2, 3, 4]
                return {"tokens": tokens, "raw_tokens": tokens, "scored_tokens": tokens,
                        "margins": [0.1, 0.2, 0.3, 0.4], "ended": False}
            geo.decode = decode
            with patch.dict(sys.modules, {"numpy": fake_numpy, "torch": fake_torch, "transformers": fake_transformers}):
                result = module._run_bound(attempt / worker.OUT_NAME, packet / "freeze.json", "f", "t", attempt,
                                           _bind_geo=lambda: (geo, bound, geometry))
            marker = json.loads((attempt / worker.LOAD_BEGUN_NAME).read_bytes())
            events = [json.loads(line) for line in (attempt / worker.PROGRESS_NAME).read_text(encoding="utf-8").splitlines()]
            return result, called, marker, events, (attempt / worker.INCOMPLETE_NAME).is_file(), (attempt / worker.OUT_NAME).is_file()

    def test_real_preload_path_marks_before_fake_loader_without_importing_torch(self):
        result, called, marker, events, incomplete, terminal = self._real_worker_fixture()
        self.assertEqual(called, [True], result)
        self.assertEqual(result["model_loads"], 0)
        self.assertEqual(result["model_load_attempts"], 1)
        self.assertTrue(incomplete)
        self.assertEqual(marker["attempt_number"], 1)
        self.assertEqual(marker["freeze_sha256"], "f")
        self.assertTrue(all("utc" in e and "elapsed_seconds" in e for e in events))

    def test_padding_failure_keeps_both_raw_traces_and_margins_without_extra_decode(self):
        result, called, marker, events, incomplete, terminal = self._real_worker_fixture(fail_padding=True)
        self.assertEqual(result["status"], "STOP_NATIVE_MECHANICS_PADDING", result)
        trace = result["development_padding_parity_traces"][0]
        self.assertEqual(trace["unpadded"]["raw_tokens"], [1, 2, 3, 4])
        self.assertEqual(trace["padded"]["raw_tokens"], [1, 2, 5, 4])
        self.assertEqual(trace["unpadded"]["margins"], [0.1, 0.2, 0.3, 0.4])
        self.assertFalse(trace["tokens_identical"])
        self.assertEqual(sum(e["event"] == "padding_parity_trace" for e in events), 2)

    def test_unverified_overlap_never_persists_a_usable_qualified_verdict(self):
        for failure, expected in (("UNVERIFIED_OVERLAP", "UNVERIFIED_OVERLAP"),
                                  (ValueError("fixture overlap audit failure"), "STOP_ATTEMPT_OVERLAP_ERROR")):
            result, _, _, _, incomplete, terminal = self._real_worker_fixture(overlap_failure=failure)
            self.assertEqual(result["status"], expected)
            self.assertFalse(result["decision_usable"])
            self.assertEqual(result["verdict"]["status"], expected)
            self.assertEqual(result["verdict"]["diagnostic_verdict"]["status"], "PARENTS_QUALIFIED")
            self.assertFalse(result["verdict"]["decision_usable"])
            self.assertTrue(terminal)
            self.assertFalse(incomplete)

    def test_create_only_json_is_flushed_and_fsynced_before_return(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "marker.json"
            with patch.object(worker.os, "fsync", wraps=os.fsync) as sync:
                worker._write_json(path, {"fixture": True})
            sync.assert_called_once()
            self.assertEqual(json.loads(path.read_bytes()), {"fixture": True})

    def test_decision_error_cannot_escape_finally_helper(self):
        decide = types.SimpleNamespace(decide=lambda *a, **k: (_ for _ in ()).throw(ValueError("fixture decision failed")))
        verdict, error = worker.safe_decide(decide, None, {}, [], {}, {}, "fixture stop")
        self.assertEqual(verdict["status"], "STOP_DECISION_ERROR")
        self.assertFalse(verdict["decision_usable"])
        self.assertEqual(error["type"], "ValueError")

    def test_canary_first_divergence_metadata_is_diagnostic(self):
        tokenizer = lambda text, **kwargs: {"input_ids": [10, 11]}
        row = {"stored": {"BASE": "expected"}}
        decoded = {"scored_tokens": [10, 12, 99], "margins": [0.1, 0.2, 0.3],
                   "reference_token_margins": [0.0, 0.7, 0.0]}
        record = worker.canary_divergence(row, "BASE", "other", decoded, tokenizer, [99])
        self.assertEqual(record["first_divergence_index"], 1)
        self.assertEqual(record["first_divergence_observed_margin"], 0.2)
        self.assertEqual(record["first_divergence_chosen_minus_reference_logit"], 0.7)
        self.assertTrue(record["reference_tokens_reconstructed_from_stored_text"])
        self.assertEqual(worker.canary_divergence(row, "BASE", " expected ", decoded, tokenizer, [99]), {"text_matches": True})

    def test_attempt_overlap_detects_text_or_token_difference_without_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prior, current = root / "prior", root / "current"
            prior.mkdir()
            current.mkdir()
            worker._write_json(prior / worker.LOAD_BEGUN_NAME, {})
            worker._write_json(prior / worker.START_NAME, {"experiment": "TRISKILL-GEO-001-PARENTQ", "freeze_sha256": "f"})
            events = [{"event": "generation", "arm": "BASE", "row_id": "r", "text": "a", "scored_tokens": [1]}]
            (prior / worker.PROGRESS_NAME).write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")
            record = worker.compare_attempt_overlap(current, {"BASE": {"r": "a"}}, {"BASE": {"r": {"scored_tokens": [2]}}}, "f")
            self.assertEqual(record["status"], "NONDETERMINISM_OBSERVED")
            self.assertFalse(record["select_outcome"])
            self.assertEqual(record["retry_policy"], "PENDING_OPERATOR_DECISION")
            self.assertEqual(len(record["differences"]), 1)
            identical = worker.compare_attempt_overlap(current, {"BASE": {"r": "a"}}, {"BASE": {"r": {"scored_tokens": [1]}}}, "f")
            self.assertEqual(identical["status"], "NO_DIFFERENCE_ON_OBSERVED_OVERLAP")
            self.assertEqual(worker.compare_attempt_overlap(current, {}, {}, "other")["comparisons"], [])

    def test_unreadable_or_missing_prior_trace_is_a_hold_not_determinism(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prior, current = root / "prior", root / "current"
            prior.mkdir()
            current.mkdir()
            worker._write_json(prior / worker.LOAD_BEGUN_NAME, {})
            start_path = prior / worker.START_NAME
            for start in (None, b"{", json.dumps({"experiment": "TRISKILL-GEO-001-PARENTQ", "freeze_sha256": "f"}).encode()):
                if start is not None:
                    start_path.write_bytes(start)
                record = worker.compare_attempt_overlap(current, {}, {}, "f")
                self.assertEqual(record["status"], "UNVERIFIED_OVERLAP")
                self.assertTrue(record["diagnostic_errors"])
                self.assertFalse(record["select_outcome"])
            progress = prior / worker.PROGRESS_NAME
            for content in ("{", "null\n", '{"event":"generation"}\n', "{\n{}\n"):
                progress.write_text(content, encoding="utf-8")
                record = worker.compare_attempt_overlap(current, {}, {}, "f")
                self.assertEqual(record["status"], "UNVERIFIED_OVERLAP")
                self.assertFalse(record["differences"])

    def test_supervisor_attempt_binding_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory)
            start = {"lease_token": "t", "operator_grant": fixture_grant("f"), "freeze_sha256": "f", "attempt_number": 1,
                     "experiment": "TRISKILL-GEO-001-PARENTQ"}
            (attempt / worker.START_NAME).write_text(json.dumps(start), encoding="utf-8")
            self.assertEqual(worker.verify_supervisor_start(attempt, "t", "f"), start)
            with self.assertRaisesRegex(RuntimeError, "STOP_SUPERVISOR_ATTEMPT_BINDING"):
                worker.verify_supervisor_start(attempt, "t", "g")

    def test_pinned_cpu_receipt_truth_counts_and_gpu_lease_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            here = Path(directory)
            for name in worker.REQUIRED_FILES:
                (here / name).write_text(name, encoding="utf-8")
            receipt_path = here / "CPU.json"
            geo_pin = json.loads((worker.GEO_DIR / worker.GEO_FREEZE_NAME).read_bytes())["supervisor_dependencies"]["gpu_lease"]
            freeze = {"files": {name: digest(here / name) for name in worker.REQUIRED_FILES},
                      "runtime_root": str(worker.RUNTIME_ROOT.resolve()),
                      "geo_freeze": {"sha256": worker.GEO_FREEZE_SHA256},
                      "supervisor_dependencies": {"gpu_lease": geo_pin}}
            freeze_path = here / "FREEZE.json"
            def validate(receipt):
                receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
                freeze["cpu_validation"] = {"path": str(receipt_path.resolve()), "sha256": digest(receipt_path), "passed": True}
                freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
                return worker.validate_freeze(freeze_path, digest(freeze_path), here)
            self.assertEqual(validate(passing_receipt())["supervisor_dependencies"]["gpu_lease"], geo_pin)
            for mutate in (lambda r: r.update(passed=False), lambda r: r.update(canary_eligible_rows=55),
                           lambda r: r.update(validation_profile="LIGHTWEIGHT_NO_TORCH"), lambda r: r.update(design_ready=False),
                           lambda r: r.update(canary_pairwise_different_before_length_filter=57),
                           lambda r: r["delta_H_training_rows"].update(screen_contract_rows=9),
                           lambda r: r["delta_H_training_rows"]["native_prompt_counts"]["native_b"].update(rows=27)):
                receipt = passing_receipt()
                mutate(receipt)
                with self.assertRaisesRegex(ValueError, "STOP_CPU_VALIDATION_FACTS_NOT_VERIFIED"):
                    validate(receipt)
            freeze["supervisor_dependencies"]["gpu_lease"] = {**geo_pin, "sha256": "0" * 64}
            with self.assertRaisesRegex(ValueError, "STOP_GPU_LEASE_PIN_DIFFERS_FROM_GEO"):
                validate(passing_receipt())

    def test_no_refusal_output_can_be_written_inside_source(self):
        with patch.object(worker, "_write_json") as write:
            result = worker._refusal_receipt(worker.HERE, "f", ValueError("fixture refusal"))
        write.assert_not_called()
        self.assertEqual(result["receipt_write_error"]["message"], "STOP_WORKER_OUTPUT_LOCATION")

    def test_native_runtime_root_cannot_rotate_with_a_new_freeze(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "freeze.json"
            path.write_text(json.dumps({"runtime_root": str(Path(directory).resolve())}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "STOP_NONCANONICAL_NATIVE_RUNTIME_ROOT"):
                worker.validate_freeze(path, digest(path))


if __name__ == "__main__":
    unittest.main()
