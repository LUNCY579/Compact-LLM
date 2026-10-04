"""TRAIN-only binding, mocked worker and unchanged guard controls; no native run."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
import inference_contract as contract
import inference_qual_worker as worker
import native_contract as native
from inference_component import ROW_FIELDS, digest
from native_worker import Journal
from test_inference_component import FakeTorch, Model, Parameter, Logits, Scalar, row


class IsolatedCLIControls(unittest.TestCase):
    def test_actual_isolated_dispatch_argv_imports_worker_before_help(self):
        result = subprocess.run([sys.executable, "-I", "-B", str(PACKAGE / "inference_qual_worker.py"), "--help"],
                                text=True, capture_output=True, timeout=30)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("--output-dir", result.stdout)
        self.assertNotIn("ModuleNotFoundError", result.stderr)


def source_fixture():
    records, pools = [], {"M1": [], "M2": []}
    for arm_index, arm in enumerate(("M1", "M2")):
        for skill_index, skill in enumerate(("R", "S")):
            for candidate in range(2):
                rid = digest([arm, skill, candidate])
                length = 2 + candidate
                records.append({"row_id": rid, "arm": arm, "split": "train", "skill": skill,
                                "target": "MUST_NEVER_BE_READ", "verification": {"gold": "PRIVATE"}})
                pools[arm].append({"row_id": rid, "prompt_ids": [5] * length, "prompt_tokens": length,
                                   "prompt_text_sha256": digest(["prompt", rid]), "completion_ids": [999],
                                   "labels": [999], "input_ids": [999], "target": "PRIVATE"})
    return records, pools


class ContractTests(unittest.TestCase):
    def test_imports_are_framework_free_and_do_not_open_model_or_banks(self):
        code = ("import sys; sys.path.insert(0,sys.argv[1]); import inference_contract,inference_qual_worker; "
                "assert not any(x in sys.modules for x in ['torch','transformers','rs_verifier','cpu_tokenizer']); "
                "print('NO_FRAMEWORK_OR_SCORER_IMPORT')")
        result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(PACKAGE)], text=True,
                                capture_output=True, timeout=30, check=True)
        self.assertEqual("NO_FRAMEWORK_OR_SCORER_IMPORT", result.stdout.strip())

    def test_longest_four_train_prompts_have_only_public_fields_no_targets(self):
        records, pools = source_fixture()
        before = deepcopy((records, pools))
        selected = contract.select_prompts(records, pools)
        self.assertEqual(before, (records, pools))
        self.assertEqual([("M1", "R"), ("M1", "S"), ("M2", "R"), ("M2", "S")],
                         [(x["arm"], x["skill"]) for x in selected["selection_bindings"]])
        self.assertTrue(all(set(x) == ROW_FIELDS and x["prompt_tokens"] == 3 for x in selected["rows"]))
        self.assertNotIn("PRIVATE", json.dumps(selected))
        self.assertNotIn("MUST_NEVER_BE_READ", json.dumps(selected))
        self.assertEqual(digest(selected["rows"]), selected["source_packet_sha256"])

    def test_tie_break_selection_is_stable_under_source_order_changes(self):
        records, pools = source_fixture()
        for pool in pools.values():
            for candidate in pool: candidate.update(prompt_ids=[5, 6], prompt_tokens=2)
        first = contract.select_prompts(records, pools)
        records.reverse()
        for pool in pools.values(): pool.reverse()
        self.assertEqual(first, contract.select_prompts(records, pools))
        for selected, binding in zip(first["rows"], first["selection_bindings"]):
            candidates = [r["row_id"] for r in records if r["arm"] == binding["arm"] and r["skill"] == binding["skill"]]
            self.assertEqual(min(candidates), selected["row_id"])

    def test_target_changes_do_not_change_selection_or_generated_inputs(self):
        records, pools = source_fixture()
        selected = contract.select_prompts(records, pools)
        for record in records: record["target"] = object()
        for pool in pools.values():
            for candidate in pool:
                candidate.update(completion_ids=object(), labels=object(), input_ids=object())
        self.assertEqual(selected, contract.select_prompts(records, pools))

    def test_dev_rs_missing_rows_duplicates_and_overlong_longest_cannot_be_filtered(self):
        mutations = (lambda records, pools: records[0].update(split="dev"),
                     lambda records, pools: records[0].update(skill="RS"),
                     lambda records, pools: records.append(deepcopy(records[0])),
                     lambda records, pools: pools["M1"].pop(),
                     lambda records, pools: pools["M1"][1].update(prompt_ids=[5] * 481, prompt_tokens=481))
        for mutate in mutations:
            records, pools = source_fixture()
            mutate(records, pools)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError): contract.select_prompts(records, pools)

    def test_actual_corpus_selection_is_cpu_only_train_before_outputs(self):
        selected = contract.select_prompts(native.read_jsonl(contract.BUNDLE / "candidate_records.jsonl"),
                       {arm: native.read_jsonl(contract.BUNDLE / ("token_pool_%s.jsonl" % arm)) for arm in ("M1", "M2")})
        self.assertEqual(4, len(selected["rows"]))
        self.assertLessEqual(max(r["prompt_tokens"] for r in selected["rows"]), 480)
        self.assertTrue(all(b["split"] == "train" for b in selected["selection_bindings"]))

    def test_native7_current_helpers_and_passed_evidence_have_exact_pins(self):
        prior, inherited = contract._native7()
        self.assertEqual("PASS_SHORT_SCOPED_NATIVE_MECHANICS", inherited["status"])
        self.assertEqual(native.RESOURCE_BOUNDS, prior["resources"])
        self.assertEqual(8, len(inherited["qualified_helper_pins"]))
        fake = deepcopy(prior)
        fake["source_pins"][str((PACKAGE / "native_precision.py").resolve())] = "0" * 64
        fake["sha256"] = digest({k: v for k, v in fake.items() if k != "sha256"})
        original = native.read_json
        with patch.object(native, "read_json", side_effect=lambda path: fake if Path(path) == contract.NATIVE7_CONFIG else original(path)):
            with self.assertRaisesRegex(ValueError, "helper source drift"): contract._native7()

    def test_profile_caps_and_authority_remain_closed(self):
        self.assertEqual(4 * 65 + 2, contract.FORWARD_CAP)
        self.assertEqual(262 * 512, contract.POSITION_CAP)
        self.assertEqual("inference_native_qualification_only", contract.AUTHORIZATION["scope"])
        for key in ("production_training", "DEV_collection", "scoring", "protected_bank_access", "paid_services"):
            self.assertIs(False, contract.AUTHORIZATION[key])
        self.assertEqual(.03125, native.TOLERANCES["selected_logprob_max_error"])
        self.assertEqual(.01, native.TOLERANCES["selected_logprob_mean_error"])
        self.assertEqual((2, 1.25, 3.3125, 4, .5, 4, 600, 100, .2, .5, 2, 2),
            tuple(native.RESOURCE_BOUNDS[k] for k in ("worker_process_tree_RSS_GiB", "system_available_reserve_GiB",
            "preworker_available_GiB", "GPU_free_at_admission_GiB", "GPU_free_during_work_GiB", "disk_free_GiB",
            "whole_scope_seconds", "external_watchdog_milliseconds", "CPU_sample_max_age_seconds",
            "GPU_probe_interval_seconds", "GPU_probe_timeout_seconds", "GPU_sample_max_age_seconds")))

    def test_write_once_requires_clean_sources_and_resigned_amendment_fails_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            (package / "configs").mkdir()
            path = package / "configs" / "scope.json"
            frozen = {"id": contract.ID, "authorization": deepcopy(contract.AUTHORIZATION),
                      "resources": deepcopy(native.RESOURCE_BOUNDS), "dose": {"maximum_forward_calls": 262},
                      "source_pins": {}, "runtime_pins": {}, "model_pins": {}, "data_pins": {}, "native7": {}}
            with patch.object(contract, "PACKAGE", package), patch.object(contract, "build_scope", side_effect=lambda: deepcopy(frozen)), \
                    patch.object(contract, "_committed_checkout", return_value="a" * 40) as clean:
                body = contract.write_scope(path)
                clean.assert_called_with(True)
                self.assertEqual(contract.ID, contract.preflight(path)["id"])
                with self.assertRaises(ValueError): contract.write_scope(path)
                body["dose"]["maximum_forward_calls"] = 263
                body["sha256"] = digest({k: v for k, v in body.items() if k != "sha256"})
                path.write_text(json.dumps(body))
                with self.assertRaisesRegex(ValueError, "scope differs"): contract.preflight(path)
            path.unlink()
            with patch.object(contract, "PACKAGE", package), patch.object(contract, "_committed_checkout", side_effect=ValueError("dirty")):
                with self.assertRaisesRegex(ValueError, "dirty"): contract.write_scope(path)
            self.assertFalse(path.exists())


class Hook:
    def __init__(self, remove): self.remove = remove


class HookedModel(Model):
    def __init__(self, torch):
        super().__init__(torch, tokens=(9, 10, 151645))
        self.parameters.pop("adapter.weight")
        self.prehooks, self.observations = [], {"fake.norm": 0}
    def register_forward_pre_hook(self, hook, *, with_kwargs):
        assert with_kwargs is True
        self.prehooks.append(hook)
        return Hook(lambda: self.prehooks.remove(hook))
    def __call__(self, **kwargs):
        for hook in list(self.prehooks): hook(self, (), kwargs)
        self.observations["fake.norm"] += 1
        return super().__call__(**kwargs)


class WorkerTests(unittest.TestCase):
    def fixture(self, directory):
        torch = FakeTorch()
        torch.equal = lambda x, y: x.shape == y.shape and [(r.token, r.selected, r.baseline) for r in x.rows] == [(r.token, r.selected, r.baseline) for r in y.rows]
        model, journal = HookedModel(torch), Journal(Path(directory))
        rows = [row(i) for i in range(4)]
        config = {"rows": rows, "source_packet_sha256": digest(rows), "dose": {
                    "maximum_forward_calls": 262, "maximum_forward_input_positions": 134144}}
        report, dose, helpers = {}, worker.ForwardDose(torch), {}
        helpers["compute"] = lambda model, torch: {"scope": "CPU_FAKE_COMPONENT"}
        def adapters(model, torch):
            model.parameters["fake.lora_A"] = Parameter(requires_grad=True)
            model.parameters["fake.lora_A"].dtype = "float32"
            model.parameters["fake.lora_B"] = Parameter(requires_grad=True)
            model.parameters["fake.lora_B"].dtype = "float32"
            return {"scope": "CPU_FAKE_COMPONENT", "rank": 8}
        helpers["adapters"] = adapters
        helpers["activation_hooks"] = lambda model, torch: {"observations": model.observations, "handles": []}
        helpers["zero"] = lambda model, torch: {"all_B_zero": True, "scope": "CPU_FAKE_COMPONENT"}
        helpers["base_verify"] = lambda *args: {"verified_source_tensors": 310, "frozen_base_unchanged": True,
                                                "scope": "CPU_FAKE_COMPONENT"}
        return torch, model, journal, config, report, dose, helpers

    def execute(self, fixture):
        torch, model, journal, config, report, dose, helpers = fixture
        return worker.qualify_loaded(torch, model, config, {}, journal, lambda stage: None, report, dose,
                                     device="cpu", **helpers)

    def test_fake_worker_four_controls_zero_b_parity_no_training_and_separate_costs(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            report = self.execute(fixture)
            torch, model, journal, _, _, dose, _ = fixture
            self.assertEqual(4, len(report["numerical_controls"]))
            self.assertTrue(report["zero_adapter_parity"]["zero_adapter_output_equal"])
            self.assertEqual(310, report["base_postcheck"]["verified_source_tensors"])
            self.assertEqual(30, dose.snapshot()["attempted_calls"])
            self.assertEqual(28, report["qualification_costs"]["attempted_forwards"])
            self.assertEqual(4, sum(x["input_positions"] for name, x in dose.snapshot()["operations"].items() if "parity" in name))
            self.assertEqual([], model.prehooks)
            self.assertTrue(all(p.grad is None for p in model.parameters.values()))
            self.assertTrue(all(x["scope"] == "CPU_FAKE_COMPONENT" and not x["collection_scope_qualified"] for x in report["numerical_controls"]))
            self.assertFalse(any("backward" in name or "optimizer" in name for name in journal.snapshot()["operations"]))
            journal.stream.close()
            files = list(Path(directory).glob("stage_inference_q*_r*_t*.json"))
            self.assertEqual(24, len(files))

    def test_zero_b_fault_aborts_before_numerical_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            fixture[-1]["zero"] = lambda *args: {"all_B_zero": False}
            with self.assertRaisesRegex(ValueError, "exact zero B"): self.execute(fixture)
            self.assertEqual(1, len(fixture[1].calls))
            self.assertFalse(fixture[1].prehooks)
            fixture[2].stream.close()

    def test_base_byte_failure_prevents_mechanics_success_and_retains_all_costs(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            fixture[-1]["base_verify"] = lambda *args: {"verified_source_tensors": 309, "frozen_base_unchanged": True}
            with self.assertRaisesRegex(ValueError, "all310"): self.execute(fixture)
            self.assertEqual(30, fixture[4]["forward_dose"]["attempted_calls"])
            self.assertEqual(4, len(fixture[4]["numerical_controls"]))
            fixture[2].stream.close()

    def test_forward_failure_retains_partial_tokens_and_actual_attempts_before_abort(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self.fixture(directory)
            def fault(model, logits):
                if len(model.calls) == 5: raise RuntimeError("fake native-forward failure")
            fixture[1].fault = fault
            with self.assertRaisesRegex(RuntimeError, "native-forward failure"): self.execute(fixture)
            report = fixture[4]
            self.assertEqual(5, report["forward_dose"]["attempted_calls"])
            self.assertEqual(3, report["qualification_costs"]["attempted_forwards"])
            self.assertEqual([9, 10], [x["token"] for x in report["last_partial_receipt"]["traces"][0]])
            self.assertEqual("COMPONENT_NUMERICAL_CONTROL_FAILED", report["last_partial_receipt"]["status"])
            self.assertEqual(1, fixture[2].snapshot()["operations"]["inference_qualification_forward"]["failed"])
            self.assertFalse(fixture[1].prehooks)
            fixture[2].stream.close()

    def test_global_caps_refuse_before_next_actual_forward_and_unsafe_operations(self):
        torch, dose = FakeTorch(), worker.ForwardDose(FakeTorch())
        dose.current_operation = "inference_qualification_forward"
        ids = torch.tensor([[5] * 512], dtype=torch.long, device="cpu")
        dose.attempted_calls, dose.input_positions = 262, 134144
        with self.assertRaisesRegex(ValueError, "cap exhausted"): dose.before(None, (), {"input_ids": ids})
        self.assertEqual(262, dose.attempted_calls)
        dose.current_operation = "training_forward"
        with self.assertRaisesRegex(ValueError, "unregistered"): dose.before(None, (), {"input_ids": ids})

    def test_worker_guard_unchanged_floors_deadline_and_owned_heartbeat(self):
        owner, state = {"job_name": "fixture_owned", "supervisor_pid": 1}, {"rss": 2 * worker.GIB, "available": 1.25 * worker.GIB, "now": 10., "stamp": 10., "pid": 1}
        guard = worker.make_guard(Path("fixture"), owner, 0, lambda: None,
            owned_resources=lambda name: {"tree_rss_bytes": state["rss"]},
            available=lambda: state["available"], monotonic=lambda: state["now"], wall=lambda: state["now"],
            heartbeat_reader=lambda path: {"unix_seconds": state["stamp"], "supervisor_pid": state["pid"]})
        guard("exact_floors")
        for key, bad in (("rss", 2 * worker.GIB + 1), ("available", 1.25 * worker.GIB - 1),
                         ("now", 601), ("stamp", 7.9), ("stamp", 10.3), ("pid", 2)):
            original = state[key]
            state[key] = bad
            with self.subTest(key=key, bad=bad), self.assertRaises(ValueError): guard("failure")
            state[key] = original

    def test_cli_run_rejects_unowned_output_before_framework_import(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "owned output"): worker.run(Path("config"), Path(directory))
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("transformers", sys.modules)

    def test_missing_ownership_preserves_terminal_failure_before_any_framework_import(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            output = package / "artifacts" / "attempt"
            output.mkdir(parents=True)
            with patch.object(contract, "PACKAGE", package):
                self.assertEqual(2, worker.run(package / "config.json", output))
            receipt = native.read_json(output / "worker_result.json")
            self.assertEqual("NATIVE_INFERENCE_QUALIFICATION_FAILED", receipt["status"])
            self.assertIn("ownership receipt", receipt["error"]["message"])
            self.assertFalse(receipt["base_postcheck_completed"])
            self.assertEqual(0, receipt["optimizer_steps"])
            self.assertFalse(receipt["DEV_collection_authorized"])
            self.assertFalse(receipt["full_DEV_collection_qualified"])
            self.assertEqual({}, receipt["operation_accounting"]["operations"])
            self.assertFalse((output / "worker_GPU_final.json").exists())
        self.assertNotIn("torch", sys.modules)

    def test_injected_cuda_free_floor_without_native_cuda_calls(self):
        state = {"free": .5 * worker.GIB}
        fake = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: True, mem_get_info=lambda: (state["free"], 8 * worker.GIB)))
        guard = worker.make_guard(Path("fixture"), {"job_name": "fixture", "supervisor_pid": 1}, 0, lambda: fake,
            owned_resources=lambda name: {"tree_rss_bytes": 1}, available=lambda: 2 * worker.GIB,
            monotonic=lambda: 1, wall=lambda: 1,
            heartbeat_reader=lambda path: {"unix_seconds": 1, "supervisor_pid": 1})
        guard("exact_GPU_floor")
        state["free"] -= 1
        with self.assertRaisesRegex(ValueError, "GPU free floor"): guard("too_low")


if __name__ == "__main__": unittest.main()
