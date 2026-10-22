"""Synthetic qualification-chain fixtures; zero native capability evidence."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.rineya_recursive_self_distillation_v1.contracts import Plan, file_hash
from experiments.rineya_recursive_self_distillation_v1.e0_qualification import PACKAGES, validate_e0_chain
from experiments.rineya_recursive_self_distillation_v1.readiness import external_output, prepare, tiny_design

ROOT = Path(__file__).resolve().parents[3]
CENTRAL = ROOT / "experiments/post_training_mimo_centralized_v1"


class E0ChainTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=ROOT.parent)
        self.addCleanup(temporary.cleanup)
        self.temp = Path(temporary.name)
        self.config = CENTRAL / "configs/E0_TITAN_BF16.json"
        self.directory = self.temp / "e0"
        self.supervision = self.temp / "e0_supervisor"
        self.directory.mkdir()
        self.supervision.mkdir()
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        self.result = {"experiment_id": "RINEYA-0P6B-E0", "status": "PASS_E0_SCOPED_NO_EFFICACY_CLAIM",
                       "config_sha256": file_hash(self.config), "code_sha": commit,
                       "dirty_at_launch": False, "pretrained_loads": 1, "optimizer_steps": 5,
                       "checks": {name: True for name in ("exact_native_load", "save_reload_resume_bitwise",
                           "frozen_base_unchanged", "finite_scoped_backward", "eos_pad_mask", "native_padding_and_causality")}}
        self.result["checks"].update(zero_adapter_parity={"prompts": 32, "max_abs": 0},
            sampled_logprobs={"max_abs": 0, "mean_abs": 0, "tokens": [10, 151645],
                            "backend": "full_prefix_no_cache_temperature1_topk0_topp1"})
        self.source = {"dirty": False, "code_sha": commit,
                       "script_sha256": file_hash(CENTRAL / "scripts/qualify_native.py")}
        self.runtime = {"python": "3.11.15 (synthetic metadata fixture)", "packages": dict(PACKAGES),
                        "cuda": "12.6", "attention": "eager", "base_dtype": "bfloat16", "adapter_dtype": "float32"}
        self.launch = {"source_sha": commit, "config_sha256": file_hash(self.config),
                       "supervisor_sha256": file_hash(CENTRAL / "scripts/supervise_native.py"),
                       "watchdog_sha256": file_hash(ROOT / "experiments/opallo_native_collector_v1/supervisor.py")}
        self.supervisor = {"status": "PASS_PROCESS_ONLY", "exit_code": 0, "reason": None,
                           "limit_seconds": 1800, "peak_observed_process_tree_rss_bytes": 1000}
        self.go = {"config_sha256": file_hash(self.config)}
        self.write()

    def write(self):
        for directory, name, value in ((self.directory, "E0_RESULT.json", self.result),
                (self.directory, "SOURCE_RECEIPT.json", self.source),
                (self.directory, "RUNTIME_RECEIPT.json", self.runtime),
                (self.supervision, "LAUNCH.json", self.launch),
                (self.supervision, "SUPERVISOR.json", self.supervisor), (self.supervision, "GO.json", self.go)):
            (directory / name).write_text(json.dumps(value), encoding="utf-8")

    def validate(self):
        return validate_e0_chain(self.config, self.directory / "E0_RESULT.json", ROOT)

    def test_synthetic_metadata_never_grants_native_permission(self):
        evidence = self.validate()
        self.assertFalse(evidence["execution_authorized"])
        self.assertFalse(evidence["capability_qualified"])
        self.assertEqual(len(evidence["companion_sha256"]), 5)

    def test_historical_actual_failed_e0_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "E0 PASS"):
            validate_e0_chain(self.config, CENTRAL / "results/e0_titan_v2/E0_RESULT.json", ROOT)

    def test_process_pass_cannot_replace_e0(self):
        self.result["status"] = "PASS_PROCESS_ONLY"
        self.write()
        with self.assertRaises(ValueError):
            self.validate()

    def test_missing_causality_or_parity_is_rejected(self):
        original = copy.deepcopy(self.result)
        for key in ("native_padding_and_causality", "zero_adapter_parity", "sampled_logprobs"):
            self.result = copy.deepcopy(original)
            del self.result["checks"][key]
            self.write()
            with self.assertRaises(ValueError):
                self.validate()

    def test_containment_and_source_drift_are_rejected(self):
        for document, key, value in ((self.source, "script_sha256", "0" * 64),
                                     (self.launch, "watchdog_sha256", "0" * 64),
                                     (self.supervisor, "status", "BLOCKED"),
                                     (self.go, "config_sha256", "0" * 64)):
            old = document[key]
            document[key] = value
            self.write()
            with self.assertRaises(ValueError):
                self.validate()
            document[key] = old

    def test_runtime_or_incomplete_native_updates_are_rejected(self):
        self.runtime["packages"]["torch"] = "2.10.0+cpu"
        self.write()
        with self.assertRaises(ValueError):
            self.validate()
        self.runtime["packages"] = dict(PACKAGES)
        self.result["optimizer_steps"] = 0
        self.write()
        with self.assertRaises(ValueError):
            self.validate()

    def test_historical_transitive_source_drift_is_rejected(self):
        with patch("experiments.rineya_recursive_self_distillation_v1.e0_qualification.subprocess.check_output",
                   return_value=b"changed dependency bytes"):
            with self.assertRaisesRegex(ValueError, "transitive producer source drift"):
                self.validate()

    def test_native_input_preflight_rejects_actual_blocked_e0_without_model_imports(self):
        opaque = self.temp / "unqualified-input.json"
        opaque.write_text("{}", encoding="utf-8")
        bound = lambda path: {"path": str(path), "sha256": file_hash(path)}
        cfg = {"schema": 2, "parent": None, "e0_config": bound(self.config),
               "e0_result": bound(CENTRAL / "results/e0_titan_v2/E0_RESULT.json"),
               "catalog": bound(opaque), "exclusions": bound(opaque), "catalog_review": bound(opaque)}
        binding = self.temp / "binding.json"
        binding.write_text(json.dumps(cfg), encoding="utf-8")
        code = ("import sys; from pathlib import Path; "
                "from experiments.rineya_recursive_self_distillation_v1.native import inputs; "
                "from experiments.rineya_recursive_self_distillation_v1.contracts import Plan;\n"
                "try:\n inputs(Path(" + repr(str(binding)) + "), Plan(), Path(" + repr(str(ROOT)) + "), parent_stage=True)\n"
                "except ValueError as exc:\n assert 'E0 PASS' in str(exc)\n"
                "else:\n raise AssertionError('failed E0 admitted')\n"
                "assert not {'torch','transformers','safetensors'} & sys.modules.keys()")
        subprocess.run([sys.executable, "-B", "-c", code], check=True)

    def test_legacy_binding_cannot_skip_provenance_contract(self):
        from experiments.rineya_recursive_self_distillation_v1.native import inputs
        path = self.temp / "legacy.json"
        path.write_text(json.dumps({"schema": 1, "parent": None, "e0_config": None,
                                   "e0_result": None, "catalog": None, "exclusions": None}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "schema2"):
            inputs(path, Plan(), ROOT, parent_stage=True)


class ReadinessTests(unittest.TestCase):
    def test_tiny_design_prespecifies_refresh_contrast_and_control(self):
        plan = Plan.load(ROOT / "experiments/rineya_recursive_self_distillation_v1/configs/tiny_guidance_v1.json")
        design = tiny_design(plan)
        self.assertEqual(design["selected_arms"], ["sft", "frozen", "evolving"])
        self.assertEqual(design["primary_contrast"]["evaluated_step"], 2)
        self.assertEqual(design["treatment_optimizer_ceiling"], 6)
        self.assertFalse(design["srcl_selected"])
        self.assertFalse(design["native_execution_authorized"])
        with self.assertRaises(ValueError):
            tiny_design(Plan(rounds=1))

    def test_output_must_be_external_businessproject_and_new(self):
        with self.assertRaises(ValueError):
            external_output(ROOT / "artifacts/new-readiness", ROOT)
        with self.assertRaises(ValueError):
            external_output(ROOT.parent / ".codex/new-readiness", ROOT)
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as directory:
            path = Path(directory)
            self.assertEqual(external_output(path / "new", ROOT), path / "new")
            with self.assertRaises(ValueError):
                external_output(path, ROOT)

    def test_readiness_import_is_model_free(self):
        subprocess.run([sys.executable, "-B", "-c",
            "import sys; import experiments.rineya_recursive_self_distillation_v1.readiness; "
            "assert not {'torch','transformers','safetensors'} & sys.modules.keys()"], check=True)
