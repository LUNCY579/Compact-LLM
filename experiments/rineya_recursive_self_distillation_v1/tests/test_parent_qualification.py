"""Synthetic stdlib metadata fixtures only; never Rineya capability evidence.

The caller redirects TEMP/TMP to a BusinessProject artifact directory. Opaque
adapter bytes are deliberately not a model and no native commands are invoked.
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from experiments.rineya_recursive_self_distillation_v1.contracts import BASE_SHA256, digest, file_hash
from experiments.rineya_recursive_self_distillation_v1.parent_qualification import (
    CORE_SOURCES, INPUT_NAMES, PACKAGES, validate_parent,
)


class ParentQualificationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[4])
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.adapter = self.root / "adapter.safetensors"
        self.adapter.write_bytes(b"opaque non-model fixture bytes; metadata checks only")
        self.receipt_path = self.root / "receipt.json"
        self.manifest_path = self.root / "parent.json"
        sources = {name: "1" * 64 for name in sorted(CORE_SOURCES)}
        self.receipt = {
            "schema": 2, "experiment": "RSD-001", "mode": "parent", "arm": "sft",
            "status": "COMPLETED_SFT_CANDIDATE_REVIEW_REQUIRED",
            "exported_adapter": {"path": "adapter.safetensors", "sha256": file_hash(self.adapter)},
            "identity": {"evidence": "native_exact_titan", "base": BASE_SHA256,
                         "adapter": "ZERO_LORA_ON_EXACT_TITAN", "parent_evidence": "parent_stage",
                         "plan": "2" * 64, "catalog": "3" * 64,
                         "files": {name: "4" * 64 for name in sorted(INPUT_NAMES)},
                         "sources": sources, "source": digest(sources)},
            "source_commit": "5" * 40,
            "runtime": {"python": "3.11.15 (synthetic metadata test)", "packages": dict(PACKAGES),
                        "platform": "Windows-synthetic-metadata", "cuda": "12.6",
                        "gpu": "synthetic metadata GPU", "base_dtype": "bfloat16",
                        "adapter_dtype": "float32", "attention": "eager",
                        "sampler": "full-prefix-no-cache-temp1", "evaluation": "greedy-fixed-budget"},
            "result": {"status": "COMPLETE", "steps": 2, "last_evaluated_step": 2,
                       "costs": {"optimizer_steps": 2}, "native_efficacy_claim": False},
            "native_efficacy_claim": False,
        }
        self.manifest = {"schema": 1, "base_weight_sha256": BASE_SHA256,
                         "native_sft_completed": True, "parent_evaluation_reviewed": True,
                         "optimizer_steps": 2, "reviewed_by": "synthetic metadata reviewer",
                         "adapter": copy.deepcopy(self.receipt["exported_adapter"]),
                         "source_run_receipt": {"path": "receipt.json", "sha256": "0" * 64}}
        self.write()

    def write(self):
        self.receipt_path.write_text(json.dumps(self.receipt), encoding="utf-8")
        self.manifest["source_run_receipt"]["sha256"] = file_hash(self.receipt_path)
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def rejected(self, message=None):
        self.write()
        with self.assertRaisesRegex(ValueError, message or ".+"):
            validate_parent(self.manifest_path)

    def test_source_bound_metadata_returns_resolved_paths_without_permission(self):
        adapter, receipt, evidence = validate_parent(self.manifest_path)
        self.assertEqual(adapter, self.adapter.resolve())
        self.assertEqual(receipt, self.receipt_path.resolve())
        self.assertEqual(evidence["optimizer_steps"], 2)
        self.assertFalse(evidence["quality_certified"])
        self.assertFalse(evidence["native_execution_authorized"])
        self.assertFalse(evidence["native_efficacy_claim"])

    def test_unreviewed_candidate_only_passes_explicit_pre_review_check(self):
        self.manifest["parent_evaluation_reviewed"] = False
        self.manifest["reviewed_by"] = ""
        self.rejected("explicit parent evaluation review")
        _, _, evidence = validate_parent(self.manifest_path, require_review=False)
        self.assertFalse(evidence["review_attested"])
        self.assertFalse(evidence["native_execution_authorized"])

    def test_manifest_completion_base_steps_and_review_are_required(self):
        pristine = copy.deepcopy(self.manifest)
        for name, value in (("schema", 2), ("schema", True), ("base_weight_sha256", "0" * 64),
                            ("native_sft_completed", False), ("optimizer_steps", 0),
                            ("optimizer_steps", True), ("reviewed_by", "   "),
                            ("parent_evaluation_reviewed", "true")):
            with self.subTest(name=name, value=value):
                self.manifest = copy.deepcopy(pristine)
                self.manifest[name] = value
                self.rejected()

    def test_unsupported_external_e1_wrong_arm_and_fixture_receipts_fail_closed(self):
        pristine = copy.deepcopy(self.receipt)
        for name, value in (("schema", 1), ("experiment", "E1"), ("mode", "train"),
                            ("arm", "evolving"), ("status", "COMPLETE")):
            with self.subTest(name=name, value=value):
                self.receipt = copy.deepcopy(pristine)
                self.receipt[name] = value
                self.rejected("unsupported parent receipt")
        self.receipt = copy.deepcopy(pristine)
        self.receipt["identity"]["evidence"] = "random_CPU_fixture_NOT_native"
        self.rejected("source-bound native exact-Titan")

    def test_paused_retention_stopped_or_missing_terminal_evaluation_rejected(self):
        pristine = copy.deepcopy(self.receipt)
        for name, value in (("status", "PAUSED"), ("status", "STOPPED_RETENTION"),
                            ("last_evaluated_step", 0), ("last_evaluated_step", True)):
            with self.subTest(name=name, value=value):
                self.receipt = copy.deepcopy(pristine)
                self.receipt["result"][name] = value
                self.rejected()

    def test_steps_must_match_manifest_result_and_optimizer_ledger(self):
        self.receipt["result"]["steps"] = 3
        self.rejected("mismatched parent training result")
        self.receipt["result"]["steps"] = 2
        self.receipt["result"]["costs"]["optimizer_steps"] = 1
        self.rejected("optimizer-step evidence mismatch")
        self.receipt["result"]["costs"]["optimizer_steps"] = True
        self.rejected("optimizer-step evidence mismatch")
        self.receipt["result"]["costs"]["optimizer_steps"] = 2
        self.receipt["result"]["optimizer_steps"] = 1
        self.rejected("top-level optimizer-step mismatch")

    def test_adapter_bytes_receipt_bytes_and_export_identity_are_bound(self):
        self.adapter.write_bytes(b"changed fixture adapter")
        with self.assertRaisesRegex(ValueError, "file drift"):
            validate_parent(self.manifest_path)
        self.adapter.write_bytes(b"opaque non-model fixture bytes; metadata checks only")
        self.receipt_path.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "file drift"):
            validate_parent(self.manifest_path)
        other = self.root / "different.safetensors"
        other.write_bytes(self.adapter.read_bytes())
        self.receipt["exported_adapter"]["path"] = other.name
        self.rejected("manifest/export adapter mismatch")

    def test_missing_or_wrong_adapter_type_rejected(self):
        self.manifest["adapter"]["path"] = "missing.safetensors"
        self.rejected("missing parent evidence file")
        self.manifest["adapter"]["path"] = "adapter.bin"
        (self.root / "adapter.bin").write_bytes(self.adapter.read_bytes())
        self.rejected("must be safetensors")

    def test_source_digest_and_core_dependency_map_must_agree(self):
        self.receipt["identity"]["source"] = "f" * 64
        self.rejected("source digest/map drift")
        self.receipt["identity"]["sources"].pop(sorted(CORE_SOURCES)[0])
        self.receipt["identity"]["source"] = digest(self.receipt["identity"]["sources"])
        self.rejected("incomplete parent source")

    def test_historical_source_is_not_required_to_match_current_checkout(self):
        self.receipt["identity"]["sources"] = {name: "a" * 64 for name in CORE_SOURCES}
        self.receipt["identity"]["source"] = digest(self.receipt["identity"]["sources"])
        self.receipt["source_commit"] = "b" * 40
        self.write()
        _, _, evidence = validate_parent(self.manifest_path)
        self.assertEqual(evidence["source_commit"], "b" * 40)

    def test_invalid_source_paths_hashes_commits_and_input_bindings_rejected(self):
        pristine = copy.deepcopy(self.receipt)
        cases = [("commit", "not-a-commit"), ("input", None), ("hash", "placeholder"),
                 ("path", "../outside.py"), ("path", "C:/outside.py")]
        for kind, value in cases:
            with self.subTest(kind=kind, value=value):
                self.receipt = copy.deepcopy(pristine)
                if kind == "commit":
                    self.receipt["source_commit"] = value
                elif kind == "input":
                    self.receipt["identity"]["files"].pop("catalog_review")
                elif kind == "hash":
                    self.receipt["identity"]["files"]["catalog"] = value
                else:
                    self.receipt["identity"]["sources"][value] = "1" * 64
                    self.receipt["identity"]["source"] = digest(self.receipt["identity"]["sources"])
                self.rejected()

    def test_nonzero_warm_start_wrong_base_and_efficacy_claims_rejected(self):
        pristine = copy.deepcopy(self.receipt)
        for name, value in (("adapter", "nonzero-parent"), ("base", "0" * 64),
                            ("parent_evidence", "borrowed-permit")):
            with self.subTest(name=name):
                self.receipt = copy.deepcopy(pristine)
                self.receipt["identity"][name] = value
                self.rejected("source-bound native exact-Titan")
        self.receipt = copy.deepcopy(pristine)
        self.receipt["native_efficacy_claim"] = True
        self.rejected("may not certify efficacy")
        self.receipt = copy.deepcopy(pristine)
        self.receipt["result"]["native_efficacy_claim"] = True
        self.rejected("mismatched parent training result")

    def test_unregistered_runtime_rejected(self):
        pristine = copy.deepcopy(self.receipt)
        for name, value in (("python", "3.13.5"), ("python", "3.11.150"),
                            ("platform", "Linux"), ("cuda", None), ("base_dtype", "float32"),
                            ("sampler", "cached")):
            with self.subTest(name=name, value=value):
                self.receipt = copy.deepcopy(pristine)
                self.receipt["runtime"][name] = value
                self.rejected("unregistered parent native")
        self.receipt = copy.deepcopy(pristine)
        self.receipt["runtime"]["packages"]["torch"] = "2.10.0+cpu"
        self.rejected("unregistered parent native packages")

    def test_import_remains_stdlib_only(self):
        code = ("import sys; from experiments.rineya_recursive_self_distillation_v1 "
                "import parent_qualification; "
                "assert not {'torch', 'transformers', 'safetensors'} & set(sys.modules)")
        result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True,
                                cwd=Path(__file__).resolve().parents[3], timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
