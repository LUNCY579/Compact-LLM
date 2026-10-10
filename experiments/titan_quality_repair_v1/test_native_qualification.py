"""Injected protocol/counter fixtures only: no Torch, model or genuine permit."""
from __future__ import annotations

import copy
import hashlib
import os
import random
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import custody, generator, native_qualification as n, registry, schema


def fixture_rows():
    return [generator._row(family, generator._payload(family, index, 64, random.Random(100 + index)),
                           "dev", "canonical")
            for family in schema.REPAIR_FAMILIES for index in range(2)]


def fixture_proposal(rows):
    ids = [r["row_id"] for r in rows]
    p = {"experiment": schema.EXPERIMENT, "proposal_version": 2, "stage": "native_qualification",
         "native_worker_implementation_in_this_milestone": True,
         "source_checkpoint_sha256": registry.SOURCE_WEIGHTS_SHA256,
         "registry_sha256": registry.build_registry()["registry_sha256"],
         "source_manifest_sha256": "a" * 64, "architecture": "Qwen3ForCausalLM", "dtype": "float32",
         "source_sidecar_sha256": dict.fromkeys(custody.SIDECARS, "a" * 64),
         "source_identity_descriptor": {"path": "fixture-only://source-identity", "sha256": "b" * 64},
         "custom_abstention": False, "attempt": {
             "generation_row_ids_in_order": ids, "optimizer_row_ids_in_order": ids + ids,
             "padding_check_row_ids_in_order": ids[:2], "reload_readout_row_id": ids[0],
             "generation_rows": 8, "max_output_tokens": 64, "max_input_tokens": 1984,
             "generation_forward_requests_max": 512, "padding_check_rows": 2, "padding_check_steps": 8,
             "padding_check_forwards": 32, "padding_layout": "left_singleton_vs_unpadded",
             "padding_input_slots": 1984,
             "disposable_optimizer_updates": 2, "optimizer_microbatches": 16,
             "teacher_forced_forward_requests": 16, "backward_requests": 16,
             "reload_readout_rows": 1, "reload_teacher_forced_forward_requests": 2,
             "registered_forward_requests_without_checkpoint_recompute": 562,
             "eos_ids": [151645, 151643], "training_defaults": copy.deepcopy(registry.TRAINING),
             "greedy": True, "thinking": False, "tf32": False, "attention": "eager",
             "development_path": "fixture-only://no-development-bank-read", "development_sha256": "b" * 64}}
    p["proposal_payload_sha256"] = n.proposal_payload_digest(p)
    return p


def fixture_admission(proposal):
    # Structurally shaped claims deliberately have no actual grant, lease or Job.
    # They are used only by pure validation and never by production qualify.
    root = custody.ARTIFACT_ROOT / "native-qualification"
    return {"experiment": schema.EXPERIMENT, "stage": "native_qualification",
            "source_checkpoint_sha256": registry.SOURCE_WEIGHTS_SHA256,
            "registry_sha256": registry.build_registry()["registry_sha256"],
            "source_manifest_sha256": "a" * 64, "proposal_payload_sha256": proposal["proposal_payload_sha256"],
            "proposal_file_sha256": "b" * 64, "permit_file_sha256": "c" * 64,
            "attempt_id": "fixture-claims-only", "hardware_slot": "fixture-no-hardware",
            "runtime_root": str(root), "attempt_dir": str(root / "attempts" / "fixture-claims-only"),
            "proposal_path": "fixture-only://unissued", "permit_path": "fixture-only://unissued",
            "job_assigned_before_payload": True, "guardian_admission_passed": True,
            "worker_pid": 123, "supervisor_pid": 122, "worker_create_time": 100.0, "supervisor_create_time": 99.0,
            "supervisor_hard_wall_seconds": 600, "worker_wall_seconds": 540,
            "cleanup_wall_seconds": 120, "declared_attempt_cost_seconds": 720,
            "limits": {"host_admission_gib": 12, "gpu_admission_gib": 6, "disk_admission_gib": 20,
                       "host_reserve_gib": 1.5, "gpu_reserve_gib": .5, "tree_rss_max_gib": 10, "output_bytes_max": 6 * 2 ** 30}}


class FakeFramework:
    def __init__(self):
        self.events = []
        self.bad = None
        self.synchronizations = 0
    def synchronize(self):
        self.synchronizations += 1
    def verify_source_assets(self):
        return {"fixture_only": True}
    def load_source(self):
        self.events.append("load")
    def serialize(self, rows):
        self.events.append("serialize")
        return [{"row_id": r["row_id"], "full_hf_equivalence": self.bad != "template",
                 "loss_tokens": 0 if self.bad == "mask" else 1, "final_eos_label_positive": True} for r in rows]
    def generate(self, item):
        self.events.append("generate:" + item["row_id"])
        return {"forward_requests": 65 if self.bad == "generation" else 64, "greedy": True, "eos_ids": list(n.EOS_IDS)}
    def padding_probe(self, item, steps):
        self.events.append("padding:" + item["row_id"])
        return {"forward_requests": 16, "layout": "left_singleton_vs_unpadded", "parity": self.bad != "padding"}
    def install_lora(self):
        self.events.append("lora")
    def microbatch(self, item, accumulation):
        self.events.append("microbatch:" + item["row_id"])
        if accumulation != 8:
            raise AssertionError("fixture accumulation drift")
        return {"forward_requests": 1, "backward_requests": 1, "finite_loss": self.bad != "loss", "positive_labels": True}
    def optimizer_step(self):
        self.events.append("update")
        return {"finite_gradients": True, "gradient_norm": 0 if self.bad == "gradients" else 1.0}
    def merge_dense(self):
        self.events.append("merge")
    def readout(self, item):
        self.events.append("readout:" + item["row_id"])
        return (1, 2, 3)
    def export_dense(self):
        self.events.append("export")
    def release_dense_before_reload(self):
        self.events.append("release")
    def reload_dense(self):
        self.events.append("reload")
    def readout_equal(self, before, after):
        return before == after and self.bad != "reload"
    def recompute_accounting(self):
        return {"model_forward_requests_during_backward": 0,
                "decoder_layer_forward_requests_during_backward": -1 if self.bad == "recompute" else 448}


class QualificationTests(unittest.TestCase):
    def setUp(self):
        self.rows = fixture_rows()
        self.proposal = fixture_proposal(self.rows)

    def _run(self, framework):
        return n._run_protocol(framework, self.rows, self.proposal, check=lambda phase: None,
                               mark=lambda event: framework.events.append(event), fixture_only=True)

    def test_exact_protocol_and_checkpoint_recomputation_are_accounted(self):
        framework = FakeFramework()
        result = self._run(framework)
        self.assertEqual(result["forward_requests_without_checkpoint_recompute"], 562)
        self.assertEqual(result["phase_forward_requests"], {"generation": 512, "padding": 32,
            "teacher_forced": 16, "reload_readout": 2, "backward_requests": 16, "optimizer_updates": 2})
        self.assertEqual(result["checkpoint_recompute"]["decoder_layer_forward_requests_during_backward"], 448)
        self.assertEqual(framework.events[0:3], ["LOAD_BEGUN", "load", "serialize"])
        self.assertLess(framework.events.index("DENSE_RELOAD_BEGUN"), framework.events.index("reload"))
        self.assertTrue(result["fixture_only"])
        self.assertFalse(result["native_qualification_passed"])
        self.assertFalse(result["native_mechanics_passed"])
        self.assertFalse(result["capability_evidence"])
        self.assertGreater(framework.synchronizations, 0)
        self.assertEqual(result["stage_timing"], "FIXTURE_STUB_NOT_DEVICE_TIMING")

    def test_optimizer_order_and_disposable_export_readout(self):
        framework = FakeFramework()
        result = self._run(framework)
        self.assertEqual(framework.synchronizations, 2 * len(result["stage_wall_seconds"]))
        micro = [e.split(":", 1)[1] for e in framework.events if e.startswith("microbatch:")]
        ids = self.proposal["attempt"]["generation_row_ids_in_order"]
        self.assertEqual(micro, ids + ids)
        self.assertEqual(framework.events.count("update"), 2)
        before = framework.events.index("merge")
        self.assertEqual(framework.events[before:], ["merge", "readout:" + ids[0], "export", "release", "DENSE_RELOAD_BEGUN", "reload", "readout:" + ids[0]])

    def test_hf_equivalence_and_positive_labels_stop_before_any_training(self):
        for bad in ("template", "mask"):
            with self.subTest(bad=bad):
                framework = FakeFramework()
                framework.bad = bad
                with self.assertRaisesRegex(ValueError, "STOP_QUALIFICATION_FULL_HF_LABEL_MASK"):
                    self._run(framework)
                self.assertNotIn("lora", framework.events)

    def test_nonfinite_padding_counter_and_reload_fail_closed(self):
        cases = {"generation": "GENERATION_ACCOUNTING", "padding": "PADDING_PARITY",
                 "loss": "NONFINITE_LOSS", "gradients": "NONFINITE_OR_ZERO_GRADIENTS",
                 "reload": "RELOADED_FP32_READOUT_PARITY", "recompute": "UNMEASURED_RECOMPUTATION"}
        for bad, error in cases.items():
            with self.subTest(bad=bad):
                framework = FakeFramework()
                framework.bad = bad
                with self.assertRaisesRegex(ValueError, error):
                    self._run(framework)

    def test_ownership_check_interrupts_before_loading(self):
        framework = FakeFramework()
        def check(phase):
            raise RuntimeError("fixture ownership lost")
        with self.assertRaisesRegex(RuntimeError, "ownership lost"):
            n._run_protocol(framework, self.rows, self.proposal, check=check, mark=lambda event: None)
        self.assertEqual(framework.events, [])

    def test_proposal_rejects_dtype_count_contract_and_training_drift(self):
        cases = [("dtype", "float16"), ("architecture", "CustomRouter"), ("custom_abstention", True),
                 ("proposal_version", 1), ("stage", "parentq"), ("native_worker_implementation_in_this_milestone", False)]
        for key, value in cases:
            proposal = copy.deepcopy(self.proposal)
            proposal[key] = value
            with self.assertRaises(ValueError):
                n.validate_proposal(proposal)
        for mutate in (lambda a: a.update(max_output_tokens=63), lambda a: a.update(backward_requests=True),
                       lambda a: a.update(padding_layout="right"), lambda a: a["training_defaults"].update(microbatch=True)):
            proposal = copy.deepcopy(self.proposal)
            mutate(proposal["attempt"])
            with self.assertRaises(ValueError):
                n.validate_proposal(proposal)

    def test_final_data_and_unregistered_membership_are_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["split"] = "final"
        with self.assertRaisesRegex(ValueError, "DEVELOPMENT_ONLY"):
            n._checked_rows(rows, self.proposal)
        rows = copy.deepcopy(self.rows)
        rows[0]["row_id"] += "-different"
        with self.assertRaisesRegex(ValueError, "ROW_MISSING"):
            n._checked_rows(rows, self.proposal)

    def test_pure_admission_is_exactly_source_and_experiment_bound(self):
        admission = fixture_admission(self.proposal)
        n.validate_admission(admission, self.proposal)
        changes = [("experiment", "PARENTQ"), ("source_checkpoint_sha256", "d" * 64),
                   ("worker_pid", True), ("worker_wall_seconds", 600), ("fixture_only", True),
                   ("guardian_admission_passed", False), ("runtime_root", "C:/Users/clclc/.codex/native")]
        for key, value in changes:
            with self.subTest(key=key):
                bad = copy.deepcopy(admission)
                bad[key] = value
                with self.assertRaises(ValueError):
                    n.validate_admission(bad, self.proposal)

    def test_proposal_review_digest_cycle_exclusion_does_not_drop_workload_binding(self):
        changed = copy.deepcopy(self.proposal)
        changed["review_descriptor"] = {"fixture": "not-an-attestation"}
        self.assertEqual(n.proposal_payload_digest(changed), n.proposal_payload_digest(self.proposal))
        changed["attempt"]["max_output_tokens"] = 63
        self.assertNotEqual(n.proposal_payload_digest(changed), n.proposal_payload_digest(self.proposal))

    def test_public_factory_or_fake_guardian_never_admits_fixture(self):
        with self.assertRaisesRegex(ValueError, "INJECTED_FRAMEWORK"):
            n.qualify({}, guardian_check=lambda **kwargs: None, framework_factory=FakeFramework)
        with self.assertRaisesRegex(ValueError, "LIVE_BOUND_GUARDIAN"):
            n.qualify({}, guardian_check=lambda **kwargs: {"passed": True})


class FakeCore:
    """Simulated denied writers; never opens a real core or native file handle."""
    def __init__(self):
        self.assets = {name: ("fixture-" + name).encode() for name in custody.SIDECARS}
        self.assets["model.safetensors"] = b"tiny fixture checkpoint" * 10
        self.readers = []
        self.fail_name = None
    def opener(self, path, directory=False):
        name = Path(path).name
        if name == self.fail_name:
            raise PermissionError("fixture writer already holds file")
        core = self
        class Reader:
            def __init__(self):
                self.closed = False
                self.initial = None if directory else core.assets[name]
            def verify(self):
                if self.closed or (not directory and core.assets[name] != self.initial):
                    raise RuntimeError("STOP_HELD_SOURCE_IDENTITY_CHANGED")
            def read_chunks(self):
                self.verify()
                data = core.assets[name]
                for start in range(0, len(data), 4):
                    yield data[start:start + 4]
            def close(self):
                self.closed = True
        reader = Reader()
        self.readers.append(reader)
        return reader
    def pins(self):
        return {name: hashlib.sha256(self.assets[name]).hexdigest() for name in custody.SIDECARS}
    def custody(self, pins=None):
        return n._CoreSourceCustody(Path("fixture-core-no-real-io"), self.pins() if pins is None else pins,
            opener=self.opener, inventory=list(self.assets), fixture_only=True)


class SourceCustodyTests(unittest.TestCase):
    def test_transferred_crt_descriptor_is_closed_if_stream_wrapper_was_not_created(self):
        reader = object.__new__(n._WindowsReadOnlyFile)
        reader.stream, reader.descriptor, reader.handle = None, 73, 999
        class Kernel:
            def CloseHandle(self, handle):
                raise AssertionError("raw handle would leave a stale CRT descriptor")
        reader.kernel = Kernel()
        with patch.object(n.os, "close") as close:
            reader.close()
        close.assert_called_once_with(73)
        self.assertIsNone(reader.handle)

    @unittest.skipUnless(os.name == "nt", "Win32 error accessor")
    def test_raw_handle_close_failure_is_reported_and_retains_handle_identity(self):
        reader = object.__new__(n._WindowsReadOnlyFile)
        reader.stream, reader.descriptor, reader.handle = None, None, 999
        class Kernel:
            def CloseHandle(self, handle):
                return 0
        reader.kernel = Kernel()
        with self.assertRaisesRegex(RuntimeError, "SOURCE_HANDLE_CLOSE_FAILED"):
            reader.close()
        self.assertEqual(reader.handle, 999)

    def test_same_held_reader_hash_and_source_handles_retained_until_close(self):
        core = FakeCore()
        protected = core.custody()
        events = []
        data = core.assets["model.safetensors"]
        value = protected.weight_digest(events.append, expected_bytes=len(data), expected_sha256=hashlib.sha256(data).hexdigest())
        self.assertEqual(value, hashlib.sha256(data).hexdigest())
        self.assertEqual(protected.weight_bytes_hashed, len(data))
        self.assertIn("source_hash_progress", events)
        self.assertTrue(all(not reader.closed for reader in core.readers))
        self.assertTrue(protected.verify()["fixture_only"])
        protected.close()
        self.assertTrue(all(reader.closed for reader in core.readers))

    def test_original_sidecar_pin_tamper_blocks_and_closes_partial_readers(self):
        core = FakeCore()
        original = core.pins()
        core.assets["tokenizer_config.json"] = b"changed fixture template"
        with self.assertRaisesRegex(ValueError, "SOURCE_SIDECAR_DIGEST"):
            core.custody(original)
        self.assertTrue(all(reader.closed for reader in core.readers))

    def test_read_lock_acquisition_failure_closes_already_owned_handles(self):
        core = FakeCore()
        core.fail_name = "tokenizer.json"
        with self.assertRaisesRegex(PermissionError, "writer already holds"):
            core.custody()
        self.assertTrue(all(reader.closed for reader in core.readers))

    def test_source_drift_is_rejected_before_framework_loader(self):
        core = FakeCore()
        protected = core.custody()
        framework = FakeFramework()
        framework.verify_source_assets = protected.verify
        core.assets["config.json"] = b"changed configuration"
        rows = fixture_rows()
        with self.assertRaisesRegex(RuntimeError, "SOURCE_IDENTITY_CHANGED"):
            n._run_protocol(framework, rows, fixture_proposal(rows), check=lambda phase: None, mark=lambda event: None)
        self.assertEqual(framework.events, [])
        protected.close()

    def test_production_source_factory_cannot_use_fixture_reader_without_guardian(self):
        core = FakeCore()
        with self.assertRaisesRegex(ValueError, "ACTUAL_GUARDIAN"):
            n._CoreSourceCustody(Path("fixture-core-no-real-io"), core.pins(), opener=core.opener,
                inventory=list(core.assets), fixture_only=False)
        self.assertEqual(core.readers, [])

    @unittest.skipUnless(os.name == "nt", "Win32 sharing fixture")
    def test_actual_windows_toy_read_share_denies_write_delete_then_releases(self):
        # Only a toy payload under the supervisor's lane-contained TEMP; never
        # a model file. The resolved deletion target stays inside this temp root.
        temp_root = Path(tempfile.gettempdir()).resolve()
        self.assertTrue(temp_root.is_relative_to(custody.ARTIFACT_ROOT.resolve()))
        with tempfile.TemporaryDirectory(prefix="core-read-share-", dir=temp_root) as directory:
            resolved = Path(directory).resolve()
            self.assertTrue(resolved.is_relative_to(temp_root))
            path = resolved / "toy.txt"
            path.write_bytes(b"original tiny fixture")
            reader = n._WindowsReadOnlyFile(path)
            try:
                self.assertEqual(path.read_bytes(), b"original tiny fixture")
                self.assertEqual(b"".join(reader.read_chunks()), b"original tiny fixture")
                with self.assertRaises(PermissionError):
                    path.write_bytes(b"cannot overwrite")
                with self.assertRaises(PermissionError):
                    path.unlink()
                reader.verify()
                self.assertEqual(path.read_bytes(), b"original tiny fixture")
            finally:
                reader.close()
            path.write_bytes(b"write allowed after close")
            self.assertEqual(path.read_bytes(), b"write allowed after close")


if __name__ == "__main__":
    unittest.main()
