"""Tiny CPU tensors exercise merge geometry, actual delta norms and custody gates."""
from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

if __package__:
    from . import materialize as m
else:
    import materialize as m

import torch
from safetensors.torch import load_file, save_file
from experiments.model_quality_gpu_v1.integrity import digest, file_hash, seal, write_new


class MaterializerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = self.root / "journal/runs/train-001"
        self.run.mkdir(parents=True)
        self.source, self.lora, self.native = self.root / "source", self.run / "checkpoint-00193", self.root / "native"
        for p in (self.source, self.lora, self.native):
            p.mkdir()
        tensors = {}
        for i in range(28):
            for suffix in m.PROJECTIONS:
                tensors[f"model.layers.{i}.{suffix}.weight"] = torch.full((2, 2), 2.0)
            for suffix in m.NORMS:
                tensors[f"model.layers.{i}.{suffix}"] = torch.full((2,), 2.0)
        tensors["model.embed_tokens.weight"] = torch.full((2, 2), 2.0)
        tensors["model.norm.weight"] = torch.full((2,), 2.0)
        save_file(tensors, str(self.source / "model.safetensors"))
        for name in m.SIDECARS:
            (self.source / name).write_bytes(b"{}\n")
        (self.source / "config.json").write_text(json.dumps({"num_hidden_layers": 28, "tie_word_embeddings": True}))
        patches = [patch.object(m, "PRODUCTION_SHAPES", {s: (2, 2) for s in m.PROJECTIONS}),
                   patch.object(m, "NORM_WIDTHS", {s: 2 for s in m.NORMS}),
                   patch.object(m, "BASE_PARAMETERS", sum(t.numel() for t in tensors.values())),
                   patch.object(m, "ADAPTER_PARAMETERS", 8960), patch.object(m, "NORM_PARAMETERS", 160),
                   patch.object(m, "require_ram", return_value=8 * m.GIB),
                   patch.object(m, "resource_lock_path", return_value=self.root / "fixture.lock")]
        # disk_usage has a namedtuple result in production.
        from collections import namedtuple
        patches.append(patch.object(m.shutil, "disk_usage",
                                    return_value=namedtuple("Disk", "total used free")(100*m.GIB, 10*m.GIB, 90*m.GIB)))
        for context in patches:
            context.start(); self.addCleanup(context.stop)
        assets = {name: file_hash(self.source / name) for name in ("model.safetensors", *m.SIDECARS)}
        prepared = self.root / "prepared"; prepared.mkdir()
        (prepared / "microfit-train.jsonl").write_text(json.dumps({"ids": [1, 2, 3, 4]}) + "\n")
        (prepared / "ARM_BUILD.json").write_bytes(b"{}\n")
        files = {name: file_hash(prepared / name) for name in ("microfit-train.jsonl", "ARM_BUILD.json")}
        parent_contract = self.root / "parent_contract.json"
        parent_contract.write_text(json.dumps({"experiment": "RRP-008", "files": files}))
        (prepared / "DATA_CONTRACT.json").write_text(json.dumps({"experiment": "DOSELORA-001", "files": files,
                                                                "parent": file_hash(parent_contract)}))
        plan_path = self.root / "plan.json"
        plan = {"experiment": "DOSELORA-001", "fixture": True}
        plan_path.write_text(json.dumps(plan))
        self.grant = {"experiment": "DOSELORA-001", "limits": {"total_owned_work_wall_seconds": 100,
                      "model_worker_wall_seconds_and_conservative_gpu_seconds": 100, "model_worker_launches": 1}}
        self.binding = {"experiment": "MQ-001", "plan_sha256": "1" * 64,
                        "data_contract_sha256": file_hash(parent_contract), "grant_sha256": "3" * 64,
                        "runtime": {"fixture": True}, "source": {"assets": assets, "sha256": digest(assets)}}
        lora_binding = {**self.binding, "experiment": "DOSELORA-001", "plan_path": str(plan_path),
                        "grant_sha256": digest(self.grant),
                        "plan_sha256": digest(plan), "prepared": str(prepared),
                        "data_contract_sha256": file_hash(prepared / "DATA_CONTRACT.json"),
                        "parent_data_contract_path": str(parent_contract),
                        "parent_data_contract_sha256": file_hash(parent_contract), "training_files_sha256": files,
                        "parameterization": {"rank": 16, "alpha": 32.0, "dropout": 0, "bias": "none",
                                             "layers": list(range(8, 28)), "train_norms": True,
                                             "target_modules": [p.rsplit(".", 1)[1] for p in m.PROJECTIONS]}}
        header = m.tensor_header(self.source / "model.safetensors")
        spec = m.expected_lora_spec(header)
        a = torch.zeros((16, 2)); a[0] = torch.tensor([1.0, 2.0])
        b = torch.zeros((2, 16)); b[:, 0] = torch.tensor([1/16, 1/8])
        parameters = {name: (a.clone() if name.endswith(".lora_A") else
                             b.clone() if name.endswith(".lora_B") else torch.full((2,), 2.5))
                      for name in spec["trainable_names"]}
        self.lora_manifest = self.checkpoint(self.lora, parameters,
                                            {"record": "DOSELORA_TRAINING_CHECKPOINT", "adapter_spec": spec},
                                            lora_binding, "DOSELORA-001")
        self.native_manifest = self.checkpoint(self.native, {n: tensors[n] + 1 for n in m.DENSE_NAMES},
                                              {}, self.binding, "MQ-001")
        pin = patch.object(m, "NATIVE_MANIFEST_SHA256", file_hash(self.native / "manifest.json"))
        pin.start(); self.addCleanup(pin.stop)
        self.out, self.norm_path = self.root / "merged", self.root / "NORM_MATCH.json"
        self.qualify_fixture()

    @staticmethod
    def checkpoint(path, parameters, extra, binding, experiment):
        entries = []
        for i, (name, value) in enumerate(sorted(parameters.items())):
            filename = f"tensor-{i:04d}.safetensors"
            save_file({"parameter": value, "exp_avg": torch.zeros_like(value),
                       "exp_avg_sq": torch.zeros_like(value)}, str(path / filename))
            entries.append({"name": name, "file": filename, "shape": list(value.shape),
                            "sha256": file_hash(path / filename)})
        manifest = {"schema": 1, "experiment": experiment, "status": "COMPLETE", "binding": binding,
                    "progress": dict(m.FINAL_PROGRESS), "rng": {"fixture": True}, "entries": entries, **extra}
        write_new(path / "manifest.json", seal(manifest))
        return manifest

    def reseal_lora(self):
        (self.lora / "manifest.json").write_text(json.dumps(seal(self.lora_manifest)))

    def qualify_fixture(self):
        """Mechanical stand-in for an independently completed trainer qualification."""
        source = load_file(str(self.source / "model.safetensors"))
        params = {e["name"]: load_file(str(self.lora / e["file"]))["parameter"]
                  for e in self.lora_manifest["entries"]}
        hashes = {}
        for name in m.DENSE_NAMES:
            if name in m.NORM_NAMES:
                value = params[name]
            else:
                prefix = name.removesuffix(".weight")
                value = source[name] + (params[prefix + ".lora_B"] @ params[prefix + ".lora_A"]) * 2
            metadata = json.dumps([name, list(value.shape), "torch.float32"],
                                  ensure_ascii=False, separators=(",", ":")).encode()
            hashes[name] = hashlib.sha256(metadata + value.contiguous().numpy().tobytes()).hexdigest()
        qualification_path = self.run / "merge_qualification.json"
        inputs = []
        for label, token_ids in (("reply_7", list(range(31))), ("first_training_row", [1, 2, 3, 4]),
                                 ("synthetic_2048_arange", list(range(2048)))):
            inputs.append({"input": label, "input_token_count": len(token_ids), "input_ids_sha256": digest(token_ids),
                           "checked_last_positions": min(16, len(token_ids)),
                           "checked_logits_shape": [1, min(16, len(token_ids)), 151936],
                           "max_abs_difference": 0.0, "tolerance_violations": 0, "argmax_mismatches": 0, "passed": True})
        qualification_path.write_text(json.dumps({"schema": 1, "record": "DOSELORA_FINAL_MERGE_QUALIFICATION",
                         "experiment": "DOSELORA-001", "status": "PASS", "complete": True,
                         "checkpoint": str(self.lora), "binding": self.lora_manifest["binding"],
                         "dense_tensor_sha256": hashes, "atol": 2e-4, "rtol": 2e-4,
                         "argmax_equality_required": True, "original_modules_restored": True,
                         "frozen_base_sha256_before_after": "f" * 64, "inputs": inputs,
                         "budget_charges": {"processed_training_tokens_all_attempts": 2 * (31 + 4 + 2048),
                                            "evaluation_and_qualification_model_calls": 6},
                         "checkpoint_manifest_sha256": file_hash(self.lora / "manifest.json")}))
        (self.run / "result.json").write_text(json.dumps({"record": "DOSELORA_TRAINING_RESULT",
                    "experiment": "DOSELORA-001", "status": "COMPLETE", "complete": True,
                    "binding": self.lora_manifest["binding"], "adapter_spec": self.lora_manifest["adapter_spec"],
                    "frozen_base_sha256_before_after": "f" * 64,
                    "progress": dict(m.FINAL_PROGRESS), "checkpoint": str(self.lora),
                    "checkpoint_manifest_sha256": file_hash(self.lora / "manifest.json"),
                    "merge_qualification_path": str(qualification_path),
                    "merge_qualification_sha256": file_hash(qualification_path)}))
        self.supervise_fixture()

    def supervise_fixture(self):
        journal = self.root / "journal"
        events = journal / "events"; events.mkdir(exist_ok=True)
        (journal / "grant.json").write_text(json.dumps(self.grant))
        launch = {"schema": 1, "out": str(self.run), "supervisor_pid": 7,
                  "argv": ["python", "train.py", "--journal", str(journal), "--out", str(self.run),
                           "--run-id", self.run.name, "--plan", str(self.root / "plan.json"),
                           "--prepared", str(self.root / "prepared"), "--grant", str(journal / "grant.json"), "--_worker"]}
        supervisor = {"status": "EXITED", "returncode": 0, "child_pid": 11, "cleanup": None, "wall_seconds": 1.0}
        records = {"launch.json": launch, "supervisor.json": supervisor,
                   "spawn.json": {"child_pid": 11, "supervisor_pid": 7, "launch_sha256": digest(launch)},
                   "worker_claim.json": {"pid": 11, "parent_pid": 7, "supervisor_pid": 7,
                                         "launcher_redirected": False, "launch_sha256": digest(launch)}}
        for filename, value in records.items():
            (self.run / filename).write_text(json.dumps(value))
        walls = ("total_owned_work_wall_seconds", "model_worker_wall_seconds_and_conservative_gpu_seconds")
        start = seal({"index": 0, "previous": None, "kind": "start", "run_id": self.run.name, "phase": "train",
                      "charges": {walls[0]: 10, walls[1]: 10, "model_worker_launches": 1}})
        settle = seal({"index": 1, "previous": start["sha256"], "kind": "settle", "run_id": self.run.name,
                       "receipt_sha256": file_hash(self.run / "supervisor.json"), "charges": {},
                       "actual_wall": {walls[0]: 1, walls[1]: 1}})
        for index, value in enumerate((start, settle)):
            (events / f"{index:06d}.json").write_text(json.dumps(value))
        used = {walls[0]: 1, walls[1]: 1, "model_worker_launches": 1}
        budget = {"used_or_reserved": used, "active_reservations": [], "budget_breached": False,
                  "remaining": {k: self.grant["limits"][k] - v for k, v in used.items()}}
        (self.run / "run_summary.json").write_text(json.dumps({"experiment": "DOSELORA-001", "run_id": self.run.name,
                            "returncode": 0, "worker_failed": False, "supervisor": supervisor, "budget_after": budget}))

    def build_lora(self):
        return m.materialize_lora(self.source, self.lora, self.native, self.out, self.norm_path)

    def test_exact_merge_actual_dense_norm_and_scalar_match(self):
        norm = self.build_lora()
        self.assertEqual(norm["status"], "PASS")
        self.assertEqual(norm["dense_updated_tensor_count"], 220)
        self.assertEqual(norm["projection_count"], 140)
        self.assertEqual(norm["trained_norm_count"], 80)
        # Each 2BA matrix has squares .125²+.25²+.25²+.5²; each trained norm contributes 2*.5².
        self.assertAlmostEqual(norm["lora_dense_delta_norm"], math.sqrt(140 * .390625 + 80 * .5), places=12)
        self.assertAlmostEqual(norm["native_dense_delta_norm"], math.sqrt(140 * 4 + 80 * 2), places=12)
        self.assertAlmostEqual(norm["beta"], math.sqrt(94.6875 / 720), places=12)
        self.assertEqual(norm["training_run_summary_sha256"], file_hash(self.run / "run_summary.json"))
        self.assertEqual(norm["training_supervisor_sha256"], file_hash(self.run / "supervisor.json"))
        self.assertEqual(len(norm["training_journal_event_sha256"]), 2)
        merged = load_file(str(self.out / "model.safetensors"))
        self.assertEqual(len(merged), 310)
        self.assertTrue(torch.equal(merged["model.layers.8.self_attn.q_proj.weight"],
                                    torch.tensor([[2.125, 2.25], [2.25, 2.5]])))
        self.assertTrue(torch.equal(merged["model.layers.8.self_attn.q_norm.weight"], torch.full((2,), 2.5)))
        self.assertTrue(torch.equal(merged["model.layers.0.self_attn.q_proj.weight"], torch.full((2, 2), 2.0)))
        del merged
        scalar = m.materialize_scalar(self.source, self.native, self.root / "scalar", self.norm_path)
        self.assertEqual(scalar["alpha"], norm["beta"])
        self.assertLessEqual(scalar["relative_norm_discrepancy"], 1e-5)
        self.assertEqual(scalar["checkpoint_manifest_sha256"], file_hash(self.native / "manifest.json"))
        self.assertTrue((self.root / "scalar/NORM_MATCH_VERIFICATION.json").is_file())

    def test_adapter_factor_rescaling_preserves_dense_norm(self):
        src = torch.zeros((2, 2)); a = torch.arange(32, dtype=torch.float32).reshape(16, 2) / 16
        b = torch.arange(32, dtype=torch.float32).reshape(2, 16) / 16
        first = m.merge_projection(src, a, b)
        second = m.merge_projection(src, a * 8, b / 8)
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(m.delta_square_sum(first, src), m.delta_square_sum(second, src))

    def test_scalar_endpoints_copy_exactly_and_do_not_alias(self):
        src, trained = torch.tensor([2.0, -0.0]), torch.tensor([4.0, 7.0])
        for beta, expected in ((0.0, src), (1.0, trained)):
            value = m.scalar_value(src, trained, beta)
            self.assertTrue(torch.equal(value.view(torch.uint8), expected.view(torch.uint8)))
            self.assertNotEqual(value.data_ptr(), expected.data_ptr())
        with self.assertRaisesRegex(ValueError, "outside"):
            m.scalar_value(src, trained, 2.01)

    def test_wrong_hash_refused_before_output(self):
        path = self.lora / self.lora_manifest["entries"][0]["file"]
        path.write_bytes(path.read_bytes() + b"corrupt")
        with self.assertRaisesRegex(ValueError, "tensor hash mismatch"):
            self.build_lora()
        self.assertFalse(self.out.exists()); self.assertFalse(self.norm_path.exists())

    def test_no_overwrite_preserves_bytes(self):
        self.out.mkdir(); sentinel = self.out / "precious"; sentinel.write_bytes(b"keep")
        with self.assertRaises(FileExistsError): self.build_lora()
        self.assertEqual(sentinel.read_bytes(), b"keep")
        different_out = self.root / "other"
        self.norm_path.write_bytes(b"old norm")
        with self.assertRaises(FileExistsError):
            m.materialize_lora(self.source, self.lora, self.native, different_out, self.norm_path)
        self.assertFalse(different_out.exists()); self.assertEqual(self.norm_path.read_bytes(), b"old norm")

    def test_seal_final_progress_and_complete_scope_are_load_bearing(self):
        original = (self.lora / "manifest.json").read_bytes()
        self.lora_manifest["progress"]["updates"] = 192
        self.reseal_lora()
        with self.assertRaisesRegex(ValueError, "final progress"): self.build_lora()
        self.lora_manifest["progress"]["updates"] = 193
        self.lora_manifest["entries"][0]["name"] = "model.layers.0.self_attn.q_proj.lora_A"
        self.reseal_lora()
        with self.assertRaisesRegex(ValueError, "full tensor scope"): self.build_lora()
        (self.lora / "manifest.json").write_bytes(original)
        value = json.loads(original); value["status"] = "FAILED"
        (self.lora / "manifest.json").write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "digest mismatch"): self.build_lora()
        self.assertFalse(self.out.exists())

    def test_nonfinite_optimizer_rejected_even_if_rehashed_and_resealed(self):
        entry = self.lora_manifest["entries"][0]
        path = self.lora / entry["file"]
        state = load_file(str(path)); state["exp_avg"].flatten()[0] = float("nan")
        save_file(state, str(path)); entry["sha256"] = file_hash(path); self.reseal_lora(); self.qualify_fixture()
        with self.assertRaisesRegex(ValueError, "nonfinite parameter/optimizer"):
            self.build_lora()
        self.assertFalse(self.out.exists())

    def test_beta_out_of_band_blocks_without_arm_export_or_clamping(self):
        for entry in self.lora_manifest["entries"]:
            if entry["name"] in m.NORM_NAMES:
                path = self.lora / entry["file"]
                state = load_file(str(path)); state["parameter"].fill_(100)
                save_file(state, str(path)); entry["sha256"] = file_hash(path)
        self.reseal_lora(); self.qualify_fixture()
        norm = self.build_lora()
        self.assertEqual(norm["status"], "BLOCKED_NORM_RATIO_OUT_OF_RANGE")
        self.assertGreater(norm["beta"], 2)
        self.assertFalse(self.out.exists()); self.assertTrue(self.norm_path.exists())

    def test_norm_receipt_tamper_and_scalar_no_overwrite_fail_closed(self):
        self.build_lora()
        scalar_out = self.root / "scalar"
        scalar_out.mkdir()
        with self.assertRaises(FileExistsError):
            m.materialize_scalar(self.source, self.native, scalar_out, self.norm_path)
        arm = self.out / "ARM.json"; arm.write_bytes(arm.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "LoRA arm identity differs"):
            m.materialize_scalar(self.source, self.native, self.root / "scalar-other", self.norm_path)
        self.assertFalse((self.root / "scalar-other").exists())

    def test_scalar_norm_discrepancy_blocks_export_without_search(self):
        self.build_lora()
        scalar_out = self.root / "scalar"
        with patch.object(m, "scalar_value", side_effect=lambda source, trained, beta: source.clone()):
            result = m.materialize_scalar(self.source, self.native, scalar_out, self.norm_path)
        self.assertEqual(result["status"], "BLOCKED_NORM_MATCH")
        self.assertEqual(result["relative_discrepancy"], 1.0)
        self.assertFalse(result["coefficient_search_performed"])
        self.assertFalse(scalar_out.exists())
        self.assertTrue((self.root / "scalar-NORM_MATCH_BLOCKED.json").is_file())

    def test_lease_and_resource_failures_create_no_arm(self):
        lock = self.root / "fixture.lock"; lock.write_bytes(b"another worker")
        with self.assertRaises(FileExistsError): self.build_lora()
        self.assertEqual(lock.read_bytes(), b"another worker")
        self.assertFalse(self.out.exists())

    def test_complete_manifest_without_passed_merge_qualification_cannot_export(self):
        qualification = self.run / "merge_qualification.json"
        value = json.loads(qualification.read_text()); value["status"] = "FAILED"
        qualification.write_text(json.dumps(value))
        result_path = self.run / "result.json"
        result = json.loads(result_path.read_text()); result["merge_qualification_sha256"] = file_hash(qualification)
        result_path.write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError, "not qualified"):
            self.build_lora()
        self.assertFalse(self.out.exists()); self.assertFalse(self.norm_path.exists())

    def test_rehashed_qualification_with_wrong_dense_hash_cannot_export(self):
        qualification = self.run / "merge_qualification.json"
        value = json.loads(qualification.read_text())
        value["dense_tensor_sha256"]["model.layers.8.self_attn.q_proj.weight"] = "0" * 64
        qualification.write_text(json.dumps(value))
        result_path = self.run / "result.json"
        result = json.loads(result_path.read_text()); result["merge_qualification_sha256"] = file_hash(qualification)
        result_path.write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError, "differs from qualified CPU merge"):
            self.build_lora()
        self.assertFalse(self.out.exists())

    def test_rehashed_qualification_cannot_relax_tolerance_or_hide_failed_comparison(self):
        qualification = self.run / "merge_qualification.json"
        result_path = self.run / "result.json"
        original_q, original_r = qualification.read_bytes(), result_path.read_bytes()
        for mutate in (lambda value: value.update(atol=.1),
                       lambda value: value["inputs"][0].update(argmax_mismatches=1),
                       lambda value: value["inputs"][1].update(input_ids_sha256="0" * 64)):
            value = json.loads(original_q); mutate(value); qualification.write_text(json.dumps(value))
            result = json.loads(original_r); result["merge_qualification_sha256"] = file_hash(qualification)
            result_path.write_text(json.dumps(result))
            with self.assertRaises(ValueError): self.build_lora()
            self.assertFalse(self.out.exists())
        qualification.write_bytes(original_q); result_path.write_bytes(original_r)

    def test_source_and_actual_training_bytes_remain_bound(self):
        source_sidecar = self.source / "tokenizer.json"
        saved = source_sidecar.read_bytes(); source_sidecar.write_bytes(saved + b" ")
        with self.assertRaisesRegex(ValueError, "source asset hash mismatch"):
            self.build_lora()
        source_sidecar.write_bytes(saved)
        (self.root / "prepared/microfit-train.jsonl").write_bytes(b"changed data")
        with self.assertRaisesRegex(ValueError, "actual prepared training bytes differ"):
            self.build_lora()
        self.assertFalse(self.out.exists())

    def test_late_worker_failure_overrules_complete_result_and_passed_merge(self):
        self.assertEqual(json.loads((self.run / "result.json").read_text())["status"], "COMPLETE")
        (self.run / "failure.json").write_text(json.dumps({"phase": "lease_cleanup", "failure": "late failure"}))
        with self.assertRaisesRegex(ValueError, "failure receipt"):
            self.build_lora()
        self.assertFalse(self.out.exists()); self.assertFalse(self.norm_path.exists())

    def test_failed_or_mismatched_supervisor_overrules_complete_result(self):
        path = self.run / "supervisor.json"
        summary_path = self.run / "run_summary.json"
        supervisor = json.loads(path.read_text()); supervisor["returncode"] = 2
        path.write_text(json.dumps(supervisor))
        with self.assertRaisesRegex(ValueError, "disagrees with its supervisor"):
            self.build_lora()
        summary = json.loads(summary_path.read_text()); summary["supervisor"] = supervisor
        summary_path.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError, "clean successful exit"):
            self.build_lora()
        self.assertFalse(self.out.exists())

    def test_supervisor_launch_wrong_output_reference_is_rejected(self):
        path = self.run / "launch.json"
        launch = json.loads(path.read_text()); launch["out"] = str(self.root / "another-run")
        path.write_text(json.dumps(launch))
        with self.assertRaisesRegex(ValueError, "different output directory"):
            self.build_lora()
        self.assertFalse(self.out.exists())

    def test_unsettled_actual_journal_overrules_success_summary(self):
        path = self.root / "journal/events/000001.json"
        event = json.loads(path.read_text()); event.pop("sha256")
        event["kind"] = "charge"; event.pop("actual_wall"); event.pop("receipt_sha256")
        path.write_text(json.dumps(seal(event)))
        with self.assertRaisesRegex(ValueError, "unsettled launch"):
            self.build_lora()
        self.assertFalse(self.out.exists())

    def test_actual_budget_breach_overrules_success_summary(self):
        path = self.root / "journal/events/000001.json"
        event = json.loads(path.read_text()); event.pop("sha256")
        event["actual_wall"]["total_owned_work_wall_seconds"] = 101
        path.write_text(json.dumps(seal(event)))
        with self.assertRaisesRegex(ValueError, "budget is breached"):
            self.build_lora()
        self.assertFalse(self.out.exists())

    def test_settlement_must_name_actual_successful_supervisor(self):
        path = self.root / "journal/events/000001.json"
        event = json.loads(path.read_text()); event.pop("sha256"); event["receipt_sha256"] = "0" * 64
        path.write_text(json.dumps(seal(event)))
        with self.assertRaisesRegex(ValueError, "different supervisor receipt"):
            self.build_lora()
        self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main()
