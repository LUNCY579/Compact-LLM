"""Offline custody/refusal/arithmetic integration fixtures; no model asset reads."""
import copy
import json
import gzip
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import custody, exclusion, generator, native_proposal, registry, schema
from .packet import read_training_dataset


class CustodyTests(unittest.TestCase):
    def test_external_paths_exclude_git_and_codex(self):
        with self.assertRaisesRegex(ValueError, "STOP_OUTPUT"):
            custody.external(custody.REPO / "output")
        with self.assertRaisesRegex(ValueError, "STOP_OUTPUT"):
            custody.external(Path("C:/Users/clclc/.codex/output"))
        with tempfile.TemporaryDirectory(dir=custody.ARTIFACT_ROOT) as temporary:
            root = Path(temporary)
            (root / ".git").write_text("fixture")
            with self.assertRaisesRegex(ValueError, "STOP_OUTPUT"):
                custody.external(root / "out")

    def test_create_only_outputs_and_pinned_same_bytes(self):
        with tempfile.TemporaryDirectory(dir=custody.ARTIFACT_ROOT) as temporary:
            path = Path(temporary) / "receipt.json"
            custody.write_json(path, {"fixture_only": True})
            before = path.read_bytes()
            self.assertEqual(custody.read_pinned(path, custody.sha_bytes(before)), before)
            with self.assertRaises(FileExistsError):
                custody.write_json(path, {"rewrite": True})
            with self.assertRaisesRegex(ValueError, "DIGEST_MISMATCH"):
                custody.read_pinned(path, "0" * 64)
            self.assertEqual(path.read_bytes(), before)

    def test_source_metadata_verification_does_not_hash_weights(self):
        with tempfile.TemporaryDirectory(dir=custody.ARTIFACT_ROOT) as temporary:
            core = Path(temporary)
            manifest = {"core_id": "fixture", "checks": {"weight_sha256": custody.WEIGHT_SHA256,
                "weight_bytes": 16, "architecture": "Qwen3ForCausalLM", "dtype_counts": {"F32": 310}}}
            for name in custody.SIDECARS:
                (core / name).write_text("{}")
            (core / "CORE.json").write_text(json.dumps(manifest))
            (core / "config.json").write_text(json.dumps({"architectures": ["Qwen3ForCausalLM"], "torch_dtype": "float32"}))
            (core / "model.safetensors").write_bytes(b"fixture123456789")
            with patch.object(custody, "WEIGHT_BYTES", 16):
                identity = custody.source_identity(core)
            self.assertTrue(identity["metadata_verified"])
            self.assertFalse(identity["fresh_weight_digest_verified"])
            self.assertEqual(identity["checkpoint_weight_reads"], 0)
            (core / "config.json").write_text(json.dumps({"architectures": ["Qwen3AbstainForCausalLM"], "torch_dtype": "float32"}))
            with patch.object(custody, "WEIGHT_BYTES", 16), self.assertRaisesRegex(ValueError, "SOURCE_IDENTITY"):
                custody.source_identity(core)

    def test_native_dispatch_always_requires_separate_authorization(self):
        with self.assertRaisesRegex(RuntimeError, "SEPARATE_SLOT"):
            native_proposal.dispatch_native(permit={"fixture_fake_grant": True})

    def test_proposal_is_costed_but_not_native_qualification(self):
        fake = {"offline_validation_passed": True, "registry_sha256": registry.build_registry()["registry_sha256"],
                "source_manifest": {"fixture": "a" * 64},
                "development": {"qualification_row_ids": [f"fixture{i}" for i in range(8)],
                    "reload_readout_row_id": "fixture0", "path": "fixture-development.jsonl", "sha256": "b" * 64}}
        result = native_proposal.proposal(fake)
        self.assertEqual(result["attempt"]["registered_forward_requests_without_checkpoint_recompute"], 562)
        self.assertFalse(result["native_execution_authorized"])
        self.assertFalse(result["native_worker_implementation_in_this_milestone"])
        changed = copy.deepcopy(fake)
        changed["registry_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "REGISTRY_BINDING"):
            native_proposal.proposal(changed)

    def test_root_exclusion_matches_independent_generator_roles(self):
        p = {"before": {"A": 19, "B": 2, "C": 3}, "event": {"kind": "scale", "account": "A", "other": "B", "amount": 3, "status": "posted", "tick": 6}}
        self.assertEqual(exclusion.arithmetic_key("ledger", p), generator.operative_key("ledger", p))
        p["before"]["C"] = 900
        p["event"]["tick"] = 99
        self.assertEqual(exclusion.arithmetic_key("ledger", p), schema.canonical(("ledger_replay", "scale", "posted", 19, 3, 57)))

    def test_cross_split_pair_arithmetic_or_world_cannot_be_renamed_away(self):
        row = generator.build_split("dev")[0]
        altered = copy.deepcopy(row)
        altered.update(row_id="renamed", split="train")
        report = exclusion.audit_partition({"dev": [row], "train": [altered]})
        self.assertFalse(report["passed"])
        self.assertTrue(any("operative" in e for e in report["errors"]))
        self.assertTrue(any("world" in e for e in report["errors"]))

    def test_compressed_training_rows_preserve_exact_dose_and_payload(self):
        rows = generator.training_rows(registry.build_registry()["recipes"][0], 17, [])
        payload = "".join(schema.canonical(r) + "\n" for r in rows).encode("utf-8")
        compressed = gzip.compress(payload, mtime=0)
        with tempfile.TemporaryDirectory(dir=custody.ARTIFACT_ROOT) as temporary:
            path = Path(temporary) / "fixture.jsonl.gz"
            path.write_bytes(compressed)
            descriptor = {"path": str(path), "encoding": "gzip-jsonl", "examples": 2000,
                "sha256": custody.sha_bytes(compressed), "payload_sha256": custody.sha_bytes(payload),
                "payload_bytes": len(payload), "replay_examples": 0}
            self.assertEqual(read_training_dataset(descriptor), rows)
            with self.assertRaisesRegex(ValueError, "PAYLOAD_DIGEST"):
                read_training_dataset({**descriptor, "payload_sha256": "0" * 64})
            with self.assertRaisesRegex(ValueError, "DATASET_DOSE"):
                read_training_dataset({**descriptor, "examples": 1999})
