"""Tiny CPU fixtures test custody failures and FP32 interpolation without a model."""
from __future__ import annotations

import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from research.retention_dose_parameterization_v1 import build_arm as builder
from experiments.model_quality_gpu_v1.integrity import digest, file_hash, seal, write_new


def tensor_file(path, tensors):
    header, data = {}, b""
    for name, values in tensors.items():
        raw = struct.pack("<" + "f" * len(values), *values)
        header[name] = {"dtype": "F32", "shape": [len(values)],
                        "data_offsets": [len(data), len(data) + len(raw)]}
        data += raw
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)


class ArmBuilderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source, self.checkpoint = self.root / "source", self.root / "checkpoint"
        self.source.mkdir(); self.checkpoint.mkdir()
        names = sorted(builder.EXPECTED_NAMES)
        tensor_file(self.source / "model.safetensors",
                    {name: [2.0, 4.0] for name in [*names, "model.embed_tokens.weight"]})
        for name in builder.SIDECARS:
            (self.source / name).write_bytes(b"{}\r\n")
        (self.source / "config.json").write_text(json.dumps({"num_hidden_layers": 28, "tie_word_embeddings": True}))
        assets = {name: file_hash(self.source / name) for name in ("model.safetensors", *builder.SIDECARS)}
        entries = []
        for i, name in enumerate(names):
            filename = f"tensor-{i:04d}.safetensors"
            tensor_file(self.checkpoint / filename,
                        {"parameter": [6.0, 12.0], "exp_avg": [0.0, 0.0], "exp_avg_sq": [0.0, 0.0]})
            entries.append({"name": name, "file": filename, "shape": [2],
                            "sha256": file_hash(self.checkpoint / filename)})
        binding = {key: "fixture" for key in ("experiment", "plan_sha256", "data_contract_sha256",
                    "grant_sha256", "code_sha256", "mq_code_sha256", "checkout")}
        binding.update(runtime={"fixture": True}, source={"assets": assets, "sha256": digest(assets)})
        self.manifest = {"schema": 1, "experiment": "MQ-001", "status": "COMPLETE",
                         "binding": binding, "progress": {"updates": 2}, "rng": {}, "entries": entries}
        write_new(self.checkpoint / "manifest.json", seal(self.manifest))

    def test_corrupt_tensor_refused_before_output(self):
        tensor = self.checkpoint / self.manifest["entries"][0]["file"]
        with tensor.open("ab") as stream:
            stream.write(b"corruption")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            builder.inspect_inputs(self.source, self.checkpoint)

    def test_resealed_wrong_layer_inventory_refused(self):
        self.manifest["entries"][0]["name"] = "model.layers.0.input_layernorm.weight"
        (self.checkpoint / "manifest.json").write_text(json.dumps(seal(self.manifest)))
        with self.assertRaisesRegex(ValueError, "220 trainable"):
            builder.inspect_inputs(self.source, self.checkpoint)

    def test_source_bytes_and_manifest_seal_bound(self):
        self.assertEqual(len(builder.inspect_inputs(self.source, self.checkpoint)["manifest"]["entries"]), 220)
        raw = json.loads((self.checkpoint / "manifest.json").read_text())
        raw["progress"]["updates"] = 3
        (self.checkpoint / "manifest.json").write_text(json.dumps(raw))
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            builder.inspect_inputs(self.source, self.checkpoint)
        (self.checkpoint / "manifest.json").write_text(json.dumps(seal(self.manifest)))
        (self.source / "tokenizer.json").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "source asset hash"):
            builder.inspect_inputs(self.source, self.checkpoint)

    def test_resource_refusals_preserve_other_lease_and_create_no_arm(self):
        lock, out = self.root / "fixture.lock", self.root / "refused-arm"
        lock.write_bytes(b"existing-worker-lease")
        with patch.object(builder, "resource_lock_path", return_value=lock):
            with self.assertRaises(FileExistsError):
                builder.materialize(self.source, self.checkpoint, 0.5, out)
            self.assertEqual(lock.read_bytes(), b"existing-worker-lease")
            lock.unlink()
            with patch("psutil.virtual_memory") as memory:
                memory.return_value.available = int(4.5 * builder.GIB)
                with self.assertRaisesRegex(RuntimeError, "RAM guard"):
                    builder.materialize(self.source, self.checkpoint, 0.5, out)
            self.assertFalse(lock.exists())
            self.assertFalse(out.exists())

    def test_interpolation_endpoints_frozen_weights_receipts_and_no_overwrite(self):
        from safetensors import safe_open
        with patch.object(builder, "require_ram", return_value=8 * builder.GIB), \
                patch.object(builder, "resource_lock_path", return_value=self.root / "fixture.lock"), \
                patch.object(builder.shutil, "disk_usage") as disk:
            disk.return_value.free = 100 * builder.GIB
            name = sorted(builder.EXPECTED_NAMES)[0]
            for alpha, expected in ((0.0, [2.0, 4.0]), (0.25, [3.0, 6.0]),
                                    (0.5, [4.0, 8.0]), (0.75, [5.0, 10.0]), (1.0, [6.0, 12.0])):
                out = self.root / f"arm-{alpha}"
                receipt = builder.materialize(self.source, self.checkpoint, alpha, out)
                with safe_open(str(out / "model.safetensors"), framework="pt") as tensors:
                    self.assertEqual(tensors.get_tensor(name).tolist(), expected)
                    self.assertEqual(tensors.get_tensor("model.embed_tokens.weight").tolist(), [2.0, 4.0])
                    self.assertEqual(str(tensors.get_tensor(name).dtype), "torch.float32")
                self.assertEqual(receipt["out_weight_sha256"], file_hash(out / "model.safetensors"))
                self.assertEqual(receipt["checkpoint_binding"], self.manifest["binding"])
                for sidecar in builder.SIDECARS:
                    self.assertEqual((out / sidecar).read_bytes(), (self.source / sidecar).read_bytes())
                with self.assertRaises(FileExistsError):
                    builder.materialize(self.source, self.checkpoint, alpha, out)


if __name__ == "__main__":
    unittest.main()
