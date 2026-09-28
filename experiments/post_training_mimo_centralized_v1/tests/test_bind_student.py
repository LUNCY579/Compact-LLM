"""Identity and corruption controls; tiny payload fixtures, no ML imports."""
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "bind_student.py"
SPEC = importlib.util.spec_from_file_location("central_bind_student", SCRIPT)
binder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(binder)


class BindingControls(unittest.TestCase):
    def fixture(self, root, header, payload=b"\0" * 8):
        path = Path(root) / "fixture.safetensors"
        encoded = json.dumps(header).encode()
        path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
        return path

    def test_header_count_without_tensor_import(self):
        with tempfile.TemporaryDirectory() as root:
            path = self.fixture(root, {"w": {"shape": [2, 2], "dtype": "BF16", "data_offsets": [0, 8]}})
            result = binder.inspect_safetensors(path)
            self.assertEqual(result["parameter_count_serialized"], 4)
            self.assertEqual(result["tensor_count"], 1)

    def test_corrupt_offsets_and_trailing_payload_rejected(self):
        for offsets, payload in (([1, 9], b"\0" * 9), ([0, 7], b"\0" * 7), ([0, 8], b"\0" * 9)):
            with self.subTest(offsets=offsets, size=len(payload)), tempfile.TemporaryDirectory() as root:
                path = self.fixture(root, {"w": {"shape": [2, 2], "dtype": "BF16", "data_offsets": offsets}}, payload)
                with self.assertRaises(ValueError):
                    binder.inspect_safetensors(path)

    def test_duplicate_tensor_names_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "fixture.safetensors"
            entry = '"w":{"shape":[2,2],"dtype":"BF16","data_offsets":[0,8]}'
            header = ("{" + entry + "," + entry + "}").encode()
            path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 8)
            with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                binder.inspect_safetensors(path)

    def test_config_alias_and_changed_shape_rejected(self):
        config = {**binder.GEOMETRY, "torch_dtype": "bfloat16"}
        with self.assertRaisesRegex(ValueError, "tensor names/shapes"):
            binder.validate_geometry(config, {"tensors": {}})
        with self.assertRaisesRegex(ValueError, "num_hidden_layers"):
            binder.validate_geometry({**config, "num_hidden_layers": 12}, {"tensors": {}})
        with self.assertRaisesRegex(ValueError, "custom loader"):
            binder.validate_geometry({**config, "auto_map": {"loader": "custom"}}, {"tensors": {}})

    def test_nonprimary_weight_hash_rejected_before_asset_reads(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "exact preregistered dense Titan"):
                binder.create_receipts(Path(root), "0" * 64)

    def test_write_once_keeps_old_receipts(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "new_binding"
            binder.write_receipts(target, {"receipt.json": {"version": 1}})
            with self.assertRaisesRegex(ValueError, "write-once"):
                binder.write_receipts(target, {"receipt.json": {"version": 2}})
            self.assertEqual(json.loads((target / "receipt.json").read_text())["version"], 1)


if __name__ == "__main__":
    unittest.main()
