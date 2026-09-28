"""Mocked Stage C worker/hook tests: no torch import, checkpoints or native calls."""
from contextlib import ExitStack, nullcontext
import copy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from . import native_capture as n
from . import receipts as r


class FakeBackend:
    instances = []
    fail_ordinal = None

    def __init__(self, runtime, role):
        self.role, self.calls = role, 0
        self.tokenizer_encodings = 0
        self.width = 1024 if role == "student_base" else 2560
        self.instances.append(self)

    def free_gpu_bytes(self):
        return 20 * n.GIB

    def load(self, spec, rows):
        self.loaded = True
        self.tokenizer_encodings = 32
        assert len(rows) == 32

    def capture(self, ids, layers):
        self.calls += 1
        if self.calls == self.fail_ordinal:
            raise RuntimeError("injected forward failure")
        return {layer: np.full(self.width, layer + self.calls / 100, dtype="float32") for layer in layers}


def synthetic_spec(root):
    lease = root / "rineya-gpu-0.lock"
    r.write_once(lease, {"pid": 1234, "token": "test-owner", "created": 0})
    reservation = root / "reservations"
    reservation.mkdir()
    assets = {}
    for name in ("student", "donor", "adapter"):
        directory = root / name
        directory.mkdir()
        (directory / "model.safetensors").write_bytes(b"not-a-model")
        files = {"model.safetensors": r.file_digest(directory / "model.safetensors")}
        assets[name] = {"directory": str(directory), "files": files, "model_sha256": r.digest(files)}
    token_dir = root / "tokenizer"
    token_dir.mkdir()
    (token_dir / "tokenizer.json").write_text("not-a-tokenizer")
    tokenizer = {"directory": str(token_dir), "files": {"tokenizer.json": r.file_digest(token_dir / "tokenizer.json")}}
    rows, original = [], []
    for i in range(32):
        row = {"id": f"row-{i}", "parent_id": f"parent-{i}", "split": "fit" if i < 24 else "map_check", "open": True}
        original.append({**row, "prompt_sha256": r.digest([{"role": "user", "content": f"prompt {i}"}])})
        rows.append({**row, "messages_sha256": original[-1]["prompt_sha256"],
                     "prompt": f"rendered {i}", "prompt_sha256": r.digest(f"rendered {i}"),
                     "input_ids": [1, 2, i + 3], "input_ids_sha256": r.digest([1, 2, i + 3])})
    coordinates = [{"id": row["id"], "input_ids_sha256": row["input_ids_sha256"], "position": 2,
                    "token_id": row["input_ids"][-1]} for row in rows]
    calibration = {"rows": rows, "source_binding_rows": original, "rows_sha256": r.digest(rows),
                   "calibration_sha256": r.digest(original), "dataset_sha256": r.digest("data"),
                   "source_binding_file_sha256": r.digest("original-file"), "tokenizer_sha256": r.digest(tokenizer["files"]),
                   "prompt_positions_sha256": r.digest(coordinates), "layer_map_sha256": r.digest("frozen-map")}
    r.write_once(root / "CALIBRATION.json", {"rows": original, "calibration_sha256": calibration["calibration_sha256"],
                                           "dataset_sha256": calibration["dataset_sha256"]})
    calibration["source_binding_file_sha256"] = r.file_digest(root / "CALIBRATION.json")
    mapping = {"capture_pairs_student_donor": [[0, 0], [13, 17], [27, 35]],
               "donor_to_student": [min(range(28), key=lambda j: (abs(i * 27 - j * 35), j)) for i in range(36)]}
    r.write_once(root / "ARCHITECTURE.json", {"layer_map": mapping})
    calibration["layer_map_sha256"] = r.digest(mapping)
    calibration["architecture_file_sha256"] = r.file_digest(root / "ARCHITECTURE.json")
    source_files = {"experiments/cross_scale_composable_001/native_capture.py": r.file_digest(Path(n.__file__))}
    student_runtime = {"python_executable": sys.executable, "packages": dict.fromkeys(
        ("torch", "transformers", "safetensors", "numpy", "psutil"), "mock"),
        "device": "cuda:0", "seed": 1729, "attention": "eager"}
    donor_runtime = {**copy.deepcopy(student_runtime), "attention": "sdpa", "quantization": {
        "type": "nf4", "double_quant": True, "compute_dtype": "bfloat16", "nonquantized_dtype_after_prepare": "float32"}}
    donor_runtime["packages"].update(dict.fromkeys(("bitsandbytes", "peft", "accelerate"), "mock"))
    return {"schema": 1, "experiment": "CSC-001", "stage": "C", "native_capture_authorized": True,
            "source_commit": "a" * 40, "source_files": source_files, "implementation_digest": r.digest(source_files),
            "max_total_forwards": 96, "max_role_forwards": 32, "session_id": "mock-session",
            "frozen_registration_sha256": r.digest("immutable-registration"),
            "deadline_utc": (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat(),
            "lease": {"path": str(lease), "token": "test-owner", "owner_pid": 1234},
            "reservation_dir": str(reservation), "assets": assets, "tokenizer": tokenizer, "calibration": calibration,
            "runtime": {"student_base": student_runtime, "donor": donor_runtime}}


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="csc-mocked-worker-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.spec = synthetic_spec(self.root)
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(patch.object(n.r, "verify_source", return_value=self.spec["source_commit"]))
        self.patches.enter_context(patch.object(n.importlib.metadata, "version", return_value="mock"))
        self.patches.enter_context(patch.object(n, "available_host_bytes", return_value=20 * n.GIB))
        self.patches.enter_context(patch.object(n, "NativeBackend", FakeBackend))
        self.patches.enter_context(patch.object(n, "CALIBRATION_FILE", self.root / "CALIBRATION.json"))
        self.patches.enter_context(patch.object(n, "ARCHITECTURE_FILE", self.root / "ARCHITECTURE.json"))
        self.patches.enter_context(patch.object(n.tempfile, "gettempdir", return_value=str(self.root)))
        FakeBackend.instances, FakeBackend.fail_ordinal = [], None

    def execute(self, role="student_base", name="attempt"):
        directory = self.root / name
        directory.mkdir()
        supervision = directory / "supervision"
        supervision.mkdir()
        r.write_once(supervision / "launch.json", {"supervisor_pid": 1234})
        spec_path = directory / "SPEC.json"
        r.write_once(spec_path, self.spec)
        return n.run_worker(spec_path, r.file_digest(spec_path), role, supervision / "capture", supervision)

    def test_exact32_calls_three_layers_and_persisted_hashes(self):
        result = self.execute()
        self.assertEqual(FakeBackend.instances[0].calls, 32)
        self.assertEqual(result["native_model_forwards"], 32)
        self.assertEqual(result["generation_calls"], 0)
        self.assertEqual(result["optimizer_steps"], 0)
        self.assertEqual(result["native_tokenizer_encodings"], 32)
        output = self.root / "attempt/supervision/capture"
        self.assertEqual(len(list(output.glob("row-*-layer-*.npy"))), 96)
        for layer, capture in result["captures"].items():
            value = np.load(output / capture["file"], allow_pickle=False)
            self.assertEqual(value.shape, (32, 1024))
            self.assertEqual(value.dtype, np.dtype("float32"))
            self.assertEqual(capture["file_sha256"], r.file_digest(output / capture["file"]))
            self.assertEqual(capture["metadata"]["capture_site"], "block_output_raw")
        journal = Path(self.spec["reservation_dir"])
        self.assertEqual(len(list((journal / "student_base").glob("*-RESERVED.json"))), 32)
        self.assertFalse((journal / "ACTIVE.json").exists())
        self.assertTrue((output / "RESULT.json").is_file())

    def test_failed_forward_keeps_charged_reservation_and_no_success(self):
        FakeBackend.fail_ordinal = 2
        with self.assertRaisesRegex(RuntimeError, "injected"):
            self.execute()
        output = self.root / "attempt/supervision/capture"
        failure = n.read_json(output / "FAILURE.json")
        self.assertEqual(failure["reserved_forwards"], 2)
        self.assertFalse((output / "RESULT.json").exists())
        journal = Path(self.spec["reservation_dir"]) / "student_base"
        self.assertTrue((journal / "002-RESERVED.json").exists())
        self.assertFalse((journal / "002-COMPLETED.json").exists())
        self.assertEqual(n.read_json(journal / "TERMINAL.json")["status"], "FAILED")

    def test_failed_role_cannot_be_retried(self):
        FakeBackend.fail_ordinal = 1
        with self.assertRaises(RuntimeError):
            self.execute()
        FakeBackend.fail_ordinal = None
        with self.assertRaises(FileExistsError):
            self.execute(name="retry")
        self.assertEqual(len(FakeBackend.instances), 1)

    def test_ram_refusal_precedes_native_import(self):
        with patch.object(n, "available_host_bytes", return_value=6 * n.GIB):
            with self.assertRaisesRegex(r.Blocked, "RAM"):
                self.execute()
        self.assertEqual(FakeBackend.instances, [])
        failure = n.read_json(self.root / "attempt/supervision/capture/FAILURE.json")
        self.assertEqual(failure["reserved_forwards"], 0)

    def test_donor12gb_guard_is_not_student_guard(self):
        with patch.object(n, "available_host_bytes", return_value=11 * n.GIB):
            with self.assertRaisesRegex(r.Blocked, "12 GiB"):
                self.execute("donor_base")
        self.assertEqual(FakeBackend.instances, [])

    def test_runtime_and_owner_and_calibration_negatives(self):
        modes = ("nonce", "owner", "expiry", "max_calls", "rowhash", "messages", "tokenhash", "overflow", "dtype")
        for mode in modes:
            spec = copy.deepcopy(self.spec)
            if mode == "nonce": spec["lease"]["token"] = "wrong"
            if mode == "owner": spec["lease"]["owner_pid"] = 99
            if mode == "expiry": spec["deadline_utc"] = "2000-01-01T00:00:00+00:00"
            if mode == "max_calls": spec["max_role_forwards"] = 33
            if mode == "rowhash": spec["calibration"]["rows_sha256"] = "0" * 64
            if mode == "messages": spec["calibration"]["rows"][0]["messages_sha256"] = "0" * 64
            if mode == "tokenhash": spec["calibration"]["rows"][0]["input_ids_sha256"] = "0" * 64
            if mode == "overflow": spec["calibration"]["rows"][0]["input_ids"] = [1] * 257
            if mode == "dtype": spec["runtime"]["donor"]["quantization"]["compute_dtype"] = "float16"
            if mode in ("messages", "tokenhash", "overflow"):
                spec["calibration"]["rows_sha256"] = r.digest(spec["calibration"]["rows"])
            with self.subTest(mode=mode), self.assertRaises(r.Blocked):
                n.validate_spec(spec, "donor_base" if mode == "dtype" else "student_base")

    def test_asset_corruption_and_extra_weights_refused_before_import(self):
        asset = self.spec["assets"]["student"]
        (Path(asset["directory"]) / "extra.bin").write_bytes(b"extra")
        with self.assertRaisesRegex(r.Blocked, "unexpected"):
            self.execute()
        self.assertEqual(FakeBackend.instances, [])

    def test_session_excludes_concurrent_role_and_enforces_budget(self):
        journal = n.CaptureJournal(self.spec, "c" * 64, "student_base")
        self.addCleanup(journal.release)
        with self.assertRaises(FileExistsError):
            n.CaptureJournal(self.spec, "c" * 64, "donor_base")
        row = self.spec["calibration"]["rows"][0]
        for _ in range(32): journal.reserve(row)
        with self.assertRaisesRegex(r.Blocked, "exhausted"):
            journal.reserve(row)

    def test_same_donor_runtime_binding_and_distinct_skilled_identity(self):
        base = self.execute("donor_base", "base")
        skilled = self.execute("donor_skilled", "skilled")
        a, b = base["captures"]["35"]["metadata"], skilled["captures"]["35"]["metadata"]
        self.assertEqual(a["runtime_binding_sha256"], b["runtime_binding_sha256"])
        self.assertEqual(b["anchor_checkpoint_sha256"], a["model_sha256"])
        self.assertNotEqual(a["model_sha256"], b["model_sha256"])

    def test_role_worker_specs_differ_but_session_binding_is_stable(self):
        first = n.CaptureJournal(self.spec, "a" * 64, "student_base")
        first.finish("COMPLETE")
        first.release()
        second = n.CaptureJournal(self.spec, "b" * 64, "donor_base")
        second.finish("COMPLETE")
        second.release()
        changed = copy.deepcopy(self.spec)
        changed["frozen_registration_sha256"] = "c" * 64
        with self.assertRaisesRegex(r.Blocked, "session changed"):
            n.CaptureJournal(changed, "d" * 64, "donor_skilled")

    def test_self_consistent_replaced_calibration_refused(self):
        changed = copy.deepcopy(self.spec)
        changed["calibration"]["source_binding_rows"][0]["prompt_sha256"] = "c" * 64
        changed["calibration"]["calibration_sha256"] = r.digest(changed["calibration"]["source_binding_rows"])
        with self.assertRaisesRegex(r.Blocked, "committed calibration"):
            n.validate_spec(changed, "student_base")

    def test_wrong_map_hash_and_alternate_same_name_lease_refused(self):
        changed = copy.deepcopy(self.spec)
        changed["calibration"]["layer_map_sha256"] = "c" * 64
        with self.assertRaisesRegex(r.Blocked, "layer map"):
            n.validate_spec(changed, "student_base")
        changed = copy.deepcopy(self.spec)
        changed["lease"]["path"] = str(self.root / "other" / "rineya-gpu-0.lock")
        with self.assertRaisesRegex(r.Blocked, "lease path"):
            n.validate_spec(changed, "student_base")

    def test_module_import_has_no_native_model_stack(self):
        from experiments.rab001_exploration.torchfree import assert_import_is_clean
        assert_import_is_clean(self, "experiments.cross_scale_composable_001.native_capture")


class Tensor:
    def __init__(self, array): self.array = np.asarray(array)
    @property
    def shape(self): return self.array.shape
    requires_grad = False
    def __getitem__(self, index): return Tensor(self.array[index])
    def detach(self): return self
    def to(self, **kwargs): return Tensor(self.array.astype("float32"))
    def contiguous(self): return self
    def numpy(self): return self.array


class Handle:
    def __init__(self, hooks, callback): self.hooks, self.callback = hooks, callback
    def remove(self): self.hooks.remove(self.callback)


class Block:
    def __init__(self, index): self.index, self.hooks = index, []
    def register_forward_hook(self, callback):
        self.hooks.append(callback)
        return Handle(self.hooks, callback)


class Core:
    def __init__(self, layers, width): self.layers, self.width, self.hooks, self.calls = layers, width, [], 0
    def register_forward_pre_hook(self, callback):
        self.hooks.append(callback)
        return Handle(self.hooks, callback)
    def __call__(self, **kwargs):
        self.calls += 1
        assert kwargs["use_cache"] is False
        assert kwargs["return_dict"] is True
        for hook in self.hooks: hook(self, ())
        tokens = kwargs["input_ids"].array[0]
        for block in self.layers:
            array = np.tile(tokens[None, :, None], (1, 1, self.width)) + block.index
            for hook in block.hooks: hook(block, (), (Tensor(array),))
        return object()


class FakeTorch:
    long, float32 = "long", "float32"
    @staticmethod
    def inference_mode(): return nullcontext()
    @staticmethod
    def tensor(value, **kwargs): return Tensor(value)
    @staticmethod
    def ones_like(value): return Tensor(np.ones_like(value.array))
    @staticmethod
    def isfinite(value): return np.isfinite(value.array)


class HookTests(unittest.TestCase):
    def test_raw_block_last_token_hooks_one_core_forward_cleanup(self):
        backend = n.NativeBackend.__new__(n.NativeBackend)
        backend.torch, backend.width = FakeTorch(), 4
        backend.layers = [Block(i) for i in range(3)]
        backend.core = Core(backend.layers, backend.width)
        result = backend.capture([9, 15, 22], [0, 2])
        np.testing.assert_array_equal(result[0], np.full(4, 22, dtype="float32"))
        np.testing.assert_array_equal(result[2], np.full(4, 24, dtype="float32"))
        self.assertEqual(backend.core.calls, 1)
        self.assertFalse(backend.core.hooks)
        self.assertTrue(all(not layer.hooks for layer in backend.layers))


if __name__ == "__main__":
    unittest.main()
