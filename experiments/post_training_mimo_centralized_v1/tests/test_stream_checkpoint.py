"""Bounded loader controls; framework-free by default, optional tiny CPU BF16 test."""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import struct
import tempfile
import unittest

from experiments.post_training_mimo_centralized_v1.scripts import stream_checkpoint as loader
from experiments.post_training_mimo_centralized_v1.scripts.qualify_native import CallAccounting, verify_call_journal


class FakeTensor:
    def __init__(self, shape, dtype, contiguous=True):
        self.shape, self.dtype, self.contiguous = tuple(shape), dtype, contiguous
        self.storage = bytearray(math.prod(shape) * 2)
        self.copy_calls = []
        self.corrupt = False

    def is_contiguous(self):
        return self.contiguous

    def detach(self):
        return self

    def reshape(self, value):
        assert value == -1
        return FakeFlat(self)


class FakeFlat:
    def __init__(self, tensor):
        self.tensor = tensor

    def __getitem__(self, index):
        return FakeDestination(self.tensor, index.start * 2, index.stop * 2)


class FakeDestination:
    def __init__(self, tensor, begin, end):
        self.tensor, self.begin, self.end = tensor, begin, end

    def copy_(self, source, non_blocking):
        assert non_blocking is False
        assert self.end - self.begin == len(source)
        self.tensor.storage[self.begin:self.end] = source
        if self.tensor.corrupt:
            self.tensor.storage[self.begin] ^= 1
        self.tensor.copy_calls.append((self.begin, self.end, non_blocking))


class FakeTorch:
    bfloat16 = object()

    def __init__(self):
        self.buffer_ids = set()
        self.buffer_lengths = []

    def frombuffer(self, view, dtype):
        assert dtype is self.bfloat16 and len(view) % 2 == 0
        self.buffer_ids.add(id(view.obj))
        self.buffer_lengths.append(len(view))
        return view


class FakeModel:
    def __init__(self, states):
        self.states = states

    def state_dict(self):
        return dict(self.states)


def fixture(large=False):
    torch = FakeTorch()
    states, header, payload = {}, {}, bytearray()
    for index in range(loader.SERIALIZED_TENSORS):
        name = f"weight{index:03d}"
        count = loader.CHUNK_BYTES // 2 + 3 if large and index == 0 else 1
        data = b"\x80\x3f" * count
        states[name] = FakeTensor([count], torch.bfloat16)
        header[name] = {"dtype": "BF16", "shape": [count],
                        "data_offsets": [len(payload), len(payload) + len(data)]}
        payload.extend(data)
    states["lm_head.weight"] = states["weight000"]
    return torch, FakeModel(states), header, payload


def write_fixture(path, header, payload, raw_header=None):
    raw = json.dumps(header, separators=(",", ":")).encode() if raw_header is None else raw_header
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


def execute(torch, model, path, digest=None):
    events = []
    accounting = CallAccounting(lambda event, **values: events.append({"event": event, **values}), 0)
    digest = digest or (lambda framework, tensor: hashlib.sha256(tensor.storage).hexdigest())
    result = loader.load_streamed(torch, model, path, accounting, digest)
    verify_call_journal(events, accounting.snapshot(), 0)
    return result, events, accounting.snapshot()


class StreamCheckpointTests(unittest.TestCase):
    def test_chunked_bytes_identity_single_buffer_and_exact_journal(self):
        torch, model, header, payload = fixture(large=True)
        state_before = random.getstate()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.safetensors"
            write_fixture(path, header, payload)
            (hashes, diagnostics), events, snapshot = execute(torch, model, path)
        self.assertEqual(random.getstate(), state_before)
        for name, entry in header.items():
            begin, end = entry["data_offsets"]
            self.assertEqual(model.states[name].storage, payload[begin:end])
            self.assertEqual(hashes[name], hashlib.sha256(payload[begin:end]).hexdigest())
        self.assertEqual(len(torch.buffer_ids), 1)
        self.assertEqual(max(torch.buffer_lengths), loader.CHUNK_BYTES)
        self.assertEqual(torch.buffer_lengths[:2], [loader.CHUNK_BYTES, 6])
        self.assertEqual(diagnostics["payload_chunk_count"], 311)
        self.assertFalse(diagnostics["memory_mapped"])
        self.assertFalse(diagnostics["cpu_model_allocated"])
        self.assertEqual(snapshot["operations"]["tensor_copy"], dict(attempted=310, returned=310, failed=0, pending=0))
        self.assertEqual([row["stage"] for row in events if row["event"] == "operation_attempt"],
                         ["tensor_copy:" + name for name in sorted(header)])

    def test_metadata_and_reordered_header_do_not_change_bytes(self):
        torch, model, header, payload = fixture()
        header = {"__metadata__": {"format": "pt"}, **dict(reversed(list(header.items())))}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.safetensors"
            write_fixture(path, header, payload)
            (hashes, diagnostics), _, _ = execute(torch, model, path)
        self.assertEqual(len(hashes), 310)
        self.assertEqual(diagnostics["payload_bytes"], 620)

    def test_header_prefix_bounds_and_bad_json(self):
        torch, model, _, _ = fixture()
        cases = [b"", b"\x00" * 7, struct.pack("<Q", loader.MAX_HEADER_BYTES + 1),
                 struct.pack("<Q", 0), struct.pack("<Q", 8) + b"{}",
                 struct.pack("<Q", 2) + b"[]", struct.pack("<Q", 2) + b"\xff\xff",
                 struct.pack("<Q", 4) + b"nope"]
        for raw in cases:
            with self.subTest(raw=raw[:12]), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "weights.safetensors"
                path.write_bytes(raw)
                with self.assertRaises(RuntimeError):
                    execute(torch, model, path)

    def test_duplicate_keys_and_nonfinite_or_float_numbers_rejected(self):
        torch, model, _, _ = fixture()
        cases = [b'{"x":{},"x":{}}', b'{"x":{"shape":[1],"shape":[1]}}',
                 b'{"x":NaN}', b'{"x":Infinity}', b'{"x":1e999}', b'{"x":1.0}']
        for raw in cases:
            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "weights.safetensors"
                write_fixture(path, {}, b"", raw_header=raw)
                with self.assertRaises(RuntimeError):
                    execute(torch, model, path)

    def test_malformed_tensor_headers_rejected_before_any_copy(self):
        cases = []
        for field, value in (("dtype", "F32"), ("shape", [2]), ("shape", [True]), ("shape", [-1]),
                             ("shape", [0]), ("data_offsets", [0, 1]), ("data_offsets", [True, 2]),
                             ("data_offsets", [-1, 1]), ("data_offsets", [0, 99999])):
            cases.append((field, value))
        for field, value in cases:
            torch, model, header, payload = fixture()
            header["weight000"][field] = value
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "weights.safetensors"
                write_fixture(path, header, payload)
                with self.assertRaises(RuntimeError):
                    execute(torch, model, path)
                self.assertEqual(torch.buffer_lengths, [])

    def test_offset_overlap_gap_and_trailing_unclaimed_payload_rejected(self):
        for mode in ("overlap", "gap", "trailing", "truncated"):
            torch, model, header, payload = fixture()
            if mode == "overlap":
                header["weight001"]["data_offsets"] = [0, 2]
            elif mode == "gap":
                header["weight001"]["data_offsets"] = [3, 5]
            elif mode == "trailing":
                payload.extend(b"xx")
            else:
                del payload[-1]
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "weights.safetensors"
                write_fixture(path, header, payload)
                with self.assertRaises(RuntimeError):
                    execute(torch, model, path)

    def test_key_count_fields_metadata_and_target_contract_rejected(self):
        for mode in ("missing", "unknown", "extra_field", "metadata", "dtype", "contiguity"):
            torch, model, header, payload = fixture()
            if mode == "missing":
                del header["weight001"]
            elif mode == "unknown":
                header["unknown"] = header.pop("weight001")
            elif mode == "extra_field":
                header["weight001"]["extra"] = True
            elif mode == "metadata":
                header["__metadata__"] = {"format": 1}
            elif mode == "dtype":
                model.states["weight001"].dtype = "float32"
            else:
                model.states["weight001"].contiguous = False
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "weights.safetensors"
                write_fixture(path, header, payload)
                with self.assertRaises(RuntimeError):
                    execute(torch, model, path)

    def test_copy_corruption_detected_by_loaded_tensor_hash(self):
        torch, model, header, payload = fixture()
        model.states["weight000"].corrupt = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.safetensors"
            write_fixture(path, header, payload)
            with self.assertRaisesRegex(RuntimeError, "loaded tensor mismatch"):
                execute(torch, model, path)


@unittest.skipUnless(os.environ.get("RINEYA_STREAM_TORCH_CPU_TEST") == "1", "explicit tiny CPU Torch test only")
class TorchCpuStreamTests(unittest.TestCase):
    def test_actual_bfloat16_bytes_and_torch_rng_are_unchanged(self):
        import torch
        _, _, header, payload = fixture()
        class CpuModel:
            def __init__(self):
                self.states = {name: torch.zeros(entry["shape"], dtype=torch.bfloat16) for name, entry in header.items()}
                self.states["lm_head.weight"] = self.states["weight000"]
            def state_dict(self):
                return dict(self.states)
        model = CpuModel()
        before = torch.get_rng_state().clone()
        digest = lambda framework, value: hashlib.sha256(value.detach().view(torch.uint8).numpy().tobytes()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.safetensors"
            write_fixture(path, header, payload)
            (hashes, diagnostics), _, _ = execute(torch, model, path, digest)
        self.assertTrue(torch.equal(torch.get_rng_state(), before))
        self.assertEqual(diagnostics["serialized_tensors"], 310)
        self.assertTrue(all(torch.equal(value, torch.ones_like(value)) for value in model.states.values()))
        self.assertEqual(len(hashes), 310)


if __name__ == "__main__":
    unittest.main()
