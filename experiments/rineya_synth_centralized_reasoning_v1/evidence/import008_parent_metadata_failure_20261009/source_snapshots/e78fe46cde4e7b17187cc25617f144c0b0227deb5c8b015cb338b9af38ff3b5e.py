"""Stdlib-only loader fixtures; no real Torch, model assets or CUDA execution."""
from __future__ import annotations

from contextlib import contextmanager, ExitStack, nullcontext
import copy
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parents[1]
sys.path.insert(0, str(PACKAGE))
import native_loader as loader


def write_source(path, entries=None, *, payload=None, raw_header=None):
    if entries is None:
        entries = {"a": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}
    raw = json.dumps(entries, separators=(",", ":")).encode() if raw_header is None else raw_header
    if payload is None:
        payload = bytes(4)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
    return path


class HeaderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "toy.safetensors"

    def test_valid_header_metadata_offsets_and_shape(self):
        entries = {"__metadata__": {"format": "pt"},
                   "b": {"dtype": "BF16", "shape": [1, 2], "data_offsets": [4, 8]},
                   "a": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}
        write_source(self.path, entries, payload=bytes(8))
        result = loader.read_header(self.path)
        self.assertEqual({"format": "pt"}, result["metadata"])
        self.assertEqual({"a", "b"}, set(result["entries"]))
        self.assertEqual(8, result["payload_bytes"])
        self.assertEqual(self.path.stat().st_size, result["file_bytes"])

    def test_duplicate_top_level_and_nested_keys_are_rejected(self):
        for raw in (b'{"a":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]},'
                    b'"a":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}',
                    b'{"a":{"dtype":"BF16","dtype":"BF16","shape":[2],"data_offsets":[0,4]}}',
                    b'{"__metadata__":{"x":"a","x":"b"},'
                    b'"a":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}'):
            with self.subTest(raw=raw):
                write_source(self.path, raw_header=raw)
                with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
                    loader.read_header(self.path)

    def test_shape_dtype_offset_field_and_metadata_errors(self):
        normal = {"a": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}
        malformed = []
        for field, values in {"dtype": ["F32", None, False],
                              "shape": [[], [0], [-2], [True], [2.0], "2"],
                              "data_offsets": [[1, 5], [0, 6], [4, 0], [-1, 3], [False, 4], [0.0, 4], [0], "0,4"]}.items():
            for value in values:
                entry = copy.deepcopy(normal)
                entry["a"][field] = value
                malformed.append(entry)
        malformed += [{"a": {"dtype": "BF16", "shape": [2]}},
                      {"a": {**normal["a"], "extra": "value"}},
                      {" a ": normal["a"]}, {"": normal["a"]},
                      {"__metadata__": {"x": 1}, **normal}, {"__metadata__": [], **normal},
                      {"__metadata__": {}}, {}, []]
        for entries in malformed:
            with self.subTest(entries=entries):
                write_source(self.path, entries)
                with self.assertRaises(ValueError):
                    loader.read_header(self.path)

    def test_overlap_gap_unclaimed_and_truncated_payload_are_rejected(self):
        first = {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}
        cases = [({"a": first, "b": {**first, "data_offsets": [2, 6]}}, bytes(6)),
                 ({"a": first, "b": {**first, "data_offsets": [6, 10]}}, bytes(10)),
                 ({"a": first}, bytes(6)), ({"a": first}, bytes(2))]
        for entries, payload in cases:
            with self.subTest(entries=entries, bytes=len(payload)):
                write_source(self.path, entries, payload=payload)
                with self.assertRaises(ValueError):
                    loader.read_header(self.path)

    def test_invalid_bounded_length_utf8_json_and_numbers(self):
        for prefix in (b"", b"1234567", struct.pack("<Q", 1) + b"{}",
                       struct.pack("<Q", loader.MAX_HEADER_BYTES + 1) + b"{}",
                       struct.pack("<Q", 100) + b"{}"):
            with self.subTest(prefix=prefix):
                self.path.write_bytes(prefix)
                with self.assertRaises(ValueError):
                    loader.read_header(self.path)
        for raw in (b"\xff\xfe", b"not JSON", b"null", b"[1,2]",
                    b'{"a":{"dtype":"BF16","shape":[NaN],"data_offsets":[0,4]}}',
                    b'{"a":{"dtype":"BF16","shape":[Infinity],"data_offsets":[0,4]}}'):
            with self.subTest(raw=raw):
                write_source(self.path, raw_header=raw)
                with self.assertRaises(ValueError):
                    loader.read_header(self.path)

    def test_partial_read_exactness_and_invalid_read_count(self):
        class ShortRead:
            def __init__(self): self.position = 0
            def readinto(self, view):
                if self.position >= 4: return 0
                view[0] = self.position
                self.position += 1
                return 1
        target = bytearray(4)
        loader._read_exact_into(ShortRead(), memoryview(target))
        self.assertEqual(bytes(range(4)), target)
        for count in (0, -1, True, 100):
            with self.subTest(count=count), self.assertRaises(ValueError):
                loader._read_exact_into(SimpleNamespace(readinto=lambda view: count), memoryview(bytearray(4)))


class FakeTensor:
    """Byte views backed only by tiny fixture storage, never a real device."""
    def __init__(self, torch, shape, dtype="bfloat16", device="cuda:0", *, data=None,
                 start=0, end=None, requires_grad=False):
        self.torch, self.shape, self.dtype = torch, tuple(shape), dtype
        self.device = SimpleNamespace(type=device.split(":")[0], index=0 if device == "cuda:0" else None)
        self.requires_grad, self.grad = requires_grad, None
        self._data = bytearray(math.prod(shape) * (2 if dtype == "bfloat16" else 4)) if data is None else data
        self._start, self._end = start, len(self._data) if end is None else end
    def numel(self): return math.prod(self.shape)
    def data_ptr(self): return id(self._data) + self._start
    def is_contiguous(self): return True
    def detach(self): return self
    def reshape(self, *shape):
        return FakeTensor(self.torch, [self.numel()], self.dtype, self.device.type + ":0",
                          data=self._data, start=self._start, end=self._end)
    def __getitem__(self, index):
        width = 2 if self.dtype == "bfloat16" else 4
        stop = min(self.numel(), index.stop)
        return FakeTensor(self.torch, [max(0, stop - index.start)], self.dtype, self.device.type + ":0",
                          data=self._data, start=self._start + width * index.start,
                          end=self._start + width * stop)
    def copy_(self, source, *, non_blocking):
        assert non_blocking is False
        assert source.dtype == self.dtype == "bfloat16"
        assert self._end - self._start == len(source.view)
        self._data[self._start:self._end] = source.view
        self.torch.copy_sizes.append(len(source.view))
        return self
    def contiguous(self): return self
    def view(self, dtype):
        assert dtype == self.torch.uint8
        return self
    def cpu(self):
        count = self._end - self._start
        assert count <= loader.CHUNK_BYTES
        self.torch.cpu_sizes.append(count)
        return SimpleNamespace(numpy=lambda: memoryview(self._data)[self._start:self._end])


class FakeTorch:
    bfloat16, float32, uint8 = "bfloat16", "float32", "uint8"
    def __init__(self):
        self.dtype, self.device_name = self.float32, "cpu"
        self.copy_sizes, self.cpu_sizes, self.frombuffer_sizes = [], [], []
    def get_default_dtype(self): return self.dtype
    def set_default_dtype(self, dtype): self.dtype = dtype
    @contextmanager
    def device(self, name):
        previous, self.device_name = self.device_name, name
        try: yield
        finally: self.device_name = previous
    def no_grad(self): return nullcontext()
    def frombuffer(self, view, *, dtype):
        assert dtype == self.bfloat16 and len(view) <= loader.CHUNK_BYTES
        self.frombuffer_sizes.append(len(view))
        return SimpleNamespace(view=view, dtype=dtype)
    def isfinite(self, tensor): return SimpleNamespace(all=lambda: True)
    def equal(self, left, right):
        return left._data[left._start:left._end] == right._data[right._start:right._end]
    def arange(self, begin, end, step, *, dtype, device):
        assert (begin, end, step, dtype, device) == (0, 128, 2, "float32", "cuda:0")
        return FakeArithmetic(self, list(range(begin, end, step)))


def float32(value): return struct.unpack("<f", struct.pack("<f", value))[0]


class FakeArithmetic:
    def __init__(self, torch, values): self.torch, self.values = torch, values
    def __truediv__(self, denominator):
        return FakeArithmetic(self.torch, [float32(value / denominator) for value in self.values])
    def __rpow__(self, base):
        return FakeArithmetic(self.torch, [float32(base ** value) for value in self.values])
    def __rtruediv__(self, numerator):
        data = bytearray(b"".join(struct.pack("<f", float32(numerator / value)) for value in self.values))
        return FakeTensor(self.torch, [len(self.values)], "float32", "cuda:0", data=data)


def frozen_config():
    return SimpleNamespace(**loader.DIMENSIONS, model_type="qwen3", tie_word_embeddings=True,
                           attention_bias=False, use_sliding_window=False, sliding_window=None,
                           hidden_act="silu", rms_norm_eps=1e-6, attention_dropout=0.0,
                           rope_parameters={"rope_type": "default", "rope_theta": 1000000},
                           layer_types=["full_attention"] * 28, _attn_implementation="eager")


class Fixture:
    def __init__(self, directory):
        self.path, self.torch = Path(directory), FakeTorch()
        self.events, self.guards = [], []
        self.shapes = {"model.embed_tokens.weight": [loader.CHUNK_BYTES // 2 + 7],
                       "model.norm.weight": [3], "model.layers.0.self_attn.q_proj.weight": [4]}
        self.payloads, entries, position = {}, {}, 0
        for number, (name, shape) in enumerate(self.shapes.items()):
            payload = bytes([number + 1]) * (math.prod(shape) * 2)
            entries[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [position, position + len(payload)]}
            position += len(payload)
            self.payloads[name] = payload
        write_source(self.path / "model.safetensors", entries, payload=b"".join(self.payloads.values()))
        self.pins = {}
        for name in loader.ASSET_PINS:
            if name != "model.safetensors": (self.path / name).write_bytes((name + " fixture").encode())
            self.pins[name] = hashlib.sha256((self.path / name).read_bytes()).hexdigest()
        fixture = self

        class Config:
            @classmethod
            def from_pretrained(cls, path, *, local_files_only):
                assert local_files_only is True and Path(path) == fixture.path
                return frozen_config()

        class Rotary:
            def __init__(self, config, device=None):
                self.rope_type, self.attention_scaling = "default", 1.0
                self.max_seq_len_cached = self.original_max_seq_len = 40960
                self._non_persistent_buffers_set = {"inv_freq", "original_inv_freq"}
                data = bytearray(b"".join(struct.pack("<f", float32(1 / float32(1000000 ** (i / 128))))
                                         for i in range(0, 128, 2)))
                self.inv_freq = FakeTensor(fixture.torch, [64], "float32", device or fixture.torch.device_name, data=bytearray(data))
                self.original_inv_freq = FakeTensor(fixture.torch, [64], "float32", device or fixture.torch.device_name, data=bytearray(data))
            def named_buffers(self):
                return [("inv_freq", self.inv_freq), ("original_inv_freq", self.original_inv_freq)]

        class Model:
            def __init__(self, config):
                assert fixture.torch.device_name == "meta" and fixture.torch.dtype == "bfloat16"
                self.values = {name: FakeTensor(fixture.torch, shape, fixture.torch.dtype, "meta", requires_grad=True)
                               for name, shape in fixture.shapes.items()}
                self.model = SimpleNamespace(embed_tokens=SimpleNamespace(weight=self.values["model.embed_tokens.weight"]),
                                             rotary_emb=Rotary(config))
                self.lm_head = SimpleNamespace(weight=self.model.embed_tokens.weight)
                self.training = True
            def named_parameters(self, remove_duplicate=True):
                result = list(self.values.items())
                if not remove_duplicate: result.append(("lm_head.weight", self.lm_head.weight))
                return result
            def parameters(self): return [parameter for _, parameter in self.named_parameters()]
            def to_empty(self, *, device):
                assert device == "cuda:0"
                self.values = {name: FakeTensor(fixture.torch, list(parameter.shape), parameter.dtype, device)
                               for name, parameter in self.values.items()}
                self.model.embed_tokens.weight = self.values["model.embed_tokens.weight"]
                self.lm_head.weight = FakeTensor(fixture.torch, fixture.shapes["model.embed_tokens.weight"])
                return self
            def tie_weights(self): self.lm_head.weight = self.model.embed_tokens.weight
            def eval(self): self.training = False; return self
            def requires_grad_(self, value):
                for parameter in self.parameters(): parameter.requires_grad = value
                return self
            def state_dict(self):
                result = dict(self.values, **{"lm_head.weight": self.lm_head.weight})
                for name, value in self.model.rotary_emb.named_buffers():
                    if name not in self.model.rotary_emb._non_persistent_buffers_set:
                        result["model.rotary_emb." + name] = value
                return result

        self.config_class, self.model_class = Config, Model

    @contextmanager
    def bindings(self):
        with ExitStack() as stack:
            for name, value in {"MODEL_DIR": self.path, "ASSET_PINS": self.pins,
                                "SERIALIZED_TENSORS": len(self.shapes),
                                "BASE_PARAMETERS": sum(map(math.prod, self.shapes.values()))}.items():
                stack.enter_context(patch.object(loader, name, value))
            stack.enter_context(patch.object(loader, "_source_shapes", lambda: copy.deepcopy(self.shapes)))
            yield

    def call(self, operation, function):
        self.events.append((operation, "attempted"))
        try: value = function()
        except BaseException:
            self.events.append((operation, "failed"))
            raise
        self.events.append((operation, "returned"))
        return value

    def guard(self, stage): self.guards.append(stage)

    def load(self):
        return loader.load_titan(self.torch, self.config_class, self.model_class, self.path, self.call, self.guard)


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.fixture = Fixture(self.directory.name)

    def test_fake_stream_exact_hash_tie_dtype_config_rope_and_bounded_copies(self):
        fixture = self.fixture
        with fixture.bindings():
            model, receipt, hashes = fixture.load()
            verified = loader.verify_frozen_base(fixture.torch, model, hashes, fixture.call, fixture.guard)
        self.assertEqual({name: hashlib.sha256(payload).hexdigest() for name, payload in fixture.payloads.items()}, hashes)
        self.assertEqual([loader.CHUNK_BYTES, 14, 6, 8], fixture.torch.copy_sizes)
        self.assertEqual(fixture.torch.copy_sizes, fixture.torch.frombuffer_sizes)
        self.assertLessEqual(max(fixture.torch.cpu_sizes), loader.CHUNK_BYTES)
        self.assertIs(model.lm_head.weight, model.model.embed_tokens.weight)
        self.assertFalse(model.training)
        self.assertEqual("float32", fixture.torch.get_default_dtype())
        self.assertEqual("cpu", fixture.torch.device_name)
        self.assertTrue(verified["frozen_base_unchanged"])
        self.assertEqual(3, verified["verified_source_tensors"])
        self.assertEqual(4, receipt["payload_chunk_count"])
        self.assertFalse(receipt["memory_mapped"])
        self.assertFalse(receipt["cpu_model_allocated"])
        self.assertFalse(receipt["FP32_base_copy_allocated"])
        self.assertTrue(receipt["blocking_chunk_copies"])
        self.assertEqual(28, len(receipt["normalized_config"]["layer_types"]))
        self.assertEqual({"inv_freq", "original_inv_freq"}, set(receipt["rotary_buffers"]))
        self.assertEqual({"model_construct", "checkpoint_load", "tensor_copy"}, {op for op, _ in fixture.events})
        self.assertEqual(3, fixture.events.count(("tensor_copy", "attempted")))
        self.assertEqual(3, fixture.events.count(("tensor_copy", "returned")))
        self.assertIn("tensor_copy_before:model.embed_tokens.weight", fixture.guards)
        self.assertIn("device_digest_after:model.embed_tokens.weight", fixture.guards)

    def test_pins_and_path_fail_before_model_construction(self):
        fixture = self.fixture
        with fixture.bindings():
            with self.assertRaisesRegex(ValueError, "unregistered Titan model directory"):
                loader.load_titan(fixture.torch, fixture.config_class, fixture.model_class,
                                  fixture.path / "wrong", fixture.call, fixture.guard)
            self.assertFalse(fixture.events)
            (fixture.path / "config.json").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "source asset SHA256 drift"):
                fixture.load()
            self.assertFalse(fixture.events)

    def test_source_tensor_metadata_mismatch_prevents_construction(self):
        fixture = self.fixture
        entries = loader.read_header(fixture.path / "model.safetensors")["entries"]
        entries["unexpected"] = entries.pop("model.norm.weight")
        write_source(fixture.path / "model.safetensors", entries, payload=b"".join(fixture.payloads.values()))
        fixture.pins["model.safetensors"] = hashlib.sha256((fixture.path / "model.safetensors").read_bytes()).hexdigest()
        with fixture.bindings(), self.assertRaisesRegex(ValueError, "serialized Titan tensor inventory mismatch"):
            fixture.load()
        self.assertFalse(fixture.events)

    def test_accounted_construct_failure_restores_default_dtype(self):
        fixture = self.fixture
        def raising_model(config): raise RuntimeError("fixture construct failure")
        fixture.model_class = raising_model
        with fixture.bindings(), self.assertRaisesRegex(RuntimeError, "fixture construct failure"):
            fixture.load()
        self.assertEqual("float32", fixture.torch.dtype)
        self.assertEqual("cpu", fixture.torch.device_name)
        self.assertEqual([("model_construct", "attempted"), ("model_construct", "failed")], fixture.events)

    def test_guard_failure_before_copy_has_failed_accounting_and_no_copy(self):
        fixture = self.fixture
        def guard(stage):
            if stage.startswith("tensor_copy_before:"): raise RuntimeError("fixture resource ceiling")
        fixture.guard = guard
        with fixture.bindings(), self.assertRaisesRegex(RuntimeError, "fixture resource ceiling"):
            fixture.load()
        self.assertFalse(fixture.torch.copy_sizes)
        self.assertIn(("tensor_copy", "failed"), fixture.events)
        self.assertIn(("checkpoint_load", "failed"), fixture.events)

    def test_source_mutation_between_preflight_and_copy_is_rejected(self):
        fixture = self.fixture
        original_call = fixture.call
        def call(operation, function):
            if operation == "checkpoint_load":
                with (fixture.path / "model.safetensors").open("r+b") as stream:
                    stream.seek(-1, 2)
                    stream.write(b"\xff")
            return original_call(operation, function)
        fixture.call = call
        with fixture.bindings(), self.assertRaisesRegex(ValueError, "source weights changed after preflight"):
            fixture.load()
        self.assertIn(("checkpoint_load", "failed"), fixture.events)

    def test_device_copy_corruption_is_rejected(self):
        fixture = self.fixture
        original_copy = FakeTensor.copy_
        def corrupt(tensor, source, *, non_blocking):
            original_copy(tensor, source, non_blocking=non_blocking)
            tensor._data[tensor._start] ^= 1
        with fixture.bindings(), patch.object(FakeTensor, "copy_", corrupt), self.assertRaisesRegex(ValueError, "loaded tensor differs"):
            fixture.load()

    def test_frozen_verification_rejects_hash_key_byte_grad_dtype_device_shape_and_tie_drift(self):
        fixture = self.fixture
        mutations = [lambda model: model.values["model.norm.weight"]._data.__setitem__(0, 255),
                     lambda model: setattr(model.values["model.norm.weight"], "requires_grad", True),
                     lambda model: setattr(model.values["model.norm.weight"], "grad", object()),
                     lambda model: setattr(model.values["model.norm.weight"], "dtype", "float32"),
                     lambda model: setattr(model.values["model.norm.weight"].device, "type", "cpu"),
                     lambda model: setattr(model.values["model.norm.weight"].device, "index", 1),
                     lambda model: setattr(model.values["model.norm.weight"], "shape", (4,)),
                     lambda model: setattr(model.lm_head, "weight", FakeTensor(fixture.torch, fixture.shapes["model.embed_tokens.weight"]))]
        with fixture.bindings():
            for mutation in mutations:
                with self.subTest(mutation=mutation):
                    model, _, hashes = fixture.load()
                    mutation(model)
                    with self.assertRaises(ValueError):
                        loader.verify_frozen_base(fixture.torch, model, hashes, fixture.call, fixture.guard)
            model, _, hashes = fixture.load()
            for bad_hashes in ({}, {**hashes, "extra": "a" * 64}, {**hashes, "model.norm.weight": "A" * 64}):
                with self.subTest(hashes=bad_hashes), self.assertRaises(ValueError):
                    loader.verify_frozen_base(fixture.torch, model, bad_hashes, fixture.call, fixture.guard)

    def test_rope_persistence_dtype_device_and_inventory_are_fail_closed(self):
        fixture = self.fixture
        mutations = [lambda rope: rope._non_persistent_buffers_set.remove("inv_freq"),
                     lambda rope: setattr(rope.inv_freq, "dtype", "bfloat16"),
                     lambda rope: setattr(rope.inv_freq.device, "type", "meta"),
                     lambda rope: setattr(rope.original_inv_freq, "shape", (63,)),
                     lambda rope: setattr(rope, "rope_type", "linear"),
                     lambda rope: setattr(rope, "attention_scaling", 2.0),
                     lambda rope: (rope.inv_freq._data.__setitem__(0, 255),
                                   rope.original_inv_freq._data.__setitem__(0, 255))]
        with fixture.bindings():
            for mutation in mutations:
                with self.subTest(mutation=mutation):
                    model, _, _ = fixture.load()
                    mutation(model.model.rotary_emb)
                    with self.assertRaises(ValueError): loader._validate_rope(fixture.torch, model)

    def test_normalized_configuration_rejects_dimensions_types_rope_and_sliding(self):
        mutations = [{field: value + 1} for field, value in loader.DIMENSIONS.items()]
        mutations += [{"hidden_size": True}, {"num_hidden_layers": 28.0}, {"tie_word_embeddings": False},
                      {"attention_bias": True}, {"use_sliding_window": True}, {"sliding_window": 4096},
                      {"layer_types": ["full_attention"] * 27},
                      {"layer_types": ["sliding_attention"] * 28}, {"_attn_implementation": "sdpa"},
                      {"rope_parameters": None}, {"rope_parameters": {"rope_type": "default", "rope_theta": 10000}},
                      {"rope_parameters": {"rope_type": "linear", "rope_theta": 1000000}},
                      {"rope_parameters": {"rope_type": "default", "rope_theta": 1000000, "factor": 2}}]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                config = frozen_config()
                for field, value in mutation.items(): setattr(config, field, value)
                with self.assertRaises(ValueError): loader._validate_config(config)

    def test_exact_adapter_extras_only_and_source_names_preserved(self):
        fixture = self.fixture
        with fixture.bindings():
            model, _, hashes = fixture.load()
            for name, shape in loader._adapter_shapes().items():
                model.values[name] = FakeTensor(fixture.torch, shape, "float32", "cuda:0", requires_grad=True)
            self.assertTrue(loader.verify_frozen_base(fixture.torch, model, hashes, fixture.call, fixture.guard)["frozen_base_unchanged"])
            model.values.pop(next(iter(loader._adapter_shapes())))
            with self.assertRaisesRegex(ValueError, "parameter key mismatch"):
                loader.verify_frozen_base(fixture.torch, model, hashes, fixture.call, fixture.guard)

    def test_published_static_shapes_and_pilot_pins_match_implementation(self):
        inventory = json.loads((ROOT / "experiments/post_training_mimo_centralized_v1/receipts/titan_static_v1/PARAMETER_INVENTORY.json").read_text())
        decisions = json.loads((PACKAGE / "PILOT_DECISIONS.json").read_text())
        self.assertEqual({name: entry["shape"] for name, entry in inventory["tensors"].items()}, loader._source_shapes())
        self.assertEqual(310, len(loader._source_shapes()))
        self.assertEqual(596049920, sum(map(math.prod, loader._source_shapes().values())))
        self.assertEqual(inventory["checkpoint_path"].replace("\\", "/"), loader.MODEL_DIR.as_posix())
        for filename, decision_field in [("model.safetensors", "weight_sha256"), ("config.json", "config_sha256"),
                                         ("tokenizer.json", "tokenizer_sha256"), ("tokenizer_config.json", "tokenizer_config_sha256")]:
            self.assertEqual(decisions["student"][decision_field], loader.ASSET_PINS[filename])
        expected = json.loads((PACKAGE / "pilot_trainable_inventory.json").read_text())
        self.assertEqual({entry["name"]: entry["shape"] for entry in expected["tensors"]}, loader._adapter_shapes())

    def test_module_import_is_framework_and_file_read_free(self):
        code = """import sys
sys.path.insert(0, sys.argv[1])
from pathlib import Path
def fail(*args, **kwargs): raise AssertionError('unexpected asset read on import')
Path.open = fail
Path.read_bytes = fail
Path.read_text = fail
import native_loader
assert 'torch' not in sys.modules and 'transformers' not in sys.modules
print('FRAMEWORK_AND_ASSET_FREE_IMPORT')
"""
        result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(PACKAGE)],
                                capture_output=True, text=True, timeout=30, check=True)
        self.assertEqual("FRAMEWORK_AND_ASSET_FREE_IMPORT", result.stdout.strip())


if __name__ == "__main__":
    unittest.main()
