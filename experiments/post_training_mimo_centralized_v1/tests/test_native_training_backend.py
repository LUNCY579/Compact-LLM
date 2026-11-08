"""Fake tensor/optimizer dispatch controls; no framework or tensor-physics proof."""
from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import sys

from experiments.post_training_mimo_centralized_v1.scripts import native_training_backend as b
from experiments.post_training_mimo_centralized_v1.scripts import native_backend_contract as c
from experiments.post_training_mimo_centralized_v1.scripts.e1_training_accounting import E1CallAccounting


class Tensor:
    def __init__(self, shape, dtype="float32", device="cuda:0", *, value=0.0, grad=False, data=None):
        self.shape, self.dtype, self.device = tuple(shape), dtype, device
        self.value, self.requires_grad, self.grad, self.data = value, grad, None, data
        self.on_backward = None
    def numel(self): return math.prod(self.shape)
    def detach(self):
        copied = self.clone(); copied.requires_grad = False; return copied
    def cpu(self): copied = self.clone(); copied.device = "cpu"; return copied
    def clone(self):
        copied = Tensor(self.shape, self.dtype, self.device, value=self.value, grad=self.requires_grad, data=deepcopy(self.data))
        copied.on_backward = self.on_backward; return copied
    def item(self): return self.value
    def copy_(self, other): self.value, self.data = other.value, deepcopy(other.data); return self
    def __mul__(self, scale):
        copied = self.clone(); copied.value *= scale
        copied.on_backward = lambda: self.on_backward(scale)
        return copied
    def backward(self): self.on_backward()


def tensor_hash(torch, tensor):
    return hashlib.sha256(c.canonical({"shape": list(tensor.shape), "dtype": tensor.dtype,
                                      "value": tensor.value, "data": tensor.data})).hexdigest()


class Model:
    def __init__(self):
        self.config = SimpleNamespace(_attn_implementation="eager")
        self.is_gradient_checkpointing = False; self.training = True; self.calls = []
        self.base = {}
        for name, shape in b.base_shapes().items():
            wrapped = name.replace(".q_proj.weight", ".q_proj.base.weight").replace(".v_proj.weight", ".v_proj.base.weight")
            self.base[wrapped] = Tensor(shape, "bfloat16", value=1.0)
        self.adapters = {item["name"]: Tensor(item["shape"], grad=True) for item in c.expected_adapter_inventory()}
        self.model = SimpleNamespace(embed_tokens=SimpleNamespace(weight=self.base["model.embed_tokens.weight"]))
        self.lm_head = SimpleNamespace(weight=self.model.embed_tokens.weight)
        self.sites = {}
        for item in c.expected_adapter_inventory()[::2]:
            name = item["name"].rsplit(".", 1)[0]
            self.sites[name] = SimpleNamespace(lora_A=self.adapters[name + ".lora_A"], lora_B=self.adapters[name + ".lora_B"],
                                              base=SimpleNamespace(bias=None))
    def eval(self): self.training = False; return self
    def named_parameters(self): return list(self.base.items()) + list(self.adapters.items())
    def named_modules(self): return list(self.sites.items())
    def __call__(self, ids, **kwargs):
        self.calls.append((ids, kwargs))
        return SimpleNamespace(logits=Tensor([1, ids.shape[1], 151936], "bfloat16"))


class Optimizer:
    def __init__(self, parameters, **kwargs):
        self.parameters = list(parameters); self.state = {}; self.steps = 0
        self.param_groups = [dict(kwargs, params=self.parameters, amsgrad=False, maximize=False, decoupled_weight_decay=True)]
    def zero_grad(self, *, set_to_none):
        assert set_to_none
        for p in self.parameters: p.grad = None
    def step(self):
        self.steps += 1
        for p in self.parameters:
            prior = self.state.get(p)
            self.state[p] = {"step": Tensor([], device="cpu", value=float(self.steps)),
                "exp_avg": Tensor(p.shape, value=(prior["exp_avg"].value if prior else 0) + 1),
                "exp_avg_sq": Tensor(p.shape, value=(prior["exp_avg_sq"].value if prior else 0) + 1)}
            p.value -= self.param_groups[0]["lr"]
    def state_dict(self):
        return {"state": {index: deepcopy(self.state[p]) for index, p in enumerate(self.parameters) if p in self.state},
                "param_groups": [dict(self.param_groups[0], params=list(range(112)))]}
    def load_state_dict(self, state):
        self.state = {}
        for index, values in state["state"].items():
            self.state[self.parameters[index]] = deepcopy(values)
            for key in ("exp_avg", "exp_avg_sq"): self.state[self.parameters[index]][key].device = "cuda:0"
        self.steps = int(next(iter(state["state"].values()))["step"].value) if state["state"] else 0
        self.param_groups[0] = dict(state["param_groups"][0], params=self.parameters)


class Torch:
    long = "long"
    def __init__(self, model):
        self.model = model; self.saved = []; self.loaded = []; self.restored_rng = []; self.scales = []; self.clips = []
        self.cpu_rng = Tensor([8], "uint8", "cpu", data=list(range(8)))
        self.cuda_rng = Tensor([4], "uint8", "cpu", data=list(range(4)))
        self.cuda = SimpleNamespace(device_count=lambda: 1, get_rng_state=lambda device: self.cuda_rng.clone(),
                                    set_rng_state=lambda value, device: self.restored_rng.append((device, value.clone())))
        self.optim = SimpleNamespace(AdamW=Optimizer)
        self.nn = SimpleNamespace(utils=SimpleNamespace(clip_grad_norm_=self.clip))
    def is_tensor(self, value): return isinstance(value, Tensor)
    def isfinite(self, value): return SimpleNamespace(all=lambda: math.isfinite(value.value))
    def is_grad_enabled(self): return True
    def tensor(self, value, *, dtype, device): return Tensor([len(value), len(value[0])], dtype, device, data=deepcopy(value))
    def get_rng_state(self): return self.cpu_rng.clone()
    def set_rng_state(self, value): self.restored_rng.append(("cpu", value.clone()))
    def no_grad(self):
        class Context:
            def __enter__(self): pass
            def __exit__(self, *args): pass
        return Context()
    def clip(self, parameters, max_norm, *, error_if_nonfinite):
        self.clips.append((max_norm, error_if_nonfinite)); return Tensor([], value=0.5)
    def completion_loss(self, torch, logits, ids, labels):
        loss = Tensor([], value=2.0, grad=True)
        def backward(scale):
            self.scales.append(scale)
            for p in self.model.adapters.values(): p.grad = Tensor(p.shape, value=scale)
        loss.on_backward = backward
        return loss
    def encode(self, value):
        if isinstance(value, Tensor): return {"__tensor__": [list(value.shape), value.dtype, value.device, value.value, value.requires_grad, value.data]}
        if type(value) is dict: return {"__dict__": [[self.encode(k), self.encode(v)] for k, v in value.items()]}
        if type(value) is tuple: return {"__tuple__": [self.encode(x) for x in value]}
        if type(value) is list: return [self.encode(x) for x in value]
        return value
    def decode(self, value):
        if type(value) is list: return [self.decode(x) for x in value]
        if type(value) is dict:
            if "__tensor__" in value:
                shape, dtype, device, number, grad, data = value["__tensor__"]
                return Tensor(shape, dtype, device, value=number, grad=grad, data=data)
            if "__dict__" in value: return {self.decode(k): self.decode(v) for k, v in value["__dict__"]}
            if "__tuple__" in value: return tuple(self.decode(x) for x in value["__tuple__"])
        return value
    def save(self, payload, stream):
        self.saved.append(deepcopy(payload)); stream.write(json.dumps(self.encode(payload)).encode())
    def load(self, stream, *, map_location, weights_only):
        assert type(stream) is io.BytesIO and map_location == "cpu" and weights_only is True
        self.loaded.append((map_location, weights_only))
        return self.decode(json.loads(stream.read()))


def row():
    ids = [10, 11, 12, 13, 151645]
    return {"row_id": "bridge-fake", "parent_id": "fake-parent", "family": "bounded_integer_arithmetic", "role": "TRAIN",
            "component": "acquisition",
            "training_role_eligible": True, "training_eligibility": False, "prompt_tokens": 3, "completion_tokens": 2,
            "nonpadding_tokens": 5, "padding_tokens": 0, "completion_eos_id": 151645, "input_ids": ids,
            "labels": [-100, -100, -100, 13, 151645], "attention_mask": [1] * 5, "completion_mask": [0, 0, 0, 1, 1],
            "input_token_ids_sha256": c.digest(ids), "tokenized_prompt_sha256": c.digest(ids[:3]),
            "completion_token_ids_sha256": c.digest(ids[3:]), "source_row_sha256": "a" * 64}


class BackendControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.run = "artifacts/mimo_central/fake_backend"
        (self.root / self.run).mkdir(parents=True)
        self.model = Model(); self.torch = Torch(self.model); self.guards = []; self.events = []
        self.helpers = SimpleNamespace(install_lora=lambda *args: None, completion_loss=self.torch.completion_loss, tensor_digest=tensor_hash)
        self.bindings = {"code_sha256": "c" * 64, "data_sha256": c.DATA_SHA256, "plan_sha256": c.PLAN_SHA256,
                         "weight_sha256": c.WEIGHT_SHA256, "inventory_sha256": c.digest(c.expected_adapter_inventory())}
        self.hashes = {c.canonical_base_name(name): tensor_hash(self.torch, p) for name, p in self.model.base.items()}
    def backend(self, **overrides):
        options = dict(helpers=self.helpers, guard=self.guards.append,
            emit=lambda kind, **fields: self.events.append((kind, fields)),
            repository_root=self.root, approved_run_relative=self.run, bindings=self.bindings,
            source_base_hashes=self.hashes, expected_rng_shapes={"torch_cpu": [8], "cuda:0": [4]})
        options.update(overrides)
        return b.NativeTrainingBackend.create(self.torch, self.model, **options)
    def boundary0(self, backend):
        ledger = E1CallAccounting(lambda *args, **kwargs: None)
        cursor = {"completed_optimizer_steps": 0, "plan_sha256": c.PLAN_SHA256, "accumulated_sequences": 0,
                  "identities": {"data_sha256": c.DATA_SHA256, "parent_sha256": c.WEIGHT_SHA256}}
        pre_save = ledger.snapshot()
        state = ledger.call("checkpoint_save", lambda: backend.save_checkpoint(cursor, pre_save), stage="save0")
        envelope = {"schema_version": 1, "kind": "E1_CONTROLLED_BOUNDARY_COMPONENT_CHECKPOINT_V2", "plan_sha256": c.PLAN_SHA256,
                    "cursor": cursor, "cost_snapshot": ledger.snapshot(), "backend_state": state,
                    "save_call_id": ledger.last_call_id, "native_resume_qualified": False, "E1_authorized": False}
        return state, envelope

    def test_constructor_exact_virtual_inventory_and_no_framework(self):
        backend = self.backend()
        self.assertFalse(self.model.training)
        self.assertEqual(len(backend.base), 310); self.assertEqual(len(backend.adapters), 112)
        self.assertEqual(sum(p.numel() for p in backend.base.values()), 596049920)
        self.assertEqual(backend.execution_scope, "UNQUALIFIED_NATIVE_COMPONENT")
        self.assertFalse(set(("torch", "transformers", "safetensors", "tokenizers")).intersection(sys.modules))

    def test_constructor_rejects_hash_alias_shape_dtype_and_checkpointing(self):
        cases = [lambda: self.hashes.update({"model.norm.weight": "0" * 64}),
                 lambda: setattr(self.model, "is_gradient_checkpointing", True),
                 lambda: setattr(self.model.config, "_attn_implementation", "sdpa"),
                 lambda: setattr(next(iter(self.model.adapters.values())), "dtype", "bfloat16"),
                 lambda: setattr(self.model.lm_head, "weight", Tensor([151936, 1024], "bfloat16"))]
        for mutation in cases:
            self.setUp(); mutation()
            with self.assertRaises(ValueError): self.backend()

    def test_full20_dispatch_scales_clip_once_and_step(self):
        backend = self.backend(); backend.reset_gradients()
        for _ in range(20):
            loss = backend.forward_loss(row()); self.assertEqual(backend.loss_value(loss), 2.0)
            backend.backward(loss, scale=1 / 20)
        self.assertEqual(backend.clip_gradients(max_norm=1.0), 0.5)
        backend.optimizer_step(lr=1e-4 / 13)
        self.assertEqual(self.torch.scales, [1 / 20] * 20)
        self.assertEqual(self.torch.clips, [(1.0, True)])
        self.assertEqual(backend.completed_updates, 1)
        self.assertEqual(self.model.calls[0][1], {"use_cache": False, "logits_to_keep": 0})

    def test_partial_scale_clip_step_base_leak_and_nonfinite_rejected(self):
        backend = self.backend(); backend.reset_gradients(); loss = backend.forward_loss(row())
        with self.assertRaises(ValueError): backend.backward(loss, scale=1.0)
        with self.assertRaises(ValueError): backend.clip_gradients(max_norm=1.0)
        with self.assertRaises(ValueError): backend.optimizer_step(lr=1e-4 / 13)
        next(iter(self.model.base.values())).grad = Tensor([1])
        with self.assertRaises(ValueError): backend.reset_gradients()

    def test_bad_row_or_eval_change_rejected_before_model(self):
        backend = self.backend(); backend.reset_gradients(); bad = row(); bad["labels"][0] = 10
        with self.assertRaises(ValueError): backend.forward_loss(bad)
        self.assertEqual(self.model.calls, [])
        self.model.training = True
        with self.assertRaises(ValueError): backend.forward_loss(row())

    def test_save_only_cpu_payload_refs_and_published_same_bytes_restore(self):
        backend = self.backend(); state, envelope = self.boundary0(backend)
        payload = self.torch.saved[-1]
        self.assertEqual(set(payload["parameters"]), set(self.model.adapters))
        self.assertTrue(all(p.device == "cpu" and not p.requires_grad for p in payload["parameters"].values()))
        self.assertNotIn("base", payload); self.assertNotIn("gradients", payload)
        self.assertIsInstance(state["parameters"]["artifact"]["sha256"], str)
        with self.assertRaises(ValueError): backend.restore_checkpoint(state)
        published = backend.publish_checkpoint(envelope)
        self.assertIs(published["E1_authorized"], False)
        next(iter(self.model.adapters.values())).value = 17
        backend.restore_checkpoint(state)
        self.assertEqual(next(iter(self.model.adapters.values())).value, 0)
        self.assertEqual(self.torch.loaded, [("cpu", True)])
        self.assertEqual([entry[0] for entry in self.torch.restored_rng], ["cpu", 0])

    def test_write_once_publication_and_mutated_bytes_refused(self):
        backend = self.backend(); state, envelope = self.boundary0(backend)
        backend.publish_checkpoint(envelope)
        with self.assertRaises(FileExistsError): backend.publish_checkpoint(envelope)
        path = self.root / state["parameters"]["artifact"]["relative_path"]
        path.write_bytes(path.read_bytes() + b"changed")
        with self.assertRaises(ValueError): backend.restore_checkpoint(state)
        self.assertEqual(self.torch.loaded, [])

    def test_publication_cost_or_source_change_refused(self):
        backend = self.backend(); state, envelope = self.boundary0(backend)
        envelope["save_call_id"] = 99
        with self.assertRaises(ValueError): backend.publish_checkpoint(envelope)
        self.assertEqual(backend._published, {})
        self.bindings["code_sha256"] = "a" * 40
        with self.assertRaises(ValueError): self.backend()

    def test_duplicate_phase_calls_and_partial_save_rejected(self):
        backend = self.backend()
        with self.assertRaises(ValueError): backend.forward_loss(row())
        backend.reset_gradients()
        with self.assertRaises(ValueError): backend.reset_gradients()
        loss = backend.forward_loss(row())
        with self.assertRaises(ValueError): backend.forward_loss(row())
        with self.assertRaises(ValueError): self.boundary0(backend)
        backend.backward(loss, scale=1 / 20)
        with self.assertRaises(ValueError): backend.backward(loss, scale=1 / 20)
        self.assertEqual(self.torch.scales, [1 / 20])
        self.assertEqual(self.torch.saved, [])

    def test_explicit_component_missing_and_invalid_refused(self):
        backend = self.backend(); backend.reset_gradients()
        for value in (None, "other", True):
            bad = row()
            if value is None: bad.pop("component")
            else: bad["component"] = value
            with self.assertRaises(ValueError): backend.forward_loss(bad)
        self.assertEqual(self.model.calls, [])

    def test_final_lr_zero_advances_adam_state_without_parameter_change(self):
        backend = self.backend(); backend.completed_updates = 255
        backend.optimizer.steps = 255
        backend.optimizer.param_groups[0]["lr"] = 1e-4 / 243
        backend.optimizer.state = {p: {"step": Tensor([], device="cpu", value=255.),
            "exp_avg": Tensor(p.shape, value=1.), "exp_avg_sq": Tensor(p.shape, value=1.)} for p in backend.adapters.values()}
        backend.reset_gradients()
        for _ in range(20): backend.backward(backend.forward_loss(row()), scale=1 / 20)
        backend.clip_gradients(max_norm=1.0)
        before = [p.value for p in backend.adapters.values()]
        backend.optimizer_step(lr=0.)
        self.assertEqual([p.value for p in backend.adapters.values()], before)
        self.assertTrue(all(state["step"].value == 256 and state["exp_avg"].value == 2 for state in backend.optimizer.state.values()))

    def test_partial_serialization_failure_retains_file_without_reference(self):
        backend = self.backend()
        def fail(payload, stream): stream.write(b"partial"); raise OSError("synthetic disk failure")
        with patch.object(self.torch, "save", fail):
            with self.assertRaises(OSError): self.boundary0(backend)
        self.assertEqual((self.root / self.run / "adapter_000.pt").read_bytes(), b"partial")
        self.assertEqual(backend._saved, {}); self.assertEqual(backend._published, {})

    def test_publication_wrong_version_kind_and_boolean_cursor_refused(self):
        backend = self.backend(); state, original = self.boundary0(backend)
        for field, value in (("schema_version", True), ("kind", "E1_CONTROLLED_BOUNDARY_COMPONENT_CHECKPOINT")):
            changed = deepcopy(original); changed[field] = value
            with self.assertRaises(ValueError): backend.publish_checkpoint(changed)
        changed = deepcopy(original["cursor"]); changed["completed_optimizer_steps"] = False
        with self.assertRaises(ValueError): backend.save_checkpoint(changed, E1CallAccounting(lambda *args, **kwargs: None).snapshot())

    def test_physical_payload_shape_rejected_before_parameter_mutation(self):
        backend = self.backend(); state, envelope = self.boundary0(backend); backend.publish_checkpoint(envelope)
        first = next(iter(backend.adapters.values())); first.value = 17.
        original_load = self.torch.load
        def corrupted(stream, **kwargs):
            payload = original_load(stream, **kwargs)
            payload["parameters"][next(iter(payload["parameters"]))].shape = (1,)
            return payload
        with patch.object(self.torch, "load", corrupted):
            with self.assertRaises(ValueError): backend.restore_checkpoint(state)
        self.assertEqual(first.value, 17.)

    def test_foreign_optimizer_group_and_oversize_manifest_refused_early(self):
        backend = self.backend(); backend.optimizer.param_groups[0]["foreign_base"] = self.model.model.embed_tokens.weight
        with self.assertRaises(ValueError): backend._optimizer_metadata()
        backend.optimizer.param_groups[0].pop("foreign_base")
        state, envelope = self.boundary0(backend); publication = backend.publish_checkpoint(envelope)
        (self.root / publication["relative_path"]).write_bytes(b"X" * (1024 * 1024 + 1))
        with self.assertRaises(ValueError): backend.restore_checkpoint(state)
        self.assertEqual(self.torch.loaded, [])

    def prepare_update(self, backend):
        backend.reset_gradients()
        for _ in range(20): backend.backward(backend.forward_loss(row()), scale=1 / 20)
        backend.clip_gradients(max_norm=1.0)

    def test_primitive_optimizer_return_survives_later_guard_or_finite_failure(self):
        for mode in ("guard", "finite"):
            self.setUp(); backend = self.backend(); self.prepare_update(backend)
            if mode == "guard":
                def guard(stage):
                    if stage == "after_optimizer_step": raise OSError("post-return guard failure")
                backend.guard = guard
            else:
                original = backend.optimizer.step
                def nonfinite(): original(); next(iter(backend.adapters.values())).value = float("nan")
                backend.optimizer.step = nonfinite
            ledger = E1CallAccounting(lambda *args, **kwargs: None)
            with self.assertRaises((OSError, ValueError)):
                ledger.call("optimizer_step", lambda: backend.optimizer_step(lr=1e-4 / 13), stage="step1")
            self.assertEqual(ledger.snapshot()["operations"]["optimizer_step"]["failed"], 1)
            self.assertEqual(backend.primitive_operation_snapshot()["operations"]["optimizer_step"],
                             {"attempted": 1, "returned": 1, "failed": 0, "pending": 0})
            self.assertEqual(backend.completed_updates, 1)

    def test_primitive_return_event_io_failure_does_not_reclassify_torch_work(self):
        backend = self.backend(); self.prepare_update(backend)
        def emit(kind, **fields):
            if kind == "primitive_operation_returned" and fields["operation"] == "optimizer_step": raise OSError("journal disk failure")
        backend.emit = emit
        with self.assertRaises(OSError): backend.optimizer_step(lr=1e-4 / 13)
        self.assertEqual(backend.optimizer.steps, 1)
        self.assertEqual(backend.primitive_operation_snapshot()["operations"]["optimizer_step"]["returned"], 1)
        self.assertEqual(backend.primitive_operation_snapshot()["operations"]["optimizer_step"]["failed"], 0)

    def test_primitive_attempt_io_failure_prevents_execution_and_remains_pending(self):
        backend = self.backend(); backend.reset_gradients()
        def emit(kind, **fields): raise OSError("journal unavailable")
        backend.emit = emit
        with self.assertRaises(OSError): backend.forward_loss(row())
        self.assertEqual(self.model.calls, [])
        self.assertEqual(backend.primitive_operation_snapshot()["operations"]["forward"],
                         {"attempted": 1, "returned": 0, "failed": 0, "pending": 1})

    def test_primitive_costs_not_reset_by_controlled_restore(self):
        backend = self.backend(); state, envelope = self.boundary0(backend); backend.publish_checkpoint(envelope)
        before = backend.primitive_operation_snapshot()
        backend.restore_checkpoint(state)
        after = backend.primitive_operation_snapshot()
        self.assertGreater(after["last_call_id"], before["last_call_id"])
        self.assertEqual(after["operations"]["checkpoint_save"]["returned"], 1)
        self.assertEqual(after["operations"]["checkpoint_load"]["returned"], 1)
        self.assertEqual(after["operations"]["optimizer_construct"]["returned"], 1)
        for flag in ("native_qualified", "native_launch_authorized", "E1_authorized"): self.assertIs(after[flag], False)

    def test_factory_exposes_constructor_primitive_return_after_event_io_failure(self):
        original = OSError("constructor return journal unavailable")
        def emit(kind, **fields):
            if kind == "primitive_operation_returned" and fields["operation"] == "optimizer_construct": raise original
        with self.assertRaises(OSError) as observed: self.backend(emit=emit)
        self.assertIs(observed.exception, original)
        snapshot = observed.exception.native_backend_primitive_accounting
        self.assertEqual(snapshot["operations"]["optimizer_construct"], {"attempted": 1, "returned": 1, "failed": 0, "pending": 0})
        self.assertIs(observed.exception.native_backend_constructor_failed, True)

    def test_factory_preserves_final_constructor_guard_error_and_cost_snapshot(self):
        original = RuntimeError("post-construction resource failure")
        def guard(stage):
            if stage == "backend_constructed_unqualified": raise original
        with self.assertRaises(RuntimeError) as observed: self.backend(guard=guard)
        self.assertIs(observed.exception, original)
        self.assertEqual(observed.exception.native_backend_primitive_accounting["operations"]["optimizer_construct"]["returned"], 1)
        self.assertEqual(observed.exception.native_backend_primitive_accounting["pending_calls"], [])


if __name__ == "__main__": unittest.main()
