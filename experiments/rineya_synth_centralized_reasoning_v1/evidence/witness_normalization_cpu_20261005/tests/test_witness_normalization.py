"""Stdlib persisted-A/fresh-B witness controls; no frameworks or native restore."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
STAGE = Path(__file__).resolve().parents[1]
PACKAGE = STAGE.parents[1]
sys.path.insert(0, str(PACKAGE))
sys.path.insert(0, str(PACKAGE / "tests"))
import backend_qual_contract as contract
import backend_qual_core as core
from test_backend_qualification import FakeAdapter, Sink
from test_study_native_chunk import Journal


def load_adapter(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


STAGED = load_adapter("staged_witness_adapter", STAGE / "backend_qual_native.py")
ORIGINAL = load_adapter("original_witness_adapter", STAGE / "original_backend_qual_native.py")


class NativeWitnessFakeAdapter(FakeAdapter):
    """Scalar fake controller with the actual injected native witness formatter."""
    module = STAGED
    change_witness = None

    def __init__(self, *args):
        super().__init__(*args)
        self.backend.source_hashes = {"CPU_FAKE_BASE_NOT_NATIVE_TENSOR": "0" * 64}
        self.backend._binding = self.current_binding
        # Bypass the constructor: no Torch object, model, CUDA context or restore.
        self.native = self.module.NativeAdapter.__new__(self.module.NativeAdapter)
        self.native.backend = self.backend
        self.native.entries = [("CPU_FAKE.lora_A", None)]
        self.native._hash_tensor = self.fake_hash

    @staticmethod
    def fake_hash(value):
        raw = contract.canonical(value).encode("ascii")
        return {"shape": [], "dtype": "CPU_FAKE_SCALAR_NOT_NATIVE_TENSOR",
                "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}

    def current_binding(self, supplied):
        del supplied
        backend = self.backend
        return {"completed_updates": backend.completed_updates,
                "consumed_occurrences": len(backend.order),
                "input_tokens": backend.input_tokens,
                "supervised_tokens": backend.supervised_tokens,
                "consumed_order_sha256": contract.digest(backend.order),
                "next_update_ordinal": backend.completed_updates + 1,
                "learning_rate": backend.lr}

    def raw_state(self):
        backend = self.backend
        return {"adapters": {"CPU_FAKE.lora_A": backend.weight},
                "optimizer": {"state": {0: {"step": backend.completed_updates,
                                                "exp_avg": backend.m, "exp_avg_sq": backend.v}},
                              "param_groups": [{"lr": backend.lr, "betas": (.9, .999),
                                                "eps": 1e-8, "weight_decay": 0,
                                                "amsgrad": False, "maximize": False,
                                                "foreach": False, "capturable": False,
                                                "differentiable": False, "fused": False,
                                                "params": [0]}]},
                "rng": {"python": backend.state()["rng"],
                        "cpu": "CPU_FAKE_NO_GLOBAL_CPU_RNG",
                        "cuda0": "CPU_FAKE_NO_CUDA_RNG"}}

    def witness(self, binding):
        self.check()
        state = self.raw_state()
        original = deepcopy(state)
        witness = self.native._state_witness(state, binding)
        if state != original or type(state["optimizer"]["param_groups"][0]["betas"]) is not tuple:
            raise AssertionError("witness formatting mutated the raw optimizer state")
        if self.scope["phase"] == "B" and self.change_witness is not None:
            self.change_witness(witness)
            witness["sha256"] = self.module.digest({k: v for k, v in witness.items() if k != "sha256"})
        return witness


class WitnessNormalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scope_a = contract.build_scope("A")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)

    def execute(self, scope, name, adapter_type=NativeWitnessFakeAdapter):
        directory = self.path / name
        directory.mkdir()
        sink = Sink(directory)
        result = core.run_case(scope, adapter_type, Journal(), lambda stage: None,
                               sink, cpu_control=True)
        return result, sink

    def predecessor(self, result, sink):
        path = sink.directory / "worker_result.json"
        path.write_text(contract.canonical(result), encoding="ascii")
        return {"result_path": str(path), "result_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "supervisor_path": str(sink.directory / "CPU_fake_unused.json"),
                "supervisor_sha256": "0" * 64}

    def fresh_scope(self, result, sink):
        return contract.build_scope("B", predecessor=self.predecessor(result, sink))

    def assert_stopped_at_c9(self, result):
        self.assertEqual("FAILED_SCOPED_BACKEND_QUALIFICATION", result["status"])
        self.assertEqual("fresh-process C9 state must match independently detached A witness bitwise",
                         result["error"]["message"])
        self.assertEqual(2, result["dose"]["attempted_calls"])
        self.assertEqual([], result["update_summaries"])
        for name in ("forward", "backward", "step", "reference_control"):
            self.assertEqual(0, result["protocol_operations"][name]["attempted"])
        self.assertFalse(result["native_execution_performed"])

    def test_original_tuple_list_failure_reproduced_before_reference_or_training(self):
        class OriginalAdapter(NativeWitnessFakeAdapter):
            module = ORIGINAL
        a, sink = self.execute(self.scope_a, "original_A", OriginalAdapter)
        self.assertEqual("PASS_CPU_FAKE_BACKEND_A9_CONT10_CONTROL", a["status"], a.get("error"))
        self.assertIs(type(a["witness9"]["optimizer_groups"][0]["betas"]), tuple)
        b, b_sink = self.execute(self.fresh_scope(a, sink), "original_B", OriginalAdapter)
        self.assert_stopped_at_c9(b)
        self.assertEqual(json.loads((sink.directory / "detached_c9_witness.json").read_text()),
                         json.loads((b_sink.directory / "restored_c9.json").read_text()))

    def test_staged_persisted_A_to_fresh_B_succeeds_with_raw_tuples_unchanged(self):
        a, sink = self.execute(self.scope_a, "staged_A")
        self.assertEqual("PASS_CPU_FAKE_BACKEND_A9_CONT10_CONTROL", a["status"], a.get("error"))
        for key in ("witness9", "witness10"):
            self.assertIs(type(a[key]["optimizer_groups"][0]["betas"]), list)
            self.assertEqual(a[key], json.loads(contract.canonical(a[key])))
        b, _ = self.execute(self.fresh_scope(a, sink), "staged_B")
        self.assertEqual("PASS_CPU_FAKE_BACKEND_FRESH_CONT10_CONTROL", b["status"], b.get("error"))
        self.assertEqual(a["witness10"], b["witness10"])
        self.assertEqual(42, b["dose"]["attempted_calls"])
        self.assertFalse(b["native_execution_performed"])
        self.assertFalse(b["study_warmstart_allowed"])

    def test_altered_beta_tensor_hash_rng_and_cursor_are_rejected(self):
        a, sink = self.execute(self.scope_a, "controls_A")
        self.assertEqual("PASS_CPU_FAKE_BACKEND_A9_CONT10_CONTROL", a["status"], a.get("error"))
        scope = self.fresh_scope(a, sink)
        changes = {
            "beta": lambda w: w["optimizer_groups"][0]["betas"].__setitem__(0, .8),
            "tensor_hash": lambda w: w["adapters"]["CPU_FAKE.lora_A"].update(sha256="1" * 64),
            "rng": lambda w: w["rng"].update(python_sha256="2" * 64),
            "cursor": lambda w: w["binding"].update(input_tokens=w["binding"]["input_tokens"] + 1),
        }
        for name, change in changes.items():
            with self.subTest(name=name):
                class ChangedAdapter(NativeWitnessFakeAdapter):
                    change_witness = staticmethod(change)
                result, _ = self.execute(scope, "altered_" + name, ChangedAdapter)
                self.assert_stopped_at_c9(result)

    def test_only_betas_representation_changes_and_serialized_output_is_identical(self):
        a, _ = self.execute(self.scope_a, "serialize_A")
        self.assertEqual("PASS_CPU_FAKE_BACKEND_A9_CONT10_CONTROL", a["status"], a.get("error"))
        for key in ("witness9", "witness10"):
            witness = a[key]
            old = deepcopy(witness)
            old["optimizer_groups"][0]["betas"] = tuple(old["optimizer_groups"][0]["betas"])
            self.assertNotEqual(old, witness)
            self.assertEqual(ORIGINAL.digest(old), STAGED.digest(witness))
            self.assertEqual(contract.canonical(old).encode("ascii"), contract.canonical(witness).encode("ascii"))

    def test_no_framework_modules_loaded(self):
        forbidden = {"torch", "transformers", "jax", "tensorflow", "flax"}
        self.assertFalse({name.split(".", 1)[0] for name in sys.modules} & forbidden)


if __name__ == "__main__":
    unittest.main(verbosity=2)
