"""Pure synthetic metadata/byte controls; no framework or native checkpoint."""
from copy import deepcopy
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


from experiments.post_training_mimo_centralized_v1.scripts import native_backend_contract as c


def model_case():
    adapters = c.expected_adapter_inventory()
    sources = {name: hashlib.sha256(name.encode()).hexdigest() for name in c.expected_base_names()}
    names = [name.replace(".q_proj.weight", ".q_proj.base.weight").replace(".v_proj.weight", ".v_proj.base.weight")
             for name in c.expected_base_names()]
    base = {"unique_parameters": 596049920, "dtype": "bfloat16", "device": "cuda:0", "requires_grad": False,
            "weight_sha256": c.WEIGHT_SHA256, "embedding_tied": True, "serialized_tensor_count": 310,
            "parameter_names": names, "source_tensor_sha256": deepcopy(sources),
            "tied_alias": {"alias": "lm_head.weight", "target": "model.embed_tokens.weight", "same_object": True,
                           "serialized_separately": False}}
    return adapters, base, sources


def tensor(shape, device, *, step=None):
    value = {"shape": list(shape), "dtype": "float32", "device": device, "numel": math.prod(shape),
             "requires_grad": False, "finite": True}
    if step is not None:
        value["value"] = float(step)
    return value


def adam(updates=64):
    adapters = c.expected_adapter_inventory()
    lr = 1e-4 if updates in (0, 13) else (1e-4 * updates / 13 if updates < 13 else 1e-4 * (256 - updates) / 243)
    states = [{"name": item["name"], "exp_avg": tensor(item["shape"], "cuda:0"),
               "exp_avg_sq": tensor(item["shape"], "cuda:0"), "step": tensor([], "cpu", step=updates)} for item in adapters] if updates else []
    return {"metadata_scope": c.WITNESS_SCOPE, "name": "AdamW", "betas": [0.9, 0.999], "epsilon": 1e-8,
            "weight_decay": 0, "foreach": None, "fused": None, "capturable": False, "differentiable": False,
            "amsgrad": False, "maximize": False,
            "parameter_groups": [{"parameter_names": [item["name"] for item in adapters], "learning_rate": lr}], "state": states}


RNG_SHAPES = {"torch_cpu": [8], "cuda:0": [4]}  # Explicitly synthetic; not measured native generator sizes.


def rng():
    def bytes_meta(shape):
        return {"shape": list(shape), "dtype": "uint8", "device": "cpu", "numel": shape[0], "sha256": "f" * 64}
    return {"metadata_scope": c.WITNESS_SCOPE,
            "python": {"algorithm": "MT19937", "state_version": 3, "internal_state_words": 625,
                       "state_sha256": "e" * 64, "gaussian_cache": None},
            "torch_cpu": bytes_meta(RNG_SHAPES["torch_cpu"]),
            "cuda": [{"device": "cuda:0", "state": bytes_meta(RNG_SHAPES["cuda:0"])}]}


class ContractControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = "artifacts/mimo_central/contract_fixture"
        self.relative = self.run + "/adapter_064.pt"
        self.path = self.root / self.relative; self.path.parent.mkdir(parents=True)
        self.raw = b"Synthetic bytes only. This is deliberately not a Torch checkpoint.\n"
        self.path.write_bytes(self.raw)
        self.inventory_sha = c.digest(c.expected_adapter_inventory())
        self.bindings = {"code_sha256": "c" * 64, "data_sha256": c.DATA_SHA256, "plan_sha256": c.PLAN_SHA256,
                         "weight_sha256": c.WEIGHT_SHA256, "inventory_sha256": self.inventory_sha}
        artifact = {"relative_path": self.relative, "sha256": hashlib.sha256(self.raw).hexdigest(), "bytes": len(self.raw),
                    "format": "torch_save_adapter_only_v1", "stored_sections": ["parameters", "optimizer", "training_rng"],
                    "base_parameters_stored": False, "gradient_tensors_stored": False}
        self.state = {role: {"artifact": deepcopy(artifact), "bindings": deepcopy(self.bindings), "metadata": metadata}
                      for role, metadata in (("parameters", {"metadata_scope": c.WITNESS_SCOPE, "inventory": c.expected_adapter_inventory()}),
                                             ("optimizer", adam()), ("training_rng", rng()))}

    def check(self):
        return c.validate_checkpoint_descriptors(self.state, repository_root=self.root, approved_run_relative=self.run,
            expected_bindings=self.bindings, expected_inventory_sha256=self.inventory_sha, completed_updates=64,
            expected_rng_shapes=RNG_SHAPES)

    def rejects(self, code, function):
        with self.assertRaises(c.ContractError) as context:
            function()
        self.assertEqual(context.exception.code, code)

    def test_exact_model_namespace_counts_shapes_and_only_qv_aliases(self):
        adapters, base, sources = model_case()
        self.assertEqual(c.validate_model_metadata(adapters, base, source_tensor_sha256=sources), self.inventory_sha)
        self.assertEqual(len(c.expected_base_names()), 310)
        self.assertEqual(sum(item["numel"] for item in adapters), 1146880)
        self.assertEqual({item["name"].rsplit(".", 1)[0] for item in adapters}.__len__(), 56)

    def test_missing_foreign_reordered_or_wrong_shape_adapters_refused(self):
        for mode in ("missing", "foreign", "reordered", "shape", "dtype", "device", "grad"):
            with self.subTest(mode=mode):
                adapters, base, sources = model_case()
                if mode == "missing": adapters.pop()
                if mode == "foreign": adapters[0]["name"] = "model.layers.0.self_attn.k_proj.lora_A"
                if mode == "reordered": adapters[0], adapters[1] = adapters[1], adapters[0]
                if mode == "shape": adapters[1]["shape"] = [1024, 8]
                if mode == "dtype": adapters[0]["dtype"] = "bfloat16"
                if mode == "device": adapters[0]["device"] = "cpu"
                if mode == "grad": adapters[0]["requires_grad"] = False
                self.rejects("ADAPTER_INVENTORY", lambda: c.validate_model_metadata(adapters, base, source_tensor_sha256=sources))

    def test_boolean_adapter_shape_or_count_cannot_equal_integer(self):
        for field, value in (("shape", [True, 1024]), ("numel", True), ("requires_grad", 1)):
            adapters, base, sources = model_case(); adapters[0][field] = value
            self.rejects("ADAPTER_TYPES", lambda: c.validate_model_metadata(adapters, base, source_tensor_sha256=sources))

    def test_base_alias_duplication_missing_tensor_and_wrong_source_hash(self):
        adapters, base, sources = model_case()
        base["parameter_names"][0] = "lm_head.weight"
        self.rejects("BASE_NAME", lambda: c.validate_model_metadata(adapters, base, source_tensor_sha256=sources))
        adapters, base, sources = model_case(); base["parameter_names"][0] = base["parameter_names"][1]
        self.rejects("BASE_NAMES", lambda: c.validate_model_metadata(adapters, base, source_tensor_sha256=sources))
        adapters, base, sources = model_case(); base["source_tensor_sha256"].pop("model.norm.weight")
        self.rejects("SOURCE_TENSOR_KEYS", lambda: c.validate_model_metadata(adapters, base, source_tensor_sha256=sources))
        adapters, base, sources = model_case(); base["source_tensor_sha256"]["model.norm.weight"] = "0" * 64
        self.rejects("SOURCE_TENSOR_BINDING", lambda: c.validate_model_metadata(adapters, base, source_tensor_sha256=sources))

    def test_unsupported_base_wrapper_bias_layer_and_unfrozen_tie_refused(self):
        for bad in ("model.layers.0.self_attn.k_proj.base.weight", "model.layers.28.self_attn.q_proj.base.weight",
                    "model.layers.00.self_attn.q_proj.base.weight", "model.layers.0.self_attn.q_proj.base.bias"):
            self.rejects("BASE_NAME", lambda: c.canonical_base_name(bad))
        for field, value in (("unique_parameters", 596049920.0), ("requires_grad", True), ("embedding_tied", False),
                             ("dtype", "float32"), ("serialized_tensor_count", 311)):
            adapters, base, sources = model_case(); base[field] = value
            self.rejects("BASE_CONTRACT", lambda: c.validate_model_metadata(adapters, base, source_tensor_sha256=sources))

    def test_artifact_hash_success_explicitly_has_no_tensor_or_native_proof(self):
        result = self.check()
        self.assertEqual(result["artifact"]["sha256"], hashlib.sha256(self.raw).hexdigest())
        for field in ("checkpoint_deserialized", "artifact_tensor_contents_verified", "finite_tensors_physically_verified", "RNG_restored",
                      "native_qualified", "native_resume_qualified", "native_launch_authorized", "training_authorized", "E1_authorized", "CHECK_FINAL_access"):
            self.assertIs(result[field], False)
        self.assertEqual(result["framework_imports"], 0)
        self.assertEqual(result["model_calls"], 0)

    def test_non_json_tuple_inventory_group_and_storage_witnesses_refused(self):
        original = deepcopy(self.state)
        self.state["parameters"]["metadata"]["inventory"] = tuple(c.expected_adapter_inventory())
        self.rejects("ADAPTER_INVENTORY", self.check)
        self.state = deepcopy(original); self.state["parameters"]["metadata"]["inventory"][0]["shape"] = (8, 1024)
        self.rejects("ADAPTER_TYPES", self.check)
        self.state = deepcopy(original)
        metadata = self.state["optimizer"]["metadata"]
        metadata["parameter_groups"][0]["parameter_names"] = tuple(metadata["parameter_groups"][0]["parameter_names"])
        self.rejects("OPTIMIZER_TYPES", self.check)
        self.state = deepcopy(original)
        for section in self.state.values(): section["artifact"]["stored_sections"] = ("parameters", "optimizer", "training_rng")
        self.rejects("ARTIFACT_STORAGE_TYPES", self.check)

    def test_mixed_artifact_refs_storage_and_foreign_bindings_refused(self):
        original = deepcopy(self.state)
        self.state["optimizer"]["artifact"]["relative_path"] = self.run + "/other.pt"
        self.rejects("MIXED_ARTIFACT_REFERENCES", self.check)
        self.state = deepcopy(original); self.state["training_rng"]["bindings"]["code_sha256"] = "b" * 64
        self.rejects("SOURCE_BINDING", self.check)
        for field in ("base_parameters_stored", "gradient_tensors_stored"):
            self.state = deepcopy(original); self.state["parameters"]["artifact"][field] = True
            self.rejects("ARTIFACT_STORAGE_CONTRACT", self.check)

    def test_artifact_byte_budget_and_boolean_size_fail_before_file_reads(self):
        original = deepcopy(self.state)
        for size in (True, 0, c.MAX_ARTIFACT_BYTES + 1):
            self.state = deepcopy(original)
            for section in self.state.values(): section["artifact"]["bytes"] = size
            with patch.object(c.Path, "open", side_effect=AssertionError("file read before budget rejection")):
                self.rejects("ARTIFACT_BUDGET", self.check)

    def test_unapproved_closed_published_absolute_and_escape_paths_refused(self):
        for path in ("../adapter.pt", "C:/adapter.pt", "C:adapter.pt", "artifacts/mimo_central/CHECK/adapter.pt",
                     "artifacts/mimo_central/fresh_sealed/adapter.pt", "Rineya_Published_Cores/adapter.pt",
                     "artifacts/mimo_central/elsewhere/adapter.pt", self.run + "/./adapter.pt"):
            with self.subTest(path=path):
                code = "ARTIFACT_PATH" if path.startswith(("..", "C:")) or "/./" in path else (
                    "CLOSED_OR_PUBLISHED_PATH" if any(word in path for word in ("CHECK", "fresh_sealed", "Published")) else "ARTIFACT_PATH_ESCAPE")
                self.rejects(code, lambda: c.safe_artifact_path(self.root, self.run, path))
        self.rejects("APPROVED_RUN_PATH", lambda: c.safe_artifact_path(self.root, "outside/run", self.relative))

    def test_hash_and_size_mismatch_refused_without_deserialization(self):
        self.path.write_bytes(self.raw[:-1] + b"X")
        self.rejects("ARTIFACT_HASH", self.check)
        self.path.write_bytes(self.raw + b"X")
        self.rejects("ARTIFACT_SIZE", self.check)

    def test_mutation_during_streaming_hash_is_detected(self):
        original_open = Path.open; path = self.path; raw = self.raw
        class Mutation:
            def __init__(self, stream): self.stream = stream; self.done = False
            def __enter__(self): return self
            def __exit__(self, *args): self.stream.close()
            def fileno(self): return self.stream.fileno()
            def read(self, size):
                result = self.stream.read(size)
                if not self.done:
                    self.done = True
                    with original_open(path, "wb") as writer: writer.write(raw + b"changed")
                return result
        def opening(value, *args, **kwargs):
            stream = original_open(value, *args, **kwargs)
            return Mutation(stream) if value == path and args and args[0] == "rb" else stream
        with patch.object(Path, "open", opening):
            self.rejects("ARTIFACT_CHANGED", self.check)

    def test_same_size_mtime_restored_mutation_after_hashed_read_is_detected(self):
        original_open = Path.open; path = self.path; raw = self.raw; original_stat = path.stat()
        changed = raw.replace(b"Synthetic", b"SynthetiX", 1)
        self.assertEqual(len(changed), len(raw))
        class MutationAfterEOF:
            def __init__(self, stream): self.stream = stream; self.done = False
            def __enter__(self): return self
            def __exit__(self, *args): self.stream.close()
            def fileno(self): return self.stream.fileno()
            def read(self, size):
                result = self.stream.read(size)
                if not result and not self.done:
                    self.done = True
                    with original_open(path, "wb") as writer: writer.write(changed)
                    os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
                return result
        def opening(value, *args, **kwargs):
            stream = original_open(value, *args, **kwargs)
            return MutationAfterEOF(stream) if value == path and args and args[0] == "rb" else stream
        with patch.object(Path, "open", opening):
            self.rejects("ARTIFACT_CHANGED", self.check)
        self.assertEqual(path.stat().st_size, original_stat.st_size)
        self.assertEqual(path.stat().st_mtime_ns, original_stat.st_mtime_ns)
        self.assertEqual(path.read_bytes(), changed)

    @unittest.skipUnless(os.name == "nt", "Windows handle-query control")
    def test_windows_identity_query_failure_is_closed(self):
        self.check()  # Initialize the one cached API/type.
        ctypes, msvcrt, basic_type, _ = c._WINDOWS_FILE_API
        with patch.object(c, "_WINDOWS_FILE_API", (ctypes, msvcrt, basic_type, lambda *args: 0)):
            self.rejects("ARTIFACT_IDENTITY_QUERY", self.check)

    def test_returned_byte_receipt_does_not_claim_future_file_immutability(self):
        observed = self.check()
        self.path.write_bytes(self.raw[:-1] + b"X")
        self.assertEqual(observed["artifact"]["sha256"], hashlib.sha256(self.raw).hexdigest())
        self.assertIn("no immutability guarantee after return", observed["byte_identity_scope"])
        self.assertIn("same bounded bytes or opened handle", observed["future_restore_requirement"])
        self.rejects("ARTIFACT_HASH", self.check)

    def test_optimizer_initial_empty_state_exact_steps_and_lr_zero(self):
        for updates in (0, 1, 13, 64, 256):
            self.assertTrue(c.validate_adamw_metadata(adam(updates), completed_updates=updates))
        initial = adam(0); initial["state"] = adam(1)["state"]
        self.rejects("OPTIMIZER_STATE", lambda: c.validate_adamw_metadata(initial, completed_updates=0))
        initialized = adam(); initialized["state"][0]["step"]["device"] = "cuda:0"
        self.assertTrue(c.validate_adamw_metadata(initialized, completed_updates=64))

    def test_optimizer_flags_parameter_order_missing_moments_and_step_drift(self):
        for field, value in (("foreach", True), ("fused", True), ("capturable", True), ("differentiable", True),
                             ("epsilon", 1e-6), ("betas", [0.8, 0.999]), ("weight_decay", True)):
            metadata = adam(); metadata[field] = value
            self.rejects("OPTIMIZER_CONTRACT", lambda: c.validate_adamw_metadata(metadata, completed_updates=64))
        metadata = adam(); metadata["parameter_groups"][0]["parameter_names"].reverse()
        self.rejects("OPTIMIZER_PARAMETER_ORDER", lambda: c.validate_adamw_metadata(metadata, completed_updates=64))
        metadata = adam(); metadata["state"].pop()
        self.rejects("OPTIMIZER_STATE", lambda: c.validate_adamw_metadata(metadata, completed_updates=64))
        for value in (True, 63, float("nan"), float("inf")):
            metadata = adam(); metadata["state"][0]["step"]["value"] = value
            self.rejects("OPTIMIZER_STEP_VALUE", lambda: c.validate_adamw_metadata(metadata, completed_updates=64))

    def test_optimizer_tensor_dtype_device_shape_and_finite_witness_refused(self):
        for field, value, code in (("dtype", "bfloat16", "OPTIMIZER_TENSOR_CONTRACT"),
                                  ("device", "cpu", "OPTIMIZER_TENSOR_CONTRACT"),
                                  ("shape", [True, 1024], "OPTIMIZER_TENSOR_TYPES"),
                                  ("finite", False, "OPTIMIZER_TENSOR_CONTRACT"),
                                  ("requires_grad", True, "OPTIMIZER_TENSOR_CONTRACT")):
            metadata = adam(); metadata["state"][0]["exp_avg"][field] = value
            self.rejects(code, lambda: c.validate_adamw_metadata(metadata, completed_updates=64))

    def test_rng_wrong_shape_dtype_bool_and_extra_cuda_device_refused(self):
        self.assertTrue(c.validate_rng_metadata(rng(), expected_rng_shapes=RNG_SHAPES))
        for field, value, code in (("shape", [7], "RNG_TENSOR_CONTRACT"), ("shape", [True], "RNG_TENSOR_TYPES"),
                                  ("dtype", "float32", "RNG_TENSOR_CONTRACT"), ("device", "cuda:0", "RNG_TENSOR_CONTRACT")):
            metadata = rng(); metadata["torch_cpu"][field] = value
            self.rejects(code, lambda: c.validate_rng_metadata(metadata, expected_rng_shapes=RNG_SHAPES))
        metadata = rng(); metadata["cuda"].append(deepcopy(metadata["cuda"][0]))
        self.rejects("CUDA_RNG_INVENTORY", lambda: c.validate_rng_metadata(metadata, expected_rng_shapes=RNG_SHAPES))
        metadata = rng(); metadata["cuda"][0]["device"] = "cuda:1"
        self.rejects("CUDA_RNG_INVENTORY", lambda: c.validate_rng_metadata(metadata, expected_rng_shapes=RNG_SHAPES))

    def test_python_rng_strict_version_and_gaussian_cache(self):
        for field, value, code in (("state_version", 3.0, "PYTHON_RNG_CONTRACT"),
                                  ("internal_state_words", 624, "PYTHON_RNG_CONTRACT"),
                                  ("gaussian_cache", True, "PYTHON_RNG_GAUSSIAN"),
                                  ("gaussian_cache", float("inf"), "PYTHON_RNG_GAUSSIAN")):
            metadata = rng(); metadata["python"][field] = value
            self.rejects(code, lambda: c.validate_rng_metadata(metadata, expected_rng_shapes=RNG_SHAPES))
        metadata = rng(); metadata["python"]["gaussian_cache"] = -1.5
        self.assertTrue(c.validate_rng_metadata(metadata, expected_rng_shapes=RNG_SHAPES))

    def test_contract_never_imports_framework_modules(self):
        self.check()
        self.assertFalse(set(("torch", "transformers", "safetensors", "tokenizers")).intersection(sys.modules))


if __name__ == "__main__":
    unittest.main()
