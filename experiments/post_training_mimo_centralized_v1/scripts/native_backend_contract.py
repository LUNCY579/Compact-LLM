"""Draft stdlib contracts for native-backend metadata witnesses.

No tensor is inspected, framework imported, checkpoint deserialized, RNG restored,
or native/scientific authority granted. Artifact hashing proves byte identity
only; adapter-only contents and finite tensors require later physical evidence.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import re


WEIGHT_SHA256 = "537201676814ade103c4efa63f5ebe758ea309971803c4ad633dbca5b6b531b7"
DATA_SHA256 = "51b87fe4fd70d2fb23d16f270d1023716172dbbe1c88efd79dff8c3548623d6d"
PLAN_SHA256 = "319c780924adb6ea17d3fabd1850f097848c5e11272e65e730c9e791c843b156"
WITNESS_SCOPE = "RUNTIME_METADATA_WITNESS_ONLY"
MAX_ARTIFACT_BYTES = 32 * 1024 * 1024
ARTIFACT_FIELDS = {"relative_path", "sha256", "bytes", "format", "stored_sections", "base_parameters_stored", "gradient_tensors_stored"}
BINDING_FIELDS = {"code_sha256", "data_sha256", "plan_sha256", "weight_sha256", "inventory_sha256"}
_WINDOWS_FILE_API = None


class ContractError(ValueError):
    def __init__(self, code, detail):
        super().__init__(detail)
        self.code = code


def require(value, code, detail):
    if not value:
        raise ContractError(code, detail)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def fields(value, expected, code):
    require(type(value) is dict and set(value) == set(expected), code, "exact metadata fields required")


def sha(value, code="SHA256"):
    require(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value), code, "lowercase SHA256 required")


def same(actual, expected, code):
    require(canonical(actual) == canonical(expected), code, "metadata differs")


def expected_adapter_inventory():
    inventory = []
    for layer in range(28):
        for projection, outputs in (("q_proj", 2048), ("v_proj", 1024)):
            for factor, shape in (("A", [8, 1024]), ("B", [outputs, 8])):
                inventory.append({"name": f"model.layers.{layer}.self_attn.{projection}.lora_{factor}",
                    "shape": shape, "dtype": "float32", "device": "cuda:0", "requires_grad": True,
                    "numel": shape[0] * shape[1]})
    return inventory


def expected_base_names():
    names = ["model.embed_tokens.weight", "model.norm.weight"]
    suffixes = ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                "self_attn.q_norm.weight", "self_attn.k_norm.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
                "mlp.down_proj.weight", "input_layernorm.weight", "post_attention_layernorm.weight")
    return names + [f"model.layers.{layer}.{suffix}" for layer in range(28) for suffix in suffixes]


def canonical_base_name(name):
    require(type(name) is str, "BASE_NAME", "base parameter name required")
    for projection in ("q_proj", "v_proj"):
        pattern = rf"model\.layers\.([0-9]+)\.self_attn\.{projection}\.base\.weight"
        match = re.fullmatch(pattern, name)
        if match:
            require(str(int(match.group(1))) == match.group(1) and 0 <= int(match.group(1)) < 28,
                    "BASE_NAME", name)
            return f"model.layers.{match.group(1)}.self_attn.{projection}.weight"
    require(name in set(expected_base_names()), "BASE_NAME", name)
    return name


def validate_adapter_inventory(adapters):
    require(type(adapters) is list and len(adapters) == 112, "ADAPTER_INVENTORY", "112 ordered adapter tensors required")
    for actual, expected in zip(adapters, expected_adapter_inventory()):
        fields(actual, expected, "ADAPTER_FIELDS")
        require(type(actual["shape"]) is list and all(type(size) is int for size in actual["shape"])
                and type(actual["numel"]) is int and type(actual["requires_grad"]) is bool,
                "ADAPTER_TYPES", actual.get("name", "adapter"))
        same(actual, expected, "ADAPTER_INVENTORY")
    require(sum(item["numel"] for item in adapters) == 1146880, "ADAPTER_TOTAL", "1,146,880 adapter parameters required")
    return digest(adapters)


def validate_model_metadata(adapters, base, *, source_tensor_sha256):
    inventory_sha = validate_adapter_inventory(adapters)
    fields(base, {"unique_parameters", "dtype", "device", "requires_grad", "weight_sha256", "embedding_tied",
                  "serialized_tensor_count", "parameter_names", "source_tensor_sha256", "tied_alias"}, "BASE_FIELDS")
    for name, value in {"unique_parameters": 596049920, "dtype": "bfloat16", "device": "cuda:0", "requires_grad": False,
                        "weight_sha256": WEIGHT_SHA256, "embedding_tied": True, "serialized_tensor_count": 310,
                        "tied_alias": {"alias": "lm_head.weight", "target": "model.embed_tokens.weight", "same_object": True,
                                       "serialized_separately": False}}.items():
        same(base[name], value, "BASE_CONTRACT")
    names = base["parameter_names"]
    require(type(names) is list and len(names) == 310, "BASE_NAMES", "310 unique base parameters required")
    normalized = [canonical_base_name(name) for name in names]
    require(len(set(normalized)) == 310 and set(normalized) == set(expected_base_names()), "BASE_NAMES", "missing or aliased unique base parameter")
    for mapping in (source_tensor_sha256, base["source_tensor_sha256"]):
        require(type(mapping) is dict and set(mapping) == set(expected_base_names()), "SOURCE_TENSOR_KEYS", "exact310 serialized keys required")
        for value in mapping.values():
            sha(value, "SOURCE_TENSOR_SHA")
    same(base["source_tensor_sha256"], source_tensor_sha256, "SOURCE_TENSOR_BINDING")
    return inventory_sha


def tensor_witness(value, expected_shape, device, *, step=None):
    required = {"shape", "dtype", "device", "numel", "requires_grad", "finite"} | ({"value"} if step is not None else set())
    fields(value, required, "OPTIMIZER_TENSOR_FIELDS")
    require(type(value["shape"]) is list and all(type(size) is int for size in value["shape"])
            and type(value["numel"]) is int, "OPTIMIZER_TENSOR_TYPES", "typed shape/count required")
    numel = math.prod(expected_shape)
    for field, wanted in {"shape": expected_shape, "dtype": "float32", "device": device, "numel": numel,
                          "requires_grad": False, "finite": True}.items():
        same(value[field], wanted, "OPTIMIZER_TENSOR_CONTRACT")
    if step is not None:
        require(type(value["value"]) in (int, float) and math.isfinite(value["value"]) and value["value"] == step,
                "OPTIMIZER_STEP_VALUE", "exact finite completed-update counter required")


def validate_adamw_metadata(metadata, *, completed_updates):
    require(type(completed_updates) is int and 0 <= completed_updates <= 256, "COMPLETED_UPDATES", "0..256 integer required")
    fields(metadata, {"metadata_scope", "name", "betas", "epsilon", "weight_decay", "foreach", "fused", "capturable",
                      "differentiable", "amsgrad", "maximize", "parameter_groups", "state"}, "OPTIMIZER_FIELDS")
    for field, value in {"metadata_scope": WITNESS_SCOPE, "name": "AdamW", "betas": [0.9, 0.999], "epsilon": 1e-8,
                         "foreach": None, "fused": None, "capturable": False,
                         "differentiable": False, "amsgrad": False, "maximize": False}.items():
        same(metadata[field], value, "OPTIMIZER_CONTRACT")
    require(type(metadata["betas"]) is list and all(type(value) is float for value in metadata["betas"]),
            "OPTIMIZER_TYPES", "JSON float-array betas required")
    require(type(metadata["weight_decay"]) in (int, float) and metadata["weight_decay"] == 0,
            "OPTIMIZER_CONTRACT", "zero numeric weight decay required")
    names = [item["name"] for item in expected_adapter_inventory()]
    groups = metadata["parameter_groups"]
    require(type(groups) is list and len(groups) == 1, "OPTIMIZER_GROUPS", "one ordered group required")
    fields(groups[0], {"parameter_names", "learning_rate"}, "OPTIMIZER_GROUP_FIELDS")
    require(type(groups[0]["parameter_names"]) is list and all(type(value) is str for value in groups[0]["parameter_names"]),
            "OPTIMIZER_TYPES", "ordered JSON parameter-name array required")
    same(groups[0]["parameter_names"], names, "OPTIMIZER_PARAMETER_ORDER")
    lr = groups[0]["learning_rate"]
    expected_lr = 1e-4 if completed_updates in (0, 13) else (1e-4 * completed_updates / 13 if completed_updates < 13 else 1e-4 * (256 - completed_updates) / 243)
    require(type(lr) in (int, float) and math.isfinite(lr) and lr == expected_lr, "OPTIMIZER_LEARNING_RATE", "fixed saved-boundary LR required")
    state = metadata["state"]
    require(type(state) is list and len(state) == (112 if completed_updates else 0), "OPTIMIZER_STATE", "complete initialized state or initial empty state required")
    for item, adapter in zip(state, expected_adapter_inventory()):
        fields(item, {"name", "exp_avg", "exp_avg_sq", "step"}, "OPTIMIZER_STATE_FIELDS")
        same(item["name"], adapter["name"], "OPTIMIZER_PARAMETER_ORDER")
        for moment in ("exp_avg", "exp_avg_sq"):
            tensor_witness(item[moment], adapter["shape"], "cuda:0")
        require(type(item["step"]) is dict and item["step"].get("device") in ("cpu", "cuda:0"), "OPTIMIZER_STEP_DEVICE", "CPU scalar default or CUDA0 required")
        tensor_witness(item["step"], [], item["step"]["device"], step=completed_updates)
    return True


def validate_rng_metadata(metadata, *, expected_rng_shapes):
    fields(expected_rng_shapes, {"torch_cpu", "cuda:0"}, "RNG_EXPECTED_SHAPES")
    for shape in expected_rng_shapes.values():
        require(type(shape) is list and len(shape) == 1 and type(shape[0]) is int and 0 < shape[0] <= 65536,
                "RNG_EXPECTED_SHAPES", "caller-frozen bounded1D byte shape required")
    fields(metadata, {"metadata_scope", "python", "torch_cpu", "cuda"}, "RNG_FIELDS")
    same(metadata["metadata_scope"], WITNESS_SCOPE, "RNG_SCOPE")
    fields(metadata["python"], {"algorithm", "state_version", "internal_state_words", "state_sha256", "gaussian_cache"}, "PYTHON_RNG_FIELDS")
    for field, value in {"algorithm": "MT19937", "state_version": 3, "internal_state_words": 625}.items():
        same(metadata["python"][field], value, "PYTHON_RNG_CONTRACT")
    sha(metadata["python"]["state_sha256"], "RNG_SHA")
    gaussian = metadata["python"]["gaussian_cache"]
    require(gaussian is None or type(gaussian) is float and math.isfinite(gaussian), "PYTHON_RNG_GAUSSIAN", "optional finite float required")
    def check_tensor(value, shape):
        fields(value, {"shape", "dtype", "device", "numel", "sha256"}, "RNG_TENSOR_FIELDS")
        require(type(value["shape"]) is list and all(type(size) is int for size in value["shape"])
                and type(value["numel"]) is int, "RNG_TENSOR_TYPES", "typed byte shape required")
        for field, wanted in {"shape": shape, "dtype": "uint8", "device": "cpu", "numel": shape[0]}.items():
            same(value[field], wanted, "RNG_TENSOR_CONTRACT")
        sha(value["sha256"], "RNG_SHA")
    check_tensor(metadata["torch_cpu"], expected_rng_shapes["torch_cpu"])
    require(type(metadata["cuda"]) is list and len(metadata["cuda"]) == 1, "CUDA_RNG_INVENTORY", "exactly CUDA0 RNG required")
    fields(metadata["cuda"][0], {"device", "state"}, "CUDA_RNG_FIELDS")
    same(metadata["cuda"][0]["device"], "cuda:0", "CUDA_RNG_INVENTORY")
    check_tensor(metadata["cuda"][0]["state"], expected_rng_shapes["cuda:0"])
    return True


def safe_artifact_path(repository_root, approved_run_relative, artifact_relative):
    root = Path(repository_root).resolve()
    def relative(value):
        require(type(value) is str and value and not PureWindowsPath(value).drive and "\\" not in value,
                "ARTIFACT_PATH", "normalized relative POSIX path required")
        parts = Path(value).parts
        require(not Path(value).is_absolute() and not any(part in ("..", ".") for part in parts), "ARTIFACT_PATH", value)
        require(Path(value).as_posix() == value, "ARTIFACT_PATH", "normalized relative path required")
        require(not any(re.search(r"(^|[^a-z])(check|final|test2|fresh_sealed|published)([^a-z]|$)", part.lower())
                        or "rineya_published_cores" in part.lower() for part in parts), "CLOSED_OR_PUBLISHED_PATH", value)
        return root / value
    approved = relative(approved_run_relative)
    require(approved_run_relative.startswith("artifacts/mimo_central/") and len(approved.parts) > len((root / "artifacts/mimo_central").parts),
            "APPROVED_RUN_PATH", "named local run directory required")
    artifact = relative(artifact_relative)
    artifact_root = (root / "artifacts/mimo_central").resolve()
    run = approved.resolve(); target = artifact.resolve()
    require(artifact_root.is_relative_to(root) and run.is_relative_to(artifact_root) and target.is_relative_to(run)
            and target != run, "ARTIFACT_PATH_ESCAPE", "artifact must stay inside approved run")
    # Also reject forbidden components reached through symlink resolution.
    relative(target.relative_to(root).as_posix())
    return target


def _windows_basic_times(stream):
    """Normalize native creation/change times on the already opened handle.

    Windows Python path-stat and fstat can assign different meanings to ctime;
    do not compare those ctime fields with each other.
    """
    global _WINDOWS_FILE_API
    if _WINDOWS_FILE_API is None:
        import ctypes
        import msvcrt
        class BasicInfo(ctypes.Structure):
            _fields_ = [(name, ctypes.c_longlong) for name in ("CreationTime", "LastAccessTime", "LastWriteTime", "ChangeTime")] + [
                ("FileAttributes", ctypes.c_ulong)]
        query = ctypes.WinDLL("kernel32", use_last_error=True).GetFileInformationByHandleEx
        query.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong]
        query.restype = ctypes.c_int
        _WINDOWS_FILE_API = (ctypes, msvcrt, BasicInfo, query)
    ctypes, msvcrt, BasicInfo, query = _WINDOWS_FILE_API
    record = BasicInfo()
    require(bool(query(msvcrt.get_osfhandle(stream.fileno()), 0, ctypes.byref(record), ctypes.sizeof(record)))
            and record.CreationTime > 0 and record.ChangeTime > 0,
            "ARTIFACT_IDENTITY_QUERY", "positive Windows creation/change times required")
    return record.CreationTime, record.ChangeTime


def _file_identity(stat, change_time=None, creation_time=None):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns,
            stat.st_ctime_ns if creation_time is None else creation_time,
            stat.st_ctime_ns if change_time is None else change_time)


def _stream_identity(stream):
    observed = os.fstat(stream.fileno())
    if os.name == "nt":
        creation, change = _windows_basic_times(stream)
        return _file_identity(observed, change, creation)
    return _file_identity(observed)


def _path_identity(path):
    before = _file_identity(path.stat())
    comparable = 4 if os.name == "nt" else 5
    with path.open("rb") as stream:
        observed = _stream_identity(stream)
        require(observed[:comparable] == before[:comparable], "ARTIFACT_CHANGED", "path/opened file identity differs")
    require(_file_identity(path.stat())[:comparable] == before[:comparable], "ARTIFACT_CHANGED", "path changed during identity observation")
    return observed


def verify_artifact_bytes(path, artifact):
    require(type(artifact.get("bytes")) is int and 0 < artifact["bytes"] <= MAX_ARTIFACT_BYTES,
            "ARTIFACT_BUDGET", "positive artifact up to32MiB required")
    before = _path_identity(path)
    require(path.is_file() and before[2] == artifact["bytes"], "ARTIFACT_SIZE", "exact artifact byte size required")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        require(_stream_identity(stream) == before, "ARTIFACT_CHANGED", "opened file identity differs")
        while block := stream.read(1024 * 1024):
            h.update(block)
        require(_stream_identity(stream) == before, "ARTIFACT_CHANGED", "opened file mutated during hash")
    require(_path_identity(path) == before, "ARTIFACT_CHANGED", "artifact path identity changed during hash")
    require(h.hexdigest() == artifact["sha256"], "ARTIFACT_HASH", "artifact byte hash differs")
    return {"sha256": h.hexdigest(), "bytes": before[2]}


def validate_checkpoint_descriptors(state, *, repository_root, approved_run_relative, expected_bindings,
                                    expected_inventory_sha256, completed_updates, expected_rng_shapes):
    fields(state, {"parameters", "optimizer", "training_rng"}, "CHECKPOINT_SECTIONS")
    fields(expected_bindings, BINDING_FIELDS, "BINDINGS")
    for value in expected_bindings.values():
        sha(value, "BINDING_SHA")
    sha(expected_inventory_sha256, "INVENTORY_SHA")
    require(expected_bindings["data_sha256"] == DATA_SHA256 and expected_bindings["plan_sha256"] == PLAN_SHA256
            and expected_bindings["weight_sha256"] == WEIGHT_SHA256
            and expected_bindings["inventory_sha256"] == expected_inventory_sha256 == digest(expected_adapter_inventory()),
            "SOURCE_BINDING", "exact Titan/data/plan/inventory required")
    common = None
    for role in ("parameters", "optimizer", "training_rng"):
        section = state[role]
        fields(section, {"artifact", "bindings", "metadata"}, "CHECKPOINT_DESCRIPTOR")
        same(section["bindings"], expected_bindings, "SOURCE_BINDING")
        artifact = section["artifact"]
        fields(artifact, ARTIFACT_FIELDS, "ARTIFACT_FIELDS")
        sha(artifact["sha256"], "ARTIFACT_SHA")
        require(type(artifact["bytes"]) is int and 0 < artifact["bytes"] <= MAX_ARTIFACT_BYTES, "ARTIFACT_BUDGET", "positive adapter-only artifact up to32MiB required")
        require(type(artifact["stored_sections"]) is list, "ARTIFACT_STORAGE_TYPES", "JSON section-name array required")
        for field, value in {"format": "torch_save_adapter_only_v1", "stored_sections": ["parameters", "optimizer", "training_rng"],
                             "base_parameters_stored": False, "gradient_tensors_stored": False}.items():
            same(artifact[field], value, "ARTIFACT_STORAGE_CONTRACT")
        if common is None:
            common = artifact
        else:
            same(artifact, common, "MIXED_ARTIFACT_REFERENCES")
    fields(state["parameters"]["metadata"], {"metadata_scope", "inventory"}, "PARAMETER_METADATA")
    same(state["parameters"]["metadata"]["metadata_scope"], WITNESS_SCOPE, "PARAMETER_SCOPE")
    require(validate_adapter_inventory(state["parameters"]["metadata"]["inventory"]) == expected_inventory_sha256,
            "ADAPTER_INVENTORY", "parameter inventory identity differs")
    validate_adamw_metadata(state["optimizer"]["metadata"], completed_updates=completed_updates)
    validate_rng_metadata(state["training_rng"]["metadata"], expected_rng_shapes=expected_rng_shapes)
    path = safe_artifact_path(repository_root, approved_run_relative, common["relative_path"])
    observed = verify_artifact_bytes(path, common)
    return {"status": "PASS_METADATA_WITNESS_AND_ARTIFACT_BYTES_ONLY", "artifact": observed,
            "byte_identity_scope": "Bytes read during this validation; no immutability guarantee after return.",
            "future_restore_requirement": "Verify and deserialize the same bounded bytes or opened handle; an unverified later reopen is not covered.",
            "inventory_sha256": expected_inventory_sha256, "framework_imports": 0, "checkpoint_deserialized": False,
            "artifact_tensor_contents_verified": False, "finite_tensors_physically_verified": False,
            "RNG_restored": False, "native_qualified": False, "native_resume_qualified": False,
            "native_launch_authorized": False, "training_authorized": False, "E1_authorized": False,
            "CHECK_FINAL_access": False, "model_calls": 0, "training_updates": 0}
