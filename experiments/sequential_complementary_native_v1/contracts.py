"""Stdlib-only identities, inventories and a new *non-training* approval contract.

An asset hash identifies bytes, not a model qualification or a human approval.
The inherited RRP validator intentionally remains untouched: it requires training
permission and therefore must NOT be used to manufacture an inference-only grant.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import struct
import subprocess

from experiments.model_quality_gpu_v1.integrity import (
    ROOT, canonical, decode, digest, file_hash, new_dir, read_json, regular, write_new,
)

PACKAGE = Path(__file__).resolve().parent
PARENT = "3ad2769a9f8343a9697e1720fcdc9efa18c3845a"
BRANCH = "research/sequential-complementary-native-v1"
EXPERIMENT = "SCP-NQ-001"
SUBJECT = "rineya.core.titan-sparse-experimental.v1.0.0-devfreeze.28L.supa17-rrp015-control-seed17-magprune-k0p25"
SOURCE_SHA = "962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b"
TRAINED_SHA = "07156fb5aa4a8c510c63e8af2118a08a8b6bf004549532d497c2e89e2deec24d"
SUBJECT_SHA = "a28629c9d3454fbf2e884fd7ca442c1b406e7caf4b36d34b2c0614d0add31284"
PHASES = ("student", "mr")
CLOSED = ("training_permission", "teacher_comparisons_permission", "confirmation_permission",
          "quality_permission", "release_permission")
EXCLUDES = ("training", "teacher_comparisons", "protected_banks", "confirmation",
            "quality", "release", "cloud", "RL", "compression_candidates")
# Proposed maxima, never an approval. RRP Journal's own launch overhead is retained.
CEILING = {
    "total_owned_work_wall_seconds": 3600,
    "model_worker_wall_seconds_and_conservative_gpu_seconds": 2600,
    "processed_training_tokens_all_attempts": 0,
    "evaluation_and_qualification_model_calls": 1024,  # EACH actual forward
    "generated_tokens_all_candidates": 576,          # MR + repeated direct
    "on_policy_rl_rollout_tokens": 0,
    "new_disk_bytes": 128 * 2**20,
    "adaptive_model_selection_rounds": 0,
    "model_worker_launches": 2,
    "retries": 0,
    "cloud_or_teacher_spend": 0,
    "generation_requests": 10,
}
HOST = {"max_host_rss_bytes": 8 * 2**30, "min_free_disk_bytes": 14 * 2**30,
        "max_sequence": 2048, "max_stage_timeout_seconds": 1200}
# These are parent Git BLOB SHA-1s, NOT checkpoint SHA-256s. Never modify the
# inherited validators/optimizer/loss/checkpoint implementation to pass this lane.
INHERITED = {
    "model_quality_gpu_v1/integrity.py": "223d3aec8378ad316356e5a7fced6c7cb2690405",
    "model_quality_gpu_v1/native.py": "79f0d467c2c4bd9ae8004c82afb0d4c84a637bf6",
    "model_quality_gpu_v1/resources.py": "deffbe9ece3cb106c99e27d8a526452cd68d09e7",
    "model_quality_gpu_v1/supervisor.py": "9377c5dd178b78fe64a4b07aa6ee510060aaa947",
    "model_quality_gpu_v1/source_lock.json": "50dd384c2eac68385634599012bdf068188f04fa",
    "model_quality_gpu_v1/losses.py": "91e06434a3c5b3a662fde0e84a891339da2f47aa",
    "model_quality_gpu_v1/optimizer.py": "04a748d2f706be8091d884fb59feea023aa650a5",
    "model_quality_gpu_v1/checkpoints.py": "8c78fa38b9095ca9bf68a9273484801b3b8bd6ec",
    "reasoning_posttraining_v1/budget.py": "c9e5054154123ad18b2472cfc335baf25ed296ef",
    "reasoning_posttraining_v1/preflight.py": "a527f47ab236d56f28a0d2aab25482b71cb0b4bf",
    "reasoning_posttraining_v1/native_runtime.py": "5b87b7a49559fb342eb51cd0077b3c82186f2206",
    "reasoning_posttraining_v1/engine.py": "b6c9748f75ee55bc8c244ee82733d1931a857ace",
    "reasoning_posttraining_v1/serialization.py": "bf2912f8d65b49fd9de0ef010558967ca2e883ec",
    "sequential_complementary_v1/delta.py": "8cf2090d1e72fcfb0b088e77f286575bb3d62c63",
    "sequential_complementary_v1/data.py": "d529aaf9dec2896008d5e4d66ecd52db65c36112",
    "sequential_complementary_v1/analysis.py": "d97c5f647648239b1ee6cc56dc72691f30f3bf29",
    "sequential_complementary_v1/integrity.py": "85b13f4ecb7fcd663df6edcbd4dc4f164971d931",
    "sequential_complementary_v1/protocol.json": "e7cb7a748a110ea29676e5aebd54cb69e1229964",
    "sequential_complementary_v1/SOURCE_PINS.json": "0c5336d6864720f6a4744eb84869ace7a091fbff",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_path(path: Path) -> Path:
    path = Path(path).absolute()
    require(".." not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)),
            "symlink/traversal path refused")
    return path


def external(path: Path) -> Path:
    path = safe_path(path)
    require(path != ROOT and ROOT not in path.parents, "mutable evidence must be outside checkout")
    return path


def checkout(expected=None) -> str:
    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True,
                                       stderr=subprocess.STDOUT, timeout=20).strip()
    require(Path(git("rev-parse", "--show-toplevel")).resolve() == ROOT, "full checkout required")
    require(git("branch", "--show-current") == BRANCH, "wrong dedicated native branch")
    require(not git("status", "--porcelain", "--untracked-files=all"), "clean checkout required")
    # A clean status must not be hiding modified executable files behind index flags.
    flags = git("ls-files", "-v").splitlines()
    require(all(line.startswith("H ") for line in flags), "hidden/skip-worktree index entries refused")
    subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", PARENT, "HEAD"],
                   check=True, capture_output=True, timeout=20)
    head = git("rev-parse", "HEAD")
    require(expected is None or expected == head, "source commit differs from frozen binding")
    return head


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def source_code() -> dict:
    files = {}
    for name, wanted in INHERITED.items():
        data = regular(ROOT / "experiments" / name).read_bytes().replace(b"\r\n", b"\n")
        require(git_blob(data) == wanted, f"inherited implementation changed: {name}")
        files["experiments/" + name] = hashlib.sha256(data).hexdigest()
    paths = list(PACKAGE.rglob("*.py")) + [ROOT / "experiments/sequential_complementary_v1/reasoning.py"]
    for path in sorted(paths):
        data = regular(path).read_bytes().replace(b"\r\n", b"\n")
        files[path.relative_to(ROOT).as_posix()] = hashlib.sha256(data).hexdigest()
    return {"sha256": digest(files), "files": files}


def shapes() -> dict[str, list[int]]:
    result = {"model.embed_tokens.weight": [151936, 1024], "model.norm.weight": [1024]}
    for layer in range(28):
        for suffix, shape in {
            "self_attn.q_proj.weight": [2048, 1024], "self_attn.k_proj.weight": [1024, 1024],
            "self_attn.v_proj.weight": [1024, 1024], "self_attn.o_proj.weight": [1024, 2048],
            "self_attn.q_norm.weight": [128], "self_attn.k_norm.weight": [128],
            "mlp.gate_proj.weight": [3072, 1024], "mlp.up_proj.weight": [3072, 1024],
            "mlp.down_proj.weight": [1024, 3072], "input_layernorm.weight": [1024],
            "post_attention_layernorm.weight": [1024],
        }.items():
            result[f"model.layers.{layer}.{suffix}"] = shape
    return dict(sorted(result.items()))


def trained_shapes() -> dict[str, list[int]]:
    prefixes = tuple(f"model.layers.{i}." for i in range(8, 28))
    return {name: shape for name, shape in shapes().items() if name.startswith(prefixes)}


def inventory(path: Path, expected: dict, dtype: str) -> dict:
    """Bounded safetensors HEADER inspection; no torch import or tensor allocation."""
    path = regular(path)
    with path.open("rb") as stream:
        raw = stream.read(8)
        require(len(raw) == 8, "short safetensors header")
        size = struct.unpack("<Q", raw)[0]
        require(2 <= size <= 2**21 and size + 8 <= path.stat().st_size, "invalid safetensors header length")
        header = decode(stream.read(size).decode("utf-8"))
    require(type(header) is dict, "safetensors header is not an object")
    metadata = header.pop("__metadata__", {})
    require(type(metadata) is dict and all(type(k) is str and type(v) is str for k, v in metadata.items()),
            "invalid safetensors metadata")
    require(set(header) == set(expected), "exact tensor-name inventory differs")
    ranges, count = [], 0
    require(dtype in ("BF16", "F32"), "unregistered storage dtype")
    for name, shape in expected.items():
        spec = header[name]
        require(type(spec) is dict and set(spec) == {"shape", "dtype", "data_offsets"}, "invalid tensor header")
        require(type(spec["shape"]) is list and all(type(x) is int and x > 0 for x in spec["shape"])
                and spec["shape"] == shape and spec["dtype"] == dtype, "tensor shape/dtype differs")
        offsets = spec["data_offsets"]
        require(type(offsets) is list and len(offsets) == 2 and all(type(x) is int and x >= 0 for x in offsets),
                "invalid tensor offsets")
        elements = math.prod(shape)
        require(offsets[1] - offsets[0] == elements * (2 if dtype == "BF16" else 4), "tensor byte extent differs")
        ranges.append(offsets)
        count += elements
    cursor = 0
    for start, end in sorted(ranges):
        require(start == cursor, "overlapping or gapped tensor payload")
        cursor = end
    require(cursor + size + 8 == path.stat().st_size, "trailing/truncated tensor payload")
    return {"tensor_count": len(header), "parameters": count, "stored_dtype": dtype,
            "inventory_sha256": digest({name: {"shape": expected[name], "dtype": dtype} for name in sorted(expected)})}


def source_lock() -> dict:
    value = read_json(ROOT / "experiments/model_quality_gpu_v1/source_lock.json")
    require(value["assets"]["model.safetensors"] == SOURCE_SHA, "foreign Supa source lock")
    return value


def _directory(path: Path, expected: dict) -> dict:
    path = safe_path(path)
    require(path.is_dir(), "local asset directory missing")
    inert = {"CORE.json", "README.md", "LICENSE", "LICENSE.txt", "RENAME.json", "AMENDMENT_001.json"}
    # No hidden HF-consumed override, adapter, remote code, extra weights or template.
    for child in path.iterdir():
        require(child.is_file() and not child.is_symlink() and child.name in set(expected) | inert,
                f"unregistered asset-directory entry: {child.name}")
    return {name: file_hash(path / name) for name in expected}


def audit_assets(source: Path, trained: Path, incumbent: Path) -> dict:
    wanted = source_lock()["assets"]
    base = _directory(source, wanted)
    require(base == wanted, "exact Supa assets differ")
    sparse = _directory(incumbent, wanted)
    require(sparse["model.safetensors"] == SUBJECT_SHA, "exact Titan-Sparse weight hash differs")
    for name in wanted.keys() - {"model.safetensors", "config.json"}:
        require(sparse[name] == wanted[name], f"Titan tokenizer/generation sidecar differs: {name}")
    config, candidate = read_json(Path(source) / "config.json"), read_json(Path(incumbent) / "config.json")
    variant = "SOURCE_SIDECAR_WITH_EXPLICIT_FP32_LOADER"
    if candidate != config:
        require("torch_dtype" in config and candidate == {**config, "torch_dtype": "float32"},
                "only documented Titan config torch_dtype->float32 deviation is permitted")
        variant = "PUBLISHED_FP32_SIDECAR"
    trained = regular(trained)
    require(file_hash(trained) == TRAINED_SHA, "exact trained-partial FP32 hash differs")
    full, tail = shapes(), trained_shapes()
    stored = {
        "source": inventory(Path(source) / "model.safetensors", full, "BF16"),
        "trained": inventory(trained, tail, "F32"),
        "subject": inventory(Path(incumbent) / "model.safetensors", full, "F32"),
    }
    return {"source_id": source_lock()["source_id"], "subject_id": SUBJECT,
            "source_path": str(safe_path(source)), "trained_path": str(trained),
            "subject_path": str(safe_path(incumbent)), "source_assets": base,
            "trained_sha256": TRAINED_SHA, "subject_assets": sparse, "inventories": stored,
            "subject_config_variant": variant, "compute_dtype": "float32",
            "trained_names": list(tail), "trainable_tensors_during_qualification": 0}


def revalidate(binding: dict) -> dict:
    require(type(binding) is dict and binding.get("schema") == "SCP_NATIVE_BINDING_V1"
            and binding.get("experiment") == EXPERIMENT, "foreign native binding")
    require(binding.get("scope") == "INFERENCE_QUALIFICATION_ONLY" and binding.get("host") == HOST,
            "native scope/resources changed")
    require(all(binding.get(k) is False for k in CLOSED), "binding opens a closed claim")
    journal, lease = external(Path(binding["journal_path"])), external(Path(binding["lease_path"]))
    require(str(journal) == binding["journal_path"] and str(lease) == binding["lease_path"]
            and lease != journal and journal not in lease.parents, "noncanonical session/lease binding")
    checkout(binding["checkout"])
    require(binding["code"] == source_code(), "executable identity changed")
    assets = binding["assets"]
    actual = audit_assets(Path(assets["source_path"]), Path(assets["trained_path"]), Path(assets["subject_path"]))
    require(actual == assets, "bound asset bytes or inventory changed")
    require(binding["scp_protocol_sha256"] == file_hash(ROOT / "experiments/sequential_complementary_v1/protocol.json"),
            "published SCP protocol changed")
    return actual


def grant_request(binding: dict) -> dict:
    return {"schema": "SCP_NATIVE_GRANT_V1", "experiment": EXPERIMENT, "decision": "PENDING",
            "approved_by": None, "authorization_reference": None, "binding_sha256": digest(binding),
            "inference_qualification_permission": False, **{k: False for k in CLOSED},
            "limits": dict(CEILING), "excludes": list(EXCLUDES)}


def validate_approval(grant: dict, binding: dict) -> None:
    expected = set(grant_request(binding))
    require(type(grant) is dict and set(grant) == expected, "incomplete/expanded approval schema")
    require(grant["schema"] == "SCP_NATIVE_GRANT_V1" and grant["experiment"] == EXPERIMENT,
            "old/foreign permits cannot authorize this subject")
    require(grant["decision"] == "APPROVED" and grant["inference_qualification_permission"] is True,
            "explicit inference-only numeric approval absent")
    for name in ("approved_by", "authorization_reference"):
        require(type(grant[name]) is str and bool(grant[name].strip()), "human authorization attribution absent")
    require(grant["binding_sha256"] == digest(binding), "approval belongs to another subject/source/plan")
    require(all(grant[k] is False for k in CLOSED), "inference qualification cannot grant training/quality")
    require(grant["excludes"] == list(EXCLUDES), "excluded lanes changed")
    limits = grant["limits"]
    require(type(limits) is dict and set(limits) == set(CEILING), "complete numeric budget required")
    require(all(type(v) is int and 0 <= v <= CEILING[k] for k, v in limits.items()), "invalid/expanded numeric budget")


def propose(source: Path, trained: Path, incumbent: Path, out: Path, journal: Path, lease: Path) -> dict:
    journal, lease, out = external(journal), external(lease), external(out)
    require(not journal.exists(), "a new binding requires an unused journal path")
    require(journal != out and journal not in out.parents and out not in journal.parents,
            "binding and budget journal directories must be separate")
    require(lease != journal and journal not in lease.parents and lease != out and out not in lease.parents,
            "shared model lease must be outside session evidence directories")
    head, code = checkout(), source_code()
    assets = audit_assets(source, trained, incumbent)
    binding = {"schema": "SCP_NATIVE_BINDING_V1", "experiment": EXPERIMENT,
               "scope": "INFERENCE_QUALIFICATION_ONLY", "checkout": head, "code": code,
               "journal_path": str(journal), "lease_path": str(lease),
               "assets": assets, "host": dict(HOST), **{k: False for k in CLOSED},
               "scp_protocol_sha256": file_hash(ROOT / "experiments/sequential_complementary_v1/protocol.json")}
    out = new_dir(external(out))
    write_new(out / "BINDING.json", binding)
    write_new(out / "BUDGET_REQUEST.json", grant_request(binding))
    write_new(out / "INVENTORY.json", {"full": shapes(), "eligible_tail20": trained_shapes(),
                                      "currently_trainable": [], "not_a_training_permit": True})
    write_new(out / "PREFLIGHT.json", {"status": "ASSETS_BOUND_NATIVE_NOT_RUN", "model_calls": 0,
        "training_steps": 0, "binding_sha256": digest(binding), "native_pass": False,
        "requires": ["explicit inference-only approval", "pinned runtime", "unaltered MQ guards", "student then MR qualification"]})
    return binding
