from __future__ import annotations

import json
import math
import re
import struct
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .hashing import bundle_sha256, canonical_sha256, sha256_file, sha256_text


LAYER_KEY = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
)
REPRESENTATIVE_TENSORS = (
    "model.embed_tokens.weight",
    "model.layers.0.input_layernorm.weight",
    "model.layers.0.post_attention_layernorm.weight",
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.mlp.gate_proj.weight",
    "model.layers.0.mlp.up_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
    "model.norm.weight",
    "lm_head.weight",
)


class ArchitectureAuditError(RuntimeError):
    pass


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_safetensors_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ArchitectureAuditError(f"truncated safetensors prefix: {path}")
        header_bytes = struct.unpack("<Q", prefix)[0]
        if header_bytes <= 0 or header_bytes > 256 * 1024 * 1024:
            raise ArchitectureAuditError(
                f"invalid safetensors header length {header_bytes}: {path}"
            )
        payload = handle.read(header_bytes)
        if len(payload) != header_bytes:
            raise ArchitectureAuditError(f"truncated safetensors header: {path}")
    header = json.loads(payload.decode("utf-8"))
    if not isinstance(header, dict):
        raise ArchitectureAuditError(f"non-object safetensors header: {path}")
    return header


def resolve_weight_files(model_path: Path) -> tuple[list[Path], dict[str, str] | None]:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        index = _read_json(index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ArchitectureAuditError(f"invalid weight_map: {index_path}")
        shard_names = sorted(set(weight_map.values()))
        paths = [model_path / name for name in shard_names]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise ArchitectureAuditError(f"missing weight shards: {missing}")
        return paths, {str(key): str(value) for key, value in weight_map.items()}

    single = model_path / "model.safetensors"
    if single.is_file():
        return [single], None
    raise ArchitectureAuditError(f"no safetensors checkpoint found in {model_path}")


def inspect_tensor_headers(
    weight_files: list[Path], weight_map: dict[str, str] | None
) -> dict[str, Any]:
    tensors: dict[str, dict[str, Any]] = {}
    shard_for_key: dict[str, str] = {}
    metadata: dict[str, Any] = {}
    for path in weight_files:
        header = read_safetensors_header(path)
        metadata[path.name] = header.get("__metadata__", {})
        for key, entry in header.items():
            if key == "__metadata__":
                continue
            if key in tensors:
                raise ArchitectureAuditError(f"duplicate tensor key {key}")
            if not isinstance(entry, dict) or "shape" not in entry or "dtype" not in entry:
                raise ArchitectureAuditError(f"invalid tensor header for {key}")
            shape = [int(value) for value in entry["shape"]]
            if any(value < 0 for value in shape):
                raise ArchitectureAuditError(f"negative tensor dimension for {key}")
            tensors[key] = {"dtype": str(entry["dtype"]), "shape": shape}
            shard_for_key[key] = path.name

    if weight_map is not None:
        if set(weight_map) != set(tensors):
            missing_header = sorted(set(weight_map) - set(tensors))
            missing_index = sorted(set(tensors) - set(weight_map))
            raise ArchitectureAuditError(
                "index/header key mismatch: "
                f"missing_header={missing_header[:5]}, missing_index={missing_index[:5]}"
            )
        wrong_shards = [
            key for key, shard in weight_map.items() if shard_for_key[key] != shard
        ]
        if wrong_shards:
            raise ArchitectureAuditError(
                f"index points to wrong shard for {wrong_shards[:5]}"
            )

    dtype_parameters: Counter[str] = Counter()
    total_parameters = 0
    for entry in tensors.values():
        parameters = math.prod(entry["shape"])
        total_parameters += parameters
        dtype_parameters[entry["dtype"]] += parameters

    schema: dict[str, dict[str, Any]] = {}
    layer_ids: set[int] = set()
    schema_accumulator: dict[str, list[tuple[list[int], str]]] = defaultdict(list)
    non_layer_keys: list[str] = []
    for key, entry in sorted(tensors.items()):
        match = LAYER_KEY.match(key)
        if match:
            layer_ids.add(int(match.group(1)))
            schema_accumulator[match.group(2)].append((entry["shape"], entry["dtype"]))
        else:
            non_layer_keys.append(key)
    for suffix, entries in sorted(schema_accumulator.items()):
        unique = sorted({(tuple(shape), dtype) for shape, dtype in entries})
        schema[suffix] = {
            "count": len(entries),
            "signatures": [
                {"shape": list(shape), "dtype": dtype} for shape, dtype in unique
            ],
        }

    key_digest_payload = [
        [key, entry["dtype"], entry["shape"]] for key, entry in sorted(tensors.items())
    ]
    shape_digest_payload = [
        [key, entry["shape"]] for key, entry in sorted(tensors.items())
    ]
    return {
        "tensor_count": len(tensors),
        "parameter_count_serialized": total_parameters,
        "parameters_by_dtype": dict(sorted(dtype_parameters.items())),
        "layer_ids": sorted(layer_ids),
        "layer_schema": schema,
        "non_layer_keys": non_layer_keys,
        "representative_tensors": {
            key: tensors[key] for key in REPRESENTATIVE_TENSORS if key in tensors
        },
        "bias_keys": sorted(key for key in tensors if key.endswith(".bias")),
        "state_dict_key_layout_sha256": canonical_sha256(key_digest_payload),
        "state_dict_shape_layout_sha256": canonical_sha256(shape_digest_payload),
        "state_dict_tensor_layout": [
            {"key": key, "dtype": entry["dtype"], "shape": entry["shape"]}
            for key, entry in sorted(tensors.items())
        ],
        "state_dict_keys": sorted(tensors),
        "shard_metadata": metadata,
        "_tensors": tensors,
    }


def _safe_library_metadata(model_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": sys.version.split()[0],
    }
    try:
        import torch
        import transformers
        from transformers import AutoConfig, AutoTokenizer

        result["torch_version"] = torch.__version__
        result["transformers_version"] = transformers.__version__
        config = AutoConfig.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False
        )
        result["config_class"] = type(config).__name__
        result["resolved_attention_implementation"] = getattr(
            config, "_attn_implementation", None
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False
        )
        result["tokenizer_class"] = type(tokenizer).__name__
        result["special_token_ids"] = {
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "unk_token_id": tokenizer.unk_token_id,
        }
        try:
            from transformers.models.qwen3.modeling_qwen3 import (
                Qwen3Attention,
                Qwen3RMSNorm,
            )

            result["attention_class"] = (
                f"{Qwen3Attention.__module__}.{Qwen3Attention.__name__}"
            )
            result["normalization_class"] = (
                f"{Qwen3RMSNorm.__module__}.{Qwen3RMSNorm.__name__}"
            )
        except Exception as error:  # pragma: no cover - version-specific metadata
            result["implementation_introspection_error"] = repr(error)
    except Exception as error:
        result["loader_error"] = repr(error)
    return result


def _tokenizer_report(model_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for name in TOKENIZER_FILES:
        path = model_path / name
        if path.is_file():
            files[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    tokenizer_config_path = model_path / "tokenizer_config.json"
    tokenizer_config = (
        _read_json(tokenizer_config_path) if tokenizer_config_path.is_file() else {}
    )
    chat_template = tokenizer_config.get("chat_template")
    special_ids = {
        key: config.get(key)
        for key in ("bos_token_id", "eos_token_id", "pad_token_id")
    }
    return {
        "files": files,
        "bundle_sha256": bundle_sha256(
            (name, entry["sha256"]) for name, entry in files.items()
        ),
        "tokenizer_class_declared": tokenizer_config.get("tokenizer_class"),
        "special_token_ids_from_config": special_ids,
        "chat_template_sha256": (
            sha256_text(chat_template) if isinstance(chat_template, str) else None
        ),
    }


def _verify_tied_embeddings(
    model_path: Path, config: dict[str, Any], tensors: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    tied = bool(config.get("tie_word_embeddings"))
    embed_key = "model.embed_tokens.weight"
    head_key = "lm_head.weight"
    if not tied:
        return {"configured_tied": False, "status": "not_tied"}
    if embed_key not in tensors:
        return {
            "configured_tied": True,
            "status": "failed",
            "error": f"missing {embed_key}",
        }
    if head_key not in tensors:
        return {
            "configured_tied": True,
            "status": "verified_implicit",
            "serialized_head": False,
            "equal": True,
        }
    if tensors[embed_key]["shape"] != tensors[head_key]["shape"]:
        return {
            "configured_tied": True,
            "status": "failed",
            "serialized_head": True,
            "equal": False,
            "error": "embedding and lm_head shapes differ",
        }
    from .checkpoint import CheckpointReader

    with CheckpointReader(model_path) as reader:
        embedding = reader.get_tensor(embed_key)
        head = reader.get_tensor(head_key)
        equal = bool(torch.equal(embedding, head))
        maximum_absolute_difference = (
            0.0
            if equal
            else float((embedding.float() - head.float()).abs().max().item())
        )
    return {
        "configured_tied": True,
        "status": "verified_redundant" if equal else "failed",
        "serialized_head": True,
        "equal": equal,
        "maximum_absolute_difference": maximum_absolute_difference,
    }


def _verify_release_manifest(
    model_path: Path, manifest_path: Path | None = None
) -> dict[str, Any] | None:
    manifest_path = manifest_path or (model_path / "FILES.json")
    if not manifest_path.is_file():
        return None
    manifest = _read_json(manifest_path)
    records: dict[str, Any] = {}
    failures: list[str] = []
    for relative, expected in sorted(manifest.get("files", {}).items()):
        path = model_path / relative
        if not path.is_file():
            records[relative] = {"status": "missing"}
            failures.append(f"missing:{relative}")
            continue
        actual_bytes = path.stat().st_size
        actual_hash = sha256_file(path)
        status = (
            "verified"
            if actual_bytes == expected.get("bytes")
            and actual_hash == expected.get("sha256")
            else "mismatch"
        )
        records[relative] = {
            "status": status,
            "bytes": actual_bytes,
            "sha256": actual_hash,
            "expected_bytes": expected.get("bytes"),
            "expected_sha256": expected.get("sha256"),
        }
        if status != "verified":
            failures.append(f"mismatch:{relative}")
    return {
        "manifest": str(manifest_path.resolve()),
        "override": manifest_path.resolve() != (model_path / "FILES.json").resolve(),
        "status": "verified" if not failures else "failed",
        "failures": failures,
        "files": records,
    }


def inspect_checkpoint(
    *,
    name: str,
    role: str,
    source_id: str,
    revision: str,
    model_path: Path,
    hash_weights: bool = True,
    release_manifest_path: Path | None = None,
) -> dict[str, Any]:
    model_path = model_path.resolve()
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise ArchitectureAuditError(f"missing config.json: {model_path}")
    config = _read_json(config_path)
    weight_files, weight_map = resolve_weight_files(model_path)
    tensor_report = inspect_tensor_headers(weight_files, weight_map)
    tensors = tensor_report.pop("_tensors")
    tied_verification = _verify_tied_embeddings(model_path, config, tensors)
    effective_parameters = tensor_report["parameter_count_serialized"]
    if tied_verification.get("status") == "verified_redundant":
        effective_parameters -= math.prod(tensors["lm_head.weight"]["shape"])
    tensor_report["parameter_count_effective"] = effective_parameters

    weight_records = []
    for path in weight_files:
        record = {"name": path.name, "bytes": path.stat().st_size}
        if hash_weights:
            record["sha256"] = sha256_file(path)
        weight_records.append(record)

    index_path = model_path / "model.safetensors.index.json"
    index_record = None
    if index_path.is_file():
        index_record = {
            "bytes": index_path.stat().st_size,
            "sha256": sha256_file(index_path),
            "mapped_tensor_count": len(weight_map or {}),
        }

    hidden_size = config.get("hidden_size")
    q_heads = config.get("num_attention_heads")
    kv_heads = config.get("num_key_value_heads")
    head_dim = config.get("head_dim")
    inferred_head_dim = (
        hidden_size // q_heads
        if isinstance(hidden_size, int)
        and isinstance(q_heads, int)
        and q_heads
        and hidden_size % q_heads == 0
        else None
    )
    embedding = tensors.get("model.embed_tokens.weight")
    lm_head = tensors.get("lm_head.weight")
    if lm_head is None and config.get("tie_word_embeddings"):
        lm_head = embedding
        lm_head_storage = "tied_to_model.embed_tokens.weight"
    else:
        lm_head_storage = "serialized" if lm_head is not None else "missing"

    return {
        "schema_version": "1.0",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "name": name,
        "role": role,
        "source_id": source_id,
        "revision": revision,
        "resolved_path": str(model_path),
        "config": {
            "sha256": sha256_file(config_path),
            "architectures": config.get("architectures"),
            "model_type": config.get("model_type"),
            "hidden_size": hidden_size,
            "intermediate_size": config.get("intermediate_size"),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "num_attention_heads": q_heads,
            "num_key_value_heads": kv_heads,
            "head_dim": head_dim,
            "head_dim_inferred": inferred_head_dim,
            "vocab_size": config.get("vocab_size"),
            "attention_bias": config.get("attention_bias"),
            "attention_dropout": config.get("attention_dropout"),
            "hidden_act": config.get("hidden_act"),
            "rms_norm_eps": config.get("rms_norm_eps"),
            "rope_theta": config.get("rope_theta")
            or (config.get("rope_parameters") or {}).get("rope_theta"),
            "rope_scaling": config.get("rope_scaling"),
            "rope_parameters": config.get("rope_parameters"),
            "max_position_embeddings": config.get("max_position_embeddings"),
            "sliding_window": config.get("sliding_window"),
            "use_sliding_window": config.get("use_sliding_window"),
            "tie_word_embeddings": config.get("tie_word_embeddings"),
            "torch_dtype": config.get("torch_dtype") or config.get("dtype"),
            "auto_map": config.get("auto_map"),
            "abstain_token_id": config.get("abstain_token_id"),
            "abstain_logit_bias": config.get("abstain_logit_bias"),
            "raw": config,
        },
        "library": _safe_library_metadata(model_path),
        "tokenizer": _tokenizer_report(model_path, config),
        "weights": {
            "files": weight_records,
            "index": index_record,
            **tensor_report,
        },
        "embedding_shape": embedding["shape"] if embedding else None,
        "output_head_shape": lm_head["shape"] if lm_head else None,
        "output_head_storage": lm_head_storage,
        "tied_embedding_verification": tied_verification,
        "release_manifest_verification": _verify_release_manifest(
            model_path,
            release_manifest_path.resolve() if release_manifest_path else None,
        ),
    }


def uniform_depth_map(target_layers: int, donor_layers: int) -> list[int]:
    if target_layers <= 0 or donor_layers <= 0:
        raise ValueError("layer counts must be positive")
    if target_layers == 1:
        return [0]
    return [
        math.floor(index * (donor_layers - 1) / (target_layers - 1) + 0.5)
        for index in range(target_layers)
    ]


def center_preserving_depth_map(target_layers: int, donor_layers: int) -> list[int]:
    """Preserve edge quarters 1:1; compress only the donor middle trajectory."""
    if donor_layers < target_layers:
        raise ValueError("center-preserving mapping requires donor depth >= target")
    if donor_layers == target_layers:
        return list(range(target_layers))
    edge = max(1, math.ceil(target_layers / 4))
    middle_target = target_layers - 2 * edge
    if middle_target <= 0:
        return uniform_depth_map(target_layers, donor_layers)
    first = list(range(edge))
    donor_middle_start = edge
    donor_middle_end = donor_layers - edge - 1
    if middle_target == 1:
        middle = [math.floor((donor_middle_start + donor_middle_end) / 2 + 0.5)]
    else:
        middle = [
            math.floor(
                donor_middle_start
                + index
                * (donor_middle_end - donor_middle_start)
                / (middle_target - 1)
                + 0.5
            )
            for index in range(middle_target)
        ]
    last = list(range(donor_layers - edge, donor_layers))
    result = first + middle + last
    if len(result) != target_layers or result != sorted(result):
        raise AssertionError("invalid center-preserving mapping")
    return result


def depth_mappings(target_layers: int, donor_layers: int) -> dict[str, Any]:
    return {
        "D1_prefix": {
            "rule": "target i maps to donor i",
            "mapping": list(range(target_layers)),
        },
        "D2_uniform_full_depth": {
            "rule": "floor(i*(Ld-1)/(Ls-1)+0.5)",
            "mapping": uniform_depth_map(target_layers, donor_layers),
        },
        "D3_center_preserving": {
            "rule": (
                "preserve ceil(Ls/4) early and late layers one-to-one; "
                "uniformly compress the donor middle into the target middle"
            ),
            "edge_layers": max(1, math.ceil(target_layers / 4)),
            "mapping": center_preserving_depth_map(target_layers, donor_layers),
        },
    }


def compatibility_report(audits: dict[str, dict[str, Any]]) -> dict[str, Any]:
    required = {"qwen06", "supa", "qwen4b", "qwen8b"}
    if set(audits) != required:
        raise ArchitectureAuditError(
            f"expected audits {sorted(required)}, received {sorted(audits)}"
        )
    target = audits["supa"]
    reference = audits["qwen06"]
    target_config = target["config"]
    target_keys = target["weights"]["state_dict_keys"]
    reference_keys = reference["weights"]["state_dict_keys"]
    def canonical_shapes(audit: dict[str, Any]) -> dict[str, tuple[int, ...]]:
        result = {
            entry["key"]: tuple(entry["shape"])
            for entry in audit["weights"]["state_dict_tensor_layout"]
        }
        if audit["config"].get("tie_word_embeddings"):
            result.pop("lm_head.weight", None)
        return result

    exact_small_shape_match = canonical_shapes(target) == canonical_shapes(reference)

    donors: dict[str, Any] = {}
    global_fatal: list[str] = []
    for donor_name in ("qwen4b", "qwen8b"):
        donor = audits[donor_name]
        donor_config = donor["config"]
        fatal: list[str] = []
        notes: list[str] = []
        for field in ("model_type", "vocab_size", "head_dim"):
            if donor_config.get(field) != target_config.get(field):
                fatal.append(
                    f"{field} mismatch: target={target_config.get(field)}, "
                    f"donor={donor_config.get(field)}"
                )
        if donor["tokenizer"]["bundle_sha256"] != target["tokenizer"]["bundle_sha256"]:
            fatal.append("tokenizer bundle mismatch")
        if donor_config["num_attention_heads"] < target_config["num_attention_heads"]:
            fatal.append("donor has fewer query heads than target")
        if donor_config["num_key_value_heads"] < target_config["num_key_value_heads"]:
            fatal.append("donor has fewer KV heads than target")
        if donor_config["num_hidden_layers"] < target_config["num_hidden_layers"]:
            fatal.append("donor has fewer layers than target")
        if donor_config["hidden_size"] > target_config["hidden_size"]:
            notes.append("residual channels require deterministic selection/projection")
        if donor_config["intermediate_size"] > target_config["intermediate_size"]:
            notes.append("MLP channels require deterministic selection/projection")
        if donor_config["num_attention_heads"] > target_config["num_attention_heads"]:
            notes.append("query heads require complete-head selection")
        if donor_config["num_key_value_heads"] == target_config["num_key_value_heads"]:
            notes.append("all KV heads can be retained; residual input width still projects")
        tied_projection_policies: list[str] = []
        if target_config.get("tie_word_embeddings") and not donor_config.get(
            "tie_word_embeddings"
        ):
            notes.append(
                "donor output head is untied but target is tied; projection must "
                "declare embedding, output_head, or mean collapse policy"
            )
            tied_projection_policies = ["embedding", "output_head", "mean"]
        mappings = depth_mappings(
            int(target_config["num_hidden_layers"]),
            int(donor_config["num_hidden_layers"]),
        )
        donors[donor_name] = {
            "projection_go": not fatal,
            "fatal_issues": fatal,
            "notes": notes,
            "tied_projection_policies": tied_projection_policies,
            "depth_mappings": mappings,
            "layer_schema_names_equal": (
                set(target["weights"]["layer_schema"])
                == set(donor["weights"]["layer_schema"])
            ),
        }
        global_fatal.extend(f"{donor_name}: {item}" for item in fatal)

    if not exact_small_shape_match:
        global_fatal.append("Supa and Qwen3-0.6B state-dict layouts differ")
    for name, audit in audits.items():
        tied_status = audit["tied_embedding_verification"].get("status")
        if audit["config"].get("tie_word_embeddings") and not str(tied_status).startswith(
            "verified_"
        ):
            global_fatal.append(f"{name}: tied embedding contract failed verification")
    target_manifest = target.get("release_manifest_verification")
    if not target_manifest or target_manifest.get("status") != "verified":
        global_fatal.append("Supa release manifest did not verify")
    return {
        "schema_version": "1.0",
        "small_reference": {
            "exact_state_dict_layout_match": exact_small_shape_match,
            "target_only_keys": sorted(set(target_keys) - set(reference_keys)),
            "reference_only_keys": sorted(set(reference_keys) - set(target_keys)),
            "weight_space_note": (
                "Supa-A1.1's abstention offset is implemented in forward and is not "
                "stored in the state dict; weight-space comparisons use the ungated "
                "dense checkpoint and report the gate as a separate behavioral control."
            ),
        },
        "tokenizer_bundle_equal_all_models": len(
            {audit["tokenizer"]["bundle_sha256"] for audit in audits.values()}
        )
        == 1,
        "donors": donors,
        "projection_gate": "GO" if not global_fatal else "NO_GO",
        "fatal_issues": global_fatal,
    }


def render_compatibility_markdown(
    audits: dict[str, dict[str, Any]], compatibility: dict[str, Any]
) -> str:
    lines = [
        "# Architecture Compatibility Report",
        "",
        f"Projection gate: **{compatibility['projection_gate']}**",
        "",
        "## Verified architecture",
        "",
        "| Model | Class | Layers | Hidden | MLP | Q heads | KV heads | Head dim | Vocab | Stored dtype | Effective parameters |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for name in ("qwen06", "supa", "qwen4b", "qwen8b"):
        audit = audits[name]
        config = audit["config"]
        lines.append(
            "| {name} | {architecture} | {layers} | {hidden} | {mlp} | {q} | "
            "{kv} | {head_dim} | {vocab} | {dtype} | {parameters} |".format(
                name=name,
                architecture=", ".join(config.get("architectures") or []),
                layers=config.get("num_hidden_layers"),
                hidden=config.get("hidden_size"),
                mlp=config.get("intermediate_size"),
                q=config.get("num_attention_heads"),
                kv=config.get("num_key_value_heads"),
                head_dim=config.get("head_dim"),
                vocab=config.get("vocab_size"),
                dtype=config.get("torch_dtype"),
                parameters=audit["weights"]["parameter_count_effective"],
            )
        )
    lines.extend(
        [
            "",
            "All values above come from checkpoint configs and safetensor headers, not model-card assumptions.",
            "",
            "## Small-model identity",
            "",
            f"- Exact Supa/Qwen3-0.6B state-dict layout match: `{compatibility['small_reference']['exact_state_dict_layout_match']}`",
            f"- Tokenizer bundle equal across all four models: `{compatibility['tokenizer_bundle_equal_all_models']}`",
            f"- Supa embedding shape: `{audits['supa']['embedding_shape']}`",
            f"- Supa output-head shape: `{audits['supa']['output_head_shape']}` ({audits['supa']['output_head_storage']})",
            f"- Official 0.6B tied-head verification: `{audits['qwen06']['tied_embedding_verification']['status']}`",
            f"- Supa release manifest verification: `{audits['supa']['release_manifest_verification']['status']}`",
            "- Supa-A1.1 abstention is a non-weight forward logit offset. It is excluded from parameter deltas and retained only as a separately labeled behavioral control.",
            "",
            "## Projection semantics",
            "",
            "Safetensor matrices follow the PyTorch linear layout `[out_features, in_features]`. W2 retains complete query/KV heads along attention output-feature axes, then applies the declared residual-channel selection along hidden axes. Embeddings, norms, MLP channels, and output projections use the same deterministic residual-channel index set. Tied output embeddings remain tied.",
            "",
        ]
    )
    for donor_name in ("qwen4b", "qwen8b"):
        donor = compatibility["donors"][donor_name]
        lines.extend(
            [
                f"### {donor_name}",
                "",
                f"Projection GO: `{donor['projection_go']}`",
                "",
            ]
        )
        for note in donor["notes"]:
            lines.append(f"- {note}")
        for issue in donor["fatal_issues"]:
            lines.append(f"- FATAL: {issue}")
        if donor["tied_projection_policies"]:
            lines.append(
                "- Required tied-collapse policy: "
                + ", ".join(f"`{value}`" for value in donor["tied_projection_policies"])
            )
        lines.extend(["", "Depth mappings:", ""])
        for mapping_name, mapping in donor["depth_mappings"].items():
            lines.append(f"- `{mapping_name}`: `{mapping['mapping']}`")
        lines.append("")
    lines.extend(
        [
            "## Numerical and loader notes",
            "",
            "- Merge arithmetic must cast source tensors to FP32 before any projection or combination.",
            "- Official donors are BF16. Supa-A1.1 is stored FP16; storage-precision comparisons are separate finalist controls.",
            "- Architecture audit reads only headers. Functional GO still requires a full load, finite forward pass, deterministic generation, EOS behavior, and exact post-serialization validation.",
            "- Runtime attention kernels may vary by Transformers/device. Each run manifest must record the resolved implementation.",
        ]
    )
    if compatibility["fatal_issues"]:
        lines.extend(["", "## Fatal issues", ""])
        lines.extend(f"- {issue}" for issue in compatibility["fatal_issues"])
    return "\n".join(lines)
