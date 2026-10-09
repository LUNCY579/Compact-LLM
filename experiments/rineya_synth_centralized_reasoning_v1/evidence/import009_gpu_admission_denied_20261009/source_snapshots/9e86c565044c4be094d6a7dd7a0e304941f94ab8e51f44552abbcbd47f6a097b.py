"""Callable-only strict Titan loader; no framework imports or launch authority.

The caller injects Torch/Qwen3 classes, an attempt-accounting ``call(operation,
function)`` wrapper and a ``guard(stage)`` resource/deadline check. A single
2 MiB writable CPU buffer supplies blocking BF16 copies into meta-constructed
CUDA storage. Device checksums also visit at most 2 MiB at a time, without an
FP32 conversion, CPU model, complete CPU tensor copy or memory mapping.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import struct

VERSION = "synth-titan-stream-meta-2MiB-1"
CHUNK_BYTES = 2 * 1024 * 1024
MAX_HEADER_BYTES = 1024 * 1024
SERIALIZED_TENSORS = 310
BASE_PARAMETERS = 596049920
MODEL_DIR = Path("C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/"
                 "Rineya-Titan-Experimental-1.0.0-devfreeze")
ASSET_PINS = {
    "config.json": "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
    "model.safetensors": "537201676814ade103c4efa63f5ebe758ea309971803c4ad633dbca5b6b531b7",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "generation_config.json": "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
}
DIMENSIONS = {"hidden_size": 1024, "intermediate_size": 3072, "num_hidden_layers": 28,
              "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128,
              "vocab_size": 151936, "max_position_embeddings": 40960}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def _invalid_number(value):
    raise ValueError("noninteger/nonfinite safetensors header number")


def _read_exact_into(stream, view):
    position = 0
    while position < len(view):
        count = stream.readinto(view[position:])
        _require(type(count) is int and 0 < count <= len(view) - position,
                 "truncated safetensors header/payload")
        position += count


def _read_header(stream, file_bytes):
    prefix = bytearray(8)
    _read_exact_into(stream, memoryview(prefix))
    header_bytes = struct.unpack("<Q", prefix)[0]
    _require(2 <= header_bytes <= MAX_HEADER_BYTES and header_bytes <= file_bytes - 8,
             "invalid or oversized safetensors header length")
    raw = bytearray(header_bytes)
    _read_exact_into(stream, memoryview(raw))
    try:
        entries = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object,
                             parse_float=_invalid_number, parse_constant=_invalid_number)
    except (UnicodeError, ValueError, RecursionError) as error:
        raise ValueError("invalid safetensors header JSON: " + str(error)) from error
    _require(type(entries) is dict and bool(entries), "safetensors header must be a nonempty object")
    metadata = entries.pop("__metadata__", {})
    _require(type(metadata) is dict and all(type(k) is str and type(v) is str for k, v in metadata.items()),
             "invalid safetensors metadata")
    _require(bool(entries), "safetensors header has no tensors")
    payload_bytes = file_bytes - 8 - header_bytes
    intervals = []
    for name, entry in entries.items():
        _require(type(name) is str and bool(name) and name == name.strip(), "invalid tensor name")
        _require(type(entry) is dict and set(entry) == {"dtype", "shape", "data_offsets"},
                 "invalid tensor header fields: " + name)
        shape, offsets = entry["shape"], entry["data_offsets"]
        _require(entry["dtype"] == "BF16", "source tensor must be BF16: " + name)
        _require(type(shape) is list and bool(shape) and
                 all(type(dim) is int and dim > 0 for dim in shape), "invalid tensor shape: " + name)
        _require(type(offsets) is list and len(offsets) == 2 and
                 all(type(value) is int and value >= 0 for value in offsets), "invalid tensor offsets: " + name)
        begin, end = offsets
        _require(begin <= end <= payload_bytes and end - begin == math.prod(shape) * 2,
                 "tensor byte span mismatch: " + name)
        intervals.append((begin, end, name))
    previous_end = 0
    for begin, end, name in sorted(intervals):
        _require(begin == previous_end, "overlapping or gapped safetensors payload: " + name)
        previous_end = end
    _require(previous_end == payload_bytes, "unclaimed safetensors payload bytes")
    return {"entries": entries, "metadata": metadata, "header_bytes": header_bytes,
            "payload_bytes": payload_bytes, "file_bytes": file_bytes}, prefix, raw


def read_header(path):
    """Validate bounded header metadata and exact BF16 spans without reading tensors."""
    path = Path(path)
    with path.open("rb", buffering=0) as stream:
        return _read_header(stream, path.stat().st_size)[0]


def _source_shapes():
    result = {"model.embed_tokens.weight": [151936, 1024], "model.norm.weight": [1024]}
    shapes = {"input_layernorm": [1024], "post_attention_layernorm": [1024],
              "mlp.down_proj": [1024, 3072], "mlp.gate_proj": [3072, 1024],
              "mlp.up_proj": [3072, 1024], "self_attn.q_proj": [2048, 1024],
              "self_attn.k_proj": [1024, 1024], "self_attn.v_proj": [1024, 1024],
              "self_attn.o_proj": [1024, 2048], "self_attn.q_norm": [128], "self_attn.k_norm": [128]}
    for layer in range(28):
        for suffix, shape in shapes.items():
            result[f"model.layers.{layer}.{suffix}.weight"] = list(shape)
    return result


def _adapter_shapes():
    return {f"model.layers.{layer}.self_attn.{projection}.lora_{factor}": shape
            for layer in range(28)
            for projection, factor, shape in (("q_proj", "A", [8, 1024]), ("q_proj", "B", [2048, 8]),
                                               ("v_proj", "A", [8, 1024]), ("v_proj", "B", [1024, 8]))}


def _asset_hash(path, guard):
    digest, total = hashlib.sha256(), 0
    buffer = bytearray(CHUNK_BYTES)
    with Path(path).open("rb", buffering=0) as stream:
        while True:
            guard("asset_hash_before:" + Path(path).name)
            count = stream.readinto(buffer)
            _require(type(count) is int and 0 <= count <= CHUNK_BYTES, "invalid asset read")
            if count:
                digest.update(memoryview(buffer)[:count])
                total += count
            guard("asset_hash_after:" + Path(path).name)
            if not count:
                break
    return {"sha256": digest.hexdigest(), "bytes": total}


def _validate_config(config):
    for field, expected in DIMENSIONS.items():
        _require(type(getattr(config, field, None)) is int and getattr(config, field) == expected,
                 "normalized configuration drift: " + field)
    _require(getattr(config, "model_type", None) == "qwen3" and
             getattr(config, "tie_word_embeddings", None) is True and
             getattr(config, "attention_bias", None) is False and
             getattr(config, "use_sliding_window", None) is False and
             getattr(config, "sliding_window", None) is None and
             getattr(config, "hidden_act", None) == "silu" and
             getattr(config, "rms_norm_eps", None) == 1e-6 and
             getattr(config, "attention_dropout", None) == 0.0,
             "normalized attention/activation/tie configuration drift")
    rope = getattr(config, "rope_parameters", None)
    _require(type(rope) is dict and set(rope) == {"rope_type", "rope_theta"} and
             rope["rope_type"] == "default" and type(rope["rope_theta"]) in (int, float) and
             rope["rope_theta"] == 1000000, "normalized RoPE configuration drift")
    layers = getattr(config, "layer_types", None)
    _require(type(layers) is list and layers == ["full_attention"] * 28,
             "normalized attention layer types drift")
    _require(getattr(config, "_attn_implementation", None) == "eager", "eager attention required")
    return {**DIMENSIONS, "model_type": "qwen3", "tie_word_embeddings": True,
            "rope_parameters": dict(rope), "layer_types": list(layers), "attention_implementation": "eager"}


def _validate_rope(torch, model):
    rotary = model.model.rotary_emb
    buffers = dict(rotary.named_buffers())
    _require(set(buffers) == {"inv_freq", "original_inv_freq"} and
             set(rotary._non_persistent_buffers_set) == set(buffers),
             "RoPE buffers must be exactly the two nonpersistent current Qwen3 buffers")
    _require(getattr(rotary, "rope_type", None) == "default" and
             getattr(rotary, "attention_scaling", None) == 1.0 and
             getattr(rotary, "max_seq_len_cached", None) == 40960 and
             getattr(rotary, "original_max_seq_len", None) == 40960,
             "RoPE runtime metadata drift")
    for name, value in buffers.items():
        _require(value.dtype == torch.float32 and value.device.type == "cuda" and value.device.index == 0 and
                 list(value.shape) == [64] and not value.requires_grad and
                 bool(torch.isfinite(value).all()), "invalid resident RoPE buffer: " + name)
    _require(torch.equal(buffers["inv_freq"], buffers["original_inv_freq"]), "RoPE original/current frequencies differ")
    expected = 1.0 / (1000000 ** (torch.arange(0, 128, 2, dtype=torch.float32, device="cuda:0") / 128))
    _require(torch.equal(buffers["inv_freq"], expected), "RoPE frequencies differ from the frozen default formula")
    _require(not any("rotary_emb" in name for name in model.state_dict()), "RoPE buffer became persistent")
    return {name: {"shape": [64], "dtype": "float32", "device": "cuda:0", "persistent": False}
            for name in buffers}


def _base_inventory(torch, model, *, allow_adapters):
    pairs = list(model.named_parameters(remove_duplicate=False))
    named = dict(pairs)
    _require(len(pairs) == len(named), "duplicate parameter names")
    shapes = _source_shapes()
    aliases = set(shapes) | {"lm_head.weight"}
    extras = set(named) - aliases
    _require(aliases <= set(named) and
             (not extras or (allow_adapters and extras == set(_adapter_shapes()))),
             "resident base/adapter parameter key mismatch")
    for name, shape in shapes.items():
        parameter = named[name]
        _require(list(parameter.shape) == shape and parameter.numel() == math.prod(shape) and
                 parameter.dtype == torch.bfloat16 and parameter.device.type == "cuda" and
                 parameter.device.index == 0 and not parameter.requires_grad and parameter.grad is None and
                 parameter.is_contiguous(), "resident base tensor mismatch: " + name)
    parameters = [named[name] for name in shapes]
    _require(len(parameters) == SERIALIZED_TENSORS and len({id(p) for p in parameters}) == SERIALIZED_TENSORS and
             len({p.data_ptr() for p in parameters}) == SERIALIZED_TENSORS and
             sum(p.numel() for p in parameters) == BASE_PARAMETERS, "resident unique base inventory mismatch")
    _require(model.lm_head.weight is model.model.embed_tokens.weight and
             named["lm_head.weight"] is named["model.embed_tokens.weight"] and
             model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr(), "embedding/head tie lost")
    for name, shape in _adapter_shapes().items():
        if name in extras:
            parameter = named[name]
            _require(list(parameter.shape) == shape and parameter.dtype == torch.float32 and
                     parameter.device.type == "cuda" and parameter.device.index == 0 and parameter.requires_grad,
                     "resident adapter mismatch: " + name)
    return {name: named[name] for name in shapes}


def _tensor_digest(torch, tensor, guard, name):
    digest = hashlib.sha256()
    flat = tensor.detach().reshape(-1)
    for offset in range(0, flat.numel(), CHUNK_BYTES // 2):
        guard("device_digest_before:" + name)
        # ndarray exposes the same bounded CPU allocation through the buffer protocol.
        part = flat[offset:offset + CHUNK_BYTES // 2].contiguous().view(torch.uint8).cpu().numpy()
        view = memoryview(part)
        _require(view.nbytes <= CHUNK_BYTES and view.c_contiguous, "unbounded/noncontiguous device digest")
        digest.update(view)
        del view, part
        guard("device_digest_after:" + name)
    return digest.hexdigest()


def verify_frozen_base(torch, model, source_hashes, call, guard):
    """Checksum all source tensors, permitting only the frozen q/v LoRA inventory.

    Content checks do not authorize inference/training or establish capability.
    No source file is opened by this verification operation.
    """
    _require(callable(call) and callable(guard), "accounting call and guard required")
    _require(type(source_hashes) is dict and set(source_hashes) == set(_source_shapes()) and
             all(type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) for value in source_hashes.values()),
             "source tensor digest inventory mismatch")
    guard("verify_frozen_base_before")
    base = _base_inventory(torch, model, allow_adapters=True)
    for name, parameter in base.items():
        _require(_tensor_digest(torch, parameter, guard, name) == source_hashes[name],
                 "frozen base tensor bytes changed: " + name)
    buffers = _validate_rope(torch, model)
    guard("verify_frozen_base_after")
    return {"verified_source_tensors": len(base), "unique_base_parameters": BASE_PARAMETERS,
            "base_dtype": "bfloat16", "embedding_head_tie": True, "frozen_base_unchanged": True,
            "device_digest_maximum_chunk_bytes": CHUNK_BYTES, "rotary_buffers": buffers}


def load_titan(torch, config_class, model_class, model_dir, call, guard):
    """Return ``(model, receipt, source_hashes)`` after strict pinned loading.

    This function grants no launch permission. Its caller must separately satisfy
    the repository's source, runtime, host, resource and explicit launch gates.
    """
    _require(callable(call) and callable(guard), "accounting call and guard required")
    path = Path(model_dir)
    _require(path.resolve() == MODEL_DIR.resolve(), "unregistered Titan model directory")
    guard("load_titan_before")
    assets = {}
    for name, expected in ASSET_PINS.items():
        assets[name] = _asset_hash(path / name, guard)
        _require(assets[name]["sha256"] == expected, "source asset SHA256 drift: " + name)
    header = read_header(path / "model.safetensors")
    shapes = _source_shapes()
    _require(len(header["entries"]) == SERIALIZED_TENSORS and set(header["entries"]) == set(shapes) and
             all(header["entries"][name]["shape"] == shape for name, shape in shapes.items()),
             "serialized Titan tensor inventory mismatch")
    guard("config_load_before")
    config = config_class.from_pretrained(str(path), local_files_only=True)
    config._attn_implementation = "eager"
    normalized = _validate_config(config)
    guard("config_load_after")

    def construct():
        guard("model_construct_before")
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.bfloat16)
            with torch.device("meta"):
                model = model_class(config)
        finally:
            torch.set_default_dtype(previous_dtype)
        _require(all(p.device.type == "meta" and p.dtype == torch.bfloat16 for p in model.parameters()),
                 "construction allocated resident/non-BF16 parameters")
        guard("model_to_empty_before")
        model.to_empty(device="cuda:0")
        # to_empty visits modules independently and can break tied object identity.
        model.tie_weights()
        _require(model.lm_head.weight is model.model.embed_tokens.weight, "post-allocation embedding tie lost")
        # Meta-created nonpersistent buffers have no data to restore from weights.
        model.model.rotary_emb = type(model.model.rotary_emb)(config, device="cuda:0")
        model.eval().requires_grad_(False)
        _base_inventory(torch, model, allow_adapters=False)
        _validate_rope(torch, model)
        guard("model_construct_after")
        return model

    model = call("model_construct", construct)
    source_hashes = {}
    diagnostics = {"loader_mode": VERSION, "read_buffer_bytes": CHUNK_BYTES,
                   "maximum_payload_read_bytes": 0, "payload_chunk_count": 0,
                   "memory_mapped": False, "cpu_model_allocated": False,
                   "FP32_base_copy_allocated": False, "blocking_chunk_copies": True}

    def checkpoint_load():
        guard("checkpoint_load_before")
        states = model.state_dict()
        _require(set(states) - set(shapes) == {"lm_head.weight"} and not set(shapes) - set(states),
                 "only absent serialized key must be tied lm_head.weight")
        buffer = bytearray(CHUNK_BYTES)
        full_digest = hashlib.sha256()
        with (path / "model.safetensors").open("rb", buffering=0) as stream, torch.no_grad():
            current, prefix, raw_header = _read_header(stream, (path / "model.safetensors").stat().st_size)
            _require(current == header, "source header changed after preflight")
            full_digest.update(prefix)
            full_digest.update(raw_header)
            for name in sorted(shapes, key=lambda name: current["entries"][name]["data_offsets"][0]):
                begin, end = current["entries"][name]["data_offsets"]
                target = states[name].detach().reshape(-1)

                def copy_tensor():
                    source_digest, offset = hashlib.sha256(), 0
                    _require(stream.tell() == 8 + current["header_bytes"] + begin, "source stream position drift")
                    while offset < end - begin:
                        guard("tensor_copy_before:" + name)
                        count = min(CHUNK_BYTES, end - begin - offset)
                        view = memoryview(buffer)[:count]
                        _read_exact_into(stream, view)
                        source_digest.update(view)
                        full_digest.update(view)
                        source = torch.frombuffer(view, dtype=torch.bfloat16)
                        target[offset // 2:(offset + count) // 2].copy_(source, non_blocking=False)
                        del source, view
                        diagnostics["payload_chunk_count"] += 1
                        diagnostics["maximum_payload_read_bytes"] = max(diagnostics["maximum_payload_read_bytes"], count)
                        offset += count
                        guard("tensor_copy_after:" + name)
                    return source_digest.hexdigest()

                expected = call("tensor_copy", copy_tensor)
                _require(_tensor_digest(torch, states[name], guard, name) == expected,
                         "loaded tensor differs from serialized BF16 bytes: " + name)
                source_hashes[name] = expected
        _require(full_digest.hexdigest() == ASSET_PINS["model.safetensors"], "source weights changed after preflight")
        _require((path / "model.safetensors").stat().st_size == header["file_bytes"],
                 "source weight size changed during loading")
        _base_inventory(torch, model, allow_adapters=False)
        guard("checkpoint_load_after")

    call("checkpoint_load", checkpoint_load)
    receipt = {"loader": VERSION, "qualification": "EXACT_LOAD_ONLY_NOT_NATIVE_NUMERICS_OR_CAPABILITY",
               "model_dir": str(path.resolve()), "asset_hashes": assets,
               "normalized_config": normalized, "rotary_buffers": _validate_rope(torch, model),
               "unique_base_tensors": SERIALIZED_TENSORS, "unique_base_parameters": BASE_PARAMETERS,
               "serialized_tensors": len(source_hashes), "base_dtype": "bfloat16", "base_frozen": True,
               "embedding_head_tie": True, "source_tensor_sha256": dict(source_hashes),
               "header_bytes": header["header_bytes"], "payload_bytes": header["payload_bytes"],
               "file_bytes": header["file_bytes"], "device_digest_maximum_chunk_bytes": CHUNK_BYTES,
               **diagnostics}
    guard("load_titan_after")
    return model, receipt, source_hashes
