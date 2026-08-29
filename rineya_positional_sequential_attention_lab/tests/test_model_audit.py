"""Tests for Phase 1.

Nothing here reads a real checkpoint: the panel is ~6.5 GB and hashing it twice
would put minutes into the fast gate. Instead every test builds tiny synthetic
safetensors directories in ``tmp_path`` with a 4-query-head / 2-KV-head / 2-layer
architecture, which exercises exactly the same code paths as Qwen3-0.6B.

The synthetic architecture is deliberately chosen so ``n_kv * head_dim != hidden``
(4 != 6). Qwen3-0.6B has ``n_kv * head_dim == hidden == 1024``, i.e. its k_proj and
v_proj are square and their orientation is genuinely unmeasurable from shape alone;
``test_head_layout_square_kv_inherits_convention`` pins that case separately with a
Qwen3-shaped ``Arch``.

The single real-checkpoint test is marked ``model`` and skipped unless
``PSA_MODEL_TESTS=1``.
"""

from __future__ import annotations

import json
import os
import struct
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from rineya_psa.model_audit import distances as dist
from rineya_psa.model_audit import hashing as H
from rineya_psa.model_audit import inspect_ckpt as I
from rineya_psa.model_audit import run_audit as R
from rineya_psa.types import Arch, ModelSpec

# --------------------------------------------------------------------------- #
# synthetic checkpoint fixtures
# --------------------------------------------------------------------------- #

TINY = Arch(
    num_layers=2, hidden_size=6, num_q_heads=4, num_kv_heads=2, head_dim=2, vocab_size=10
)
INTERMEDIATE = 12

QWEN3 = Arch(
    num_layers=28,
    hidden_size=1024,
    num_q_heads=16,
    num_kv_heads=8,
    head_dim=128,
    vocab_size=151936,
)

_NP_TO_ST = {
    np.dtype("<f4"): "F32",
    np.dtype("<f2"): "F16",
    np.dtype("<f8"): "F64",
}


def write_safetensors(
    path: Path, tensors: Mapping[str, np.ndarray], metadata: Mapping[str, str] | None = None
) -> dict[str, bytes]:
    """Minimal safetensors writer. Returns the raw payload of each tensor.

    Written by hand rather than via the ``safetensors`` package so the test
    validates the reader against the on-disk format, not against the same library
    that produced the file.
    """
    header: dict[str, object] = {}
    payloads: dict[str, bytes] = {}
    offset = 0
    for name in sorted(tensors):
        arr = np.ascontiguousarray(tensors[name])
        st = _NP_TO_ST[arr.dtype.newbyteorder("<")]
        raw = arr.tobytes()
        header[name] = {
            "dtype": st,
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        payloads[name] = raw
        offset += len(raw)
    if metadata:
        header["__metadata__"] = dict(metadata)
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        for name in sorted(tensors):
            fh.write(payloads[name])
    return payloads


def write_bf16_safetensors(path: Path, tensors: Mapping[str, np.ndarray]) -> None:
    """Store float32 arrays as bfloat16 (top 16 bits, truncated)."""
    header: dict[str, object] = {}
    payloads: dict[str, bytes] = {}
    offset = 0
    for name in sorted(tensors):
        arr = np.ascontiguousarray(tensors[name], dtype="<f4")
        raw = (arr.view("<u4") >> 16).astype("<u2").tobytes()
        header[name] = {
            "dtype": "BF16",
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        payloads[name] = raw
        offset += len(raw)
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        for name in sorted(tensors):
            fh.write(payloads[name])


def tiny_state_dict(
    arch: Arch, seed: int, *, tie_head: bool, dtype: str = "<f4"
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)

    def r(*shape: int) -> np.ndarray:
        return rng.standard_normal(shape).astype(dtype)

    sd: dict[str, np.ndarray] = {
        "model.embed_tokens.weight": r(arch.vocab_size, arch.hidden_size),
        "model.norm.weight": r(arch.hidden_size),
    }
    for layer in range(arch.num_layers):
        p = f"model.layers.{layer}"
        sd[f"{p}.self_attn.q_proj.weight"] = r(arch.num_q_heads * arch.head_dim, arch.hidden_size)
        sd[f"{p}.self_attn.k_proj.weight"] = r(arch.num_kv_heads * arch.head_dim, arch.hidden_size)
        sd[f"{p}.self_attn.v_proj.weight"] = r(arch.num_kv_heads * arch.head_dim, arch.hidden_size)
        sd[f"{p}.self_attn.o_proj.weight"] = r(arch.hidden_size, arch.num_q_heads * arch.head_dim)
        sd[f"{p}.self_attn.q_norm.weight"] = r(arch.head_dim)
        sd[f"{p}.self_attn.k_norm.weight"] = r(arch.head_dim)
        sd[f"{p}.mlp.gate_proj.weight"] = r(INTERMEDIATE, arch.hidden_size)
        sd[f"{p}.mlp.up_proj.weight"] = r(INTERMEDIATE, arch.hidden_size)
        sd[f"{p}.mlp.down_proj.weight"] = r(arch.hidden_size, INTERMEDIATE)
        sd[f"{p}.input_layernorm.weight"] = r(arch.hidden_size)
        sd[f"{p}.post_attention_layernorm.weight"] = r(arch.hidden_size)
    if tie_head:
        sd["lm_head.weight"] = sd["model.embed_tokens.weight"].copy()
    return sd


def flat_config(arch: Arch, **over: object) -> dict[str, object]:
    """The transformers-4 spelling: flat rope_theta / rope_scaling / torch_dtype."""
    cfg: dict[str, object] = {
        "architectures": ["Qwen3ForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "head_dim": arch.head_dim,
        "hidden_act": "silu",
        "hidden_size": arch.hidden_size,
        "initializer_range": 0.02,
        "intermediate_size": INTERMEDIATE,
        "max_position_embeddings": 4096,
        "max_window_layers": arch.num_layers,
        "model_type": "qwen3",
        "num_attention_heads": arch.num_q_heads,
        "num_hidden_layers": arch.num_layers,
        "num_key_value_heads": arch.num_kv_heads,
        "rms_norm_eps": 1e-06,
        "rope_scaling": None,
        "rope_theta": 1000000,
        "sliding_window": None,
        "tie_word_embeddings": True,
        "torch_dtype": "float32",
        "transformers_version": "4.51.0",
        "use_cache": True,
        "use_sliding_window": False,
        "vocab_size": arch.vocab_size,
    }
    cfg.update(over)
    return cfg


def modern_config(arch: Arch, **over: object) -> dict[str, object]:
    """The transformers-5 spelling: rope_parameters dict, ``dtype``, ``layer_types``."""
    cfg = flat_config(arch)
    for k in ("rope_theta", "rope_scaling", "torch_dtype"):
        cfg.pop(k)
    cfg["rope_parameters"] = {"rope_theta": 1000000, "rope_type": "default"}
    cfg["dtype"] = "float32"
    cfg["layer_types"] = ["full_attention"] * arch.num_layers
    cfg["transformers_version"] = "5.14.1"
    cfg["pad_token_id"] = None
    cfg.update(over)
    return cfg


TOKENIZER_JSON = {
    "version": "1.0",
    "added_tokens": [
        {"id": 1, "content": "<|start|>", "special": True},
        {"id": 2, "content": "<|end|>", "special": True},
    ],
    "model": {"type": "BPE", "vocab": {"a": 0, "b": 1}, "merges": []},
}
CHAT_TEMPLATE = "{% for m in messages %}{{ m.content }}{% endfor %}"


def write_tokenizer(directory: Path, *, modern: bool) -> None:
    (directory / "tokenizer.json").write_text(
        json.dumps(TOKENIZER_JSON, separators=(",", ":")), encoding="utf-8"
    )
    (directory / "vocab.json").write_text('{"a":0,"b":1}', encoding="utf-8")
    (directory / "merges.txt").write_text("#version: 0.2\na b\n", encoding="utf-8")
    if modern:
        # transformers 5 spelling: no added_tokens_decoder, template in its own file,
        # extra_special_tokens instead of additional_special_tokens, loader noise.
        cfg = {
            "backend": "tokenizers",
            "bos_token": None,
            "eos_token": "<|end|>",
            "extra_special_tokens": ["<|end|>", "<|start|>"],
            "is_local": True,
            "local_files_only": False,
            "model_max_length": 4096,
            "pad_token": "<|end|>",
            "tokenizer_class": "Qwen2Tokenizer",
            "unk_token": None,
        }
        (directory / "chat_template.jinja").write_text(CHAT_TEMPLATE + "\n", encoding="utf-8")
    else:
        cfg = {
            "added_tokens_decoder": {
                "1": {"content": "<|start|>", "special": True},
                "2": {"content": "<|end|>", "special": True},
            },
            "additional_special_tokens": ["<|start|>", "<|end|>"],
            "bos_token": None,
            "chat_template": CHAT_TEMPLATE,
            "eos_token": {"content": "<|end|>", "special": True},
            "model_max_length": 4096,
            "pad_token": "<|end|>",
            "tokenizer_class": "Qwen2Tokenizer",
            "unk_token": None,
        }
    (directory / "tokenizer_config.json").write_text(
        json.dumps(cfg, indent=2), encoding="utf-8"
    )


def make_ckpt(
    root: Path,
    key: str,
    *,
    state_dict: Mapping[str, np.ndarray],
    config: Mapping[str, object],
    modern_tokenizer: bool = False,
    bf16: bool = False,
    extra_files: Mapping[str, str] | None = None,
) -> Path:
    directory = root / key
    directory.mkdir(parents=True, exist_ok=True)
    if bf16:
        write_bf16_safetensors(directory / "model.safetensors", state_dict)
    else:
        write_safetensors(directory / "model.safetensors", state_dict, {"format": "pt"})
    (directory / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (directory / "generation_config.json").write_text('{"do_sample": false}', encoding="utf-8")
    write_tokenizer(directory, modern=modern_tokenizer)
    for name, text in (extra_files or {}).items():
        (directory / name).write_text(text, encoding="utf-8")
    return directory


def spec_for(key: str, directory: Path, **over: object) -> ModelSpec:
    defaults: dict[str, object] = {
        "key": key,
        "display_name": key,
        "path": directory,
        "store_dtype": "float32",
        "trust_remote_code": False,
        "expected_arch": "Qwen3ForCausalLM",
        "ungatable": False,
    }
    defaults.update(over)
    return ModelSpec(**defaults)  # type: ignore[arg-type]


@pytest.fixture
def panel(tmp_path: Path) -> dict[str, ModelSpec]:
    """A four-model synthetic panel mirroring the real one's schema differences."""
    root = tmp_path / "ckpts"
    base_sd = tiny_state_dict(TINY, seed=0, tie_head=True)

    # base: transformers-4 spelling, materialised tied lm_head
    make_ckpt(root, "base", state_dict=base_sd, config=flat_config(TINY))

    # twin: identical config and schema, different weights
    twin_sd = tiny_state_dict(TINY, seed=1, tie_head=True)
    make_ckpt(root, "twin", state_dict=twin_sd, config=flat_config(TINY))

    # modern: transformers-5 spelling, lm_head omitted
    modern_sd = tiny_state_dict(TINY, seed=2, tie_head=False)
    make_ckpt(
        root, "modern", state_dict=modern_sd, config=modern_config(TINY), modern_tokenizer=True
    )

    # gated: modern spelling plus a forward-time abstain gate and custom code
    gated_sd = tiny_state_dict(TINY, seed=3, tie_head=False)
    make_ckpt(
        root,
        "gated",
        state_dict=gated_sd,
        config=modern_config(
            TINY,
            architectures=["Qwen3AbstainForCausalLM"],
            auto_map={"AutoModelForCausalLM": "modeling_tiny.TinyAbstainForCausalLM"},
            abstain_token_id=7,
            abstain_logit_bias=5.15625,
        ),
        modern_tokenizer=True,
        extra_files={"modeling_tiny.py": "class TinyAbstainForCausalLM: pass\n"},
    )

    return {
        "base": spec_for("base", root / "base"),
        "twin": spec_for("twin", root / "twin"),
        "modern": spec_for("modern", root / "modern"),
        "gated": spec_for(
            "gated",
            root / "gated",
            trust_remote_code=True,
            expected_arch="Qwen3AbstainForCausalLM",
            ungatable=True,
        ),
    }


# --------------------------------------------------------------------------- #
# hashing
# --------------------------------------------------------------------------- #


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    import hashlib

    p = tmp_path / "blob.bin"
    data = os.urandom(3 * H.CHUNK_BYTES + 17)
    p.write_bytes(data)
    assert H.sha256_file(p, chunk_size=4096) == hashlib.sha256(data).hexdigest()


def test_one_pass_hashing_matches_independent_digests(tmp_path: Path) -> None:
    import hashlib

    tensors = {
        "a": np.arange(12, dtype="<f4").reshape(3, 4),
        "b": np.arange(6, dtype="<f2").reshape(2, 3),
        "c": np.array([1.5], dtype="<f4"),
    }
    p = tmp_path / "model.safetensors"
    payloads = write_safetensors(p, tensors, {"format": "pt"})

    file_sha, per_tensor = H.hash_safetensors_file(p, chunk_size=8)
    assert file_sha == hashlib.sha256(p.read_bytes()).hexdigest()
    for name, raw in payloads.items():
        assert per_tensor[name] == hashlib.sha256(raw).hexdigest()

    locs = H.tensor_locations(p)
    for name in tensors:
        assert H.sha256_tensor(locs[name]) == per_tensor[name]
        assert locs[name].shape == tensors[name].shape
        assert locs[name].n_params == tensors[name].size


def test_header_metadata_and_offsets(tmp_path: Path) -> None:
    p = tmp_path / "m.safetensors"
    write_safetensors(p, {"x": np.zeros((2, 2), dtype="<f4")}, {"format": "pt", "note": "hi"})
    entries, meta, start = H.read_safetensors_header(p)
    assert meta == {"format": "pt", "note": "hi"}
    assert "__metadata__" not in entries
    assert entries["x"]["data_offsets"] == [0, 16]
    assert start == p.stat().st_size - 16


def test_bf16_decode_is_exact(tmp_path: Path) -> None:
    """bf16 is the top 16 bits of a float32; widening must be lossless."""
    vals = np.array([0.0, -0.0, 1.0, -2.5, 3.5, 1e-6, 65536.0, -1.0 / 3], dtype="<f4")
    truncated = (vals.view("<u4") & np.uint32(0xFFFF0000)).view("<f4")
    p = tmp_path / "bf.safetensors"
    write_bf16_safetensors(p, {"w": vals})
    loc = H.tensor_locations(p)["w"]
    mm = np.memmap(p, dtype=np.uint8, mode="r")
    got = H.decode_to_float64(np.asarray(mm[loc.begin : loc.end]), "BF16")
    np.testing.assert_array_equal(got, truncated.astype(np.float64))


def test_decode_rejects_unknown_dtype() -> None:
    with pytest.raises(ValueError, match="cannot decode"):
        H.decode_to_float64(np.zeros(4, dtype=np.uint8), "F8_E4M3")


def test_tokenizer_identity_normalises_spelling_differences(tmp_path: Path) -> None:
    """The two transformers spellings of the same tokenizer must hash the same."""
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir()
    new.mkdir()
    write_tokenizer(old, modern=False)
    write_tokenizer(new, modern=True)

    a, b = H.tokenizer_identity(old), H.tokenizer_identity(new)
    assert a["config_normalized"] == b["config_normalized"]
    assert a["identity_sha256"] == b["identity_sha256"]
    assert a["chat_template_location"] == "tokenizer_config.json"
    assert b["chat_template_location"] == "chat_template.jinja"
    # Volatile loader keys must not survive into the hashed view.
    assert "is_local" not in b["config_normalized"]
    assert "backend" not in b["config_normalized"]
    # added_tokens is recovered from tokenizer.json when the config omits it.
    assert b["config_normalized"]["added_tokens"] == [[1, "<|start|>"], [2, "<|end|>"]]


def test_tokenizer_identity_detects_a_real_difference(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    write_tokenizer(a, modern=False)
    write_tokenizer(b, modern=False)
    (b / "merges.txt").write_text("#version: 0.2\nb a\n", encoding="utf-8")
    ia, ib = H.tokenizer_identity(a), H.tokenizer_identity(b)
    assert ia["identity_sha256"] != ib["identity_sha256"]
    assert ia["vocab_merges_sha256"] != ib["vocab_merges_sha256"]


def test_json_diff_paths() -> None:
    assert H.json_diff_paths({"a": 1, "b": {"c": 2}}, {"a": 1, "b": {"c": 2}}) == []
    assert H.json_diff_paths({"a": 1}, {"a": 2}) == ["a"]
    assert H.json_diff_paths({"b": {"c": 2}}, {"b": {"c": 3}}) == ["b.c"]
    assert H.json_diff_paths({"a": 1}, {}) == ["a"]
    assert H.json_diff_paths({"a": [1, 2]}, {"a": [1, 3]}) == ["a[1]"]


def test_directory_fingerprint_ignores_hf_cache(tmp_path: Path) -> None:
    d = tmp_path / "d"
    (d / ".cache" / "huggingface").mkdir(parents=True)
    (d / "config.json").write_text("{}", encoding="utf-8")
    before = H.directory_fingerprint(d)
    (d / ".cache" / "huggingface" / "download.json").write_text("{}", encoding="utf-8")
    assert H.directory_fingerprint(d) == before
    assert [f["name"] for f in H.hash_directory(d)] == ["config.json"]


# --------------------------------------------------------------------------- #
# config normalisation and pair classification
# --------------------------------------------------------------------------- #


def test_rope_and_dtype_schemas_normalise_to_the_same_view() -> None:
    old = I.normalize_config(flat_config(TINY))
    new = I.normalize_config(modern_config(TINY))
    assert old == new, H.json_diff_paths(old, new)
    assert old["rope_theta"] == 1000000.0
    assert old["rope_type"] == "default"
    assert old["rope_scaling"] is None
    assert old["store_dtype"] == "float32"
    assert old["layer_types_unique"] == ["full_attention"]

    assert I.config_schema_report(flat_config(TINY))["rope_schema"] == "flat"
    assert I.config_schema_report(modern_config(TINY))["rope_schema"] == "rope_parameters"
    assert I.config_schema_report(flat_config(TINY))["dtype_field"] == "torch_dtype"
    assert I.config_schema_report(modern_config(TINY))["dtype_field"] == "dtype"


def test_rope_scaling_survives_normalisation() -> None:
    scaled = I.normalize_config(
        modern_config(
            TINY,
            rope_parameters={"rope_theta": 1e6, "rope_type": "yarn", "factor": 4.0},
        )
    )
    assert scaled["rope_type"] == "yarn"
    assert scaled["rope_scaling"] == {"factor": 4.0}
    assert scaled != I.normalize_config(flat_config(TINY))


def _audit(spec: ModelSpec) -> I.CheckpointAudit:
    return I.audit_checkpoint(spec.key, spec=spec, probe_transformers=False)


def test_audit_records_every_required_field(panel: dict[str, ModelSpec]) -> None:
    a = _audit(panel["base"])
    assert a.path == str(panel["base"].path.resolve())
    assert a.model_files == ["model.safetensors"]
    assert {f["name"] for f in a.files} >= {
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    }
    assert all(len(f["sha256"]) == 64 for f in a.files)

    n_per_layer = 11
    assert a.tensor_count == 2 + TINY.num_layers * n_per_layer + 1  # +lm_head
    assert a.tensor_dtypes == ["F32"]
    assert a.architecture_class == "Qwen3ForCausalLM"
    assert (a.hidden_size, a.intermediate_size, a.num_hidden_layers) == (6, INTERMEDIATE, 2)
    assert (a.num_attention_heads, a.num_key_value_heads, a.head_dim) == (4, 2, 2)
    assert a.vocab_size == TINY.vocab_size
    assert a.max_position_embeddings == 4096
    assert a.rope_theta == 1000000.0
    assert a.rope_scaling is None
    assert a.tie_word_embeddings is True
    assert a.lm_head_present is True
    assert a.lm_head_is_embedding_copy is True
    assert a.attention_bias_declared is False
    assert a.bias_tensors == []
    assert a.custom_code_files == []
    assert a.gate is None
    assert a.spec_store_dtype_matches is True
    assert a.generation_config == {"do_sample": False}
    assert a.arch() == TINY

    # param accounting: the tied head is counted once, and separately from the rest
    embed = TINY.vocab_size * TINY.hidden_size
    assert a.param_count_total == a.param_count_unique + embed
    assert a.param_count_embedding == embed
    assert a.param_count_non_embedding == a.param_count_unique - embed

    # GQA
    assert a.gqa["kv_group_size"] == 2
    assert a.gqa["q_heads_for_kv"]["1"] == [2, 3]
    assert a.gqa["shapes_match_config"] is True
    assert a.gqa["layer0_projection_shapes"]["o_proj"] == [6, 8]

    # norms, including the Qwen3-style per-head q/k norm over head_dim
    assert a.norms["has_qk_norm"] is True
    assert a.norms["q_norm_shape"] == [TINY.head_dim]
    assert a.norms["qk_norm_over_head_dim"] is True
    assert a.norms["final_norm_over_hidden"] is True
    assert a.norms["weight_only"] is True

    assert a.tokenizer["identity_sha256"]
    assert a.attn_implementation["lab_default"] == "sdpa"
    assert a.warnings == []


def test_audit_records_the_abstain_gate_and_custom_code(panel: dict[str, ModelSpec]) -> None:
    a = _audit(panel["gated"])
    assert a.architecture_class == "Qwen3AbstainForCausalLM"
    assert a.trust_remote_code_required is True
    assert a.auto_map == {"AutoModelForCausalLM": "modeling_tiny.TinyAbstainForCausalLM"}
    assert [f["name"] for f in a.custom_code_files] == ["modeling_tiny.py"]
    assert len(a.custom_code_files[0]["sha256"]) == 64
    assert a.gate is not None
    assert a.gate["abstain_token_id"] == 7
    assert a.gate["abstain_logit_bias"] == pytest.approx(5.15625)
    assert a.gate["modeling_files"] == a.custom_code_files
    assert a.lm_head_present is False
    assert a.param_count_total == a.param_count_unique


def test_audit_round_trips_through_json(panel: dict[str, ModelSpec]) -> None:
    a = _audit(panel["gated"])
    b = I.CheckpointAudit.from_json(json.loads(json.dumps(a.to_json())))
    assert b == a


def test_audit_warns_when_config_and_spec_disagree(panel: dict[str, ModelSpec]) -> None:
    bad = spec_for("base", panel["base"].path, expected_arch="LlamaForCausalLM")
    a = I.audit_checkpoint("base", spec=bad, probe_transformers=False)
    assert any("LlamaForCausalLM" in w for w in a.warnings)


def test_pair_classification_ladder(panel: dict[str, ModelSpec], tmp_path: Path) -> None:
    audits = {k: _audit(v) for k, v in panel.items()}

    # byte-identical: a literal copy of the base directory
    import shutil

    clone_dir = tmp_path / "ckpts" / "clone"
    shutil.copytree(panel["base"].path, clone_dir)
    clone = I.audit_checkpoint("clone", spec=spec_for("clone", clone_dir), probe_transformers=False)
    pc = I.classify_pair(audits["base"], clone)
    assert pc.pair_class == "byte_identical"
    assert pc.weights_identical and pc.configs_identical and pc.delta_comparable

    # config-identical: same schema and config, different weights
    pc = I.classify_pair(audits["base"], audits["twin"])
    assert pc.pair_class == "config_identical"
    assert not pc.weights_identical
    assert pc.configs_identical and pc.schema_identical
    assert pc.n_shared_tensors == audits["base"].tensor_count

    # architecture-compatible: only the tied lm_head convention differs
    pc = I.classify_pair(audits["base"], audits["modern"])
    assert pc.pair_class == "architecture_compatible"
    assert pc.only_in_a == ["lm_head.weight"]
    assert pc.tied_lm_head_convention is True
    assert pc.configs_identical is True  # the two spellings normalise to the same view
    assert pc.delta_comparable

    # architecture-compatible with a behavioural config difference (the gate)
    pc = I.classify_pair(audits["base"], audits["gated"])
    assert pc.pair_class == "architecture_compatible"
    assert pc.architectures_differ is True
    assert set(pc.behavioural_config_diff) >= {
        "abstain_logit_bias",
        "abstain_token_id",
        "architectures[0]",
        "auto_map",
    }

    # same-shape-different-weights: an extra tensor beyond the tied-head convention
    odd_sd = tiny_state_dict(TINY, seed=9, tie_head=True)
    odd_sd["model.layers.0.self_attn.rotary_emb.inv_freq"] = np.ones(
        TINY.head_dim // 2, dtype="<f4"
    )
    make_ckpt(tmp_path / "ckpts", "odd", state_dict=odd_sd, config=flat_config(TINY))
    odd = I.audit_checkpoint(
        "odd", spec=spec_for("odd", tmp_path / "ckpts" / "odd"), probe_transformers=False
    )
    pc = I.classify_pair(audits["base"], odd)
    assert pc.pair_class == "same_shape_different_weights"
    assert pc.delta_comparable

    # incompatible: a different architecture entirely
    big = Arch(
        num_layers=2, hidden_size=8, num_q_heads=4, num_kv_heads=2, head_dim=2, vocab_size=10
    )
    make_ckpt(
        tmp_path / "ckpts",
        "big",
        state_dict=tiny_state_dict(big, seed=4, tie_head=True),
        config=flat_config(big),
    )
    other = I.audit_checkpoint(
        "big", spec=spec_for("big", tmp_path / "ckpts" / "big"), probe_transformers=False
    )
    pc = I.classify_pair(audits["base"], other)
    assert pc.pair_class == "incompatible"
    assert not pc.delta_comparable

    assert set(I.PAIR_CLASSES) == {
        "byte_identical",
        "config_identical",
        "architecture_compatible",
        "same_shape_different_weights",
        "incompatible",
    }


def test_tokenizer_comparison_across_the_panel(panel: dict[str, ModelSpec]) -> None:
    audits = {k: _audit(v) for k, v in panel.items()}
    pc = I.classify_pair(audits["base"], audits["modern"])
    assert pc.tokenizer_identical is True
    assert pc.tokenizer_vocab_identical is True
    assert pc.tokenizer_diff == []


def test_tokenizer_diff_names_the_field_that_moved(
    panel: dict[str, ModelSpec], tmp_path: Path
) -> None:
    """The real panel differs in tokenizer.json's ByteLevel decoder flags only.

    Reporting that as a path beats reporting it as a hash mismatch, so the diff is
    exercised on exactly that field.
    """
    import copy

    tweaked = copy.deepcopy(TOKENIZER_JSON)
    tweaked["decoder"] = {"type": "ByteLevel", "add_prefix_space": True}
    directory = make_ckpt(
        tmp_path / "ckpts",
        "reserialised",
        state_dict=tiny_state_dict(TINY, seed=0, tie_head=True),
        config=flat_config(TINY),
    )
    (directory / "tokenizer.json").write_text(
        json.dumps(tweaked, separators=(",", ":")), encoding="utf-8"
    )
    other = I.audit_checkpoint(
        "reserialised", spec=spec_for("reserialised", directory), probe_transformers=False
    )
    pc = I.classify_pair(_audit(panel["base"]), other)
    assert pc.tokenizer_identical is False
    assert pc.tokenizer_vocab_identical is True  # the vocabulary itself is untouched
    assert pc.tokenizer_diff == ["tokenizer.json:decoder"]


# --------------------------------------------------------------------------- #
# per-head decomposition (pure functions, hand-computed)
# --------------------------------------------------------------------------- #


def test_head_layout_axes_are_measured_not_assumed() -> None:
    shapes = {
        "model.layers.0.self_attn.q_proj.weight": (8, 6),
        "model.layers.0.self_attn.k_proj.weight": (4, 6),
        "model.layers.0.self_attn.v_proj.weight": (4, 6),
        "model.layers.0.self_attn.o_proj.weight": (6, 8),
    }
    rep = dist.resolve_orientation(shapes, TINY)
    assert rep.stored_out_first is True
    assert set(rep.evidence) == {"q_proj", "k_proj", "v_proj", "o_proj"}
    assert rep.ambiguous == ()

    assert dist.head_layout("q_proj", (8, 6), TINY).axis == 0
    assert dist.head_layout("k_proj", (4, 6), TINY).axis == 0
    assert dist.head_layout("v_proj", (4, 6), TINY).axis == 0
    assert dist.head_layout("o_proj", (6, 8), TINY).axis == 1
    assert dist.head_layout("q_proj", (8, 6), TINY).n_heads == 4
    assert dist.head_layout("k_proj", (4, 6), TINY).n_heads == 2
    assert dist.head_layout("gate_proj", (12, 6), TINY) is None


def test_head_layout_detects_a_transposed_store() -> None:
    transposed = {
        "model.layers.0.self_attn.q_proj.weight": (6, 8),
        "model.layers.0.self_attn.o_proj.weight": (8, 6),
    }
    rep = dist.resolve_orientation(transposed, TINY)
    assert rep.stored_out_first is False
    assert dist.head_layout("q_proj", (6, 8), TINY, stored_out_first=False).axis == 1
    assert dist.head_layout("o_proj", (8, 6), TINY, stored_out_first=False).axis == 0


def test_head_layout_rejects_contradictory_shapes() -> None:
    contradictory = {
        "model.layers.0.self_attn.q_proj.weight": (8, 6),  # says out-first
        "model.layers.0.self_attn.o_proj.weight": (8, 6),  # says transposed
    }
    with pytest.raises(ValueError, match="mutually inconsistent"):
        dist.resolve_orientation(contradictory, TINY)
    with pytest.raises(ValueError, match="head axis at"):
        dist.head_layout("q_proj", (8, 6), TINY, stored_out_first=False)
    with pytest.raises(ValueError, match="neither axis"):
        dist.head_layout("q_proj", (7, 6), TINY)


def test_head_layout_square_kv_inherits_convention() -> None:
    """Qwen3-0.6B's k_proj is [1024,1024]: shape alone cannot say which axis is heads."""
    assert QWEN3.num_kv_heads * QWEN3.head_dim == QWEN3.hidden_size
    shapes = {
        "model.layers.0.self_attn.q_proj.weight": (2048, 1024),
        "model.layers.0.self_attn.k_proj.weight": (1024, 1024),
        "model.layers.0.self_attn.v_proj.weight": (1024, 1024),
        "model.layers.0.self_attn.o_proj.weight": (1024, 2048),
    }
    rep = dist.resolve_orientation(shapes, QWEN3)
    assert rep.stored_out_first is True
    assert set(rep.evidence) == {"q_proj", "o_proj"}
    assert set(rep.ambiguous) == {"k_proj", "v_proj"}

    k = dist.head_layout("k_proj", (1024, 1024), QWEN3)
    assert (k.axis, k.n_heads, k.head_dim, k.space, k.evidence) == (0, 8, 128, "kv", "inherited")
    q = dist.head_layout("q_proj", (2048, 1024), QWEN3)
    assert (q.axis, q.n_heads, q.space, q.evidence) == (0, 16, "q", "measured")
    o = dist.head_layout("o_proj", (1024, 2048), QWEN3)
    assert (o.axis, o.n_heads, o.space, o.evidence) == (1, 16, "q", "measured")


def test_per_head_split_against_a_hand_computed_example(tmp_path: Path) -> None:
    """q_proj rows and o_proj columns, with numbers worked out by hand.

    ``q_proj`` is 8x6 = 4 query heads of 2 rows. Base is all ones, so each head's
    base Frobenius norm is sqrt(2*6) = sqrt(12). Head h is perturbed by a constant
    ``h``, giving a delta norm of ``h*sqrt(12)`` and a relative Frobenius of exactly
    ``h``. ``o_proj`` is 6x8 = 4 heads of 2 columns and is perturbed the same way on
    its column blocks, so head h again has relative Frobenius exactly ``h``.
    """
    base_q = np.ones((8, 6), dtype="<f4")
    other_q = base_q.copy()
    for h in range(4):
        other_q[h * 2 : (h + 1) * 2, :] += h

    base_o = np.ones((6, 8), dtype="<f4")
    other_o = base_o.copy()
    for h in range(4):
        other_o[:, h * 2 : (h + 1) * 2] += h

    qn = "model.layers.0.self_attn.q_proj.weight"
    on = "model.layers.0.self_attn.o_proj.weight"
    bp, op = tmp_path / "b.safetensors", tmp_path / "o.safetensors"
    write_safetensors(bp, {qn: base_q, on: base_o})
    write_safetensors(op, {qn: other_q, on: other_o})
    bl, ol = H.tensor_locations(bp), H.tensor_locations(op)

    # A byte budget far below one head block, to force multi-chunk streaming and
    # prove the chunker never straddles a head boundary.
    with dist.MemmapPool() as pool:
        td_q = dist.tensor_delta(pool, bl[qn], ol[qn], arch=TINY, chunk_bytes=1)
        td_o = dist.tensor_delta(pool, bl[on], ol[on], arch=TINY, chunk_bytes=1)

    assert td_q.layout.axis == 0 and td_o.layout.axis == 1
    assert sorted(td_q.heads) == [0, 1, 2, 3]
    for h in range(4):
        mq = td_q.heads[h].metrics()
        assert mq["n_params"] == 12
        assert mq["base_frobenius"] == pytest.approx(np.sqrt(12.0))
        assert mq["frobenius"] == pytest.approx(h * np.sqrt(12.0))
        assert mq["rel_frobenius"] == pytest.approx(float(h))
        assert mq["max_abs_delta"] == pytest.approx(float(h))
        assert mq["frac_changed"] == pytest.approx(0.0 if h == 0 else 1.0)

        mo = td_o.heads[h].metrics()
        assert mo["n_params"] == 12
        assert mo["rel_frobenius"] == pytest.approx(float(h))
        assert mo["cosine"] == pytest.approx(1.0)  # ones scaled by (1+h) stays parallel

    # The whole-tensor norm must equal the quadrature sum of the head norms.
    total = np.sqrt(sum(td_q.heads[h].metrics()["frobenius"] ** 2 for h in range(4)))
    assert td_q.metrics()["frobenius"] == pytest.approx(total)


def test_kv_heads_split_into_the_right_number_of_blocks(tmp_path: Path) -> None:
    kn = "model.layers.0.self_attn.k_proj.weight"
    base = np.ones((4, 6), dtype="<f4")
    other = base.copy()
    other[2:4, :] = 3.0  # only KV head 1 moves
    bp, op = tmp_path / "b.safetensors", tmp_path / "o.safetensors"
    write_safetensors(bp, {kn: base})
    write_safetensors(op, {kn: other})
    with dist.MemmapPool() as pool:
        td = dist.tensor_delta(
            pool, H.tensor_locations(bp)[kn], H.tensor_locations(op)[kn], arch=TINY
        )
    assert td.layout.space == "kv"
    assert sorted(td.heads) == [0, 1]
    assert td.heads[0].metrics()["rel_frobenius"] == pytest.approx(0.0)
    assert td.heads[1].metrics()["rel_frobenius"] == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# streaming distances
# --------------------------------------------------------------------------- #


def test_streaming_metrics_match_a_dense_computation(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    base = rng.standard_normal((37, 13)).astype("<f4")
    other = base + rng.standard_normal((37, 13)).astype("<f4") * 0.01
    other[3, 4] = base[3, 4]  # one entry deliberately unchanged

    bp, op = tmp_path / "b.safetensors", tmp_path / "o.safetensors"
    write_safetensors(bp, {"model.norm.weight": base})
    write_safetensors(op, {"model.norm.weight": other})

    with dist.MemmapPool() as pool:
        td = dist.tensor_delta(
            pool,
            H.tensor_locations(bp)["model.norm.weight"],
            H.tensor_locations(op)["model.norm.weight"],
            chunk_bytes=64,  # forces many chunks
        )
    m = td.metrics()

    b64, o64 = base.astype(np.float64), other.astype(np.float64)
    d = o64 - b64
    assert m["n_params"] == b64.size
    assert m["frobenius"] == pytest.approx(np.linalg.norm(d), rel=1e-12)
    assert m["base_frobenius"] == pytest.approx(np.linalg.norm(b64), rel=1e-12)
    assert m["rel_frobenius"] == pytest.approx(np.linalg.norm(d) / np.linalg.norm(b64), rel=1e-12)
    assert m["cosine"] == pytest.approx(
        float(b64.ravel() @ o64.ravel() / (np.linalg.norm(b64) * np.linalg.norm(o64))), rel=1e-12
    )
    assert m["max_abs_delta"] == pytest.approx(np.abs(d).max(), rel=1e-12)
    assert m["rmse"] == pytest.approx(np.sqrt((d**2).mean()), rel=1e-12)
    assert m["frac_changed"] == pytest.approx((b64.size - 1) / b64.size)


def test_identical_tensors_give_zero_delta_and_unit_cosine(tmp_path: Path) -> None:
    w = np.linspace(-1, 1, 24, dtype="<f4").reshape(4, 6)
    bp, op = tmp_path / "b.safetensors", tmp_path / "o.safetensors"
    write_safetensors(bp, {"model.norm.weight": w})
    write_safetensors(op, {"model.norm.weight": w.copy()})
    with dist.MemmapPool() as pool:
        m = dist.tensor_delta(
            pool,
            H.tensor_locations(bp)["model.norm.weight"],
            H.tensor_locations(op)["model.norm.weight"],
        ).metrics()
    assert m["frobenius"] == 0.0
    assert m["rel_frobenius"] == 0.0
    assert m["cosine"] == pytest.approx(1.0)
    assert m["frac_changed"] == 0.0
    assert m["frac_changed_rel"] == 0.0


def test_zero_base_gives_nan_relative_norm_not_a_lie(tmp_path: Path) -> None:
    bp, op = tmp_path / "b.safetensors", tmp_path / "o.safetensors"
    write_safetensors(bp, {"model.norm.weight": np.zeros(4, dtype="<f4")})
    write_safetensors(op, {"model.norm.weight": np.ones(4, dtype="<f4")})
    with dist.MemmapPool() as pool:
        m = dist.tensor_delta(
            pool,
            H.tensor_locations(bp)["model.norm.weight"],
            H.tensor_locations(op)["model.norm.weight"],
        ).metrics()
    assert np.isnan(m["rel_frobenius"])
    assert np.isnan(m["cosine"])
    assert m["frobenius"] == pytest.approx(2.0)


def test_mixed_dtype_pair_is_compared_in_float64(tmp_path: Path) -> None:
    """A bf16 store vs a float32 store: the delta is the rounding, and it is exact."""
    vals = np.array([[0.1, 0.2, 0.3, 0.4]], dtype="<f4")
    bf, f32 = tmp_path / "bf.safetensors", tmp_path / "f32.safetensors"
    write_bf16_safetensors(bf, {"model.norm.weight": vals})
    write_safetensors(f32, {"model.norm.weight": vals})

    def measure(rel_tol: float) -> dict[str, float]:
        with dist.MemmapPool() as pool:
            return dist.tensor_delta(
                pool,
                H.tensor_locations(bf)["model.norm.weight"],
                H.tensor_locations(f32)["model.norm.weight"],
                rel_tol=rel_tol,
            ).metrics()

    truncated = (vals.view("<u4") & np.uint32(0xFFFF0000)).view("<f4").astype(np.float64)
    expected = np.linalg.norm(vals.astype(np.float64) - truncated)
    m = measure(dist.DEFAULT_REL_TOL)
    assert m["frobenius"] == pytest.approx(expected, rel=1e-12)
    assert m["frac_changed"] == 1.0
    # Truncating to bf16 costs up to 2^-8 relative, i.e. above the 1e-3 default, so
    # a bf16-vs-f32 pair legitimately reads as "everything moved" at that tolerance.
    assert m["frac_changed_rel"] == 1.0
    # Widen past the storage noise and the two files agree.
    assert measure(1e-2)["frac_changed_rel"] == 0.0


def test_pair_deltas_aggregates_and_reports_skips(panel: dict[str, ModelSpec]) -> None:
    base_locs = H.collect_tensor_locations(H.discover_model_files(panel["base"].path))
    other_locs = H.collect_tensor_locations(H.discover_model_files(panel["modern"].path))
    pd = dist.pair_deltas(
        base_locs, other_locs, base_key="base", other_key="modern", arch=TINY
    )
    assert pd.skipped == ["lm_head.weight"]
    assert pd.orientation is not None and pd.orientation.stored_out_first

    # every (layer, family) cell holds exactly one tensor for this architecture
    assert len(pd.by_layer_family) == len(pd.per_tensor)
    families = {f for _l, f in pd.by_layer_family}
    assert families == set(dist.FAMILIES)

    # head counts: q/o give 4 heads per layer, k/v give 2
    per_kind: dict[str, int] = {}
    for _layer, kind, _h in pd.by_head:
        per_kind[kind] = per_kind.get(kind, 0) + 1
    assert per_kind == {
        "q_proj": 4 * TINY.num_layers,
        "o_proj": 4 * TINY.num_layers,
        "k_proj": 2 * TINY.num_layers,
        "v_proj": 2 * TINY.num_layers,
    }

    # the overall accumulator is the quadrature sum of the per-tensor ones
    total = np.sqrt(sum(td.metrics()["frobenius"] ** 2 for td in pd.per_tensor))
    assert pd.overall_metrics()["frobenius"] == pytest.approx(total, rel=1e-12)


def test_pair_deltas_without_arch_skips_head_split(panel: dict[str, ModelSpec]) -> None:
    base_locs = H.collect_tensor_locations(H.discover_model_files(panel["base"].path))
    twin_locs = H.collect_tensor_locations(H.discover_model_files(panel["twin"].path))
    pd = dist.pair_deltas(base_locs, twin_locs, base_key="base", other_key="twin", arch=None)
    assert pd.by_head == {}
    assert pd.orientation is None
    assert pd.skipped == []


# --------------------------------------------------------------------------- #
# end-to-end runner
# --------------------------------------------------------------------------- #


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    import csv

    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


ARTIFACTS = (
    ("manifests", R.MANIFEST_NAME),
    ("manifests", R.TENSOR_HASHES_NAME),
    ("results", R.PAIRWISE_NAME),
    ("results", R.BY_LAYER_NAME),
    ("results", R.BY_HEAD_NAME),
)


def _run(panel: dict[str, ModelSpec], out: Path, **over: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "models": panel,
        "panel": ["base", "twin", "modern", "gated"],
        "base_key": "base",
        "out_root": out,
        "probe_transformers": False,
        "quiet": True,
    }
    kwargs.update(over)
    return R.run_audit(**kwargs)  # type: ignore[arg-type]


def test_run_audit_writes_every_artifact(panel: dict[str, ModelSpec], tmp_path: Path) -> None:
    out = tmp_path / "out"
    summary = _run(panel, out)
    assert summary["base_key"] == "base"

    for sub, name in ARTIFACTS:
        path = out / sub / name
        assert path.is_file(), name
        prov_path = R.provenance_path(path)
        assert prov_path.is_file(), prov_path.name
        prov = json.loads(prov_path.read_text(encoding="utf-8"))
        assert prov["artifact"] == name
        assert prov["artifact_sha256"] == H.sha256_bytes(path.read_bytes())
        assert prov["config_hash"] and prov["generated_at"] and prov["idempotency_key"]
        assert set(prov["model_sha256"]) == set(panel)
        assert prov["seeds"] == {}
        assert prov["stamp_source"]
        assert "git" in prov["stamp"]

    manifest = json.loads((out / "manifests" / R.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["base_key"] == "base"
    assert set(manifest["models"]) == set(panel)
    assert len(manifest["pairs"]) == 6
    assert manifest["models"]["gated"]["gate"]["abstain_token_id"] == 7
    assert manifest["head_decomposition"]["orientation_by_pair"]["twin"]["stored_out_first"]


def test_run_audit_pairwise_csv_covers_all_six_pairs(
    panel: dict[str, ModelSpec], tmp_path: Path
) -> None:
    out = tmp_path / "out"
    _run(panel, out)
    cols, rows = _read_csv(out / "results" / R.PAIRWISE_NAME)
    assert len(rows) == 6
    assert {(r["model_a"], r["model_b"]) for r in rows} == {
        ("base", "twin"),
        ("base", "modern"),
        ("base", "gated"),
        ("twin", "modern"),
        ("twin", "gated"),
        ("modern", "gated"),
    }
    # base-anchored pairs are emitted first and always with base in column A
    assert [r["model_a"] for r in rows[:3]] == ["base", "base", "base"]
    for r in rows:
        assert r["scope"] == R.PAIRWISE_SCOPE
        assert r["pair_class"] in I.PAIR_CLASSES
        assert float(r["rel_frobenius"]) >= 0.0
        assert -1.0 <= float(r["cosine"]) <= 1.0
        assert int(r["n_shared_tensors"]) > 0
    assert {"config_diff", "behavioural_config_diff", "tokenizer_identical"} <= set(cols)

    by_pair = {(r["model_a"], r["model_b"]): r for r in rows}
    assert by_pair[("base", "twin")]["pair_class"] == "config_identical"
    assert by_pair[("base", "modern")]["pair_class"] == "architecture_compatible"
    assert by_pair[("base", "modern")]["only_in_a"] == "lm_head.weight"
    assert by_pair[("base", "gated")]["architectures_differ"] == "true"


def test_run_audit_layer_and_head_tables(panel: dict[str, ModelSpec], tmp_path: Path) -> None:
    out = tmp_path / "out"
    _run(panel, out)

    _cols, layer_rows = _read_csv(out / "results" / R.BY_LAYER_NAME)
    assert {r["model_key"] for r in layer_rows} == {"twin", "modern", "gated"}
    assert {r["base_key"] for r in layer_rows} == {"base"}
    assert {r["row_kind"] for r in layer_rows} == {"tensor", "layer_family"}
    per_tensor = [r for r in layer_rows if r["row_kind"] == "tensor"]
    grouped = [r for r in layer_rows if r["row_kind"] == "layer_family"]
    assert len(per_tensor) == len(grouped)  # one tensor per (layer, family) here
    # twin still ships the tied head, so lm_head appears alongside the 13 families
    assert set(dist.FAMILIES) <= {r["family"] for r in grouped} <= set(dist.ALL_FAMILIES)
    assert {r["family"] for r in grouped if r["model_key"] == "modern"} == set(dist.FAMILIES)
    assert {int(r["layer"]) for r in grouped} == {dist.NO_LAYER, 0, 1}
    # the aggregate row reproduces the tensor row it summarises
    key = ("twin", "0", "q_proj")
    t = next(r for r in per_tensor if (r["model_key"], r["layer"], r["family"]) == key)
    g = next(r for r in grouped if (r["model_key"], r["layer"], r["family"]) == key)
    assert float(t["frobenius"]) == pytest.approx(float(g["frobenius"]))
    assert g["n_tensors"] == "1"

    _cols, head_rows = _read_csv(out / "results" / R.BY_HEAD_NAME)
    assert {r["head_kind"] for r in head_rows} == set(dist.HEAD_KINDS)
    assert {r["base_key"] for r in head_rows} == {"base"}
    q_rows = [r for r in head_rows if r["model_key"] == "twin" and r["head_kind"] == "q_proj"]
    assert len(q_rows) == 4 * TINY.num_layers
    assert {r["head_axis"] for r in q_rows} == {"rows"}
    o_rows = [r for r in head_rows if r["model_key"] == "twin" and r["head_kind"] == "o_proj"]
    assert {r["head_axis"] for r in o_rows} == {"cols"}
    k_rows = [r for r in head_rows if r["model_key"] == "twin" and r["head_kind"] == "k_proj"]
    assert {int(r["head_index"]) for r in k_rows} == {0, 1}
    assert {r["head_space"] for r in k_rows} == {"kv"}

    # head norms recompose into the tensor norm they were cut from
    layer0_q = [r for r in q_rows if r["layer"] == "0"]
    recomposed = np.sqrt(sum(float(r["frobenius"]) ** 2 for r in layer0_q))
    assert recomposed == pytest.approx(float(t["frobenius"]), rel=1e-9)


def test_tensor_hashes_csv_matches_the_manifest(
    panel: dict[str, ModelSpec], tmp_path: Path
) -> None:
    out = tmp_path / "out"
    _run(panel, out)
    _cols, rows = _read_csv(out / "manifests" / R.TENSOR_HASHES_NAME)
    manifest = json.loads((out / "manifests" / R.MANIFEST_NAME).read_text(encoding="utf-8"))
    for key, audit in manifest["models"].items():
        csv_for = {r["tensor"]: r for r in rows if r["model_key"] == key}
        assert len(csv_for) == audit["tensor_count"]
        for t in audit["tensors"]:
            row = csv_for[t["name"]]
            assert row["sha256"] == t["sha256"]
            assert row["dtype"] == t["st_dtype"]
            assert row["shape"] == "x".join(str(d) for d in t["shape"])
            assert int(row["n_params"]) == t["n_params"]
    # every tensor hash is a real digest of the bytes on disk
    locs = H.collect_tensor_locations(H.discover_model_files(panel["base"].path))
    for name, loc in locs.items():
        row = next(r for r in rows if r["model_key"] == "base" and r["tensor"] == name)
        assert row["sha256"] == H.sha256_tensor(loc)


def test_run_audit_is_idempotent(panel: dict[str, ModelSpec], tmp_path: Path) -> None:
    out = tmp_path / "out"
    _run(panel, out)
    before = {
        p.name: p.read_bytes()
        for sub, name in ARTIFACTS
        for p in (out / sub / name, R.provenance_path(out / sub / name))
    }
    summary = _run(panel, out)
    assert summary["written"] == []
    after = {
        p.name: p.read_bytes()
        for sub, name in ARTIFACTS
        for p in (out / sub / name, R.provenance_path(out / sub / name))
    }
    assert after == before


def test_run_audit_resumes_from_cache_without_rereading(
    panel: dict[str, ModelSpec], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    _run(panel, out)
    assert (out / "manifests" / R.CACHE_DIRNAME).is_dir()

    # Second run must not touch a single checkpoint byte.
    def explode(*_a: object, **_k: object) -> None:
        raise AssertionError("cache miss: the audit re-read a checkpoint it had already hashed")

    monkeypatch.setattr(R, "audit_checkpoint", explode)
    monkeypatch.setattr(dist, "pair_deltas", explode)
    _run(panel, out)

    # And a deleted artifact is restored from cache alone.
    monkeypatch.undo()
    monkeypatch.setattr(R, "audit_checkpoint", explode)
    monkeypatch.setattr(dist, "pair_deltas", explode)
    victim = out / "results" / R.BY_HEAD_NAME
    expected = victim.read_bytes()
    victim.unlink()
    summary = _run(panel, out)
    assert victim.is_file()
    assert victim.read_bytes() == expected
    assert R.BY_HEAD_NAME in summary["written"]


def test_touching_a_checkpoint_invalidates_its_cache(
    panel: dict[str, ModelSpec], tmp_path: Path
) -> None:
    out = tmp_path / "out"
    _run(panel, out)
    before = (out / "results" / R.PAIRWISE_NAME).read_bytes()

    replacement = tiny_state_dict(TINY, seed=99, tie_head=True)
    write_safetensors(panel["twin"].path / "model.safetensors", replacement, {"format": "pt"})
    summary = _run(panel, out)
    assert R.PAIRWISE_NAME in summary["written"]
    assert (out / "results" / R.PAIRWISE_NAME).read_bytes() != before


def test_cache_key_tracks_the_code_that_produced_it(
    panel: dict[str, ModelSpec], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A directory fingerprint alone would serve results computed by older code."""
    out = tmp_path / "out"
    _run(panel, out)

    real = R.code_fingerprint
    monkeypatch.setattr(R, "code_fingerprint", lambda *m: "edited:" + real(*m))
    recomputed: list[str] = []
    real_audit, real_deltas = R.audit_checkpoint, dist.pair_deltas
    monkeypatch.setattr(
        R,
        "audit_checkpoint",
        lambda *a, **k: (recomputed.append("audit"), real_audit(*a, **k))[1],
    )
    monkeypatch.setattr(
        dist,
        "pair_deltas",
        lambda *a, **k: (recomputed.append("delta"), real_deltas(*a, **k))[1],
    )
    _run(panel, out)
    assert "audit" in recomputed and "delta" in recomputed


def test_code_fingerprint_is_stable_and_discriminating() -> None:
    a = R.code_fingerprint(*R.AUDIT_CODE_MODULES)
    assert a == R.code_fingerprint(*R.AUDIT_CODE_MODULES)
    assert a != R.code_fingerprint(*R.DELTA_CODE_MODULES)
    assert len(a) == 64


def test_cache_directory_is_self_ignoring(panel: dict[str, ModelSpec], tmp_path: Path) -> None:
    out = tmp_path / "out"
    _run(panel, out)
    marker = out / "manifests" / R.CACHE_DIRNAME / ".gitignore"
    assert marker.is_file()
    assert marker.read_text(encoding="utf-8").strip().endswith("*")


def test_force_recomputes(panel: dict[str, ModelSpec], tmp_path: Path) -> None:
    out = tmp_path / "out"
    _run(panel, out)
    before = (out / "manifests" / R.MANIFEST_NAME).read_bytes()
    _run(panel, out, force=True)
    assert (out / "manifests" / R.MANIFEST_NAME).read_bytes() == before


def test_incompatible_pairs_are_classified_but_not_measured(
    panel: dict[str, ModelSpec], tmp_path: Path
) -> None:
    big = Arch(
        num_layers=2, hidden_size=8, num_q_heads=4, num_kv_heads=2, head_dim=2, vocab_size=10
    )
    directory = make_ckpt(
        tmp_path / "ckpts",
        "big",
        state_dict=tiny_state_dict(big, seed=5, tie_head=True),
        config=flat_config(big),
    )
    models = {**panel, "big": spec_for("big", directory)}
    out = tmp_path / "out"
    R.run_audit(
        models=models,
        panel=["base", "big"],
        base_key="base",
        out_root=out,
        probe_transformers=False,
        quiet=True,
    )
    _cols, rows = _read_csv(out / "results" / R.PAIRWISE_NAME)
    assert len(rows) == 1
    assert rows[0]["pair_class"] == "incompatible"
    assert rows[0]["delta_comparable"] == "false"
    assert rows[0]["frobenius"] == ""  # no distance is claimed
    _cols, head_rows = _read_csv(out / "results" / R.BY_HEAD_NAME)
    assert head_rows == []


def test_run_audit_refuses_to_write_into_a_checkpoint(
    panel: dict[str, ModelSpec], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The read-only guard from paths.py is wired into every write."""
    from rineya_psa import paths

    monkeypatch.setitem(paths.MODELS, "synthetic_base", panel["base"])
    with pytest.raises(PermissionError):
        R._write_if_changed(panel["base"].path / "config.json", "{}")


def test_cli_parses_and_runs(panel: dict[str, ModelSpec], tmp_path: Path,
                             monkeypatch: pytest.MonkeyPatch) -> None:
    from rineya_psa import paths

    monkeypatch.setattr(paths, "MODELS", panel, raising=False)
    monkeypatch.setattr(R.paths, "MODELS", panel, raising=False)
    monkeypatch.setattr(R.paths, "PANEL", ("base", "twin"), raising=False)
    monkeypatch.setattr(R.paths, "BASE_KEY", "base", raising=False)
    out = tmp_path / "cli"
    assert R.main(["--out-root", str(out), "--models", "base,twin", "--no-probe", "--quiet"]) == 0
    assert (out / "results" / R.PAIRWISE_NAME).is_file()
    assert (out / "manifests" / R.MANIFEST_NAME).is_file()


def test_pair_order_puts_base_first() -> None:
    assert R.pair_order(["a", "b", "c"], "b") == [("b", "a"), ("b", "c"), ("a", "c")]
    assert R.pair_order(["base", "x"], "base") == [("base", "x")]


# --------------------------------------------------------------------------- #
# real checkpoints -- opt-in only
# --------------------------------------------------------------------------- #

requires_real_checkpoints = pytest.mark.skipif(
    os.environ.get("PSA_MODEL_TESTS") != "1",
    reason="reads ~6.5 GB of real checkpoints; set PSA_MODEL_TESTS=1 to enable",
)


@pytest.mark.model
@pytest.mark.slow
@requires_real_checkpoints
def test_real_panel_matches_the_spec_table() -> None:
    """Pins SPEC.md §1 against the checkpoints actually on disk."""
    from rineya_psa import paths

    audits = {
        key: I.audit_checkpoint(key, probe_transformers=False) for key in paths.PANEL
    }
    for key, a in audits.items():
        assert a.arch() == Arch(
            num_layers=28,
            hidden_size=1024,
            num_q_heads=16,
            num_kv_heads=8,
            head_dim=128,
            vocab_size=151936,
        ), key
        assert a.intermediate_size == 3072
        assert a.rope_theta == 1e6
        assert a.tie_word_embeddings is True
        assert a.attention_bias_declared is False
        assert a.norms["qk_norm_over_head_dim"] is True
        assert a.spec_store_dtype_matches, key
        assert a.param_count_unique == 596_049_920
        assert a.param_count_non_embedding == 440_467_456

    # The schema split the brief calls out.
    assert audits["qwen3"].config_schema["rope_schema"] == "flat"
    assert audits["nimbus"].config_schema["rope_schema"] == "flat"
    assert audits["halo"].config_schema["rope_schema"] == "rope_parameters"
    assert audits["supa_a11"].config_schema["rope_schema"] == "rope_parameters"
    assert audits["halo"].config_schema["dtype_field"] == "dtype"
    assert audits["qwen3"].config_schema["dtype_field"] == "torch_dtype"

    gate = audits["supa_a11"].gate
    assert gate is not None
    assert gate["abstain_token_id"] == 51378
    assert gate["abstain_logit_bias"] == pytest.approx(5.15625, abs=1e-6)
    assert [f["name"] for f in audits["supa_a11"].custom_code_files] == ["modeling_supa_a11.py"]

    for other in ("halo", "nimbus", "supa_a11"):
        pc = I.classify_pair(audits["qwen3"], audits[other])
        assert pc.delta_comparable, other
        assert pc.arch_dims_identical
