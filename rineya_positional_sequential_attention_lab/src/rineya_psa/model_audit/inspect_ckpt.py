"""Structural audit of one checkpoint directory, and pairwise classification.

Everything here is derived from bytes on disk: the safetensors header, the JSON
configs, and SHA-256 digests. No model is instantiated, nothing touches the GPU.

Two real schema differences in this panel are normalised here rather than left to
every downstream consumer:

* **rope**: Halo and Supa-A1.1 were re-serialised by transformers 5.14 and carry
  ``rope_parameters: {rope_theta, rope_type}``; Qwen3 and Nimbus carry the flat
  ``rope_theta`` / ``rope_scaling`` pair. Both collapse to
  ``(rope_theta, rope_type, rope_scaling)``.
* **store dtype**: the same two carry ``dtype``; the other two carry
  ``torch_dtype``. Both collapse to ``store_dtype``.

``layer_types`` (written only by the newer serialiser, and uniformly
``full_attention`` here) collapses to ``layer_types_unique`` so its presence alone
does not make two otherwise identical configs look different. Every collapse is
reported in ``config_schema`` so nothing is hidden.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rineya_psa.model_audit.hashing import (
    CHUNK_BYTES,
    collect_tensor_locations,
    directory_fingerprint,
    discover_model_files,
    hash_directory,
    hash_safetensors_file,
    json_diff_paths,
    read_json,
    read_safetensors_header,
    sha256_json,
    tokenizer_identity,
)
from rineya_psa.paths import model_spec
from rineya_psa.types import Arch, ModelSpec

AUDIT_SCHEMA_VERSION: str = "1.1"

PAIR_CLASSES: tuple[str, ...] = (
    "byte_identical",
    "config_identical",
    "architecture_compatible",
    "same_shape_different_weights",
    "incompatible",
)

# Pair classes for which W_m - W_base is a meaningful quantity.
DELTA_COMPARABLE_CLASSES: frozenset[str] = frozenset(PAIR_CLASSES[:4])

# Folded into normalised equivalents, or purely a record of who wrote the file.
DROPPED_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "transformers_version",
        "use_cache",
        "_name_or_path",
        "name_or_path",
        "torch_dtype",
        "dtype",
        "rope_theta",
        "rope_scaling",
        "rope_parameters",
        "layer_types",
        "output_attentions",
        "output_hidden_states",
        "return_dict",
        "torchscript",
        "_attn_implementation",
        "_attn_implementation_autoset",
    }
)

# Config fields that change what forward() computes, as opposed to how the file was
# written. Reported separately so a "cosmetic" diff can be told from a real one.
BEHAVIOURAL_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "architectures",
        "attention_bias",
        "attention_dropout",
        "head_dim",
        "hidden_act",
        "hidden_size",
        "intermediate_size",
        "max_position_embeddings",
        "num_attention_heads",
        "num_hidden_layers",
        "num_key_value_heads",
        "rms_norm_eps",
        "rope_scaling",
        "rope_theta",
        "rope_type",
        "sliding_window",
        "tie_word_embeddings",
        "use_sliding_window",
        "vocab_size",
        "abstain_token_id",
        "abstain_logit_bias",
        "auto_map",
    }
)

EMBED_SUFFIX: str = "embed_tokens.weight"
LM_HEAD_NAME: str = "lm_head.weight"


# --------------------------------------------------------------------------- #
# config normalisation
# --------------------------------------------------------------------------- #


def _rope_view(cfg: Mapping[str, Any]) -> tuple[str, float | None, str, dict[str, Any] | None]:
    """Collapse both rope spellings to ``(schema, theta, rope_type, scaling)``."""
    params = cfg.get("rope_parameters")
    if isinstance(params, Mapping):
        theta = params.get("rope_theta", params.get("theta"))
        rope_type = str(params.get("rope_type", params.get("type", "default")))
        rest = {
            k: v
            for k, v in params.items()
            if k not in {"rope_theta", "theta", "rope_type", "type"}
        }
        return "rope_parameters", _as_float(theta), rope_type, (rest or None)

    scaling = cfg.get("rope_scaling")
    if isinstance(scaling, Mapping) and scaling:
        rope_type = str(scaling.get("rope_type", scaling.get("type", "default")))
        rest = {k: v for k, v in scaling.items() if k not in {"rope_type", "type"}}
        return "flat", _as_float(cfg.get("rope_theta")), rope_type, (rest or None)
    return "flat", _as_float(cfg.get("rope_theta")), "default", None


def _as_float(value: Any) -> float | None:
    return None if value is None else float(value)


def normalize_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Serialiser-independent view of ``config.json``.

    Two checkpoints written by different transformers versions from the same
    architecture must produce equal output here; anything that changes the forward
    computation must survive.
    """
    out = {k: v for k, v in dict(cfg).items() if k not in DROPPED_CONFIG_KEYS}
    _schema, theta, rope_type, scaling = _rope_view(cfg)
    out["rope_theta"] = theta
    out["rope_type"] = rope_type
    out["rope_scaling"] = scaling
    out["store_dtype"] = cfg.get("dtype") or cfg.get("torch_dtype")

    layer_types = cfg.get("layer_types")
    if isinstance(layer_types, Sequence) and not isinstance(layer_types, str):
        out["layer_types_unique"] = sorted({str(t) for t in layer_types})
    else:
        # Absent means "no per-layer override"; Qwen3 with sliding_window=null is
        # full attention everywhere, which is what the newer serialiser writes out.
        out["layer_types_unique"] = ["full_attention"] if not cfg.get("sliding_window") else []
    out["architectures"] = [str(a) for a in (cfg.get("architectures") or [])]
    if out.get("pad_token_id") is None:
        out.pop("pad_token_id", None)
    return out


def config_schema_report(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Which spelling of each normalised field this file actually used."""
    schema, _theta, _rope_type, _scaling = _rope_view(cfg)
    dtype_field = "dtype" if "dtype" in cfg else ("torch_dtype" if "torch_dtype" in cfg else None)
    return {
        "rope_schema": schema,
        "dtype_field": dtype_field,
        "has_layer_types": "layer_types" in cfg,
        "declared_transformers_version": cfg.get("transformers_version"),
    }


# --------------------------------------------------------------------------- #
# tensor bookkeeping
# --------------------------------------------------------------------------- #


def _is_embedding_tensor(name: str) -> bool:
    return name.endswith(EMBED_SUFFIX) or name == LM_HEAD_NAME


@dataclass(frozen=True)
class TensorRecord:
    name: str
    file: str
    st_dtype: str
    shape: tuple[int, ...]
    n_params: int
    nbytes: int
    sha256: str

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["shape"] = list(self.shape)
        return d


# --------------------------------------------------------------------------- #
# the audit record
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CheckpointAudit:
    """Every structural fact about one checkpoint that Phase 1 is asked to record."""

    schema_version: str
    key: str
    display_name: str
    path: str
    path_exists: bool
    dir_fingerprint: str
    files: list[dict[str, Any]]
    model_files: list[str]
    total_bytes: int
    model_bytes: int
    safetensors_metadata: dict[str, Any]
    tensors: list[TensorRecord]
    tensor_count: int
    tensor_dtypes: list[str]
    param_count_total: int
    param_count_unique: int
    param_count_embedding: int
    param_count_non_embedding: int
    architecture_class: str | None
    model_type: str | None
    hidden_size: int | None
    intermediate_size: int | None
    num_hidden_layers: int | None
    num_attention_heads: int | None
    num_key_value_heads: int | None
    head_dim: int | None
    vocab_size: int | None
    max_position_embeddings: int | None
    rope_theta: float | None
    rope_type: str | None
    rope_scaling: dict[str, Any] | None
    store_dtype_declared: str | None
    spec_store_dtype: str
    spec_store_dtype_matches: bool
    tie_word_embeddings: bool | None
    lm_head_present: bool
    lm_head_is_embedding_copy: bool | None
    attention_bias_declared: bool | None
    bias_tensors: list[str]
    gqa: dict[str, Any]
    norms: dict[str, Any]
    tokenizer: dict[str, Any]
    custom_code_files: list[dict[str, Any]]
    auto_map: dict[str, Any] | None
    trust_remote_code_required: bool
    gate: dict[str, Any] | None
    attn_implementation: dict[str, Any]
    generation_config: dict[str, Any] | None
    config_raw: dict[str, Any]
    config_normalized: dict[str, Any]
    config_normalized_sha256: str
    config_schema: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    # -- derived helpers ---------------------------------------------------- #

    @property
    def tensor_shapes(self) -> dict[str, tuple[int, ...]]:
        return {t.name: t.shape for t in self.tensors}

    @property
    def tensor_hashes(self) -> dict[str, str]:
        return {t.name: t.sha256 for t in self.tensors}

    @property
    def tensor_dtype_map(self) -> dict[str, str]:
        return {t.name: t.st_dtype for t in self.tensors}

    def arch(self) -> Arch:
        missing = [
            n
            for n, v in (
                ("num_hidden_layers", self.num_hidden_layers),
                ("hidden_size", self.hidden_size),
                ("num_attention_heads", self.num_attention_heads),
                ("num_key_value_heads", self.num_key_value_heads),
                ("head_dim", self.head_dim),
                ("vocab_size", self.vocab_size),
            )
            if v is None
        ]
        if missing:
            raise ValueError(f"{self.key}: config is missing {missing}; cannot build Arch")
        return Arch(
            num_layers=int(self.num_hidden_layers),  # type: ignore[arg-type]
            hidden_size=int(self.hidden_size),  # type: ignore[arg-type]
            num_q_heads=int(self.num_attention_heads),  # type: ignore[arg-type]
            num_kv_heads=int(self.num_key_value_heads),  # type: ignore[arg-type]
            head_dim=int(self.head_dim),  # type: ignore[arg-type]
            vocab_size=int(self.vocab_size),  # type: ignore[arg-type]
        )

    def arch_dims(self) -> tuple[Any, ...]:
        return (
            self.num_hidden_layers,
            self.hidden_size,
            self.intermediate_size,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.head_dim,
            self.vocab_size,
        )

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["tensors"] = [t.to_json() for t in self.tensors]
        return d

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> CheckpointAudit:
        payload = dict(d)
        payload["tensors"] = [
            TensorRecord(
                name=t["name"],
                file=t["file"],
                st_dtype=t["st_dtype"],
                shape=tuple(int(x) for x in t["shape"]),
                n_params=int(t["n_params"]),
                nbytes=int(t["nbytes"]),
                sha256=t["sha256"],
            )
            for t in d.get("tensors", [])
        ]
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})


# --------------------------------------------------------------------------- #
# the audit itself
# --------------------------------------------------------------------------- #


@functools.lru_cache(maxsize=1)
def probe_attn_default() -> dict[str, Any]:
    """What the installed ``transformers`` would pick with ``attn_implementation`` unset.

    A bare ``PretrainedConfig`` reports ``_attn_implementation = None``: the value is
    not decided until ``PreTrainedModel`` resolves it at load time, from the kernels
    the class advertises. So the probe records the class capability flags and the
    resolution transformers 5 performs (SDPA when supported) rather than inventing a
    single string.

    Imported lazily: SPEC.md §2 keeps ``model_audit`` importable without torch, and
    ``transformers`` drags torch in. Degrades to the documented default if the probe
    fails for any reason.
    """
    try:  # pragma: no cover - exercised only when transformers is installed
        import transformers
        from transformers.configuration_utils import PretrainedConfig
        from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

        supports = {
            flag.removeprefix("_supports_"): bool(getattr(Qwen3ForCausalLM, flag, False))
            for flag in ("_supports_sdpa", "_supports_flash_attn", "_supports_flex_attn")
        }
        return {
            "config_default": getattr(PretrainedConfig(), "_attn_implementation", None),
            "resolved": "sdpa" if supports.get("sdpa") else "eager",
            "resolved_by": "transformers_probe",
            "supports": supports,
            "transformers_version": transformers.__version__,
        }
    except Exception as exc:  # noqa: BLE001 - the probe is best effort by design
        return {
            "config_default": None,
            "resolved": "sdpa",
            "resolved_by": "documented_default",
            "probe_error": f"{type(exc).__name__}: {exc}",
        }


def audit_checkpoint(
    key: str,
    *,
    spec: ModelSpec | None = None,
    chunk_size: int = CHUNK_BYTES,
    probe_transformers: bool = False,
) -> CheckpointAudit:
    """Audit one panel entry. Reads every byte of the directory exactly once."""
    spec = spec or model_spec(key)
    root = Path(spec.path)
    warnings: list[str] = []
    if not root.is_dir():
        raise FileNotFoundError(f"{key}: checkpoint directory not found: {root}")

    # -- weights ------------------------------------------------------------ #
    model_files = discover_model_files(root)
    if not model_files:
        raise FileNotFoundError(f"{key}: no *.safetensors under {root}")
    non_st = [p for p in model_files if p.suffix != ".safetensors"]
    if non_st:
        raise ValueError(
            f"{key}: only safetensors are auditable without executing pickle code; "
            f"found {[p.name for p in non_st]}"
        )

    file_digests: dict[str, str] = {}
    tensor_digests: dict[str, str] = {}
    st_metadata: dict[str, Any] = {}
    for p in model_files:
        fsha, tshas = hash_safetensors_file(p, chunk_size=chunk_size)
        file_digests[p.name] = fsha
        tensor_digests.update(tshas)
        _entries, meta, _start = read_safetensors_header(p)
        if meta:
            st_metadata[p.name] = meta

    locs = collect_tensor_locations(model_files)
    tensors = [
        TensorRecord(
            name=name,
            file=loc.file.name,
            st_dtype=loc.st_dtype,
            shape=loc.shape,
            n_params=loc.n_params,
            nbytes=loc.nbytes,
            sha256=tensor_digests[name],
        )
        for name, loc in sorted(locs.items())
    ]

    files = hash_directory(root, chunk_size=chunk_size, precomputed=file_digests)
    total_bytes = sum(int(f["size_bytes"]) for f in files)
    model_bytes = sum(p.stat().st_size for p in model_files)

    # -- config ------------------------------------------------------------- #
    cfg_path = root / "config.json"
    cfg: dict[str, Any] = read_json(cfg_path) if cfg_path.is_file() else {}
    if not cfg:
        warnings.append("config.json missing or empty")
    normalized = normalize_config(cfg)
    schema = config_schema_report(cfg)

    architectures = normalized["architectures"]
    arch_class = architectures[0] if architectures else None
    if arch_class != spec.expected_arch:
        warnings.append(
            f"config declares {arch_class!r} but ModelSpec expects {spec.expected_arch!r}"
        )

    # -- params ------------------------------------------------------------- #
    by_name = {t.name: t for t in tensors}
    param_total = sum(t.n_params for t in tensors)
    lm_head = by_name.get(LM_HEAD_NAME)
    embed = next((t for t in tensors if t.name.endswith(EMBED_SUFFIX)), None)
    tie = cfg.get("tie_word_embeddings")
    lm_head_copy: bool | None = None
    if lm_head is not None and embed is not None:
        lm_head_copy = lm_head.sha256 == embed.sha256
        if tie and not lm_head_copy:
            warnings.append(
                "tie_word_embeddings is true but lm_head.weight differs from "
                "embed_tokens.weight; the stored head will be discarded at load time"
            )
    duplicated = lm_head.n_params if (lm_head is not None and lm_head_copy) else 0
    param_unique = param_total - duplicated
    embedding_params = sum(t.n_params for t in tensors if _is_embedding_tensor(t.name)) - duplicated
    param_non_embedding = param_unique - embedding_params

    # -- GQA + projection shapes -------------------------------------------- #
    n_q = _int_or_none(cfg.get("num_attention_heads"))
    n_kv = _int_or_none(cfg.get("num_key_value_heads"))
    head_dim = _int_or_none(cfg.get("head_dim"))
    hidden = _int_or_none(cfg.get("hidden_size"))
    if head_dim is None and hidden is not None and n_q:
        head_dim = hidden // n_q
    gqa = _gqa_report(by_name, n_q=n_q, n_kv=n_kv, head_dim=head_dim, hidden=hidden)
    warnings.extend(gqa.pop("warnings", []))

    norms = _norm_report(by_name, cfg, head_dim=head_dim, hidden=hidden)

    bias_tensors = sorted(n for n in by_name if n.endswith(".bias"))
    attention_bias = cfg.get("attention_bias")
    attn_bias_tensors = [n for n in bias_tensors if ".self_attn." in n]
    if bool(attention_bias) != bool(attn_bias_tensors):
        warnings.append(
            f"attention_bias={attention_bias!r} but {len(attn_bias_tensors)} attention "
            "bias tensors are stored"
        )

    # -- tokenizer ----------------------------------------------------------- #
    tokenizer = tokenizer_identity(
        root, file_sha256={f["name"]: f["sha256"] for f in files if "/" not in f["name"]}
    )

    # -- custom code + gate --------------------------------------------------- #
    custom_code = [
        {"name": f["name"], "sha256": f["sha256"], "size_bytes": f["size_bytes"]}
        for f in files
        if f["name"].endswith(".py")
    ]
    auto_map = cfg.get("auto_map") if isinstance(cfg.get("auto_map"), Mapping) else None
    if auto_map and not custom_code:
        warnings.append("config declares auto_map but no .py file ships with the checkpoint")
    if custom_code and not spec.trust_remote_code:
        warnings.append(
            "checkpoint ships custom modelling code but ModelSpec.trust_remote_code is False"
        )

    gate: dict[str, Any] | None = None
    if "abstain_token_id" in cfg or "abstain_logit_bias" in cfg:
        gate = {
            "abstain_token_id": _int_or_none(cfg.get("abstain_token_id")),
            "abstain_logit_bias": _as_float(cfg.get("abstain_logit_bias")),
            "modeling_files": custom_code,
            "auto_map": dict(auto_map) if auto_map else None,
            "note": (
                "The offset is applied in forward(), not stored in any weight; "
                "loading without trust_remote_code silently disables it."
            ),
        }

    gen_cfg_path = root / "generation_config.json"
    gen_cfg = read_json(gen_cfg_path) if gen_cfg_path.is_file() else None

    attn_impl = {
        "declared_in_config": cfg.get("_attn_implementation"),
        "lab_default": "sdpa",
        "lab_default_source": "rineya_psa.loading.load_model(attn='sdpa')",
        "note": (
            "The lab never relies on the implicit default: load_model always passes "
            "attn_implementation explicitly, and the 'psa' interface of SPEC.md 5 "
            "loads as 'eager' before the custom kernel is registered."
        ),
        **(probe_attn_default() if probe_transformers else {"resolved_by": "not_probed"}),
    }

    return CheckpointAudit(
        schema_version=AUDIT_SCHEMA_VERSION,
        key=key,
        display_name=spec.display_name,
        path=str(root.resolve()),
        path_exists=True,
        dir_fingerprint=directory_fingerprint(root),
        files=files,
        model_files=[p.name for p in model_files],
        total_bytes=total_bytes,
        model_bytes=model_bytes,
        safetensors_metadata=st_metadata,
        tensors=tensors,
        tensor_count=len(tensors),
        tensor_dtypes=sorted({t.st_dtype for t in tensors}),
        param_count_total=param_total,
        param_count_unique=param_unique,
        param_count_embedding=embedding_params,
        param_count_non_embedding=param_non_embedding,
        architecture_class=arch_class,
        model_type=cfg.get("model_type"),
        hidden_size=hidden,
        intermediate_size=_int_or_none(cfg.get("intermediate_size")),
        num_hidden_layers=_int_or_none(cfg.get("num_hidden_layers")),
        num_attention_heads=n_q,
        num_key_value_heads=n_kv,
        head_dim=head_dim,
        vocab_size=_int_or_none(cfg.get("vocab_size")),
        max_position_embeddings=_int_or_none(cfg.get("max_position_embeddings")),
        rope_theta=normalized["rope_theta"],
        rope_type=normalized["rope_type"],
        rope_scaling=normalized["rope_scaling"],
        store_dtype_declared=normalized["store_dtype"],
        spec_store_dtype=spec.store_dtype,
        spec_store_dtype_matches=(normalized["store_dtype"] == spec.store_dtype),
        tie_word_embeddings=None if tie is None else bool(tie),
        lm_head_present=lm_head is not None,
        lm_head_is_embedding_copy=lm_head_copy,
        attention_bias_declared=None if attention_bias is None else bool(attention_bias),
        bias_tensors=bias_tensors,
        gqa=gqa,
        norms=norms,
        tokenizer=tokenizer,
        custom_code_files=custom_code,
        auto_map=dict(auto_map) if auto_map else None,
        trust_remote_code_required=bool(auto_map),
        gate=gate,
        attn_implementation=attn_impl,
        generation_config=gen_cfg,
        config_raw=cfg,
        config_normalized=normalized,
        config_normalized_sha256=sha256_json(normalized),
        config_schema=schema,
        warnings=warnings,
    )


def _int_or_none(value: Any) -> int | None:
    return None if value is None else int(value)


def _gqa_report(
    by_name: Mapping[str, TensorRecord],
    *,
    n_q: int | None,
    n_kv: int | None,
    head_dim: int | None,
    hidden: int | None,
) -> dict[str, Any]:
    """GQA structure, cross-checked against the projection shapes actually stored."""
    warnings: list[str] = []
    report: dict[str, Any] = {
        "num_q_heads": n_q,
        "num_kv_heads": n_kv,
        "head_dim": head_dim,
        "hidden_size": hidden,
        "warnings": warnings,
    }
    if not n_q or not n_kv:
        warnings.append("cannot describe GQA without num_attention_heads/num_key_value_heads")
        return report
    if n_q % n_kv:
        warnings.append(f"{n_q} query heads do not divide into {n_kv} KV heads")
        return report

    group = n_q // n_kv
    report["kv_group_size"] = group
    report["q_heads_for_kv"] = {str(kv): list(range(kv * group, (kv + 1) * group))
                                for kv in range(n_kv)}
    report["is_gqa"] = group > 1
    report["is_mqa"] = n_kv == 1

    def shape_of(suffix: str) -> tuple[int, ...] | None:
        rec = by_name.get(f"model.layers.0.self_attn.{suffix}.weight")
        return rec.shape if rec else None

    shapes = {s: shape_of(s) for s in ("q_proj", "k_proj", "v_proj", "o_proj")}
    report["layer0_projection_shapes"] = {k: (list(v) if v else None) for k, v in shapes.items()}
    if head_dim:
        expect = {
            "q_proj": (n_q * head_dim, hidden),
            "k_proj": (n_kv * head_dim, hidden),
            "v_proj": (n_kv * head_dim, hidden),
            "o_proj": (hidden, n_q * head_dim),
        }
        report["expected_projection_shapes"] = {k: list(v) for k, v in expect.items()}
        mismatched = [k for k, v in shapes.items() if v is not None and v != expect[k]]
        report["shapes_match_config"] = not mismatched
        if mismatched:
            warnings.append(f"projection shapes disagree with config for {mismatched}")
    return report


def _norm_report(
    by_name: Mapping[str, TensorRecord],
    cfg: Mapping[str, Any],
    *,
    head_dim: int | None,
    hidden: int | None,
) -> dict[str, Any]:
    """Norm placement and shape, including the Qwen3-specific per-head q/k norms."""
    def shape_of(name: str) -> tuple[int, ...] | None:
        rec = by_name.get(name)
        return rec.shape if rec else None

    q_norm = shape_of("model.layers.0.self_attn.q_norm.weight")
    k_norm = shape_of("model.layers.0.self_attn.k_norm.weight")
    final = shape_of("model.norm.weight")
    norm_names = [n for n in by_name if "norm" in n.rsplit(".", 1)[0].rsplit(".", 1)[-1]]
    return {
        "norm_class": "RMSNorm",
        "rms_norm_eps": _as_float(cfg.get("rms_norm_eps")),
        "weight_only": not any(n.endswith(".bias") and "norm" in n for n in by_name),
        "input_layernorm_shape": _listify(shape_of("model.layers.0.input_layernorm.weight")),
        "post_attention_layernorm_shape": _listify(
            shape_of("model.layers.0.post_attention_layernorm.weight")
        ),
        "q_norm_shape": _listify(q_norm),
        "k_norm_shape": _listify(k_norm),
        "has_qk_norm": q_norm is not None and k_norm is not None,
        "qk_norm_over_head_dim": bool(
            head_dim and q_norm == (head_dim,) and k_norm == (head_dim,)
        ),
        "final_norm_shape": _listify(final),
        "final_norm_over_hidden": bool(hidden and final == (hidden,)),
        "norm_tensor_count": len(norm_names),
        "note": (
            "Qwen3 applies q_norm/k_norm as RMSNorm over head_dim to each head "
            "before RoPE, which is why their weight is [head_dim] and not [hidden]."
        ),
    }


def _listify(shape: tuple[int, ...] | None) -> list[int] | None:
    return None if shape is None else list(shape)


# --------------------------------------------------------------------------- #
# pairwise classification
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PairClassification:
    model_a: str
    model_b: str
    pair_class: str
    delta_comparable: bool
    weights_identical: bool
    configs_identical: bool
    schema_identical: bool
    arch_dims_identical: bool
    dtype_differs: bool
    architectures_differ: bool
    tokenizer_identical: bool
    tokenizer_semantically_identical: bool
    tokenizer_vocab_identical: bool
    tokenizer_diff: list[str]
    n_shared_tensors: int
    n_identical_tensors: int
    only_in_a: list[str]
    only_in_b: list[str]
    shape_mismatches: list[str]
    tied_lm_head_convention: bool
    config_diff: list[str]
    behavioural_config_diff: list[str]
    reason: str

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _tied_lm_head_only(
    only_in_a: set[str], only_in_b: set[str], a: CheckpointAudit, b: CheckpointAudit
) -> bool:
    """True when the only schema difference is whether the tied head was materialised.

    Qwen3-0.6B and Nimbus store ``lm_head.weight`` even though
    ``tie_word_embeddings`` is true; Halo and Supa-A1.1 (re-serialised by
    transformers 5) omit it. Nothing about the model differs -- the stored head is
    a byte copy of the embedding and is discarded at load time.
    """
    extra = only_in_a | only_in_b
    if not extra:
        return True
    if extra != {LM_HEAD_NAME}:
        return False
    owner = a if LM_HEAD_NAME in only_in_a else b
    other = b if LM_HEAD_NAME in only_in_a else a
    return bool(owner.tie_word_embeddings) and bool(other.tie_word_embeddings) and bool(
        owner.lm_head_is_embedding_copy
    )


def _tokenizer_diff(a: CheckpointAudit, b: CheckpointAudit) -> list[str]:
    """Field-level tokenizer differences, prefixed by which file they live in.

    Two of this panel's four cores were re-serialised by transformers 5, which
    rewrote the ByteLevel decoder flags. Reporting that as a path
    (``tokenizer.json:decoder.add_prefix_space``) rather than as a hash mismatch is
    the difference between an actionable finding and a shrug.
    """
    out = [
        f"tokenizer.json:{p}"
        for p in json_diff_paths(
            a.tokenizer.get("tokenizer_json_structure", {}),
            b.tokenizer.get("tokenizer_json_structure", {}),
        )
    ]
    out += [
        f"tokenizer_config:{p}"
        for p in json_diff_paths(
            a.tokenizer.get("config_normalized", {}), b.tokenizer.get("config_normalized", {})
        )
    ]
    if a.tokenizer["vocab_merges_sha256"] != b.tokenizer["vocab_merges_sha256"]:
        out.append("vocab_or_merges:bytes")
    return out


def classify_pair(a: CheckpointAudit, b: CheckpointAudit) -> PairClassification:
    """Place a pair on the compatibility ladder, strongest level that holds.

    ``byte_identical`` > ``config_identical`` > ``architecture_compatible`` >
    ``same_shape_different_weights`` > ``incompatible``. The ladder is about the
    *weights*: config differences that do not change tensor layout are recorded in
    ``config_diff`` / ``behavioural_config_diff`` rather than demoting the pair, so
    Supa-A1.1's forward-time abstention gate does not hide the fact that its
    weights transplant cleanly into a plain Qwen3.
    """
    shapes_a, shapes_b = a.tensor_shapes, b.tensor_shapes
    names_a, names_b = set(shapes_a), set(shapes_b)
    shared = names_a & names_b
    only_a, only_b = names_a - names_b, names_b - names_a
    shape_mismatch = sorted(n for n in shared if shapes_a[n] != shapes_b[n])

    hashes_a, hashes_b = a.tensor_hashes, b.tensor_hashes
    identical = sorted(n for n in shared if hashes_a[n] == hashes_b[n])

    cfg_diff = json_diff_paths(a.config_normalized, b.config_normalized)
    behavioural = [p for p in cfg_diff if p.split(".")[0].split("[")[0] in BEHAVIOURAL_CONFIG_KEYS]
    configs_identical = not cfg_diff
    dtypes_a, dtypes_b = a.tensor_dtype_map, b.tensor_dtype_map
    schema_identical = (
        not only_a
        and not only_b
        and not shape_mismatch
        and all(dtypes_a[n] == dtypes_b[n] for n in shared)
    )
    arch_dims_identical = a.arch_dims() == b.arch_dims()
    tied_only = _tied_lm_head_only(only_a, only_b, a, b)
    weights_identical = not only_a and not only_b and len(identical) == len(shared) and bool(shared)

    if weights_identical:
        pair_class = "byte_identical"
        reason = "every tensor payload hashes the same"
    elif not arch_dims_identical or shape_mismatch:
        pair_class = "incompatible"
        reason = (
            f"architecture dims differ ({a.arch_dims()} vs {b.arch_dims()})"
            if not arch_dims_identical
            else f"{len(shape_mismatch)} shared tensors have different shapes"
        )
    elif configs_identical and schema_identical:
        pair_class = "config_identical"
        reason = "identical normalised config and tensor schema; weights differ"
    elif tied_only:
        pair_class = "architecture_compatible"
        reason = (
            "same dims and shapes; tensor sets differ only by the tied lm_head "
            "convention, so a state dict transplants cleanly"
            if (only_a or only_b)
            else "same dims and shapes; configs differ"
        )
    else:
        pair_class = "same_shape_different_weights"
        reason = (
            f"same dims and shapes but {len(only_a | only_b)} tensors exist on one "
            "side only, beyond the tied lm_head convention"
        )

    return PairClassification(
        model_a=a.key,
        model_b=b.key,
        pair_class=pair_class,
        delta_comparable=pair_class in DELTA_COMPARABLE_CLASSES,
        weights_identical=weights_identical,
        configs_identical=configs_identical,
        schema_identical=schema_identical,
        arch_dims_identical=arch_dims_identical,
        dtype_differs=a.tensor_dtypes != b.tensor_dtypes,
        architectures_differ=a.architecture_class != b.architecture_class,
        tokenizer_identical=(
            a.tokenizer["identity_sha256"] == b.tokenizer["identity_sha256"]
        ),
        tokenizer_semantically_identical=(
            a.tokenizer["semantic_sha256"] == b.tokenizer["semantic_sha256"]
        ),
        tokenizer_vocab_identical=(
            a.tokenizer["vocab_merges_sha256"] == b.tokenizer["vocab_merges_sha256"]
        ),
        tokenizer_diff=_tokenizer_diff(a, b),
        n_shared_tensors=len(shared),
        n_identical_tensors=len(identical),
        only_in_a=sorted(only_a),
        only_in_b=sorted(only_b),
        shape_mismatches=shape_mismatch,
        tied_lm_head_convention=tied_only and bool(only_a or only_b),
        config_diff=cfg_diff,
        behavioural_config_diff=behavioural,
        reason=reason,
    )
