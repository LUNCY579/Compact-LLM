"""Model / tokenizer loading with the panel's dtype and gating policy.

The only place in the lab that calls ``from_pretrained``.
"""

from __future__ import annotations

import contextlib
import functools
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rineya_psa.paths import model_spec
from rineya_psa.types import Arch, ModelSpec

_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

# Halo ships float32 (2.4 GB). On an 8.6 GB card that is affordable alone but not
# alongside activations at 8K, so the default run dtype is bfloat16 and the
# downcast is recorded as an approximation rather than hidden. Supa-A1.1 is the
# opposite case: its float16 store is the point of the checkpoint, so it must not
# be re-rounded to bfloat16.
_RUN_DTYPE_OVERRIDE: dict[str, str] = {"halo": "bfloat16"}
_DTYPE_LOCKED: frozenset[str] = frozenset({"supa_a11"})


def run_dtype(key: str) -> torch.dtype:
    spec = model_spec(key)
    return _DTYPES[_RUN_DTYPE_OVERRIDE.get(key, spec.store_dtype)]


def dtype_is_approximation(key: str) -> bool:
    """True when the run dtype differs from the checkpoint's store dtype."""
    return key in _RUN_DTYPE_OVERRIDE


@functools.lru_cache(maxsize=8)
def load_tokenizer(key: str):
    spec = model_spec(key)
    tok = AutoTokenizer.from_pretrained(
        str(spec.path), trust_remote_code=spec.trust_remote_code
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def arch_from_config(config: Any) -> Arch:
    return Arch(
        num_layers=int(config.num_hidden_layers),
        hidden_size=int(config.hidden_size),
        num_q_heads=int(config.num_attention_heads),
        num_kv_heads=int(config.num_key_value_heads),
        head_dim=int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)),
        vocab_size=int(config.vocab_size),
    )


def load_model(
    key: str,
    *,
    device: str = "cuda",
    dtype: torch.dtype | None = None,
    attn: str = "sdpa",
    ungate: bool = False,
    eval_mode: bool = True,
) -> tuple[Any, Arch]:
    """Load a panel checkpoint.

    ``ungate=True`` zeroes Supa-A1.1's abstention logit offset **on the loaded
    instance only**, so teacher-forced NLL and causal significance measure
    attention rather than the constant. It is a no-op for the other three.
    """
    spec: ModelSpec = model_spec(key)
    if dtype is None:
        dtype = run_dtype(key)
    elif key in _DTYPE_LOCKED and dtype != _DTYPES[spec.store_dtype]:
        raise ValueError(
            f"{key} must be loaded in {spec.store_dtype}; a different dtype re-rounds "
            "the weights this checkpoint exists to preserve"
        )

    model = AutoModelForCausalLM.from_pretrained(
        str(spec.path),
        dtype=dtype,
        trust_remote_code=spec.trust_remote_code,
        attn_implementation=attn if attn != "psa" else "eager",
    )
    actual = type(model).__name__
    if actual != spec.expected_arch:
        raise RuntimeError(
            f"{key} loaded as {actual}, expected {spec.expected_arch}. For supa_a11 this "
            "usually means trust_remote_code was not honoured, which silently disables "
            "the abstention gate."
        )

    # No-op for the plain cores; the caller records the requested mode either way.
    if ungate and spec.ungatable:
        model.config.abstain_logit_bias = 0.0

    model = model.to(device)
    if eval_mode:
        model.eval()
    model.config.use_cache = False
    return model, arch_from_config(model.config)


def gate_is_active(model: Any) -> bool:
    return bool(float(getattr(model.config, "abstain_logit_bias", 0.0) or 0.0))


def free_model(model: Any) -> None:
    """Release a model and its VRAM. The 8.6 GB card needs this between arms."""
    with contextlib.suppress(Exception):  # best-effort teardown
        model.to("cpu")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
