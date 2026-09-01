"""Supa-A1.1 loader. Checkpoint is read-only."""

from __future__ import annotations

from typing import Any

import torch

from rineya_fwa.paths import resolve_supa_path


def load_tokenizer():
    from transformers import AutoTokenizer

    path = resolve_supa_path()
    tok = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def load_supa(
    *,
    device: str = "cuda",
    ungate: bool = False,
    eval_mode: bool = True,
) -> Any:
    from transformers import AutoModelForCausalLM

    path = resolve_supa_path()
    if not (path / "model.safetensors").is_file():
        raise FileNotFoundError(f"Supa-A1.1 weights missing at {path}")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        str(path),
        dtype=torch.float16 if device != "cpu" else torch.float32,
        trust_remote_code=True,
    )
    actual = type(model).__name__
    if actual != "Qwen3AbstainForCausalLM":
        raise RuntimeError(
            f"loaded {actual}, expected Qwen3AbstainForCausalLM (trust_remote_code missing?)"
        )
    if ungate:
        model.config.abstain_logit_bias = 0.0
    model.config.use_cache = False
    model = model.to(device)
    if eval_mode:
        model.eval()
    return model


def freeze_backbone(model: Any) -> None:
    for param in model.parameters():
        param.requires_grad = False
