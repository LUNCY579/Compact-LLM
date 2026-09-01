"""Supa-A1.1 loader. Checkpoint is read-only."""

from __future__ import annotations

import sys
from typing import Any

import torch

from rineya_fwa.paths import REPO_ROOT, SUPA_PATH

PSA_SRC = REPO_ROOT / "rineya_positional_sequential_attention_lab" / "src"
if str(PSA_SRC) not in sys.path:
    sys.path.insert(0, str(PSA_SRC))


def load_tokenizer():
    try:
        from rineya_psa.loading import load_tokenizer as _lt

        return _lt("supa_a11")
    except Exception:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(SUPA_PATH), trust_remote_code=True)
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
    try:
        from rineya_psa.loading import load_model

        model, _arch = load_model("supa_a11", device=device, ungate=ungate, eval_mode=eval_mode)
        return model
    except Exception:
        from transformers import AutoModelForCausalLM

        if not torch.cuda.is_available():
            device = "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            str(SUPA_PATH),
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
