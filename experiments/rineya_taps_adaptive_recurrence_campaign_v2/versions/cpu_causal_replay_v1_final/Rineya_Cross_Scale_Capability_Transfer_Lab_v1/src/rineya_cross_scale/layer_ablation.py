"""Identity residual skip for single decoder-layer ablation screens.

Registers a forward hook on `model.model.layers[i]` that returns the block
input unchanged, so attention and MLP for that layer contribute nothing.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

import torch


class LayerAblationError(RuntimeError):
    pass


def decoder_layers(model: Any) -> Any:
    """Return the ModuleList of decoder blocks for Qwen3 / Supa wrappers."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise LayerAblationError("could not locate decoder layers on model")


def _identity_hook(_module: Any, inputs: tuple[Any, ...], output: Any) -> Any:
    """Replace block hidden state with its input (residual passthrough)."""
    if not inputs:
        raise LayerAblationError("decoder block received empty inputs")
    hidden = inputs[0]
    if isinstance(output, tuple):
        return (hidden,) + tuple(output[1:])
    return hidden


@contextmanager
def skip_layer(model: Any, layer_index: int) -> Iterator[None]:
    layers = decoder_layers(model)
    n = len(layers)
    if not 0 <= layer_index < n:
        raise LayerAblationError(f"layer_index {layer_index} out of range 0..{n - 1}")
    handle = layers[layer_index].register_forward_hook(_identity_hook)
    try:
        yield
    finally:
        handle.remove()


def final_token_logits(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    kwargs: dict[str, Any] = {"input_ids": input_ids}
    if attention_mask is not None:
        kwargs["attention_mask"] = attention_mask
    with torch.no_grad():
        out = model(**kwargs)
    return out.logits[0, -1, :].float().cpu()


def logit_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    denom = float(a.norm().item() * b.norm().item())
    if denom <= 0:
        return float("nan")
    return float(torch.dot(a, b).item() / denom)


def verify_skip_hook(
    model: Any,
    tokenizer: Any,
    *,
    prompt: str = "Say hello in one word.",
    probe_layer: int = 14,
) -> dict[str, Any]:
    """Sanity: skip changes logits; restoring restores bit-identical logits."""
    messages = [{"role": "user", "content": prompt}]
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    encoded = tokenizer(rendered, return_tensors="pt")
    device = next(model.parameters()).device
    encoded = {key: value.to(device) for key, value in encoded.items()}

    base = final_token_logits(model, encoded["input_ids"], encoded.get("attention_mask"))
    with skip_layer(model, probe_layer):
        skipped = final_token_logits(
            model, encoded["input_ids"], encoded.get("attention_mask")
        )
    restored = final_token_logits(
        model, encoded["input_ids"], encoded.get("attention_mask")
    )

    delta_skip = float((base - skipped).abs().max().item())
    delta_restored = float((base - restored).abs().max().item())
    ok = delta_skip > 1e-4 and delta_restored == 0.0
    return {
        "probe_layer": probe_layer,
        "max_abs_delta_skip_vs_intact": delta_skip,
        "max_abs_delta_restored_vs_intact": delta_restored,
        "cosine_skip_vs_intact": logit_cosine(base, skipped),
        "hook_live": delta_skip > 1e-4,
        "deterministic_restore": delta_restored == 0.0,
        "ok": ok,
    }
