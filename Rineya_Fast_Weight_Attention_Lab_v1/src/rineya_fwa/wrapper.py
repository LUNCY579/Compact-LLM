"""Install / uninstall auxiliary FWA around frozen Qwen3 attention modules."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import torch
from torch import nn

from rineya_fwa.falcon2 import FWAConfig, FastWeightBlock


def decoder_layers(model: Any) -> Sequence[nn.Module]:
    for holder in (getattr(model, "model", None), model):
        layers = getattr(holder, "layers", None)
        if layers is not None:
            return layers
    raise AttributeError(f"{type(model).__name__} exposes no .layers")


def resolve_layers(model: Any, layers: Iterable[int] | None) -> tuple[int, ...]:
    n = len(decoder_layers(model))
    if layers is None:
        return tuple(range(n))
    resolved = tuple(sorted({int(i) if int(i) >= 0 else n + int(i) for i in layers}))
    for i in resolved:
        if not 0 <= i < n:
            raise IndexError(f"layer {i} out of range [0, {n})")
    return resolved


def token_valid_mask(attention_mask: torch.Tensor | None, hidden: torch.Tensor) -> torch.Tensor:
    batch, length, _ = hidden.shape
    if attention_mask is None:
        return torch.ones(batch, length, dtype=torch.bool, device=hidden.device)
    mask = attention_mask
    if mask.dim() == 2:
        return mask.to(device=hidden.device).bool()
    if mask.dim() == 4:
        key_slice = mask[:, 0, -1, :length]
        return key_slice > -1.0e4
    raise ValueError(f"unsupported attention_mask shape {tuple(mask.shape)}")


class FWAAttentionWrapper(nn.Module):
    """Calls frozen inner attention, then adds α · P(m). α=0 skips the add in eval."""

    def __init__(self, inner: nn.Module, block: FastWeightBlock) -> None:
        super().__init__()
        self.inner = inner
        self.block = block

    def forward(self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        past = kwargs.get("past_key_values")
        past_len = 0
        if past is not None:
            getter = getattr(past, "get_seq_length", None)
            if callable(getter):
                past_len = int(getter() or 0)
            else:
                past_len = 1
        if past_len > 0:
            raise RuntimeError(
                "FWA refuses KV-cache mixing; load with use_cache=False and recompute the prefix"
            )
        out = self.inner(hidden_states, *args, **kwargs)
        if isinstance(out, tuple):
            attn_h, rest = out[0], out[1:]
        else:
            attn_h, rest = out, ()

        alpha = self.block.alpha
        if not self.training and float(alpha.detach()) == 0.0:
            return out

        mask = kwargs.get("attention_mask")
        if mask is None:
            for arg in args:
                if torch.is_tensor(arg) and arg.dim() in (2, 4) and arg.shape[-1] == hidden_states.shape[1]:
                    mask = arg
                    break
        valid = token_valid_mask(mask, hidden_states)
        delta = self.block(hidden_states, valid)
        delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
        merged = attn_h + alpha.to(dtype=attn_h.dtype) * delta.to(dtype=attn_h.dtype)
        if rest:
            return (merged, *rest)
        return merged


class InstalledFWA:
    def __init__(
        self,
        model: Any,
        layers: tuple[int, ...],
        wrappers: dict[int, FWAAttentionWrapper],
        originals: dict[int, nn.Module],
    ) -> None:
        self.model = model
        self.layers = layers
        self.wrappers = wrappers
        self.originals = originals
        self._live = True

    def blocks(self) -> list[FastWeightBlock]:
        return [w.block for w in self.wrappers.values()]

    def trainable_parameters(self) -> list[nn.Parameter]:
        params: list[nn.Parameter] = []
        for block in self.blocks():
            params.extend(p for p in block.parameters() if p.requires_grad)
        return params

    def uninstall(self) -> None:
        if not self._live:
            return
        layers = decoder_layers(self.model)
        for idx, original in self.originals.items():
            layers[idx].self_attn = original
        self._live = False

    def __enter__(self) -> InstalledFWA:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.uninstall()


def install_fwa(model: Any, cfg: FWAConfig, layers: Iterable[int]) -> InstalledFWA:
    blocks = decoder_layers(model)
    indices = resolve_layers(model, layers)
    originals: dict[int, nn.Module] = {}
    wrappers: dict[int, FWAAttentionWrapper] = {}
    try:
        for idx in indices:
            current = blocks[idx].self_attn
            if isinstance(current, FWAAttentionWrapper):
                raise RuntimeError(f"layer {idx} already has FWA installed")
            hidden = int(getattr(model.config, "hidden_size"))
            n_kv = int(getattr(model.config, "num_key_value_heads"))
            layer_cfg = FWAConfig(
                hidden_size=hidden,
                n_kv_heads=n_kv,
                rank=cfg.rank,
                eps=cfg.eps,
                recurrent=cfg.recurrent,
                significance_gate=cfg.significance_gate,
                init_alpha=cfg.init_alpha,
                beta_raw_init=cfg.beta_raw_init,
                lambda_raw_init=cfg.lambda_raw_init,
                significance_span=cfg.significance_span,
            )
            block = FastWeightBlock(layer_cfg)
            wrapper = FWAAttentionWrapper(current, block)
            param = next(model.parameters())
            wrapper.to(device=param.device)
            wrapper.block.float()
            originals[idx] = current
            wrappers[idx] = wrapper
            blocks[idx].self_attn = wrapper
    except Exception:
        for idx, original in originals.items():
            blocks[idx].self_attn = original
        raise
    return InstalledFWA(model, indices, wrappers, originals)
