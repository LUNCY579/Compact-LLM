"""Frozen Qwen3 augmentation without replacing any native layer.

Pinned target: transformers==4.57.1. This module has no import-time transformers
requirement. Structural test doubles are NOT evidence of native Qwen3 success.
Native cache and sidecar/history state travel together and cannot be reused,
branched, or switched between enabled/disabled modes. Use a fresh prefill.
"""
from dataclasses import dataclass
from types import SimpleNamespace
import uuid
import time
import torch
from torch import nn, Tensor
from .core import MemoryState


@dataclass
class CombinedCache:
    native: object
    memory: MemoryState | None
    length: int
    owner: str
    mode: tuple
    consumed: bool = False


class FrozenQwenSidecar(nn.Module):
    def __init__(self, backbone: nn.Module, sidecar: nn.Module | None, layer: int = 13):
        super().__init__()
        if getattr(backbone.config, "model_type", None) != "qwen3":
            raise ValueError("only the dense Qwen3 contract is supported")
        if not 0 <= layer < len(backbone.model.layers):
            raise ValueError("insertion layer is outside the backbone")
        if getattr(backbone, "is_gradient_checkpointing", False):
            raise ValueError("gradient checkpointing is not supported by this hook adapter")
        if getattr(backbone, "hf_device_map", None):
            raise ValueError("sharded/device_map backbones are not qualified")
        if hasattr(backbone, "_twm_owner"):
            raise ValueError("backbone already has a TWM wrapper")
        self.backbone, self.sidecar, self.layer = backbone, sidecar, layer
        self._owner = uuid.uuid4().hex
        backbone._twm_owner = self._owner
        self._context = None
        self._closed = False
        for p in backbone.parameters():
            p.requires_grad_(False)
        backbone.eval()
        self._handle = backbone.model.layers[layer].register_forward_hook(self._hook)

    def _hook(self, module, args, output):
        if self._context is None:
            return output
        if not isinstance(output, Tensor):
            raise TypeError("expected pinned Qwen3DecoderLayer tensor output")
        ctx = self._context
        if not ctx["enabled"] or self.sidecar is None:
            return output
        updated, memory = self.sidecar(output, state=ctx["memory"], valid=ctx["valid"],
                                       write=ctx["write"], ablation=ctx["ablation"])
        ctx["memory"] = memory
        return updated

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()  # Never enable frozen-backbone dropout.
        return self

    def assert_frozen(self):
        if any(p.requires_grad or p.grad is not None for p in self.backbone.parameters()):
            raise RuntimeError("frozen-backbone contract violated")

    def forward(self, input_ids: Tensor, attention_mask: Tensor | None = None,
                labels: Tensor | None = None, cache: CombinedCache | None = None,
                use_cache: bool = False, enabled: bool = True,
                write: Tensor | None = None, ablation: str = "none"):
        if self._closed or self._context is not None:
            raise RuntimeError("closed/reentrant wrapper; use independent models for concurrent requests")
        self.assert_frozen()
        if getattr(self.backbone, "is_gradient_checkpointing", False):
            raise ValueError("disable gradient checkpointing before invoking this wrapper")
        if input_ids.ndim != 2 or input_ids.shape[1] < 1:
            raise ValueError("nonempty [B,T] token IDs required")
        if (use_cache or cache is not None) and (self.training or torch.is_grad_enabled()):
            raise ValueError("caching is inference-only; use eval() and torch.no_grad()")
        if cache is not None and not use_cache:
            raise ValueError("a cache requires use_cache=True")
        if labels is not None and use_cache:
            raise ValueError("cached training/loss is unsupported")
        if self.sidecar is None and ablation != "none":
            raise ValueError("baseline has no memory ablations")
        b, t = input_ids.shape
        prefix = cache.length if cache is not None else 0
        shape = (b, prefix + t)
        attention_mask = torch.ones(shape, dtype=torch.long, device=input_ids.device) if attention_mask is None else attention_mask
        if attention_mask.shape != shape or attention_mask.device != input_ids.device:
            raise ValueError("attention mask must cover the complete native prefix")
        if not torch.all((attention_mask == 0) | (attention_mask == 1)):
            raise ValueError("binary 2-D attention mask required; packed documents unsupported")
        if use_cache and not bool(attention_mask.all()):
            raise ValueError("padded cached batches are not qualified; use separate unpadded rows")
        if not bool(attention_mask[:, -t:].any(-1).all()):
            raise ValueError("every sequence must contain at least one valid token")
        if write is not None and (write.shape != (b, t) or write.dtype != torch.bool or write.device != input_ids.device):
            raise ValueError("write mask must be boolean [B,T] on input device")
        mode = (enabled, ablation)
        memory = None
        native_cache = None
        if cache is not None:
            if cache.owner != self._owner or cache.mode != mode or cache.consumed:
                raise ValueError("foreign, stale, consumed, or mode-switched cache; start a new prefill")
            if cache.native.get_seq_length() != prefix:
                raise ValueError("native KV length and sidecar prefix are desynchronized")
            cache.consumed = True  # Invalidate before a possibly in-place native update.
            memory, native_cache = cache.memory, cache.native
        ctx = dict(enabled=enabled, memory=memory, valid=attention_mask[:, -t:].bool(), write=write, ablation=ablation)
        self._context = ctx
        try:
            result = self.backbone(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                                   past_key_values=native_cache, use_cache=use_cache, return_dict=True)
        finally:
            self._context = None
        combined = None
        if use_cache:
            if result.past_key_values is None or result.past_key_values.get_seq_length() != prefix + t:
                raise RuntimeError("native cache did not advance by the expected token count")
            memory = ctx["memory"]
            combined = CombinedCache(result.past_key_values, memory.detach() if memory else None,
                                     prefix+t, self._owner, mode)
        return SimpleNamespace(logits=result.logits, loss=getattr(result, "loss", None), cache=combined)

    def close(self):
        self._handle.remove()
        self._closed = True
        if getattr(self.backbone, "_twm_owner", None) == self._owner:
            delattr(self.backbone, "_twm_owner")


@torch.no_grad()
def greedy_decode(model, prompt_ids: Tensor, max_new_tokens: int = 24,
                  eos_token_id: int | None = None, enabled: bool = True,
                  ablation: str = "none", cached: bool = True, return_profile: bool = False):
    if model.training or prompt_ids.shape[0] != 1 or max_new_tokens <= 0:
        raise ValueError("greedy decoding requires eval mode, batch=1, and positive budget")
    def synchronize():
        if prompt_ids.device.type == "cuda":
            torch.cuda.synchronize(prompt_ids.device)
    synchronize()
    started = time.perf_counter()
    result = model(prompt_ids, use_cache=cached, enabled=enabled, ablation=ablation)
    synchronize()
    prefill_done = time.perf_counter()
    sequence, generated = prompt_ids, []
    stop = "length"
    for i in range(max_new_tokens):
        token = result.logits[:, -1].argmax(-1, keepdim=True)
        value = int(token.item())
        generated.append(value)
        if eos_token_id is not None and value == eos_token_id:
            stop = "eos"
            break
        if i + 1 < max_new_tokens:
            if cached:
                result = model(token, cache=result.cache, use_cache=True, enabled=enabled, ablation=ablation)
            else:
                sequence = torch.cat((sequence, token), 1)
                result = model(sequence, enabled=enabled, ablation=ablation)
    synchronize()
    profile = dict(prefill_seconds=prefill_done-started,
                   decode_seconds=time.perf_counter()-prefill_done,
                   continuation_forwards=max(0, len(generated)-1),
                   prompt_tokens=prompt_ids.shape[1], output_tokens=len(generated))
    return (generated, stop, profile) if return_profile else (generated, stop)
